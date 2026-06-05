"""Scene and object models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import TaskKind
from retarget.core.pose import PoseSequence

ObjectQposMode = Literal["appended", "external"]


class ObjectTrajectory(BaseModel):
    """Dynamic object poses over time.

    Attributes:
        poses (PoseSequence): Per-frame object rigid transform track.
        name (str): Object label used in scene exports and logs.
    """

    poses: PoseSequence
    name: str = "object"

    @classmethod
    def identity(cls, frame_count: int, *, fps: float = 30.0, name: str = "object") -> ObjectTrajectory:
        """Return an identity object trajectory."""

        return cls(name=name, poses=PoseSequence.identity(frame_count, fps=fps))

    def resampled(self, fps: float) -> ObjectTrajectory:
        """Return this object trajectory sampled on a new FPS grid."""

        return ObjectTrajectory(name=self.name, poses=self.poses.resampled(fps))


class ObjectSpec(BaseModel):
    """Object asset and sampling information.

    Attributes:
        name (str): Object identifier referenced by the scene and optimizer.
        mesh_path (Path | None): Optional triangle mesh file for visualization or sampling.
        urdf_path (Path | None): Optional URDF describing articulated object geometry.
        sample_points (FloatArray | None): Precomputed surface points with shape ``(N, 3)``.
        trajectory (ObjectTrajectory | None): Time-varying object pose track.
        qpos_mode (ObjectQposMode): Whether the trajectory is appended to qpos or treated as an external scene pose.
        metadata (dict[str, Any]): Opaque sidecar fields (mass, scale, asset ids, …).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    mesh_path: Path | None = None
    urdf_path: Path | None = None
    sample_points: FloatArray | None = None
    trajectory: ObjectTrajectory | None = None
    qpos_mode: ObjectQposMode = "appended"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mesh_path", "urdf_path", mode="before")
    @classmethod
    def _path_or_none(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)

    @field_validator("sample_points", mode="before")
    @classmethod
    def _validate_points(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="sample_points")
        if arr.ndim != 2:
            raise ValueError("sample_points must have shape (N, 3)")
        return arr


class TerrainSpec(BaseModel):
    """Static terrain information.

    Attributes:
        name (str): Terrain label used in scene exports and logs.
        mesh_path (Path | None): Optional terrain mesh file.
        sample_points (FloatArray | None): Precomputed terrain surface points with shape ``(N, 3)``.
        metadata (dict[str, Any]): Opaque sidecar fields (friction, resolution, …).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ground"
    mesh_path: Path | None = None
    sample_points: FloatArray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mesh_path", mode="before")
    @classmethod
    def _path_or_none(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)


class SceneSpec(BaseModel):
    """Scene configuration for a retargeting run.

    Attributes:
        task_kind (TaskKind): High-level workflow (robot-only, object interaction, climbing).
        object (ObjectSpec | None): Manipulated or climbable object definition.
        terrain (TerrainSpec | None): Static ground or climbable terrain definition.
        ground_range (tuple[float, float]): XY extent of the procedural ground grid (meters).
        ground_size (int): Number of samples per axis for :meth:`ground_points`.
        metadata (dict[str, Any]): Opaque sidecar fields passed through to the solver.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_kind: TaskKind
    object: ObjectSpec | None = None
    terrain: TerrainSpec | None = None
    ground_range: tuple[float, float] = (-1.0, 1.0)
    ground_size: int = 15
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scene(self) -> SceneSpec:
        if self.ground_size <= 0:
            raise ValueError("ground_size must be positive")
        if self.ground_range[1] < self.ground_range[0]:
            raise ValueError("ground_range upper value must be >= lower value")
        if self.task_kind == TaskKind.OBJECT_INTERACTION and self.object is None:
            raise ValueError("object_interaction requires an object")
        if self.task_kind == TaskKind.CLIMBING and self.terrain is None and self.object is None:
            raise ValueError("climbing requires terrain or an object")
        return self

    @classmethod
    def robot_only(cls) -> SceneSpec:
        """Create a robot-only ground scene.

        Returns:
            SceneSpec: ``ROBOT_ONLY`` task with default flat terrain.
        """

        return cls(task_kind=TaskKind.ROBOT_ONLY, terrain=TerrainSpec())

    @classmethod
    def object_interaction(cls, object_spec: ObjectSpec) -> SceneSpec:
        """Create an object interaction scene.

        Args:
            object_spec (ObjectSpec): Manipulated object definition.

        Returns:
            SceneSpec: ``OBJECT_INTERACTION`` task referencing ``object_spec``.
        """

        return cls(task_kind=TaskKind.OBJECT_INTERACTION, object=object_spec)

    @classmethod
    def climbing(cls, terrain: TerrainSpec | None = None, object_spec: ObjectSpec | None = None) -> SceneSpec:
        """Create a climbing or terrain-interaction scene.

        Args:
            terrain (TerrainSpec | None): Climbable terrain mesh or samples.
            object_spec (ObjectSpec | None): Optional climbable object instead of terrain.

        Returns:
            SceneSpec: ``CLIMBING`` task; inserts a default terrain spec when both arguments are omitted.
        """

        if terrain is None and object_spec is None:
            terrain = TerrainSpec(name="terrain")
        return cls(task_kind=TaskKind.CLIMBING, terrain=terrain, object=object_spec)

    def has_dynamic_object(self) -> bool:
        """Whether qpos should include appended object poses."""

        return (
            self.object is not None
            and self.object.trajectory is not None
            and self.object.qpos_mode == "appended"
        )

    def resampled(self, fps: float) -> SceneSpec:
        """Return a scene with dynamic trajectories sampled on a new FPS grid."""

        if self.object is None or self.object.trajectory is None:
            return self.model_copy(deep=True)
        object_spec = self.object.model_copy(update={"trajectory": self.object.trajectory.resampled(fps)})
        return self.model_copy(update={"object": object_spec}, deep=True)

    def ground_points(self) -> FloatArray:
        """Create an xy grid of ground points."""

        coords = np.linspace(self.ground_range[0], self.ground_range[1], self.ground_size)
        x, y = np.meshgrid(coords, coords)
        return np.stack([x.reshape(-1), y.reshape(-1), np.zeros(x.size)], axis=1)
