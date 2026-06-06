"""Scene and object models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import ObjectQposMode, ObjectSampleSpace, TaskKind
from retarget.core.pose import PoseSequence


def _coerce_asset_scale(value: Any, *, allow_none: bool = False) -> tuple[float, float, float] | None:
    if value is None:
        return None if allow_none else (1.0, 1.0, 1.0)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(()), 3)
    arr = arr.reshape(-1)
    if arr.shape != (3,):
        raise ValueError("asset_scale must be a scalar or three values")
    if not np.all(np.isfinite(arr)):
        raise ValueError("asset_scale must contain finite values")
    if np.any(arr <= 0.0):
        raise ValueError("asset_scale values must be positive")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _coerce_rgba(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (4,):
        raise ValueError("rgba must contain four values")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rgba must contain finite values")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("rgba values must be in [0, 1]")
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


class ObjectVisualPart(BaseModel):
    """A renderable object mesh part with optional material styling.

    Attributes:
        name (str): Stable part label used in scene paths.
        mesh_path (Path): Mesh file for this visual part.
        asset_scale (tuple[float, float, float] | None): Optional part-local scale; inherits the
            parent object scale when omitted by playback metadata builders.
        rgba (tuple[float, float, float, float] | None): Optional material color with alpha in ``[0, 1]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    mesh_path: Path
    asset_scale: tuple[float, float, float] | None = None
    rgba: tuple[float, float, float, float] | None = None

    @field_validator("mesh_path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        if value in (None, ""):
            raise ValueError("mesh_path is required for object visual parts")
        return Path(value)

    @field_validator("asset_scale", mode="before")
    @classmethod
    def _validate_asset_scale(cls, value: Any) -> tuple[float, float, float] | None:
        return _coerce_asset_scale(value, allow_none=True)

    @field_validator("rgba", mode="before")
    @classmethod
    def _validate_rgba(cls, value: Any) -> tuple[float, float, float, float] | None:
        return _coerce_rgba(value)


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
        asset_scale (tuple[float, float, float]): Scale applied to asset-local geometry.
        visual_parts (tuple[ObjectVisualPart, ...]): Optional visual mesh parts with typed materials.
        sample_points (FloatArray | None): Precomputed asset-local surface points with shape ``(N, 3)``.
        sample_space (ObjectSampleSpace): Coordinate space for ``sample_points``.
        trajectory (ObjectTrajectory | None): Time-varying object pose track.
        qpos_mode (ObjectQposMode): Whether the trajectory is appended to qpos or treated as an external scene pose.
        metadata (dict[str, Any]): Opaque sidecar fields (mass, scale, asset ids, …).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    mesh_path: Path | None = None
    urdf_path: Path | None = None
    asset_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    visual_parts: tuple[ObjectVisualPart, ...] = ()
    sample_points: FloatArray | None = None
    sample_space: ObjectSampleSpace = ObjectSampleSpace.OBJECT_LOCAL
    trajectory: ObjectTrajectory | None = None
    qpos_mode: ObjectQposMode = ObjectQposMode.APPENDED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mesh_path", "urdf_path", mode="before")
    @classmethod
    def _path_or_none(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)

    @field_validator("asset_scale", mode="before")
    @classmethod
    def _validate_asset_scale(cls, value: Any) -> tuple[float, float, float]:
        result = _coerce_asset_scale(value)
        if result is None:
            raise ValueError("asset_scale is required")
        return result

    @field_validator("visual_parts", mode="before")
    @classmethod
    def _validate_visual_parts(cls, value: Any) -> tuple[ObjectVisualPart, ...]:
        if value is None or value == "":
            return ()
        if not isinstance(value, list | tuple):
            raise ValueError("visual_parts must be a list or tuple")
        return tuple(
            item if isinstance(item, ObjectVisualPart) else ObjectVisualPart.model_validate(item)
            for item in value
        )

    @field_validator("sample_points", mode="before")
    @classmethod
    def _validate_points(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="sample_points")
        if arr.ndim != 2:
            raise ValueError("sample_points must have shape (N, 3)")
        return arr

    def scaled_sample_points(self, default: FloatArray | None = None) -> FloatArray | None:
        """Return object sample points in the active object-local geometry frame."""

        raw_points = self.sample_points if self.sample_points is not None else default
        if raw_points is None:
            return None
        points = as_float_array(raw_points, shape_tail=(3,), name="sample_points")
        if points.ndim != 2:
            raise ValueError("sample_points must have shape (N, 3)")
        if self.sample_space in (ObjectSampleSpace.OBJECT_LOCAL, ObjectSampleSpace.OBJECT_ASSET_LOCAL):
            return points * np.asarray(self.asset_scale, dtype=np.float64)
        return points.copy()


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
            and self.object.qpos_mode == ObjectQposMode.APPENDED
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
