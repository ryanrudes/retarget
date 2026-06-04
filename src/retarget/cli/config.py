"""Typed CLI run-spec loading."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.enums import FrameConvention, QuaternionOrder, TaskKind
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.registry import Registry
from retarget.mesh import InteractionMeshSpec, sample_mesh_points
from retarget.motion import load_motion, motion_formats
from retarget.motion.contact import ContactPlan, SupportPlane
from retarget.optimization import validate_optimization_references
from retarget.optimization.spec import ConstraintSpec, ObjectiveSpec, OptimizationProfile, SolverSpec
from retarget.pipeline import RetargetingProblem
from retarget.robots import robot_providers, robots
from retarget.scene import ObjectSpec, ObjectTrajectory, SceneSpec, TerrainSpec


class ObjectConfig(BaseModel):
    """Serializable object-scene options for CLI run specs.

    Attributes:
        name (str): Object identifier (default ``"object"``).
        mesh_path (Path | None): Mesh file used to sample interaction points.
        urdf_path (Path | None): Optional URDF for object geometry.
        sample_points (tuple[tuple[float, float, float], ...] | None): Inline object sample points.
        sample_points_path (Path | None): File containing sample points (``.npy``, ``.npz``, etc.).
        mesh_sample_count (int): Number of points to sample from ``mesh_path`` when inline/path
            points are omitted (default ``128``).
        sample_points_frame (FrameConvention): Frame convention of inline/path sample points.
        identity_trajectory (bool): Use a fixed identity object pose for every frame.
        trajectory_path (Path | None): File with object pose trajectory.
        trajectory_positions (tuple[tuple[float, float, float], ...] | None): Inline positions.
        trajectory_quaternions (tuple[tuple[float, float, float, float], ...] | None): Inline
            orientations paired with positions.
        trajectory_quaternion_order (QuaternionOrder): Storage order of inline quaternions.
        trajectory_frame (FrameConvention): Frame convention of inline trajectory data.
        metadata (dict[str, Any]): Free-form object metadata passed to :class:`~retarget.scene.ObjectSpec`.
    """

    name: str = "object"
    mesh_path: Path | None = None
    urdf_path: Path | None = None
    sample_points: tuple[tuple[float, float, float], ...] | None = None
    sample_points_path: Path | None = None
    mesh_sample_count: int = 128
    sample_points_frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    identity_trajectory: bool = False
    trajectory_path: Path | None = None
    trajectory_positions: tuple[tuple[float, float, float], ...] | None = None
    trajectory_quaternions: tuple[tuple[float, float, float, float], ...] | None = None
    trajectory_quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    trajectory_frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mesh_sample_count")
    @classmethod
    def _non_negative_mesh_sample_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mesh_sample_count must be non-negative")
        return value

    def resolve_paths(self, base_dir: Path) -> ObjectConfig:
        """Return a copy with relative asset paths resolved against `base_dir`."""

        return self.model_copy(
            update={
                "mesh_path": _resolve_relative(self.mesh_path, base_dir),
                "urdf_path": _resolve_relative(self.urdf_path, base_dir),
                "sample_points_path": _resolve_relative(self.sample_points_path, base_dir),
                "trajectory_path": _resolve_relative(self.trajectory_path, base_dir),
            }
        )


class TerrainConfig(BaseModel):
    """Serializable terrain-scene options for CLI run specs.

    Attributes:
        name (str): Terrain identifier (default ``"terrain"``).
        mesh_path (Path | None): Terrain mesh file.
        sample_points (tuple[tuple[float, float, float], ...] | None): Inline terrain sample points.
        sample_points_path (Path | None): File containing terrain sample points.
        mesh_sample_count (int): Points to sample from ``mesh_path`` when others are omitted.
        sample_points_frame (FrameConvention): Frame convention of sample points.
        metadata (dict[str, Any]): Free-form terrain metadata passed to :class:`~retarget.scene.TerrainSpec`.
    """

    name: str = "terrain"
    mesh_path: Path | None = None
    sample_points: tuple[tuple[float, float, float], ...] | None = None
    sample_points_path: Path | None = None
    mesh_sample_count: int = 128
    sample_points_frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mesh_sample_count")
    @classmethod
    def _non_negative_mesh_sample_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mesh_sample_count must be non-negative")
        return value

    def resolve_paths(self, base_dir: Path) -> TerrainConfig:
        """Return a copy with relative asset paths resolved against `base_dir`."""

        return self.model_copy(
            update={
                "mesh_path": _resolve_relative(self.mesh_path, base_dir),
                "sample_points_path": _resolve_relative(self.sample_points_path, base_dir),
            }
        )


class SceneConfig(BaseModel):
    """Serializable scene options for CLI run specs.

    Attributes:
        object (ObjectConfig | None): Manipulated object configuration for interaction tasks.
        terrain (TerrainConfig | None): Terrain mesh or samples for climbing tasks.
        ground_range (tuple[float, float]): Horizontal ground sampling range for support meshes.
        ground_size (int): Grid resolution for ground support sampling.
        metadata (dict[str, Any]): Free-form scene metadata passed to :class:`~retarget.scene.SceneSpec`.
    """

    object: ObjectConfig | None = None
    terrain: TerrainConfig | None = None
    ground_range: tuple[float, float] = (-1.0, 1.0)
    ground_size: int = 15
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_paths(self, base_dir: Path) -> SceneConfig:
        """Return a copy with relative asset paths resolved against `base_dir`."""

        return self.model_copy(
            update={
                "object": self.object.resolve_paths(base_dir) if self.object is not None else None,
                "terrain": self.terrain.resolve_paths(base_dir) if self.terrain is not None else None,
            }
        )


class RetargetingRunConfig(BaseModel):
    """Human-editable run spec used by the CLI.

    Attributes:
        name (str | None): Optional run or result name; defaults to the motion clip name.
        motion (Path): Source motion file path.
        format_name (str): Registered motion format key (TOML alias ``format``; default ``"minimal"``).
        robot (str): Robot name passed to the selected provider.
        robot_provider (str): Registered provider key (default ``"registry"``).
        robot_options (dict[str, Any]): Provider-specific options (``path``, ``store``, etc.).
        task_kind (TaskKind): Retargeting workflow kind.
        output (Path): Destination ``.npz`` result path.
        imports (tuple[str, ...]): Extension modules or ``.py`` plugins to import before validation.
        scale_to_robot (bool): Scale source motion to the target robot height.
        output_fps (float | None): Optional result frame rate override.
        show_progress (bool): Show a Rich per-frame progress bar during optimization.
        joint_mapping (dict[str, str] | None): Source-to-robot joint name overrides.
        mesh (InteractionMeshSpec): Interaction mesh construction settings.
        solver (SolverSpec): Optimization solver configuration.
        objectives (tuple[ObjectiveSpec, ...] | None): Objective terms; defaults when omitted.
        constraints (tuple[ConstraintSpec, ...] | None): Constraint terms; defaults when omitted.
        scene (SceneConfig): Scene object, terrain, and ground options.
        metadata (dict[str, Any]): Free-form run metadata stored on the result.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    motion: Path
    format_name: str = Field(default="minimal", alias="format")
    robot: str = "synthetic_humanoid"
    robot_provider: str = "registry"
    robot_options: dict[str, Any] = Field(default_factory=dict)
    task_kind: TaskKind = TaskKind.ROBOT_ONLY
    output: Path
    imports: tuple[str, ...] = ()
    scale_to_robot: bool = True
    output_fps: float | None = None
    show_progress: bool = False
    joint_mapping: dict[str, str] | None = None
    mesh: InteractionMeshSpec = Field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    objectives: tuple[ObjectiveSpec, ...] | None = None
    constraints: tuple[ConstraintSpec, ...] | None = None
    scene: SceneConfig = Field(default_factory=SceneConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("motion", "output", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("imports", mode="before")
    @classmethod
    def _coerce_imports(cls, value: Any) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a `.toml`, `.yaml`, `.yml`, or `.json` run spec."""

        config_path = Path(path)
        data = _load_mapping(config_path)
        config = cls.model_validate(data)
        return config.resolve_paths(config_path.parent)

    def resolve_paths(self, base_dir: Path) -> Self:
        """Return a copy with relative file paths resolved against `base_dir`."""

        return self.model_copy(
            update={
                "motion": _resolve_relative(self.motion, base_dir),
                "output": _resolve_relative(self.output, base_dir),
                "imports": _resolve_import_refs(self.imports, base_dir),
                "robot_options": _resolve_robot_options(self.robot_options, base_dir),
                "scene": self.scene.resolve_paths(base_dir),
            }
        )

    def with_overrides(
        self,
        *,
        motion: Path | None = None,
        output: Path | None = None,
        format_name: str | None = None,
        robot: str | None = None,
        task_kind: TaskKind | None = None,
        name: str | None = None,
        show_progress: bool | None = None,
    ) -> Self:
        """Return a copy with explicit CLI overrides applied."""

        updates: dict[str, Any] = {}
        if motion is not None:
            updates["motion"] = motion
        if output is not None:
            updates["output"] = output
        if format_name is not None:
            updates["format_name"] = format_name
        if robot is not None:
            updates["robot"] = robot
            updates["robot_provider"] = "registry"
            updates["robot_options"] = {}
        if task_kind is not None:
            updates["task_kind"] = task_kind
        if name is not None:
            updates["name"] = name
        if show_progress is not None:
            updates["show_progress"] = show_progress
        return self.model_copy(update=updates)

    def build_problem(self) -> RetargetingProblem:
        """Resolve registries and build a `RetargetingProblem`."""

        self.validate_registry_references()
        motion = load_motion(self.motion, self.format_name, name=self.name)
        robot_spec = robot_providers.get(self.robot_provider).load(self.robot, **self.robot_options)
        scene = self._build_scene(frame_count=motion.frame_count, fps=motion.fps)
        contacts = _contact_plan_from_motion(motion, robot_spec.contact_links)
        return RetargetingProblem(
            name=self.name or motion.name,
            task_kind=self.task_kind,
            robot=robot_spec,
            motion=motion,
            contacts=contacts,
            scene=scene,
            motion_format=motion_formats.get(self.format_name),
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            objectives=self.objectives if self.objectives is not None else _default_objectives(),
            constraints=(
                self.constraints if self.constraints is not None else _default_constraints()
            ),
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps,
            show_progress=self.show_progress,
            metadata=self.metadata,
        )

    def import_extensions(self) -> None:
        """Import explicitly configured extension modules."""

        for reference in self.imports:
            _import_extension(reference)

    def validate_registry_references(self) -> None:
        """Validate named extension references before loading files or assets."""

        self.import_extensions()
        messages: list[str] = []
        _append_missing_registry_messages(messages, motion_formats, (self.format_name,))
        _append_missing_registry_messages(messages, robot_providers, (self.robot_provider,))
        if self.robot_provider == "registry":
            _append_missing_registry_messages(messages, robots, (self.robot,))
        if messages:
            raise KeyError("Unknown run-config registry references: " + "; ".join(messages))
        validate_optimization_references(
            solver=self.solver,
            objectives=self.objectives if self.objectives is not None else _default_objectives(),
            constraints=self.constraints if self.constraints is not None else _default_constraints(),
        )

    def _build_scene(self, *, frame_count: int, fps: float) -> SceneSpec:
        object_spec = _object_spec(self.scene.object, frame_count=frame_count, fps=fps)
        terrain_spec = _terrain_spec(self.scene.terrain)
        if self.task_kind == TaskKind.ROBOT_ONLY:
            return SceneSpec(
                task_kind=self.task_kind,
                terrain=terrain_spec or TerrainSpec(),
                ground_range=self.scene.ground_range,
                ground_size=self.scene.ground_size,
                metadata=self.scene.metadata,
            )
        if self.task_kind == TaskKind.OBJECT_INTERACTION:
            return SceneSpec(
                task_kind=self.task_kind,
                object=object_spec or ObjectSpec(name="object"),
                terrain=terrain_spec,
                ground_range=self.scene.ground_range,
                ground_size=self.scene.ground_size,
                metadata=self.scene.metadata,
            )
        return SceneSpec(
            task_kind=self.task_kind,
            object=object_spec,
            terrain=terrain_spec or TerrainSpec(name="terrain"),
            ground_range=self.scene.ground_range,
            ground_size=self.scene.ground_size,
            metadata=self.scene.metadata,
        )


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return dict(tomllib.loads(path.read_text()))
    if suffix in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    if suffix == ".json":
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    raise ValueError(f"Unsupported run config suffix {suffix!r}; expected .toml, .yaml, .yml, or .json")


def _resolve_relative(path: Path | None, base_dir: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_import_refs(references: tuple[str, ...], base_dir: Path) -> tuple[str, ...]:
    resolved: list[str] = []
    for reference in references:
        if _is_path_like_import(reference):
            path = Path(reference)
            resolved.append(str(path if path.is_absolute() else (base_dir / path).resolve()))
        else:
            resolved.append(reference)
    return tuple(resolved)


def _is_path_like_import(reference: str) -> bool:
    return reference.endswith(".py") or "/" in reference or "\\" in reference


def _import_extension(reference: str) -> None:
    if not _is_path_like_import(reference):
        importlib.import_module(reference)
        return

    path = Path(reference)
    if path.suffix != ".py":
        raise ValueError(f"Extension import path {reference!r} must point to a .py file")
    if not path.exists():
        raise FileNotFoundError(path)

    module_name = _module_name_for_path(path)
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extension module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _module_name_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"_retarget_plugin_{digest}"


def _append_missing_registry_messages(messages: list[str], registry: Registry[Any], keys: tuple[str, ...]) -> None:
    missing = registry.missing(keys)
    if not missing:
        return
    available = ", ".join(registry.names()) or "<none>"
    messages.append(f"{registry.name}: {', '.join(missing)} (available: {available})")


def _resolve_robot_options(options: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(options)
    for key in ("path", "store"):
        value = resolved.get(key)
        if value is None or value == "":
            continue
        resolved[key] = _resolve_relative(Path(str(value)), base_dir)
    return resolved


def _contact_plan_from_motion(motion: Any, contact_links: tuple[str, ...]) -> ContactPlan | None:
    if not motion.contacts:
        return None
    subjects = tuple(dict.fromkeys(name for frame in motion.contacts for name in frame))
    link_mapping = {subject: _links_for_contact_subject(subject, contact_links) for subject in subjects}
    metadata_provenance = motion.metadata.get("contact_provenance", {})
    return ContactPlan.from_binary_contacts(
        motion.contacts,
        link_mapping=link_mapping,
        support=_support_plane_from_metadata(motion.metadata),
        provenance={
            "source": "motion_sequence.contacts",
            "motion": motion.name,
            **(dict(metadata_provenance) if isinstance(metadata_provenance, dict) else {}),
        },
    )


def _links_for_contact_subject(subject: str, contact_links: tuple[str, ...]) -> tuple[str, ...]:
    lower = subject.lower()
    if "left" in lower or lower.startswith(("l_", "l-")):
        return tuple(link for link in contact_links if "left" in link.lower() or link.lower().startswith(("l_", "l-")))
    if "right" in lower or lower.startswith(("r_", "r-")):
        return tuple(
            link for link in contact_links if "right" in link.lower() or link.lower().startswith(("r_", "r-"))
        )
    return contact_links


def _support_plane_from_metadata(metadata: dict[str, Any]) -> SupportPlane | None:
    raw = metadata.get("support_plane")
    if isinstance(raw, dict):
        normal = raw.get("normal")
        origin = raw.get("origin")
        up_axis = int(raw.get("up_axis", 2))
    else:
        normal = metadata.get("support_plane_normal")
        origin = metadata.get("support_plane_origin")
        up_axis = int(metadata.get("support_plane_up_axis", 2))
    if normal is None or origin is None:
        return None
    return SupportPlane(
        normal=np.asarray(normal, dtype=np.float64),
        origin=np.asarray(origin, dtype=np.float64),
        up_axis=up_axis,
    )


def _default_objectives() -> tuple[ObjectiveSpec, ...]:
    return OptimizationProfile.defaults().objectives


def _default_constraints() -> tuple[ConstraintSpec, ...]:
    return OptimizationProfile.defaults().constraints


def _object_spec(config: ObjectConfig | None, *, frame_count: int, fps: float) -> ObjectSpec | None:
    if config is None:
        return None
    trajectory = _object_trajectory(config, frame_count=frame_count, fps=fps)
    return ObjectSpec(
        name=config.name,
        mesh_path=config.mesh_path,
        urdf_path=config.urdf_path,
        sample_points=_scene_points(
            config.sample_points,
            config.sample_points_path,
            config.sample_points_frame,
            mesh_path=config.mesh_path,
            mesh_sample_count=config.mesh_sample_count,
        ),
        trajectory=trajectory,
        metadata=config.metadata,
    )


def _terrain_spec(config: TerrainConfig | None) -> TerrainSpec | None:
    if config is None:
        return None
    return TerrainSpec(
        name=config.name,
        mesh_path=config.mesh_path,
        sample_points=_scene_points(
            config.sample_points,
            config.sample_points_path,
            config.sample_points_frame,
            mesh_path=config.mesh_path,
            mesh_sample_count=config.mesh_sample_count,
        ),
        metadata=config.metadata,
    )


def _object_trajectory(config: ObjectConfig, *, frame_count: int, fps: float) -> ObjectTrajectory | None:
    if config.trajectory_path is not None:
        poses = _load_pose_sequence(config.trajectory_path, fps=fps).to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
        return ObjectTrajectory(name=config.name, poses=poses)
    if config.trajectory_positions is not None or config.trajectory_quaternions is not None:
        positions = (
            np.asarray(config.trajectory_positions, dtype=np.float64)
            if config.trajectory_positions is not None
            else np.zeros((len(config.trajectory_quaternions or ()), 3), dtype=np.float64)
        )
        quaternions = (
            np.asarray(config.trajectory_quaternions, dtype=np.float64)
            if config.trajectory_quaternions is not None
            else _identity_quaternions(len(positions))
        )
        poses = PoseSequence.from_arrays(
            positions,
            quaternions,
            fps=fps,
            quaternion_order=config.trajectory_quaternion_order,
            frame=config.trajectory_frame,
        ).to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
        return ObjectTrajectory(name=config.name, poses=poses)
    if config.identity_trajectory:
        return ObjectTrajectory.identity(frame_count, fps=fps, name=config.name)
    return None


def _scene_points(
    inline_points: tuple[tuple[float, float, float], ...] | None,
    points_path: Path | None,
    frame: FrameConvention,
    *,
    mesh_path: Path | None = None,
    mesh_sample_count: int = 128,
) -> np.ndarray | None:
    if points_path is not None:
        points = _load_points(points_path)
    elif inline_points is not None:
        points = np.asarray(inline_points, dtype=np.float64)
    elif mesh_path is not None and mesh_sample_count > 0:
        points = sample_mesh_points(mesh_path, count=mesh_sample_count)
    else:
        return None
    return np.asarray(convert_points_frame(points, frame, FrameConvention.Z_UP_RIGHT_HANDED), dtype=np.float64)


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _validate_points(np.load(path))
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return _validate_points(_first_present_array(data, "sample_points", "points", "vertices"))
    if suffix == ".json":
        data = json.loads(path.read_text())
        values = data.get("sample_points", data.get("points", data)) if isinstance(data, dict) else data
        return _validate_points(values)
    if suffix == ".csv":
        rows = list(csv.DictReader(path.read_text().splitlines()))
        return _validate_points([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows])
    raise ValueError(f"Unsupported point file suffix {suffix!r}; expected .npy, .npz, .json, or .csv")


def _load_pose_sequence(path: Path, *, fps: float) -> PoseSequence:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        positions = np.asarray(np.load(path), dtype=np.float64)
        return _pose_sequence_from_arrays(positions, None, fps=fps)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        positions = _first_present_array(data, "positions", "translations", "object_positions")
        quaternions = _maybe_present_array(data, "quaternions", "rotations", "object_quaternions")
        file_fps = _scalar_float(data, "fps", fps)
        frame = _enum_value(data, "frame_convention", FrameConvention.Z_UP_RIGHT_HANDED)
        order = _enum_value(data, "quaternion_order", QuaternionOrder.WXYZ)
        return _pose_sequence_from_arrays(positions, quaternions, fps=file_fps, frame=frame, order=order)
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        json_positions: Any = data.get("positions", data.get("translations", data.get("object_positions")))
        if json_positions is None:
            raise KeyError(f"{path} must contain positions, translations, or object_positions")
        json_quaternions: Any | None = data.get("quaternions", data.get("rotations", data.get("object_quaternions")))
        frame = FrameConvention(str(data.get("frame_convention", data.get("frame", FrameConvention.Z_UP_RIGHT_HANDED))))
        order = QuaternionOrder(str(data.get("quaternion_order", QuaternionOrder.WXYZ)))
        return _pose_sequence_from_arrays(
            json_positions,
            json_quaternions,
            fps=float(data.get("fps", fps)),
            frame=frame,
            order=order,
        )
    if suffix == ".csv":
        rows = list(csv.DictReader(path.read_text().splitlines()))
        positions = np.asarray([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=np.float64)
        if rows and all(key in rows[0] for key in ("qw", "qx", "qy", "qz")):
            quaternions = np.asarray(
                [[float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])] for row in rows],
                dtype=np.float64,
            )
            order = QuaternionOrder.WXYZ
        elif rows and all(key in rows[0] for key in ("qx", "qy", "qz", "qw")):
            quaternions = np.asarray(
                [[float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])] for row in rows],
                dtype=np.float64,
            )
            order = QuaternionOrder.XYZW
        else:
            quaternions = None
            order = QuaternionOrder.WXYZ
        return _pose_sequence_from_arrays(positions, quaternions, fps=fps, order=order)
    raise ValueError(f"Unsupported trajectory file suffix {suffix!r}; expected .npy, .npz, .json, or .csv")


def _pose_sequence_from_arrays(
    positions: Any,
    quaternions: Any | None,
    *,
    fps: float,
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    order: QuaternionOrder = QuaternionOrder.WXYZ,
) -> PoseSequence:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("object trajectory positions must have shape (frames, 3)")
    quat = _identity_quaternions(len(pos)) if quaternions is None else np.asarray(quaternions, dtype=np.float64)
    return PoseSequence.from_arrays(pos, quat, fps=fps, quaternion_order=order, frame=frame)


def _identity_quaternions(frame_count: int) -> np.ndarray:
    quaternions = np.zeros((frame_count, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    return quaternions


def _validate_points(values: Any) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("sample points must have shape (N, 3)")
    return points


def _first_present_array(data: Any, *keys: str) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise KeyError(f"Expected one of {keys}")


def _maybe_present_array(data: Any, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    return None


def _scalar_float(data: Any, key: str, default: float) -> float:
    if key not in data:
        return default
    return float(np.asarray(data[key]).reshape(()))


def _enum_value(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode()
    return type(default)(str(value))
