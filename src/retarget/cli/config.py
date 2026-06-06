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
from typing import Any, Protocol, Self, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.enums import FrameConvention, QuaternionOrder, RunSourceKind, TaskKind
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.registry import Registry
from retarget.mesh import InteractionMeshSpec, sample_mesh_points
from retarget.motion import load_motion, motion_formats
from retarget.motion.contact import ContactPlan
from retarget.optimization import constraint_terms, objective_terms, validate_optimization_references
from retarget.optimization.spec import (
    ConstraintConfig,
    ConstraintConfigUnion,
    ObjectiveConfig,
    ObjectiveConfigUnion,
    OptimizationProfile,
    SolverSpec,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline import PreparedRetargetingInputs, RetargetingProblem
from retarget.robots import robot_providers, robots
from retarget.robots.spec import RobotSpec
from retarget.scene import ObjectSpec, ObjectTrajectory, ObjectVisualPart, SceneSpec, TerrainSpec


class BaseRunSourceConfig(BaseModel):
    """Base class for typed run-source config blocks."""

    model_config = ConfigDict(extra="forbid")

    kind: RunSourceKind

    def resolve_paths(self, base_dir: Path) -> BaseRunSourceConfig:
        """Return a copy with relative paths resolved."""

        return self


class RunSourceBuilder(Protocol):
    """Build prepared retarget inputs from one run-source config type."""

    config_type: type[BaseRunSourceConfig]

    def validate_registry_references(self, config: BaseRunSourceConfig, messages: list[str]) -> None:
        """Append missing registry-reference messages for this source."""

    def prepare(
        self,
        config: BaseRunSourceConfig,
        robot: RobotSpec,
        *,
        run_name: str | None,
        run_config: Any,
    ) -> PreparedRetargetingInputs:
        """Load and adapt source data for a run."""


run_sources: Registry[RunSourceBuilder] = Registry("run source")


class MotionFileSourceConfig(BaseRunSourceConfig):
    """Load a motion file through a registered motion format."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: RunSourceKind = RunSourceKind.MOTION_FILE
    path: Path
    format_name: str = Field(default="minimal", alias="format")

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_paths(self, base_dir: Path) -> MotionFileSourceConfig:
        """Return a copy with relative paths resolved."""

        return self.model_copy(update={"path": _resolve_relative(self.path, base_dir)})


class MotionSyncSkateboardingSourceConfig(BaseRunSourceConfig):
    """Prepare a skateboarding ``motion_sync`` clip directly."""

    model_config = ConfigDict(extra="forbid")

    kind: RunSourceKind = RunSourceKind.MOTION_SYNC_SKATEBOARDING
    demo: str = "pushoff5_twoshoes"
    synced: Path | None = None
    synced_root: Path = Path("motion_sync_output/synced")
    max_frames: int | None = None
    height_m: float | None = None
    force_contacts: bool = False
    save_contact_layer: bool = False

    @field_validator("synced", "synced_root", mode="before")
    @classmethod
    def _coerce_optional_path(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)

    @field_validator("max_frames")
    @classmethod
    def _positive_max_frames(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_frames must be positive")
        return value

    @field_validator("height_m")
    @classmethod
    def _positive_height(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("height_m must be positive")
        return None if value is None else float(value)

    def resolve_paths(self, base_dir: Path) -> MotionSyncSkateboardingSourceConfig:
        """Return a copy with relative paths resolved."""

        return self.model_copy(
            update={
                "synced": _resolve_relative(self.synced, base_dir),
                "synced_root": _resolve_relative(self.synced_root, base_dir),
            }
        )

    @property
    def synced_path(self) -> Path:
        """Resolved synced clip path."""

        if self.synced is not None:
            return self.synced
        return self.synced_root / self.demo


ObjectiveConfigInput: TypeAlias = ObjectiveConfigUnion | dict[str, Any]
ConstraintConfigInput: TypeAlias = ConstraintConfigUnion | dict[str, Any]


class ObjectVisualPartConfig(BaseModel):
    """Serializable visual mesh part for object playback."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mesh_path: Path
    asset_scale: float | tuple[float, float, float] | None = None
    rgba: tuple[float, float, float, float] | None = None

    @field_validator("mesh_path", mode="before")
    @classmethod
    def _coerce_mesh_path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_paths(self, base_dir: Path) -> ObjectVisualPartConfig:
        """Return a copy with relative asset paths resolved against `base_dir`."""

        return self.model_copy(update={"mesh_path": _resolve_relative(self.mesh_path, base_dir)})


class ObjectConfig(BaseModel):
    """Serializable object-scene options for CLI run specs."""

    model_config = ConfigDict(extra="forbid")

    name: str = "object"
    mesh_path: Path | None = None
    urdf_path: Path | None = None
    asset_scale: float | tuple[float, float, float] = 1.0
    visual_parts: tuple[ObjectVisualPartConfig, ...] = ()
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
                "visual_parts": tuple(part.resolve_paths(base_dir) for part in self.visual_parts),
                "sample_points_path": _resolve_relative(self.sample_points_path, base_dir),
                "trajectory_path": _resolve_relative(self.trajectory_path, base_dir),
            }
        )


class TerrainConfig(BaseModel):
    """Serializable terrain-scene options for CLI run specs."""

    model_config = ConfigDict(extra="forbid")

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
    """Serializable scene options for CLI run specs."""

    model_config = ConfigDict(extra="forbid")

    object: ObjectConfig | None = None
    terrain: TerrainConfig | None = None
    ground_range: tuple[float, float] = (-1.0, 1.0)
    ground_size: int = 15
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_paths(self, base_dir: Path) -> SceneConfig:
        """Return a copy with relative file paths resolved against `base_dir`."""

        return self.model_copy(
            update={
                "object": self.object.resolve_paths(base_dir) if self.object is not None else None,
                "terrain": self.terrain.resolve_paths(base_dir) if self.terrain is not None else None,
            }
        )


class RetargetingRunConfig(BaseModel):
    """Human-editable run spec used by the CLI."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = None
    source: BaseRunSourceConfig
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
    variables: QposVariableSpec = Field(default_factory=QposVariableSpec.actuated)
    objectives: tuple[ObjectiveConfigInput, ...] | None = None
    constraints: tuple[ConstraintConfigInput, ...] | None = None
    scene: SceneConfig = Field(default_factory=SceneConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: Any) -> BaseRunSourceConfig:
        if isinstance(value, BaseRunSourceConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("source must be a typed source mapping")
        raw_kind = value.get("kind")
        if raw_kind is None:
            raise ValueError("source.kind is required")
        kind = RunSourceKind(str(raw_kind))
        builder = run_sources.get(kind)
        return builder.config_type.model_validate(value)

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
                "source": self.source.resolve_paths(base_dir),
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
            current_format = self.source.format_name if isinstance(self.source, MotionFileSourceConfig) else "minimal"
            updates["source"] = MotionFileSourceConfig(path=motion, format_name=format_name or current_format)
        elif format_name is not None:
            if not isinstance(self.source, MotionFileSourceConfig):
                raise ValueError("--format can only override motion_file sources")
            updates["source"] = self.source.model_copy(update={"format_name": format_name})
        if output is not None:
            updates["output"] = output
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
        robot_spec = robot_providers.get(self.robot_provider).load(self.robot, **self.robot_options)
        prepared = self._build_inputs(robot_spec)
        return RetargetingProblem(
            name=self.name or prepared.motion.name,
            task_kind=self.task_kind,
            robot=robot_spec,
            motion=prepared.motion,
            scene=prepared.scene,
            contacts=prepared.contacts,
            targets=prepared.targets,
            nominal_qpos=prepared.nominal_qpos,
            motion_format=prepared.motion_format,
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            variables=self.variables,
            objectives=self.resolved_objectives(),
            constraints=self.resolved_constraints(),
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps,
            show_progress=self.show_progress,
            metadata={**self.metadata, **prepared.metadata},
        )

    def import_extensions(self) -> None:
        """Import explicitly configured extension modules."""

        for reference in self.imports:
            _import_extension(reference)

    def resolved_objectives(self) -> tuple[ObjectiveConfig, ...]:
        """Return config entries validated through registered objective terms."""

        return _resolve_objective_configs(self.objectives if self.objectives is not None else _default_objectives())

    def resolved_constraints(self) -> tuple[ConstraintConfig, ...]:
        """Return config entries validated through registered constraint terms."""

        return _resolve_constraint_configs(self.constraints if self.constraints is not None else _default_constraints())

    def validate_registry_references(self) -> None:
        """Validate named extension references before loading files or assets."""

        self.import_extensions()
        messages: list[str] = []
        run_sources.get(self.source.kind).validate_registry_references(self.source, messages)
        _append_missing_registry_messages(messages, robot_providers, (self.robot_provider,))
        if self.robot_provider == "registry":
            _append_missing_registry_messages(messages, robots, (self.robot,))
        objective_inputs = self.objectives if self.objectives is not None else _default_objectives()
        constraint_inputs = self.constraints if self.constraints is not None else _default_constraints()
        _append_missing_registry_messages(messages, objective_terms, _config_kinds(objective_inputs, label="objective"))
        _append_missing_registry_messages(
            messages,
            constraint_terms,
            _config_kinds(constraint_inputs, label="constraint"),
        )
        if messages:
            raise KeyError("Unknown run-config registry references: " + "; ".join(messages))
        validate_optimization_references(
            solver=self.solver,
            objectives=self.resolved_objectives(),
            constraints=self.resolved_constraints(),
        )

    def _build_inputs(self, robot: RobotSpec) -> PreparedRetargetingInputs:
        return run_sources.get(self.source.kind).prepare(self.source, robot, run_name=self.name, run_config=self)

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


class MotionFileSourceBuilder:
    """Build retarget inputs from a registered motion file loader."""

    config_type: type[BaseRunSourceConfig] = MotionFileSourceConfig

    def validate_registry_references(self, config: BaseRunSourceConfig, messages: list[str]) -> None:
        source = cast(MotionFileSourceConfig, config)
        _append_missing_registry_messages(messages, motion_formats, (source.format_name,))

    def prepare(
        self,
        config: BaseRunSourceConfig,
        robot: RobotSpec,
        *,
        run_name: str | None,
        run_config: Any,
    ) -> PreparedRetargetingInputs:
        source = cast(MotionFileSourceConfig, config)
        motion_format = motion_formats.get(source.format_name)
        motion = load_motion(source.path, source.format_name, name=run_name)
        return PreparedRetargetingInputs(
            motion=motion,
            scene=run_config._build_scene(frame_count=motion.frame_count, fps=motion.fps),
            contacts=_contact_plan_from_motion(motion, robot.contact_links),
            motion_format=motion_format,
        )


class MotionSyncSkateboardingSourceBuilder:
    """Build retarget inputs from the bundled skateboarding motion_sync recipe."""

    config_type: type[BaseRunSourceConfig] = MotionSyncSkateboardingSourceConfig

    def validate_registry_references(self, _config: BaseRunSourceConfig, _messages: list[str]) -> None:
        return

    def prepare(
        self,
        config: BaseRunSourceConfig,
        robot: RobotSpec,
        *,
        run_name: str | None,
        run_config: Any,
    ) -> PreparedRetargetingInputs:
        del run_config
        source = cast(MotionSyncSkateboardingSourceConfig, config)
        from retarget.integrations.motion_sync.skateboarding import from_skateboarding_clip

        prepared = from_skateboarding_clip(
            source.synced_path,
            name=run_name or source.demo,
            max_frames=source.max_frames,
            height_m=source.height_m,
            force_contacts=source.force_contacts,
            save_contact_layer=source.save_contact_layer,
            contact_links=robot.contact_links,
        )
        return PreparedRetargetingInputs(
            motion=prepared.motion,
            scene=prepared.scene,
            contacts=prepared.contacts,
            targets=prepared.targets,
            nominal_qpos=prepared.nominal_qpos,
            motion_format=prepared.motion_format,
            metadata=dict(prepared.metadata),
        )


run_sources.register(RunSourceKind.MOTION_FILE, MotionFileSourceBuilder())
run_sources.register(RunSourceKind.MOTION_SYNC_SKATEBOARDING, MotionSyncSkateboardingSourceBuilder())


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


def _config_kinds(items: tuple[Any, ...], *, label: str) -> tuple[str, ...]:
    kinds: list[str] = []
    for item in items:
        if isinstance(item, (ObjectiveConfig, ConstraintConfig)):
            kinds.append(item.kind)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"{label} config entries must be typed configs or mappings")
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{label} config entries must include a non-empty kind")
        kinds.append(kind)
    return tuple(kinds)


def _resolve_objective_configs(items: tuple[Any, ...]) -> tuple[ObjectiveConfig, ...]:
    configs: list[ObjectiveConfig] = []
    for item in items:
        if isinstance(item, ObjectiveConfig):
            configs.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError("objective config entries must be typed configs or mappings")
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("objective config entries must include a non-empty kind")
        term = objective_terms.get(kind)
        configs.append(term.config_type.model_validate(item))
    return tuple(configs)


def _resolve_constraint_configs(items: tuple[Any, ...]) -> tuple[ConstraintConfig, ...]:
    configs: list[ConstraintConfig] = []
    for item in items:
        if isinstance(item, ConstraintConfig):
            configs.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError("constraint config entries must be typed configs or mappings")
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("constraint config entries must include a non-empty kind")
        term = constraint_terms.get(kind)
        configs.append(term.config_type.model_validate(item))
    return tuple(configs)


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
    contact_provenance = motion.contact_provenance
    return ContactPlan.from_binary_contacts(
        motion.contacts,
        link_mapping=link_mapping,
        support=motion.support,
        provenance={
            "source": "motion_sequence.contacts",
            "motion": motion.name,
            **dict(contact_provenance),
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


def _default_objectives() -> tuple[ObjectiveConfig, ...]:
    return OptimizationProfile.defaults().objectives


def _default_constraints() -> tuple[ConstraintConfig, ...]:
    return OptimizationProfile.defaults().constraints


def _object_spec(config: ObjectConfig | None, *, frame_count: int, fps: float) -> ObjectSpec | None:
    if config is None:
        return None
    trajectory = _object_trajectory(config, frame_count=frame_count, fps=fps)
    return ObjectSpec(
        name=config.name,
        mesh_path=config.mesh_path,
        urdf_path=config.urdf_path,
        asset_scale=config.asset_scale,
        visual_parts=tuple(
            ObjectVisualPart(
                name=part.name,
                mesh_path=part.mesh_path,
                asset_scale=part.asset_scale,
                rgba=part.rgba,
            )
            for part in config.visual_parts
        ),
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
