"""Declarative frontends for the public experiment recipe hierarchy."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.capture import GvhmrOutputSource, ViconRecordingSource
from retarget.core.enums import (
    CropPolicy,
    FrameConvention,
    ObservationRecipeKind,
    QuaternionOrder,
    RetargetingRecipeKind,
    SolverBackend,
    TaskKind,
    TimelineSelection,
)
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.registry import Registry
from retarget.mesh import InteractionMeshSpec, sample_mesh_points
from retarget.motion import motion_formats
from retarget.optimization import (
    constraint_terms,
    objective_terms,
    validate_optimization_references,
)
from retarget.optimization.spec import (
    ConstraintConfig,
    ConstraintConfigUnion,
    ObjectiveConfig,
    ObjectiveConfigUnion,
    OptimizationProfile,
    SolverSpec,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline import (
    ObservationRecipe,
    Retargeter,
    RetargetingExperiment,
    RetargetingProblem,
    RetargetingRecipe,
)
from retarget.recipes import MotionFileObservationRecipe, RoleRetargetingRecipe
from retarget.recipes.holosoma import (
    HolosomaClimbObservationPolicy,
    HolosomaClimbObservationRecipe,
    HolosomaClimbOptimizationPolicy,
    HolosomaClimbRetargetingRecipe,
)
from retarget.recipes.skateboarding import (
    GVHMR_SCHEMA,
    VICON_SCHEMA,
    SkateboardingObservationRecipe,
    SkateboardingRetargetingRecipe,
)
from retarget.robots import robot_providers, robots
from retarget.robots.spec import RobotSpec
from retarget.scene import (
    ObjectSpec,
    ObjectTrajectory,
    ObjectVisualPart,
    SceneSpec,
    TerrainSpec,
)


class BaseObservationConfig(BaseModel):
    """Serializable configuration for one observation recipe."""

    model_config = ConfigDict(extra="forbid")

    kind: ObservationRecipeKind

    def resolve_paths(self, base_dir: Path) -> BaseObservationConfig:
        """Return a copy with relative paths resolved."""

        return self


class BaseAdaptationConfig(BaseModel):
    """Serializable configuration for one robot-adaptation recipe."""

    model_config = ConfigDict(extra="forbid")

    kind: RetargetingRecipeKind

    def resolve_paths(self, base_dir: Path) -> BaseAdaptationConfig:
        """Return a copy with relative paths resolved."""

        return self


class ObservationConfigBuilder(Protocol):
    """Deserialize one observation config into a public recipe."""

    config_type: type[BaseObservationConfig]

    def build(
        self,
        config: BaseObservationConfig,
        *,
        run_name: str | None,
    ) -> ObservationRecipe:
        """Build the observation recipe."""

    def validate_registry_references(
        self,
        config: BaseObservationConfig,
        messages: list[str],
    ) -> None:
        """Append invalid registry references."""


class AdaptationConfigBuilder(Protocol):
    """Deserialize one adaptation config into a public recipe."""

    config_type: type[BaseAdaptationConfig]

    def build(
        self,
        config: BaseAdaptationConfig,
        *,
        observation: BaseObservationConfig,
        robot: RobotSpec,
        run_name: str | None,
    ) -> RetargetingRecipe:
        """Build the robot-adaptation recipe."""

    def validate_registry_references(
        self,
        config: BaseAdaptationConfig,
        messages: list[str],
    ) -> None:
        """Append invalid registry references."""


observation_config_builders: Registry[ObservationConfigBuilder] = Registry("observation recipe config")
adaptation_config_builders: Registry[AdaptationConfigBuilder] = Registry("retargeting recipe config")


class MotionFileObservationConfig(BaseObservationConfig):
    """Load one file through a registered typed motion format."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: ObservationRecipeKind = ObservationRecipeKind.MOTION_FILE
    path: Path
    format_name: str = Field(default="minimal", alias="format")

    @field_validator("path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_paths(self, base_dir: Path) -> MotionFileObservationConfig:
        return self.model_copy(update={"path": _resolve_relative(self.path, base_dir)})


class SkateboardingObservationConfig(BaseObservationConfig):
    """Fuse native Vicon and GVHMR recordings in memory."""

    kind: ObservationRecipeKind = ObservationRecipeKind.SKATEBOARDING
    vicon: Path
    gvhmr: Path
    video_fps: float
    source_height_m: float | None = None
    max_frames: int | None = None
    timeline_selection: TimelineSelection = TimelineSelection.HUMAN_POSE
    crop_policy: CropPolicy = CropPolicy.OVERLAP
    uniform_fps: float | None = None

    @field_validator("vicon", "gvhmr", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("video_fps", "source_height_m", "uniform_fps")
    @classmethod
    def _positive_float(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("sampling rates and heights must be positive")
        return value

    @field_validator("max_frames")
    @classmethod
    def _positive_frames(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_frames must be positive")
        return value

    def resolve_paths(self, base_dir: Path) -> SkateboardingObservationConfig:
        return self.model_copy(
            update={
                "vicon": _resolve_relative(self.vicon, base_dir),
                "gvhmr": _resolve_relative(self.gvhmr, base_dir),
            }
        )


class HolosomaClimbObservationPolicyConfig(BaseModel):
    """Serializable Holosoma capture-processing policy."""

    model_config = ConfigDict(extra="forbid")

    source_height_m: float = Field(default=1.78, gt=0.0)
    mat_height_m: float = Field(default=0.1, ge=0.0)
    contact_velocity_threshold: float = Field(default=0.01, ge=0.0)
    object_sample_count: int = Field(default=100, gt=0)
    object_sample_seed: int = 42

    def build(self) -> HolosomaClimbObservationPolicy:
        return HolosomaClimbObservationPolicy(**self.model_dump())


class HolosomaClimbObservationConfig(BaseObservationConfig):
    """Load the public Holosoma climbing fixture as a native observation."""

    kind: ObservationRecipeKind = ObservationRecipeKind.HOLOSOMA_CLIMB
    holosoma_root: Path
    frame_count: int | None = None
    source_fps: float = Field(default=30.0, gt=0.0)
    downsample: int = Field(default=4, gt=0)
    policy: HolosomaClimbObservationPolicyConfig = Field(default_factory=HolosomaClimbObservationPolicyConfig)

    @field_validator("holosoma_root", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("frame_count")
    @classmethod
    def _positive_frames(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("frame_count must be positive")
        return value

    def resolve_paths(self, base_dir: Path) -> HolosomaClimbObservationConfig:
        return self.model_copy(update={"holosoma_root": _resolve_relative(self.holosoma_root, base_dir)})


ObjectiveConfigInput: TypeAlias = ObjectiveConfigUnion | dict[str, Any]
ConstraintConfigInput: TypeAlias = ConstraintConfigUnion | dict[str, Any]


class ObjectVisualPartConfig(BaseModel):
    """Serializable visual mesh part."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mesh_path: Path
    asset_scale: float | tuple[float, float, float] | None = None
    rgba: tuple[float, float, float, float] | None = None

    @field_validator("mesh_path", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_paths(self, base_dir: Path) -> ObjectVisualPartConfig:
        return self.model_copy(update={"mesh_path": _resolve_relative(self.mesh_path, base_dir)})


class ObjectConfig(BaseModel):
    """Serializable object scene specification."""

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
    def _sample_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mesh_sample_count must be non-negative")
        return value

    def resolve_paths(self, base_dir: Path) -> ObjectConfig:
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
    """Serializable terrain scene specification."""

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
    def _sample_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("mesh_sample_count must be non-negative")
        return value

    def resolve_paths(self, base_dir: Path) -> TerrainConfig:
        return self.model_copy(
            update={
                "mesh_path": _resolve_relative(self.mesh_path, base_dir),
                "sample_points_path": _resolve_relative(self.sample_points_path, base_dir),
            }
        )


class SceneConfig(BaseModel):
    """Serializable scene recipe fields."""

    model_config = ConfigDict(extra="forbid")

    object: ObjectConfig | None = None
    terrain: TerrainConfig | None = None
    ground_range: tuple[float, float] = (-1.0, 1.0)
    ground_size: int = 15
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_paths(self, base_dir: Path) -> SceneConfig:
        return self.model_copy(
            update={
                "object": (self.object.resolve_paths(base_dir) if self.object is not None else None),
                "terrain": (self.terrain.resolve_paths(base_dir) if self.terrain is not None else None),
            }
        )


class RoleMappingRecipeConfig(BaseAdaptationConfig):
    """Adapt typed motion joints through semantic robot roles."""

    kind: RetargetingRecipeKind = RetargetingRecipeKind.ROLE_MAPPING
    task_kind: TaskKind = TaskKind.ROBOT_ONLY
    joint_roles: dict[str, str] = Field(default_factory=dict)
    link_roles: dict[str, str] = Field(default_factory=dict)
    scale_to_robot: bool = True
    output_fps: float | None = None
    show_progress: bool = False
    mesh: InteractionMeshSpec = Field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    variables: QposVariableSpec = Field(default_factory=QposVariableSpec.actuated)
    objectives: tuple[ObjectiveConfigInput, ...] | None = None
    constraints: tuple[ConstraintConfigInput, ...] | None = None
    scene: SceneConfig = Field(default_factory=SceneConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolve_paths(self, base_dir: Path) -> RoleMappingRecipeConfig:
        return self.model_copy(update={"scene": self.scene.resolve_paths(base_dir)})

    def resolved_objectives(self) -> tuple[ObjectiveConfig, ...]:
        return _resolve_objective_configs(
            self.objectives if self.objectives is not None else OptimizationProfile.defaults().objectives
        )

    def resolved_constraints(self) -> tuple[ConstraintConfig, ...]:
        return _resolve_constraint_configs(
            self.constraints if self.constraints is not None else OptimizationProfile.defaults().constraints
        )


class SkateboardingAdaptationConfig(BaseAdaptationConfig):
    """Configure the public skateboarding adaptation recipe."""

    kind: RetargetingRecipeKind = RetargetingRecipeKind.SKATEBOARDING
    scale_to_robot: bool = False
    output_fps: float | None = None
    show_progress: bool = False
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL


class HolosomaClimbOptimizationPolicyConfig(BaseModel):
    """Serializable Holosoma optimization policy."""

    model_config = ConfigDict(extra="forbid")

    collision_activation_distance: float = Field(default=0.1, gt=0.0)
    qpos_regularization: tuple[tuple[int, float], ...] = ((19, 0.2), (20, 0.2))
    nominal_qpos_indices: tuple[int, ...] = tuple(range(19))

    def build(self) -> HolosomaClimbOptimizationPolicy:
        return HolosomaClimbOptimizationPolicy(
            collision_activation_distance=self.collision_activation_distance,
            qpos_regularization=self.qpos_regularization,
            nominal_qpos_indices=self.nominal_qpos_indices,
        )


class HolosomaClimbAdaptationConfig(BaseAdaptationConfig):
    """Configure the public Holosoma climbing adaptation recipe."""

    kind: RetargetingRecipeKind = RetargetingRecipeKind.HOLOSOMA_CLIMB
    holosoma_root: Path
    include_object_collision: bool = True
    show_progress: bool = False
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL
    policy: HolosomaClimbOptimizationPolicyConfig = Field(default_factory=HolosomaClimbOptimizationPolicyConfig)

    @field_validator("holosoma_root", mode="before")
    @classmethod
    def _path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_paths(self, base_dir: Path) -> HolosomaClimbAdaptationConfig:
        return self.model_copy(update={"holosoma_root": _resolve_relative(self.holosoma_root, base_dir)})


@dataclass(frozen=True)
class ConfiguredSceneRecipe:
    """Build a frame-aligned scene from serialized scene fields."""

    task_kind: TaskKind
    config: SceneConfig

    def build_scene(self, observation: Any) -> SceneSpec:
        fps = observation.timeline.nominal_fps
        if fps is None:
            raise ValueError("configured scenes require at least two observation samples")
        object_spec = _object_spec(
            self.config.object,
            frame_count=observation.timeline.sample_count,
            fps=fps,
        )
        terrain_spec = _terrain_spec(self.config.terrain)
        if self.task_kind == TaskKind.ROBOT_ONLY:
            return SceneSpec(
                task_kind=self.task_kind,
                terrain=terrain_spec or TerrainSpec(),
                ground_range=self.config.ground_range,
                ground_size=self.config.ground_size,
                metadata=self.config.metadata,
            )
        if self.task_kind == TaskKind.OBJECT_INTERACTION:
            return SceneSpec(
                task_kind=self.task_kind,
                object=object_spec or ObjectSpec(name="object"),
                terrain=terrain_spec,
                ground_range=self.config.ground_range,
                ground_size=self.config.ground_size,
                metadata=self.config.metadata,
            )
        return SceneSpec(
            task_kind=self.task_kind,
            object=object_spec,
            terrain=terrain_spec or TerrainSpec(name="terrain"),
            ground_range=self.config.ground_range,
            ground_size=self.config.ground_size,
            metadata=self.config.metadata,
        )


class RetargetingRunConfig(BaseModel):
    """Serialized constructor arguments for a `RetargetingExperiment`."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    observation: BaseObservationConfig
    recipe: BaseAdaptationConfig
    robot: str = "synthetic_humanoid"
    robot_provider: str = "registry"
    robot_options: dict[str, Any] = Field(default_factory=dict)
    output: Path
    imports: tuple[str, ...] = ()

    @field_validator("output", mode="before")
    @classmethod
    def _output_path(cls, value: Any) -> Path:
        return Path(value)

    @field_validator("observation", mode="before")
    @classmethod
    def _observation(cls, value: Any) -> BaseObservationConfig:
        if isinstance(value, BaseObservationConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("observation must be a typed mapping")
        kind = ObservationRecipeKind(str(value.get("kind", "")))
        return observation_config_builders.get(kind).config_type.model_validate(value)

    @field_validator("recipe", mode="before")
    @classmethod
    def _recipe(cls, value: Any) -> BaseAdaptationConfig:
        if isinstance(value, BaseAdaptationConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("recipe must be a typed mapping")
        kind = RetargetingRecipeKind(str(value.get("kind", "")))
        return adaptation_config_builders.get(kind).config_type.model_validate(value)

    @field_validator("imports", mode="before")
    @classmethod
    def _imports(cls, value: Any) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and resolve one TOML, YAML, or JSON experiment config."""

        config_path = Path(path)
        data = _load_mapping(config_path)
        raw_imports = data.get("imports", ())
        references = (str(raw_imports),) if isinstance(raw_imports, str) else tuple(str(value) for value in raw_imports)
        for reference in _resolve_import_refs(references, config_path.parent):
            _import_extension(reference)
        return cls.model_validate(data).resolve_paths(config_path.parent)

    def resolve_paths(self, base_dir: Path) -> Self:
        return self.model_copy(
            update={
                "observation": self.observation.resolve_paths(base_dir),
                "recipe": self.recipe.resolve_paths(base_dir),
                "output": _resolve_relative(self.output, base_dir),
                "imports": _resolve_import_refs(self.imports, base_dir),
                "robot_options": _resolve_robot_options(self.robot_options, base_dir),
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
        updates: dict[str, Any] = {}
        if motion is not None:
            current_format = (
                self.observation.format_name if isinstance(self.observation, MotionFileObservationConfig) else "minimal"
            )
            updates["observation"] = MotionFileObservationConfig(
                path=motion,
                format_name=format_name or current_format,
            )
        elif format_name is not None:
            if not isinstance(self.observation, MotionFileObservationConfig):
                raise ValueError("--format can only override motion-file observations")
            updates["observation"] = self.observation.model_copy(update={"format_name": format_name})
        if output is not None:
            updates["output"] = output
        if robot is not None:
            updates["robot"] = robot
            updates["robot_provider"] = "registry"
            updates["robot_options"] = {}
        if task_kind is not None:
            if not isinstance(self.recipe, RoleMappingRecipeConfig):
                raise ValueError("--task-kind can only override role-mapping recipes")
            updates["recipe"] = self.recipe.model_copy(update={"task_kind": task_kind})
        if name is not None:
            updates["name"] = name
        if show_progress is not None:
            updates["recipe"] = self.recipe.model_copy(update={"show_progress": show_progress})
        return self.model_copy(update=updates)

    def import_extensions(self) -> None:
        for reference in self.imports:
            _import_extension(reference)

    def load_robot(self) -> RobotSpec:
        """Resolve the configured target robot."""

        return robot_providers.get(self.robot_provider).load(self.robot, **self.robot_options)

    def build_experiment(
        self,
        *,
        retargeter: Retargeter | None = None,
        retargeter_factory: Callable[[RetargetingProblem], Retargeter] | None = None,
    ) -> RetargetingExperiment:
        """Build the same public experiment object used by Python workflows."""

        self.validate_registry_references()
        robot = self.load_robot()
        observation = observation_config_builders.get(self.observation.kind).build(self.observation, run_name=self.name)
        recipe = adaptation_config_builders.get(self.recipe.kind).build(
            self.recipe,
            observation=self.observation,
            robot=robot,
            run_name=self.name,
        )
        return RetargetingExperiment(
            observation=observation,
            recipe=recipe,
            robot=robot,
            retargeter=retargeter,
            retargeter_factory=retargeter_factory,
        )

    def build_problem(self) -> RetargetingProblem:
        """Observe and adapt through `RetargetingExperiment`."""

        return self.build_experiment().build_problem()

    def validate_registry_references(self) -> None:
        self.import_extensions()
        messages: list[str] = []
        observation_config_builders.get(self.observation.kind).validate_registry_references(self.observation, messages)
        adaptation_config_builders.get(self.recipe.kind).validate_registry_references(self.recipe, messages)
        _append_missing_registry_messages(messages, robot_providers, (self.robot_provider,))
        if self.robot_provider == "registry":
            _append_missing_registry_messages(messages, robots, (self.robot,))
        if messages:
            raise KeyError("Unknown experiment-config registry references: " + "; ".join(messages))


class MotionFileObservationBuilder:
    config_type: type[BaseObservationConfig] = MotionFileObservationConfig

    def build(
        self,
        config: BaseObservationConfig,
        *,
        run_name: str | None,
    ) -> ObservationRecipe:
        source = cast(MotionFileObservationConfig, config)
        return MotionFileObservationRecipe.registered(
            source.path,
            source.format_name,
            name=run_name,
        )

    def validate_registry_references(
        self,
        config: BaseObservationConfig,
        messages: list[str],
    ) -> None:
        source = cast(MotionFileObservationConfig, config)
        _append_missing_registry_messages(messages, motion_formats, (source.format_name,))


class SkateboardingObservationBuilder:
    config_type: type[BaseObservationConfig] = SkateboardingObservationConfig

    def build(
        self,
        config: BaseObservationConfig,
        *,
        run_name: str | None,
    ) -> ObservationRecipe:
        source = cast(SkateboardingObservationConfig, config)
        name = run_name or source.gvhmr.name
        return SkateboardingObservationRecipe(
            mocap=ViconRecordingSource(
                source.vicon,
                VICON_SCHEMA,
                name=name,
            ),
            human_pose=GvhmrOutputSource(
                source.gvhmr,
                GVHMR_SCHEMA,
                fps=source.video_fps,
                name=name,
                source_height_m=source.source_height_m,
            ),
            timeline_selection=source.timeline_selection,
            crop_policy=source.crop_policy,
            uniform_fps=source.uniform_fps,
            max_frames=source.max_frames,
        )

    def validate_registry_references(
        self,
        config: BaseObservationConfig,
        messages: list[str],
    ) -> None:
        del config, messages


class HolosomaObservationBuilder:
    config_type: type[BaseObservationConfig] = HolosomaClimbObservationConfig

    def build(
        self,
        config: BaseObservationConfig,
        *,
        run_name: str | None,
    ) -> ObservationRecipe:
        del run_name
        source = cast(HolosomaClimbObservationConfig, config)
        return HolosomaClimbObservationRecipe.from_fixture(
            source.holosoma_root,
            frame_count=source.frame_count,
            source_fps=source.source_fps,
            downsample=source.downsample,
            policy=source.policy.build(),
        )

    def validate_registry_references(
        self,
        config: BaseObservationConfig,
        messages: list[str],
    ) -> None:
        del config, messages


class RoleMappingAdaptationBuilder:
    config_type: type[BaseAdaptationConfig] = RoleMappingRecipeConfig

    def build(
        self,
        config: BaseAdaptationConfig,
        *,
        observation: BaseObservationConfig,
        robot: RobotSpec,
        run_name: str | None,
    ) -> RetargetingRecipe:
        recipe_config = cast(RoleMappingRecipeConfig, config)
        if not isinstance(observation, MotionFileObservationConfig):
            raise TypeError("role_mapping recipes currently require a motion_file observation")
        motion_format = motion_formats.get(observation.format_name)
        return RoleRetargetingRecipe(
            task_kind=recipe_config.task_kind,
            motion_format=motion_format,
            scene=ConfiguredSceneRecipe(
                task_kind=recipe_config.task_kind,
                config=recipe_config.scene,
            ),
            joint_roles={
                motion_format.joint_vocabulary(source): robot.role_vocabulary(target)
                for source, target in recipe_config.joint_roles.items()
            },
            link_roles={
                motion_format.joint_vocabulary(source): robot.role_vocabulary(target)
                for source, target in recipe_config.link_roles.items()
            },
            mesh=recipe_config.mesh,
            solver=recipe_config.solver,
            variables=recipe_config.variables,
            objectives=recipe_config.resolved_objectives(),
            constraints=recipe_config.resolved_constraints(),
            scale_to_robot=recipe_config.scale_to_robot,
            output_fps=recipe_config.output_fps,
            show_progress=recipe_config.show_progress,
            name=run_name,
            metadata=recipe_config.metadata,
        )

    def validate_registry_references(
        self,
        config: BaseAdaptationConfig,
        messages: list[str],
    ) -> None:
        recipe = cast(RoleMappingRecipeConfig, config)
        objective_inputs = (
            recipe.objectives if recipe.objectives is not None else OptimizationProfile.defaults().objectives
        )
        constraint_inputs = (
            recipe.constraints if recipe.constraints is not None else OptimizationProfile.defaults().constraints
        )
        _append_missing_registry_messages(
            messages,
            objective_terms,
            _config_kinds(objective_inputs, label="objective"),
        )
        _append_missing_registry_messages(
            messages,
            constraint_terms,
            _config_kinds(constraint_inputs, label="constraint"),
        )
        validate_optimization_references(
            solver=recipe.solver,
            objectives=recipe.resolved_objectives(),
            constraints=recipe.resolved_constraints(),
        )


class SkateboardingAdaptationBuilder:
    config_type: type[BaseAdaptationConfig] = SkateboardingAdaptationConfig

    def build(
        self,
        config: BaseAdaptationConfig,
        *,
        observation: BaseObservationConfig,
        robot: RobotSpec,
        run_name: str | None,
    ) -> RetargetingRecipe:
        del robot, run_name
        if not isinstance(observation, SkateboardingObservationConfig):
            raise TypeError("skateboarding adaptation requires a skateboarding observation")
        recipe = cast(SkateboardingAdaptationConfig, config)
        return SkateboardingRetargetingRecipe(
            scale_to_robot=recipe.scale_to_robot,
            output_fps=recipe.output_fps,
            show_progress=recipe.show_progress,
            solver_backend=recipe.solver_backend,
        )

    def validate_registry_references(
        self,
        config: BaseAdaptationConfig,
        messages: list[str],
    ) -> None:
        del config, messages


class HolosomaAdaptationBuilder:
    config_type: type[BaseAdaptationConfig] = HolosomaClimbAdaptationConfig

    def build(
        self,
        config: BaseAdaptationConfig,
        *,
        observation: BaseObservationConfig,
        robot: RobotSpec,
        run_name: str | None,
    ) -> RetargetingRecipe:
        del robot, run_name
        if not isinstance(observation, HolosomaClimbObservationConfig):
            raise TypeError("holosoma_climb adaptation requires a Holosoma observation")
        recipe = cast(HolosomaClimbAdaptationConfig, config)
        return HolosomaClimbRetargetingRecipe(
            holosoma_root=recipe.holosoma_root,
            include_object_collision=recipe.include_object_collision,
            solver_backend=recipe.solver_backend,
            show_progress=recipe.show_progress,
            optimization_policy=recipe.policy.build(),
        )

    def validate_registry_references(
        self,
        config: BaseAdaptationConfig,
        messages: list[str],
    ) -> None:
        del config, messages


observation_config_builders.register(ObservationRecipeKind.MOTION_FILE, MotionFileObservationBuilder())
observation_config_builders.register(ObservationRecipeKind.SKATEBOARDING, SkateboardingObservationBuilder())
observation_config_builders.register(ObservationRecipeKind.HOLOSOMA_CLIMB, HolosomaObservationBuilder())
adaptation_config_builders.register(RetargetingRecipeKind.ROLE_MAPPING, RoleMappingAdaptationBuilder())
adaptation_config_builders.register(RetargetingRecipeKind.SKATEBOARDING, SkateboardingAdaptationBuilder())
adaptation_config_builders.register(RetargetingRecipeKind.HOLOSOMA_CLIMB, HolosomaAdaptationBuilder())


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


def _resolve_import_refs(
    references: tuple[str, ...],
    base_dir: Path,
) -> tuple[str, ...]:
    return tuple(
        str(Path(reference) if Path(reference).is_absolute() else (base_dir / reference).resolve())
        if _is_path_like_import(reference)
        else reference
        for reference in references
    )


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
    module_name = "_retarget_plugin_" + hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
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


def _append_missing_registry_messages(
    messages: list[str],
    registry: Registry[Any],
    keys: tuple[str, ...],
) -> None:
    missing = registry.missing(keys)
    if missing:
        available = ", ".join(registry.names()) or "<none>"
        messages.append(f"{registry.name}: {', '.join(missing)} (available: {available})")


def _config_kinds(items: tuple[Any, ...], *, label: str) -> tuple[str, ...]:
    kinds: list[str] = []
    for item in items:
        if isinstance(item, (ObjectiveConfig, ConstraintConfig)):
            kinds.append(item.kind)
        elif isinstance(item, dict) and isinstance(item.get("kind"), str):
            kinds.append(item["kind"])
        else:
            raise ValueError(f"{label} config entries must include a non-empty kind")
    return tuple(kinds)


def _resolve_objective_configs(
    items: tuple[Any, ...],
) -> tuple[ObjectiveConfig, ...]:
    return tuple(
        item
        if isinstance(item, ObjectiveConfig)
        else objective_terms.get(str(item["kind"])).config_type.model_validate(item)
        for item in items
    )


def _resolve_constraint_configs(
    items: tuple[Any, ...],
) -> tuple[ConstraintConfig, ...]:
    return tuple(
        item
        if isinstance(item, ConstraintConfig)
        else constraint_terms.get(str(item["kind"])).config_type.model_validate(item)
        for item in items
    )


def _resolve_robot_options(
    options: dict[str, Any],
    base_dir: Path,
) -> dict[str, Any]:
    resolved = dict(options)
    for key in ("path", "store", "holosoma_root"):
        value = resolved.get(key)
        if value not in (None, ""):
            resolved[key] = _resolve_relative(Path(str(value)), base_dir)
    return resolved


def _object_spec(
    config: ObjectConfig | None,
    *,
    frame_count: int,
    fps: float,
) -> ObjectSpec | None:
    if config is None:
        return None
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
        trajectory=_object_trajectory(
            config,
            frame_count=frame_count,
            fps=fps,
        ),
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


def _object_trajectory(
    config: ObjectConfig,
    *,
    frame_count: int,
    fps: float,
) -> ObjectTrajectory | None:
    if config.trajectory_path is not None:
        poses = _load_pose_sequence(config.trajectory_path, fps=fps).to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
        return ObjectTrajectory(name=config.name, poses=poses)
    if config.trajectory_positions is not None or config.trajectory_quaternions is not None:
        positions = (
            np.asarray(config.trajectory_positions, dtype=np.float64)
            if config.trajectory_positions is not None
            else np.zeros(
                (len(config.trajectory_quaternions or ()), 3),
                dtype=np.float64,
            )
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
    return np.asarray(
        convert_points_frame(points, frame, FrameConvention.Z_UP_RIGHT_HANDED),
        dtype=np.float64,
    )


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _validate_points(np.load(path))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
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
        return _pose_sequence_from_arrays(np.load(path), None, fps=fps)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            positions = _first_present_array(data, "positions", "translations", "object_positions")
            quaternions = _maybe_present_array(data, "quaternions", "rotations", "object_quaternions")
            file_fps = _scalar_float(data, "fps", fps)
            frame = _enum_value(
                data,
                "frame_convention",
                FrameConvention.Z_UP_RIGHT_HANDED,
            )
            order = _enum_value(data, "quaternion_order", QuaternionOrder.WXYZ)
        return _pose_sequence_from_arrays(
            positions,
            quaternions,
            fps=file_fps,
            frame=frame,
            order=order,
        )
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        json_positions: Any = data.get("positions", data.get("translations", data.get("object_positions")))
        if json_positions is None:
            raise KeyError(f"{path} must contain positions, translations, or object_positions")
        quaternions = data.get(
            "quaternions",
            data.get("rotations", data.get("object_quaternions")),
        )
        frame = FrameConvention(
            str(
                data.get(
                    "frame_convention",
                    data.get("frame", FrameConvention.Z_UP_RIGHT_HANDED),
                )
            )
        )
        order = QuaternionOrder(str(data.get("quaternion_order", QuaternionOrder.WXYZ)))
        return _pose_sequence_from_arrays(
            json_positions,
            quaternions,
            fps=float(data.get("fps", fps)),
            frame=frame,
            order=order,
        )
    if suffix == ".csv":
        rows = list(csv.DictReader(path.read_text().splitlines()))
        positions = np.asarray(
            [[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows],
            dtype=np.float64,
        )
        quaternions = None
        order = QuaternionOrder.WXYZ
        if rows and all(key in rows[0] for key in ("qw", "qx", "qy", "qz")):
            quaternions = np.asarray(
                [
                    [
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )
        return _pose_sequence_from_arrays(
            positions,
            quaternions,
            fps=fps,
            order=order,
        )
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
    return PoseSequence.from_arrays(
        pos,
        quat,
        fps=fps,
        quaternion_order=order,
        frame=frame,
    )


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
    return default if key not in data else float(np.asarray(data[key]).reshape(()))


def _enum_value(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode()
    return type(default)(str(value))
