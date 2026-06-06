"""Typed optimization configuration models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, Self, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.enums import (
    Constraint,
    ConvergenceMode,
    GeometrySource,
    NominalFallback,
    NonPenetrationSource,
    Objective,
    SolverBackend,
)


class SolverSpec(BaseModel):
    """Solver selection and common options."""

    model_config = ConfigDict(extra="forbid")

    backend: SolverBackend | str = SolverBackend.AUTO
    max_iterations: int = 10
    first_frame_iterations: int | None = None
    trust_radius: float = 0.2
    tolerance: float = 1e-6
    convergence: ConvergenceMode = ConvergenceMode.STEP_NORM
    cost_atol: float = 1e-8
    cost_rtol: float = 1e-5
    verbose: bool = False

    @property
    def backend_name(self) -> str:
        """Registry key for the selected solver backend."""

        return self.backend.value if isinstance(self.backend, SolverBackend) else self.backend

    @field_validator("max_iterations")
    @classmethod
    def _positive_iterations(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_iterations must be positive")
        return int(value)

    @field_validator("first_frame_iterations")
    @classmethod
    def _positive_optional_iterations(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("first_frame_iterations must be positive")
        return None if value is None else int(value)

    @field_validator("trust_radius", "tolerance", "cost_atol")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return float(value)

    @field_validator("cost_rtol")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cost_rtol must be non-negative")
        return float(value)


class ObjectiveConfig(BaseModel):
    """Base class for weighted objective term configs."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    weight: float = 1.0

    @property
    def name(self) -> str:
        """Registry key for this objective config."""

        return self.kind

    def with_weight(self, weight: float) -> Self:
        """Return a copy with a different objective weight."""

        return self.model_copy(update={"weight": weight})

    @field_validator("weight")
    @classmethod
    def _non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weight must be non-negative")
        return float(value)


class LaplacianObjectiveConfig(ObjectiveConfig):
    """Interaction-mesh Laplacian deformation objective."""

    kind: Literal["laplacian"] = Objective.LAPLACIAN.value


class LinkTrackingObjectiveConfig(ObjectiveConfig):
    """Track robot links to a typed target plan."""

    kind: Literal["link_tracking"] = Objective.LINK_TRACKING.value
    weight_scale: float = 1.0

    @field_validator("weight_scale")
    @classmethod
    def _positive_weight_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight_scale must be positive")
        return float(value)


class SmoothnessObjectiveConfig(ObjectiveConfig):
    """Penalize changes from the previous frame."""

    kind: Literal["smoothness"] = Objective.SMOOTHNESS.value


class NominalTrackingObjectiveConfig(ObjectiveConfig):
    """Track the robot's nominal posture for selected joints."""

    kind: Literal["nominal_tracking"] = Objective.NOMINAL_TRACKING.value
    weight: float = 5.0
    joint_names: tuple[str, ...] = ()
    qpos_indices: tuple[int, ...] = ()
    fallback: NominalFallback = NominalFallback.ZERO

    @field_validator("joint_names", mode="before")
    @classmethod
    def _coerce_joint_names(cls, value: object) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Iterable):
            raise ValueError("joint_names must be a string or iterable of strings")
        return tuple(str(item) for item in value)

    @field_validator("qpos_indices", mode="before")
    @classmethod
    def _coerce_qpos_indices(cls, value: object) -> tuple[int, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, int):
            return (int(value),)
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("qpos_indices must be an integer or iterable of integers")
        return tuple(int(item) for item in value)

    @field_validator("qpos_indices")
    @classmethod
    def _non_negative_qpos_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(index < 0 for index in value):
            raise ValueError("qpos_indices entries must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("qpos_indices entries must be unique")
        return value


class DiagonalRegularizationObjectiveConfig(ObjectiveConfig):
    """Penalize selected qpos variables toward zero with diagonal weights."""

    kind: Literal["diagonal_regularization"] = Objective.DIAGONAL_REGULARIZATION.value
    qpos_weights: tuple[float, ...] = ()
    variable_weights: tuple[float, ...] = ()
    qpos_weight_overrides: dict[int, float] = Field(default_factory=dict)

    @field_validator("qpos_weights", "variable_weights", mode="before")
    @classmethod
    def _coerce_weight_tuple(cls, value: object) -> tuple[float, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, int | float):
            return (float(value),)
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("weights must be a number or iterable of numbers")
        return tuple(float(item) for item in value)

    @field_validator("qpos_weights", "variable_weights")
    @classmethod
    def _non_negative_weights(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item < 0.0 for item in value):
            raise ValueError("diagonal weights must be non-negative")
        return value

    @field_validator("qpos_weight_overrides", mode="before")
    @classmethod
    def _coerce_qpos_weight_overrides(cls, value: object) -> dict[int, float]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("qpos_weight_overrides must be a mapping from qpos index to weight")
        out: dict[int, float] = {}
        for key, weight in value.items():
            index = int(key)
            if index < 0:
                raise ValueError("qpos_weight_overrides keys must be non-negative")
            out[index] = float(weight)
        return out

    @field_validator("qpos_weight_overrides")
    @classmethod
    def _non_negative_weight_overrides(cls, value: dict[int, float]) -> dict[int, float]:
        if any(weight < 0.0 for weight in value.values()):
            raise ValueError("qpos_weight_overrides values must be non-negative")
        return value


class ConstraintConfig(BaseModel):
    """Base class for constraint term configs."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    enabled: bool = True

    @property
    def name(self) -> str:
        """Registry key for this constraint config."""

        return self.kind

    def with_enabled(self, enabled: bool = True) -> Self:
        """Return a copy with a different enabled state."""

        return self.model_copy(update={"enabled": enabled})

    def disabled(self) -> Self:
        """Return a disabled copy."""

        return self.with_enabled(False)


ConfigT = TypeVar("ConfigT", bound=ObjectiveConfig | ConstraintConfig)


class JointLimitsConstraintConfig(ConstraintConfig):
    """Box limits on actuated joints."""

    kind: Literal["joint_limits"] = Constraint.JOINT_LIMITS.value


class TrustRegionConstraintConfig(ConstraintConfig):
    """Cap the Euclidean norm of each SQP update."""

    kind: Literal["trust_region"] = Constraint.TRUST_REGION.value
    radius: float | None = None

    @field_validator("radius")
    @classmethod
    def _positive_radius(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("radius must be positive")
        return None if value is None else float(value)


class FootStickingConstraintConfig(ConstraintConfig):
    """Constrain active support links near their previous tangent-plane position."""

    kind: Literal["foot_sticking"] = Constraint.FOOT_STICKING.value
    tolerance: float = 1e-3

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tolerance must be positive")
        return float(value)


class FootLockConstraintConfig(ConstraintConfig):
    """Pin support links to a support plane or explicit frame windows."""

    kind: Literal["foot_lock"] = Constraint.FOOT_LOCK.value
    windows: dict[str, tuple[tuple[int, int], ...]] = Field(default_factory=dict)
    z_floor: float = 0.0
    tolerance: float = 5e-3

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tolerance must be positive")
        return float(value)

    @field_validator("windows", mode="before")
    @classmethod
    def _coerce_windows(cls, value: object) -> dict[str, tuple[tuple[int, int], ...]]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("windows must be a mapping from subject/link to frame ranges")
        out: dict[str, tuple[tuple[int, int], ...]] = {}
        for key, ranges in value.items():
            if not isinstance(ranges, Iterable) or isinstance(ranges, (str, bytes)):
                raise ValueError("foot lock windows must contain iterable frame ranges")
            normalized: list[tuple[int, int]] = []
            for item in ranges:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("foot lock windows must contain (start, end) pairs")
                start, end = int(item[0]), int(item[1])
                if end < start:
                    raise ValueError("foot lock window end must be >= start")
                normalized.append((start, end))
            out[str(key)] = tuple(normalized)
        return out


class NonPenetrationConstraintConfig(ConstraintConfig):
    """Separate robot links from support and scene geometry."""

    kind: Literal["non_penetration"] = Constraint.NON_PENETRATION.value
    links: tuple[str, ...] = ()
    sources: tuple[NonPenetrationSource, ...] = (
        NonPenetrationSource.SUPPORT,
        NonPenetrationSource.SCENE_POINTS,
        NonPenetrationSource.GEOMETRY,
    )
    geometry_source: GeometrySource = GeometrySource.EXPLICIT
    geometry_pairs: tuple[tuple[str, str], ...] = ()
    scene_geometry_keywords: tuple[str, ...] = ()
    excluded_geometry_keyword_pairs: tuple[tuple[str, str], ...] = ()
    tolerance: float = 1e-3
    floor_z: float = 0.0
    scene_clearance: float = 0.02
    activation_distance: float | None = None

    @field_validator("links", mode="before")
    @classmethod
    def _coerce_links(cls, value: object) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Iterable):
            raise ValueError("links must be a string or iterable of strings")
        return tuple(str(item) for item in value)

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, value: object) -> tuple[NonPenetrationSource, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, Iterable):
            raise ValueError("sources must be a string or iterable of strings")
        normalized: list[NonPenetrationSource] = []
        for item in value:
            try:
                source = NonPenetrationSource(str(item))
            except ValueError as exc:
                raise ValueError(
                    "non-penetration sources must be one of 'support', 'scene_points', or 'geometry'"
                ) from exc
            if source not in normalized:
                normalized.append(source)
        return tuple(normalized)

    @field_validator("geometry_pairs", mode="before")
    @classmethod
    def _coerce_geometry_pairs(cls, value: object) -> tuple[tuple[str, str], ...]:
        if value in (None, ""):
            return ()
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("geometry_pairs must be iterable")
        pairs: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("geometry_pairs entries must contain exactly two names")
            pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)

    @field_validator("scene_geometry_keywords", mode="before")
    @classmethod
    def _coerce_scene_geometry_keywords(cls, value: object) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, Iterable):
            raise ValueError("scene_geometry_keywords must be a string or iterable of strings")
        return tuple(str(item) for item in value)

    @field_validator("excluded_geometry_keyword_pairs", mode="before")
    @classmethod
    def _coerce_excluded_geometry_keyword_pairs(cls, value: object) -> tuple[tuple[str, str], ...]:
        if value in (None, ""):
            return ()
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("excluded_geometry_keyword_pairs must be iterable")
        pairs: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("excluded_geometry_keyword_pairs entries must contain exactly two names")
            pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)

    @field_validator("tolerance", "scene_clearance")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return float(value)

    @field_validator("activation_distance")
    @classmethod
    def _positive_optional_float(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("activation_distance must be positive")
        return None if value is None else float(value)


class SelfCollisionConstraintConfig(ConstraintConfig):
    """Maintain minimum separation between configured robot geometry pairs."""

    kind: Literal["self_collision"] = Constraint.SELF_COLLISION.value
    pairs: tuple[tuple[str, str], ...] = ()
    minimum_distance: float = 0.02
    margin: float | None = None
    windows: tuple[tuple[int, int], ...] | None = None

    @field_validator("pairs", mode="before")
    @classmethod
    def _coerce_pairs(cls, value: object) -> tuple[tuple[str, str], ...]:
        if value in (None, ""):
            return ()
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("self-collision pairs must be iterable")
        pairs: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("self-collision pairs must contain exactly two names")
            pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)

    @field_validator("minimum_distance")
    @classmethod
    def _positive_distance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("minimum_distance must be positive")
        return float(value)

    @field_validator("margin")
    @classmethod
    def _positive_margin(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("margin must be positive")
        return None if value is None else float(value)

    @field_validator("windows", mode="before")
    @classmethod
    def _coerce_windows(cls, value: object) -> tuple[tuple[int, int], ...] | None:
        if value in (None, ""):
            return None
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ValueError("self-collision windows must be iterable")
        out: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("self-collision windows must contain (start, end) pairs")
            start, end = int(item[0]), int(item[1])
            if end < start:
                raise ValueError("self-collision window end must be >= start")
            out.append((start, end))
        return tuple(out)


ObjectiveConfigUnion: TypeAlias = Annotated[
    LaplacianObjectiveConfig
    | LinkTrackingObjectiveConfig
    | SmoothnessObjectiveConfig
    | NominalTrackingObjectiveConfig
    | DiagonalRegularizationObjectiveConfig,
    Field(discriminator="kind"),
]
ConstraintConfigUnion: TypeAlias = Annotated[
    JointLimitsConstraintConfig
    | TrustRegionConstraintConfig
    | FootStickingConstraintConfig
    | FootLockConstraintConfig
    | NonPenetrationConstraintConfig
    | SelfCollisionConstraintConfig,
    Field(discriminator="kind"),
]


class OptimizationProfile(BaseModel):
    """Reusable typed objective/constraint composition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "custom"
    objectives: tuple[ObjectiveConfig, ...] = ()
    constraints: tuple[ConstraintConfig, ...] = ()
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile(self) -> OptimizationProfile:
        if not self.name:
            raise ValueError("name must not be empty")
        return self

    @classmethod
    def defaults(cls, *, name: str = "default") -> OptimizationProfile:
        """Return the default robot-only optimization profile."""

        return cls(
            name=name,
            objectives=(
                LaplacianObjectiveConfig(weight=10.0),
                SmoothnessObjectiveConfig(weight=0.2),
            ),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(),
                FootStickingConstraintConfig(tolerance=1e-3),
            ),
        )

    @classmethod
    def object_interaction(
        cls,
        *,
        scene_clearance: float = 0.03,
        links: tuple[str, ...] = (),
        floor_z: float = 0.0,
    ) -> OptimizationProfile:
        """Return a default profile with scene non-penetration enabled."""

        return cls.defaults(name="object_interaction").with_constraint(
            NonPenetrationConstraintConfig(
                links=links,
                floor_z=floor_z,
                scene_clearance=scene_clearance,
            )
        )

    @classmethod
    def climbing(
        cls,
        *,
        scene_clearance: float = 0.025,
        floor_z: float = 0.0,
    ) -> OptimizationProfile:
        """Return a default profile for terrain/climbing experiments."""

        return cls.defaults(name="climbing").with_constraint(
            NonPenetrationConstraintConfig(floor_z=floor_z, scene_clearance=scene_clearance)
        )

    @classmethod
    def holosoma_compatible(
        cls,
        *,
        geometry_pairs: tuple[tuple[str, str], ...] = (),
        qpos_weights: tuple[float, ...] = (),
        variable_weights: tuple[float, ...] = (),
    ) -> OptimizationProfile:
        """Return Holosoma-aligned objective and constraint defaults."""

        return cls(
            name="holosoma_compatible",
            objectives=(
                LaplacianObjectiveConfig(weight=10.0),
                SmoothnessObjectiveConfig(weight=0.2),
                NominalTrackingObjectiveConfig(weight=5.0, fallback=NominalFallback.CURRENT),
                DiagonalRegularizationObjectiveConfig(
                    weight=1.0,
                    qpos_weights=qpos_weights,
                    variable_weights=variable_weights,
                ),
            ),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(radius=0.2),
                FootStickingConstraintConfig(tolerance=1e-3),
                NonPenetrationConstraintConfig(
                    sources=(NonPenetrationSource.GEOMETRY,),
                    tolerance=1e-3,
                    scene_clearance=1e-3,
                    geometry_pairs=geometry_pairs,
                ),
            ),
        )

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Objective names in profile order."""

        return tuple(objective.kind for objective in self.objectives)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        """Constraint names in profile order."""

        return tuple(constraint.kind for constraint in self.constraints)

    def objective(self, kind: str) -> ObjectiveConfig | None:
        """Return the last objective matching `kind`, if present."""

        return _last_kind(self.objectives, kind)

    def constraint(self, kind: str) -> ConstraintConfig | None:
        """Return the last constraint matching `kind`, if present."""

        return _last_kind(self.constraints, kind)

    def with_objective(self, objective: ObjectiveConfig, *, replace: bool = True) -> Self:
        """Return a copy with an objective appended or replaced by kind."""

        objectives = _replace_kind(self.objectives, objective) if replace else (*self.objectives, objective)
        return self.model_copy(update={"objectives": objectives})

    def without_objective(self, *kinds: str) -> Self:
        """Return a copy without objectives matching any supplied kind."""

        return self.model_copy(update={"objectives": _without_kind(self.objectives, kinds)})

    def with_constraint(self, constraint: ConstraintConfig, *, replace: bool = True) -> Self:
        """Return a copy with a constraint appended or replaced by kind."""

        constraints = _replace_kind(self.constraints, constraint) if replace else (*self.constraints, constraint)
        return self.model_copy(update={"constraints": constraints})

    def without_constraint(self, *kinds: str) -> Self:
        """Return a copy without constraints matching any supplied kind."""

        return self.model_copy(update={"constraints": _without_kind(self.constraints, kinds)})

    def validate_registry_references(self, solver: SolverSpec | None = None) -> None:
        """Validate this profile against registered optimization terms."""

        from retarget.optimization.validation import validate_optimization_references

        validate_optimization_references(
            solver=solver or SolverSpec(),
            objectives=self.objectives,
            constraints=self.constraints,
        )


def _last_kind(items: tuple[ConfigT, ...], kind: str) -> ConfigT | None:
    for item in reversed(items):
        if item.kind == kind:
            return item
    return None


def _replace_kind(items: tuple[ConfigT, ...], new_item: ConfigT) -> tuple[ConfigT, ...]:
    kind = new_item.kind
    out: list[ConfigT] = []
    replaced = False
    for item in items:
        if item.kind == kind:
            if not replaced:
                out.append(new_item)
                replaced = True
            continue
        out.append(item)
    if not replaced:
        out.append(new_item)
    return tuple(out)


def _without_kind(items: tuple[ConfigT, ...], kinds: tuple[str, ...]) -> tuple[ConfigT, ...]:
    blocked = set(kinds)
    return tuple(item for item in items if item.kind not in blocked)
