"""Typed optimization configuration models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Self, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.enums import (
    Constraint,
    ConstraintKind,
    ContactSubject,
    ConvergenceMode,
    NominalFallback,
    NonPenetrationSource,
    Objective,
    ObjectiveKind,
    RobotGeometry,
    RobotJoint,
    RobotLink,
    SceneGeometry,
    SolverBackend,
    SolverKind,
)


class SolverSpec(BaseModel):
    """Solver selection and common options."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    backend: SolverKind = SolverBackend.AUTO
    max_iterations: int = 10
    first_frame_iterations: int | None = None
    trust_radius: float = 0.2
    tolerance: float = 1e-6
    convergence: ConvergenceMode = ConvergenceMode.STEP_NORM
    cost_atol: float = 1e-8
    cost_rtol: float = 1e-5
    verbose: bool = False

    @field_validator("backend", mode="before")
    @classmethod
    def _deserialize_builtin_backend(cls, value: object) -> SolverKind:
        if isinstance(value, SolverKind):
            return value
        return SolverBackend(str(value))

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
        return value

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
    """Base class for objective configs keyed by an extensible enum member."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: ClassVar[ObjectiveKind]
    weight: float = 1.0

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

    kind = Objective.LAPLACIAN


class LinkTrackingObjectiveConfig(ObjectiveConfig):
    """Track robot links to a typed target plan."""

    kind = Objective.LINK_TRACKING
    weight_scale: float = 1.0

    @field_validator("weight_scale")
    @classmethod
    def _positive_weight_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("weight_scale must be positive")
        return float(value)


class SmoothnessObjectiveConfig(ObjectiveConfig):
    """Penalize changes from the previous frame."""

    kind = Objective.SMOOTHNESS


class NominalTrackingObjectiveConfig(ObjectiveConfig):
    """Track a nominal posture for typed robot joints or explicit qpos coordinates."""

    kind = Objective.NOMINAL_TRACKING
    weight: float = 5.0
    joints: tuple[RobotJoint, ...] = ()
    qpos_indices: tuple[int, ...] = ()
    fallback: NominalFallback = NominalFallback.ZERO

    @field_validator("joints")
    @classmethod
    def _one_joint_vocabulary(cls, value: tuple[RobotJoint, ...]) -> tuple[RobotJoint, ...]:
        if value and not all(type(joint) is type(value[0]) for joint in value):
            raise TypeError("nominal joints must use one RobotJoint vocabulary")
        return value

    @field_validator("qpos_indices", mode="before")
    @classmethod
    def _coerce_qpos_indices(cls, value: object) -> tuple[int, ...]:
        return _integer_tuple(value, label="qpos_indices")


class DiagonalRegularizationObjectiveConfig(ObjectiveConfig):
    """Penalize selected qpos variables toward zero with diagonal weights."""

    kind = Objective.DIAGONAL_REGULARIZATION
    qpos_weights: tuple[float, ...] = ()
    variable_weights: tuple[float, ...] = ()
    qpos_weight_overrides: dict[int, float] = Field(default_factory=dict)

    @field_validator("qpos_weights", "variable_weights", mode="before")
    @classmethod
    def _coerce_weight_tuple(cls, value: object) -> tuple[float, ...]:
        return _float_tuple(value)

    @field_validator("qpos_weights", "variable_weights")
    @classmethod
    def _non_negative_weights(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(item < 0.0 for item in value):
            raise ValueError("diagonal weights must be non-negative")
        return value

    @field_validator("qpos_weight_overrides", mode="before")
    @classmethod
    def _coerce_overrides(cls, value: object) -> dict[int, float]:
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise ValueError("qpos_weight_overrides must be a mapping")
        out = {int(index): float(weight) for index, weight in value.items()}
        if any(index < 0 or weight < 0 for index, weight in out.items()):
            raise ValueError("qpos weight overrides must be non-negative")
        return out


class ConstraintConfig(BaseModel):
    """Base class for constraint configs keyed by an extensible enum member."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: ClassVar[ConstraintKind]
    enabled: bool = True

    def with_enabled(self, enabled: bool = True) -> Self:
        """Return a copy with a different enabled state."""

        return self.model_copy(update={"enabled": enabled})

    def disabled(self) -> Self:
        """Return a disabled copy."""

        return self.with_enabled(False)


class JointLimitsConstraintConfig(ConstraintConfig):
    """Box limits on actuated joints."""

    kind = Constraint.JOINT_LIMITS


class TrustRegionConstraintConfig(ConstraintConfig):
    """Cap the Euclidean norm of each SQP update."""

    kind = Constraint.TRUST_REGION
    radius: float | None = None

    @field_validator("radius")
    @classmethod
    def _positive_radius(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("radius must be positive")
        return value


class FootStickingConstraintConfig(ConstraintConfig):
    """Constrain active support links near their previous tangent-plane position."""

    kind = Constraint.FOOT_STICKING
    tolerance: float = 1e-3

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tolerance must be positive")
        return float(value)


class FootLockWindow(BaseModel):
    """Exact typed subject or link lock window."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: ContactSubject | None = None
    link: RobotLink | None = None
    ranges: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def _validate_window(self) -> FootLockWindow:
        if (self.subject is None) == (self.link is None):
            raise ValueError("foot lock window requires exactly one of subject or link")
        _validate_ranges(self.ranges)
        return self


class FootLockConstraintConfig(ConstraintConfig):
    """Pin exact typed support links or subjects during configured windows."""

    kind = Constraint.FOOT_LOCK
    windows: tuple[FootLockWindow, ...] = ()
    z_floor: float = 0.0
    tolerance: float = 5e-3

    @field_validator("tolerance")
    @classmethod
    def _positive_tolerance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tolerance must be positive")
        return float(value)


class GeometryPair(BaseModel):
    """Explicit typed geometry pair for distance constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    first: RobotGeometry | SceneGeometry
    second: RobotGeometry | SceneGeometry

    @model_validator(mode="after")
    def _different_geometries(self) -> GeometryPair:
        if type(self.first) is type(self.second) and self.first.value == self.second.value:
            raise ValueError("geometry pair members must differ")
        return self


class NonPenetrationConstraintConfig(ConstraintConfig):
    """Separate selected typed robot links from support and explicit geometry."""

    kind = Constraint.NON_PENETRATION
    links: tuple[RobotLink, ...] = ()
    subjects: tuple[ContactSubject, ...] = ()
    sources: tuple[NonPenetrationSource, ...] = (
        NonPenetrationSource.SUPPORT,
        NonPenetrationSource.SCENE_POINTS,
        NonPenetrationSource.GEOMETRY,
    )
    geometry_pairs: tuple[GeometryPair, ...] = ()
    tolerance: float = 1e-3
    floor_z: float = 0.0
    scene_clearance: float = 0.02
    activation_distance: float | None = None

    @field_validator("links")
    @classmethod
    def _one_link_vocabulary(cls, value: tuple[RobotLink, ...]) -> tuple[RobotLink, ...]:
        if value and not all(type(link) is type(value[0]) for link in value):
            raise TypeError("non-penetration links must use one RobotLink vocabulary")
        return value

    @field_validator("subjects")
    @classmethod
    def _one_subject_vocabulary(
        cls,
        value: tuple[ContactSubject, ...],
    ) -> tuple[ContactSubject, ...]:
        if value and not all(type(subject) is type(value[0]) for subject in value):
            raise TypeError("non-penetration subjects must use one ContactSubject vocabulary")
        return value

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, value: object) -> tuple[NonPenetrationSource, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, Iterable):
            raise ValueError("sources must be iterable")
        return tuple(dict.fromkeys(NonPenetrationSource(str(item)) for item in value))

    @field_validator("tolerance", "scene_clearance")
    @classmethod
    def _positive_distance(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return float(value)

    @field_validator("activation_distance")
    @classmethod
    def _positive_optional_distance(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("activation_distance must be positive")
        return value

    @model_validator(mode="after")
    def _require_explicit_selection(self) -> NonPenetrationConstraintConfig:
        point_sources = {
            NonPenetrationSource.SUPPORT,
            NonPenetrationSource.SCENE_POINTS,
        }
        if point_sources.intersection(self.sources) and not (self.links or self.subjects):
            raise ValueError("support and scene-point non-penetration require explicit links or subjects")
        if NonPenetrationSource.GEOMETRY in self.sources and not self.geometry_pairs:
            raise ValueError("geometry non-penetration requires explicit geometry_pairs")
        return self


class SelfCollisionConstraintConfig(ConstraintConfig):
    """Maintain minimum separation between explicit typed robot geometry pairs."""

    kind = Constraint.SELF_COLLISION
    pairs: tuple[GeometryPair, ...] = ()
    minimum_distance: float = 0.02
    margin: float | None = None
    windows: tuple[tuple[int, int], ...] | None = None

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
        return value

    @field_validator("windows")
    @classmethod
    def _valid_windows(cls, value: tuple[tuple[int, int], ...] | None) -> tuple[tuple[int, int], ...] | None:
        if value is not None:
            _validate_ranges(value)
        return value

    @model_validator(mode="after")
    def _require_pairs(self) -> SelfCollisionConstraintConfig:
        if not self.pairs:
            raise ValueError("self-collision requires explicit typed geometry pairs")
        return self


ObjectiveConfigUnion: TypeAlias = (
    LaplacianObjectiveConfig
    | LinkTrackingObjectiveConfig
    | SmoothnessObjectiveConfig
    | NominalTrackingObjectiveConfig
    | DiagonalRegularizationObjectiveConfig
)
ConstraintConfigUnion: TypeAlias = (
    JointLimitsConstraintConfig
    | TrustRegionConstraintConfig
    | FootStickingConstraintConfig
    | FootLockConstraintConfig
    | NonPenetrationConstraintConfig
    | SelfCollisionConstraintConfig
)
ConfigT = TypeVar("ConfigT", bound=ObjectiveConfig | ConstraintConfig)


class OptimizationProfile(BaseModel):
    """Reusable typed objective/constraint composition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "custom"
    objectives: tuple[ObjectiveConfig, ...] = ()
    constraints: tuple[ConstraintConfig, ...] = ()
    provenance: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_profile(self) -> OptimizationProfile:
        if not self.name:
            raise ValueError("name must not be empty")
        return self

    @classmethod
    def defaults(cls, *, name: str = "default") -> OptimizationProfile:
        """Return default robot-only optimization terms."""

        return cls(
            name=name,
            objectives=(LaplacianObjectiveConfig(weight=10.0), SmoothnessObjectiveConfig(weight=0.2)),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(),
                FootStickingConstraintConfig(tolerance=1e-3),
            ),
        )

    @property
    def objective_kinds(self) -> tuple[ObjectiveKind, ...]:
        """Objective kinds in profile order."""

        return tuple(objective.kind for objective in self.objectives)

    @property
    def constraint_kinds(self) -> tuple[ConstraintKind, ...]:
        """Constraint kinds in profile order."""

        return tuple(constraint.kind for constraint in self.constraints)

    def objective(self, kind: ObjectiveKind) -> ObjectiveConfig | None:
        """Return the last objective matching ``kind``."""

        return _last_kind(self.objectives, kind)

    def constraint(self, kind: ConstraintKind) -> ConstraintConfig | None:
        """Return the last constraint matching ``kind``."""

        return _last_kind(self.constraints, kind)

    def with_objective(self, objective: ObjectiveConfig, *, replace: bool = True) -> Self:
        objectives = _replace_kind(self.objectives, objective) if replace else (*self.objectives, objective)
        return self.model_copy(update={"objectives": objectives})

    def without_objective(self, *kinds: ObjectiveKind) -> Self:
        return self.model_copy(update={"objectives": _without_kind(self.objectives, kinds)})

    def with_constraint(self, constraint: ConstraintConfig, *, replace: bool = True) -> Self:
        constraints = _replace_kind(self.constraints, constraint) if replace else (*self.constraints, constraint)
        return self.model_copy(update={"constraints": constraints})

    def without_constraint(self, *kinds: ConstraintKind) -> Self:
        return self.model_copy(update={"constraints": _without_kind(self.constraints, kinds)})

    def validate_registry_references(self, solver: SolverSpec | None = None) -> None:
        from retarget.optimization.validation import validate_optimization_references

        validate_optimization_references(
            solver=solver or SolverSpec(),
            objectives=self.objectives,
            constraints=self.constraints,
        )


def serialize_config(config: ObjectiveConfig | ConstraintConfig) -> dict[str, object]:
    """Return a JSON-ready config with its class-level typed discriminator."""

    return {"kind": config.kind.value, **config.model_dump(mode="json")}


def _last_kind(items: tuple[ConfigT, ...], kind: ObjectiveKind | ConstraintKind) -> ConfigT | None:
    return next((item for item in reversed(items) if item.kind == kind), None)


def _replace_kind(items: tuple[ConfigT, ...], new_item: ConfigT) -> tuple[ConfigT, ...]:
    return (*tuple(item for item in items if item.kind != new_item.kind), new_item)


def _without_kind(
    items: tuple[ConfigT, ...],
    kinds: tuple[ObjectiveKind, ...] | tuple[ConstraintKind, ...],
) -> tuple[ConfigT, ...]:
    blocked = set(kinds)
    return tuple(item for item in items if item.kind not in blocked)


def _integer_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, int):
        value = (value,)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an integer or iterable of integers")
    result = tuple(int(item) for item in value)
    if any(index < 0 for index in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} entries must be unique and non-negative")
    return result


def _float_tuple(value: object) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, int | float):
        value = (value,)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("weights must be a number or iterable of numbers")
    return tuple(float(item) for item in value)


def _validate_ranges(ranges: tuple[tuple[int, int], ...]) -> None:
    for start, end in ranges:
        if start < 0 or end < start:
            raise ValueError("window ranges must satisfy 0 <= start <= end")


__all__ = [
    "ConstraintConfig",
    "ConstraintConfigUnion",
    "DiagonalRegularizationObjectiveConfig",
    "FootLockConstraintConfig",
    "FootLockWindow",
    "FootStickingConstraintConfig",
    "GeometryPair",
    "JointLimitsConstraintConfig",
    "LaplacianObjectiveConfig",
    "LinkTrackingObjectiveConfig",
    "NominalTrackingObjectiveConfig",
    "NonPenetrationConstraintConfig",
    "ObjectiveConfig",
    "ObjectiveConfigUnion",
    "OptimizationProfile",
    "SelfCollisionConstraintConfig",
    "SmoothnessObjectiveConfig",
    "SolverSpec",
    "TrustRegionConstraintConfig",
    "serialize_config",
]
