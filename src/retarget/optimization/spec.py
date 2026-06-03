"""Optimization configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.enums import Constraint, Objective, SolverBackend


class SolverSpec(BaseModel):
    """Solver selection and common options.

    Attributes:
        backend (SolverBackend | str): Registry key or enum; ``auto`` picks CVXPY when installed.
        max_iterations (int): Outer SQP iterations budget per frame (scaled for the first frame).
        trust_radius (float): Default trust-region radius for subproblem steps (meters/radians in joint space).
        tolerance (float): Stop an inner solve when the joint increment norm falls below this value.
        verbose (bool): Enable verbose logging for backends that support it (for example CVXPY).
    """

    backend: SolverBackend | str = SolverBackend.AUTO
    max_iterations: int = 10
    trust_radius: float = 0.2
    tolerance: float = 1e-6
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
        return value

    @field_validator("trust_radius", "tolerance")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return float(value)


class ObjectiveSpec(BaseModel):
    """Declarative objective term configuration.

    Attributes:
        name (str): Registered objective term key (see ``Objective`` enum values).
        weight (float): Non-negative scalar multiplier on the term's least-squares residual.
        parameters (dict[str, Any]): Term-specific options passed to the registered builder.
    """

    name: str
    weight: float = 1.0
    parameters: dict[str, Any] = Field(default_factory=dict)

    def with_weight(self, weight: float) -> ObjectiveSpec:
        """Return a copy with a different objective weight."""

        return ObjectiveSpec(name=self.name, weight=weight, parameters=dict(self.parameters))

    def with_parameters(self, **parameters: Any) -> ObjectiveSpec:
        """Return a copy with merged term parameters."""

        return ObjectiveSpec(name=self.name, weight=self.weight, parameters={**self.parameters, **parameters})

    @field_validator("weight")
    @classmethod
    def _non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weight must be non-negative")
        return float(value)


class ConstraintSpec(BaseModel):
    """Declarative constraint term configuration.

    Attributes:
        name (str): Registered constraint term key (see ``Constraint`` enum values).
        enabled (bool): When ``False``, the term is skipped when assembling subproblems.
        parameters (dict[str, Any]): Term-specific options (clearances, link lists, lock windows).
    """

    name: str
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)

    def with_parameters(self, **parameters: Any) -> ConstraintSpec:
        """Return a copy with merged term parameters."""

        return ConstraintSpec(name=self.name, enabled=self.enabled, parameters={**self.parameters, **parameters})

    def with_enabled(self, enabled: bool = True) -> ConstraintSpec:
        """Return a copy with the enabled flag changed."""

        return ConstraintSpec(name=self.name, enabled=enabled, parameters=dict(self.parameters))

    def disabled(self) -> ConstraintSpec:
        """Return a disabled copy of this constraint."""

        return self.with_enabled(False)


class OptimizationProfile(BaseModel):
    """Reusable objective/constraint composition for retargeting experiments.

    Attributes:
        name (str): Profile label stored in problem metadata when applied.
        objectives (tuple[ObjectiveSpec, ...]): Ordered objective terms for a run.
        constraints (tuple[ConstraintSpec, ...]): Ordered constraint terms for a run.
        metadata (dict[str, Any]): Opaque tags describing the profile or experiment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "custom"
    objectives: tuple[ObjectiveSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

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
                ObjectiveSpec(name=Objective.LAPLACIAN, weight=10.0),
                ObjectiveSpec(name=Objective.SMOOTHNESS, weight=0.2),
            ),
            constraints=(
                ConstraintSpec(name=Constraint.JOINT_LIMITS),
                ConstraintSpec(name=Constraint.TRUST_REGION),
                ConstraintSpec(
                    name=Constraint.FOOT_CONTACT,
                    parameters={"velocity_threshold": 0.02, "tolerance": 1e-3},
                ),
            ),
        )

    @classmethod
    def object_interaction(
        cls,
        *,
        scene_clearance: float = 0.03,
        links: tuple[str, ...] = ("left_toe", "right_toe", "left_hand", "right_hand"),
        floor_z: float = 0.0,
    ) -> OptimizationProfile:
        """Return a default profile with scene non-penetration enabled."""

        return cls.defaults(name="object_interaction").with_constraint(
            Constraint.NON_PENETRATION,
            parameters={"floor_z": floor_z, "scene_clearance": scene_clearance, "links": links},
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
            Constraint.NON_PENETRATION,
            parameters={"floor_z": floor_z, "scene_clearance": scene_clearance},
        )

    @property
    def objective_names(self) -> tuple[str, ...]:
        """Objective names in profile order."""

        return tuple(objective.name for objective in self.objectives)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        """Constraint names in profile order."""

        return tuple(constraint.name for constraint in self.constraints)

    def objective(self, name: str) -> ObjectiveSpec | None:
        """Return the last objective matching `name`, if present."""

        return _last_named(self.objectives, name)

    def constraint(self, name: str) -> ConstraintSpec | None:
        """Return the last constraint matching `name`, if present."""

        return _last_named(self.constraints, name)

    def with_objective(
        self,
        objective: ObjectiveSpec | str,
        *,
        weight: float | None = None,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = True,
    ) -> Self:
        """Return a copy with an objective appended or replaced by name."""

        spec = _objective_spec(objective, weight=weight, parameters=parameters)
        objectives = _replace_named(self.objectives, spec) if replace else (*self.objectives, spec)
        return self.model_copy(update={"objectives": objectives})

    def without_objective(self, *names: str) -> Self:
        """Return a copy without objectives matching any supplied name."""

        return self.model_copy(update={"objectives": _without_named(self.objectives, names)})

    def with_constraint(
        self,
        constraint: ConstraintSpec | str,
        *,
        enabled: bool | None = None,
        parameters: Mapping[str, Any] | None = None,
        replace: bool = True,
    ) -> Self:
        """Return a copy with a constraint appended or replaced by name."""

        spec = _constraint_spec(constraint, enabled=enabled, parameters=parameters)
        constraints = _replace_named(self.constraints, spec) if replace else (*self.constraints, spec)
        return self.model_copy(update={"constraints": constraints})

    def without_constraint(self, *names: str) -> Self:
        """Return a copy without constraints matching any supplied name."""

        return self.model_copy(update={"constraints": _without_named(self.constraints, names)})

    def validate_registry_references(self, solver: SolverSpec | None = None) -> None:
        """Validate this profile against registered optimization terms."""

        from retarget.optimization.validation import validate_optimization_references

        validate_optimization_references(
            solver=solver or SolverSpec(),
            objectives=self.objectives,
            constraints=self.constraints,
        )


def _objective_spec(
    objective: ObjectiveSpec | str,
    *,
    weight: float | None,
    parameters: Mapping[str, Any] | None,
) -> ObjectiveSpec:
    if isinstance(objective, ObjectiveSpec):
        spec = objective
        if weight is not None:
            spec = spec.with_weight(weight)
        if parameters:
            spec = spec.with_parameters(**dict(parameters))
        return spec
    return ObjectiveSpec(name=objective, weight=1.0 if weight is None else weight, parameters=dict(parameters or {}))


def _constraint_spec(
    constraint: ConstraintSpec | str,
    *,
    enabled: bool | None,
    parameters: Mapping[str, Any] | None,
) -> ConstraintSpec:
    if isinstance(constraint, ConstraintSpec):
        spec = constraint
        if enabled is not None:
            spec = spec.with_enabled(enabled)
        if parameters:
            spec = spec.with_parameters(**dict(parameters))
        return spec
    return ConstraintSpec(
        name=constraint,
        enabled=True if enabled is None else enabled,
        parameters=dict(parameters or {}),
    )


NamedSpec = TypeVar("NamedSpec", ObjectiveSpec, ConstraintSpec)


def _replace_named(items: tuple[NamedSpec, ...], item: NamedSpec) -> tuple[NamedSpec, ...]:
    replaced = False
    updated: list[NamedSpec] = []
    for existing in items:
        if existing.name != item.name:
            updated.append(existing)
            continue
        if not replaced:
            updated.append(item)
            replaced = True
    if not replaced:
        updated.append(item)
    return tuple(updated)


def _without_named(
    items: tuple[NamedSpec, ...],
    names: tuple[str, ...],
) -> tuple[NamedSpec, ...]:
    remove = set(names)
    return tuple(item for item in items if item.name not in remove)


def _last_named(
    items: tuple[NamedSpec, ...],
    name: str,
) -> NamedSpec | None:
    for item in reversed(items):
        if item.name == name:
            return item
    return None
