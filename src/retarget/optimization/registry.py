"""Optimization extension registries."""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import cast

from retarget.core.protocols import ConstraintTerm, ObjectiveTerm, Solver
from retarget.core.registry import Registry
from retarget.optimization.spec import SolverSpec

SolverFactory = Callable[[SolverSpec], Solver]


def _objective_term_from_decorator(value: object) -> ObjectiveTerm:
    candidate = value
    if isclass(value) or not isinstance(value, ObjectiveTerm):
        if not callable(value):
            raise TypeError("objective term registrations must implement ObjectiveTerm or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, ObjectiveTerm):
        raise TypeError("objective term registrations must implement ObjectiveTerm")
    return candidate


def _constraint_term_from_decorator(value: object) -> ConstraintTerm:
    candidate = value
    if isclass(value) or not isinstance(value, ConstraintTerm):
        if not callable(value):
            raise TypeError("constraint term registrations must implement ConstraintTerm or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, ConstraintTerm):
        raise TypeError("constraint term registrations must implement ConstraintTerm")
    return candidate


objective_terms: Registry[ObjectiveTerm] = Registry(
    "objective term",
    decorator_transform=_objective_term_from_decorator,
)
constraint_terms: Registry[ConstraintTerm] = Registry(
    "constraint term",
    decorator_transform=_constraint_term_from_decorator,
)
solver_factories: Registry[SolverFactory] = Registry("solver factory")

__all__ = ["SolverFactory", "constraint_terms", "objective_terms", "solver_factories"]
