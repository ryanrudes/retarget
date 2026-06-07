"""Optimization extension registries."""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import Any, cast

from retarget.core.enums import ConstraintKind, ObjectiveKind, SolverKind
from retarget.core.protocols import ConstraintTerm, ObjectiveTerm, Solver
from retarget.core.registry import Registry
from retarget.optimization.spec import SolverSpec

SolverFactory = Callable[[SolverSpec], Solver]
"""Build a :class:`~retarget.core.protocols.Solver` from a :class:`~retarget.optimization.spec.SolverSpec`."""


def _objective_term_from_decorator(value: object) -> ObjectiveTerm[Any]:
    candidate = value
    if isclass(value) or not isinstance(value, ObjectiveTerm):
        if not callable(value):
            raise TypeError("objective term registrations must implement ObjectiveTerm or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, ObjectiveTerm):
        raise TypeError("objective term registrations must implement ObjectiveTerm")
    return candidate


def _constraint_term_from_decorator(value: object) -> ConstraintTerm[Any]:
    candidate = value
    if isclass(value) or not isinstance(value, ConstraintTerm):
        if not callable(value):
            raise TypeError("constraint term registrations must implement ConstraintTerm or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, ConstraintTerm):
        raise TypeError("constraint term registrations must implement ConstraintTerm")
    return candidate


objective_terms: Registry[ObjectiveKind, ObjectiveTerm[Any]] = Registry(
    "objective term",
    ObjectiveKind,
    decorator_transform=_objective_term_from_decorator,
)
"""Registry of built-in and user-registered objective terms keyed by name."""

constraint_terms: Registry[ConstraintKind, ConstraintTerm[Any]] = Registry(
    "constraint term",
    ConstraintKind,
    decorator_transform=_constraint_term_from_decorator,
)
"""Registry of built-in and user-registered constraint terms keyed by name."""

solver_factories: Registry[SolverKind, SolverFactory] = Registry("solver factory", SolverKind)
"""Registry of solver factories keyed by :class:`~retarget.core.enums.SolverBackend` name."""

__all__ = ["SolverFactory", "constraint_terms", "objective_terms", "solver_factories"]
