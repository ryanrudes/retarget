"""Validation helpers for optimization extension references."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from retarget.core.registry import Registry
from retarget.optimization.registry import constraint_terms, objective_terms, solver_factories
from retarget.optimization.solvers import resolve_solver_backend_name
from retarget.optimization.spec import ConstraintConfig, ObjectiveConfig, SolverSpec


def validate_optimization_references(
    *,
    solver: SolverSpec,
    objectives: tuple[ObjectiveConfig, ...],
    constraints: tuple[ConstraintConfig, ...],
) -> None:
    """Validate registered objective, constraint, and solver references."""

    messages: list[str] = []
    _append_missing(
        messages,
        objective_terms,
        (objective.kind for objective in objectives if objective.weight > 0),
    )
    _append_missing(
        messages,
        constraint_terms,
        (constraint.kind for constraint in constraints if constraint.enabled),
    )
    _append_missing(messages, solver_factories, (resolve_solver_backend_name(solver),))
    if messages:
        raise KeyError("Unknown optimization registry references: " + "; ".join(messages))


def _append_missing(messages: list[str], registry: Registry[Any, Any], keys: Iterable[StrEnum]) -> None:
    missing = registry.missing(keys)
    if not missing:
        return
    available = ", ".join(registry.names()) or "<none>"
    messages.append(f"{registry.name}: {', '.join(missing)} (available: {available})")
