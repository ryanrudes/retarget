"""Optimization models and solvers."""

from retarget.optimization.problem import (
    ConstraintContribution,
    LinearConstraint,
    ObjectiveContribution,
    QuadraticProblem,
    SolverResult,
    TermContext,
)
from retarget.optimization.registry import SolverFactory, constraint_terms, objective_terms, solver_factories
from retarget.optimization.solvers import (
    CvxpyClarabelSolver,
    NumpyLeastSquaresSolver,
    create_solver,
    resolve_solver_backend_name,
)
from retarget.optimization.spec import ConstraintSpec, ObjectiveSpec, OptimizationProfile, SolverSpec
from retarget.optimization.terms import (
    FootContactConstraint,
    FootLockConstraint,
    JointLimitConstraint,
    LaplacianObjective,
    LinkTrackingObjective,
    NominalTrackingObjective,
    NonPenetrationConstraint,
    SelfCollisionConstraint,
    SmoothnessObjective,
    TrustRegionConstraint,
)
from retarget.optimization.validation import validate_optimization_references

__all__ = [
    "ConstraintContribution",
    "ConstraintSpec",
    "CvxpyClarabelSolver",
    "FootContactConstraint",
    "FootLockConstraint",
    "JointLimitConstraint",
    "LaplacianObjective",
    "LinearConstraint",
    "LinkTrackingObjective",
    "NominalTrackingObjective",
    "NonPenetrationConstraint",
    "NumpyLeastSquaresSolver",
    "ObjectiveContribution",
    "ObjectiveSpec",
    "OptimizationProfile",
    "QuadraticProblem",
    "SelfCollisionConstraint",
    "SmoothnessObjective",
    "SolverFactory",
    "SolverResult",
    "SolverSpec",
    "TermContext",
    "TrustRegionConstraint",
    "constraint_terms",
    "create_solver",
    "objective_terms",
    "resolve_solver_backend_name",
    "solver_factories",
    "validate_optimization_references",
]
