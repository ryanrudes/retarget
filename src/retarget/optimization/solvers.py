"""Optimization solver implementations."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize
from scipy.optimize import LinearConstraint as ScipyLinearConstraint

from retarget.core.enums import SolverBackend
from retarget.core.protocols import Solver
from retarget.optimization.problem import LinearConstraint, QuadraticProblem, SolverResult
from retarget.optimization.registry import solver_factories
from retarget.optimization.spec import SolverSpec


class NumpyLeastSquaresSolver:
    """Deterministic bounded least-squares fallback with trust-region projection."""

    def solve(self, problem: QuadraticProblem) -> SolverResult:
        """Solve a quadratic subproblem with NumPy lstsq or SciPy SLSQP.

        Uses unconstrained least squares when no linear constraints are present;
        otherwise delegates to SLSQP with box bounds, linear constraints, and an
        optional trust-region inequality. Clips the solution to bounds and projects
        onto the trust region when configured.

        Args:
            problem: Least-squares subproblem assembled for one SQP iteration.

        Returns:
            Solution vector, residual cost, and status ``"optimal"``.
        """
        if problem.linear_constraints:
            return _solve_with_slsqp(problem)
        solution, *_ = np.linalg.lstsq(problem.matrix, problem.target, rcond=None)
        solution = cast(NDArray[np.float64], np.asarray(solution, dtype=np.float64))
        if problem.initial is not None and problem.trust_radius is not None:
            delta = solution - problem.initial
            norm = float(np.linalg.norm(delta))
            if norm > problem.trust_radius:
                solution = problem.initial + delta * (problem.trust_radius / norm)
        if problem.lower is not None or problem.upper is not None:
            lower = problem.lower if problem.lower is not None else -np.inf
            upper = problem.upper if problem.upper is not None else np.inf
            solution = cast(NDArray[np.float64], np.clip(solution, lower, upper))
        return SolverResult.from_solution(problem, solution, status="optimal")


class CvxpyClarabelSolver:
    """CVXPY/Clarabel solver loaded only when the optimize extra is installed."""

    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = verbose

    def solve(self, problem: QuadraticProblem) -> SolverResult:
        """Solve a quadratic subproblem with CVXPY and the Clarabel conic solver.

        Requires the ``retarget[optimize]`` extra. Builds a sum-of-squares objective
        with box bounds, second-order-cone trust regions, and linear inequalities.

        Args:
            problem: Least-squares subproblem assembled for one SQP iteration.

        Returns:
            Solution vector, optimal cost, and CVXPY status string.

        Raises:
            RuntimeError: If optional dependencies are missing or the solve does not converge.
        """
        try:
            import clarabel  # noqa: F401
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install retarget[optimize] to use the CVXPY Clarabel solver") from exc

        x = cp.Variable(problem.matrix.shape[1])
        constraints = []
        if problem.lower is not None:
            constraints.append(x >= problem.lower)
        if problem.upper is not None:
            constraints.append(x <= problem.upper)
        if problem.initial is not None and problem.trust_radius is not None:
            constraints.append(cp.SOC(problem.trust_radius, x - problem.initial))
        for constraint in problem.linear_constraints:
            expression = constraint.matrix @ x
            if constraint.lower is not None:
                constraints.append(expression >= constraint.lower)
            if constraint.upper is not None:
                constraints.append(expression <= constraint.upper)
        objective = cp.Minimize(cp.sum_squares(problem.matrix @ x - problem.target))
        cvx_problem = cp.Problem(objective, constraints)
        cvx_problem.solve(solver=cp.CLARABEL, verbose=self.verbose)
        if cvx_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {cvx_problem.status}")
        return SolverResult(
            solution=np.asarray(x.value, dtype=np.float64),
            cost=float(cvx_problem.value),
            status=str(cvx_problem.status),
        )


def create_solver(spec: SolverSpec) -> Solver:
    """Create a solver from a spec."""

    return solver_factories.get(resolve_solver_backend_name(spec))(spec)


def resolve_solver_backend_name(spec: SolverSpec) -> str:
    """Return the concrete solver registry key selected for a solver spec."""

    backend_name = spec.backend_name
    if backend_name != SolverBackend.AUTO.value:
        return backend_name
    return _auto_solver_backend_name()


def _auto_solver_backend_name() -> str:
    if _module_available("cvxpy") and _module_available("clarabel"):
        return SolverBackend.CVXPY_CLARABEL.value
    return SolverBackend.NUMPY_LEAST_SQUARES.value


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _solve_with_slsqp(problem: QuadraticProblem) -> SolverResult:
    n_variables = problem.matrix.shape[1]
    x0 = np.zeros(n_variables, dtype=np.float64) if problem.initial is None else problem.initial.copy()
    lower = problem.lower if problem.lower is not None else np.full(n_variables, -np.inf)
    upper = problem.upper if problem.upper is not None else np.full(n_variables, np.inf)
    x0 = np.asarray(np.clip(x0, lower, upper), dtype=np.float64)

    matrix = problem.matrix
    target = problem.target

    def objective(x: NDArray[np.float64]) -> float:
        return problem.objective_value(x)

    def jacobian(x: NDArray[np.float64]) -> NDArray[np.float64]:
        residual = matrix @ x - target
        return 2.0 * matrix.T @ residual

    constraints: list[ScipyLinearConstraint | NonlinearConstraint] = [
        _to_scipy_constraint(constraint) for constraint in problem.linear_constraints
    ]
    trust_constraint: NonlinearConstraint | None = None
    if problem.initial is not None and problem.trust_radius is not None:
        origin = problem.initial.copy()
        radius_squared = float(problem.trust_radius**2)

        def trust_fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
            delta = x - origin
            return np.asarray([float(delta @ delta)], dtype=np.float64)

        def trust_jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * (x - origin)[None, :]

        trust_constraint = NonlinearConstraint(trust_fun, -np.inf, radius_squared, jac=trust_jac)
        constraints.append(trust_constraint)

    result = _minimize_slsqp(objective, jacobian, x0, lower, upper, constraints)
    if not result.success and trust_constraint is not None:
        constraints_without_trust = [constraint for constraint in constraints if constraint is not trust_constraint]
        result = _minimize_slsqp(objective, jacobian, x0, lower, upper, constraints_without_trust)
    if not result.success:
        raise RuntimeError(f"SLSQP solve failed: {result.message}")
    return SolverResult.from_solution(problem, np.asarray(result.x, dtype=np.float64), status="optimal")


def _minimize_slsqp(
    objective: Callable[[NDArray[np.float64]], float],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    constraints: list[ScipyLinearConstraint | NonlinearConstraint],
) -> OptimizeResult:
    return minimize(
        objective,
        x0,
        jac=jacobian,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-9, "disp": False},
    )


def _to_scipy_constraint(constraint: LinearConstraint) -> ScipyLinearConstraint:
    rows = constraint.matrix.shape[0]
    lower = constraint.lower if constraint.lower is not None else np.full(rows, -np.inf)
    upper = constraint.upper if constraint.upper is not None else np.full(rows, np.inf)
    return ScipyLinearConstraint(constraint.matrix, lower, upper)


solver_factories.register(SolverBackend.NUMPY_LEAST_SQUARES.value, lambda _spec: NumpyLeastSquaresSolver())
solver_factories.register(SolverBackend.CVXPY_CLARABEL.value, lambda spec: CvxpyClarabelSolver(verbose=spec.verbose))
