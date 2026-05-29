import numpy as np

import retarget.optimization.solvers as solver_module
from retarget.optimization import (
    CvxpyClarabelSolver,
    LinearConstraint,
    NumpyLeastSquaresSolver,
    QuadraticProblem,
    SolverSpec,
    create_solver,
    resolve_solver_backend_name,
)


def test_numpy_solver_respects_bounds_and_trust_region():
    problem = QuadraticProblem(
        matrix=np.eye(2),
        target=np.array([10.0, -10.0]),
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        initial=np.array([0.0, 0.0]),
        trust_radius=0.5,
    )
    result = NumpyLeastSquaresSolver().solve(problem)
    assert np.linalg.norm(result.solution) <= 0.5 + 1e-9
    assert np.all(result.solution <= 1.0)
    assert np.all(result.solution >= -1.0)


def test_numpy_solver_respects_linear_constraints():
    problem = QuadraticProblem(
        matrix=np.eye(1),
        target=np.array([2.0]),
        linear_constraints=(
            LinearConstraint(
                matrix=np.array([[1.0]]),
                lower=None,
                upper=np.array([0.5]),
            ),
        ),
    )
    result = NumpyLeastSquaresSolver().solve(problem)
    assert result.solution[0] <= 0.5 + 1e-7


def test_auto_solver_prefers_cvxpy_clarabel_when_optimize_stack_is_available(monkeypatch):
    monkeypatch.setattr(solver_module, "_module_available", lambda name: name in {"cvxpy", "clarabel"})

    solver = create_solver(SolverSpec())

    assert isinstance(solver, CvxpyClarabelSolver)
    assert resolve_solver_backend_name(SolverSpec()) == "cvxpy_clarabel"


def test_auto_solver_falls_back_to_numpy_when_clarabel_is_missing(monkeypatch):
    monkeypatch.setattr(solver_module, "_module_available", lambda name: name == "cvxpy")

    solver = create_solver(SolverSpec())

    assert isinstance(solver, NumpyLeastSquaresSolver)
    assert resolve_solver_backend_name(SolverSpec()) == "numpy_least_squares"
