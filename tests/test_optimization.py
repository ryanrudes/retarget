import numpy as np

import retarget.optimization.solvers as solver_module
from retarget.optimization import (
    ConstraintSpec,
    CvxpyClarabelSolver,
    LinearConstraint,
    NumpyLeastSquaresSolver,
    ObjectiveSpec,
    OptimizationProfile,
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


def test_objective_and_constraint_specs_have_validating_copy_helpers():
    objective = ObjectiveSpec(name="laplacian", weight=1.0, parameters={"alpha": 1})
    constraint = ConstraintSpec(name="trust_region", parameters={"radius": 0.1})

    assert objective.with_weight(2.0).weight == 2.0
    assert objective.with_parameters(beta=2).parameters == {"alpha": 1, "beta": 2}
    assert constraint.with_parameters(radius=0.2).parameters == {"radius": 0.2}
    assert constraint.with_enabled(False).enabled is False
    assert constraint.disabled().enabled is False


def test_optimization_profile_builders_replace_and_validate_terms():
    profile = (
        OptimizationProfile.defaults()
        .with_objective("nominal_tracking", weight=3.0)
        .with_objective("smoothness", weight=0.05)
        .without_constraint("foot_contact")
        .with_constraint("non_penetration", parameters={"scene_clearance": 0.03})
    )

    assert profile.objective_names == ("laplacian", "smoothness", "nominal_tracking")
    assert profile.objective("smoothness") is not None
    assert profile.objective("smoothness").weight == 0.05
    assert profile.constraint_names == ("joint_limits", "trust_region", "non_penetration")
    assert profile.constraint("non_penetration").parameters["scene_clearance"] == 0.03
    profile.validate_registry_references(SolverSpec(backend="numpy_least_squares"))


def test_task_profile_presets_include_scene_constraints():
    object_profile = OptimizationProfile.object_interaction(scene_clearance=0.04, links=("left_hand",))
    climbing_profile = OptimizationProfile.climbing(scene_clearance=0.02)

    assert object_profile.name == "object_interaction"
    assert object_profile.constraint("non_penetration") is not None
    assert object_profile.constraint("non_penetration").parameters["links"] == ("left_hand",)
    assert climbing_profile.name == "climbing"
    assert climbing_profile.constraint("non_penetration").parameters["scene_clearance"] == 0.02
