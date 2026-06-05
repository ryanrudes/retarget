from dataclasses import replace

import numpy as np

import retarget.optimization.solvers as solver_module
from retarget import RetargetingProblem, SceneSpec, TaskKind
from retarget.kinematics.backends import SimpleKinematicsBackend
from retarget.motion import ContactPlan, ContactTrack, LinkTargetPlan, MotionSequence, SupportPlane
from retarget.optimization import (
    CvxpyClarabelSolver,
    DiagonalRegularizationObjectiveConfig,
    FootLockConstraintConfig,
    FootStickingConstraintConfig,
    LinearConstraint,
    LinkTrackingObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    NumpyLeastSquaresSolver,
    OptimizationProfile,
    QuadraticProblem,
    SmoothnessObjectiveConfig,
    SolverSpec,
    TermContext,
    create_solver,
    resolve_solver_backend_name,
)
from retarget.optimization.terms import (
    DiagonalRegularizationObjective,
    LinkTrackingObjective,
    NominalTrackingObjective,
    NonPenetrationConstraint,
    foot_lock_constraints,
    geometry_non_penetration_constraints,
    ground_non_penetration_constraints,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.robots import robots


def _basic_context(
    problem: RetargetingProblem,
    backend: SimpleKinematicsBackend,
    qpos: np.ndarray,
) -> TermContext:
    return TermContext(
        problem=problem,
        backend=backend,
        q_current=qpos,
        q_previous=qpos,
        frame_idx=0,
        contact_frame=None,
        target_frame=None,
        robot_point_names=(),
        robot_points=np.zeros((0, 3), dtype=np.float64),
        robot_jacobians=np.zeros((0, 3, problem.robot.dof), dtype=np.float64),
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=(),
        target_laplacian=np.zeros((0, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(problem.robot.dof, -1.0, dtype=np.float64),
        joint_upper=np.full(problem.robot.dof, 1.0, dtype=np.float64),
        current_joints=qpos[problem.robot.qpos_layout.joint_slice(problem.robot.dof)],
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


def test_typed_objective_and_constraint_configs_have_validating_copy_helpers():
    objective = SmoothnessObjectiveConfig(weight=1.0)
    constraint = FootStickingConstraintConfig(tolerance=0.001)

    assert objective.with_weight(2.0).weight == 2.0
    assert constraint.with_enabled(False).enabled is False
    assert constraint.disabled().enabled is False
    assert constraint.tolerance == 0.001


def test_optimization_profile_builders_replace_and_validate_terms():
    profile = (
        OptimizationProfile.defaults()
        .with_objective(NominalTrackingObjectiveConfig(weight=3.0))
        .with_objective(SmoothnessObjectiveConfig(weight=0.05))
        .without_constraint("foot_sticking")
        .with_constraint(NonPenetrationConstraintConfig(scene_clearance=0.03))
    )

    assert profile.objective_names == ("laplacian", "smoothness", "nominal_tracking")
    assert profile.objective("smoothness") is not None
    assert profile.objective("smoothness").weight == 0.05
    assert profile.constraint_names == ("joint_limits", "trust_region", "non_penetration")
    non_penetration = profile.constraint("non_penetration")
    assert isinstance(non_penetration, NonPenetrationConstraintConfig)
    assert non_penetration.scene_clearance == 0.03
    profile.validate_registry_references(SolverSpec(backend="numpy_least_squares"))


def test_task_profile_presets_include_scene_constraints():
    object_profile = OptimizationProfile.object_interaction(scene_clearance=0.04, links=("left_hand",))
    climbing_profile = OptimizationProfile.climbing(scene_clearance=0.02)

    assert object_profile.name == "object_interaction"
    assert object_profile.constraint("non_penetration") is not None
    object_constraint = object_profile.constraint("non_penetration")
    assert isinstance(object_constraint, NonPenetrationConstraintConfig)
    assert object_constraint.links == ("left_hand",)
    assert climbing_profile.name == "climbing"
    climbing_constraint = climbing_profile.constraint("non_penetration")
    assert isinstance(climbing_constraint, NonPenetrationConstraintConfig)
    assert climbing_constraint.scene_clearance == 0.02


def test_holosoma_q_a_variable_policy_includes_root_and_normalizes_quaternion():
    robot = robots.get("synthetic_humanoid")
    variables = QposVariableSpec.holosoma_q_a().resolve(robot, qpos_size=robot.qpos_size())
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    delta = np.zeros(variables.size, dtype=np.float64)
    delta[4] = 0.25

    updated = variables.apply_delta(qpos, delta)

    assert variables.indices[0] == 0
    assert variables.indices[-1] == robot.qpos_layout.joint_start + robot.dof - 1
    assert variables.size == robot.qpos_layout.joint_start + robot.dof
    assert np.isclose(np.linalg.norm(updated[3:7]), 1.0)


def test_diagonal_regularization_uses_qpos_and_variable_weights():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[0] = 2.0
    qpos[robot.qpos_layout.joint_start] = 0.5
    motion = MotionSequence(
        name="diag",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="diag",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        scene=SceneSpec.robot_only(),
        variables=QposVariableSpec.qpos_indices((0, robot.qpos_layout.joint_start)),
    )
    context = TermContext(
        problem=problem,
        backend=backend,
        q_current=qpos,
        q_previous=qpos,
        frame_idx=0,
        contact_frame=None,
        target_frame=None,
        robot_point_names=(),
        robot_points=np.zeros((0, 3), dtype=np.float64),
        robot_jacobians=np.zeros((0, 3, 2), dtype=np.float64),
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=(),
        target_laplacian=np.zeros((0, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(robot.dof, -1.0, dtype=np.float64),
        joint_upper=np.full(robot.dof, 1.0, dtype=np.float64),
        current_joints=qpos[robot.qpos_layout.joint_slice(robot.dof)],
        variable_indices=np.asarray([0, robot.qpos_layout.joint_start], dtype=np.int64),
        current_variables=np.asarray([2.0, 0.5], dtype=np.float64),
        variable_lower=np.asarray([-10.0, -1.0], dtype=np.float64),
        variable_upper=np.asarray([10.0, 1.0], dtype=np.float64),
    )

    contribution = DiagonalRegularizationObjective().build(
        context,
        DiagonalRegularizationObjectiveConfig(variable_weights=(4.0, 9.0)),
    )[0]

    assert np.allclose(contribution.matrix, [[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(contribution.target, [-4.0, -1.5])


def test_nominal_tracking_current_fallback_penalizes_step_not_absolute_pose():
    robot = robots.get("synthetic_humanoid")
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[robot.qpos_layout.joint_start] = 0.75
    motion = MotionSequence(
        name="nominal",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="nominal",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        scene=SceneSpec.robot_only(),
    )
    context = TermContext(
        problem=problem,
        backend=SimpleKinematicsBackend(robot),
        q_current=qpos,
        q_previous=qpos,
        frame_idx=0,
        contact_frame=None,
        target_frame=None,
        robot_point_names=(),
        robot_points=np.zeros((0, 3), dtype=np.float64),
        robot_jacobians=np.zeros((0, 3, robot.dof), dtype=np.float64),
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=(),
        target_laplacian=np.zeros((0, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(robot.dof, -1.0, dtype=np.float64),
        joint_upper=np.full(robot.dof, 1.0, dtype=np.float64),
        current_joints=qpos[robot.qpos_layout.joint_slice(robot.dof)],
    )

    zero_target = NominalTrackingObjective().build(
        context,
        NominalTrackingObjectiveConfig(qpos_indices=(robot.qpos_layout.joint_start,), fallback="zero"),
    )[0]
    current_target = NominalTrackingObjective().build(
        context,
        NominalTrackingObjectiveConfig(qpos_indices=(robot.qpos_layout.joint_start,), fallback="current"),
    )[0]

    assert np.allclose(zero_target.target, [-0.75])
    assert np.allclose(current_target.target, [0.0])


def test_geometry_non_penetration_uses_penetration_tolerance_not_scene_clearance():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    motion = MotionSequence(
        name="geom",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="geom",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        scene=SceneSpec.robot_only(),
    )
    context = _basic_context(problem, backend, qpos)
    distance = backend.geom_distances(qpos, (("left_toe", "right_toe"),), max_distance=10.0)[0].distance

    constraints = geometry_non_penetration_constraints(
        context=context,
        config=NonPenetrationConstraintConfig(
            geometry_pairs=(("left_toe", "right_toe"),),
            tolerance=0.001,
            scene_clearance=0.05,
            activation_distance=10.0,
        ),
    )

    assert len(constraints) == 1
    assert constraints[0].lower is not None
    assert np.isclose(constraints[0].lower[0], -distance - 0.001)


def test_link_tracking_objective_builds_active_weighted_rows():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    current, _jacobians = backend.point_jacobians(qpos, ("left_toe",))
    target = current.copy()
    target[0, 2] += 0.1
    motion = MotionSequence(
        name="targets",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    targets = LinkTargetPlan.from_arrays(
        link_names=("left_toe", "right_toe"),
        positions=np.asarray([[target[0], [0.0, 0.0, 0.0]]], dtype=np.float64),
        weights=np.asarray([[4.0, 1.0]], dtype=np.float64),
        active_mask=np.asarray([[True, False]], dtype=bool),
    )
    problem = RetargetingProblem(
        name="link_tracking_test",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        targets=targets,
        scene=SceneSpec.robot_only(),
    )
    context = TermContext(
        problem=problem,
        backend=backend,
        q_current=qpos,
        q_previous=qpos,
        frame_idx=0,
        contact_frame=None,
        target_frame=targets.frame(0),
        robot_point_names=("left_toe",),
        robot_points=current,
        robot_jacobians=np.zeros((1, 3, robot.dof), dtype=np.float64),
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=((),),
        target_laplacian=np.zeros((1, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(robot.dof, -1.0, dtype=np.float64),
        joint_upper=np.full(robot.dof, 1.0, dtype=np.float64),
        current_joints=np.zeros(robot.dof, dtype=np.float64),
    )

    contributions = LinkTrackingObjective().build(context, LinkTrackingObjectiveConfig())

    assert len(contributions) == 1
    contribution = contributions[0]
    assert contribution.matrix.shape == (3, robot.dof)
    assert np.allclose(contribution.target, [0.0, 0.0, 0.2])


def test_contact_constraints_use_typed_support_plane_normal():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    positions, jacobians = backend.point_jacobians(qpos, ("left_toe",))
    support = SupportPlane(normal=np.array([0.0, 1.0, 1.0]), origin=np.zeros(3))
    contacts = ContactPlan(
        tracks=(ContactTrack("left_foot", np.array([1]), link_names=("left_toe",)),),
        support=support,
    )
    motion = MotionSequence(
        name="contact",
        joint_names=("left_foot",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="contact_terms",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        contacts=contacts,
        scene=SceneSpec.robot_only(),
    )
    context = TermContext(
        problem=problem,
        backend=backend,
        q_current=qpos,
        q_previous=qpos,
        frame_idx=0,
        contact_frame=contacts.frame(0),
        target_frame=None,
        robot_point_names=("left_toe",),
        robot_points=positions,
        robot_jacobians=jacobians,
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=((),),
        target_laplacian=np.zeros((1, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(robot.dof, -1.0, dtype=np.float64),
        joint_upper=np.full(robot.dof, 1.0, dtype=np.float64),
        current_joints=np.zeros(robot.dof, dtype=np.float64),
    )

    non_penetration = ground_non_penetration_constraints(
        context=context,
        config=NonPenetrationConstraintConfig(links=("left_toe",), tolerance=0.01),
    )
    foot_lock = foot_lock_constraints(
        context=context,
        config=FootLockConstraintConfig(tolerance=0.02),
    )

    expected_row = (support.normal @ jacobians[0]).reshape(1, -1)
    assert len(non_penetration) == 1
    assert np.allclose(non_penetration[0].matrix, expected_row)
    assert len(foot_lock) == 1
    assert np.allclose(foot_lock[0].matrix, expected_row)
    assert foot_lock[0].lower is not None
    assert foot_lock[0].upper is not None


def test_non_penetration_sources_gate_constraint_builders():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    contacts = ContactPlan(
        tracks=(ContactTrack("left_foot", np.array([1]), link_names=("left_toe",)),),
        support=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3)),
    )
    motion = MotionSequence(
        name="contact",
        joint_names=("left_foot",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="nonpen_sources",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        contacts=contacts,
        scene=SceneSpec.robot_only(),
    )
    context = replace(_basic_context(problem, backend, qpos), contact_frame=contacts.frame(0))

    config = NonPenetrationConstraintConfig(links=("left_toe",), sources=("geometry", "geometry"))
    contribution = NonPenetrationConstraint().build(context, config)

    assert config.sources == ("geometry",)
    assert contribution.linear_constraints == ()
