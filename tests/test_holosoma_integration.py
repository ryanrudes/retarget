import numpy as np
import pytest

from retarget import RetargetingProblem, SceneSpec, TaskKind
from retarget.integrations.holosoma import (
    G1_LEFT_FOOT_STICKING_LINKS,
    G1_NOMINAL_TRACKING_QPOS_INDICES,
    G1_RIGHT_FOOT_STICKING_LINKS,
    default_holosoma_root,
    from_mocap_climb_fixture,
    holosoma_initial_qpos_plan,
)
from retarget.motion import InitialQposPlan, MotionSequence
from retarget.optimization import NonPenetrationConstraintConfig
from retarget.robots import robots
from retarget.scene.spec import ObjectSpec, ObjectTrajectory

HOLOSOMA_FIXTURE = default_holosoma_root() / "tests" / "fixtures" / "climb_seq_0"


def test_external_object_trajectory_does_not_append_qpos() -> None:
    robot = robots.get("synthetic_humanoid")
    motion = MotionSequence(
        name="external_object",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
    )
    scene = SceneSpec.climbing(
        object_spec=ObjectSpec(
            name="box",
            trajectory=ObjectTrajectory.identity(1),
            qpos_mode="external",
        )
    )
    initial_qpos = InitialQposPlan.from_array(np.zeros((1, robot.qpos_size()), dtype=np.float64))

    problem = RetargetingProblem(
        name="external_object",
        task_kind=TaskKind.CLIMBING,
        robot=robot,
        motion=motion,
        scene=scene,
        initial_qpos=initial_qpos,
    )

    assert problem.scene.has_dynamic_object() is False
    assert problem.robot.qpos_size(has_object=problem.scene.has_dynamic_object()) == robot.qpos_size()


def test_holosoma_initial_qpos_plan_matches_locked_seed_overlay() -> None:
    q_init = np.arange(36, dtype=np.float64)
    object_poses = np.zeros((2, 7), dtype=np.float64)
    object_poses[:, 3] = 1.0

    plan = holosoma_initial_qpos_plan(frame_count=2, q_init=q_init, object_poses_mujoco=object_poses)

    assert plan.qpos.shape == (2, 36)
    assert np.allclose(plan.qpos[0, :29], q_init[:29])
    assert np.allclose(plan.qpos[:, -7:], object_poses)
    assert np.allclose(plan.qpos[1, :29], 0.0)


@pytest.mark.skipif(not HOLOSOMA_FIXTURE.exists(), reason="Holosoma climb fixture is not available")
def test_holosoma_mocap_climb_fixture_builds_typed_problem_contract() -> None:
    prep = from_mocap_climb_fixture(frame_count=3, include_object_collision=False)
    resolved_variables = prep.problem.variables.resolve(
        prep.robot,
        qpos_size=prep.robot.qpos_size(has_object=prep.scene.has_dynamic_object()),
    )
    nominal = prep.problem.objectives[1]
    diagonal = prep.problem.objectives[2]

    assert prep.motion.joint_positions.shape == (3, 53, 3)
    assert prep.scene.object is not None
    assert prep.scene.object.qpos_mode == "external"
    assert prep.scene.has_dynamic_object() is False
    assert prep.problem.initial_qpos is prep.initial_qpos
    assert prep.initial_qpos.qpos.shape == (3, 36)
    assert prep.problem.solver.max_iterations == 10
    assert prep.problem.solver.first_frame_iterations == 50
    assert prep.problem.solver.convergence == "cost_plateau"
    assert tuple(resolved_variables.indices) == tuple(range(36))
    assert nominal.qpos_indices == G1_NOMINAL_TRACKING_QPOS_INDICES
    assert diagonal.qpos_weights[19] == pytest.approx(0.2)
    assert diagonal.qpos_weights[20] == pytest.approx(0.2)
    assert not any(isinstance(constraint, NonPenetrationConstraintConfig) for constraint in prep.problem.constraints)


@pytest.mark.skipif(not HOLOSOMA_FIXTURE.exists(), reason="Holosoma climb fixture is not available")
def test_holosoma_mocap_climb_contacts_and_geometry_pairs_are_typed() -> None:
    prep = from_mocap_climb_fixture(frame_count=2, include_object_collision=True)

    assert prep.contacts.tracks[0].subject == "LeftToeBase"
    assert prep.contacts.tracks[0].link_names == G1_LEFT_FOOT_STICKING_LINKS
    assert prep.contacts.tracks[1].subject == "RightToeBase"
    assert prep.contacts.tracks[1].link_names == G1_RIGHT_FOOT_STICKING_LINKS
    assert prep.geometry_pairs
    assert ("pelvis", "multi_boxes_link_1") in prep.geometry_pairs
    assert ("pelvis", "ground") in prep.geometry_pairs
    assert all("mocap" not in first for first, _second in prep.geometry_pairs)
    non_penetration = prep.problem.constraints[-1]
    assert isinstance(non_penetration, NonPenetrationConstraintConfig)
    assert non_penetration.sources == ("geometry",)
    assert non_penetration.geometry_source == "backend_candidates"
    assert non_penetration.scene_geometry_keywords == ("multi_boxes", "ground")
    assert non_penetration.excluded_geometry_keyword_pairs == (("multi_boxes", "ground"),)
