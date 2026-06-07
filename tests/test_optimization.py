import numpy as np
import pytest

from retarget.core.enums import (
    Constraint,
    NominalFallback,
    NonPenetrationSource,
    Objective,
)
from retarget.optimization import (
    FootLockConstraintConfig,
    FootLockWindow,
    GeometryPair,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    QposVariableSpec,
    SelfCollisionConstraintConfig,
    SmoothnessObjectiveConfig,
)
from tests.typed_fixtures import (
    FixtureContactSubject,
    FixtureRobotGeometry,
    FixtureRobotJoint,
    FixtureRobotLink,
    SameValueRobotJoint,
    fixture_problem,
    fixture_robot,
)


def test_optimization_configs_use_class_level_enum_kinds():
    smoothness = SmoothnessObjectiveConfig(weight=0.2)
    profile = OptimizationProfile.defaults().with_objective(smoothness)

    assert smoothness.kind is Objective.SMOOTHNESS
    assert profile.objective_kinds[-1] is Objective.SMOOTHNESS
    assert Constraint.JOINT_LIMITS in profile.constraint_kinds
    assert "kind" not in smoothness.model_dump()


def test_nominal_joint_selection_is_typed_and_wrong_vocab_is_rejected_by_problem():
    config = NominalTrackingObjectiveConfig(
        joints=(FixtureRobotJoint.ROOT,),
        fallback=NominalFallback.CURRENT,
    )
    problem = fixture_problem().model_copy(update={"objectives": (config,)})
    problem._validate_problem()

    wrong = NominalTrackingObjectiveConfig(joints=(SameValueRobotJoint.ROOT,))
    broken = fixture_problem().model_copy(update={"objectives": (wrong,)})
    with pytest.raises(TypeError):
        broken._validate_typed_optimization_selections()


def test_non_penetration_requires_explicit_typed_selection():
    with pytest.raises(ValueError, match="explicit links or subjects"):
        NonPenetrationConstraintConfig(
            sources=(NonPenetrationSource.SCENE_POINTS,),
        )

    config = NonPenetrationConstraintConfig(
        links=(FixtureRobotLink.LEFT_FOOT,),
        sources=(NonPenetrationSource.SCENE_POINTS,),
    )
    assert config.links == (FixtureRobotLink.LEFT_FOOT,)


def test_geometry_constraints_require_explicit_pairs():
    with pytest.raises(ValueError, match="geometry_pairs"):
        NonPenetrationConstraintConfig(
            sources=(NonPenetrationSource.GEOMETRY,),
        )
    with pytest.raises(ValueError, match="explicit typed geometry pairs"):
        SelfCollisionConstraintConfig()

    pair = GeometryPair(
        first=FixtureRobotGeometry.LEFT_FOOT,
        second=FixtureRobotGeometry.RIGHT_FOOT,
    )
    assert SelfCollisionConstraintConfig(pairs=(pair,)).pairs == (pair,)


def test_foot_lock_window_uses_exact_subject_or_link():
    by_subject = FootLockWindow(
        subject=FixtureContactSubject.LEFT_FOOT,
        ranges=((0, 2),),
    )
    by_link = FootLockWindow(
        link=FixtureRobotLink.RIGHT_FOOT,
        ranges=((1, 3),),
    )

    config = FootLockConstraintConfig(windows=(by_subject, by_link))
    assert config.windows[0].subject is FixtureContactSubject.LEFT_FOOT
    assert config.windows[1].link is FixtureRobotLink.RIGHT_FOOT


def test_qpos_variable_resolution_uses_typed_robot_joint_order():
    robot = fixture_robot()
    resolved = QposVariableSpec.actuated().resolve(
        robot,
        qpos_size=robot.qpos_size(),
        joint_limits=robot.joint_limits,
    )

    assert resolved.indices.tolist() == [7, 8, 9]
    assert np.allclose(resolved.lower, -1.0)
    assert np.allclose(resolved.upper, 1.0)


def test_floating_base_variable_policy_normalizes_quaternion():
    robot = fixture_robot()
    resolved = QposVariableSpec.holosoma_q_a().resolve(
        robot,
        qpos_size=robot.qpos_size(),
        joint_limits=robot.joint_limits,
    )
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    delta = np.zeros(resolved.size, dtype=np.float64)
    delta[3:7] = (1.0, 1.0, 0.0, 0.0)

    updated = resolved.apply_delta(qpos, delta)

    assert np.linalg.norm(updated[3:7]) == pytest.approx(1.0)
