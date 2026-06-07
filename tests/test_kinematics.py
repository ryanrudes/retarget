from dataclasses import replace

import numpy as np
import pytest

from retarget.kinematics import SimpleKinematicsBackend
from retarget.pipeline.compiled import compile_problem, compile_robot
from tests.typed_fixtures import (
    FixtureRobotJoint,
    FixtureRobotLink,
    fixture_problem,
    fixture_robot,
)


def test_compiled_robot_has_exact_names_indices_and_limits():
    robot = fixture_robot()
    compiled = compile_robot(robot)

    assert compiled.joint_names == tuple(joint.value for joint in FixtureRobotJoint)
    assert compiled.link_names == tuple(link.value for link in FixtureRobotLink)
    assert compiled.joint_index(FixtureRobotJoint.LEFT_LEG.value) == 1
    assert compiled.joint_limits[FixtureRobotJoint.LEFT_LEG.value] == (-1.0, 1.0)


def test_simple_backend_uses_explicit_typed_fixture_points():
    compiled = compile_problem(fixture_problem())
    backend = SimpleKinematicsBackend(compiled.robot)
    qpos = np.zeros(compiled.robot.qpos_size(), dtype=np.float64)
    qpos[8] = 0.5

    positions, jacobians = backend.point_jacobians(
        qpos,
        (FixtureRobotLink.LEFT_FOOT.value,),
    )

    assert np.allclose(positions[0], [-0.15, 0.1, -0.3])
    assert jacobians.shape == (1, 3, compiled.robot.dof)
    assert np.allclose(jacobians[0, :, 1], [0.0, 0.0, 1.0])


def test_simple_backend_qpos_index_jacobians_include_root_translation():
    compiled = compile_problem(fixture_problem())
    backend = SimpleKinematicsBackend(compiled.robot)
    qpos = np.zeros(compiled.robot.qpos_size(), dtype=np.float64)
    indices = np.asarray([0, 1, 2, 8], dtype=np.int64)

    _, jacobians = backend.point_jacobians_for_qpos_indices(
        qpos,
        (FixtureRobotLink.LEFT_FOOT.value,),
        indices,
    )

    assert np.allclose(jacobians[0, :, :3], np.eye(3))
    assert np.allclose(jacobians[0, :, 3], [0.0, 0.0, 1.0])


def test_simple_backend_velocity_round_trip_and_quaternion_normalization():
    compiled = compile_problem(fixture_problem())
    backend = SimpleKinematicsBackend(compiled.robot)
    qpos = np.zeros(compiled.robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    next_qpos = qpos.copy()
    next_qpos[8] = 0.2

    qvel = backend.qpos_to_qvel(next_qpos, qpos, 0.1)
    integrated = backend.integrate_qvel(qpos, qvel, 0.1)

    assert np.allclose(integrated, next_qpos)
    assert np.linalg.norm(integrated[3:7]) == pytest.approx(1.0)


def test_simple_backend_validation_fails_before_unknown_link_lookup():
    compiled = compile_problem(fixture_problem())
    broken_robot = replace(
        compiled.robot,
        simple_kinematics={
            name: point
            for name, point in compiled.robot.simple_kinematics.items()
            if name != FixtureRobotLink.LEFT_FOOT.value
        },
    )
    backend = SimpleKinematicsBackend(broken_robot)

    with pytest.raises(KeyError, match="left_foot"):
        backend.validate_compiled_problem(compiled)
