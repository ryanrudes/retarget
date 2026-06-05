from types import SimpleNamespace

import numpy as np
import pytest

from retarget.core.protocols import KinematicsBackend
from retarget.kinematics.backends import MuJoCoKinematicsBackend, SimpleKinematicsBackend
from retarget.pipeline.engine import _joint_limit_arrays
from retarget.robots import robots
from retarget.robots.spec import RobotSpec


def test_simple_backend_returns_point_jacobians():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size())
    positions, jacobians = backend.point_jacobians(qpos, ("left_toe", "right_hand"))
    assert positions.shape == (2, 3)
    assert jacobians.shape == (2, 3, robot.dof)
    assert np.any(jacobians[0])


def test_simple_backend_qpos_index_jacobians_include_root_translation():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size())
    qpos[3] = 1.0
    left_ankle = robot.joint_index("left_ankle")
    indices = np.asarray([0, 1, 2, robot.qpos_layout.joint_start + left_ankle], dtype=np.int64)

    _positions, jacobians = backend.point_jacobians_for_qpos_indices(qpos, ("left_toe",), indices)

    assert jacobians.shape == (1, 3, 4)
    assert np.allclose(jacobians[0, :, 0], [1.0, 0.0, 0.0])
    assert np.allclose(jacobians[0, :, 1], [0.0, 1.0, 0.0])
    assert np.allclose(jacobians[0, :, 2], [0.0, 0.0, 1.0])
    assert np.any(jacobians[0, :, 3])


def test_simple_backend_exposes_full_kinematics_protocol():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size())
    qpos[3] = 1.0
    qpos_next = qpos.copy()
    qpos_next[0] = 0.1
    qpos_next[robot.qpos_layout.joint_slice(robot.dof).start] = 0.2

    positions, translational, rotational = backend.body_jacobians(qpos, ("left_toe", "right_hand"))
    qvel = backend.qpos_to_qvel(qpos_next, qpos, 0.5)
    integrated = backend.integrate_qvel(qpos, qvel, 0.5)
    distances = backend.geom_distances(qpos, (("left_toe", "right_toe"),), max_distance=10.0)
    candidates = backend.collision_candidates(qpos, margin=10.0, geom_pairs=(("left_toe", "right_toe"),))

    assert isinstance(backend, KinematicsBackend)
    assert positions.shape == (2, 3)
    assert translational.shape == (2, 3, robot.dof)
    assert rotational.shape == (2, 3, robot.dof)
    assert np.allclose(integrated[:3], qpos_next[:3])
    assert np.isclose(np.linalg.norm(integrated[3:7]), 1.0)
    assert distances[0].first == "left_toe"
    assert distances[0].distance >= 0.0
    assert len(candidates) == len(distances)
    assert np.allclose(candidates[0].point_on_first, distances[0].point_on_first)


def test_mujoco_qdot_transform_supports_ball_joints():
    backend = MuJoCoKinematicsBackend.__new__(MuJoCoKinematicsBackend)
    backend._mujoco = _FakeMujoco
    backend.model = SimpleNamespace(
        nq=13,
        nv=11,
        njnt=4,
        jnt_type=np.array(
            [
                _FakeMujoco.mjtJoint.mjJNT_FREE,
                _FakeMujoco.mjtJoint.mjJNT_BALL,
                _FakeMujoco.mjtJoint.mjJNT_HINGE,
                _FakeMujoco.mjtJoint.mjJNT_SLIDE,
            ]
        ),
        jnt_qposadr=np.array([0, 7, 11, 12]),
        jnt_dofadr=np.array([0, 6, 9, 10]),
    )
    qpos = np.zeros(13, dtype=np.float64)
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qpos[7:11] = np.array([0.5, 0.5, 0.5, 0.5])
    backend.data = SimpleNamespace(qpos=qpos)

    transform = backend._qdot_to_qvel_transform()

    assert transform.shape == (11, 13)
    assert np.allclose(transform[0:3, 0:3], np.eye(3))
    assert np.allclose(
        transform[3:6, 3:7],
        MuJoCoKinematicsBackend._quat_qdot_to_angular_velocity_matrix(qpos[3:7]),
    )
    assert np.allclose(
        transform[6:9, 7:11],
        MuJoCoKinematicsBackend._quat_qdot_to_angular_velocity_matrix(qpos[7:11]),
    )
    assert transform[9, 11] == 1.0
    assert transform[10, 12] == 1.0


def test_mujoco_backend_collision_candidate_jacobians_filter_scene_keywords(tmp_path):
    pytest.importorskip("mujoco")
    xml_path = tmp_path / "collision.xml"
    xml_path.write_text(
        """
<mujoco model="collision_fixture">
  <worldbody>
    <geom name="ground" type="plane" size="1 1 0.1" pos="0 0 0" contype="1" conaffinity="1"/>
    <body name="robot" pos="0 0 0">
      <freejoint/>
      <geom name="robot_geom" type="sphere" size="0.05" contype="1" conaffinity="1"/>
      <body name="hinge_link" pos="0 0 0">
        <joint name="joint" type="hinge" axis="0 0 1" range="-1 1" limited="true"/>
        <geom name="hinge_geom" type="sphere" size="0.01" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="object" pos="0 0 0.13">
      <geom name="multi_boxes_link_1" type="sphere" size="0.05" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    robot = RobotSpec(
        name="collision_fixture",
        dof=1,
        height_m=1.0,
        joint_names=("joint",),
        link_names=("robot",),
    )
    backend = MuJoCoKinematicsBackend(robot, xml_path=xml_path)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[2] = 0.05
    qpos[3] = 1.0
    original_margins = backend.model.geom_margin.copy()

    candidates = backend.collision_candidate_jacobians(
        qpos,
        np.asarray([0, 1, 2, 7], dtype=np.int64),
        max_distance=0.1,
        scene_geometry_keywords=("multi_boxes", "ground"),
        excluded_geometry_keyword_pairs=(("multi_boxes", "ground"),),
    )

    pairs = {frozenset((candidate.distance.first, candidate.distance.second)) for candidate in candidates}
    assert frozenset(("robot_geom", "multi_boxes_link_1")) in pairs
    assert frozenset(("ground", "multi_boxes_link_1")) not in pairs
    assert all(candidate.jacobian.shape == (4,) for candidate in candidates)
    assert np.allclose(backend.model.geom_margin, original_margins)


def test_mujoco_backend_extracts_limited_hinge_and_slide_ranges_with_spec_overrides():
    robot = RobotSpec(
        name="mujoco_fixture",
        dof=3,
        height_m=1.0,
        joint_names=("hip", "knee", "ankle"),
        joint_limits={"ankle": (-0.25, 0.25)},
    )
    backend = MuJoCoKinematicsBackend.__new__(MuJoCoKinematicsBackend)
    backend._mujoco = _FakeMujoco
    backend.robot = robot
    backend.model = SimpleNamespace(
        njnt=4,
        jnt_type=np.array(
            [
                _FakeMujoco.mjtJoint.mjJNT_FREE,
                _FakeMujoco.mjtJoint.mjJNT_HINGE,
                _FakeMujoco.mjtJoint.mjJNT_SLIDE,
                _FakeMujoco.mjtJoint.mjJNT_HINGE,
            ]
        ),
        jnt_limited=np.array([False, True, True, True]),
        jnt_range=np.asarray(
            [
                [0.0, 0.0],
                [-1.0, 1.0],
                [-2.0, 2.0],
                [-3.0, 3.0],
            ],
            dtype=np.float64,
        ),
        jnt_qposadr=np.array([0, 7, 8, 9]),
        joint_names=("root", "hip", "sim_knee", "ankle"),
    )

    limits = backend.joint_limits()

    assert limits["hip"] == (-1.0, 1.0)
    assert limits["knee"] == (-2.0, 2.0)
    assert limits["ankle"] == (-0.25, 0.25)


def test_engine_joint_limit_arrays_use_backend_limits_then_robot_fallback():
    robot = RobotSpec(
        name="limit_fixture",
        dof=2,
        height_m=1.0,
        joint_names=("hip", "knee"),
        joint_limits={"knee": (-3.0, 3.0)},
    )
    backend = SimpleNamespace(joint_limits=lambda: {"hip": (-1.0, 1.0)})

    lower, upper = _joint_limit_arrays(robot, backend)

    assert lower == [-1.0, -3.0]
    assert upper == [1.0, 3.0]


class _FakeJoint:
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_HINGE = 2
    mjJNT_SLIDE = 3


class _FakeObject:
    mjOBJ_GEOM = 0
    mjOBJ_JOINT = 1


class _FakeMujoco:
    mjtJoint = _FakeJoint
    mjtObj = _FakeObject

    @staticmethod
    def mj_id2name(model, obj_type, object_id):
        if obj_type == _FakeObject.mjOBJ_JOINT:
            return model.joint_names[object_id]
        return f"geom_{object_id}"
