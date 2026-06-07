"""MOCAP motion preprocessing for Holosoma-compatible recipes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.motion.spec import MotionFormatSpec

from .vocabulary import HolosomaMocapJoint

_MOCAP_JOINTS = tuple(HolosomaMocapJoint)


def mocap_motion_format(
    *,
    default_fps: float = 30.0,
    default_height_m: float = 1.78,
) -> MotionFormatSpec:
    """Return the Holosoma MOCAP format contract."""

    return MotionFormatSpec(
        name="holosoma_mocap",
        joint_vocabulary=HolosomaMocapJoint,
        root_joint=HolosomaMocapJoint.HIPS,
        default_fps=default_fps,
        default_height_m=default_height_m,
        description="Holosoma MOCAP climbing joint order.",
    )


def preprocess_mocap_climb(
    human_joints: np.ndarray,
    *,
    scale: float,
    mat_height: float = 0.1,
    demo_joints: tuple[HolosomaMocapJoint, ...] = _MOCAP_JOINTS,
    foot_names: tuple[HolosomaMocapJoint, HolosomaMocapJoint] = (
        HolosomaMocapJoint.LEFT_TOE_BASE,
        HolosomaMocapJoint.RIGHT_TOE_BASE,
    ),
) -> np.ndarray:
    """Apply Holosoma's climbing MOCAP height normalization and scaling."""

    joints = np.asarray(human_joints, dtype=np.float64).copy()
    left_idx = demo_joints.index(foot_names[0])
    right_idx = demo_joints.index(foot_names[1])
    z_min = float(joints[:, (left_idx, right_idx), 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    joints[:, :, 2] -= z_min
    return joints * float(scale)


def compute_climb_q_init(
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    *,
    robot_dof: int,
    demo_joints: tuple[HolosomaMocapJoint, ...] = _MOCAP_JOINTS,
    spine_joint_name: HolosomaMocapJoint = HolosomaMocapJoint.SPINE1,
) -> np.ndarray:
    """Compute Holosoma's initial floating-base qpos for climbing."""

    _translation, quaternion = transform_from_human_to_world(
        human_joints[0, 0, :],
        object_poses[0],
        np.zeros(3, dtype=np.float64),
    )
    spine_idx = demo_joints.index(spine_joint_name)
    return np.concatenate([human_joints[0, spine_idx], quaternion, np.zeros(robot_dof, dtype=np.float64)])


def transform_from_human_to_world(
    human_initial_root: np.ndarray,
    object_initial_pose: np.ndarray,
    local_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Holosoma's human-local to world transform helper."""

    human_to_object_2d = (
        np.asarray(object_initial_pose, dtype=np.float64)[-3:-1]
        - np.asarray(
            human_initial_root,
            dtype=np.float64,
        )[:2]
    )
    norm = float(np.linalg.norm(human_to_object_2d))
    if norm <= 1e-12:
        raise ValueError("human root and object origin must not coincide in the horizontal plane")
    x_axis_2d = human_to_object_2d / norm
    x_axis = np.array([x_axis_2d[0], x_axis_2d[1], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= float(np.linalg.norm(y_axis))
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return rotation_matrix @ np.asarray(local_translation, dtype=np.float64), quaternion
