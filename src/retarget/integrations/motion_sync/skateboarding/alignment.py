"""Alignment helpers for skateboarding synchronized clips."""

from __future__ import annotations

from typing import Any

import numpy as np

from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.pose import convert_points_frame, reorder_quaternions

from .vocabulary import SkateboardingMotionJoint


def estimate_fps(time_s: np.ndarray) -> float:
    """Estimate FPS from monotonically increasing timestamps."""

    if time_s.shape[0] < 2:
        return 30.0
    deltas = np.diff(time_s[np.isfinite(time_s)])
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        return 30.0
    return float(1.0 / np.mean(positive))


def aligned_smplx_joints(clip: Any, ecosystem: dict[str, Any]) -> np.ndarray:
    """Convert video joints to Z-up and align them into the Vicon lab frame."""

    joints_z_up = convert_points_frame(
        clip.core_joint_positions(),
        FrameConvention.Y_UP_RIGHT_HANDED,
        FrameConvention.Z_UP_RIGHT_HANDED,
    )
    stance = np.asarray(clip.contact(ecosystem["SKATE_FOOT_SUPPORT"]).stance_matrix(), dtype=bool)
    return align_human_to_vicon(joints_z_up, clip, ecosystem, stance)


def align_human_to_vicon(
    joint_positions: np.ndarray,
    clip: Any,
    ecosystem: dict[str, Any],
    stance: np.ndarray,
) -> np.ndarray:
    """Rigidly align video joints to Vicon shoe tracks."""

    bodies = ecosystem["Bodies"]
    joints = ecosystem["SmplxCoreJoints"]
    video_schema = ecosystem.get("SKATE_VIDEO")
    left_track = clip.body(bodies.LEFT_SHOE)
    right_track = clip.body(bodies.RIGHT_SHOE)
    left_idx = video_schema.core_index(joints.L_FOOT) if video_schema is not None else _joint_index(
        tuple(member.value for member in joints),
        SkateboardingMotionJoint.LEFT_FOOT,
    )
    right_idx = video_schema.core_index(joints.R_FOOT) if video_schema is not None else _joint_index(
        tuple(member.value for member in joints),
        SkateboardingMotionJoint.RIGHT_FOOT,
    )
    source_pairs: list[np.ndarray] = []
    target_pairs: list[np.ndarray] = []

    candidate_frames = np.flatnonzero(stance.any(axis=1))
    if candidate_frames.size == 0:
        candidate_frames = np.arange(joint_positions.shape[0])
    for frame in candidate_frames:
        source = np.vstack([joint_positions[frame, left_idx], joint_positions[frame, right_idx]])
        target = np.vstack([left_track.positions[frame], right_track.positions[frame]])
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        source_pairs.extend(source)
        target_pairs.extend(target)
    if len(source_pairs) < 3:
        raise ValueError("not enough finite paired foot targets to align video joints into the Vicon frame")

    rotation, translation = kabsch_transform(
        np.asarray(source_pairs, dtype=np.float64),
        np.asarray(target_pairs, dtype=np.float64),
    )
    return np.asarray(joint_positions @ rotation.T + translation, dtype=np.float64)


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rigid transform aligning paired point sets."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)
    covariance = (source - src_centroid).T @ (target - dst_centroid)
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return rotation, translation


def board_trajectory(clip: Any, ecosystem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return skateboard positions and scalar-first quaternions."""

    board = clip.body(ecosystem["Bodies"].SKATEBOARD)
    if board.orientations is None:
        raise ValueError("skateboard rigid body has no orientations")
    quaternions = reorder_quaternions(
        np.asarray(board.orientations, dtype=np.float64),
        QuaternionOrder.XYZW,
        QuaternionOrder.WXYZ,
    )
    return np.asarray(board.positions, dtype=np.float64), quaternions


def _joint_index(joint_names: tuple[str, ...], name: SkateboardingMotionJoint) -> int:
    try:
        return joint_names.index(name.value)
    except ValueError as exc:
        raise KeyError(f"SMPL-X core joint {name.value!r} is not present") from exc
