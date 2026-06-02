"""Synthetic skateboarding fixtures for docs and examples (no external datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retarget import MotionFormat, ObjectSpec, ObjectTrajectory, SceneSpec, TaskKind
from retarget.core.enums import QuaternionOrder
from retarget.core.pose import PoseSequence
from retarget.motion import MotionFormatSpec, MotionSequence, motion_formats

FRAME_COUNT = 12
FPS = 30.0
FORMAT_NAME = MotionFormat.MINIMAL


def motion_format() -> MotionFormatSpec:
    return motion_formats.get(FORMAT_NAME)


def synthetic_skate_motion() -> MotionSequence:
    """Fused human motion with explicit foot contacts.

    This is a tiny deterministic stand-in for a fused SMPL-X clip: the pelvis,
    feet, head, and wrists move forward with the board so examples remain
    inspectable without external datasets.
    """

    fmt = motion_format()
    names = fmt.joint_names
    positions = np.zeros((FRAME_COUNT, len(names), 3), dtype=np.float64)
    x = _board_progress()
    sway = 0.025 * np.sin(np.linspace(0.0, np.pi, FRAME_COUNT))
    left_foot = np.column_stack([x - 0.18, -0.07 + sway, np.full(FRAME_COUNT, 0.07)])
    right_foot = np.column_stack([x + 0.18, 0.07 - sway, np.full(FRAME_COUNT, 0.07)])

    _set_joint(positions, names, "Pelvis", np.column_stack([x, np.zeros(FRAME_COUNT), np.full(FRAME_COUNT, 0.95)]))
    _set_joint(positions, names, "L_Hip", np.column_stack([x - 0.12, _constant(-0.08), _constant(0.72)]))
    _set_joint(positions, names, "L_Knee", np.column_stack([x - 0.15, _constant(-0.08), _constant(0.38)]))
    _set_joint(positions, names, "L_Toe", left_foot)
    _set_joint(positions, names, "R_Hip", np.column_stack([x + 0.12, _constant(0.08), _constant(0.72)]))
    _set_joint(positions, names, "R_Knee", np.column_stack([x + 0.15, _constant(0.08), _constant(0.38)]))
    _set_joint(positions, names, "R_Toe", right_foot)
    _set_joint(positions, names, "Spine", np.column_stack([x, np.zeros(FRAME_COUNT), np.full(FRAME_COUNT, 1.12)]))
    _set_joint(positions, names, "Head", np.column_stack([x, np.zeros(FRAME_COUNT), np.full(FRAME_COUNT, 1.36)]))
    _set_joint(positions, names, "L_Wrist", np.column_stack([x - 0.42, _constant(-0.04), _constant(0.92)]))
    _set_joint(positions, names, "R_Wrist", np.column_stack([x + 0.42, _constant(0.04), _constant(0.92)]))

    contacts = tuple(
        {
            "L_Toe": True,
            "R_Toe": frame >= FRAME_COUNT // 3,
        }
        for frame in range(FRAME_COUNT)
    )

    return MotionSequence(
        name="synthetic_skate_clip",
        joint_names=names,
        joint_positions=positions,
        fps=FPS,
        contacts=contacts,
        metadata={"height_m": 1.2, "source": "examples/skateboarding/_synthetic.py"},
    )


def synthetic_deck_sample_points() -> np.ndarray:
    """Deck sample points in the skateboard object frame (meters)."""

    return np.asarray(
        [
            [x, y, z]
            for x in (-0.38, 0.38)
            for y in (-0.10, 0.10)
            for z in (-0.01, 0.01)
        ],
        dtype=np.float64,
    )


def synthetic_board_trajectory() -> ObjectTrajectory:
    """Board pose track; swap for tracked board poses in production."""

    positions = np.column_stack([_board_progress(), np.zeros(FRAME_COUNT), np.full(FRAME_COUNT, 0.04)])
    quaternions = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (FRAME_COUNT, 1))
    return ObjectTrajectory(
        name="skateboard",
        poses=PoseSequence.from_arrays(
            positions,
            quaternions,
            fps=FPS,
            quaternion_order=QuaternionOrder.WXYZ,
        ),
    )


def synthetic_skate_scene() -> SceneSpec:
    return SceneSpec(
        task_kind=TaskKind.OBJECT_INTERACTION,
        object=ObjectSpec(
            name="skateboard",
            sample_points=synthetic_deck_sample_points(),
            trajectory=synthetic_board_trajectory(),
        ),
        ground_size=12,
        ground_range=(-2.0, 2.0),
    )


def write_fixture_files(data_dir: Path) -> None:
    """Write motion, deck samples, and board trajectory files for CLI run configs."""

    data_dir.mkdir(parents=True, exist_ok=True)
    motion = synthetic_skate_motion()
    fmt = motion_format()

    contact_names = np.asarray(fmt.contact_joints)
    contact_matrix = np.asarray(
        [[frame[name] for name in fmt.contact_joints] for frame in motion.contacts],
        dtype=bool,
    )

    np.savez(
        data_dir / "skate_motion.npz",
        joint_positions=motion.joint_positions,
        fps=motion.fps,
        contact_states=contact_matrix,
        contact_names=contact_names,
    )
    np.save(data_dir / "deck_samples.npy", synthetic_deck_sample_points())

    trajectory = synthetic_board_trajectory()
    np.savez(
        data_dir / "board_trajectory.npz",
        positions=trajectory.poses.positions,
        quaternions=trajectory.poses.quaternions(),
        fps=np.array(trajectory.poses.fps),
    )


def _board_progress() -> np.ndarray:
    return np.linspace(-0.45, 0.45, FRAME_COUNT, dtype=np.float64)


def _constant(value: float) -> np.ndarray:
    return np.full(FRAME_COUNT, value, dtype=np.float64)


def _set_joint(positions: np.ndarray, names: tuple[str, ...], joint: str, values: np.ndarray) -> None:
    positions[:, names.index(joint), :] = values
