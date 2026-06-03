"""Fuse motion-sync synced clips + foot support for retarget skateboarding.

Example (pushoff5_twoshoes):

    uv run python examples/skateboarding/fuse_unified.py \\
        --synced /path/to/motion-sync/output/synced/pushoff5_twoshoes/synced.npz \\
        --output examples/skateboarding/data/pushoff5_twoshoes

Requires ``motion-sync`` and ``contact_detection`` (event_detection):

    uv pip install -e ../motion-sync
    uv pip install -e ../event_detection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rich.console import Console

from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.pose import convert_points_frame, reorder_quaternion

CONTACT_JOINTS = ("L_Foot", "R_Foot")

# Standard deck sample box in skateboard object frame (meters).
DECK_SAMPLE_POINTS = np.asarray(
    [[x, y, z] for x in (-0.38, 0.38) for y in (-0.10, 0.10) for z in (-0.01, 0.01)],
    dtype=np.float64,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synced",
        type=Path,
        required=True,
        help="Path to synced.npz or demo directory (motion-sync output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for skate_motion.npz, board_trajectory.npz, deck_samples.npy.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Clip name for metadata (defaults to synced.npz parent directory name).",
    )
    return parser.parse_args()


def _load_clip(synced_path: Path):
    try:
        from motion_sync import SyncClip
        from motion_sync.schemas.skateboarding import (
            Bodies,
            SKATE_FOOT_SUPPORT,
            SKATE_SESSION,
            SKATE_VIDEO,
            SmplxCoreJoints,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install motion-sync on PYTHONPATH:\n"
            "  uv pip install -e /path/to/motion-sync"
        ) from exc

    clip = SyncClip.load(synced_path, session=SKATE_SESSION)
    if clip.frame_count == 0:
        raise ValueError(f"{synced_path} has no frames.")

    max_fk_index = max(SKATE_VIDEO.source_indices)
    if clip.video.joint_count <= max_fk_index:
        raise ValueError(
            f"video joints has {clip.video.joint_count} joints; need at least "
            f"{max_fk_index + 1} for SMPL-X core extraction."
        )

    if clip.vicon.body_orientations is None:
        raise ValueError("synced clip is missing vicon body orientations (vicon__body_quat).")

    if not clip.contact_is_fresh(SKATE_FOOT_SUPPORT):
        clip = clip.detect(SKATE_FOOT_SUPPORT)

    return clip, Bodies, SmplxCoreJoints


def _extract_smplx_joints(clip) -> np.ndarray:
    """Pull retarget smplx-order joints from FK and convert Y-up to Z-up."""
    picked = clip.core_joint_positions()
    return convert_points_frame(
        picked,
        FrameConvention.Y_UP_RIGHT_HANDED,
        FrameConvention.Z_UP_RIGHT_HANDED,
    )


def _stance_from_clip(clip) -> np.ndarray:
    from motion_sync.schemas.skateboarding import SKATE_FOOT_SUPPORT

    return clip.contact(SKATE_FOOT_SUPPORT).stance_matrix()


def _kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform mapping source Nx3 points to target Nx3 points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("need at least three paired points for Kabsch alignment")

    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    dst_centered = target - dst_centroid
    covariance = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return rotation, translation


def _align_human_to_vicon(
    joint_positions: np.ndarray,
    clip,
    bodies,
    joints,
    contact_states: np.ndarray,
) -> np.ndarray:
    """Register Y-up-converted SMPL-X feet into the Vicon Z-up lab frame."""
    left_track = clip.body(bodies.LEFT_SHOE)
    right_track = clip.body(bodies.RIGHT_SHOE)
    from motion_sync.schemas.skateboarding import SKATE_VIDEO

    left_idx = SKATE_VIDEO.core_index(joints.L_FOOT)
    right_idx = SKATE_VIDEO.core_index(joints.R_FOOT)

    stance_mask = contact_states.any(axis=1)
    if not np.any(stance_mask):
        stance_mask = np.ones(len(joint_positions), dtype=bool)

    source_pairs: list[np.ndarray] = []
    target_pairs: list[np.ndarray] = []
    for frame in np.flatnonzero(stance_mask):
        source_pairs.append(joint_positions[frame, left_idx])
        source_pairs.append(joint_positions[frame, right_idx])
        target_pairs.append(left_track.positions[frame])
        target_pairs.append(right_track.positions[frame])

    rotation, translation = _kabsch_transform(
        np.asarray(source_pairs, dtype=np.float64),
        np.asarray(target_pairs, dtype=np.float64),
    )
    return joint_positions @ rotation.T + translation


def _board_trajectory(clip, bodies, fps: float) -> tuple[np.ndarray, np.ndarray, float]:
    board = clip.body(bodies.SKATEBOARD)
    positions = np.asarray(board.positions, dtype=np.float64)
    if board.orientations is None:
        raise ValueError("skateboard rigid body has no orientations on clip")
    quaternions_xyzw = np.asarray(board.orientations, dtype=np.float64)
    quaternions_wxyz = np.stack(
        [
            reorder_quaternion(q, QuaternionOrder.XYZW, QuaternionOrder.WXYZ)
            for q in quaternions_xyzw
        ],
        axis=0,
    )
    return positions, quaternions_wxyz, float(fps)


def fuse_unified(
    synced_path: Path,
    output_dir: Path,
    *,
    name: str = "",
) -> None:
    clip, bodies, joints = _load_clip(synced_path)
    t = np.asarray(clip.time_s, dtype=np.float64)
    fps = float(1.0 / np.mean(np.diff(t))) if len(t) > 1 else 30.0
    clip_name = name or (synced_path.parent.name if synced_path.name == "synced.npz" else synced_path.stem)

    joint_positions = _extract_smplx_joints(clip)
    contact_states = _stance_from_clip(clip)
    joint_positions = _align_human_to_vicon(
        joint_positions, clip, bodies, joints, contact_states
    )
    board_positions, board_quaternions, board_fps = _board_trajectory(clip, bodies, fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "skate_motion.npz",
        joint_positions=joint_positions,
        fps=np.float64(fps),
        contact_states=contact_states,
        contact_names=np.array(CONTACT_JOINTS),
        frame_convention=np.array("z_up_right_handed"),
        name=np.array(clip_name),
    )
    np.save(output_dir / "deck_samples.npy", DECK_SAMPLE_POINTS)
    np.savez(
        output_dir / "board_trajectory.npz",
        positions=board_positions,
        quaternions=board_quaternions,
        fps=board_fps,
        frame_convention=np.array("z_up_right_handed"),
        quaternion_order=np.array("wxyz"),
    )


def main() -> None:
    args = _parse_args()
    console = Console()
    fuse_unified(args.synced.resolve(), args.output.resolve(), name=args.name)
    console.print(
        f"Fused [bold]{args.synced.parent.name}[/bold] → [cyan]{args.output.resolve()}[/cyan]\n"
        "Next:\n"
        f"  uv run retarget run --config examples/skateboarding/run_config_pushoff5.toml"
    )


if __name__ == "__main__":
    main()
