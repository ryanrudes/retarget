"""Prepare a motion_sync skateboarding clip for retargeting.

This script consumes a synced skateboarding clip produced by ``motion_sync`` and
writes the three assets used by the retargeting example:

* ``skate_motion.npz`` with SMPL-X core joints, contact labels, and link targets
* ``board_trajectory.npz`` with the Vicon skateboard rigid-body trajectory
* ``deck_samples.npy`` with object-frame deck samples for scene constraints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.pose import convert_points_frame, reorder_quaternion

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO = "pushoff5_twoshoes"
CONTACT_JOINTS = ("L_Foot", "R_Foot")
FOOT_TARGET_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")

LOWER_BODY_LINK_TARGETS = (
    ("L_Hip", "left_hip_pitch_link", 2.0),
    ("R_Hip", "right_hip_pitch_link", 2.0),
    ("L_Knee", "left_knee_link", 3.0),
    ("R_Knee", "right_knee_link", 3.0),
    ("L_Ankle", "left_ankle_pitch_link", 2.0),
    ("R_Ankle", "right_ankle_pitch_link", 2.0),
)
UPPER_COM_JOINTS = (
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
)
UPPER_COM_TARGET_LINK = "torso_link"

DECK_SAMPLE_POINTS = np.asarray(
    [[x, y, z] for x in (-0.38, 0.38) for y in (-0.10, 0.10) for z in (-0.015, 0.015)],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default=DEFAULT_DEMO, help="Demo id under --synced-root.")
    parser.add_argument("--synced", type=Path, help="Path to synced.npz or a motion_sync demo directory.")
    parser.add_argument(
        "--synced-root",
        type=Path,
        default=REPO_ROOT / "motion_sync_output" / "synced",
        help="Directory containing motion_sync synced demo folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to examples/skateboarding/generated/<demo>.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame cap for smoke runs.")
    parser.add_argument("--force-contacts", action="store_true", help="Re-run foot-support detection.")
    parser.add_argument(
        "--save-contact-layer",
        action="store_true",
        help="Persist a freshly detected contact layer back to the synced clip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synced_path = _resolve_synced_path(args)
    output_dir = (args.output or (Path(__file__).resolve().parent / "generated" / args.demo)).resolve()
    summary = prepare_clip(
        synced_path,
        output_dir,
        name=args.demo,
        max_frames=args.max_frames,
        force_contacts=args.force_contacts,
        save_contact_layer=args.save_contact_layer,
    )
    Console().print(
        f"Prepared [bold]{summary['name']}[/bold] with {summary['frames']} frames at "
        f"{summary['fps']:.3f} Hz -> [cyan]{output_dir}[/cyan]"
    )


def prepare_clip(
    synced_path: Path,
    output_dir: Path,
    *,
    name: str = "",
    max_frames: int | None = None,
    force_contacts: bool = False,
    save_contact_layer: bool = False,
) -> dict[str, Any]:
    """Fuse synchronized video, Vicon, and detected contacts into retarget inputs."""

    ecosystem = _load_ecosystem()
    clip = ecosystem["SyncClip"].load(synced_path, session=ecosystem["SKATE_SESSION"])
    if clip.frame_count == 0:
        raise ValueError(f"{synced_path} has no frames")
    if clip.vicon.body_orientations is None:
        raise ValueError("synced clip is missing Vicon rigid-body orientations")

    foot_support = ecosystem["SKATE_FOOT_SUPPORT"]
    if force_contacts or not clip.contact_is_fresh(foot_support):
        clip = clip.detect(foot_support, force=force_contacts)
        if save_contact_layer:
            clip.save(synced_path)

    fps = _estimate_fps(np.asarray(clip.time_s, dtype=np.float64))
    clip_name = name or clip.name or (synced_path.parent.name if synced_path.name == "synced.npz" else synced_path.stem)
    joint_names = tuple(member.value for member in ecosystem["SmplxCoreJoints"])
    joint_positions = _aligned_smplx_joints(clip, ecosystem)
    stance = np.asarray(clip.contact(foot_support).stance_matrix(), dtype=bool)
    board_positions, board_quaternions = _board_trajectory(clip, ecosystem)
    link_target_names, link_target_positions, link_target_weights, link_target_masks = _link_targets(
        joint_positions,
        joint_names,
        clip,
        ecosystem,
        stance,
    )

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frame_slice = slice(0, min(max_frames, joint_positions.shape[0]))
        joint_positions = joint_positions[frame_slice]
        stance = stance[frame_slice]
        board_positions = board_positions[frame_slice]
        board_quaternions = board_quaternions[frame_slice]
        link_target_positions = link_target_positions[frame_slice]
        link_target_weights = link_target_weights[frame_slice]
        link_target_masks = link_target_masks[frame_slice]

    output_dir.mkdir(parents=True, exist_ok=True)
    root_positions = joint_positions[:, _joint_index(joint_names, "Pelvis"), :]
    root_quaternions = np.zeros((joint_positions.shape[0], 4), dtype=np.float64)
    root_quaternions[:, 0] = 1.0
    np.savez(
        output_dir / "skate_motion.npz",
        joint_positions=joint_positions,
        joint_names=np.asarray(joint_names, dtype=object),
        fps=np.asarray(fps, dtype=np.float64),
        frame_convention=np.asarray(FrameConvention.Z_UP_RIGHT_HANDED.value),
        root_positions=root_positions,
        root_quaternions=root_quaternions,
        root_quaternion_order=np.asarray(QuaternionOrder.WXYZ.value),
        contact_states=stance,
        contact_names=np.asarray(CONTACT_JOINTS, dtype=object),
        link_target_names=np.asarray(link_target_names, dtype=object),
        link_target_positions=link_target_positions,
        link_target_weights=link_target_weights,
        link_target_masks=link_target_masks,
        link_target_source=np.asarray("motion_sync:skate_foot_support+video_core_joints", dtype=object),
        name=np.asarray(clip_name, dtype=object),
    )
    np.save(output_dir / "deck_samples.npy", DECK_SAMPLE_POINTS)
    np.savez(
        output_dir / "board_trajectory.npz",
        positions=board_positions,
        quaternions=board_quaternions,
        fps=np.asarray(fps, dtype=np.float64),
        frame_convention=np.asarray(FrameConvention.Z_UP_RIGHT_HANDED.value),
        quaternion_order=np.asarray(QuaternionOrder.WXYZ.value),
    )
    return {"name": clip_name, "frames": int(joint_positions.shape[0]), "fps": fps}


def _load_ecosystem() -> dict[str, Any]:
    for path in (REPO_ROOT / "vendor" / "event_detection" / "src", REPO_ROOT / "vendor" / "motion_sync"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from motion_sync import SyncClip
        from motion_sync.schemas.skateboarding import (
            SKATE_FOOT_SUPPORT,
            SKATE_SESSION,
            SKATE_VIDEO,
            Bodies,
            SmplxCoreJoints,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import motion_sync/contact_detection. Run git submodule update --init, "
            "or install the sibling packages into this environment."
        ) from exc
    return {
        "SyncClip": SyncClip,
        "SKATE_FOOT_SUPPORT": SKATE_FOOT_SUPPORT,
        "SKATE_SESSION": SKATE_SESSION,
        "SKATE_VIDEO": SKATE_VIDEO,
        "Bodies": Bodies,
        "SmplxCoreJoints": SmplxCoreJoints,
    }


def _resolve_synced_path(args: argparse.Namespace) -> Path:
    if args.synced is not None:
        return args.synced.expanduser().resolve()
    return (args.synced_root.expanduser() / args.demo).resolve()


def _estimate_fps(time_s: np.ndarray) -> float:
    if time_s.shape[0] < 2:
        return 30.0
    deltas = np.diff(time_s[np.isfinite(time_s)])
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        return 30.0
    return float(1.0 / np.mean(positive))


def _aligned_smplx_joints(clip: Any, ecosystem: dict[str, Any]) -> np.ndarray:
    joints_z_up = convert_points_frame(
        clip.core_joint_positions(),
        FrameConvention.Y_UP_RIGHT_HANDED,
        FrameConvention.Z_UP_RIGHT_HANDED,
    )
    stance = np.asarray(clip.contact(ecosystem["SKATE_FOOT_SUPPORT"]).stance_matrix(), dtype=bool)
    return _align_human_to_vicon(joints_z_up, clip, ecosystem, stance)


def _align_human_to_vicon(
    joint_positions: np.ndarray,
    clip: Any,
    ecosystem: dict[str, Any],
    stance: np.ndarray,
) -> np.ndarray:
    bodies = ecosystem["Bodies"]
    joints = ecosystem["SmplxCoreJoints"]
    video_schema = ecosystem["SKATE_VIDEO"]
    left_track = clip.body(bodies.LEFT_SHOE)
    right_track = clip.body(bodies.RIGHT_SHOE)
    left_idx = video_schema.core_index(joints.L_FOOT)
    right_idx = video_schema.core_index(joints.R_FOOT)
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

    rotation, translation = _kabsch_transform(
        np.asarray(source_pairs, dtype=np.float64),
        np.asarray(target_pairs, dtype=np.float64),
    )
    return joint_positions @ rotation.T + translation


def _kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _board_trajectory(clip: Any, ecosystem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    board = clip.body(ecosystem["Bodies"].SKATEBOARD)
    if board.orientations is None:
        raise ValueError("skateboard rigid body has no orientations")
    quaternions = np.asarray(
        [
            reorder_quaternion(quat, QuaternionOrder.XYZW, QuaternionOrder.WXYZ)
            for quat in np.asarray(board.orientations, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    return np.asarray(board.positions, dtype=np.float64), quaternions


def _link_targets(
    joint_positions: np.ndarray,
    joint_names: tuple[str, ...],
    clip: Any,
    ecosystem: dict[str, Any],
    stance: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    positions: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    bodies = ecosystem["Bodies"]

    for side, body, link_name in (
        (0, bodies.LEFT_SHOE, FOOT_TARGET_LINKS[0]),
        (1, bodies.RIGHT_SHOE, FOOT_TARGET_LINKS[1]),
    ):
        track = clip.body(body)
        target = np.asarray(track.positions, dtype=np.float64)
        names.append(link_name)
        positions.append(target)
        weights.append(np.where(stance[:, side], 80.0, 8.0))
        masks.append(np.isfinite(target).all(axis=1))

    for joint_name, link_name, weight in LOWER_BODY_LINK_TARGETS:
        target = joint_positions[:, _joint_index(joint_names, joint_name), :]
        names.append(link_name)
        positions.append(target)
        weights.append(np.full(joint_positions.shape[0], weight, dtype=np.float64))
        masks.append(np.isfinite(target).all(axis=1))

    upper_positions = np.stack([joint_positions[:, _joint_index(joint_names, name), :] for name in UPPER_COM_JOINTS])
    upper_com = np.mean(upper_positions, axis=0)
    names.append(UPPER_COM_TARGET_LINK)
    positions.append(upper_com)
    weights.append(np.full(joint_positions.shape[0], 1.0, dtype=np.float64))
    masks.append(np.isfinite(upper_com).all(axis=1))

    return (
        tuple(names),
        np.stack(positions, axis=1),
        np.stack(weights, axis=1),
        np.stack(masks, axis=1),
    )


def _joint_index(joint_names: tuple[str, ...], name: str) -> int:
    try:
        return joint_names.index(name)
    except ValueError as exc:
        raise KeyError(f"SMPL-X core joint {name!r} is not present") from exc


if __name__ == "__main__":
    main()
