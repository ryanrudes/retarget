"""Skateboarding ``motion_sync`` adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core.enums import FrameConvention, QuaternionOrder, TaskKind
from retarget.core.pose import convert_points_frame, reorder_quaternion
from retarget.integrations.motion_sync import PreparedRetargetInputs, contact_plan_from_sync_clip, from_sync_clip
from retarget.motion.targets import LinkTargetPlan

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEMO = "pushoff5_twoshoes"
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


def from_skateboarding_clip(
    synced_path: Path,
    *,
    name: str = "",
    max_frames: int | None = None,
    height_m: float | None = None,
    force_contacts: bool = False,
    save_contact_layer: bool = False,
    contact_links: tuple[str, ...] = (),
) -> PreparedRetargetInputs:
    """Build retarget-ready inputs from a synchronized skateboarding clip."""

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
    foot_support_data = clip.contact(foot_support)

    fps = _estimate_fps(np.asarray(clip.time_s, dtype=np.float64))
    clip_name = name or clip.name or (synced_path.parent.name if synced_path.name == "synced.npz" else synced_path.stem)
    joint_names = tuple(member.value for member in ecosystem["SmplxCoreJoints"])
    joint_positions = _aligned_smplx_joints(clip, ecosystem)
    stance = np.asarray(foot_support_data.stance_matrix(), dtype=bool)
    board_positions, board_quaternions = _board_trajectory(clip, ecosystem)
    target_names, target_positions, target_weights, target_masks = _link_targets(
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
        target_positions = target_positions[frame_slice]
        target_weights = target_weights[frame_slice]
        target_masks = target_masks[frame_slice]

    contact_mapping = _contact_link_mapping(ecosystem, contact_links)
    contact_plan = contact_plan_from_sync_clip(
        clip,
        contact_type=foot_support,
        contact_link_mapping=contact_mapping,
        frame_count=clip.frame_count,
    )
    if contact_plan is not None and max_frames is not None:
        indices = np.arange(joint_positions.shape[0])
        contact_plan = contact_plan.__class__(
            tracks=tuple(track.resampled_indices(indices) for track in contact_plan.tracks),
            frame_count=joint_positions.shape[0],
            support=contact_plan.support,
            provenance=dict(contact_plan.provenance),
        )

    root_positions = joint_positions[:, _joint_index(joint_names, "Pelvis"), :]
    root_quaternions = np.zeros((joint_positions.shape[0], 4), dtype=np.float64)
    root_quaternions[:, 0] = 1.0
    targets = LinkTargetPlan.from_arrays(
        link_names=target_names,
        positions=target_positions,
        weights=target_weights,
        active_mask=target_masks,
        provenance={"source": "motion_sync:foot_support+video_core_joints"},
    )
    metadata = {"source": "motion_sync_skateboarding", "demo": clip_name}
    prepared = from_sync_clip(
        clip,
        joint_names=joint_names,
        joint_positions=joint_positions,
        name=clip_name,
        fps=fps,
        root_positions=root_positions,
        root_quaternions=root_quaternions,
        height_m=height_m,
        targets=targets,
        contact_layer=None,
        support=contact_plan.support if contact_plan is not None else None,
        object_name="skateboard",
        object_positions=board_positions,
        object_quaternions=board_quaternions,
        object_sample_points=DECK_SAMPLE_POINTS,
        task_kind=TaskKind.OBJECT_INTERACTION,
        metadata=metadata,
    )
    return PreparedRetargetInputs(
        motion=prepared.motion,
        scene=prepared.scene.model_copy(update={"ground_range": (-3.0, 3.0), "ground_size": 15}),
        contacts=contact_plan,
        targets=targets,
        metadata={**prepared.metadata, **metadata},
    )


def _load_ecosystem() -> dict[str, Any]:
    for path in (REPO_ROOT / "vendor" / "event_detection" / "src", REPO_ROOT / "vendor" / "motion_sync"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from motion_sync import SyncClip  # type: ignore[import-untyped]
        from motion_sync.schemas.skateboarding import (  # type: ignore[import-untyped]
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


def _contact_link_mapping(
    ecosystem: dict[str, Any],
    contact_links: tuple[str, ...],
) -> dict[str, tuple[str, ...] | str]:
    bodies = ecosystem["Bodies"]
    left = tuple(link for link in contact_links if "left" in link.lower() or link.lower().startswith(("l_", "l-")))
    right = tuple(link for link in contact_links if "right" in link.lower() or link.lower().startswith(("r_", "r-")))
    return {
        bodies.LEFT_SHOE.value: left or FOOT_TARGET_LINKS[0],
        bodies.RIGHT_SHOE.value: right or FOOT_TARGET_LINKS[1],
    }


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
    video_schema = ecosystem.get("SKATE_VIDEO")
    left_track = clip.body(bodies.LEFT_SHOE)
    right_track = clip.body(bodies.RIGHT_SHOE)
    left_idx = video_schema.core_index(joints.L_FOOT) if video_schema is not None else _joint_index(
        tuple(member.value for member in joints),
        "L_Foot",
    )
    right_idx = video_schema.core_index(joints.R_FOOT) if video_schema is not None else _joint_index(
        tuple(member.value for member in joints),
        "R_Foot",
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

    rotation, translation = _kabsch_transform(
        np.asarray(source_pairs, dtype=np.float64),
        np.asarray(target_pairs, dtype=np.float64),
    )
    return np.asarray(joint_positions @ rotation.T + translation, dtype=np.float64)


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
        masks.append(np.asarray(np.isfinite(target).all(axis=1), dtype=bool))

    for joint_name, link_name, weight in LOWER_BODY_LINK_TARGETS:
        target = joint_positions[:, _joint_index(joint_names, joint_name), :]
        names.append(link_name)
        positions.append(target)
        weights.append(np.full(joint_positions.shape[0], weight, dtype=np.float64))
        masks.append(np.asarray(np.isfinite(target).all(axis=1), dtype=bool))

    upper_positions = np.stack([joint_positions[:, _joint_index(joint_names, name), :] for name in UPPER_COM_JOINTS])
    upper_com = np.mean(upper_positions, axis=0)
    names.append(UPPER_COM_TARGET_LINK)
    positions.append(upper_com)
    weights.append(np.full(joint_positions.shape[0], 1.0, dtype=np.float64))
    masks.append(np.asarray(np.isfinite(upper_com).all(axis=1), dtype=bool))

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
