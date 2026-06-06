"""Target schemas for the skateboarding recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retarget.motion.targets import LinkTargetPlan

from .vocabulary import (
    FOOT_TARGET_LINKS,
    UPPER_COM_JOINTS,
    UPPER_COM_TARGET_LINK,
    SkateboardingMotionJoint,
    SkateboardingRobotLink,
)


@dataclass(frozen=True)
class LowerBodyTarget:
    """Mapping from a source motion joint to a robot link target."""

    joint: SkateboardingMotionJoint
    link: SkateboardingRobotLink
    weight: float


LOWER_BODY_TARGETS = (
    LowerBodyTarget(SkateboardingMotionJoint.LEFT_HIP, SkateboardingRobotLink.LEFT_HIP_PITCH, 2.0),
    LowerBodyTarget(SkateboardingMotionJoint.RIGHT_HIP, SkateboardingRobotLink.RIGHT_HIP_PITCH, 2.0),
    LowerBodyTarget(SkateboardingMotionJoint.LEFT_KNEE, SkateboardingRobotLink.LEFT_KNEE, 3.0),
    LowerBodyTarget(SkateboardingMotionJoint.RIGHT_KNEE, SkateboardingRobotLink.RIGHT_KNEE, 3.0),
    LowerBodyTarget(SkateboardingMotionJoint.LEFT_ANKLE, SkateboardingRobotLink.LEFT_ANKLE_PITCH, 2.0),
    LowerBodyTarget(SkateboardingMotionJoint.RIGHT_ANKLE, SkateboardingRobotLink.RIGHT_ANKLE_PITCH, 2.0),
)


def link_target_plan(
    *,
    joint_positions: np.ndarray,
    joint_names: tuple[str, ...],
    clip: Any,
    ecosystem: dict[str, Any],
    stance: np.ndarray,
) -> LinkTargetPlan:
    """Build named link targets for the skateboarding recipe."""

    target_names, target_positions, target_weights, target_masks = link_target_arrays(
        joint_positions=joint_positions,
        joint_names=joint_names,
        clip=clip,
        ecosystem=ecosystem,
        stance=stance,
    )
    return LinkTargetPlan.from_arrays(
        link_names=target_names,
        positions=target_positions,
        weights=target_weights,
        active_mask=target_masks,
        provenance={"source": "motion_sync:foot_support+video_core_joints"},
    )


def target_plan_at_indices(plan: LinkTargetPlan, indices: np.ndarray) -> LinkTargetPlan:
    """Return a target plan sampled at explicit frame indices."""

    return LinkTargetPlan(
        tracks=tuple(track.resampled_indices(indices) for track in plan.tracks),
        frame_count=len(indices),
        provenance=dict(plan.provenance),
    )


def link_target_arrays(
    *,
    joint_positions: np.ndarray,
    joint_names: tuple[str, ...],
    clip: Any,
    ecosystem: dict[str, Any],
    stance: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return link target names, positions, weights, and active masks."""

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
        names.append(link_name.value)
        positions.append(target)
        weights.append(np.where(stance[:, side], 80.0, 8.0))
        masks.append(np.asarray(np.isfinite(target).all(axis=1), dtype=bool))

    for target_spec in LOWER_BODY_TARGETS:
        target = joint_positions[:, _joint_index(joint_names, target_spec.joint), :]
        names.append(target_spec.link.value)
        positions.append(target)
        weights.append(np.full(joint_positions.shape[0], target_spec.weight, dtype=np.float64))
        masks.append(np.asarray(np.isfinite(target).all(axis=1), dtype=bool))

    upper_positions = np.stack([joint_positions[:, _joint_index(joint_names, name), :] for name in UPPER_COM_JOINTS])
    upper_com = np.mean(upper_positions, axis=0)
    names.append(UPPER_COM_TARGET_LINK.value)
    positions.append(upper_com)
    weights.append(np.full(joint_positions.shape[0], 1.0, dtype=np.float64))
    masks.append(np.asarray(np.isfinite(upper_com).all(axis=1), dtype=bool))

    return (
        tuple(names),
        np.stack(positions, axis=1),
        np.stack(weights, axis=1),
        np.stack(masks, axis=1),
    )


def _joint_index(joint_names: tuple[str, ...], name: SkateboardingMotionJoint) -> int:
    try:
        return joint_names.index(name.value)
    except ValueError as exc:
        raise KeyError(f"SMPL-X core joint {name.value!r} is not present") from exc
