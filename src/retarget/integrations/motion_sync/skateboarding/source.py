"""Skateboarding source preparation from synchronized clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core.enums import MotionFormat, TaskKind
from retarget.integrations.motion_sync import contact_plan_from_sync_clip, from_sync_clip
from retarget.motion import motion_formats
from retarget.pipeline.recipe import PreparedRetargetingInputs
from retarget.robots.spec import RobotSpec

from .alignment import aligned_smplx_joints, board_trajectory, estimate_fps
from .ecosystem import load_ecosystem
from .targets import link_target_plan, target_plan_at_indices
from .vocabulary import (
    DECK_SAMPLE_POINTS,
    FOOT_TARGET_LINKS,
    SkateboardingContactSubject,
    SkateboardingMotionJoint,
)


@dataclass(frozen=True)
class SkateboardingClipSource:
    """Typed loader for a synchronized skateboarding capture clip."""

    synced_path: Path
    name: str = ""
    max_frames: int | None = None
    height_m: float | None = None
    force_contacts: bool = False
    save_contact_layer: bool = False

    def prepare(self, robot: RobotSpec) -> PreparedRetargetingInputs:
        """Load source data and map contact subjects to ``robot`` contact links."""

        return from_skateboarding_clip(
            self.synced_path,
            name=self.name,
            max_frames=self.max_frames,
            height_m=self.height_m,
            force_contacts=self.force_contacts,
            save_contact_layer=self.save_contact_layer,
            contact_links=robot.contact_links,
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
) -> PreparedRetargetingInputs:
    """Build retarget-ready inputs from a synchronized skateboarding clip."""

    ecosystem = load_ecosystem()
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

    fps = estimate_fps(np.asarray(clip.time_s, dtype=np.float64))
    clip_name = name or clip.name or (synced_path.parent.name if synced_path.name == "synced.npz" else synced_path.stem)
    joint_names = tuple(member.value for member in ecosystem["SmplxCoreJoints"])
    joint_positions = aligned_smplx_joints(clip, ecosystem)
    stance = np.asarray(foot_support_data.stance_matrix(), dtype=bool)
    board_positions, board_quaternions = board_trajectory(clip, ecosystem)
    targets = link_target_plan(
        joint_positions=joint_positions,
        joint_names=joint_names,
        clip=clip,
        ecosystem=ecosystem,
        stance=stance,
    )

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frame_slice = slice(0, min(max_frames, joint_positions.shape[0]))
        joint_positions = joint_positions[frame_slice]
        stance = stance[frame_slice]
        board_positions = board_positions[frame_slice]
        board_quaternions = board_quaternions[frame_slice]
        targets = target_plan_at_indices(targets, np.arange(joint_positions.shape[0]))

    contact_plan = contact_plan_from_sync_clip(
        clip,
        contact_type=foot_support,
        contact_link_mapping=_contact_link_mapping(ecosystem, contact_links),
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

    root_positions = joint_positions[:, joint_names.index(SkateboardingMotionJoint.PELVIS.value), :]
    root_quaternions = np.zeros((joint_positions.shape[0], 4), dtype=np.float64)
    root_quaternions[:, 0] = 1.0
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
        object_name=SkateboardingContactSubject.SKATEBOARD.value,
        object_positions=board_positions,
        object_quaternions=board_quaternions,
        object_sample_points=DECK_SAMPLE_POINTS,
        task_kind=TaskKind.OBJECT_INTERACTION,
        metadata=metadata,
    )
    return PreparedRetargetingInputs(
        motion=prepared.motion,
        scene=prepared.scene.model_copy(update={"ground_range": (-3.0, 3.0), "ground_size": 15}),
        contacts=contact_plan,
        targets=targets,
        motion_format=motion_formats.get(MotionFormat.SMPLX),
        metadata={**prepared.metadata, **metadata},
    )


def _contact_link_mapping(
    ecosystem: dict[str, Any],
    contact_links: tuple[str, ...],
) -> dict[str, tuple[str, ...] | str]:
    bodies = ecosystem["Bodies"]
    left = tuple(link for link in contact_links if "left" in link.lower() or link.lower().startswith(("l_", "l-")))
    right = tuple(link for link in contact_links if "right" in link.lower() or link.lower().startswith(("r_", "r-")))
    return {
        bodies.LEFT_SHOE.value: left or FOOT_TARGET_LINKS[0].value,
        bodies.RIGHT_SHOE.value: right or FOOT_TARGET_LINKS[1].value,
    }
