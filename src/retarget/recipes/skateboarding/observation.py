"""Observation recipe for native skateboarding capture."""

from __future__ import annotations

from dataclasses import dataclass, field

from retarget.capture import (
    HumanPoseRecording,
    MocapRecording,
    ObservationSource,
    TemporalRegistrationConfig,
)
from retarget.core.enums import (
    CropPolicy,
    FrameConvention,
    TimelineSelection,
)
from retarget.observation import (
    FootSupportClassificationConfig,
    SceneObservation,
)

from .alignment import (
    estimate_skateboarding_clock,
    register_actor_to_shoes,
    resample_human_pose,
    resample_mocap_body,
    select_observation_timeline,
)
from .scene import (
    classify_skateboarding_contacts,
    observed_board,
    semantic_landmarks,
)
from .vocabulary import SkateboardingMotionJoint, SkateboardingRigidBody


@dataclass(frozen=True)
class SkateboardingObservationRecipe:
    """Fuse native mocap and human-pose recordings into one observation."""

    mocap: ObservationSource[MocapRecording]
    human_pose: ObservationSource[HumanPoseRecording]
    timeline_selection: TimelineSelection = TimelineSelection.HUMAN_POSE
    crop_policy: CropPolicy = CropPolicy.OVERLAP
    uniform_fps: float | None = None
    temporal_registration: TemporalRegistrationConfig = field(
        default_factory=lambda: TemporalRegistrationConfig(
            max_abs_offset_s=20.0,
            min_overlap_s=1.0,
            minimum_score=0.28,
        )
    )
    support_classification: FootSupportClassificationConfig = field(default_factory=FootSupportClassificationConfig)
    max_frames: int | None = None
    world_frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED

    def observe(self) -> SceneObservation:
        """Load, align, reconstruct, and classify the capture in memory."""

        mocap = self.mocap.load()
        human = self.human_pose.load()
        temporal_report = estimate_skateboarding_clock(
            mocap,
            human,
            self.temporal_registration,
        )
        clock = temporal_report.clock_transform
        if clock is None:
            raise RuntimeError("temporal registration did not produce a clock transform")
        timeline = select_observation_timeline(
            mocap,
            human,
            clock,
            selection=self.timeline_selection,
            crop_policy=self.crop_policy,
            uniform_fps=self.uniform_fps,
            max_frames=self.max_frames,
        )
        actor = resample_human_pose(
            human,
            timeline,
            world_frame=self.world_frame,
        )
        bodies = tuple(
            resample_mocap_body(
                track,
                mocap,
                timeline,
                clock,
                world_frame=self.world_frame,
            )
            for track in mocap.rigid_bodies
        )
        body_map = {track.role: track for track in bodies}
        left_shoe = body_map[SkateboardingRigidBody.LEFT_SHOE]
        right_shoe = body_map[SkateboardingRigidBody.RIGHT_SHOE]
        board = body_map[SkateboardingRigidBody.BOARD]
        contacts = classify_skateboarding_contacts(
            timeline=timeline,
            left_shoe=left_shoe,
            right_shoe=right_shoe,
            board=board,
            config=self.support_classification,
        )
        actor, spatial_report = register_actor_to_shoes(
            actor,
            left_shoe,
            right_shoe,
            contacts,
        )
        return SceneObservation(
            name=human.name or mocap.name,
            timeline=timeline,
            world_frame=self.world_frame,
            actor=actor.to_motion_sequence(SkateboardingMotionJoint.PELVIS),
            landmarks=semantic_landmarks(actor),
            rigid_bodies=bodies,
            objects=(observed_board(board),),
            contacts=contacts,
            alignment_reports=(temporal_report, spatial_report),
            provenance={
                "mocap": mocap.provenance,
                "human_pose": human.provenance,
            },
        )
