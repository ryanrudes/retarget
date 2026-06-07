"""Temporal and spatial registration for skateboarding capture."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.capture import (
    AlignmentReport,
    ClockTransform,
    HumanPoseRecording,
    JointTrack,
    MocapRecording,
    PoseTrack,
    RigidBodyTrack,
    SampleTimeline,
    TemporalRegistrationConfig,
    estimate_clock_offset,
    register_rigid_points,
)
from retarget.core.enums import (
    CropPolicy,
    FrameConvention,
    QuaternionOrder,
    TimelineSelection,
)
from retarget.core.pose import (
    convert_points_frame,
    frame_transform_matrix,
    reorder_quaternions,
)
from retarget.observation import SemanticContactSequence

from .vocabulary import (
    SkateboardingContactState,
    SkateboardingMotionJoint,
    SkateboardingRigidBody,
)

LEFT_FOOT_CUE_JOINTS = (
    SkateboardingMotionJoint.LEFT_ANKLE,
    SkateboardingMotionJoint.LEFT_FOOT,
    SkateboardingMotionJoint.LEFT_BIG_TOE,
    SkateboardingMotionJoint.LEFT_SMALL_TOE,
    SkateboardingMotionJoint.LEFT_HEEL,
)
RIGHT_FOOT_CUE_JOINTS = (
    SkateboardingMotionJoint.RIGHT_ANKLE,
    SkateboardingMotionJoint.RIGHT_FOOT,
    SkateboardingMotionJoint.RIGHT_BIG_TOE,
    SkateboardingMotionJoint.RIGHT_SMALL_TOE,
    SkateboardingMotionJoint.RIGHT_HEEL,
)


def estimate_skateboarding_clock(
    mocap: MocapRecording,
    human: HumanPoseRecording,
    config: TemporalRegistrationConfig,
) -> AlignmentReport:
    """Estimate the mocap-to-observation clock from paired foot-speed cues."""

    mocap_timeline, mocap_signal = _mocap_speed_signal(mocap)
    human_timeline, human_signal = _human_speed_signal(human)
    return estimate_clock_offset(
        human_timeline,
        human_signal,
        mocap_timeline,
        mocap_signal,
        config,
        target_clock="observation",
    )


def select_observation_timeline(
    mocap: MocapRecording,
    human: HumanPoseRecording,
    clock: ClockTransform,
    *,
    selection: TimelineSelection,
    crop_policy: CropPolicy,
    uniform_fps: float | None,
    max_frames: int | None,
) -> SampleTimeline:
    """Select and crop the shared observation sampling grid."""

    human_times = human.timeline.timestamps
    mocap_times = clock.apply(mocap.timeline.timestamps)
    overlap_start = max(float(human_times[0]), float(mocap_times[0]))
    overlap_end = min(float(human_times[-1]), float(mocap_times[-1]))
    if crop_policy == CropPolicy.OVERLAP and overlap_end < overlap_start:
        raise ValueError("capture sources have no shared observation interval")

    if selection == TimelineSelection.HUMAN_POSE:
        selected_times = human_times
    elif selection == TimelineSelection.MOCAP:
        selected_times = mocap_times
    elif selection == TimelineSelection.UNIFORM:
        if uniform_fps is None or uniform_fps <= 0.0:
            raise ValueError("uniform timeline selection requires positive uniform_fps")
        if crop_policy == CropPolicy.OVERLAP:
            start_s, end_s = overlap_start, overlap_end
        else:
            start_s = min(float(human_times[0]), float(mocap_times[0]))
            end_s = max(float(human_times[-1]), float(mocap_times[-1]))
        sample_count = int(np.floor((end_s - start_s) * uniform_fps + 1e-9)) + 1
        selected_times = start_s + np.arange(sample_count, dtype=np.float64) / uniform_fps
    else:  # pragma: no cover - closed enum
        raise ValueError(f"unsupported timeline selection {selection!r}")

    if crop_policy == CropPolicy.OVERLAP and selection != TimelineSelection.UNIFORM:
        selected_times = selected_times[(selected_times >= overlap_start) & (selected_times <= overlap_end)]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        selected_times = selected_times[:max_frames]
    if selected_times.size == 0:
        raise ValueError("capture sources have no shared observation samples")
    return SampleTimeline(selected_times, clock="observation")


def resample_human_pose(
    human: HumanPoseRecording,
    timeline: SampleTimeline,
    *,
    world_frame: FrameConvention,
) -> HumanPoseRecording:
    """Convert and resample a native human-pose recording."""

    clock = ClockTransform(
        source_clock=human.timeline.clock,
        target_clock=timeline.clock,
    )
    joints = tuple(
        JointTrack(
            role=track.role,
            values=convert_points_frame(track.values, human.frame, world_frame),
            validity=track.validity,
            provenance=track.provenance,
        ).resample(human.timeline, timeline, clock=clock)
        for track in human.joints
    )
    root_pose = (
        _convert_pose_frame(human.root_pose, human.frame, world_frame).resample(
            human.timeline,
            timeline,
            clock=clock,
        )
        if human.root_pose is not None
        else None
    )
    return HumanPoseRecording(
        name=human.name,
        timeline=timeline,
        frame=world_frame,
        joints=joints,
        root_pose=root_pose,
        source_height_m=human.source_height_m,
        provenance=dict(human.provenance or {}),
    )


def resample_mocap_body(
    track: RigidBodyTrack,
    mocap: MocapRecording,
    timeline: SampleTimeline,
    clock: ClockTransform,
    *,
    world_frame: FrameConvention,
) -> RigidBodyTrack:
    """Convert and resample one native mocap rigid body."""

    converted = _convert_pose_frame(track, mocap.frame, world_frame)
    rigid_body = RigidBodyTrack(
        role=converted.role,
        positions=converted.positions,
        quaternions=converted.quaternions,
        quaternion_order=converted.quaternion_order,
        validity=converted.validity,
        provenance=converted.provenance,
    )
    return rigid_body.resample(mocap.timeline, timeline, clock=clock)


def register_actor_to_shoes(
    actor: HumanPoseRecording,
    left_shoe: PoseTrack,
    right_shoe: PoseTrack,
    contacts: SemanticContactSequence,
) -> tuple[HumanPoseRecording, AlignmentReport]:
    """Rigidly register estimated actor joints to observed shoe positions."""

    left_foot = actor.joint(SkateboardingMotionJoint.LEFT_FOOT)
    right_foot = actor.joint(SkateboardingMotionJoint.RIGHT_FOOT)
    active = np.asarray(
        [state != SkateboardingContactState.AIR for state in contacts.tracks[0].states],
        dtype=bool,
    )
    active |= np.asarray(
        [state != SkateboardingContactState.AIR for state in contacts.tracks[1].states],
        dtype=bool,
    )
    finite = (
        active
        & np.asarray(left_foot.validity, dtype=bool)
        & np.asarray(right_foot.validity, dtype=bool)
        & np.asarray(left_shoe.validity, dtype=bool)
        & np.asarray(right_shoe.validity, dtype=bool)
    )
    if np.count_nonzero(finite) < 2:
        finite = (
            np.asarray(left_foot.validity, dtype=bool)
            & np.asarray(right_foot.validity, dtype=bool)
            & np.asarray(left_shoe.validity, dtype=bool)
            & np.asarray(right_shoe.validity, dtype=bool)
        )
    source = np.stack(
        [left_foot.values[finite], right_foot.values[finite]],
        axis=1,
    ).reshape(-1, 3)
    target = np.stack(
        [left_shoe.positions[finite], right_shoe.positions[finite]],
        axis=1,
    ).reshape(-1, 3)
    transform, report = register_rigid_points(source, target)
    joints = tuple(
        JointTrack(
            role=track.role,
            values=transform.apply(track.values),
            validity=track.validity,
            provenance={
                **dict(track.provenance or {}),
                "spatially_registered": True,
            },
        )
        for track in actor.joints
    )
    root_pose = (
        _apply_rigid_transform(actor.root_pose, transform.rotation, transform.translation)
        if actor.root_pose is not None
        else None
    )
    return (
        HumanPoseRecording(
            name=actor.name,
            timeline=actor.timeline,
            frame=actor.frame,
            joints=joints,
            root_pose=root_pose,
            source_height_m=actor.source_height_m,
            provenance=dict(actor.provenance or {}),
        ),
        report,
    )


def _mocap_speed_signal(
    mocap: MocapRecording,
) -> tuple[SampleTimeline, np.ndarray]:
    left = mocap.rigid_body(SkateboardingRigidBody.LEFT_SHOE).positions
    right = mocap.rigid_body(SkateboardingRigidBody.RIGHT_SHOE).positions
    return _speed_signal(mocap.timeline, left, right)


def _human_speed_signal(
    human: HumanPoseRecording,
) -> tuple[SampleTimeline, np.ndarray]:
    left = np.mean(
        np.stack([human.joint(role).values for role in LEFT_FOOT_CUE_JOINTS]),
        axis=0,
    )
    right = np.mean(
        np.stack([human.joint(role).values for role in RIGHT_FOOT_CUE_JOINTS]),
        axis=0,
    )
    return _speed_signal(human.timeline, left, right)


def _speed_signal(
    timeline: SampleTimeline,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[SampleTimeline, np.ndarray]:
    deltas = np.diff(timeline.timestamps)
    if np.any(deltas <= 0.0):
        raise ValueError("speed signal requires increasing timestamps")
    speed = np.column_stack(
        [
            np.linalg.norm(np.diff(left, axis=0), axis=1) / deltas,
            np.linalg.norm(np.diff(right, axis=0), axis=1) / deltas,
        ]
    )
    times = 0.5 * (timeline.timestamps[1:] + timeline.timestamps[:-1])
    return SampleTimeline(times, clock=timeline.clock), speed


def _convert_pose_frame(
    track: PoseTrack,
    source_frame: FrameConvention,
    target_frame: FrameConvention,
) -> PoseTrack:
    if source_frame == target_frame:
        return track
    frame_rotation = frame_transform_matrix(source_frame, target_frame)
    xyzw = reorder_quaternions(
        track.quaternions,
        track.quaternion_order,
        QuaternionOrder.XYZW,
    )
    matrices = Rotation.from_quat(xyzw).as_matrix()
    converted_matrices = frame_rotation @ matrices @ frame_rotation.T
    quaternions = reorder_quaternions(
        Rotation.from_matrix(converted_matrices).as_quat(),
        QuaternionOrder.XYZW,
        track.quaternion_order,
    )
    return PoseTrack(
        role=track.role,
        positions=convert_points_frame(
            track.positions,
            source_frame,
            target_frame,
        ),
        quaternions=quaternions,
        quaternion_order=track.quaternion_order,
        validity=track.validity,
        provenance=track.provenance,
    )


def _apply_rigid_transform(
    track: PoseTrack,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> PoseTrack:
    xyzw = reorder_quaternions(
        track.quaternions,
        track.quaternion_order,
        QuaternionOrder.XYZW,
    )
    orientations = rotation @ Rotation.from_quat(xyzw).as_matrix()
    return PoseTrack(
        role=track.role,
        positions=track.positions @ rotation.T + translation,
        quaternions=reorder_quaternions(
            Rotation.from_matrix(orientations).as_quat(),
            QuaternionOrder.XYZW,
            track.quaternion_order,
        ),
        quaternion_order=track.quaternion_order,
        validity=track.validity,
        provenance={
            **dict(track.provenance or {}),
            "spatially_registered": True,
        },
    )
