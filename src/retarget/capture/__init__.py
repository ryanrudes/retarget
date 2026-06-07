"""Typed native recordings and capture processing."""

from retarget.capture.alignment import (
    AlignmentError,
    AlignmentReport,
    RigidTransform,
    TemporalRegistrationConfig,
    estimate_clock_offset,
    register_rigid_points,
)
from retarget.capture.estimators import GvhmrEstimator, HumanPoseEstimator, VideoPoseSource
from retarget.capture.recordings import HumanPoseRecording, MocapRecording, VideoRecording
from retarget.capture.sources import (
    GvhmrOutputSource,
    HumanPoseSourceSchema,
    InMemorySource,
    MocapArraySource,
    ObservationSource,
    ViconBagSource,
    ViconBagTopics,
    ViconRecordingSource,
    ViconSourceSchema,
)
from retarget.capture.timeline import ClockTransform, SampleTimeline
from retarget.capture.tracks import (
    CategoricalTrack,
    JointTrack,
    MarkerTrack,
    PointTrack,
    PoseTrack,
    RigidBodyTrack,
)

__all__ = [
    "AlignmentError",
    "AlignmentReport",
    "CategoricalTrack",
    "ClockTransform",
    "GvhmrEstimator",
    "GvhmrOutputSource",
    "HumanPoseEstimator",
    "HumanPoseRecording",
    "HumanPoseSourceSchema",
    "InMemorySource",
    "JointTrack",
    "MarkerTrack",
    "MocapArraySource",
    "MocapRecording",
    "ObservationSource",
    "PointTrack",
    "PoseTrack",
    "RigidBodyTrack",
    "RigidTransform",
    "SampleTimeline",
    "TemporalRegistrationConfig",
    "ViconBagSource",
    "ViconBagTopics",
    "ViconRecordingSource",
    "ViconSourceSchema",
    "VideoPoseSource",
    "VideoRecording",
    "estimate_clock_offset",
    "register_rigid_points",
]
