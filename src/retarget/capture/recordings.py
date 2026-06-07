"""Typed native recording models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retarget.capture.timeline import SampleTimeline
from retarget.capture.tracks import JointTrack, MarkerTrack, PointTrack, PoseTrack, RigidBodyTrack
from retarget.core.array import FloatArray
from retarget.core.enums import (
    FrameConvention,
    MocapMarker,
    MocapRigidBody,
    MotionJoint,
    NameEnum,
)


@dataclass(frozen=True)
class VideoRecording:
    """A video stream and its native frame timeline."""

    name: str
    path: Path
    timeline: SampleTimeline
    frame_size: tuple[int, int] | None = None
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("video recording name must not be empty")
        if self.frame_size is not None and (
            len(self.frame_size) != 2 or any(dimension <= 0 for dimension in self.frame_size)
        ):
            raise ValueError("video frame_size must contain positive (width, height)")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))


@dataclass(frozen=True)
class MocapRecording:
    """Joints, rigid bodies, and markers sampled in one native mocap clock."""

    name: str
    timeline: SampleTimeline
    frame: FrameConvention
    joints: tuple[JointTrack, ...] = ()
    rigid_bodies: tuple[RigidBodyTrack, ...] = ()
    markers: tuple[MarkerTrack, ...] = ()
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mocap recording name must not be empty")
        if not all(isinstance(track.role, MotionJoint) for track in self.joints):
            raise TypeError("mocap joint roles must be MotionJoint members")
        if not all(isinstance(track.role, MocapRigidBody) for track in self.rigid_bodies):
            raise TypeError("mocap rigid-body roles must be MocapRigidBody members")
        if not all(isinstance(track.role, MocapMarker) for track in self.markers):
            raise TypeError("mocap marker roles must be MocapMarker members")
        _require_one_vocabulary(self.joints, label="mocap joint")
        _require_one_vocabulary(self.rigid_bodies, label="mocap rigid-body")
        _require_one_vocabulary(self.markers, label="mocap marker")
        for joint_track in self.joints:
            if joint_track.sample_count != self.timeline.sample_count:
                raise ValueError("mocap track lengths must match the native timeline")
        for body_track in self.rigid_bodies:
            if body_track.sample_count != self.timeline.sample_count:
                raise ValueError("mocap track lengths must match the native timeline")
        for marker_track in self.markers:
            if marker_track.sample_count != self.timeline.sample_count:
                raise ValueError("mocap track lengths must match the native timeline")
        rigid_body_roles = tuple(track.role for track in self.rigid_bodies)
        marker_roles = tuple(track.role for track in self.markers)
        joint_roles = tuple(track.role for track in self.joints)
        if len(set(joint_roles)) != len(joint_roles):
            raise ValueError("joint roles must be unique")
        if len(set(rigid_body_roles)) != len(rigid_body_roles):
            raise ValueError("rigid-body roles must be unique")
        if len(set(marker_roles)) != len(marker_roles):
            raise ValueError("marker roles must be unique")
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    def rigid_body(self, role: NameEnum) -> RigidBodyTrack:
        """Return one rigid-body track."""

        for track in self.rigid_bodies:
            if track.role == role:
                return track
        raise KeyError(f"Unknown rigid body role {role.value!r}")

    def joint(self, role: NameEnum) -> JointTrack:
        """Return one mocap joint track."""

        for track in self.joints:
            if track.role == role:
                return track
        raise KeyError(f"Unknown mocap joint role {role.value!r}")

    def marker(self, role: NameEnum) -> MarkerTrack:
        """Return one marker track."""

        for track in self.markers:
            if track.role == role:
                return track
        raise KeyError(f"Unknown marker role {role.value!r}")


@dataclass(frozen=True)
class HumanPoseRecording:
    """Estimated or measured human pose in one native clock."""

    name: str
    timeline: SampleTimeline
    frame: FrameConvention
    joints: tuple[JointTrack, ...]
    root_pose: PoseTrack | None = None
    source_height_m: float | None = None
    vertices: FloatArray | None = None
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("human-pose recording name must not be empty")
        if not self.joints:
            raise ValueError("human-pose recording requires joint tracks")
        if not all(isinstance(track.role, MotionJoint) for track in self.joints):
            raise TypeError("human-pose joint roles must be MotionJoint members")
        _require_one_vocabulary(self.joints, label="human-pose joint")
        if self.root_pose is not None and not isinstance(
            self.root_pose.role,
            MotionJoint,
        ):
            raise TypeError("human-pose root role must be a MotionJoint member")
        for track in self.joints:
            if track.sample_count != self.timeline.sample_count:
                raise ValueError("human-pose track lengths must match the native timeline")
        if self.root_pose is not None and self.root_pose.sample_count != self.timeline.sample_count:
            raise ValueError("human-pose root length must match the native timeline")
        joint_roles = tuple(track.role for track in self.joints)
        if len(set(joint_roles)) != len(joint_roles):
            raise ValueError("joint roles must be unique")
        if self.source_height_m is not None and self.source_height_m <= 0.0:
            raise ValueError("source_height_m must be positive")
        if self.vertices is not None:
            vertices = np.asarray(self.vertices, dtype=np.float64)
            if vertices.ndim != 3 or vertices.shape[0] != self.timeline.sample_count or vertices.shape[2] != 3:
                raise ValueError("human-pose vertices must have shape (samples, vertices, 3)")
            object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def joint_roles(self) -> tuple[NameEnum, ...]:
        """Ordered joint vocabulary."""

        return tuple(track.role for track in self.joints)

    def joint(self, role: NameEnum) -> JointTrack:
        """Return one joint track."""

        for track in self.joints:
            if track.role == role:
                return track
        raise KeyError(f"Unknown joint role {role.value!r}")


def _require_one_vocabulary(
    tracks: Sequence[PointTrack | PoseTrack],
    *,
    label: str,
) -> None:
    if not tracks:
        return
    roles = tuple(track.role for track in tracks)
    vocabulary = type(roles[0])
    if not all(type(role) is vocabulary for role in roles):
        raise TypeError(f"{label} tracks must use one enum vocabulary")
