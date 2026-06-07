"""Typed support-contact classification for observed point tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.capture.timeline import SampleTimeline
from retarget.capture.tracks import PointTrack
from retarget.core.enums import ContactPatch, ContactState, ContactSubject
from retarget.motion.support import SupportPlane
from retarget.observation.contact import SemanticContactSequence, SemanticContactTrack


@dataclass(frozen=True)
class FootSupportClassificationConfig:
    """Thresholds for ground and moving-object foot support."""

    ground_height_percentile: float = 8.0
    ground_clearance_m: float = 0.035
    ground_speed_mps: float = 0.18
    object_horizontal_distance_m: float = 0.35
    object_height_min_m: float = 0.0
    object_height_max_m: float = 0.14
    object_relative_speed_mps: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.ground_height_percentile <= 100.0:
            raise ValueError("ground_height_percentile must be in [0, 100]")
        values = (
            self.ground_clearance_m,
            self.ground_speed_mps,
            self.object_horizontal_distance_m,
            self.object_relative_speed_mps,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("support thresholds must be positive")
        if self.object_height_max_m < self.object_height_min_m:
            raise ValueError("object support height bounds are reversed")


@dataclass(frozen=True)
class FootSupportStates:
    """Typed categorical values emitted by support classification."""

    air: ContactState
    ground: ContactState
    observed_object: ContactState


def classify_foot_support(
    *,
    timeline: SampleTimeline,
    left_foot: PointTrack,
    right_foot: PointTrack,
    observed_object: PointTrack,
    left_subject: ContactSubject,
    right_subject: ContactSubject,
    left_patch: ContactPatch,
    right_patch: ContactPatch,
    states: FootSupportStates,
    config: FootSupportClassificationConfig | None = None,
) -> SemanticContactSequence:
    """Classify feet as air, ground, or supported by an observed object."""

    policy = config or FootSupportClassificationConfig()
    tracks = (left_foot, right_foot, observed_object)
    if any(track.sample_count != timeline.sample_count for track in tracks):
        raise ValueError("support-classification tracks must match the timeline")
    object_velocity = _velocity(observed_object.values, timeline.timestamps)
    finite_feet = np.concatenate(
        [
            left_foot.values[np.asarray(left_foot.validity, dtype=bool), 2],
            right_foot.values[np.asarray(right_foot.validity, dtype=bool), 2],
        ]
    )
    if finite_feet.size == 0:
        raise ValueError("support classification requires finite foot samples")
    floor_height = float(np.percentile(finite_feet, policy.ground_height_percentile))
    support = SupportPlane(
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        origin=np.asarray([0.0, 0.0, floor_height], dtype=np.float64),
    )

    semantic_tracks: list[SemanticContactTrack] = []
    for foot, subject, patch in (
        (left_foot, left_subject, left_patch),
        (right_foot, right_subject, right_patch),
    ):
        foot_velocity = _velocity(foot.values, timeline.timestamps)
        relative = foot.values - observed_object.values
        horizontal_distance = np.linalg.norm(relative[:, :2], axis=1)
        relative_height = relative[:, 2]
        relative_speed = np.linalg.norm(foot_velocity - object_velocity, axis=1)
        object_contact = (
            (horizontal_distance <= policy.object_horizontal_distance_m)
            & (relative_height >= policy.object_height_min_m)
            & (relative_height <= policy.object_height_max_m)
            & (relative_speed <= policy.object_relative_speed_mps)
        )
        ground_contact = (
            (np.abs(foot.values[:, 2] - floor_height) <= policy.ground_clearance_m)
            & (np.linalg.norm(foot_velocity, axis=1) <= policy.ground_speed_mps)
            & ~object_contact
        )
        values = tuple(
            states.observed_object
            if object_contact[index]
            else states.ground
            if ground_contact[index]
            else states.air
            for index in range(timeline.sample_count)
        )
        validity = (
            np.asarray(foot.validity, dtype=bool)
            & np.asarray(observed_object.validity, dtype=bool)
        )
        semantic_tracks.append(
            SemanticContactTrack(
                subject=subject,
                patch=patch,
                states=values,
                active_states=(states.ground, states.observed_object),
                support_states=(states.ground,),
                validity=validity,
                provenance={"classifier": "foot_support"},
            )
        )
    return SemanticContactSequence(
        timeline=timeline,
        tracks=tuple(semantic_tracks),
        support=support,
        provenance={"source": "foot_support", "config": policy.__dict__},
    )


def _velocity(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 1:
        return np.zeros_like(values)
    return np.asarray(np.gradient(values, timestamps, axis=0), dtype=np.float64)
