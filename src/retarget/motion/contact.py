"""Contact-state inference helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from retarget.motion.spec import MotionFormatSpec, MotionSequence
from retarget.motion.support import SupportPlane


@dataclass(frozen=True)
class ContactTrack:
    """One subject's discrete contact state over time."""

    subject: str
    states: np.ndarray
    link_names: tuple[str, ...] = ()
    active_states: tuple[int, ...] = (1, 2)
    support_states: tuple[int, ...] = (1,)
    labels: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        if states.ndim != 1:
            raise ValueError("contact track states must have shape (frames,)")
        if not self.subject:
            raise ValueError("contact track subject must not be empty")
        object.__setattr__(self, "states", states.astype(np.int16, copy=False))
        object.__setattr__(self, "link_names", tuple(str(name) for name in self.link_names))
        object.__setattr__(self, "active_states", tuple(int(state) for state in self.active_states))
        object.__setattr__(self, "support_states", tuple(int(state) for state in self.support_states))
        object.__setattr__(self, "labels", tuple(str(label) for label in self.labels))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return int(self.states.shape[0])

    @property
    def active_mask(self) -> np.ndarray:
        """Frames where the subject is weight-bearing."""

        return np.isin(self.states, self.active_states)

    @property
    def support_mask(self) -> np.ndarray:
        """Frames where the subject contacts the support plane."""

        return np.isin(self.states, self.support_states)

    def active_at(self, frame_idx: int) -> bool:
        """Whether this subject is active at one frame."""

        return bool(self.active_mask[frame_idx])

    def support_at(self, frame_idx: int) -> bool:
        """Whether this subject contacts the support plane at one frame."""

        return bool(self.support_mask[frame_idx])

    def resampled_indices(self, indices: np.ndarray) -> ContactTrack:
        """Return a copy sampled at nearest-neighbor frame indices."""

        return ContactTrack(
            subject=self.subject,
            states=self.states[indices],
            link_names=self.link_names,
            active_states=self.active_states,
            support_states=self.support_states,
            labels=self.labels,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class ContactFrame:
    """Per-frame view of a :class:`ContactPlan`."""

    frame_idx: int
    tracks: tuple[ContactTrack, ...]
    support: SupportPlane | None = None

    @property
    def active_tracks(self) -> tuple[ContactTrack, ...]:
        """Tracks active at this frame."""

        return tuple(track for track in self.tracks if track.active_at(self.frame_idx))

    @property
    def support_tracks(self) -> tuple[ContactTrack, ...]:
        """Tracks touching the support plane at this frame."""

        return tuple(track for track in self.tracks if track.support_at(self.frame_idx))

    @property
    def active_link_names(self) -> tuple[str, ...]:
        """Robot links mapped to active tracks."""

        return tuple(dict.fromkeys(link for track in self.active_tracks for link in track.link_names))

    @property
    def support_link_names(self) -> tuple[str, ...]:
        """Robot links mapped to tracks touching the support plane."""

        return tuple(dict.fromkeys(link for track in self.support_tracks for link in track.link_names))

    def as_contact_dict(self) -> dict[str, bool]:
        """Return compatibility contact booleans keyed by subject."""

        return {track.subject: track.active_at(self.frame_idx) for track in self.tracks}


@dataclass(frozen=True)
class ContactPlan:
    """Typed contact states and support geometry for a retargeting problem."""

    tracks: tuple[ContactTrack, ...]
    frame_count: int | None = None
    support: SupportPlane | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not tracks:
            raise ValueError("ContactPlan requires at least one track")
        frame_count = self.frame_count if self.frame_count is not None else tracks[0].frame_count
        if frame_count <= 0:
            raise ValueError("ContactPlan frame_count must be positive")
        for track in tracks:
            if track.frame_count != frame_count:
                raise ValueError("all contact tracks must have frame_count frames")
        if len({track.subject for track in tracks}) != len(tracks):
            raise ValueError("contact track subjects must be unique")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "frame_count", int(frame_count))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def frame(self, frame_idx: int) -> ContactFrame:
        """Return a per-frame view."""

        frame_count = cast(int, self.frame_count)
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(frame_idx)
        return ContactFrame(frame_idx=frame_idx, tracks=self.tracks, support=self.support)

    def resampled(self, source_fps: float, target_fps: float) -> ContactPlan:
        """Nearest-neighbor resample contact states."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        from retarget.core.timing import resampling_times

        source_times, target_times = resampling_times(cast(int, self.frame_count), source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        choose_previous = np.abs(target_times - source_times[previous]) <= np.abs(
            source_times[indices] - target_times
        )
        indices[choose_previous] = previous[choose_previous]
        return ContactPlan(
            tracks=tuple(track.resampled_indices(indices) for track in self.tracks),
            frame_count=len(indices),
            support=self.support,
            provenance={**self.provenance, "resampled_from_fps": source_fps, "resampled_to_fps": target_fps},
        )

    def scaled(self, factor: float) -> ContactPlan:
        """Return a copy with positional support geometry scaled."""

        return ContactPlan(
            tracks=self.tracks,
            frame_count=cast(int, self.frame_count),
            support=self.support.scaled(factor) if self.support is not None else None,
            provenance={**self.provenance, "scale_factor": float(factor)},
        )

    @classmethod
    def from_binary_contacts(
        cls,
        contacts: Sequence[Mapping[str, bool]],
        *,
        link_mapping: Mapping[str, Sequence[str] | str] | None = None,
        support: SupportPlane | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> ContactPlan:
        """Build from the legacy per-frame contact dict representation."""

        if not contacts:
            raise ValueError("contacts must not be empty")
        subjects = tuple(dict.fromkeys(name for frame in contacts for name in frame))
        link_mapping = dict(link_mapping or {})
        tracks: list[ContactTrack] = []
        for subject in subjects:
            link_names = _link_names_for_subject(subject, link_mapping.get(subject, ()))
            states = np.asarray([1 if frame.get(subject, False) else 0 for frame in contacts], dtype=np.int16)
            tracks.append(
                ContactTrack(
                    subject=subject,
                    states=states,
                    link_names=link_names,
                    active_states=(1,),
                    support_states=(1,),
                    labels=("air", "contact"),
                )
            )
        return cls(
            tracks=tuple(tracks),
            frame_count=len(contacts),
            support=support,
            provenance=dict(provenance or {}),
        )


def infer_contact_by_velocity(
    motion: MotionSequence,
    motion_format: MotionFormatSpec | None,
    *,
    velocity_threshold: float = 0.01,
) -> tuple[dict[str, bool], ...]:
    """Return per-frame binary contact states.

    Explicit contacts stored on the motion sequence take precedence. Otherwise
    contact states are inferred from contact-joint speed; the first and last
    frames reuse their nearest available finite-difference velocity.

    Args:
        motion (MotionSequence): Input motion with optional explicit contacts.
        motion_format (MotionFormatSpec | None): Format spec supplying ``contact_joints``; may be ``None``.
        velocity_threshold (float): Speed below which a contact joint is treated as in contact (m/s).

    Returns:
        tuple[dict[str, bool], ...]: One mapping per frame from contact-joint name to active flag.

    Raises:
        ValueError: If ``velocity_threshold`` is not positive.
    """

    if motion.contacts:
        return _explicit_contacts(motion, motion_format)
    if motion_format is None or not motion_format.contact_joints:
        return tuple({} for _ in range(motion.frame_count))
    if velocity_threshold <= 0:
        raise ValueError("velocity_threshold must be positive")

    contact_names = tuple(name for name in motion_format.contact_joints if name in motion.joint_names)
    if not contact_names:
        return tuple({} for _ in range(motion.frame_count))

    positions = np.stack([motion.joint(name) for name in contact_names], axis=1)
    if motion.frame_count == 1:
        speeds = np.zeros((1, len(contact_names)), dtype=np.float64)
    else:
        velocities = np.gradient(positions, 1.0 / motion.fps, axis=0)
        speeds = np.linalg.norm(velocities, axis=2)
    return tuple(
        {
            name: bool(speeds[frame_idx, contact_idx] <= velocity_threshold)
            for contact_idx, name in enumerate(contact_names)
        }
        for frame_idx in range(motion.frame_count)
    )


def _explicit_contacts(
    motion: MotionSequence,
    motion_format: MotionFormatSpec | None,
) -> tuple[dict[str, bool], ...]:
    if motion_format is None or not motion_format.contact_joints:
        return tuple(dict(frame) for frame in motion.contacts)
    allowed = set(motion_format.contact_joints)
    return tuple(
        {name: bool(active) for name, active in frame.items() if name in allowed}
        for frame in motion.contacts
    )


def _link_names_for_subject(
    subject: str,
    value: Sequence[str] | str,
) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(name) for name in value)
