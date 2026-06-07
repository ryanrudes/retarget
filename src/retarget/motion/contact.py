"""Robot-resolved contact plans that preserve semantic enum vocabularies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, cast

import numpy as np
from typing_extensions import TypeVar

from retarget.core.enums import ContactPatch, ContactState, ContactSubject, RobotLink
from retarget.motion.support import SupportPlane

SubjectT = TypeVar("SubjectT", bound=ContactSubject, default=ContactSubject)
StateT = TypeVar("StateT", bound=ContactState, default=ContactState)
PatchT = TypeVar("PatchT", bound=ContactPatch, default=ContactPatch)
LinkT = TypeVar("LinkT", bound=RobotLink, default=RobotLink)


@dataclass(frozen=True)
class ContactTrack(Generic[SubjectT, StateT, PatchT, LinkT]):
    """One semantic subject's typed contact state over time."""

    subject: SubjectT
    states: tuple[StateT, ...]
    links: tuple[LinkT, ...]
    patch: PatchT | None = None
    active_states: tuple[StateT, ...] = ()
    support_states: tuple[StateT, ...] = ()
    validity: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ContactSubject):
            raise TypeError("contact subject must be a ContactSubject member")
        states = tuple(self.states)
        links = tuple(self.links)
        if not states:
            raise ValueError("contact track requires at least one state")
        if not links:
            raise ValueError("contact track requires at least one robot link")
        state_type = type(states[0])
        link_type = type(links[0])
        if not issubclass(state_type, ContactState) or not all(type(state) is state_type for state in states):
            raise TypeError("contact states must use one ContactState vocabulary")
        if not issubclass(link_type, RobotLink) or not all(type(link) is link_type for link in links):
            raise TypeError("contact links must use one RobotLink vocabulary")
        if self.patch is not None and not isinstance(self.patch, ContactPatch):
            raise TypeError("contact patch must be a ContactPatch member")
        if not all(type(state) is state_type for state in (*self.active_states, *self.support_states)):
            raise TypeError("active and support states must use the track state vocabulary")
        validity = np.ones(len(states), dtype=bool) if self.validity is None else np.asarray(self.validity, dtype=bool)
        if validity.shape != (len(states),):
            raise ValueError("contact validity must have shape (frames,)")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "active_states", tuple(self.active_states))
        object.__setattr__(self, "support_states", tuple(self.support_states))
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return len(self.states)

    @property
    def active_mask(self) -> np.ndarray:
        """Valid frames whose state is active."""

        return np.asarray(self.validity, dtype=bool) & np.isin(self.states, self.active_states)

    @property
    def support_mask(self) -> np.ndarray:
        """Valid frames whose state touches support geometry."""

        return np.asarray(self.validity, dtype=bool) & np.isin(self.states, self.support_states)

    def active_at(self, frame_idx: int) -> bool:
        """Whether this subject is active at one frame."""

        return bool(self.active_mask[frame_idx])

    def support_at(self, frame_idx: int) -> bool:
        """Whether this subject touches support geometry at one frame."""

        return bool(self.support_mask[frame_idx])

    def resampled_indices(self, indices: np.ndarray) -> ContactTrack[SubjectT, StateT, PatchT, LinkT]:
        """Return a copy sampled at nearest-neighbor indices."""

        return ContactTrack(
            subject=self.subject,
            states=tuple(self.states[int(index)] for index in indices),
            links=self.links,
            patch=self.patch,
            active_states=self.active_states,
            support_states=self.support_states,
            validity=np.asarray(self.validity, dtype=bool)[indices],
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True)
class ContactFrame(Generic[SubjectT, StateT, PatchT, LinkT]):
    """Per-frame view of a :class:`ContactPlan`."""

    frame_idx: int
    tracks: tuple[ContactTrack[SubjectT, StateT, PatchT, LinkT], ...]
    support: SupportPlane | None = None

    @property
    def active_tracks(self) -> tuple[ContactTrack[SubjectT, StateT, PatchT, LinkT], ...]:
        """Tracks active at this frame."""

        return tuple(track for track in self.tracks if track.active_at(self.frame_idx))

    @property
    def support_tracks(self) -> tuple[ContactTrack[SubjectT, StateT, PatchT, LinkT], ...]:
        """Tracks touching support at this frame."""

        return tuple(track for track in self.tracks if track.support_at(self.frame_idx))

    @property
    def active_links(self) -> tuple[LinkT, ...]:
        """Typed robot links mapped to active tracks."""

        return tuple(dict.fromkeys(link for track in self.active_tracks for link in track.links))

    @property
    def support_links(self) -> tuple[LinkT, ...]:
        """Typed robot links mapped to support tracks."""

        return tuple(dict.fromkeys(link for track in self.support_tracks for link in track.links))


@dataclass(frozen=True)
class ContactPlan(Generic[SubjectT, StateT, PatchT, LinkT]):
    """Typed contact states and support geometry for a retargeting problem."""

    tracks: tuple[ContactTrack[SubjectT, StateT, PatchT, LinkT], ...]
    frame_count: int | None = None
    support: SupportPlane | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not tracks:
            raise ValueError("ContactPlan requires at least one track")
        frame_count = self.frame_count if self.frame_count is not None else tracks[0].frame_count
        if frame_count <= 0 or any(track.frame_count != frame_count for track in tracks):
            raise ValueError("all contact tracks must have the positive frame_count")
        subject_type = type(tracks[0].subject)
        state_type = type(tracks[0].states[0])
        patch_types = {type(track.patch) for track in tracks if track.patch is not None}
        if not all(type(track.subject) is subject_type for track in tracks):
            raise TypeError("all contact tracks must use one contact-subject vocabulary")
        if not all(all(type(state) is state_type for state in track.states) for track in tracks):
            raise TypeError("all contact tracks must use one contact-state vocabulary")
        if len(patch_types) > 1:
            raise TypeError("all contact tracks must use one contact-patch vocabulary")
        if len({(type(track.subject), track.subject.value) for track in tracks}) != len(tracks):
            raise ValueError("contact track subjects must be unique")
        link_type = type(tracks[0].links[0])
        if not all(all(type(link) is link_type for link in track.links) for track in tracks):
            raise TypeError("all contact tracks must use one robot-link vocabulary")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "frame_count", int(frame_count))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def frame(self, frame_idx: int) -> ContactFrame[SubjectT, StateT, PatchT, LinkT]:
        """Return a per-frame view."""

        frame_count = cast(int, self.frame_count)
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(frame_idx)
        return ContactFrame(frame_idx=frame_idx, tracks=self.tracks, support=self.support)

    def resampled(self, source_fps: float, target_fps: float) -> ContactPlan[SubjectT, StateT, PatchT, LinkT]:
        """Nearest-neighbor resample contact states."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        from retarget.core.timing import resampling_times

        source_times, target_times = resampling_times(cast(int, self.frame_count), source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        use_previous = np.abs(target_times - source_times[previous]) <= np.abs(source_times[indices] - target_times)
        indices[use_previous] = previous[use_previous]
        return ContactPlan(
            tracks=tuple(track.resampled_indices(indices) for track in self.tracks),
            frame_count=len(indices),
            support=self.support,
            provenance={**self.provenance, "resampled_from_fps": source_fps, "resampled_to_fps": target_fps},
        )

    def scaled(self, factor: float) -> ContactPlan[SubjectT, StateT, PatchT, LinkT]:
        """Return a copy with positional support geometry scaled."""

        return ContactPlan(
            tracks=self.tracks,
            frame_count=cast(int, self.frame_count),
            support=self.support.scaled(factor) if self.support is not None else None,
            provenance={**self.provenance, "scale_factor": float(factor)},
        )
