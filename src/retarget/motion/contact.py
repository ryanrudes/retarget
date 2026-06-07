"""Robot-resolved runtime contact plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

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
    provenance: dict[str, object] = field(default_factory=dict)

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
        object.__setattr__(self, "provenance", dict(self.provenance))

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
            provenance=dict(self.provenance),
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
        choose_previous = np.abs(target_times - source_times[previous]) <= np.abs(source_times[indices] - target_times)
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
