"""Target-independent semantic contact observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from retarget.capture.timeline import SampleTimeline
from retarget.core.enums import ContactPatch, ContactState, ContactSubject, RobotLink
from retarget.motion.contact import ContactPlan, ContactTrack
from retarget.motion.support import SupportPlane


@dataclass(frozen=True)
class SemanticContactTrack:
    """Categorical contact state for one observed subject and patch."""

    subject: ContactSubject
    states: tuple[ContactState, ...]
    patch: ContactPatch | None = None
    active_states: tuple[ContactState, ...] = ()
    support_states: tuple[ContactState, ...] = ()
    validity: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, ContactSubject):
            raise TypeError("semantic contact subject must be a ContactSubject member")
        if self.patch is not None and not isinstance(self.patch, ContactPatch):
            raise TypeError("semantic contact patch must be a ContactPatch member")
        states = tuple(self.states)
        if not states:
            raise ValueError("semantic contact track requires at least one state")
        if not all(isinstance(state, ContactState) for state in states):
            raise TypeError("semantic contact states must be ContactState members")
        state_vocabulary = type(states[0])
        if not all(type(state) is state_vocabulary for state in states):
            raise TypeError("semantic contact states must use one ContactState vocabulary")
        active_states = tuple(self.active_states)
        support_states = tuple(self.support_states)
        if not all(isinstance(state, ContactState) for state in (*active_states, *support_states)):
            raise TypeError("active and support states must be ContactState members")
        if not all(type(state) is state_vocabulary for state in (*active_states, *support_states)):
            raise TypeError("active and support states must use the track state vocabulary")
        validity = np.ones(len(states), dtype=bool) if self.validity is None else np.asarray(self.validity, dtype=bool)
        if validity.shape != (len(states),):
            raise ValueError("semantic contact validity must have shape (samples,)")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "active_states", active_states)
        object.__setattr__(self, "support_states", support_states)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def sample_count(self) -> int:
        """Number of samples."""

        return len(self.states)


@dataclass(frozen=True)
class SemanticContactSequence:
    """Robot-independent semantic contacts and support geometry."""

    timeline: SampleTimeline
    tracks: tuple[SemanticContactTrack, ...]
    support: SupportPlane | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tracks:
            raise ValueError("semantic contact sequence requires at least one track")
        keys = tuple((track.subject, track.patch) for track in self.tracks)
        if len(set(keys)) != len(keys):
            raise ValueError("semantic contact subject/patch pairs must be unique")
        if any(track.sample_count != self.timeline.sample_count for track in self.tracks):
            raise ValueError("semantic contact tracks must match the observation timeline")
        subject_vocabulary = type(self.tracks[0].subject)
        if not all(type(track.subject) is subject_vocabulary for track in self.tracks):
            raise TypeError("semantic contact subjects must use one ContactSubject vocabulary")
        state_vocabulary = type(self.tracks[0].states[0])
        if not all(type(track.states[0]) is state_vocabulary for track in self.tracks):
            raise TypeError("semantic contact tracks must use one ContactState vocabulary")
        patches = tuple(track.patch for track in self.tracks if track.patch is not None)
        if patches and not all(type(patch) is type(patches[0]) for patch in patches):
            raise TypeError("semantic contact patches must use one ContactPatch vocabulary")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def resolve(
        self,
        link_mapping: Mapping[ContactSubject, Sequence[RobotLink]],
    ) -> ContactPlan:
        """Resolve semantic subjects to concrete robot links."""

        missing = {track.subject for track in self.tracks} - set(link_mapping)
        if missing:
            values = sorted(subject.value for subject in missing)
            raise ValueError(f"contact role mapping is missing subjects: {values}")
        resolved: list[ContactTrack] = []
        for track in self.tracks:
            resolved.append(
                ContactTrack(
                    subject=track.subject,
                    states=track.states,
                    links=tuple(link_mapping[track.subject]),
                    patch=track.patch,
                    active_states=track.active_states,
                    support_states=track.support_states,
                    validity=np.asarray(track.validity, dtype=bool),
                    provenance=dict(track.provenance),
                )
            )
        return ContactPlan(
            tracks=tuple(resolved),
            frame_count=self.timeline.sample_count,
            support=self.support,
            provenance=dict(self.provenance),
        )
