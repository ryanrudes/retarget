"""Typed robot-link target tracks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, cast

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import TypeVar

from retarget.core.array import FloatArray
from retarget.core.enums import FrameConvention, RobotLink
from retarget.core.pose import convert_points_frame
from retarget.core.timing import resampling_times

LinkT = TypeVar("LinkT", bound=RobotLink, default=RobotLink)


@dataclass(frozen=True)
class LinkTargetSample(Generic[LinkT]):
    """One active typed robot-link target at one frame."""

    link: LinkT
    position: FloatArray
    weight: float
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkTargetTrack(Generic[LinkT]):
    """World-space target trajectory for one typed robot link."""

    link: LinkT
    positions: FloatArray
    weights: FloatArray | float = 1.0
    active_mask: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.link, RobotLink):
            raise TypeError("target link must be a RobotLink member")
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("target positions must have shape (positive frames, 3)")
        weights = _weights_array(self.weights, frame_count=positions.shape[0])
        active_mask = (
            np.ones(positions.shape[0], dtype=bool)
            if self.active_mask is None
            else np.asarray(self.active_mask, dtype=bool)
        )
        if active_mask.shape != (positions.shape[0],):
            raise ValueError("target active_mask must have shape (frames,)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return int(self.positions.shape[0])

    def active_at(self, frame_idx: int) -> bool:
        """Whether this target is active and finite."""

        weights = cast(FloatArray, self.weights)
        mask = cast(np.ndarray, self.active_mask)
        return bool(
            mask[frame_idx]
            and np.isfinite(self.positions[frame_idx]).all()
            and np.isfinite(weights[frame_idx])
            and weights[frame_idx] > 0.0
        )

    def sample(self, frame_idx: int) -> LinkTargetSample[LinkT] | None:
        """Return the sample for one frame when active."""

        if not self.active_at(frame_idx):
            return None
        weights = cast(FloatArray, self.weights)
        return LinkTargetSample(
            link=self.link,
            position=self.positions[frame_idx],
            weight=float(weights[frame_idx]),
            provenance=dict(self.provenance),
        )

    def resampled_indices(self, indices: np.ndarray) -> LinkTargetTrack[LinkT]:
        """Return a copy sampled at frame indices."""

        return LinkTargetTrack(
            link=self.link,
            positions=self.positions[indices],
            weights=cast(FloatArray, self.weights)[indices],
            active_mask=cast(np.ndarray, self.active_mask)[indices],
            provenance=dict(self.provenance),
        )

    def scaled(self, factor: float) -> LinkTargetTrack[LinkT]:
        """Return a copy with positions scaled."""

        return LinkTargetTrack(
            link=self.link,
            positions=self.positions * float(factor),
            weights=cast(FloatArray, self.weights).copy(),
            active_mask=cast(np.ndarray, self.active_mask).copy(),
            provenance=dict(self.provenance),
        )

    def to_frame(self, source: FrameConvention, target: FrameConvention) -> LinkTargetTrack[LinkT]:
        """Return this track represented in another coordinate frame."""

        return LinkTargetTrack(
            link=self.link,
            positions=convert_points_frame(self.positions, source, target),
            weights=cast(FloatArray, self.weights).copy(),
            active_mask=cast(np.ndarray, self.active_mask).copy(),
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True)
class TargetFrame(Generic[LinkT]):
    """Per-frame view of link targets."""

    frame_idx: int
    tracks: tuple[LinkTargetTrack[LinkT], ...]

    @property
    def samples(self) -> tuple[LinkTargetSample[LinkT], ...]:
        """Active target samples at this frame."""

        return tuple(sample for track in self.tracks if (sample := track.sample(self.frame_idx)) is not None)

    @property
    def links(self) -> tuple[LinkT, ...]:
        """Active typed robot links."""

        return tuple(sample.link for sample in self.samples)

    @property
    def positions(self) -> FloatArray:
        """Active target positions."""

        samples = self.samples
        return (
            np.asarray([sample.position for sample in samples], dtype=np.float64)
            if samples
            else np.zeros((0, 3), dtype=np.float64)
        )

    @property
    def weights(self) -> FloatArray:
        """Active target weights."""

        return np.asarray([sample.weight for sample in self.samples], dtype=np.float64)


@dataclass(frozen=True)
class LinkTargetPlan(Generic[LinkT]):
    """Validated frame-aligned targets using one robot-link vocabulary."""

    tracks: tuple[LinkTargetTrack[LinkT], ...]
    frame_count: int | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not tracks:
            raise ValueError("LinkTargetPlan requires at least one track")
        frame_count = self.frame_count if self.frame_count is not None else tracks[0].frame_count
        if frame_count <= 0 or any(track.frame_count != frame_count for track in tracks):
            raise ValueError("all target tracks must have the positive frame_count")
        link_type = type(tracks[0].link)
        if not all(type(track.link) is link_type for track in tracks):
            raise TypeError("all targets must use one RobotLink vocabulary")
        if len({track.link for track in tracks}) != len(tracks):
            raise ValueError("target links must be unique")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "frame_count", int(frame_count))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def links(self) -> tuple[LinkT, ...]:
        """Robot links targeted by this plan."""

        return tuple(track.link for track in self.tracks)

    def frame(self, frame_idx: int) -> TargetFrame[LinkT]:
        """Return a per-frame target view."""

        frame_count = cast(int, self.frame_count)
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(frame_idx)
        return TargetFrame(frame_idx=frame_idx, tracks=self.tracks)

    def resampled(self, source_fps: float, target_fps: float) -> LinkTargetPlan[LinkT]:
        """Return nearest-neighbor/masked targets on a new time grid."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        source_times, target_times = resampling_times(cast(int, self.frame_count), source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        use_previous = np.abs(target_times - source_times[previous]) <= np.abs(source_times[indices] - target_times)
        indices[use_previous] = previous[use_previous]
        return LinkTargetPlan(
            tracks=tuple(track.resampled_indices(indices) for track in self.tracks),
            frame_count=len(indices),
            provenance={**self.provenance, "resampled_from_fps": source_fps, "resampled_to_fps": target_fps},
        )

    def scaled(self, factor: float) -> LinkTargetPlan[LinkT]:
        """Return a copy with all positions scaled."""

        return LinkTargetPlan(
            tracks=tuple(track.scaled(factor) for track in self.tracks),
            frame_count=cast(int, self.frame_count),
            provenance={**self.provenance, "scale_factor": float(factor)},
        )

    def to_frame(self, source: FrameConvention, target: FrameConvention) -> LinkTargetPlan[LinkT]:
        """Return this target plan represented in another frame."""

        if source == target:
            return self
        return LinkTargetPlan(
            tracks=tuple(track.to_frame(source, target) for track in self.tracks),
            frame_count=cast(int, self.frame_count),
            provenance={**self.provenance, "frame_converted_from": source.value, "frame_converted_to": target.value},
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        links: tuple[LinkT, ...],
        positions: ArrayLike,
        weights: ArrayLike | float = 1.0,
        active_mask: ArrayLike | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> LinkTargetPlan[LinkT]:
        """Build target tracks from dense ``(frames, links, 3)`` arrays."""

        position_array = np.asarray(positions, dtype=np.float64)
        if position_array.ndim != 3 or position_array.shape[2] != 3:
            raise ValueError("target positions must have shape (frames, links, 3)")
        if position_array.shape[1] != len(links):
            raise ValueError("links length must match target positions")
        weight_array = _dense_weights_array(weights, frame_count=position_array.shape[0], link_count=len(links))
        mask_array = _dense_mask_array(active_mask, frame_count=position_array.shape[0], link_count=len(links))
        return cls(
            tracks=tuple(
                LinkTargetTrack(
                    link=link,
                    positions=position_array[:, index, :],
                    weights=weight_array[:, index],
                    active_mask=mask_array[:, index],
                    provenance=dict(provenance or {}),
                )
                for index, link in enumerate(links)
            ),
            frame_count=position_array.shape[0],
            provenance=dict(provenance or {}),
        )


def _weights_array(value: ArrayLike | float, *, frame_count: int) -> FloatArray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape == ():
        return np.full(frame_count, float(weights), dtype=np.float64)
    if weights.shape != (frame_count,):
        raise ValueError("target weights must be scalar or have shape (frames,)")
    return weights


def _dense_weights_array(value: ArrayLike | float, *, frame_count: int, link_count: int) -> FloatArray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape == ():
        return np.full((frame_count, link_count), float(weights), dtype=np.float64)
    if weights.shape == (link_count,):
        return np.tile(weights.reshape(1, link_count), (frame_count, 1))
    if weights.shape == (frame_count, link_count):
        return weights
    raise ValueError("target weights must be scalar, (links,), or (frames, links)")


def _dense_mask_array(value: ArrayLike | None, *, frame_count: int, link_count: int) -> np.ndarray:
    if value is None:
        return np.ones((frame_count, link_count), dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape == (link_count,):
        return np.tile(mask.reshape(1, link_count), (frame_count, 1))
    if mask.shape == (frame_count, link_count):
        return mask
    raise ValueError("target active mask must have shape (links,) or (frames, links)")
