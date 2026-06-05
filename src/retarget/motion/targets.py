"""Typed retargeting target tracks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from retarget.core.array import FloatArray
from retarget.core.enums import FrameConvention
from retarget.core.pose import convert_points_frame
from retarget.core.timing import resampling_times


@dataclass(frozen=True)
class LinkTargetSample:
    """One active robot-link target at one frame."""

    link_name: str
    position: FloatArray
    weight: float
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkTargetTrack:
    """World-space target trajectory for one robot link."""

    link_name: str
    positions: FloatArray
    weights: FloatArray | float = 1.0
    active_mask: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("target positions must have shape (frames, 3)")
        if positions.shape[0] == 0:
            raise ValueError("target tracks must contain at least one frame")
        if not self.link_name:
            raise ValueError("target link_name must not be empty")

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
        """Whether this target is active and finite at one frame."""

        weights = cast(FloatArray, self.weights)
        mask = cast(np.ndarray, self.active_mask)
        return bool(
            mask[frame_idx]
            and np.isfinite(self.positions[frame_idx]).all()
            and np.isfinite(weights[frame_idx])
            and weights[frame_idx] > 0.0
        )

    def sample(self, frame_idx: int) -> LinkTargetSample | None:
        """Return this track's sample for one frame, if active."""

        if not self.active_at(frame_idx):
            return None
        weights = cast(FloatArray, self.weights)
        return LinkTargetSample(
            link_name=self.link_name,
            position=self.positions[frame_idx],
            weight=float(weights[frame_idx]),
            provenance=dict(self.provenance),
        )

    def resampled_indices(self, indices: np.ndarray) -> LinkTargetTrack:
        """Return a copy sampled at frame indices."""

        weights = cast(FloatArray, self.weights)
        mask = cast(np.ndarray, self.active_mask)
        return LinkTargetTrack(
            link_name=self.link_name,
            positions=self.positions[indices],
            weights=weights[indices],
            active_mask=mask[indices],
            provenance=dict(self.provenance),
        )

    def scaled(self, factor: float) -> LinkTargetTrack:
        """Return a copy with positions scaled."""

        return LinkTargetTrack(
            link_name=self.link_name,
            positions=self.positions * float(factor),
            weights=cast(FloatArray, self.weights).copy(),
            active_mask=cast(np.ndarray, self.active_mask).copy(),
            provenance=dict(self.provenance),
        )

    def to_frame(self, source: FrameConvention, target: FrameConvention) -> LinkTargetTrack:
        """Return this track represented in another coordinate frame."""

        return LinkTargetTrack(
            link_name=self.link_name,
            positions=convert_points_frame(self.positions, source, target),
            weights=cast(FloatArray, self.weights).copy(),
            active_mask=cast(np.ndarray, self.active_mask).copy(),
            provenance=dict(self.provenance),
        )


@dataclass(frozen=True)
class TargetFrame:
    """Per-frame view of link targets."""

    frame_idx: int
    tracks: tuple[LinkTargetTrack, ...]

    @property
    def samples(self) -> tuple[LinkTargetSample, ...]:
        """Active target samples at this frame."""

        out: list[LinkTargetSample] = []
        for track in self.tracks:
            sample = track.sample(self.frame_idx)
            if sample is not None:
                out.append(sample)
        return tuple(out)

    @property
    def link_names(self) -> tuple[str, ...]:
        """Active robot link names."""

        return tuple(sample.link_name for sample in self.samples)

    @property
    def positions(self) -> FloatArray:
        """Active target positions with shape ``(targets, 3)``."""

        samples = self.samples
        if not samples:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray([sample.position for sample in samples], dtype=np.float64)

    @property
    def weights(self) -> FloatArray:
        """Active target weights with shape ``(targets,)``."""

        return np.asarray([sample.weight for sample in self.samples], dtype=np.float64)


@dataclass(frozen=True)
class LinkTargetPlan:
    """Validated set of frame-aligned robot-link targets."""

    tracks: tuple[LinkTargetTrack, ...]
    frame_count: int | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tracks = tuple(self.tracks)
        if not tracks:
            raise ValueError("LinkTargetPlan requires at least one track")
        frame_count = self.frame_count if self.frame_count is not None else tracks[0].frame_count
        if frame_count <= 0:
            raise ValueError("LinkTargetPlan frame_count must be positive")
        for track in tracks:
            if track.frame_count != frame_count:
                raise ValueError("all target tracks must have frame_count frames")
        if len({track.link_name for track in tracks}) != len(tracks):
            raise ValueError("target link names must be unique")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "frame_count", int(frame_count))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def link_names(self) -> tuple[str, ...]:
        """Robot links targeted by this plan."""

        return tuple(track.link_name for track in self.tracks)

    def frame(self, frame_idx: int) -> TargetFrame:
        """Return a per-frame target view."""

        frame_count = cast(int, self.frame_count)
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(frame_idx)
        return TargetFrame(frame_idx=frame_idx, tracks=self.tracks)

    def resampled(self, source_fps: float, target_fps: float) -> LinkTargetPlan:
        """Return nearest-neighbor/masked targets on a new time grid."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        source_times, target_times = resampling_times(cast(int, self.frame_count), source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        choose_previous = np.abs(target_times - source_times[previous]) <= np.abs(
            source_times[indices] - target_times
        )
        indices[choose_previous] = previous[choose_previous]
        return LinkTargetPlan(
            tracks=tuple(track.resampled_indices(indices) for track in self.tracks),
            frame_count=len(indices),
            provenance={**self.provenance, "resampled_from_fps": source_fps, "resampled_to_fps": target_fps},
        )

    def scaled(self, factor: float) -> LinkTargetPlan:
        """Return a copy with all positions scaled."""

        return LinkTargetPlan(
            tracks=tuple(track.scaled(factor) for track in self.tracks),
            frame_count=cast(int, self.frame_count),
            provenance={**self.provenance, "scale_factor": float(factor)},
        )

    def to_frame(self, source: FrameConvention, target: FrameConvention) -> LinkTargetPlan:
        """Return this target plan represented in another coordinate frame."""

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
        link_names: tuple[str, ...],
        positions: ArrayLike,
        weights: ArrayLike | float = 1.0,
        active_mask: ArrayLike | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> LinkTargetPlan:
        """Build target tracks from dense ``(frames, links, 3)`` arrays."""

        position_array = np.asarray(positions, dtype=np.float64)
        if position_array.ndim != 3 or position_array.shape[2] != 3:
            raise ValueError("target positions must have shape (frames, links, 3)")
        if position_array.shape[1] != len(link_names):
            raise ValueError("link_names length must match target positions")
        weight_array = _dense_weights_array(weights, frame_count=position_array.shape[0], link_count=len(link_names))
        mask_array = _dense_mask_array(active_mask, frame_count=position_array.shape[0], link_count=len(link_names))
        return cls(
            tracks=tuple(
                LinkTargetTrack(
                    link_name=link_name,
                    positions=position_array[:, idx, :],
                    weights=weight_array[:, idx],
                    active_mask=mask_array[:, idx],
                    provenance=dict(provenance or {}),
                )
                for idx, link_name in enumerate(link_names)
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
