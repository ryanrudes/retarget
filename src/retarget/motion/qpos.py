"""Typed qpos trajectory plans used by retargeting objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from retarget.core.array import FloatArray
from retarget.core.timing import resampling_times


@dataclass(frozen=True)
class InitialQposFrame:
    """Per-frame view of an initial qpos seed trajectory."""

    frame_idx: int
    plan: InitialQposPlan

    @property
    def qpos(self) -> FloatArray:
        """Initial qpos values for this frame."""

        return cast(FloatArray, self.plan.qpos[self.frame_idx])


@dataclass(frozen=True)
class InitialQposPlan:
    """Frame-aligned generalized-coordinate seeds used to initialize SQP frames."""

    qpos: FloatArray
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        if qpos.ndim != 2:
            raise ValueError("initial qpos must have shape (frames, qpos_size)")
        if qpos.shape[0] == 0 or qpos.shape[1] == 0:
            raise ValueError("initial qpos must contain at least one frame and one coordinate")
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def frame_count(self) -> int:
        """Number of frames in the plan."""

        return int(self.qpos.shape[0])

    @property
    def qpos_size(self) -> int:
        """Number of generalized coordinates in each frame."""

        return int(self.qpos.shape[1])

    def frame(self, frame_idx: int) -> InitialQposFrame:
        """Return a per-frame initial-qpos view."""

        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(frame_idx)
        return InitialQposFrame(frame_idx=frame_idx, plan=self)

    def resampled(self, source_fps: float, target_fps: float) -> InitialQposPlan:
        """Return nearest-neighbor initial qpos on a new time grid."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        source_times, target_times = resampling_times(self.frame_count, source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        choose_previous = np.abs(target_times - source_times[previous]) <= np.abs(
            source_times[indices] - target_times
        )
        indices[choose_previous] = previous[choose_previous]
        return self.resampled_indices(indices).with_provenance(
            resampled_from_fps=float(source_fps),
            resampled_to_fps=float(target_fps),
        )

    def resampled_indices(self, indices: np.ndarray) -> InitialQposPlan:
        """Return a copy sampled at frame indices."""

        return InitialQposPlan(qpos=self.qpos[indices], provenance=dict(self.provenance))

    def with_provenance(self, **items: object) -> InitialQposPlan:
        """Return a copy with additional JSON-safe provenance."""

        return InitialQposPlan(qpos=self.qpos.copy(), provenance={**self.provenance, **items})

    @classmethod
    def from_array(
        cls,
        qpos: ArrayLike,
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> InitialQposPlan:
        """Build an initial qpos plan from a dense array."""

        return cls(qpos=np.asarray(qpos, dtype=np.float64), provenance=dict(provenance or {}))


@dataclass(frozen=True)
class NominalQposFrame:
    """Per-frame view of a nominal qpos trajectory."""

    frame_idx: int
    plan: NominalQposPlan

    @property
    def qpos(self) -> FloatArray:
        """Nominal qpos values for this frame."""

        return cast(FloatArray, self.plan.qpos[self.frame_idx])

    @property
    def weights(self) -> FloatArray:
        """Per-qpos nominal weights for this frame."""

        return cast(FloatArray, cast(FloatArray, self.plan.weights)[self.frame_idx])

    @property
    def active_mask(self) -> np.ndarray:
        """Per-qpos activity mask for this frame."""

        return cast(np.ndarray, cast(np.ndarray, self.plan.active_mask)[self.frame_idx])

    def active_at(self, qpos_index: int) -> bool:
        """Whether one qpos coordinate has a finite active nominal target."""

        index = int(qpos_index)
        return bool(
            self.active_mask[index]
            and np.isfinite(self.qpos[index])
            and np.isfinite(self.weights[index])
            and self.weights[index] > 0.0
        )


@dataclass(frozen=True)
class NominalQposPlan:
    """Frame-aligned nominal generalized-coordinate trajectory.

    Unlike link targets, qpos targets are robot-configuration values. They are
    resampled in time, but are not spatially scaled with source motion.
    """

    qpos: FloatArray
    weights: FloatArray | float = 1.0
    active_mask: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        if qpos.ndim != 2:
            raise ValueError("nominal qpos must have shape (frames, qpos_size)")
        if qpos.shape[0] == 0 or qpos.shape[1] == 0:
            raise ValueError("nominal qpos must contain at least one frame and one coordinate")
        weights = _dense_weights_array(self.weights, frame_count=qpos.shape[0], qpos_size=qpos.shape[1])
        active_mask = (
            np.ones(qpos.shape, dtype=bool)
            if self.active_mask is None
            else _dense_mask_array(self.active_mask, frame_count=qpos.shape[0], qpos_size=qpos.shape[1])
        )
        if np.any(weights < 0):
            raise ValueError("nominal qpos weights must be non-negative")
        object.__setattr__(self, "qpos", qpos)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def frame_count(self) -> int:
        """Number of frames in the plan."""

        return int(self.qpos.shape[0])

    @property
    def qpos_size(self) -> int:
        """Number of generalized coordinates in each frame."""

        return int(self.qpos.shape[1])

    def frame(self, frame_idx: int) -> NominalQposFrame:
        """Return a per-frame nominal-qpos view."""

        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(frame_idx)
        return NominalQposFrame(frame_idx=frame_idx, plan=self)

    def resampled(self, source_fps: float, target_fps: float) -> NominalQposPlan:
        """Return nearest-neighbor nominal qpos on a new time grid."""

        if abs(float(source_fps) - float(target_fps)) <= 1e-9:
            return self
        source_times, target_times = resampling_times(self.frame_count, source_fps, target_fps)
        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)
        previous = np.maximum(indices - 1, 0)
        choose_previous = np.abs(target_times - source_times[previous]) <= np.abs(
            source_times[indices] - target_times
        )
        indices[choose_previous] = previous[choose_previous]
        return self.resampled_indices(indices).with_provenance(
            resampled_from_fps=float(source_fps),
            resampled_to_fps=float(target_fps),
        )

    def resampled_indices(self, indices: np.ndarray) -> NominalQposPlan:
        """Return a copy sampled at frame indices."""

        weights = cast(FloatArray, self.weights)
        mask = cast(np.ndarray, self.active_mask)
        return NominalQposPlan(
            qpos=self.qpos[indices],
            weights=weights[indices],
            active_mask=mask[indices],
            provenance=dict(self.provenance),
        )

    def with_provenance(self, **items: object) -> NominalQposPlan:
        """Return a copy with additional JSON-safe provenance."""

        return NominalQposPlan(
            qpos=self.qpos.copy(),
            weights=cast(FloatArray, self.weights).copy(),
            active_mask=cast(np.ndarray, self.active_mask).copy(),
            provenance={**self.provenance, **items},
        )

    @classmethod
    def from_array(
        cls,
        qpos: ArrayLike,
        *,
        weights: ArrayLike | float = 1.0,
        active_mask: ArrayLike | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> NominalQposPlan:
        """Build a nominal qpos plan from dense arrays."""

        weight_value: FloatArray | float
        if np.asarray(weights).shape == ():
            weight_value = float(np.asarray(weights, dtype=np.float64))
        else:
            weight_value = np.asarray(weights, dtype=np.float64)
        mask_value = None if active_mask is None else np.asarray(active_mask, dtype=bool)
        return cls(
            qpos=np.asarray(qpos, dtype=np.float64),
            weights=weight_value,
            active_mask=mask_value,
            provenance=dict(provenance or {}),
        )


def _dense_weights_array(value: ArrayLike | float, *, frame_count: int, qpos_size: int) -> FloatArray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape == ():
        return np.full((frame_count, qpos_size), float(weights), dtype=np.float64)
    if weights.shape == (qpos_size,):
        return np.tile(weights.reshape(1, qpos_size), (frame_count, 1))
    if weights.shape == (frame_count, qpos_size):
        return weights
    raise ValueError("nominal qpos weights must be scalar, (qpos_size,), or (frames, qpos_size)")


def _dense_mask_array(value: ArrayLike, *, frame_count: int, qpos_size: int) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape == (qpos_size,):
        return np.tile(mask.reshape(1, qpos_size), (frame_count, 1))
    if mask.shape == (frame_count, qpos_size):
        return mask
    raise ValueError("nominal qpos active mask must have shape (qpos_size,) or (frames, qpos_size)")


__all__ = ["InitialQposFrame", "InitialQposPlan", "NominalQposFrame", "NominalQposPlan"]
