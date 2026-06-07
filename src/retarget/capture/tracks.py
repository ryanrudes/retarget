"""Typed recording tracks with type-specific resampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from retarget.capture.timeline import ClockTransform, SampleTimeline
from retarget.core.array import FloatArray
from retarget.core.enums import NameEnum, QuaternionOrder
from retarget.core.pose import reorder_quaternions


def _validity(value: Any, sample_count: int) -> np.ndarray:
    if value is None:
        return np.ones(sample_count, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (sample_count,):
        raise ValueError("validity must have shape (samples,)")
    return mask


def _validate_values(values: Any, *, width: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (samples, {width})")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return array


def _linear_resample(
    values: FloatArray,
    validity: np.ndarray,
    source_times: FloatArray,
    target_times: FloatArray,
) -> tuple[FloatArray, np.ndarray]:
    usable = validity & np.isfinite(values).all(axis=1)
    out = np.full((target_times.size, values.shape[1]), np.nan, dtype=np.float64)
    target_valid = np.zeros(target_times.size, dtype=bool)
    if np.count_nonzero(usable) == 0:
        return out, target_valid
    times = source_times[usable]
    samples = values[usable]
    if times.size == 1:
        exact = np.isclose(target_times, times[0])
        out[exact] = samples[0]
        target_valid[exact] = True
        return out, target_valid
    inside = (target_times >= times[0]) & (target_times <= times[-1])
    for component in range(values.shape[1]):
        out[inside, component] = np.interp(
            target_times[inside],
            times,
            samples[:, component],
        )
    target_valid[inside] = True
    return out, target_valid


@dataclass(frozen=True)
class PointTrack:
    """A named 3D point trajectory."""

    role: NameEnum
    values: FloatArray
    validity: np.ndarray | None = None
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, NameEnum):
            raise TypeError("point-track role must be a NameEnum member")
        values = _validate_values(self.values, width=3, name="point-track values")
        validity = _validity(self.validity, values.shape[0])
        if not np.isfinite(values[validity]).all():
            raise ValueError("valid point-track samples must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def sample_count(self) -> int:
        """Number of samples."""

        return int(self.values.shape[0])

    def resample(
        self,
        source: SampleTimeline,
        target: SampleTimeline,
        *,
        clock: ClockTransform | None = None,
    ) -> Self:
        """Linearly interpolate this track onto ``target``."""

        if source.sample_count != self.sample_count:
            raise ValueError("source timeline length must match point track")
        mapped = clock.timeline(source) if clock is not None else source
        values, validity = _linear_resample(
            self.values,
            np.asarray(self.validity, dtype=bool),
            mapped.timestamps,
            target.timestamps,
        )
        return replace(self, values=values, validity=validity)


@dataclass(frozen=True)
class MarkerTrack(PointTrack):
    """A native motion-capture marker trajectory."""


@dataclass(frozen=True)
class JointTrack(PointTrack):
    """A native or estimated human-joint trajectory."""


@dataclass(frozen=True)
class PoseTrack:
    """A named rigid-body pose trajectory."""

    role: NameEnum
    positions: FloatArray
    quaternions: FloatArray
    quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    validity: np.ndarray | None = None
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, NameEnum):
            raise TypeError("pose-track role must be a NameEnum member")
        positions = _validate_values(self.positions, width=3, name="pose positions")
        quaternions = _validate_values(self.quaternions, width=4, name="pose quaternions")
        if positions.shape[0] != quaternions.shape[0]:
            raise ValueError("pose positions and quaternions must have matching samples")
        validity = _validity(self.validity, positions.shape[0])
        validity &= np.isfinite(positions).all(axis=1) & np.isfinite(quaternions).all(axis=1)
        if np.any(validity):
            norms = np.linalg.norm(quaternions[validity], axis=1)
            if np.any(norms <= 1e-12):
                raise ValueError("valid pose quaternions must be non-zero")
            quaternions = quaternions.copy()
            quaternions[validity] /= norms[:, None]
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "quaternions", quaternions)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def sample_count(self) -> int:
        """Number of samples."""

        return int(self.positions.shape[0])

    def resample(
        self,
        source: SampleTimeline,
        target: SampleTimeline,
        *,
        clock: ClockTransform | None = None,
    ) -> Self:
        """Linearly interpolate translation and SLERP orientation."""

        if source.sample_count != self.sample_count:
            raise ValueError("source timeline length must match pose track")
        mapped = clock.timeline(source) if clock is not None else source
        positions, position_valid = _linear_resample(
            self.positions,
            np.asarray(self.validity, dtype=bool),
            mapped.timestamps,
            target.timestamps,
        )
        usable = np.asarray(self.validity, dtype=bool) & np.isfinite(self.quaternions).all(axis=1)
        quaternions = np.full((target.sample_count, 4), np.nan, dtype=np.float64)
        orientation_valid = np.zeros(target.sample_count, dtype=bool)
        if np.count_nonzero(usable) == 1:
            source_index = int(np.flatnonzero(usable)[0])
            exact = np.isclose(target.timestamps, mapped.timestamps[source_index])
            quaternions[exact] = self.quaternions[source_index]
            orientation_valid[exact] = True
        elif np.count_nonzero(usable) >= 2:
            source_times = mapped.timestamps[usable]
            inside = (target.timestamps >= source_times[0]) & (target.timestamps <= source_times[-1])
            source_xyzw = reorder_quaternions(
                self.quaternions[usable],
                self.quaternion_order,
                QuaternionOrder.XYZW,
            )
            rotations = Slerp(source_times, Rotation.from_quat(source_xyzw))(target.timestamps[inside])
            quaternions[inside] = reorder_quaternions(
                rotations.as_quat(),
                QuaternionOrder.XYZW,
                self.quaternion_order,
            )
            orientation_valid[inside] = True
        return replace(
            self,
            positions=positions,
            quaternions=quaternions,
            validity=position_valid & orientation_valid,
        )


@dataclass(frozen=True)
class RigidBodyTrack(PoseTrack):
    """A native motion-capture rigid-body trajectory."""


@dataclass(frozen=True)
class CategoricalTrack:
    """A named categorical sequence sampled by nearest neighbor."""

    role: NameEnum
    values: tuple[NameEnum, ...]
    validity: np.ndarray | None = None
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, NameEnum):
            raise TypeError("categorical-track role must be a NameEnum member")
        values = tuple(self.values)
        if not values:
            raise ValueError("categorical track must contain at least one sample")
        if not all(isinstance(value, NameEnum) for value in values):
            raise TypeError("categorical values must be NameEnum members")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", _validity(self.validity, len(values)))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def sample_count(self) -> int:
        """Number of samples."""

        return len(self.values)

    def resample(
        self,
        source: SampleTimeline,
        target: SampleTimeline,
        *,
        clock: ClockTransform | None = None,
    ) -> Self:
        """Sample this track at the nearest native timestamp."""

        if source.sample_count != self.sample_count:
            raise ValueError("source timeline length must match categorical track")
        mapped = clock.timeline(source) if clock is not None else source
        indices = np.searchsorted(mapped.timestamps, target.timestamps, side="left")
        indices = np.clip(indices, 0, mapped.sample_count - 1)
        previous = np.maximum(indices - 1, 0)
        use_previous = np.abs(target.timestamps - mapped.timestamps[previous]) <= np.abs(
            mapped.timestamps[indices] - target.timestamps
        )
        indices[use_previous] = previous[use_previous]
        inside = (target.timestamps >= mapped.start_s) & (target.timestamps <= mapped.end_s)
        source_validity = np.asarray(self.validity, dtype=bool)
        validity = inside & source_validity[indices]
        return replace(
            self,
            values=tuple(self.values[index] for index in indices),
            validity=validity,
        )
