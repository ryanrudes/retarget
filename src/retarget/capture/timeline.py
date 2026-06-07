"""Native and observation timeline primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retarget.core.array import FloatArray


@dataclass(frozen=True)
class SampleTimeline:
    """Strictly increasing sample timestamps in one recording's native clock."""

    timestamps: FloatArray
    clock: str = "native"

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError("timeline timestamps must have shape (samples,)")
        if timestamps.size == 0:
            raise ValueError("timeline must contain at least one sample")
        if not np.isfinite(timestamps).all():
            raise ValueError("timeline timestamps must be finite")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timeline timestamps must be strictly increasing")
        if not self.clock:
            raise ValueError("timeline clock must not be empty")
        object.__setattr__(self, "timestamps", timestamps)

    @classmethod
    def uniform(
        cls,
        sample_count: int,
        fps: float,
        *,
        start_s: float = 0.0,
        clock: str = "native",
    ) -> SampleTimeline:
        """Create a uniform timeline."""

        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        return cls(start_s + np.arange(sample_count, dtype=np.float64) / float(fps), clock=clock)

    @property
    def sample_count(self) -> int:
        """Number of timestamps."""

        return int(self.timestamps.size)

    @property
    def start_s(self) -> float:
        """First timestamp."""

        return float(self.timestamps[0])

    @property
    def end_s(self) -> float:
        """Last timestamp."""

        return float(self.timestamps[-1])

    @property
    def duration_s(self) -> float:
        """Duration between first and last timestamps."""

        return self.end_s - self.start_s

    @property
    def nominal_fps(self) -> float | None:
        """Median sampling rate, or ``None`` for a single sample."""

        if self.sample_count < 2:
            return None
        return float(1.0 / np.median(np.diff(self.timestamps)))

    def slice(self, indices: slice | np.ndarray) -> SampleTimeline:
        """Return a timeline subset."""

        return SampleTimeline(self.timestamps[indices], clock=self.clock)

    def __eq__(self, other: object) -> bool:
        """Compare clock identity and timestamp values."""

        return (
            isinstance(other, SampleTimeline)
            and self.clock == other.clock
            and np.array_equal(self.timestamps, other.timestamps)
        )


@dataclass(frozen=True)
class ClockTransform:
    """Affine mapping from a native clock into an observation clock.

    The mapping is ``observation_time = scale * native_time + offset_s``.
    """

    scale: float = 1.0
    offset_s: float = 0.0
    source_clock: str = "native"
    target_clock: str = "observation"

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("clock scale must be finite and positive")
        if not np.isfinite(self.offset_s):
            raise ValueError("clock offset must be finite")
        if not self.source_clock or not self.target_clock:
            raise ValueError("clock names must not be empty")

    def apply(self, timestamps: Any) -> FloatArray:
        """Map timestamps into the target clock."""

        values = np.asarray(timestamps, dtype=np.float64)
        return np.asarray(self.scale * values + self.offset_s, dtype=np.float64)

    def timeline(self, timeline: SampleTimeline) -> SampleTimeline:
        """Map a complete timeline into the target clock."""

        if timeline.clock != self.source_clock:
            raise ValueError(
                f"clock transform expects {self.source_clock!r}, received {timeline.clock!r}"
            )
        return SampleTimeline(self.apply(timeline.timestamps), clock=self.target_clock)

    def inverse(self) -> ClockTransform:
        """Return the inverse clock mapping."""

        return ClockTransform(
            scale=1.0 / self.scale,
            offset_s=-self.offset_s / self.scale,
            source_clock=self.target_clock,
            target_clock=self.source_clock,
        )
