"""Time-grid helpers for deterministic sequence resampling."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from retarget.core.array import FloatArray, as_float_array


def resampling_times(frame_count: int, source_fps: float, target_fps: float) -> tuple[FloatArray, FloatArray]:
    """Return source and target sample times for endpoint-preserving interpolation."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps values must be positive")
    source_times = np.arange(frame_count, dtype=np.float64) / float(source_fps)
    if frame_count == 1:
        return source_times, source_times.copy()
    duration = float(source_times[-1])
    target_count = max(1, int(np.floor(duration * float(target_fps) + 1e-9)) + 1)
    target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    target_times[-1] = min(target_times[-1], duration)
    if target_times[-1] < duration and not np.isclose(target_times[-1], duration):
        target_times = np.append(target_times, duration)
    else:
        target_times[-1] = duration
    return source_times, target_times


def resample_linear(values: Any, source_fps: float, target_fps: float) -> FloatArray:
    """Linearly resample an array whose first axis is time."""

    arr = as_float_array(values, name="values")
    if arr.ndim == 0:
        raise ValueError("values must have a time axis")
    if arr.shape[0] == 0:
        raise ValueError("values must contain at least one frame")
    source_times, target_times = resampling_times(arr.shape[0], source_fps, target_fps)
    if len(source_times) == len(target_times) and np.allclose(source_times, target_times):
        return arr.copy()

    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(target_times, source_times, flat[:, dim])
    return cast(FloatArray, out.reshape((len(target_times), *arr.shape[1:])))
