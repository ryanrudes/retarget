"""Small NumPy validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_float_array(value: Any, *, shape_tail: tuple[int, ...] | None = None, name: str = "array") -> FloatArray:
    """Convert a value to a finite `float64` array and optionally validate trailing shape."""

    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if shape_tail is not None and array.shape[-len(shape_tail) :] != shape_tail:
        raise ValueError(f"{name} must have trailing shape {shape_tail}, got {array.shape}")
    return array


def normalize_quaternion(quaternion: Iterable[float], *, name: str = "quaternion") -> FloatArray:
    """Return a normalized quaternion as `float64`."""

    q = as_float_array(quaternion, shape_tail=(4,), name=name).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be zero")
    return q / norm


def ensure_2_tuple(value: Iterable[int] | Iterable[float], *, name: str) -> tuple[float, float]:
    """Validate a two-value numeric tuple."""

    vals = tuple(float(v) for v in value)
    if len(vals) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    if vals[1] < vals[0]:
        raise ValueError(f"{name} upper value must be >= lower value")
    return vals
