"""Shared kinematics data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GeometryDistance:
    """Signed or unsigned distance between two geometry-like objects.

    Negative distances indicate overlap when the backend can compute signed
    separation. Backends that only expose point distances should return
    non-negative values.

    Attributes:
        first (str): Name of the first body or collision geometry.
        second (str): Name of the second body or collision geometry.
        distance (float): Separation along ``normal_from_first_to_second`` (negative if overlapping).
        point_on_first (NDArray[np.float64]): Closest point on ``first``, shape ``(3,)``.
        point_on_second (NDArray[np.float64]): Closest point on ``second``, shape ``(3,)``.
        normal_from_first_to_second (NDArray[np.float64]): Unit normal pointing from ``first`` to ``second``.
    """

    first: str
    second: str
    distance: float
    point_on_first: NDArray[np.float64]
    point_on_second: NDArray[np.float64]
    normal_from_first_to_second: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("point_on_first", "point_on_second", "normal_from_first_to_second"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "distance", float(self.distance))


__all__ = ["GeometryDistance"]
