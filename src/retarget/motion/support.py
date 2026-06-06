"""Support geometry used by contact-aware retargeting."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from retarget.core.array import FloatArray
from retarget.core.enums import FrameConvention
from retarget.core.pose import frame_transform_matrix


class SupportPlane:
    """Planar support surface used by contact-aware terms."""

    normal: FloatArray
    origin: FloatArray
    up_axis: int

    def __init__(self, normal: ArrayLike, origin: ArrayLike, up_axis: int = 2) -> None:
        normal_array = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal_array))
        if norm <= 1e-12:
            raise ValueError("support plane normal must be non-zero")
        origin_array = np.asarray(origin, dtype=np.float64).reshape(3)
        if up_axis not in (0, 1, 2):
            raise ValueError("up_axis must be 0, 1, or 2")
        self.normal = normal_array / norm
        self.origin = origin_array
        self.up_axis = int(up_axis)

    def clearance(self, points: ArrayLike) -> FloatArray:
        """Signed clearance from the support plane."""

        arr = np.asarray(points, dtype=np.float64)
        return np.asarray((arr.reshape(-1, 3) - self.origin) @ self.normal, dtype=np.float64).reshape(
            arr.shape[:-1]
        )

    def height_at(self, points: ArrayLike) -> FloatArray:
        """Support height at the horizontal coordinates of ``points``."""

        arr = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        normal_up = float(self.normal[self.up_axis])
        if abs(normal_up) <= 1e-12:
            raise ValueError("support plane normal is parallel to the vertical axis")
        horizontal_axes = [axis for axis in range(3) if axis != self.up_axis]
        horizontal_delta = arr[:, horizontal_axes] - self.origin[horizontal_axes]
        horizontal_normal = self.normal[horizontal_axes]
        heights = self.origin[self.up_axis] - (horizontal_delta @ horizontal_normal) / normal_up
        return np.asarray(heights, dtype=np.float64).reshape(np.asarray(points).shape[:-1])

    def scaled(self, factor: float) -> SupportPlane:
        """Return a copy with positional quantities scaled."""

        return SupportPlane(normal=self.normal, origin=self.origin * float(factor), up_axis=self.up_axis)

    def to_frame(self, source: FrameConvention, target: FrameConvention) -> SupportPlane:
        """Return this support plane represented in another frame convention."""

        transform = frame_transform_matrix(source, target)
        up_axis = int(np.argmax(np.abs(transform[:, self.up_axis])))
        return SupportPlane(normal=transform @ self.normal, origin=transform @ self.origin, up_axis=up_axis)
