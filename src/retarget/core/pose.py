"""Pose primitives with explicit quaternion and frame conventions."""

from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.spatial.transform import Rotation, Slerp

from retarget.core.array import FloatArray, as_float_array, normalize_quaternion
from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.timing import resample_linear, resampling_times

Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def reorder_quaternion(
    quaternion: np.ndarray,
    source: QuaternionOrder,
    target: QuaternionOrder,
) -> np.ndarray:
    """Convert quaternion storage order without changing the represented rotation."""

    q = np.asarray(quaternion, dtype=np.float64)
    if source == target:
        return q.copy()
    if source == QuaternionOrder.WXYZ and target == QuaternionOrder.XYZW:
        return q[[1, 2, 3, 0]]
    if source == QuaternionOrder.XYZW and target == QuaternionOrder.WXYZ:
        return q[[3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion conversion: {source} -> {target}")


def frame_transform_matrix(source: FrameConvention, target: FrameConvention) -> FloatArray:
    """Return the rotation matrix that maps coordinates from `source` to `target`."""

    if source == target:
        return np.eye(3, dtype=np.float64)
    if source == FrameConvention.Y_UP_RIGHT_HANDED and target == FrameConvention.Z_UP_RIGHT_HANDED:
        return Y_UP_TO_Z_UP.copy()
    if source == FrameConvention.Z_UP_RIGHT_HANDED and target == FrameConvention.Y_UP_RIGHT_HANDED:
        return Y_UP_TO_Z_UP.T.copy()
    raise ValueError(f"Unsupported frame conversion: {source} -> {target}")


def convert_points_frame(points: Any, source: FrameConvention, target: FrameConvention) -> FloatArray:
    """Convert point coordinates between supported right-handed frame conventions."""

    arr = as_float_array(points, shape_tail=(3,), name="points")
    transform = frame_transform_matrix(source, target)
    converted = arr.reshape(-1, 3) @ transform.T
    return cast(FloatArray, converted.reshape(arr.shape))


class Pose(BaseModel):
    """Rigid transform from a local frame into a named world convention."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    translation: FloatArray = Field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quaternion: FloatArray = Field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED

    @field_validator("translation", mode="before")
    @classmethod
    def _validate_translation(cls, value: Any) -> FloatArray:
        return as_float_array(value, shape_tail=(3,), name="translation").reshape(3)

    @field_validator("quaternion", mode="before")
    @classmethod
    def _validate_quaternion(cls, value: Any) -> FloatArray:
        return normalize_quaternion(value)

    @classmethod
    def identity(cls, *, frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED) -> Self:
        """Return an identity pose."""

        return cls(frame=frame)

    def quaternion_as(self, order: QuaternionOrder) -> FloatArray:
        """Return the quaternion in the requested storage order."""

        return reorder_quaternion(self.quaternion, self.quaternion_order, order)

    def rotation(self) -> Rotation:
        """Return this pose's rotation as a SciPy `Rotation`."""

        return Rotation.from_quat(self.quaternion_as(QuaternionOrder.XYZW))

    def matrix(self) -> FloatArray:
        """Return a 4x4 homogeneous transform matrix."""

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.rotation().as_matrix()
        out[:3, 3] = self.translation
        return out

    def inverse(self) -> Pose:
        """Return the inverse transform."""

        rot_inv = self.rotation().inv()
        trans_inv = -rot_inv.apply(self.translation)
        quat = rot_inv.as_quat()
        return Pose(
            translation=trans_inv,
            quaternion=reorder_quaternion(quat, QuaternionOrder.XYZW, self.quaternion_order),
            quaternion_order=self.quaternion_order,
            frame=self.frame,
        )

    def transform_points(self, points: Any) -> FloatArray:
        """Transform points of shape `(..., 3)` from local to world coordinates."""

        arr = as_float_array(points, shape_tail=(3,), name="points")
        flat = arr.reshape(-1, 3)
        transformed = np.asarray(self.rotation().apply(flat) + self.translation, dtype=np.float64)
        return cast(FloatArray, transformed.reshape(arr.shape))

    def inverse_transform_points(self, points: Any) -> FloatArray:
        """Transform points of shape `(..., 3)` from world to local coordinates."""

        arr = as_float_array(points, shape_tail=(3,), name="points")
        flat = arr.reshape(-1, 3)
        transformed = np.asarray(self.rotation().inv().apply(flat - self.translation), dtype=np.float64)
        return cast(FloatArray, transformed.reshape(arr.shape))

    def to_frame(self, target: FrameConvention) -> Pose:
        """Return this pose represented in another coordinate frame convention."""

        transform = frame_transform_matrix(self.frame, target)
        rotation = Rotation.from_matrix(transform @ self.rotation().as_matrix() @ transform.T)
        quaternion = reorder_quaternion(rotation.as_quat(), QuaternionOrder.XYZW, self.quaternion_order)
        return Pose(
            translation=transform @ self.translation,
            quaternion=quaternion,
            quaternion_order=self.quaternion_order,
            frame=target,
        )

    def scaled(self, factor: float) -> Pose:
        """Return a copy with translation scaled and rotation preserved."""

        return Pose(
            translation=self.translation * float(factor),
            quaternion=self.quaternion.copy(),
            quaternion_order=self.quaternion_order,
            frame=self.frame,
        )


class PoseSequence(BaseModel):
    """Time-indexed sequence of poses."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    poses: tuple[Pose, ...]
    fps: float = 30.0

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("fps must be positive")
        return float(value)

    @model_validator(mode="after")
    def _validate_non_empty(self) -> PoseSequence:
        if not self.poses:
            raise ValueError("poses must not be empty")
        frames = {pose.frame for pose in self.poses}
        if len(frames) != 1:
            raise ValueError("all poses in a PoseSequence must use the same frame")
        return self

    @property
    def frame_count(self) -> int:
        """Number of poses."""

        return len(self.poses)

    @property
    def frame(self) -> FrameConvention:
        """Coordinate frame used by every pose."""

        return self.poses[0].frame

    @property
    def positions(self) -> FloatArray:
        """Stacked translations with shape `(T, 3)`."""

        return np.stack([pose.translation for pose in self.poses], axis=0)

    def quaternions(self, order: QuaternionOrder = QuaternionOrder.WXYZ) -> FloatArray:
        """Stacked quaternions with shape `(T, 4)`."""

        return np.stack([pose.quaternion_as(order) for pose in self.poses], axis=0)

    @classmethod
    def identity(
        cls,
        frame_count: int,
        *,
        fps: float = 30.0,
        frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    ) -> PoseSequence:
        """Return `frame_count` identity poses."""

        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        return cls(poses=tuple(Pose.identity(frame=frame) for _ in range(frame_count)), fps=fps)

    @classmethod
    def from_arrays(
        cls,
        positions: Any,
        quaternions: Any,
        *,
        fps: float = 30.0,
        quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ,
        frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    ) -> PoseSequence:
        """Build a pose sequence from position and quaternion arrays."""

        pos = as_float_array(positions, shape_tail=(3,), name="positions")
        quat = as_float_array(quaternions, shape_tail=(4,), name="quaternions")
        if pos.ndim != 2 or quat.ndim != 2 or pos.shape[0] != quat.shape[0]:
            raise ValueError("positions and quaternions must have matching first dimension")
        poses = tuple(
            Pose(translation=p, quaternion=q, quaternion_order=quaternion_order, frame=frame)
            for p, q in zip(pos, quat, strict=True)
        )
        return cls(poses=poses, fps=fps)

    def to_frame(self, target: FrameConvention) -> PoseSequence:
        """Return this sequence represented in another coordinate frame convention."""

        return PoseSequence(poses=tuple(pose.to_frame(target) for pose in self.poses), fps=self.fps)

    def scaled(self, factor: float) -> PoseSequence:
        """Return this sequence with translations scaled and rotations preserved."""

        return PoseSequence(poses=tuple(pose.scaled(factor) for pose in self.poses), fps=self.fps)

    def resampled(self, fps: float) -> PoseSequence:
        """Return this pose sequence sampled on a new FPS grid."""

        source_times, target_times = resampling_times(self.frame_count, self.fps, fps)
        positions = resample_linear(self.positions, self.fps, fps)
        order = self.poses[0].quaternion_order
        if self.frame_count == 1:
            quaternions = np.repeat(self.quaternions(order), len(target_times), axis=0)
            return PoseSequence.from_arrays(positions, quaternions, fps=fps, quaternion_order=order, frame=self.frame)
        rotations = Rotation.from_quat(self.quaternions(QuaternionOrder.XYZW))
        slerp = Slerp(source_times, rotations)
        resampled_quaternions_xyzw = slerp(target_times).as_quat()
        return PoseSequence.from_arrays(
            positions,
            resampled_quaternions_xyzw,
            fps=fps,
            quaternion_order=QuaternionOrder.XYZW,
            frame=self.frame,
        )
