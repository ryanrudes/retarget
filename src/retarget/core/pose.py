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
    """Convert quaternion storage order without changing the represented rotation.

    Args:
        quaternion (np.ndarray): Quaternion components in ``source`` layout.
        source (QuaternionOrder): Storage order of ``quaternion``.
        target (QuaternionOrder): Desired storage order.

    Returns:
        np.ndarray: Quaternion with the same rotation in ``target`` layout.
    """

    q = np.asarray(quaternion, dtype=np.float64)
    if source == target:
        return q.copy()
    if source == QuaternionOrder.WXYZ and target == QuaternionOrder.XYZW:
        return q[[1, 2, 3, 0]]
    if source == QuaternionOrder.XYZW and target == QuaternionOrder.WXYZ:
        return q[[3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion conversion: {source} -> {target}")


def frame_transform_matrix(source: FrameConvention, target: FrameConvention) -> FloatArray:
    """Return the rotation matrix that maps coordinates from ``source`` to ``target``.

    Args:
        source (FrameConvention): Input coordinate convention.
        target (FrameConvention): Output coordinate convention.

    Returns:
        FloatArray: ``(3, 3)`` rotation matrix ``R`` with ``p_target = R @ p_source``.
    """

    if source == target:
        return np.eye(3, dtype=np.float64)
    if source == FrameConvention.Y_UP_RIGHT_HANDED and target == FrameConvention.Z_UP_RIGHT_HANDED:
        return Y_UP_TO_Z_UP.copy()
    if source == FrameConvention.Z_UP_RIGHT_HANDED and target == FrameConvention.Y_UP_RIGHT_HANDED:
        return Y_UP_TO_Z_UP.T.copy()
    raise ValueError(f"Unsupported frame conversion: {source} -> {target}")


def convert_points_frame(points: Any, source: FrameConvention, target: FrameConvention) -> FloatArray:
    """Convert point coordinates between supported right-handed frame conventions.

    Args:
        points (Any): Array-like positions with trailing dimension 3.
        source (FrameConvention): Convention of the input coordinates.
        target (FrameConvention): Desired output convention.

    Returns:
        FloatArray: Points re-expressed in ``target``, preserving the input shape.
    """

    arr = as_float_array(points, shape_tail=(3,), name="points")
    transform = frame_transform_matrix(source, target)
    converted = arr.reshape(-1, 3) @ transform.T
    return cast(FloatArray, converted.reshape(arr.shape))


class Pose(BaseModel):
    """Rigid transform from a local frame into a named world convention.

    Attributes:
        translation (FloatArray): World-frame origin offset, shape ``(3,)``.
        quaternion (FloatArray): Unit quaternion in ``quaternion_order`` layout, shape ``(4,)``.
        quaternion_order (QuaternionOrder): Storage order of ``quaternion``.
        frame (FrameConvention): World coordinate convention for ``translation``.
    """

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
        """Return an identity pose.

        Args:
            frame (FrameConvention): World coordinate convention for the pose.

        Returns:
            Pose: Identity transform with zero translation and unit rotation.
        """

        return cls(frame=frame)

    def quaternion_as(self, order: QuaternionOrder) -> FloatArray:
        """Return the quaternion in the requested storage order.

        Args:
            order (QuaternionOrder): Desired component layout.

        Returns:
            FloatArray: Unit quaternion with shape ``(4,)`` in ``order``.
        """

        return reorder_quaternion(self.quaternion, self.quaternion_order, order)

    def rotation(self) -> Rotation:
        """Return this pose's rotation as a SciPy `Rotation`.

        Returns:
            Rotation: Orientation in SciPy's scalar-last ``(x, y, z, w)`` convention.
        """

        return Rotation.from_quat(self.quaternion_as(QuaternionOrder.XYZW))

    def matrix(self) -> FloatArray:
        """Return a 4x4 homogeneous transform matrix.

        Returns:
            FloatArray: ``(4, 4)`` matrix mapping local homogeneous coordinates to world.
        """

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.rotation().as_matrix()
        out[:3, 3] = self.translation
        return out

    def inverse(self) -> Pose:
        """Return the inverse transform.

        Returns:
            Pose: Transform that maps world coordinates back to the local frame.
        """

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
        """Transform points of shape `(..., 3)` from local to world coordinates.

        Args:
            points (Any): Local-frame positions with trailing dimension 3.

        Returns:
            FloatArray: World-frame positions with the same shape as ``points``.
        """

        arr = as_float_array(points, shape_tail=(3,), name="points")
        flat = arr.reshape(-1, 3)
        transformed = np.asarray(self.rotation().apply(flat) + self.translation, dtype=np.float64)
        return cast(FloatArray, transformed.reshape(arr.shape))

    def inverse_transform_points(self, points: Any) -> FloatArray:
        """Transform points of shape `(..., 3)` from world to local coordinates.

        Args:
            points (Any): World-frame positions with trailing dimension 3.

        Returns:
            FloatArray: Local-frame positions with the same shape as ``points``.
        """

        arr = as_float_array(points, shape_tail=(3,), name="points")
        flat = arr.reshape(-1, 3)
        transformed = np.asarray(self.rotation().inv().apply(flat - self.translation), dtype=np.float64)
        return cast(FloatArray, transformed.reshape(arr.shape))

    def to_frame(self, target: FrameConvention) -> Pose:
        """Return this pose represented in another coordinate frame convention.

        Args:
            target (FrameConvention): Desired world coordinate convention.

        Returns:
            Pose: Same rigid transform expressed in ``target``.
        """

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
        """Return a copy with translation scaled and rotation preserved.

        Args:
            factor (float): Multiplier applied to ``translation``.

        Returns:
            Pose: Copy with scaled translation and unchanged orientation.
        """

        return Pose(
            translation=self.translation * float(factor),
            quaternion=self.quaternion.copy(),
            quaternion_order=self.quaternion_order,
            frame=self.frame,
        )


class PoseSequence(BaseModel):
    """Time-indexed sequence of poses.

    Attributes:
        poses (tuple[Pose, ...]): Per-frame rigid transforms sharing one ``frame``.
        fps (float): Sampling rate in Hz for time-based resampling.
    """

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
        """Stacked quaternions with shape `(T, 4)`.

        Args:
            order (QuaternionOrder): Storage order for each frame's quaternion.

        Returns:
            FloatArray: Quaternion array with shape ``(T, 4)``.
        """

        return np.stack([pose.quaternion_as(order) for pose in self.poses], axis=0)

    @classmethod
    def identity(
        cls,
        frame_count: int,
        *,
        fps: float = 30.0,
        frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    ) -> PoseSequence:
        """Return `frame_count` identity poses.

        Args:
            frame_count (int): Number of frames to create.
            fps (float): Sampling rate in Hz.
            frame (FrameConvention): World coordinate convention for every pose.

        Returns:
            PoseSequence: Sequence of identity transforms.
        """

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
        """Build a pose sequence from position and quaternion arrays.

        Args:
            positions (Any): Translations with shape ``(T, 3)``.
            quaternions (Any): Orientations with shape ``(T, 4)`` in ``quaternion_order``.
            fps (float): Sampling rate in Hz.
            quaternion_order (QuaternionOrder): Layout of each row in ``quaternions``.
            frame (FrameConvention): World coordinate convention for every pose.

        Returns:
            PoseSequence: One ``Pose`` per matched time index.
        """

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
        """Return this sequence represented in another coordinate frame convention.

        Args:
            target (FrameConvention): Desired world coordinate convention.

        Returns:
            PoseSequence: Copy with each pose converted to ``target``.
        """

        return PoseSequence(poses=tuple(pose.to_frame(target) for pose in self.poses), fps=self.fps)

    def scaled(self, factor: float) -> PoseSequence:
        """Return this sequence with translations scaled and rotations preserved.

        Args:
            factor (float): Multiplier applied to every pose translation.

        Returns:
            PoseSequence: Copy with scaled translations and unchanged orientations.
        """

        return PoseSequence(poses=tuple(pose.scaled(factor) for pose in self.poses), fps=self.fps)

    def resampled(self, fps: float) -> PoseSequence:
        """Return this pose sequence sampled on a new FPS grid.

        Args:
            fps (float): Target sampling rate in Hz.

        Returns:
            PoseSequence: Endpoint-preserving resample with linear translation and
                spherical-linear rotation interpolation.
        """

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
