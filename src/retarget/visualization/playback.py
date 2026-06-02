"""Playback data models for visualization adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.spatial.transform import Rotation

from retarget.core.array import FloatArray, as_float_array, normalize_quaternion
from retarget.results.spec import RetargetingResult


class PlaybackFrame(BaseModel):
    """One frame of result playback data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    time_s: float
    root_position: FloatArray
    root_quaternion: FloatArray
    qpos: FloatArray
    human_points: FloatArray | None = None
    object_points: FloatArray | None = None

    @field_validator("root_position", mode="before")
    @classmethod
    def _validate_root_position(cls, value: Any) -> FloatArray:
        return as_float_array(value, shape_tail=(3,), name="root_position").reshape(3)

    @field_validator("root_quaternion", mode="before")
    @classmethod
    def _validate_root_quaternion(cls, value: Any) -> FloatArray:
        return normalize_quaternion(value, name="root_quaternion")

    @field_validator("qpos", mode="before")
    @classmethod
    def _validate_qpos(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="qpos")
        if arr.ndim != 1:
            raise ValueError("qpos must be 1D")
        return arr

    @field_validator("human_points", mode="before")
    @classmethod
    def _validate_human_points(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="human_points")
        if arr.ndim != 2:
            raise ValueError("human_points must have shape (points, 3)")
        return arr

    @field_validator("object_points", mode="before")
    @classmethod
    def _validate_object_points(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="object_points")
        if arr.ndim != 2:
            raise ValueError("object_points must have shape (points, 3)")
        return arr


class PlaybackObject(BaseModel):
    """Scene object samples transformed for result playback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    local_points: FloatArray
    world_points: FloatArray
    positions: FloatArray
    quaternions: FloatArray

    @field_validator("local_points", mode="before")
    @classmethod
    def _validate_local_points(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(3,), name="local_points")
        if arr.ndim != 2:
            raise ValueError("local_points must have shape (points, 3)")
        return arr

    @field_validator("world_points", mode="before")
    @classmethod
    def _validate_world_points(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(3,), name="world_points")
        if arr.ndim != 3:
            raise ValueError("world_points must have shape (frames, points, 3)")
        return arr

    @field_validator("positions", mode="before")
    @classmethod
    def _validate_positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(3,), name="positions")
        if arr.ndim != 2:
            raise ValueError("positions must have shape (frames, 3)")
        return arr

    @field_validator("quaternions", mode="before")
    @classmethod
    def _validate_quaternions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(4,), name="quaternions")
        if arr.ndim != 2:
            raise ValueError("quaternions must have shape (frames, 4)")
        return _normalize_quaternion_rows(arr)

    @model_validator(mode="after")
    def _validate_lengths(self) -> PlaybackObject:
        if self.world_points.shape[1] != self.local_points.shape[0]:
            raise ValueError("world point count must match local point count")
        if self.positions.shape[0] != self.world_points.shape[0]:
            raise ValueError("object positions must match world point frames")
        if self.quaternions.shape[0] != self.world_points.shape[0]:
            raise ValueError("object quaternions must match world point frames")
        return self

    @property
    def point_count(self) -> int:
        """Number of sampled object points."""

        return int(self.local_points.shape[0])


class PlaybackData(BaseModel):
    """Visualization-ready playback data derived from a retargeting result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    fps: float
    time_s: FloatArray
    qpos: FloatArray
    root_positions: FloatArray
    root_quaternions: FloatArray
    human_points: FloatArray | None = None
    object: PlaybackObject | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("time_s", mode="before")
    @classmethod
    def _validate_time(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="time_s")
        if arr.ndim != 1:
            raise ValueError("time_s must be 1D")
        return arr

    @field_validator("qpos", mode="before")
    @classmethod
    def _validate_qpos(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="qpos")
        if arr.ndim != 2:
            raise ValueError("qpos must have shape (frames, nq)")
        return arr

    @field_validator("root_positions", mode="before")
    @classmethod
    def _validate_root_positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(3,), name="root_positions")
        if arr.ndim != 2:
            raise ValueError("root_positions must have shape (frames, 3)")
        return arr

    @field_validator("root_quaternions", mode="before")
    @classmethod
    def _validate_root_quaternions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(4,), name="root_quaternions")
        if arr.ndim != 2:
            raise ValueError("root_quaternions must have shape (frames, 4)")
        return np.asarray([normalize_quaternion(quat) for quat in arr], dtype=np.float64)

    @field_validator("human_points", mode="before")
    @classmethod
    def _validate_human_points(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="human_points")
        if arr.ndim != 3:
            raise ValueError("human_points must have shape (frames, points, 3)")
        return arr

    @model_validator(mode="after")
    def _validate_lengths(self) -> PlaybackData:
        frames = self.qpos.shape[0]
        if frames == 0:
            raise ValueError("playback must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.time_s.shape != (frames,):
            raise ValueError("time_s length must match qpos frames")
        if self.root_positions.shape != (frames, 3):
            raise ValueError("root_positions shape must match qpos frames")
        if self.root_quaternions.shape != (frames, 4):
            raise ValueError("root_quaternions shape must match qpos frames")
        if self.human_points is not None and self.human_points.shape[0] != frames:
            raise ValueError("human_points frame count must match qpos")
        if self.object is not None and self.object.world_points.shape[0] != frames:
            raise ValueError("object playback frame count must match qpos")
        return self

    @property
    def frame_count(self) -> int:
        """Number of playback frames."""

        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        """Playback duration in seconds."""

        return float(self.time_s[-1]) if self.frame_count else 0.0

    def frame(self, index: int) -> PlaybackFrame:
        """Return one playback frame."""

        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        return PlaybackFrame(
            index=index,
            time_s=float(self.time_s[index]),
            root_position=self.root_positions[index],
            root_quaternion=self.root_quaternions[index],
            qpos=self.qpos[index],
            human_points=None if self.human_points is None else self.human_points[index],
            object_points=None if self.object is None else self.object.world_points[index],
        )


def build_playback_data(result: RetargetingResult) -> PlaybackData:
    """Convert a retargeting result into visualization-ready arrays."""

    frame_count = result.frame_count
    time_s = np.arange(frame_count, dtype=np.float64) / result.fps
    root_positions = _root_positions(result.qpos)
    root_quaternions = _root_quaternions(result.qpos, frame_count)
    playback_object = _object_playback(result)
    return PlaybackData(
        name=result.name,
        fps=result.fps,
        time_s=time_s,
        qpos=result.qpos,
        root_positions=root_positions,
        root_quaternions=root_quaternions,
        human_points=result.human_joints,
        object=playback_object,
        metadata={"status": result.status.value, **result.metadata},
    )


def _root_positions(qpos: FloatArray) -> FloatArray:
    if qpos.shape[1] >= 3:
        return np.asarray(qpos[:, :3], dtype=np.float64)
    positions = np.zeros((qpos.shape[0], 3), dtype=np.float64)
    positions[:, : qpos.shape[1]] = qpos
    return positions


def _root_quaternions(qpos: FloatArray, frame_count: int) -> FloatArray:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if qpos.shape[1] >= 7:
        quaternions = np.asarray(qpos[:, 3:7], dtype=np.float64).copy()
        norms = np.linalg.norm(quaternions, axis=1)
        zero_mask = norms <= 1e-12
        quaternions[zero_mask] = identity
        quaternions[~zero_mask] = quaternions[~zero_mask] / norms[~zero_mask, None]
        return quaternions
    return np.tile(identity[None, :], (frame_count, 1))


def _object_playback(result: RetargetingResult) -> PlaybackObject | None:
    object_metadata = _object_playback_metadata(result.metadata)
    if object_metadata is None:
        return None
    sample_points = object_metadata.get("sample_points")
    if sample_points is None:
        return None

    local_points = as_float_array(sample_points, shape_tail=(3,), name="object.sample_points")
    if local_points.ndim != 2:
        raise ValueError("object.sample_points must have shape (points, 3)")

    object_slice = _object_slice(object_metadata.get("qpos_slice"), result.qpos.shape[1])
    if object_slice is not None:
        start, _stop = object_slice
        positions = np.asarray(result.qpos[:, start : start + 3], dtype=np.float64)
        quaternions = _normalize_quaternion_rows(result.qpos[:, start + 3 : start + 7])
        world_points = _transform_points(local_points, positions, quaternions)
    else:
        positions = np.zeros((result.frame_count, 3), dtype=np.float64)
        quaternions = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (result.frame_count, 1))
        world_points = np.tile(local_points[None, :, :], (result.frame_count, 1, 1))

    return PlaybackObject(
        name=str(object_metadata.get("name", "object")),
        local_points=local_points,
        world_points=world_points,
        positions=positions,
        quaternions=quaternions,
    )


def _object_playback_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    playback = metadata.get("playback")
    if not isinstance(playback, dict):
        return None
    object_metadata = playback.get("object")
    return object_metadata if isinstance(object_metadata, dict) else None


def _object_slice(value: Any, qpos_width: int) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        start = int(value[0])
        stop = int(value[1])
    except (TypeError, ValueError):
        return None
    if 0 <= start < stop <= qpos_width and stop - start >= 7:
        return start, stop
    return None


def _normalize_quaternion_rows(quaternions: Any) -> FloatArray:
    arr = as_float_array(quaternions, shape_tail=(4,), name="quaternions")
    if arr.ndim != 2:
        raise ValueError("quaternions must have shape (frames, 4)")
    normalized = np.asarray(arr, dtype=np.float64).copy()
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norms = np.linalg.norm(normalized, axis=1)
    zero_mask = norms <= 1e-12
    normalized[zero_mask] = identity
    normalized[~zero_mask] = normalized[~zero_mask] / norms[~zero_mask, None]
    return normalized


def _transform_points(local_points: FloatArray, positions: FloatArray, quaternions_wxyz: FloatArray) -> FloatArray:
    rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
    return np.asarray(
        [rotation.apply(local_points) + position for rotation, position in zip(rotations, positions, strict=True)],
        dtype=np.float64,
    )


__all__ = ["PlaybackData", "PlaybackFrame", "PlaybackObject", "build_playback_data"]
