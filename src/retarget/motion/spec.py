"""Motion data models."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import FrameConvention, MotionJoint, QuaternionOrder
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.timing import resample_linear


class MotionFormatSpec(BaseModel):
    """Describes a human motion data format.

    Attributes:
        name (str): Registry key and display name for the format.
        joint_vocabulary (type[MotionJoint]): Ordered enum defining the motion joints.
        root_joint (str): Kinematic root joint (must appear in ``joint_names``).
        quaternion_order (QuaternionOrder): Expected root-rotation storage order.
        frame_convention (FrameConvention): World-frame axis convention for positions.
        default_fps (float): Fallback sampling rate when a file omits ``fps``.
        default_height_m (float | None): Optional actor height metadata for scaling.
        description (str): Human-readable format notes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    joint_vocabulary: type[MotionJoint]
    root_joint: MotionJoint
    quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    frame_convention: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    default_fps: float = 30.0
    default_height_m: float | None = None
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_vocabulary_members(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        vocabulary = value.get("joint_vocabulary")
        if not isinstance(vocabulary, type) or not issubclass(vocabulary, MotionJoint):
            return value
        normalized = dict(value)
        if "root_joint" in normalized:
            normalized["root_joint"] = vocabulary(normalized["root_joint"])
        return normalized

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered serialized values from the declared joint vocabulary."""

        return tuple(joint.value for joint in self.joint_vocabulary)

    @model_validator(mode="after")
    def _validate_names(self) -> MotionFormatSpec:
        if not self.name:
            raise ValueError("name must not be empty")
        if not issubclass(self.joint_vocabulary, MotionJoint):
            raise TypeError("joint_vocabulary must subclass MotionJoint")
        if not self.joint_names:
            raise ValueError("joint_vocabulary must declare at least one joint")
        vocabulary_type = self.joint_vocabulary
        if not isinstance(self.root_joint, vocabulary_type):
            raise ValueError(f"{self.root_joint.value!r} is not a member of {vocabulary_type.__name__}")
        if self.default_fps <= 0:
            raise ValueError("default_fps must be positive")
        return self

    def joint_index(self, name: MotionJoint) -> int:
        """Return the index for a named joint."""

        if not isinstance(name, self.joint_vocabulary):
            raise KeyError(f"{name.value!r} is not a member of motion format {self.name!r}")
        try:
            return tuple(self.joint_vocabulary).index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown joint {name.value!r} for motion format {self.name!r}") from exc


class MotionSequence(BaseModel):
    """World-space human joint positions for a sequence.

    Attributes:
        name (str): Sequence label used in logs and exports.
        joint_positions (FloatArray): Positions with shape ``(frames, joints, 3)``.
        joint_names (tuple[str, ...]): Names aligned with the joint axis of ``joint_positions``.
        fps (float): Sampling rate in Hz.
        frame (FrameConvention): World-frame convention for ``joint_positions``.
        root_poses (PoseSequence | None): Optional per-frame root transform track.
        source_height_m (float | None): Source actor height used by scale-to-robot workflows.
        metadata (dict[str, Any]): Opaque sidecar fields for provenance and diagnostics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    joint_positions: FloatArray
    joint_names: tuple[str, ...]
    fps: float = 30.0
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    root_poses: PoseSequence | None = None
    source_height_m: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("joint_positions", mode="before")
    @classmethod
    def _validate_joint_positions(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, shape_tail=(3,), name="joint_positions")
        if arr.ndim != 3:
            raise ValueError("joint_positions must have shape (frames, joints, 3)")
        return arr

    @field_validator("fps")
    @classmethod
    def _validate_fps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("fps must be positive")
        return float(value)

    @field_validator("source_height_m")
    @classmethod
    def _validate_source_height(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("source_height_m must be positive")
        return None if value is None else float(value)

    @field_validator("metadata")
    @classmethod
    def _reject_behavior_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        blocked = {
            "contacts",
            "fps",
            "frame",
            "height_m",
            "joint_names",
            "joint_positions",
            "link_targets",
            "root_poses",
            "source_height_m",
            "support",
            "support_plane",
        }
        present = sorted(blocked & set(value))
        if present:
            raise ValueError(
                "MotionSequence metadata is provenance-only; use typed fields instead: " + ", ".join(present)
            )
        return dict(value)

    @model_validator(mode="after")
    def _validate_lengths(self) -> MotionSequence:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.joint_positions.shape[0] == 0:
            raise ValueError("motion must contain at least one frame")
        if self.joint_positions.shape[1] != len(self.joint_names):
            raise ValueError("joint_names length must match joint_positions.shape[1]")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        if self.root_poses is not None and self.root_poses.frame_count != self.frame_count:
            raise ValueError("root_poses frame count must match joint_positions")
        if self.root_poses is not None and self.root_poses.frame != self.frame:
            raise ValueError("root_poses frame must match motion frame")
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return int(self.joint_positions.shape[0])

    @property
    def joint_count(self) -> int:
        """Number of joints."""

        return int(self.joint_positions.shape[1])

    @property
    def duration_s(self) -> float:
        """Time between the first and last sample in seconds."""

        return (self.frame_count - 1) / self.fps

    def joint_index(self, name: str) -> int:
        """Return the index for a named joint."""

        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown joint {name!r} for sequence {self.name!r}") from exc

    def joint(self, name: str) -> FloatArray:
        """Return positions for one joint with shape `(T, 3)`."""

        return self.joint_positions[:, self.joint_index(name), :]

    def with_positions(
        self,
        positions: FloatArray | ArrayLike,
        *,
        name: str | None = None,
    ) -> MotionSequence:
        """Return a copy with new joint positions.

        Args:
            positions (FloatArray | ArrayLike): Replacement positions with shape ``(frames, joints, 3)``.
            name (str | None): Optional new sequence name; keeps the current name when omitted.

        Returns:
            MotionSequence: Copy sharing metadata and timing with updated positions.
        """

        return MotionSequence(
            name=name or self.name,
            joint_positions=positions,
            joint_names=self.joint_names,
            fps=self.fps,
            frame=self.frame,
            root_poses=self.root_poses,
            source_height_m=self.source_height_m,
            metadata=dict(self.metadata),
        )

    def scaled(self, factor: float, *, name: str | None = None) -> MotionSequence:
        """Return a copy with all joint positions scaled."""

        return MotionSequence(
            name=name or self.name,
            joint_positions=self.joint_positions * float(factor),
            joint_names=self.joint_names,
            fps=self.fps,
            frame=self.frame,
            root_poses=self.root_poses.scaled(factor) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            metadata=dict(self.metadata),
        )

    def resampled(self, fps: float, *, name: str | None = None) -> MotionSequence:
        """Return this motion sequence sampled on a new FPS grid."""

        metadata = dict(self.metadata)
        if not np.isclose(float(fps), self.fps):
            metadata["resampled_from_fps"] = self.fps
            metadata["resampled_to_fps"] = float(fps)
        return MotionSequence(
            name=name or self.name,
            joint_positions=resample_linear(self.joint_positions, self.fps, fps),
            joint_names=self.joint_names,
            fps=fps,
            frame=self.frame,
            root_poses=self.root_poses.resampled(fps) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            metadata=metadata,
        )

    def to_frame(self, target: FrameConvention, *, name: str | None = None) -> MotionSequence:
        """Return this motion represented in another coordinate frame convention."""

        if self.frame == target:
            return MotionSequence(
                name=name or self.name,
                joint_positions=self.joint_positions.copy(),
                joint_names=self.joint_names,
                fps=self.fps,
                frame=self.frame,
                root_poses=self.root_poses,
                source_height_m=self.source_height_m,
                metadata=dict(self.metadata),
            )
        metadata = dict(self.metadata)
        metadata.setdefault("source_frame", self.frame.value)
        metadata["frame_converted_from"] = self.frame.value
        metadata["frame_converted_to"] = target.value
        return MotionSequence(
            name=name or self.name,
            joint_positions=convert_points_frame(self.joint_positions, self.frame, target),
            joint_names=self.joint_names,
            fps=self.fps,
            frame=target,
            root_poses=self.root_poses.to_frame(target) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            metadata=metadata,
        )

    def centered_on_root(self, root_joint: str) -> MotionSequence:
        """Return a sequence translated so `root_joint` starts at the origin."""

        root0 = self.joint(root_joint)[0]
        root_poses = None
        if self.root_poses is not None:
            root_poses = PoseSequence(
                poses=tuple(
                    pose.model_copy(update={"translation": pose.translation - root0}) for pose in self.root_poses.poses
                ),
                fps=self.root_poses.fps,
            )
        return MotionSequence(
            name=self.name,
            joint_positions=self.joint_positions - root0,
            joint_names=self.joint_names,
            fps=self.fps,
            frame=self.frame,
            root_poses=root_poses,
            source_height_m=self.source_height_m,
            metadata=dict(self.metadata),
        )

    @classmethod
    def zeros(
        cls,
        name: str,
        joint_names: tuple[str, ...],
        frame_count: int,
        *,
        fps: float = 30.0,
        frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    ) -> MotionSequence:
        """Create a zero-valued fixture motion."""

        return cls(
            name=name,
            joint_names=joint_names,
            joint_positions=np.zeros((frame_count, len(joint_names), 3)),
            fps=fps,
            frame=frame,
        )
