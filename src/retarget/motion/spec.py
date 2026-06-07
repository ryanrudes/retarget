"""Typed motion format and shared-timeline actor motion models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeVar

from retarget.capture.timeline import SampleTimeline
from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import FrameConvention, MotionJoint, QuaternionOrder
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.timing import resample_linear

JointT = TypeVar("JointT", bound=MotionJoint, default=MotionJoint)


class MotionFormatSpec(BaseModel, Generic[JointT]):
    """Concrete source-motion vocabulary and storage conventions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    joint_vocabulary: type[JointT]
    root_joint: JointT
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
            normalized["root_joint"] = _member(normalized["root_joint"], vocabulary)
        return normalized

    @property
    def joints(self) -> tuple[JointT, ...]:
        """Ordered joints declared by this format."""

        return tuple(self.joint_vocabulary)

    @model_validator(mode="after")
    def _validate_format(self) -> MotionFormatSpec[JointT]:
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.joint_vocabulary, type) or not issubclass(self.joint_vocabulary, MotionJoint):
            raise TypeError("joint_vocabulary must subclass MotionJoint")
        if not self.joints:
            raise ValueError("joint_vocabulary must declare at least one joint")
        if not isinstance(self.root_joint, self.joint_vocabulary):
            raise TypeError(f"root_joint must be a {self.joint_vocabulary.__name__} member")
        if self.default_fps <= 0:
            raise ValueError("default_fps must be positive")
        if self.default_height_m is not None and self.default_height_m <= 0:
            raise ValueError("default_height_m must be positive")
        return self

    def joint_index(self, joint: JointT) -> int:
        """Return the index of a typed joint."""

        _require_joint(joint, self.joint_vocabulary)
        try:
            return self.joints.index(joint)
        except ValueError as exc:
            raise KeyError(f"Unknown joint {joint.value!r} for motion format {self.name!r}") from exc


class MotionSequence(BaseModel, Generic[JointT]):
    """Canonical shared-timeline actor motion with a concrete joint vocabulary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    joint_vocabulary: type[JointT]
    joints: tuple[JointT, ...]
    root_joint: JointT
    joint_positions: FloatArray
    timeline: SampleTimeline
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    root_poses: PoseSequence | None = None
    source_height_m: float | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_joints(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        vocabulary = value.get("joint_vocabulary")
        if not isinstance(vocabulary, type) or not issubclass(vocabulary, MotionJoint):
            return value
        normalized = dict(value)
        normalized["joints"] = tuple(_member(joint, vocabulary) for joint in normalized.get("joints", ()))
        if "root_joint" in normalized:
            normalized["root_joint"] = _member(normalized["root_joint"], vocabulary)
        return normalized

    @field_validator("joint_positions", mode="before")
    @classmethod
    def _validate_joint_positions(cls, value: Any) -> FloatArray:
        array = as_float_array(value, shape_tail=(3,), name="joint_positions")
        if array.ndim != 3:
            raise ValueError("joint_positions must have shape (frames, joints, 3)")
        return array

    @model_validator(mode="after")
    def _validate_sequence(self) -> MotionSequence[JointT]:
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.joint_vocabulary, type) or not issubclass(self.joint_vocabulary, MotionJoint):
            raise TypeError("joint_vocabulary must subclass MotionJoint")
        if not self.joints:
            raise ValueError("motion must declare at least one joint")
        if not all(isinstance(joint, self.joint_vocabulary) for joint in self.joints):
            raise TypeError(f"motion joints must use {self.joint_vocabulary.__name__}")
        if not isinstance(self.root_joint, self.joint_vocabulary) or self.root_joint not in self.joints:
            raise TypeError("root_joint must be a declared member of the motion joint vocabulary")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError("motion joints must be unique")
        if self.joint_positions.shape[:2] != (self.timeline.sample_count, len(self.joints)):
            raise ValueError("joint_positions shape must match timeline samples and joints")
        if self.root_poses is not None and self.root_poses.frame_count != self.frame_count:
            raise ValueError("root_poses frame count must match joint_positions")
        if self.root_poses is not None and self.root_poses.frame != self.frame:
            raise ValueError("root_poses frame must match motion frame")
        if self.source_height_m is not None and self.source_height_m <= 0:
            raise ValueError("source_height_m must be positive")
        object.__setattr__(self, "provenance", dict(self.provenance))
        return self

    @property
    def frame_count(self) -> int:
        """Number of samples."""

        return self.timeline.sample_count

    @property
    def joint_count(self) -> int:
        """Number of joints."""

        return len(self.joints)

    @property
    def fps(self) -> float:
        """Nominal sampling rate."""

        fps = self.timeline.nominal_fps
        if fps is None:
            raise ValueError("a single-sample motion has no nominal fps")
        return fps

    @property
    def duration_s(self) -> float:
        """Duration between the first and last samples."""

        return self.timeline.duration_s

    def joint_index(self, joint: JointT) -> int:
        """Return the index of a typed joint."""

        _require_joint(joint, self.joint_vocabulary)
        try:
            return self.joints.index(joint)
        except ValueError as exc:
            raise KeyError(f"Unknown joint {joint.value!r} for sequence {self.name!r}") from exc

    def joint(self, joint: JointT) -> FloatArray:
        """Return positions for one joint with shape ``(samples, 3)``."""

        return self.joint_positions[:, self.joint_index(joint), :]

    def with_positions(self, positions: FloatArray | ArrayLike, *, name: str | None = None) -> MotionSequence[JointT]:
        """Return a copy with replacement joint positions."""

        return self.model_copy(update={"name": name or self.name, "joint_positions": positions})

    def scaled(self, factor: float, *, name: str | None = None) -> MotionSequence[JointT]:
        """Return a spatially scaled copy."""

        return MotionSequence(
            name=name or self.name,
            joint_vocabulary=self.joint_vocabulary,
            joints=self.joints,
            root_joint=self.root_joint,
            joint_positions=self.joint_positions * float(factor),
            timeline=self.timeline,
            frame=self.frame,
            root_poses=self.root_poses.scaled(factor) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            provenance={**self.provenance, "scale_factor": float(factor)},
        )

    def resampled(self, fps: float, *, name: str | None = None) -> MotionSequence[JointT]:
        """Return this motion sampled on a uniform timeline."""

        if fps <= 0:
            raise ValueError("fps must be positive")
        if np.isclose(float(fps), self.fps):
            return self
        positions = resample_linear(self.joint_positions, self.fps, fps)
        timeline = SampleTimeline.uniform(
            positions.shape[0],
            fps,
            start_s=self.timeline.start_s,
            clock=self.timeline.clock,
        )
        return MotionSequence(
            name=name or self.name,
            joint_vocabulary=self.joint_vocabulary,
            joints=self.joints,
            root_joint=self.root_joint,
            joint_positions=positions,
            timeline=timeline,
            frame=self.frame,
            root_poses=self.root_poses.resampled(fps) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            provenance={
                **self.provenance,
                "resampled_from_fps": self.fps,
                "resampled_to_fps": float(fps),
            },
        )

    def to_frame(self, target: FrameConvention, *, name: str | None = None) -> MotionSequence[JointT]:
        """Return this motion represented in another coordinate frame."""

        if self.frame == target:
            return self.model_copy(update={"name": name or self.name, "joint_positions": self.joint_positions.copy()})
        return MotionSequence(
            name=name or self.name,
            joint_vocabulary=self.joint_vocabulary,
            joints=self.joints,
            root_joint=self.root_joint,
            joint_positions=convert_points_frame(self.joint_positions, self.frame, target),
            timeline=self.timeline,
            frame=target,
            root_poses=self.root_poses.to_frame(target) if self.root_poses is not None else None,
            source_height_m=self.source_height_m,
            provenance={
                **self.provenance,
                "frame_converted_from": self.frame.value,
                "frame_converted_to": target.value,
            },
        )

    def centered_on_root(self, root_joint: JointT) -> MotionSequence[JointT]:
        """Return a sequence translated so ``root_joint`` begins at the origin."""

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
            joint_vocabulary=self.joint_vocabulary,
            joints=self.joints,
            root_joint=self.root_joint,
            joint_positions=self.joint_positions - root0,
            timeline=self.timeline,
            frame=self.frame,
            root_poses=root_poses,
            source_height_m=self.source_height_m,
            provenance=dict(self.provenance),
        )

    @classmethod
    def zeros(
        cls,
        name: str,
        *,
        joint_vocabulary: type[JointT],
        joints: tuple[JointT, ...],
        root_joint: JointT,
        frame_count: int,
        fps: float = 30.0,
        frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    ) -> MotionSequence[JointT]:
        """Create a zero-valued fixture motion."""

        return cls(
            name=name,
            joint_vocabulary=joint_vocabulary,
            joints=joints,
            root_joint=root_joint,
            joint_positions=np.zeros((frame_count, len(joints), 3)),
            timeline=SampleTimeline.uniform(frame_count, fps, clock=f"motion:{name}"),
            frame=frame,
        )


def _member(value: Any, vocabulary: type[JointT]) -> JointT:
    if isinstance(value, StrEnum):
        if not isinstance(value, vocabulary):
            raise TypeError(
                f"expected a {vocabulary.__name__} member, received {type(value).__name__}.{value.name}"
            )
        return value
    return vocabulary(str(value))


def _require_joint(joint: MotionJoint, vocabulary: type[MotionJoint]) -> None:
    if not isinstance(joint, vocabulary):
        raise TypeError(f"joint must be a {vocabulary.__name__} member")


AnyMotionFormatSpec = MotionFormatSpec[Any]
AnyMotionSequence = MotionSequence[Any]
