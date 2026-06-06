"""Motion data models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.pose import PoseSequence, convert_points_frame
from retarget.core.timing import resample_linear, resampling_times
from retarget.motion.support import SupportPlane


class MotionFormatSpec(BaseModel):
    """Describes a human motion data format.

    Attributes:
        name (str): Registry key and display name for the format.
        joint_names (tuple[str, ...]): Ordered joint labels expected in motion data.
        root_joint (str): Kinematic root joint (must appear in ``joint_names``).
        contact_joints (tuple[str, ...]): Joints used for foot/contact inference.
        quaternion_order (QuaternionOrder): Expected root-rotation storage order.
        frame_convention (FrameConvention): World-frame axis convention for positions.
        default_fps (float): Fallback sampling rate when a file omits ``fps``.
        default_height_m (float | None): Optional actor height metadata for scaling.
        description (str): Human-readable format notes.
    """

    name: str
    joint_names: tuple[str, ...]
    root_joint: str
    contact_joints: tuple[str, ...] = ()
    quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    frame_convention: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    default_fps: float = 30.0
    default_height_m: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _validate_names(self) -> MotionFormatSpec:
        if not self.name:
            raise ValueError("name must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        for required in (self.root_joint, *self.contact_joints):
            if required not in self.joint_names:
                raise ValueError(f"{required!r} is not present in joint_names")
        if self.default_fps <= 0:
            raise ValueError("default_fps must be positive")
        return self

    def joint_index(self, name: str) -> int:
        """Return the index for a named joint."""

        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown joint {name!r} for motion format {self.name!r}") from exc


class MotionSequence(BaseModel):
    """World-space human joint positions for a sequence.

    Attributes:
        name (str): Sequence label used in logs and exports.
        joint_positions (FloatArray): Positions with shape ``(frames, joints, 3)``.
        joint_names (tuple[str, ...]): Names aligned with the joint axis of ``joint_positions``.
        fps (float): Sampling rate in Hz.
        frame (FrameConvention): World-frame convention for ``joint_positions``.
        root_poses (PoseSequence | None): Optional per-frame root transform track.
        contacts (tuple[dict[str, bool], ...]): Optional per-frame contact flags keyed by joint name.
        support (SupportPlane | None): Optional support geometry associated with contact flags.
        contact_provenance (dict[str, Any]): Provenance for loader-provided contact flags.
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
    contacts: tuple[dict[str, bool], ...] = ()
    support: SupportPlane | None = None
    contact_provenance: dict[str, Any] = Field(default_factory=dict)
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

    @field_validator("contacts", mode="before")
    @classmethod
    def _validate_contacts(cls, value: Any) -> tuple[dict[str, bool], ...]:
        if value is None or value == ():
            return ()
        if isinstance(value, Mapping):
            return _contacts_from_columns(value)
        frames: list[dict[str, bool]] = []
        for frame in value:
            if not isinstance(frame, Mapping):
                raise ValueError("contacts must be a sequence of mappings or a mapping of contact columns")
            frames.append({str(name): _coerce_bool(state) for name, state in frame.items()})
        return tuple(frames)

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
        if self.contacts and len(self.contacts) != self.frame_count:
            raise ValueError("contacts length must match joint_positions frame count")
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
            MotionSequence: Copy sharing metadata, contacts, and timing with updated positions.
        """

        return MotionSequence(
            name=name or self.name,
            joint_positions=positions,
            joint_names=self.joint_names,
            fps=self.fps,
            frame=self.frame,
            root_poses=self.root_poses,
            contacts=tuple(dict(frame) for frame in self.contacts),
            support=self.support,
            contact_provenance=dict(self.contact_provenance),
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
            contacts=tuple(dict(frame) for frame in self.contacts),
            support=self.support.scaled(factor) if self.support is not None else None,
            contact_provenance=dict(self.contact_provenance),
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
            contacts=_resample_contacts(self.contacts, self.fps, fps),
            support=self.support,
            contact_provenance=dict(self.contact_provenance),
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
                contacts=tuple(dict(frame) for frame in self.contacts),
                support=self.support,
                contact_provenance=dict(self.contact_provenance),
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
            contacts=tuple(dict(frame) for frame in self.contacts),
            support=self.support.to_frame(self.frame, target) if self.support is not None else None,
            contact_provenance=dict(self.contact_provenance),
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
                    pose.model_copy(update={"translation": pose.translation - root0})
                    for pose in self.root_poses.poses
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
            contacts=tuple(dict(frame) for frame in self.contacts),
            support=self.support,
            contact_provenance=dict(self.contact_provenance),
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


def _contacts_from_columns(value: Mapping[Any, Any]) -> tuple[dict[str, bool], ...]:
    columns = {str(name): tuple(states) for name, states in value.items()}
    lengths = {len(states) for states in columns.values()}
    if len(lengths) != 1:
        raise ValueError("contact columns must have matching lengths")
    if not lengths:
        return ()
    frame_count = lengths.pop()
    return tuple(
        {name: _coerce_bool(states[frame_idx]) for name, states in columns.items()}
        for frame_idx in range(frame_count)
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "contact", "contacting"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "none", "off", ""}:
            return False
    return bool(value)


def _resample_contacts(
    contacts: tuple[dict[str, bool], ...],
    source_fps: float,
    target_fps: float,
) -> tuple[dict[str, bool], ...]:
    if not contacts:
        return ()
    source_times, target_times = resampling_times(len(contacts), source_fps, target_fps)
    indices = np.asarray(
        [int(np.argmin(np.abs(source_times - target_time))) for target_time in target_times],
        dtype=int,
    )
    return tuple(dict(contacts[int(index)]) for index in indices)
