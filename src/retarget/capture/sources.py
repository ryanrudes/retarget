"""Observation sources for native recordings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

from retarget.capture.recordings import HumanPoseRecording, MocapRecording
from retarget.capture.timeline import SampleTimeline
from retarget.capture.tracks import JointTrack, MarkerTrack, RigidBodyTrack
from retarget.core.enums import (
    FrameConvention,
    MocapMarker,
    MocapRigidBody,
    MotionJoint,
    NameEnum,
    QuaternionOrder,
)

RecordingT = TypeVar("RecordingT", covariant=True)


class ObservationSource(Protocol[RecordingT]):
    """Load one native recording without creating a shared timeline."""

    def load(self) -> RecordingT:
        """Load the recording."""


@dataclass(frozen=True)
class InMemorySource(Generic[RecordingT]):
    """Source returning an already constructed recording."""

    recording: RecordingT

    def load(self) -> RecordingT:
        """Return the stored recording."""

        return self.recording


@dataclass(frozen=True)
class ViconSourceSchema:
    """Map Vicon table names into typed native vocabularies."""

    rigid_bodies: Mapping[str, MocapRigidBody]
    markers: Mapping[str, MocapMarker]
    rigid_body_order: tuple[str, ...] | None = None
    marker_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, MocapRigidBody) for value in self.rigid_bodies.values()):
            raise TypeError("rigid-body schema values must be MocapRigidBody members")
        if not all(isinstance(value, MocapMarker) for value in self.markers.values()):
            raise TypeError("marker schema values must be MocapMarker members")
        _require_one_schema_vocabulary(self.rigid_bodies.values(), label="rigid-body")
        _require_one_schema_vocabulary(self.markers.values(), label="marker")
        if len(set(self.rigid_bodies.values())) != len(self.rigid_bodies):
            raise ValueError("rigid-body schema roles must be unique")
        if len(set(self.markers.values())) != len(self.markers):
            raise ValueError("marker schema roles must be unique")
        _validate_native_order(self.rigid_body_order, self.rigid_bodies, label="rigid-body")
        _validate_native_order(self.marker_order, self.markers, label="marker")


@dataclass(frozen=True)
class ViconRecordingSource:
    """Load a preconverted Vicon recording directly into typed tracks."""

    path: Path
    schema: ViconSourceSchema
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    name: str = ""

    def load(self) -> MocapRecording:
        """Load ``vicon.npz`` without a synchronization intermediate."""

        path = Path(self.path)
        if path.is_dir():
            path = path / "vicon.npz"
        with np.load(path, allow_pickle=False) as data:
            timestamps = np.asarray(data["stamp"], dtype=np.float64)
            timestamps -= timestamps[0]
            body_positions = np.asarray(data["body_pos"], dtype=np.float64)
            body_quaternions = np.asarray(data["body_quat"], dtype=np.float64)
            body_names = _native_names(
                data,
                key="body_names",
                explicit=self.schema.rigid_body_order,
                width=body_positions.shape[1],
            )
            body_occluded = np.asarray(
                data.get("body_occluded", np.zeros(body_positions.shape[:2], dtype=bool)),
                dtype=bool,
            )
            marker_positions = np.asarray(data["marker_pos"], dtype=np.float64)
            marker_names = _native_names(
                data,
                key="marker_names",
                explicit=self.schema.marker_order,
                width=marker_positions.shape[1],
                required=bool(self.schema.markers),
            )
            marker_occluded = np.asarray(
                data.get("marker_occluded", np.zeros(marker_positions.shape[:2], dtype=bool)),
                dtype=bool,
            )
        timeline = SampleTimeline(timestamps, clock=f"vicon:{path.stem}")
        rigid_bodies = tuple(
            RigidBodyTrack(
                role=role,
                positions=body_positions[:, body_names.index(native_name)],
                quaternions=body_quaternions[:, body_names.index(native_name)],
                quaternion_order=QuaternionOrder.WXYZ,
                validity=~body_occluded[:, body_names.index(native_name)],
                provenance={"native_name": native_name},
            )
            for native_name, role in self.schema.rigid_bodies.items()
        )
        markers = tuple(
            MarkerTrack(
                role=role,
                values=marker_positions[:, marker_names.index(native_name)],
                validity=~marker_occluded[:, marker_names.index(native_name)],
                provenance={"native_name": native_name},
            )
            for native_name, role in self.schema.markers.items()
        )
        return MocapRecording(
            name=self.name or path.parent.name or path.stem,
            timeline=timeline,
            frame=self.frame,
            rigid_bodies=rigid_bodies,
            markers=markers,
            provenance={"source_path": str(path), "source_type": "vicon_recording"},
        )


@dataclass(frozen=True)
class ViconBagTopics:
    """ROS topics carrying Vicon rigid-body transforms and marker samples."""

    transforms: tuple[str, ...] = ("/tf",)
    markers: tuple[str, ...] = ("/vicon/markers",)


@dataclass(frozen=True)
class ViconBagSource:
    """Read a ROS 2 Vicon bag directly into a native mocap recording.

    The optional ``capture-ros`` dependency is imported only when :meth:`load`
    runs. The bag is never converted into a persistent table or archive.
    """

    path: Path
    schema: ViconSourceSchema
    topics: ViconBagTopics = ViconBagTopics()
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    name: str = ""
    transform_scale_m: float = 1.0
    marker_scale_m: float = 0.001

    def load(self) -> MocapRecording:
        """Deserialize configured bag topics into typed Vicon tracks."""

        try:
            from rosbags.highlevel import AnyReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("ViconBagSource requires the capture-ros extra: `uv sync --extra capture-ros`") from exc

        path = Path(self.path)
        if self.transform_scale_m <= 0.0 or self.marker_scale_m <= 0.0:
            raise ValueError("Vicon bag unit scales must be positive")

        body_samples: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        marker_samples: dict[float, dict[str, tuple[np.ndarray, bool]]] = {}
        selected_topics = set((*self.topics.transforms, *self.topics.markers))

        with AnyReader([path]) as reader:
            connections = [connection for connection in reader.connections if connection.topic in selected_topics]
            if not connections:
                raise ValueError(f"bag {path} contains none of the configured Vicon topics {sorted(selected_topics)}")
            for connection, recorded_ns, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic in self.topics.transforms:
                    self._collect_transforms(
                        message,
                        recorded_ns,
                        body_samples,
                    )
                if connection.topic in self.topics.markers:
                    self._collect_markers(
                        message,
                        recorded_ns,
                        marker_samples,
                    )

        timestamps = np.asarray(
            sorted(set(body_samples) | set(marker_samples)),
            dtype=np.float64,
        )
        if timestamps.size == 0:
            raise ValueError(f"bag {path} contained no configured Vicon samples")
        timestamps -= timestamps[0]
        absolute_times = sorted(set(body_samples) | set(marker_samples))
        timeline = SampleTimeline(timestamps, clock=f"vicon-bag:{path.stem}")

        rigid_bodies = tuple(
            self._body_track(
                native_name,
                role,
                absolute_times,
                body_samples,
            )
            for native_name, role in self.schema.rigid_bodies.items()
        )
        markers = tuple(
            self._marker_track(
                native_name,
                role,
                absolute_times,
                marker_samples,
            )
            for native_name, role in self.schema.markers.items()
        )
        return MocapRecording(
            name=self.name or path.stem,
            timeline=timeline,
            frame=self.frame,
            rigid_bodies=rigid_bodies,
            markers=markers,
            provenance={"source_path": str(path), "source_type": "vicon_bag"},
        )

    def _collect_transforms(
        self,
        message: Any,
        recorded_ns: int,
        samples: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]],
    ) -> None:
        transforms = getattr(message, "transforms", ())
        for transform in transforms:
            native_name = _schema_name(
                str(transform.child_frame_id),
                self.schema.rigid_bodies,
            )
            if native_name is None:
                continue
            stamp = _message_time_s(transform, recorded_ns)
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            samples.setdefault(stamp, {})[native_name] = (
                self.transform_scale_m
                * np.asarray(
                    [translation.x, translation.y, translation.z],
                    dtype=np.float64,
                ),
                np.asarray(
                    [rotation.w, rotation.x, rotation.y, rotation.z],
                    dtype=np.float64,
                ),
            )

    def _collect_markers(
        self,
        message: Any,
        recorded_ns: int,
        samples: dict[float, dict[str, tuple[np.ndarray, bool]]],
    ) -> None:
        stamp = _message_time_s(message, recorded_ns)
        for marker in getattr(message, "markers", ()):
            native_name = _schema_name(
                str(marker.marker_name),
                self.schema.markers,
            )
            if native_name is None:
                continue
            translation = marker.translation
            occluded = bool(marker.occluded)
            position = self.marker_scale_m * np.asarray(
                [translation.x, translation.y, translation.z],
                dtype=np.float64,
            )
            samples.setdefault(stamp, {})[native_name] = (position, not occluded)

    @staticmethod
    def _body_track(
        native_name: str,
        role: MocapRigidBody,
        timestamps: list[float],
        samples: Mapping[float, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    ) -> RigidBodyTrack:
        positions = np.full((len(timestamps), 3), np.nan, dtype=np.float64)
        quaternions = np.full((len(timestamps), 4), np.nan, dtype=np.float64)
        validity = np.zeros(len(timestamps), dtype=bool)
        for index, timestamp in enumerate(timestamps):
            sample = samples.get(timestamp, {}).get(native_name)
            if sample is not None:
                positions[index], quaternions[index] = sample
                validity[index] = True
        return RigidBodyTrack(
            role=role,
            positions=positions,
            quaternions=quaternions,
            quaternion_order=QuaternionOrder.WXYZ,
            validity=validity,
            provenance={"native_name": native_name},
        )

    @staticmethod
    def _marker_track(
        native_name: str,
        role: MocapMarker,
        timestamps: list[float],
        samples: Mapping[float, Mapping[str, tuple[np.ndarray, bool]]],
    ) -> MarkerTrack:
        positions = np.full((len(timestamps), 3), np.nan, dtype=np.float64)
        validity = np.zeros(len(timestamps), dtype=bool)
        for index, timestamp in enumerate(timestamps):
            sample = samples.get(timestamp, {}).get(native_name)
            if sample is not None:
                positions[index], validity[index] = sample
        return MarkerTrack(
            role=role,
            values=positions,
            validity=validity,
            provenance={"native_name": native_name},
        )


def _message_time_s(message: Any, recorded_ns: int) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return recorded_ns / 1e9
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _schema_name(native_name: str, schema: Mapping[str, NameEnum]) -> str | None:
    if native_name in schema:
        return native_name
    components = tuple(component for component in native_name.split("/") if component)
    matches = [candidate for candidate in schema if candidate in components]
    if len(matches) > 1:
        raise ValueError(f"native name {native_name!r} ambiguously matches schema entries {matches}")
    return matches[0] if matches else None


@dataclass(frozen=True)
class HumanPoseSourceSchema:
    """Map semantic motion-joint vocabularies to source array columns."""

    joint_indices: Mapping[MotionJoint, int]

    def __post_init__(self) -> None:
        if not self.joint_indices:
            raise ValueError("human-pose schema requires at least one joint")
        if not all(isinstance(role, MotionJoint) for role in self.joint_indices):
            raise TypeError("human-pose schema keys must be MotionJoint members")
        _require_one_schema_vocabulary(self.joint_indices, label="human-pose joint")
        if len(set(self.joint_indices.values())) != len(self.joint_indices):
            raise ValueError("human-pose schema indices must be unique")
        if min(self.joint_indices.values()) < 0:
            raise ValueError("human-pose schema indices must be non-negative")


def _require_one_schema_vocabulary(values: Iterable[NameEnum], *, label: str) -> None:
    members = tuple(values)
    if not members:
        return
    vocabulary = type(members[0])
    if not all(type(member) is vocabulary for member in members):
        raise TypeError(f"{label} schema must use one enum vocabulary")


def _validate_native_order(
    order: tuple[str, ...] | None,
    mapping: Mapping[str, NameEnum],
    *,
    label: str,
) -> None:
    if order is None:
        return
    if len(set(order)) != len(order):
        raise ValueError(f"{label} native order must be unique")
    missing = set(mapping) - set(order)
    if missing:
        raise ValueError(f"{label} native order is missing mapped names: {sorted(missing)}")


def _native_names(
    data: Any,
    *,
    key: str,
    explicit: tuple[str, ...] | None,
    width: int,
    required: bool = True,
) -> tuple[str, ...]:
    if explicit is not None:
        if explicit or required:
            if len(explicit) != width:
                raise ValueError(f"{key} explicit order has {len(explicit)} entries, expected {width}")
            return explicit
        return tuple("" for _ in range(width))
    if not required:
        return tuple("" for _ in range(width))
    try:
        names = tuple(str(value) for value in data[key])
    except ValueError as exc:
        raise ValueError(
            f"{key} uses an unsafe object array; declare its native column order in ViconSourceSchema"
        ) from exc
    if len(names) != width:
        raise ValueError(f"{key} has {len(names)} entries, expected {width}")
    return names


@dataclass(frozen=True)
class GvhmrOutputSource:
    """Load existing GVHMR arrays as a native human-pose recording."""

    path: Path
    schema: HumanPoseSourceSchema
    fps: float
    frame: FrameConvention = FrameConvention.Y_UP_RIGHT_HANDED
    name: str = ""
    source_height_m: float | None = None
    load_vertices: bool = False

    def load(self) -> HumanPoseRecording:
        """Load GVHMR joint output without creating a synchronized file."""

        root = Path(self.path)
        joints_path = root / "joints.npy" if root.is_dir() else root
        joints = np.asarray(np.load(joints_path), dtype=np.float64)
        if joints.ndim != 3 or joints.shape[2] != 3:
            raise ValueError("GVHMR joints must have shape (frames, joints, 3)")
        maximum_index = max(self.schema.joint_indices.values())
        if maximum_index >= joints.shape[1]:
            raise ValueError(f"GVHMR schema index {maximum_index} exceeds joint axis {joints.shape[1]}")
        timeline = SampleTimeline.uniform(
            joints.shape[0],
            self.fps,
            clock=f"gvhmr:{joints_path.stem}",
        )
        tracks = tuple(
            JointTrack(
                role=role,
                values=joints[:, index],
                validity=np.asarray(np.isfinite(joints[:, index]).all(axis=1), dtype=bool),
                provenance={"source_index": index},
            )
            for role, index in self.schema.joint_indices.items()
        )
        vertices = None
        vertices_path = joints_path.with_name("vertices.npy")
        if self.load_vertices and vertices_path.exists():
            vertices = np.load(vertices_path, mmap_mode="r")
        return HumanPoseRecording(
            name=self.name or joints_path.parent.name or joints_path.stem,
            timeline=timeline,
            frame=self.frame,
            joints=tracks,
            source_height_m=self.source_height_m,
            vertices=vertices,
            provenance={"source_path": str(joints_path), "source_type": "gvhmr_output"},
        )


@dataclass(frozen=True)
class MocapArraySource:
    """Load a dense ``(frames, joints, 3)`` mocap array."""

    path: Path
    joint_vocabulary: type[MotionJoint]
    fps: float
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    downsample: int = 1
    name: str = ""

    def load(self) -> MocapRecording:
        """Load the array as typed native mocap joint tracks."""

        if self.downsample <= 0:
            raise ValueError("downsample must be positive")
        positions = np.asarray(np.load(self.path), dtype=np.float64)[:: self.downsample]
        joints = tuple(self.joint_vocabulary)
        if positions.ndim != 3 or positions.shape[1:] != (len(joints), 3):
            raise ValueError(f"mocap array must have shape (frames, {len(joints)}, 3)")
        effective_fps = self.fps / self.downsample
        timeline = SampleTimeline.uniform(
            positions.shape[0],
            effective_fps,
            clock=f"mocap:{Path(self.path).stem}",
        )
        return MocapRecording(
            name=self.name or Path(self.path).stem,
            timeline=timeline,
            frame=self.frame,
            joints=tuple(
                JointTrack(
                    role=role,
                    values=positions[:, index],
                    validity=np.asarray(
                        np.isfinite(positions[:, index]).all(axis=1),
                        dtype=bool,
                    ),
                    provenance={"source_index": index},
                )
                for index, role in enumerate(joints)
            ),
            provenance={
                "source_path": str(self.path),
                "source_type": "mocap_array",
                "downsample": self.downsample,
            },
        )
