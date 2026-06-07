"""Canonical target-independent scene observations."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np

from retarget.capture.alignment import AlignmentReport
from retarget.capture.timeline import ClockTransform, SampleTimeline
from retarget.capture.tracks import PointTrack, PoseTrack
from retarget.core.enums import (
    ContactPatch,
    ContactState,
    ContactSubject,
    FrameConvention,
    MocapRigidBody,
    NameEnum,
    ObservationRole,
    QuaternionOrder,
)
from retarget.core.pose import PoseSequence
from retarget.motion.spec import MotionSequence
from retarget.motion.support import SupportPlane
from retarget.observation.contact import SemanticContactSequence, SemanticContactTrack
from retarget.scene.spec import ObjectSpec, TerrainSpec


@dataclass(frozen=True)
class ObservedObject:
    """Observed object geometry and pose, independent of a target robot."""

    role: ObservationRole
    pose: PoseTrack
    geometry: ObjectSpec
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, ObservationRole):
            raise TypeError("observed-object role must be an ObservationRole member")
        if self.geometry.trajectory is not None:
            raise ValueError("observed-object geometry must not contain a trajectory; use pose")
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class SceneObservation:
    """Shared-timeline result of capture processing before robot adaptation."""

    name: str
    timeline: SampleTimeline
    world_frame: FrameConvention
    actor: MotionSequence[Any]
    landmarks: tuple[PointTrack, ...] = ()
    rigid_bodies: tuple[PoseTrack, ...] = ()
    objects: tuple[ObservedObject, ...] = ()
    contacts: SemanticContactSequence | None = None
    terrain: TerrainSpec | None = None
    alignment_reports: tuple[AlignmentReport, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scene observation name must not be empty")
        if not isinstance(self.actor, MotionSequence):
            raise TypeError("scene observation actor must be a MotionSequence")
        if self.actor.timeline != self.timeline:
            raise ValueError("actor must use the scene observation timeline")
        if self.actor.frame != self.world_frame:
            raise ValueError("actor frame must match the scene observation world frame")
        if not all(isinstance(track.role, ObservationRole) for track in self.landmarks):
            raise TypeError("observation landmark roles must be ObservationRole members")
        if not all(isinstance(track.role, MocapRigidBody) for track in self.rigid_bodies):
            raise TypeError("observation rigid-body roles must be MocapRigidBody members")
        for landmark_track in self.landmarks:
            if landmark_track.sample_count != self.timeline.sample_count:
                raise ValueError("observation tracks must match the shared timeline")
        for body_track in self.rigid_bodies:
            if body_track.sample_count != self.timeline.sample_count:
                raise ValueError("observation tracks must match the shared timeline")
        for observed in self.objects:
            if observed.pose.sample_count != self.timeline.sample_count:
                raise ValueError("observed object poses must match the shared timeline")
        if self.contacts is not None and self.contacts.timeline != self.timeline:
            raise ValueError("semantic contacts must use the scene observation timeline")
        landmark_roles = tuple(track.role for track in self.landmarks)
        body_roles = tuple(track.role for track in self.rigid_bodies)
        object_roles = tuple(observed.role for observed in self.objects)
        if len(set(landmark_roles)) != len(landmark_roles):
            raise ValueError("observation landmark roles must be unique")
        if len(set(body_roles)) != len(body_roles):
            raise ValueError("observation rigid body roles must be unique")
        if len(set(object_roles)) != len(object_roles):
            raise ValueError("observation object roles must be unique")
        blocked = {
            "contact_links",
            "joint_mapping",
            "link_mapping",
            "robot",
            "support_plane",
        }
        present = sorted(blocked & set(self.provenance))
        if present:
            raise ValueError(
                "SceneObservation provenance cannot contain behavior: " + ", ".join(present)
            )
        object.__setattr__(self, "provenance", dict(self.provenance))

    def landmark(self, role: ObservationRole) -> PointTrack:
        """Return a semantic landmark track."""

        for track in self.landmarks:
            if track.role == role:
                return track
        raise KeyError(f"Unknown observation landmark role {role.value!r}")

    def observed_object(self, role: ObservationRole) -> ObservedObject:
        """Return an observed object."""

        for observed in self.objects:
            if observed.role == role:
                return observed
        raise KeyError(f"Unknown observation object role {role.value!r}")

    def save_npz(self, path: str | Path) -> Path:
        """Explicitly save this observation checkpoint without pickle."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "manifest_json": json.dumps(self._manifest(), default=_json_default),
            "timeline": self.timeline.timestamps,
        }
        payload["actor_joint_positions"] = self.actor.joint_positions
        if self.actor.root_poses is not None:
            payload["actor_root_positions"] = self.actor.root_poses.positions
            payload["actor_root_quaternions"] = self.actor.root_poses.quaternions()
        for index, landmark_track in enumerate(self.landmarks):
            payload[f"landmark_{index}_values"] = landmark_track.values
            payload[f"landmark_{index}_validity"] = landmark_track.validity
        for prefix, tracks in (
            ("rigid_body", self.rigid_bodies),
            ("object_pose", tuple(observed.pose for observed in self.objects)),
        ):
            for index, pose_track in enumerate(tracks):
                payload[f"{prefix}_{index}_positions"] = pose_track.positions
                payload[f"{prefix}_{index}_quaternions"] = pose_track.quaternions
                payload[f"{prefix}_{index}_validity"] = pose_track.validity
        if self.contacts is not None:
            for index, contact_track in enumerate(self.contacts.tracks):
                payload[f"contact_{index}_states"] = np.asarray(
                    [state.value for state in contact_track.states],
                    dtype=np.str_,
                )
                payload[f"contact_{index}_validity"] = contact_track.validity
        np.savez_compressed(output, **payload)
        return output

    @classmethod
    def load_npz(cls, path: str | Path) -> SceneObservation:
        """Load an explicit observation checkpoint."""

        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            timeline = SampleTimeline(
                data["timeline"],
                clock=str(manifest["timeline"]["clock"]),
            )
            actor_joints = tuple(_load_enum(item) for item in manifest["actor"]["joints"])
            joint_vocabulary = type(actor_joints[0])
            actor = MotionSequence(
                name=str(manifest["actor"]["name"]),
                joint_vocabulary=joint_vocabulary,
                joints=actor_joints,
                root_joint=_load_enum(manifest["actor"]["root_joint"]),
                joint_positions=data["actor_joint_positions"],
                timeline=timeline,
                frame=FrameConvention(manifest["actor"]["frame"]),
                root_poses=(
                    _load_pose_sequence(data, manifest["actor"]["root_pose"], timeline)
                    if manifest["actor"]["root_pose"] is not None
                    else None
                ),
                source_height_m=manifest["actor"]["source_height_m"],
                provenance=dict(manifest["actor"]["provenance"]),
            )
            landmarks = tuple(
                PointTrack(
                    role=_load_observation_role(item["role"]),
                    values=data[f"landmark_{index}_values"],
                    validity=data[f"landmark_{index}_validity"],
                    provenance=dict(item["provenance"]),
                )
                for index, item in enumerate(manifest["landmarks"])
            )
            rigid_bodies = tuple(
                _load_pose_track(data, item, prefix="rigid_body", index=index)
                for index, item in enumerate(manifest["rigid_bodies"])
            )
            objects = tuple(
                ObservedObject(
                    role=_load_observation_role(item["role"]),
                    pose=_load_pose_track(data, item["pose"], prefix="object_pose", index=index),
                    geometry=ObjectSpec.model_validate(item["geometry"]),
                    provenance=dict(item["provenance"]),
                )
                for index, item in enumerate(manifest["objects"])
            )
            contacts = _load_contacts(data, manifest.get("contacts"), timeline)
            return cls(
                name=str(manifest["name"]),
                timeline=timeline,
                world_frame=FrameConvention(manifest["world_frame"]),
                actor=actor,
                landmarks=landmarks,
                rigid_bodies=rigid_bodies,
                objects=objects,
                contacts=contacts,
                terrain=(TerrainSpec.model_validate(manifest["terrain"]) if manifest["terrain"] is not None else None),
                alignment_reports=tuple(_load_alignment_report(item) for item in manifest["alignment_reports"]),
                provenance=dict(manifest["provenance"]),
            )

    def _manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "timeline": {"clock": self.timeline.clock},
            "world_frame": self.world_frame.value,
            "actor": {
                "name": self.actor.name,
                "frame": self.actor.frame.value,
                "source_height_m": self.actor.source_height_m,
                "provenance": self.actor.provenance,
                "root_pose": (
                    _pose_sequence_manifest(self.actor.root_poses)
                    if self.actor.root_poses is not None
                    else None
                ),
                "joints": tuple(_enum_ref(joint) for joint in self.actor.joints),
                "root_joint": _enum_ref(self.actor.root_joint),
            },
            "landmarks": tuple(
                {"role": _enum_ref(track.role), "provenance": track.provenance} for track in self.landmarks
            ),
            "rigid_bodies": tuple(_pose_manifest(track) for track in self.rigid_bodies),
            "objects": tuple(
                {
                    "role": _enum_ref(observed.role),
                    "pose": _pose_manifest(observed.pose),
                    "geometry": observed.geometry.model_dump(),
                    "provenance": observed.provenance,
                }
                for observed in self.objects
            ),
            "contacts": _contacts_manifest(self.contacts),
            "terrain": self.terrain.model_dump() if self.terrain is not None else None,
            "alignment_reports": tuple(_alignment_manifest(report) for report in self.alignment_reports),
            "provenance": self.provenance,
        }


def _enum_ref(value: NameEnum) -> dict[str, str]:
    enum_type = type(value)
    return {
        "module": enum_type.__module__,
        "qualname": enum_type.__qualname__,
        "value": value.value,
    }


def _load_enum(reference: dict[str, str]) -> NameEnum:
    module = importlib.import_module(reference["module"])
    enum_type: object = module
    for segment in reference["qualname"].split("."):
        enum_type = getattr(enum_type, segment)
    if not isinstance(enum_type, type) or not issubclass(enum_type, NameEnum):
        raise TypeError(f"{reference['qualname']} is not a NameEnum type")
    return enum_type(reference["value"])


def _load_observation_role(reference: dict[str, str]) -> ObservationRole:
    return cast(ObservationRole, _load_enum(reference))


def _load_contact_subject(reference: dict[str, str]) -> ContactSubject:
    return cast(ContactSubject, _load_enum(reference))


def _load_contact_patch(reference: dict[str, str]) -> ContactPatch:
    return cast(ContactPatch, _load_enum(reference))


def _pose_manifest(track: PoseTrack) -> dict[str, object]:
    return {
        "role": _enum_ref(track.role),
        "quaternion_order": track.quaternion_order.value,
        "provenance": track.provenance,
    }


def _pose_sequence_manifest(sequence: PoseSequence) -> dict[str, object]:
    return {
        "fps": sequence.fps,
        "frame": sequence.frame.value,
        "quaternion_order": QuaternionOrder.WXYZ.value,
    }


def _load_pose_sequence(
    data: Any,
    item: dict[str, Any],
    timeline: SampleTimeline,
) -> PoseSequence:
    fps = timeline.nominal_fps
    if fps is None:
        fps = float(item["fps"])
    return PoseSequence.from_arrays(
        data["actor_root_positions"],
        data["actor_root_quaternions"],
        fps=fps,
        quaternion_order=QuaternionOrder(item["quaternion_order"]),
        frame=FrameConvention(item["frame"]),
    )


def _load_pose_track(
    data: Any,
    item: dict[str, Any],
    *,
    prefix: str,
    index: int,
) -> PoseTrack:
    return PoseTrack(
        role=_load_enum(item["role"]),
        positions=data[f"{prefix}_{index}_positions"],
        quaternions=data[f"{prefix}_{index}_quaternions"],
        quaternion_order=QuaternionOrder(item["quaternion_order"]),
        validity=data[f"{prefix}_{index}_validity"],
        provenance=dict(item["provenance"]),
    )


def _contacts_manifest(contacts: SemanticContactSequence | None) -> dict[str, object] | None:
    if contacts is None:
        return None
    return {
        "support": (
            {
                "normal": contacts.support.normal,
                "origin": contacts.support.origin,
                "up_axis": contacts.support.up_axis,
            }
            if contacts.support is not None
            else None
        ),
        "provenance": contacts.provenance,
        "tracks": tuple(
            {
                "subject": _enum_ref(track.subject),
                "patch": _enum_ref(track.patch) if track.patch is not None else None,
                "state_type": {
                    "module": type(track.states[0]).__module__,
                    "qualname": type(track.states[0]).__qualname__,
                },
                "active_states": tuple(state.value for state in track.active_states),
                "support_states": tuple(state.value for state in track.support_states),
                "provenance": track.provenance,
            }
            for track in contacts.tracks
        ),
    }


def _load_contacts(
    data: Any,
    item: dict[str, Any] | None,
    timeline: SampleTimeline,
) -> SemanticContactSequence | None:
    if item is None:
        return None
    tracks: list[SemanticContactTrack] = []
    for index, track_item in enumerate(item["tracks"]):
        state_reference = dict(track_item["state_type"])
        module = importlib.import_module(state_reference["module"])
        state_type: object = module
        for segment in state_reference["qualname"].split("."):
            state_type = getattr(state_type, segment)
        if not isinstance(state_type, type) or not issubclass(state_type, ContactState):
            raise TypeError(f"{state_reference['qualname']} is not a ContactState type")
        tracks.append(
            SemanticContactTrack(
                subject=_load_contact_subject(track_item["subject"]),
                patch=(_load_contact_patch(track_item["patch"]) if track_item["patch"] is not None else None),
                states=tuple(state_type(str(value)) for value in data[f"contact_{index}_states"]),
                active_states=tuple(state_type(value) for value in track_item["active_states"]),
                support_states=tuple(state_type(value) for value in track_item["support_states"]),
                validity=data[f"contact_{index}_validity"],
                provenance=dict(track_item["provenance"]),
            )
        )
    support = SupportPlane(**item["support"]) if item["support"] is not None else None
    return SemanticContactSequence(
        timeline=timeline,
        tracks=tuple(tracks),
        support=support,
        provenance=dict(item["provenance"]),
    )


def _alignment_manifest(report: AlignmentReport) -> dict[str, object]:
    transform = report.clock_transform
    return {
        "strategy": report.strategy,
        "accepted": report.accepted,
        "score": report.score,
        "sample_count": report.sample_count,
        "overlap_s": report.overlap_s,
        "rms_error": report.rms_error,
        "clock_transform": (
            {
                "scale": transform.scale,
                "offset_s": transform.offset_s,
                "source_clock": transform.source_clock,
                "target_clock": transform.target_clock,
            }
            if transform is not None
            else None
        ),
        "diagnostics": report.diagnostics,
    }


def _load_alignment_report(item: dict[str, Any]) -> AlignmentReport:
    transform_data = item["clock_transform"]
    return AlignmentReport(
        strategy=str(item["strategy"]),
        accepted=bool(item["accepted"]),
        score=float(item["score"]),
        sample_count=int(item["sample_count"]),
        overlap_s=item["overlap_s"],
        rms_error=item["rms_error"],
        clock_transform=ClockTransform(**transform_data) if transform_data is not None else None,
        diagnostics=dict(item["diagnostics"]),
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
