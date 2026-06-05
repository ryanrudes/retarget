"""Adapters from ``motion_sync`` clips into retargeting inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from retarget.core.enums import FrameConvention, QuaternionOrder, TaskKind
from retarget.core.pose import PoseSequence
from retarget.motion.contact import ContactPlan, ContactTrack, SupportPlane
from retarget.motion.qpos import NominalQposPlan
from retarget.motion.spec import MotionSequence
from retarget.motion.targets import LinkTargetPlan
from retarget.scene.spec import ObjectSpec, ObjectTrajectory, SceneSpec, TerrainSpec

__all__ = [
    "PreparedRetargetInputs",
    "contact_plan_from_sync_clip",
    "from_sync_clip",
    "scene_from_sync_clip",
]


@dataclass(frozen=True)
class PreparedRetargetInputs:
    """Retarget-ready bundle derived from a synchronized capture clip."""

    motion: MotionSequence
    scene: SceneSpec
    contacts: ContactPlan | None = None
    targets: LinkTargetPlan | None = None
    nominal_qpos: NominalQposPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def from_sync_clip(
    clip: Any,
    *,
    joint_names: Sequence[str],
    joint_positions: ArrayLike,
    name: str | None = None,
    fps: float | None = None,
    frame: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED,
    root_positions: ArrayLike | None = None,
    root_quaternions: ArrayLike | None = None,
    root_quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ,
    height_m: float | None = None,
    targets: LinkTargetPlan | None = None,
    nominal_qpos: NominalQposPlan | None = None,
    contact_type: Any | None = None,
    contact_layer: Any | None = None,
    contact_link_mapping: Mapping[str, Sequence[str] | str] | None = None,
    support: SupportPlane | None = None,
    object_name: str | None = None,
    object_positions: ArrayLike | None = None,
    object_quaternions: ArrayLike | None = None,
    object_quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ,
    object_sample_points: ArrayLike | None = None,
    terrain: TerrainSpec | None = None,
    task_kind: TaskKind | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PreparedRetargetInputs:
    """Build retargeting inputs from a synchronized capture clip."""

    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("joint_positions must have shape (frames, joints, 3)")
    names = tuple(str(joint_name) for joint_name in joint_names)
    if len(names) != positions.shape[1]:
        raise ValueError("joint_names length must match joint_positions")
    clip_fps = _clip_fps(clip) if fps is None else float(fps)
    motion_metadata: dict[str, Any] = dict(metadata or {})
    if height_m is not None:
        motion_metadata["height_m"] = float(height_m)

    root_poses = None
    if root_positions is not None or root_quaternions is not None:
        if root_positions is None or root_quaternions is None:
            raise ValueError("root_positions and root_quaternions must be provided together")
        root_poses = PoseSequence.from_arrays(
            root_positions,
            root_quaternions,
            fps=clip_fps,
            quaternion_order=root_quaternion_order,
            frame=frame,
        )

    motion = MotionSequence(
        name=name or getattr(clip, "name", "") or "sync_clip",
        joint_positions=positions,
        joint_names=names,
        fps=clip_fps,
        frame=frame,
        root_poses=root_poses,
        metadata=motion_metadata,
    )
    contact_plan = contact_plan_from_sync_clip(
        clip,
        contact_type=contact_type,
        contact_layer=contact_layer,
        contact_link_mapping=contact_link_mapping,
        support=support,
        frame_count=positions.shape[0],
    )
    if targets is not None and targets.frame_count != positions.shape[0]:
        raise ValueError("targets frame count must match joint_positions")
    if nominal_qpos is not None and nominal_qpos.frame_count != positions.shape[0]:
        raise ValueError("nominal_qpos frame count must match joint_positions")

    scene = scene_from_sync_clip(
        frame_count=positions.shape[0],
        fps=clip_fps,
        object_name=object_name,
        object_positions=object_positions,
        object_quaternions=object_quaternions,
        object_quaternion_order=object_quaternion_order,
        object_sample_points=object_sample_points,
        terrain=terrain,
        task_kind=task_kind,
    )
    return PreparedRetargetInputs(
        motion=motion,
        scene=scene,
        contacts=contact_plan,
        targets=targets,
        nominal_qpos=nominal_qpos,
        metadata={
            "source": "motion_sync",
            "clip_name": getattr(clip, "name", None),
            "frame_count": int(positions.shape[0]),
            "fps": clip_fps,
        },
    )


def contact_plan_from_sync_clip(
    clip: Any,
    *,
    contact_type: Any | None = None,
    contact_layer: Any | None = None,
    contact_link_mapping: Mapping[str, Sequence[str] | str] | None = None,
    support: SupportPlane | None = None,
    frame_count: int | None = None,
) -> ContactPlan | None:
    """Read a persisted ``motion_sync`` contact layer as a typed contact plan."""

    layer = contact_layer
    if layer is None and contact_type is not None:
        data = clip.contact(contact_type)
        layer = getattr(data, "layer", None)
    if layer is None:
        return None
    if frame_count is not None and int(layer.frame_count) != int(frame_count):
        raise ValueError(f"contact layer has {layer.frame_count} frames, expected {frame_count}")
    support = support or _support_from_layer(layer)
    mapping = dict(contact_link_mapping or {})
    tracks = tuple(_tracks_from_layer(layer, mapping))
    if not tracks:
        return None
    return ContactPlan(
        tracks=tracks,
        frame_count=int(layer.frame_count),
        support=support,
        provenance=_layer_provenance(layer),
    )


def scene_from_sync_clip(
    *,
    frame_count: int,
    fps: float,
    object_name: str | None = None,
    object_positions: ArrayLike | None = None,
    object_quaternions: ArrayLike | None = None,
    object_quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ,
    object_sample_points: ArrayLike | None = None,
    terrain: TerrainSpec | None = None,
    task_kind: TaskKind | None = None,
) -> SceneSpec:
    """Build a scene from optional synchronized object trajectory arrays."""

    object_spec = None
    if object_positions is not None or object_quaternions is not None or object_sample_points is not None:
        if object_positions is None or object_quaternions is None:
            raise ValueError("object_positions and object_quaternions must be provided together")
        object_spec = ObjectSpec(
            name=object_name or "object",
            sample_points=None if object_sample_points is None else np.asarray(object_sample_points, dtype=np.float64),
            trajectory=ObjectTrajectory(
                name=object_name or "object",
                poses=PoseSequence.from_arrays(
                    object_positions,
                    object_quaternions,
                    fps=fps,
                    quaternion_order=object_quaternion_order,
                    frame=FrameConvention.Z_UP_RIGHT_HANDED,
                ),
            ),
        )
    resolved_kind = task_kind or (TaskKind.OBJECT_INTERACTION if object_spec is not None else TaskKind.ROBOT_ONLY)
    if resolved_kind == TaskKind.OBJECT_INTERACTION:
        return SceneSpec(task_kind=resolved_kind, object=object_spec or ObjectSpec(name=object_name or "object"))
    return SceneSpec(task_kind=resolved_kind, terrain=terrain or TerrainSpec())


def _tracks_from_layer(
    layer: Any,
    link_mapping: Mapping[str, Sequence[str] | str],
) -> list[ContactTrack]:
    subjects = tuple(str(subject) for subject in layer.subjects)
    if layer.kind == "binary":
        values = np.asarray(layer.mask, dtype=bool)
        return [
            ContactTrack(
                subject=subject,
                states=values[:, idx].astype(np.int16),
                link_names=_link_names(link_mapping.get(subject, ())),
                active_states=(1,),
                support_states=(1,),
                labels=("air", "contact"),
                metadata={"layer_id": layer.layer_id},
            )
            for idx, subject in enumerate(subjects)
        ]
    if layer.kind != "categorical":
        raise ValueError(f"unsupported contact layer kind {layer.kind!r}")
    values = np.asarray(layer.states, dtype=np.int16)
    labels = tuple(str(label).lower() for label in layer.labels)
    active_states = tuple(
        idx for idx, label in enumerate(labels) if label not in {"air", "none", "off", "unknown"}
    ) or tuple(int(value) for value in np.unique(values) if int(value) != 0)
    support_states = tuple(
        idx for idx, label in enumerate(labels) if label in {"ground", "floor", "support", "contact"}
    ) or active_states
    return [
        ContactTrack(
            subject=subject,
            states=values[:, idx],
            link_names=_link_names(link_mapping.get(subject, ())),
            active_states=active_states,
            support_states=support_states,
            labels=labels,
            metadata={"layer_id": layer.layer_id},
        )
        for idx, subject in enumerate(subjects)
    ]


def _support_from_layer(layer: Any) -> SupportPlane | None:
    metadata = getattr(layer, "metadata", {})
    normal = metadata.get("floor_normal")
    origin = metadata.get("floor_origin")
    if normal is None:
        normal = (0.0, 0.0, 1.0)
    if origin is None and "floor_height" in metadata:
        origin = (0.0, 0.0, float(metadata["floor_height"]))
    if origin is None:
        return None
    return SupportPlane(normal=np.asarray(normal, dtype=np.float64), origin=np.asarray(origin, dtype=np.float64))


def _layer_provenance(layer: Any) -> dict[str, Any]:
    metadata = dict(getattr(layer, "metadata", {}))
    out = {
        "source": "motion_sync.contact_layer",
        "layer_id": getattr(layer, "layer_id", None),
        "kind": getattr(layer, "kind", None),
        "subjects": list(getattr(layer, "subjects", ())),
        "labels": list(getattr(layer, "labels", ())),
        "source_frame_count": int(getattr(layer, "frame_count", 0)),
    }
    for key in (
        "floor_model",
        "floor_height",
        "detector",
        "config_hash",
        "fingerprint",
        "time_fingerprint",
        "timeline_fingerprint",
    ):
        if key in metadata:
            out[key] = metadata[key]
    return out


def _clip_fps(clip: Any) -> float:
    time_s = np.asarray(getattr(clip, "time_s", ()), dtype=np.float64)
    if time_s.shape[0] < 2:
        return 30.0
    deltas = np.diff(time_s[np.isfinite(time_s)])
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        return 30.0
    return float(1.0 / np.mean(positive))


def _link_names(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(name) for name in value)
