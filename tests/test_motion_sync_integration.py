from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retarget import LinkTargetPlan, TaskKind
from retarget.integrations.motion_sync import contact_plan_from_sync_clip, from_sync_clip


@dataclass(frozen=True)
class FakeContactLayer:
    layer_id: str = "foot_support"
    kind: str = "categorical"
    subjects: tuple[str, ...] = ("left_shoe", "right_shoe")
    labels: tuple[str, ...] = ("air", "ground", "skateboard")
    states: np.ndarray = field(
        default_factory=lambda: np.asarray([[1, 0], [2, 1], [0, 2]], dtype=np.int8)
    )
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(
        default_factory=lambda: {
            "floor_model": "plane",
            "floor_normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "floor_origin": np.asarray([0.0, 0.0, 0.1], dtype=np.float64),
            "config_hash": "cfg123",
            "time_fingerprint": "time123",
        }
    )

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class FakeClip:
    name: str = "fake_clip"
    time_s: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 0.1, 0.2], dtype=np.float64))


def test_contact_plan_from_sync_clip_reads_categorical_layer() -> None:
    clip = FakeClip()
    layer = FakeContactLayer()

    plan = contact_plan_from_sync_clip(
        clip,
        contact_layer=layer,
        contact_link_mapping={"left_shoe": "left_toe", "right_shoe": "right_toe"},
    )

    assert plan is not None
    assert plan.frame_count == 3
    assert plan.frame(0).active_link_names == ("left_toe",)
    assert plan.frame(0).support_link_names == ("left_toe",)
    assert plan.frame(1).active_link_names == ("left_toe", "right_toe")
    assert plan.frame(1).support_link_names == ("right_toe",)
    assert plan.support is not None
    assert np.allclose(plan.support.origin, [0.0, 0.0, 0.1])
    assert plan.provenance["config_hash"] == "cfg123"
    assert plan.provenance["time_fingerprint"] == "time123"


def test_from_sync_clip_returns_motion_scene_contacts_and_metadata() -> None:
    clip = FakeClip()
    layer = FakeContactLayer()
    joint_positions = np.zeros((3, 2, 3), dtype=np.float64)
    object_positions = np.zeros((3, 3), dtype=np.float64)
    object_quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float64)

    targets = LinkTargetPlan.from_arrays(
        link_names=("left_toe",),
        positions=np.zeros((3, 1, 3), dtype=np.float64),
    )

    prepared = from_sync_clip(
        clip,
        joint_names=("Pelvis", "L_Foot"),
        joint_positions=joint_positions,
        height_m=1.8,
        targets=targets,
        contact_layer=layer,
        contact_link_mapping={"left_shoe": "left_toe", "right_shoe": "right_toe"},
        object_name="board",
        object_positions=object_positions,
        object_quaternions=object_quaternions,
        object_sample_points=np.zeros((4, 3), dtype=np.float64),
    )

    assert prepared.motion.name == "fake_clip"
    assert prepared.motion.metadata["height_m"] == 1.8
    assert prepared.targets is not None
    assert prepared.targets.link_names == ("left_toe",)
    assert prepared.contacts is not None
    assert prepared.scene.task_kind == TaskKind.OBJECT_INTERACTION
    assert prepared.scene.object is not None
    assert prepared.scene.object.name == "board"
    assert prepared.scene.object.trajectory is not None
    assert prepared.metadata["source"] == "motion_sync"
