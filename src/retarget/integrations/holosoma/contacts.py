"""Contact extraction for Holosoma-compatible recipes."""

from __future__ import annotations

import numpy as np

from retarget.motion.contact import ContactPlan, ContactTrack
from retarget.motion.support import SupportPlane

from .vocabulary import (
    FOOT_STICKING_VELOCITY_THRESHOLD,
    G1_LEFT_FOOT_STICKING_LINKS,
    G1_RIGHT_FOOT_STICKING_LINKS,
    MOCAP_DEMO_JOINTS,
    HolosomaContactState,
    HolosomaMocapJoint,
)


def foot_sticking_contact_plan(
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...] = MOCAP_DEMO_JOINTS,
    velocity_threshold: float = FOOT_STICKING_VELOCITY_THRESHOLD,
) -> ContactPlan:
    """Extract Holosoma's toe-velocity foot sticking contact plan."""

    left_states, right_states = foot_sticking_states(
        human_joints,
        demo_joints=demo_joints,
        velocity_threshold=velocity_threshold,
    )
    return ContactPlan(
        tracks=(
            ContactTrack(
                subject=HolosomaMocapJoint.LEFT_TOE_BASE.value,
                states=left_states,
                link_names=G1_LEFT_FOOT_STICKING_LINKS,
                active_states=(1,),
                support_states=(1,),
                labels=(HolosomaContactState.AIR.value, HolosomaContactState.STICKING.value),
                metadata={"source": "holosoma_velocity", "velocity_threshold": velocity_threshold},
            ),
            ContactTrack(
                subject=HolosomaMocapJoint.RIGHT_TOE_BASE.value,
                states=right_states,
                link_names=G1_RIGHT_FOOT_STICKING_LINKS,
                active_states=(1,),
                support_states=(1,),
                labels=(HolosomaContactState.AIR.value, HolosomaContactState.STICKING.value),
                metadata={"source": "holosoma_velocity", "velocity_threshold": velocity_threshold},
            ),
        ),
        frame_count=int(np.asarray(human_joints).shape[0]),
        support=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3)),
        provenance={"source": "holosoma.extract_foot_sticking_sequence_velocity"},
    )


def foot_sticking_states(
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...],
    velocity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary foot-sticking states from toe XY velocity."""

    joints = np.asarray(human_joints, dtype=np.float64)
    left_idx = demo_joints.index(HolosomaMocapJoint.LEFT_TOE_BASE.value)
    right_idx = demo_joints.index(HolosomaMocapJoint.RIGHT_TOE_BASE.value)
    left_vel = np.linalg.norm(np.diff(joints[:, left_idx, :2], axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(joints[:, right_idx, :2], axis=0), axis=1)
    left_vel = np.concatenate([[velocity_threshold + 1.0], left_vel])
    right_vel = np.concatenate([[velocity_threshold + 1.0], right_vel])
    return (
        (left_vel <= velocity_threshold).astype(np.int16),
        (right_vel <= velocity_threshold).astype(np.int16),
    )
