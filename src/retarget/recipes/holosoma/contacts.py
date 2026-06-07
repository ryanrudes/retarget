"""Semantic contact extraction for Holosoma climbing observations."""

from __future__ import annotations

import numpy as np

from retarget.capture import SampleTimeline
from retarget.motion.support import SupportPlane
from retarget.observation import SemanticContactSequence, SemanticContactTrack

from .vocabulary import (
    HolosomaContactPatch,
    HolosomaContactState,
    HolosomaContactSubject,
    HolosomaMocapJoint,
)

_MOCAP_JOINT_NAMES = tuple(joint.value for joint in HolosomaMocapJoint)


def foot_sticking_contacts(
    timeline: SampleTimeline,
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...] = _MOCAP_JOINT_NAMES,
    velocity_threshold: float = 0.01,
) -> SemanticContactSequence:
    """Extract target-independent toe sticking states."""

    left_states, right_states = foot_sticking_states(
        human_joints,
        demo_joints=demo_joints,
        velocity_threshold=velocity_threshold,
    )
    return SemanticContactSequence(
        timeline=timeline,
        tracks=(
            SemanticContactTrack(
                subject=HolosomaContactSubject.LEFT_FOOT,
                patch=HolosomaContactPatch.LEFT_TOE,
                states=tuple(
                    HolosomaContactState.STICKING if state else HolosomaContactState.AIR for state in left_states
                ),
                active_states=(HolosomaContactState.STICKING,),
                support_states=(HolosomaContactState.STICKING,),
                provenance={"velocity_threshold": velocity_threshold},
            ),
            SemanticContactTrack(
                subject=HolosomaContactSubject.RIGHT_FOOT,
                patch=HolosomaContactPatch.RIGHT_TOE,
                states=tuple(
                    HolosomaContactState.STICKING if state else HolosomaContactState.AIR for state in right_states
                ),
                active_states=(HolosomaContactState.STICKING,),
                support_states=(HolosomaContactState.STICKING,),
                provenance={"velocity_threshold": velocity_threshold},
            ),
        ),
        support=SupportPlane(
            normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            origin=np.zeros(3, dtype=np.float64),
        ),
        provenance={"source": "toe_velocity"},
    )


def foot_sticking_states(
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...],
    velocity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary foot-sticking states from toe XY displacement."""

    joints = np.asarray(human_joints, dtype=np.float64)
    left_idx = demo_joints.index(HolosomaMocapJoint.LEFT_TOE_BASE.value)
    right_idx = demo_joints.index(HolosomaMocapJoint.RIGHT_TOE_BASE.value)
    left_velocity = np.linalg.norm(np.diff(joints[:, left_idx, :2], axis=0), axis=1)
    right_velocity = np.linalg.norm(np.diff(joints[:, right_idx, :2], axis=0), axis=1)
    left_velocity = np.concatenate([[velocity_threshold + 1.0], left_velocity])
    right_velocity = np.concatenate([[velocity_threshold + 1.0], right_velocity])
    return left_velocity <= velocity_threshold, right_velocity <= velocity_threshold
