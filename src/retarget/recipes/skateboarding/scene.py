"""Target-independent and runtime scene construction for skateboarding."""

from __future__ import annotations

import numpy as np

from retarget.capture import HumanPoseRecording, PointTrack, PoseTrack, SampleTimeline
from retarget.core.enums import ObjectSampleSpace
from retarget.core.pose import PoseSequence
from retarget.observation import (
    FootSupportClassificationConfig,
    FootSupportStates,
    ObservedObject,
    SceneObservation,
    SemanticContactSequence,
    classify_foot_support,
)
from retarget.scene import ObjectSpec, ObjectTrajectory, SceneSpec

from .schema import LANDMARK_JOINTS
from .vocabulary import (
    DECK_SAMPLE_POINTS,
    SkateboardingContactPatch,
    SkateboardingContactState,
    SkateboardingContactSubject,
    SkateboardingGeometryName,
    SkateboardingMotionJoint,
    SkateboardingObservationRole,
)

UPPER_BODY_JOINTS = (
    SkateboardingMotionJoint.SPINE2,
    SkateboardingMotionJoint.SPINE3,
    SkateboardingMotionJoint.NECK,
    SkateboardingMotionJoint.HEAD,
    SkateboardingMotionJoint.LEFT_SHOULDER,
    SkateboardingMotionJoint.RIGHT_SHOULDER,
    SkateboardingMotionJoint.LEFT_ELBOW,
    SkateboardingMotionJoint.RIGHT_ELBOW,
    SkateboardingMotionJoint.LEFT_WRIST,
    SkateboardingMotionJoint.RIGHT_WRIST,
)


def classify_skateboarding_contacts(
    *,
    timeline: SampleTimeline,
    left_shoe: PoseTrack,
    right_shoe: PoseTrack,
    board: PoseTrack,
    config: FootSupportClassificationConfig,
) -> SemanticContactSequence:
    """Classify shoe support without resolving any robot links."""

    return classify_foot_support(
        timeline=timeline,
        left_foot=PointTrack(
            role=SkateboardingContactSubject.LEFT_SHOE,
            values=left_shoe.positions,
            validity=left_shoe.validity,
        ),
        right_foot=PointTrack(
            role=SkateboardingContactSubject.RIGHT_SHOE,
            values=right_shoe.positions,
            validity=right_shoe.validity,
        ),
        observed_object=PointTrack(
            role=SkateboardingObservationRole.BOARD,
            values=board.positions,
            validity=board.validity,
        ),
        left_subject=SkateboardingContactSubject.LEFT_SHOE,
        right_subject=SkateboardingContactSubject.RIGHT_SHOE,
        left_patch=SkateboardingContactPatch.LEFT_SHOE_SOLE,
        right_patch=SkateboardingContactPatch.RIGHT_SHOE_SOLE,
        states=FootSupportStates(
            air=SkateboardingContactState.AIR,
            ground=SkateboardingContactState.GROUND,
            observed_object=SkateboardingContactState.BOARD,
        ),
        config=config,
    )


def semantic_landmarks(actor: HumanPoseRecording) -> tuple[PointTrack, ...]:
    """Derive target-independent actor landmarks."""

    tracks = [
        PointTrack(
            role=role,
            values=actor.joint(joint).values,
            validity=actor.joint(joint).validity,
            provenance={"source_joint": joint.value},
        )
        for role, joint in LANDMARK_JOINTS.items()
    ]
    upper = np.mean(
        np.stack(
            [actor.joint(joint).values for joint in UPPER_BODY_JOINTS],
            axis=0,
        ),
        axis=0,
    )
    upper_validity = np.logical_and.reduce(
        [np.asarray(actor.joint(joint).validity, dtype=bool) for joint in UPPER_BODY_JOINTS]
    )
    tracks.append(
        PointTrack(
            role=SkateboardingObservationRole.UPPER_BODY_CENTER,
            values=upper,
            validity=upper_validity,
            provenance={"source_joints": tuple(joint.value for joint in UPPER_BODY_JOINTS)},
        )
    )
    return tuple(tracks)


def observed_board(board: PoseTrack) -> ObservedObject:
    """Build the target-independent observed skateboard."""

    return ObservedObject(
        role=SkateboardingObservationRole.BOARD,
        pose=PoseTrack(
            role=SkateboardingObservationRole.BOARD,
            positions=board.positions,
            quaternions=board.quaternions,
            quaternion_order=board.quaternion_order,
            validity=board.validity,
            provenance=board.provenance,
        ),
        geometry=ObjectSpec(
            name=SkateboardingGeometryName.BOARD.value,
            sample_points=DECK_SAMPLE_POINTS,
            sample_space=ObjectSampleSpace.OBJECT_LOCAL,
        ),
        provenance={"source": "mocap_rigid_body"},
    )


def runtime_scene(
    observation: SceneObservation,
    *,
    fps: float,
) -> SceneSpec:
    """Resolve observed board motion into a runtime object trajectory."""

    board = observation.observed_object(SkateboardingObservationRole.BOARD)
    return SceneSpec.object_interaction(
        board.geometry.model_copy(
            update={
                "trajectory": ObjectTrajectory(
                    name=board.geometry.name,
                    poses=PoseSequence.from_arrays(
                        board.pose.positions,
                        board.pose.quaternions,
                        fps=fps,
                        quaternion_order=board.pose.quaternion_order,
                        frame=observation.world_frame,
                    ),
                )
            }
        )
    ).model_copy(update={"ground_range": (-3.0, 3.0), "ground_size": 15})
