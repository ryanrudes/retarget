"""Source schemas and semantic mappings for skateboarding."""

from __future__ import annotations

from retarget.capture import HumanPoseSourceSchema, ViconSourceSchema
from retarget.core.enums import HumanoidRobotRole

from .vocabulary import (
    SkateboardingMotionJoint,
    SkateboardingObservationRole,
    SkateboardingRigidBody,
)

VICON_SCHEMA = ViconSourceSchema(
    rigid_bodies={body.value: body for body in SkateboardingRigidBody},
    markers={},
    rigid_body_order=tuple(body.value for body in SkateboardingRigidBody),
    marker_order=(),
)

GVHMR_SCHEMA = HumanPoseSourceSchema(
    joint_indices={
        SkateboardingMotionJoint.PELVIS: 0,
        SkateboardingMotionJoint.LEFT_HIP: 1,
        SkateboardingMotionJoint.RIGHT_HIP: 2,
        SkateboardingMotionJoint.LEFT_KNEE: 4,
        SkateboardingMotionJoint.RIGHT_KNEE: 5,
        SkateboardingMotionJoint.SPINE2: 6,
        SkateboardingMotionJoint.LEFT_ANKLE: 7,
        SkateboardingMotionJoint.RIGHT_ANKLE: 8,
        SkateboardingMotionJoint.SPINE3: 9,
        SkateboardingMotionJoint.LEFT_FOOT: 10,
        SkateboardingMotionJoint.RIGHT_FOOT: 11,
        SkateboardingMotionJoint.NECK: 12,
        SkateboardingMotionJoint.HEAD: 15,
        SkateboardingMotionJoint.LEFT_SHOULDER: 16,
        SkateboardingMotionJoint.RIGHT_SHOULDER: 17,
        SkateboardingMotionJoint.LEFT_ELBOW: 18,
        SkateboardingMotionJoint.RIGHT_ELBOW: 19,
        SkateboardingMotionJoint.LEFT_WRIST: 20,
        SkateboardingMotionJoint.RIGHT_WRIST: 21,
        SkateboardingMotionJoint.LEFT_BIG_TOE: 60,
        SkateboardingMotionJoint.LEFT_SMALL_TOE: 61,
        SkateboardingMotionJoint.LEFT_HEEL: 62,
        SkateboardingMotionJoint.RIGHT_BIG_TOE: 63,
        SkateboardingMotionJoint.RIGHT_SMALL_TOE: 64,
        SkateboardingMotionJoint.RIGHT_HEEL: 65,
    }
)

LANDMARK_JOINTS = {
    SkateboardingObservationRole.PELVIS: SkateboardingMotionJoint.PELVIS,
    SkateboardingObservationRole.LEFT_HIP: SkateboardingMotionJoint.LEFT_HIP,
    SkateboardingObservationRole.RIGHT_HIP: SkateboardingMotionJoint.RIGHT_HIP,
    SkateboardingObservationRole.LEFT_KNEE: SkateboardingMotionJoint.LEFT_KNEE,
    SkateboardingObservationRole.RIGHT_KNEE: SkateboardingMotionJoint.RIGHT_KNEE,
    SkateboardingObservationRole.LEFT_ANKLE: SkateboardingMotionJoint.LEFT_ANKLE,
    SkateboardingObservationRole.RIGHT_ANKLE: SkateboardingMotionJoint.RIGHT_ANKLE,
    SkateboardingObservationRole.LEFT_FOOT: SkateboardingMotionJoint.LEFT_FOOT,
    SkateboardingObservationRole.RIGHT_FOOT: SkateboardingMotionJoint.RIGHT_FOOT,
}

OBSERVATION_TO_ROBOT_ROLE = {
    SkateboardingObservationRole.PELVIS: HumanoidRobotRole.PELVIS,
    SkateboardingObservationRole.LEFT_HIP: HumanoidRobotRole.LEFT_HIP,
    SkateboardingObservationRole.RIGHT_HIP: HumanoidRobotRole.RIGHT_HIP,
    SkateboardingObservationRole.LEFT_KNEE: HumanoidRobotRole.LEFT_KNEE,
    SkateboardingObservationRole.RIGHT_KNEE: HumanoidRobotRole.RIGHT_KNEE,
    SkateboardingObservationRole.LEFT_ANKLE: HumanoidRobotRole.LEFT_ANKLE,
    SkateboardingObservationRole.RIGHT_ANKLE: HumanoidRobotRole.RIGHT_ANKLE,
    SkateboardingObservationRole.LEFT_FOOT: HumanoidRobotRole.LEFT_FOOT,
    SkateboardingObservationRole.RIGHT_FOOT: HumanoidRobotRole.RIGHT_FOOT,
    SkateboardingObservationRole.UPPER_BODY_CENTER: HumanoidRobotRole.TORSO,
}

JOINT_TO_ROBOT_ROLE = {
    SkateboardingMotionJoint.LEFT_HIP: HumanoidRobotRole.LEFT_HIP,
    SkateboardingMotionJoint.RIGHT_HIP: HumanoidRobotRole.RIGHT_HIP,
    SkateboardingMotionJoint.LEFT_KNEE: HumanoidRobotRole.LEFT_KNEE,
    SkateboardingMotionJoint.RIGHT_KNEE: HumanoidRobotRole.RIGHT_KNEE,
    SkateboardingMotionJoint.LEFT_ANKLE: HumanoidRobotRole.LEFT_ANKLE,
    SkateboardingMotionJoint.RIGHT_ANKLE: HumanoidRobotRole.RIGHT_ANKLE,
    SkateboardingMotionJoint.SPINE3: HumanoidRobotRole.TORSO,
    SkateboardingMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
    SkateboardingMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
}

LINK_TO_ROBOT_ROLE = {
    SkateboardingMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
    SkateboardingMotionJoint.LEFT_HIP: HumanoidRobotRole.LEFT_HIP,
    SkateboardingMotionJoint.RIGHT_HIP: HumanoidRobotRole.RIGHT_HIP,
    SkateboardingMotionJoint.LEFT_KNEE: HumanoidRobotRole.LEFT_KNEE,
    SkateboardingMotionJoint.RIGHT_KNEE: HumanoidRobotRole.RIGHT_KNEE,
    SkateboardingMotionJoint.LEFT_FOOT: HumanoidRobotRole.LEFT_FOOT,
    SkateboardingMotionJoint.RIGHT_FOOT: HumanoidRobotRole.RIGHT_FOOT,
    SkateboardingMotionJoint.SPINE3: HumanoidRobotRole.TORSO,
    SkateboardingMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
    SkateboardingMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
}
