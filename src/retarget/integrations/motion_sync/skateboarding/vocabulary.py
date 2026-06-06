"""Typed vocabulary for the skateboarding recipe."""

from __future__ import annotations

import numpy as np

from retarget.core.enums import (
    ContactPatch,
    ContactState,
    ContactSubject,
    GeometryName,
    MotionJoint,
    NameEnum,
    RobotLink,
)


class SkateboardingDemo(NameEnum):
    """Bundled skateboarding demo identifiers."""

    PUSHOFF5_TWOSHOES = "pushoff5_twoshoes"


class SkateboardingMotionJoint(MotionJoint):
    """SMPL-X/core joints used by the skateboarding recipe."""

    PELVIS = "Pelvis"
    LEFT_HIP = "L_Hip"
    RIGHT_HIP = "R_Hip"
    LEFT_KNEE = "L_Knee"
    RIGHT_KNEE = "R_Knee"
    LEFT_ANKLE = "L_Ankle"
    RIGHT_ANKLE = "R_Ankle"
    LEFT_FOOT = "L_Foot"
    RIGHT_FOOT = "R_Foot"
    LEFT_TOE = "L_Toe"
    RIGHT_TOE = "R_Toe"
    SPINE2 = "Spine2"
    SPINE3 = "Spine3"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_SHOULDER = "L_Shoulder"
    RIGHT_SHOULDER = "R_Shoulder"
    LEFT_ELBOW = "L_Elbow"
    RIGHT_ELBOW = "R_Elbow"
    LEFT_WRIST = "L_Wrist"
    RIGHT_WRIST = "R_Wrist"


class SkateboardingRobotLink(RobotLink):
    """Robot links targeted by the skateboarding recipe."""

    LEFT_ANKLE_ROLL = "left_ankle_roll_link"
    RIGHT_ANKLE_ROLL = "right_ankle_roll_link"
    LEFT_HIP_PITCH = "left_hip_pitch_link"
    RIGHT_HIP_PITCH = "right_hip_pitch_link"
    LEFT_KNEE = "left_knee_link"
    RIGHT_KNEE = "right_knee_link"
    LEFT_ANKLE_PITCH = "left_ankle_pitch_link"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch_link"
    TORSO = "torso_link"


class SkateboardingContactSubject(ContactSubject):
    """Synchronized contact subjects used by the skateboarding recipe."""

    LEFT_SHOE = "left_shoe"
    RIGHT_SHOE = "right_shoe"
    SKATEBOARD = "skateboard"


class SkateboardingContactState(ContactState):
    """Contact state labels expected from the foot-support layer."""

    AIR = "air"
    GROUND = "ground"
    SKATEBOARD = "skateboard"


class SkateboardingContactPatch(ContactPatch):
    """Named contact patches for skateboarding tasks."""

    LEFT_SHOE_SOLE = "left_shoe_sole"
    RIGHT_SHOE_SOLE = "right_shoe_sole"
    DECK = "deck"


class SkateboardingGeometryName(GeometryName):
    """Geometry groups owned by the retarget skateboarding recipe."""

    DECK = "skateboard_deck"
    GROUND = "ground"


DEFAULT_DEMO = SkateboardingDemo.PUSHOFF5_TWOSHOES.value
FOOT_TARGET_LINKS = (
    SkateboardingRobotLink.LEFT_ANKLE_ROLL,
    SkateboardingRobotLink.RIGHT_ANKLE_ROLL,
)
UPPER_COM_JOINTS = (
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
UPPER_COM_TARGET_LINK = SkateboardingRobotLink.TORSO
DECK_SAMPLE_POINTS = np.asarray(
    [[x, y, z] for x in (-0.38, 0.38) for y in (-0.10, 0.10) for z in (-0.015, 0.015)],
    dtype=np.float64,
)
