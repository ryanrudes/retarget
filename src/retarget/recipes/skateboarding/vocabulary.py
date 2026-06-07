"""Typed native, semantic, and contact vocabularies for skateboarding."""

from __future__ import annotations

import numpy as np

from retarget.core.enums import (
    ContactPatch,
    ContactState,
    ContactSubject,
    MocapRigidBody,
    MotionJoint,
    NameEnum,
    ObservationRole,
    SceneGeometry,
)


class SkateboardingDemo(NameEnum):
    """Bundled skateboarding capture identifiers."""

    PUSHOFF5_TWOSHOES = "pushoff5_twoshoes"


class SkateboardingRigidBody(MocapRigidBody):
    """Native Vicon rigid bodies."""

    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"
    BOARD = "Skateboard"


class SkateboardingMotionJoint(MotionJoint):
    """GVHMR/SMPL-X joints used by the workflow."""

    PELVIS = "Pelvis"
    LEFT_HIP = "L_Hip"
    RIGHT_HIP = "R_Hip"
    LEFT_KNEE = "L_Knee"
    RIGHT_KNEE = "R_Knee"
    LEFT_ANKLE = "L_Ankle"
    RIGHT_ANKLE = "R_Ankle"
    LEFT_FOOT = "L_Foot"
    RIGHT_FOOT = "R_Foot"
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
    LEFT_BIG_TOE = "L_BigToe"
    LEFT_SMALL_TOE = "L_SmallToe"
    LEFT_HEEL = "L_Heel"
    RIGHT_BIG_TOE = "R_BigToe"
    RIGHT_SMALL_TOE = "R_SmallToe"
    RIGHT_HEEL = "R_Heel"


class SkateboardingObservationRole(ObservationRole):
    """Target-independent semantic landmarks and objects."""

    PELVIS = "pelvis"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    UPPER_BODY_CENTER = "upper_body_center"
    BOARD = "board"


class SkateboardingContactSubject(ContactSubject):
    """Semantic contact subjects."""

    LEFT_SHOE = "left_shoe"
    RIGHT_SHOE = "right_shoe"


class SkateboardingContactState(ContactState):
    """Foot support states."""

    AIR = "air"
    GROUND = "ground"
    BOARD = "board"


class SkateboardingContactPatch(ContactPatch):
    """Contact patches."""

    LEFT_SHOE_SOLE = "left_shoe_sole"
    RIGHT_SHOE_SOLE = "right_shoe_sole"


class SkateboardingGeometryName(SceneGeometry):
    """Observed geometry names."""

    BOARD = "skateboard_deck"
    GROUND = "ground"


DEFAULT_DEMO = SkateboardingDemo.PUSHOFF5_TWOSHOES.value
DECK_SAMPLE_POINTS = np.asarray(
    [[x, y, z] for x in (-0.38, 0.38) for y in (-0.10, 0.10) for z in (-0.015, 0.015)],
    dtype=np.float64,
)
