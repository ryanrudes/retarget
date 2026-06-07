"""Typed vocabulary and constants for Holosoma-compatible recipes."""

from __future__ import annotations

from retarget.core.enums import (
    ContactPatch,
    ContactState,
    ContactSubject,
    MotionJoint,
    ObservationRole,
    RobotGeometry,
    RobotJoint,
    RobotLink,
    RobotRole,
    SceneGeometry,
)


class HolosomaMocapJoint(MotionJoint):
    """Named MOCAP joints used by Holosoma's climbing fixture."""

    HIPS = "Hips"
    SPINE = "Spine"
    SPINE1 = "Spine1"
    NECK = "Neck"
    HEAD = "Head"
    LEFT_SHOULDER = "LeftShoulder"
    LEFT_ARM = "LeftArm"
    LEFT_FOREARM = "LeftForeArm"
    LEFT_HAND = "LeftHand"
    LEFT_HAND_THUMB1 = "LeftHandThumb1"
    LEFT_HAND_THUMB2 = "LeftHandThumb2"
    LEFT_HAND_THUMB3 = "LeftHandThumb3"
    LEFT_HAND_INDEX1 = "LeftHandIndex1"
    LEFT_HAND_INDEX2 = "LeftHandIndex2"
    LEFT_HAND_INDEX3 = "LeftHandIndex3"
    LEFT_HAND_MIDDLE1 = "LeftHandMiddle1"
    LEFT_HAND_MIDDLE2 = "LeftHandMiddle2"
    LEFT_HAND_MIDDLE3 = "LeftHandMiddle3"
    LEFT_HAND_RING1 = "LeftHandRing1"
    LEFT_HAND_RING2 = "LeftHandRing2"
    LEFT_HAND_RING3 = "LeftHandRing3"
    LEFT_HAND_PINKY1 = "LeftHandPinky1"
    LEFT_HAND_PINKY2 = "LeftHandPinky2"
    LEFT_HAND_PINKY3 = "LeftHandPinky3"
    RIGHT_SHOULDER = "RightShoulder"
    RIGHT_ARM = "RightArm"
    RIGHT_FOREARM = "RightForeArm"
    RIGHT_HAND = "RightHand"
    RIGHT_HAND_THUMB1 = "RightHandThumb1"
    RIGHT_HAND_THUMB2 = "RightHandThumb2"
    RIGHT_HAND_THUMB3 = "RightHandThumb3"
    RIGHT_HAND_INDEX1 = "RightHandIndex1"
    RIGHT_HAND_INDEX2 = "RightHandIndex2"
    RIGHT_HAND_INDEX3 = "RightHandIndex3"
    RIGHT_HAND_MIDDLE1 = "RightHandMiddle1"
    RIGHT_HAND_MIDDLE2 = "RightHandMiddle2"
    RIGHT_HAND_MIDDLE3 = "RightHandMiddle3"
    RIGHT_HAND_RING1 = "RightHandRing1"
    RIGHT_HAND_RING2 = "RightHandRing2"
    RIGHT_HAND_RING3 = "RightHandRing3"
    RIGHT_HAND_PINKY1 = "RightHandPinky1"
    RIGHT_HAND_PINKY2 = "RightHandPinky2"
    RIGHT_HAND_PINKY3 = "RightHandPinky3"
    LEFT_UP_LEG = "LeftUpLeg"
    LEFT_LEG = "LeftLeg"
    LEFT_FOOT = "LeftFoot"
    LEFT_TOE_BASE = "LeftToeBase"
    RIGHT_UP_LEG = "RightUpLeg"
    RIGHT_LEG = "RightLeg"
    RIGHT_FOOT = "RightFoot"
    RIGHT_TOE_BASE = "RightToeBase"
    LEFT_FOOT_MOD = "LeftFootMod"
    RIGHT_FOOT_MOD = "RightFootMod"


class G1SpherehandJoint(RobotJoint):
    """Actuated joints in the Holosoma G1 spherehand model."""

    LEFT_HIP_PITCH = "left_hip_pitch_joint"
    LEFT_HIP_ROLL = "left_hip_roll_joint"
    LEFT_HIP_YAW = "left_hip_yaw_joint"
    LEFT_KNEE = "left_knee_joint"
    LEFT_ANKLE_PITCH = "left_ankle_pitch_joint"
    LEFT_ANKLE_ROLL = "left_ankle_roll_joint"
    RIGHT_HIP_PITCH = "right_hip_pitch_joint"
    RIGHT_HIP_ROLL = "right_hip_roll_joint"
    RIGHT_HIP_YAW = "right_hip_yaw_joint"
    RIGHT_KNEE = "right_knee_joint"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch_joint"
    RIGHT_ANKLE_ROLL = "right_ankle_roll_joint"
    WAIST_YAW = "waist_yaw_joint"
    WAIST_ROLL = "waist_roll_joint"
    WAIST_PITCH = "waist_pitch_joint"
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch_joint"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll_joint"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw_joint"
    LEFT_ELBOW = "left_elbow_joint"
    LEFT_WRIST_ROLL = "left_wrist_roll_joint"
    LEFT_WRIST_PITCH = "left_wrist_pitch_joint"
    LEFT_WRIST_YAW = "left_wrist_yaw_joint"
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch_joint"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll_joint"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw_joint"
    RIGHT_ELBOW = "right_elbow_joint"
    RIGHT_WRIST_ROLL = "right_wrist_roll_joint"
    RIGHT_WRIST_PITCH = "right_wrist_pitch_joint"
    RIGHT_WRIST_YAW = "right_wrist_yaw_joint"


class G1SpherehandLink(RobotLink):
    """Bodies in the Holosoma G1 spherehand model."""

    PELVIS = "pelvis"
    PELVIS_CONTOUR = "pelvis_contour_link"
    LEFT_HIP_PITCH = "left_hip_pitch_link"
    LEFT_HIP_ROLL = "left_hip_roll_link"
    LEFT_HIP_YAW = "left_hip_yaw_link"
    LEFT_KNEE = "left_knee_link"
    LEFT_ANKLE_INTERMEDIATE_1 = "left_ankle_intermediate_1_link"
    LEFT_ANKLE_PITCH = "left_ankle_pitch_link"
    LEFT_ANKLE_ROLL = "left_ankle_roll_link"
    LEFT_ANKLE_ROLL_SPHERE_1 = "left_ankle_roll_sphere_1_link"
    LEFT_ANKLE_ROLL_SPHERE_2 = "left_ankle_roll_sphere_2_link"
    LEFT_ANKLE_ROLL_SPHERE_3 = "left_ankle_roll_sphere_3_link"
    LEFT_ANKLE_ROLL_SPHERE_4 = "left_ankle_roll_sphere_4_link"
    LEFT_ANKLE_ROLL_SPHERE_5 = "left_ankle_roll_sphere_5_link"
    RIGHT_HIP_PITCH = "right_hip_pitch_link"
    RIGHT_HIP_ROLL = "right_hip_roll_link"
    RIGHT_HIP_YAW = "right_hip_yaw_link"
    RIGHT_KNEE = "right_knee_link"
    RIGHT_ANKLE_INTERMEDIATE_1 = "right_ankle_intermediate_1_link"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch_link"
    RIGHT_ANKLE_ROLL = "right_ankle_roll_link"
    RIGHT_ANKLE_ROLL_SPHERE_1 = "right_ankle_roll_sphere_1_link"
    RIGHT_ANKLE_ROLL_SPHERE_2 = "right_ankle_roll_sphere_2_link"
    RIGHT_ANKLE_ROLL_SPHERE_3 = "right_ankle_roll_sphere_3_link"
    RIGHT_ANKLE_ROLL_SPHERE_4 = "right_ankle_roll_sphere_4_link"
    RIGHT_ANKLE_ROLL_SPHERE_5 = "right_ankle_roll_sphere_5_link"
    WAIST_YAW = "waist_yaw_link"
    WAIST_ROLL = "waist_roll_link"
    TORSO = "torso_link"
    HEAD = "head_link"
    HEAD_MOCAP = "head_mocap"
    IMU_IN_TORSO = "imu_in_torso"
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch_link"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll_link"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw_link"
    LEFT_ELBOW = "left_elbow_link"
    LEFT_WRIST_ROLL = "left_wrist_roll_link"
    LEFT_WRIST_PITCH = "left_wrist_pitch_link"
    LEFT_WRIST_YAW = "left_wrist_yaw_link"
    LEFT_SPHERE_HAND = "left_sphere_hand_link"
    LEFT_SPHERE_HAND_TIP = "left_sphere_hand_tip_link"
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch_link"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll_link"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw_link"
    RIGHT_ELBOW = "right_elbow_link"
    RIGHT_WRIST_ROLL = "right_wrist_roll_link"
    RIGHT_WRIST_PITCH = "right_wrist_pitch_link"
    RIGHT_WRIST_YAW = "right_wrist_yaw_link"
    RIGHT_SPHERE_HAND = "right_sphere_hand_link"
    RIGHT_SPHERE_HAND_TIP = "right_sphere_hand_tip_link"


class G1SpherehandGeometry(RobotGeometry):
    """Named geometries in the Holosoma G1 spherehand model."""

    GROUND = "ground"
    PELVIS = "pelvis"
    LEFT_HIP_PITCH = "left_hip_pitch_link"
    LEFT_HIP_ROLL = "left_hip_roll_link"
    LEFT_HIP_YAW = "left_hip_yaw_link"
    LEFT_KNEE = "left_knee_link"
    LEFT_ANKLE_INTERMEDIATE_1 = "left_ankle_intermediate_1_link"
    LEFT_ANKLE_PITCH = "left_ankle_pitch_link"
    LEFT_ANKLE_ROLL = "left_ankle_roll_link"
    LEFT_ANKLE_ROLL_SPHERE_1 = "left_ankle_roll_sphere_1_link"
    LEFT_ANKLE_ROLL_SPHERE_2 = "left_ankle_roll_sphere_2_link"
    LEFT_ANKLE_ROLL_SPHERE_3 = "left_ankle_roll_sphere_3_link"
    LEFT_ANKLE_ROLL_SPHERE_4 = "left_ankle_roll_sphere_4_link"
    LEFT_ANKLE_ROLL_SPHERE_5 = "left_ankle_roll_sphere_5_link"
    RIGHT_HIP_PITCH = "right_hip_pitch_link"
    RIGHT_HIP_ROLL = "right_hip_roll_link"
    RIGHT_HIP_YAW = "right_hip_yaw_link"
    RIGHT_KNEE = "right_knee_link"
    RIGHT_ANKLE_INTERMEDIATE_1 = "right_ankle_intermediate_1_link"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch_link"
    RIGHT_ANKLE_ROLL = "right_ankle_roll_link"
    RIGHT_ANKLE_ROLL_SPHERE_1 = "right_ankle_roll_sphere_1_link"
    RIGHT_ANKLE_ROLL_SPHERE_2 = "right_ankle_roll_sphere_2_link"
    RIGHT_ANKLE_ROLL_SPHERE_3 = "right_ankle_roll_sphere_3_link"
    RIGHT_ANKLE_ROLL_SPHERE_4 = "right_ankle_roll_sphere_4_link"
    RIGHT_ANKLE_ROLL_SPHERE_5 = "right_ankle_roll_sphere_5_link"
    WAIST_YAW = "waist_yaw_link"
    WAIST_ROLL = "waist_roll_link"
    TORSO = "torso_link"
    HEAD = "head_link"
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch_link"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll_link"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw_link"
    LEFT_ELBOW = "left_elbow_link"
    LEFT_WRIST_ROLL = "left_wrist_roll_link"
    LEFT_WRIST_PITCH = "left_wrist_pitch_link"
    LEFT_WRIST_YAW = "left_wrist_yaw_link"
    LEFT_SPHERE_HAND = "left_sphere_hand_link"
    LEFT_SPHERE_HAND_TIP = "left_sphere_hand_tip_link"
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch_link"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll_link"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw_link"
    RIGHT_ELBOW = "right_elbow_link"
    RIGHT_WRIST_ROLL = "right_wrist_roll_link"
    RIGHT_WRIST_PITCH = "right_wrist_pitch_link"
    RIGHT_WRIST_YAW = "right_wrist_yaw_link"
    RIGHT_SPHERE_HAND = "right_sphere_hand_link"
    RIGHT_SPHERE_HAND_TIP = "right_sphere_hand_tip_link"


class HolosomaObservationRole(ObservationRole):
    """Semantic objects observed in the climbing scene."""

    CLIMBING_STRUCTURE = "climbing_structure"


class HolosomaRobotRole(RobotRole):
    """Robot roles required by the Holosoma climbing adaptation."""

    PELVIS = "pelvis"
    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_TOE = "left_toe"
    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_TOE = "right_toe"
    LEFT_SHOULDER = "left_shoulder"
    LEFT_ELBOW = "left_elbow"
    LEFT_HAND = "left_hand"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_HAND = "right_hand"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_FOOT_CONTACT = "left_foot_contact"
    RIGHT_FOOT_CONTACT = "right_foot_contact"


class HolosomaContactSubject(ContactSubject):
    """Semantic contact subjects for climbing."""

    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class HolosomaContactPatch(ContactPatch):
    """Semantic contact patches for climbing."""

    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"


class HolosomaContactState(ContactState):
    """Contact states emitted by the Holosoma foot-sticking extractor."""

    AIR = "air"
    STICKING = "sticking"


class HolosomaGeometryName(SceneGeometry):
    """Named fixture geometries referenced by the climbing recipe."""

    MULTI_BOX_1 = "multi_boxes_link_1"
    MULTI_BOX_2 = "multi_boxes_link_2"
    MULTI_BOX_3 = "multi_boxes_link_3"
    GROUND = "ground"


G1_LEFT_FOOT_STICKING_LINKS = (
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_1,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_2,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_3,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_4,
)
G1_RIGHT_FOOT_STICKING_LINKS = (
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_1,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_2,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_3,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_4,
)
G1_FOOT_STICKING_LINKS = (
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_1,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_1,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_2,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_2,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_3,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_3,
    G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_4,
    G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_4,
)

MOCAP_TO_ROBOT_ROLE = {
    HolosomaMocapJoint.SPINE1: HolosomaRobotRole.PELVIS,
    HolosomaMocapJoint.LEFT_UP_LEG: HolosomaRobotRole.LEFT_HIP,
    HolosomaMocapJoint.LEFT_LEG: HolosomaRobotRole.LEFT_KNEE,
    HolosomaMocapJoint.LEFT_TOE_BASE: HolosomaRobotRole.LEFT_TOE,
    HolosomaMocapJoint.RIGHT_UP_LEG: HolosomaRobotRole.RIGHT_HIP,
    HolosomaMocapJoint.RIGHT_LEG: HolosomaRobotRole.RIGHT_KNEE,
    HolosomaMocapJoint.RIGHT_TOE_BASE: HolosomaRobotRole.RIGHT_TOE,
    HolosomaMocapJoint.LEFT_ARM: HolosomaRobotRole.LEFT_SHOULDER,
    HolosomaMocapJoint.LEFT_FOREARM: HolosomaRobotRole.LEFT_ELBOW,
    HolosomaMocapJoint.LEFT_HAND_MIDDLE3: HolosomaRobotRole.LEFT_HAND,
    HolosomaMocapJoint.RIGHT_ARM: HolosomaRobotRole.RIGHT_SHOULDER,
    HolosomaMocapJoint.RIGHT_FOREARM: HolosomaRobotRole.RIGHT_ELBOW,
    HolosomaMocapJoint.RIGHT_HAND_MIDDLE3: HolosomaRobotRole.RIGHT_HAND,
    HolosomaMocapJoint.LEFT_FOOT: HolosomaRobotRole.LEFT_FOOT,
    HolosomaMocapJoint.RIGHT_FOOT: HolosomaRobotRole.RIGHT_FOOT,
}
