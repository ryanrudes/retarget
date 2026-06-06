"""Typed vocabulary and constants for Holosoma-compatible recipes."""

from __future__ import annotations

from retarget.core.enums import ContactState, GeometryName, MotionJoint, RobotLink

G1_DOF = 29
G1_HEIGHT_M = 1.32
MOCAP_HUMAN_HEIGHT_M = 1.78
MOCAP_FPS = 30.0
MOCAP_DOWNSAMPLE = 4
MOCAP_MAT_HEIGHT_M = 0.1
FOOT_STICKING_VELOCITY_THRESHOLD = 0.01
COLLISION_DETECTION_THRESHOLD = 0.1
MULTI_BOX_SAMPLE_COUNT = 100
MULTI_BOX_SAMPLE_SEED = 42


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


class G1SpherehandLink(RobotLink):
    """G1 spherehand links used by the Holosoma climbing subset."""

    PELVIS_CONTOUR = "pelvis_contour_link"
    LEFT_HIP_PITCH = "left_hip_pitch_link"
    LEFT_KNEE = "left_knee_link"
    LEFT_ANKLE_ROLL_SPHERE_1 = "left_ankle_roll_sphere_1_link"
    LEFT_ANKLE_ROLL_SPHERE_2 = "left_ankle_roll_sphere_2_link"
    LEFT_ANKLE_ROLL_SPHERE_3 = "left_ankle_roll_sphere_3_link"
    LEFT_ANKLE_ROLL_SPHERE_4 = "left_ankle_roll_sphere_4_link"
    LEFT_ANKLE_ROLL_SPHERE_5 = "left_ankle_roll_sphere_5_link"
    RIGHT_HIP_PITCH = "right_hip_pitch_link"
    RIGHT_KNEE = "right_knee_link"
    RIGHT_ANKLE_ROLL_SPHERE_1 = "right_ankle_roll_sphere_1_link"
    RIGHT_ANKLE_ROLL_SPHERE_2 = "right_ankle_roll_sphere_2_link"
    RIGHT_ANKLE_ROLL_SPHERE_3 = "right_ankle_roll_sphere_3_link"
    RIGHT_ANKLE_ROLL_SPHERE_4 = "right_ankle_roll_sphere_4_link"
    RIGHT_ANKLE_ROLL_SPHERE_5 = "right_ankle_roll_sphere_5_link"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll_link"
    LEFT_ELBOW = "left_elbow_link"
    LEFT_SPHERE_HAND = "left_sphere_hand_link"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll_link"
    RIGHT_ELBOW = "right_elbow_link"
    RIGHT_SPHERE_HAND = "right_sphere_hand_link"
    LEFT_ANKLE_INTERMEDIATE_1 = "left_ankle_intermediate_1_link"
    RIGHT_ANKLE_INTERMEDIATE_1 = "right_ankle_intermediate_1_link"


class HolosomaContactState(ContactState):
    """Contact states emitted by the Holosoma foot-sticking extractor."""

    AIR = "air"
    STICKING = "sticking"


class HolosomaGeometryName(GeometryName):
    """Scene geometry groups referenced by the climbing recipe."""

    MULTI_BOXES = "multi_boxes"
    GROUND = "ground"


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
G1_LEFT_FOOT_STICKING_LINKS = tuple(link for link in G1_FOOT_STICKING_LINKS if link.startswith("left_"))
G1_RIGHT_FOOT_STICKING_LINKS = tuple(link for link in G1_FOOT_STICKING_LINKS if link.startswith("right_"))

G1_MANUAL_LOWER_QPOS = {
    3: -1.0,
    4: -1.0,
    5: -1.0,
    6: -1.0,
    20: -0.3,
    21: -0.1,
    26: -0.1,
    27: -0.1,
    28: -0.05,
    33: -0.1,
    34: -0.1,
    35: -0.05,
}
G1_MANUAL_UPPER_QPOS = {
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    20: 0.3,
    25: 1.4,
    26: 0.2,
    27: 0.3,
    28: 0.05,
    32: 1.4,
    33: 0.2,
    34: 0.3,
    35: 0.05,
}
G1_MANUAL_QPOS_COSTS = {19: 0.2, 20: 0.2}
G1_NOMINAL_TRACKING_QPOS_INDICES = tuple(range(19))

MOCAP_DEMO_JOINTS = tuple(joint.value for joint in HolosomaMocapJoint)
MOCAP_TO_G1_LINK_MAPPING = {
    HolosomaMocapJoint.SPINE1: G1SpherehandLink.PELVIS_CONTOUR,
    HolosomaMocapJoint.LEFT_UP_LEG: G1SpherehandLink.LEFT_HIP_PITCH,
    HolosomaMocapJoint.LEFT_LEG: G1SpherehandLink.LEFT_KNEE,
    HolosomaMocapJoint.LEFT_TOE_BASE: G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_5,
    HolosomaMocapJoint.RIGHT_UP_LEG: G1SpherehandLink.RIGHT_HIP_PITCH,
    HolosomaMocapJoint.RIGHT_LEG: G1SpherehandLink.RIGHT_KNEE,
    HolosomaMocapJoint.RIGHT_TOE_BASE: G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_5,
    HolosomaMocapJoint.LEFT_ARM: G1SpherehandLink.LEFT_SHOULDER_ROLL,
    HolosomaMocapJoint.LEFT_FOREARM: G1SpherehandLink.LEFT_ELBOW,
    HolosomaMocapJoint.LEFT_HAND_MIDDLE3: G1SpherehandLink.LEFT_SPHERE_HAND,
    HolosomaMocapJoint.RIGHT_ARM: G1SpherehandLink.RIGHT_SHOULDER_ROLL,
    HolosomaMocapJoint.RIGHT_FOREARM: G1SpherehandLink.RIGHT_ELBOW,
    HolosomaMocapJoint.RIGHT_HAND_MIDDLE3: G1SpherehandLink.RIGHT_SPHERE_HAND,
    HolosomaMocapJoint.LEFT_FOOT: G1SpherehandLink.LEFT_ANKLE_INTERMEDIATE_1,
    HolosomaMocapJoint.RIGHT_FOOT: G1SpherehandLink.RIGHT_ANKLE_INTERMEDIATE_1,
}
