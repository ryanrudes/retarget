"""Holosoma-compatible typed sources and recipes."""

from retarget.integrations.holosoma.contacts import foot_sticking_contact_plan, foot_sticking_states
from retarget.integrations.holosoma.geometry import included_geom_names, object_non_penetration_geometry_pairs
from retarget.integrations.holosoma.layout import (
    HolosomaClimbLayout,
    default_holosoma_root,
    holosoma_climb_layout,
    holosoma_g1_robot_dir,
)
from retarget.integrations.holosoma.motion import (
    compute_climb_q_init,
    mocap_motion_format,
    preprocess_mocap_climb,
    transform_from_human_to_world,
)
from retarget.integrations.holosoma.object import (
    convert_object_poses_to_mujoco_order,
    dummy_object_poses,
    ensure_scaled_multi_boxes_assets,
    object_visual_parts_from_urdf,
    preprocess_object_poses,
    sample_multi_boxes_like_holosoma,
)
from retarget.integrations.holosoma.profile import holosoma_climb_profile
from retarget.integrations.holosoma.recipe import (
    HolosomaClimbPreparation,
    HolosomaClimbRecipe,
    from_mocap_climb_fixture,
    holosoma_initial_qpos_plan,
)
from retarget.integrations.holosoma.robot import ensure_g1_model_assets, g1_spherehand_robot
from retarget.integrations.holosoma.vocabulary import (
    COLLISION_DETECTION_THRESHOLD,
    FOOT_STICKING_VELOCITY_THRESHOLD,
    G1_DOF,
    G1_FOOT_STICKING_LINKS,
    G1_HEIGHT_M,
    G1_LEFT_FOOT_STICKING_LINKS,
    G1_MANUAL_LOWER_QPOS,
    G1_MANUAL_QPOS_COSTS,
    G1_MANUAL_UPPER_QPOS,
    G1_NOMINAL_TRACKING_QPOS_INDICES,
    G1_RIGHT_FOOT_STICKING_LINKS,
    MOCAP_DEMO_JOINTS,
    MOCAP_DOWNSAMPLE,
    MOCAP_FPS,
    MOCAP_HUMAN_HEIGHT_M,
    MOCAP_MAT_HEIGHT_M,
    MOCAP_TO_G1_LINK_MAPPING,
    MULTI_BOX_SAMPLE_COUNT,
    MULTI_BOX_SAMPLE_SEED,
    G1SpherehandLink,
    HolosomaContactState,
    HolosomaGeometryName,
    HolosomaMocapJoint,
)

_foot_sticking_states = foot_sticking_states
_holosoma_climb_layout = holosoma_climb_layout
_holosoma_g1_robot_dir = holosoma_g1_robot_dir
_included_geom_names = included_geom_names
_object_visual_parts_from_urdf = object_visual_parts_from_urdf

__all__ = [
    "COLLISION_DETECTION_THRESHOLD",
    "FOOT_STICKING_VELOCITY_THRESHOLD",
    "G1_DOF",
    "G1_FOOT_STICKING_LINKS",
    "G1_HEIGHT_M",
    "G1_LEFT_FOOT_STICKING_LINKS",
    "G1_MANUAL_LOWER_QPOS",
    "G1_MANUAL_QPOS_COSTS",
    "G1_MANUAL_UPPER_QPOS",
    "G1_NOMINAL_TRACKING_QPOS_INDICES",
    "G1_RIGHT_FOOT_STICKING_LINKS",
    "MOCAP_DEMO_JOINTS",
    "MOCAP_DOWNSAMPLE",
    "MOCAP_FPS",
    "MOCAP_HUMAN_HEIGHT_M",
    "MOCAP_MAT_HEIGHT_M",
    "MOCAP_TO_G1_LINK_MAPPING",
    "MULTI_BOX_SAMPLE_COUNT",
    "MULTI_BOX_SAMPLE_SEED",
    "G1SpherehandLink",
    "HolosomaClimbLayout",
    "HolosomaClimbPreparation",
    "HolosomaClimbRecipe",
    "HolosomaContactState",
    "HolosomaGeometryName",
    "HolosomaMocapJoint",
    "_foot_sticking_states",
    "_holosoma_climb_layout",
    "_holosoma_g1_robot_dir",
    "_included_geom_names",
    "_object_visual_parts_from_urdf",
    "compute_climb_q_init",
    "convert_object_poses_to_mujoco_order",
    "default_holosoma_root",
    "dummy_object_poses",
    "ensure_g1_model_assets",
    "ensure_scaled_multi_boxes_assets",
    "foot_sticking_contact_plan",
    "foot_sticking_states",
    "from_mocap_climb_fixture",
    "g1_spherehand_robot",
    "holosoma_climb_layout",
    "holosoma_climb_profile",
    "holosoma_g1_robot_dir",
    "holosoma_initial_qpos_plan",
    "included_geom_names",
    "mocap_motion_format",
    "object_non_penetration_geometry_pairs",
    "object_visual_parts_from_urdf",
    "preprocess_mocap_climb",
    "preprocess_object_poses",
    "sample_multi_boxes_like_holosoma",
    "transform_from_human_to_world",
]
