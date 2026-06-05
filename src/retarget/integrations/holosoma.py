"""Holosoma-compatible task adapters.

The functions in this module are optional integration helpers. They do not
import Holosoma; they encode the data contract needed to reproduce selected
Holosoma tasks using retarget's typed problem surface.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.core.enums import FrameConvention, QuaternionOrder, TaskKind
from retarget.core.pose import PoseSequence
from retarget.mesh import InteractionMeshSpec, LaplacianWeighting
from retarget.motion.contact import ContactPlan, ContactTrack, SupportPlane
from retarget.motion.qpos import InitialQposPlan
from retarget.motion.spec import MotionFormatSpec, MotionSequence
from retarget.optimization import (
    ConstraintConfigUnion,
    DiagonalRegularizationObjectiveConfig,
    FootStickingConstraintConfig,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    SmoothnessObjectiveConfig,
    SolverSpec,
    TrustRegionConstraintConfig,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import ObjectSpec, ObjectTrajectory, SceneSpec

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

G1_FOOT_STICKING_LINKS = (
    "left_ankle_roll_sphere_1_link",
    "right_ankle_roll_sphere_1_link",
    "left_ankle_roll_sphere_2_link",
    "right_ankle_roll_sphere_2_link",
    "left_ankle_roll_sphere_3_link",
    "right_ankle_roll_sphere_3_link",
    "left_ankle_roll_sphere_4_link",
    "right_ankle_roll_sphere_4_link",
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

MOCAP_DEMO_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftFootMod",
    "RightFootMod",
)
MOCAP_TO_G1_LINK_MAPPING = {
    "Spine1": "pelvis_contour_link",
    "LeftUpLeg": "left_hip_pitch_link",
    "LeftLeg": "left_knee_link",
    "LeftToeBase": "left_ankle_roll_sphere_5_link",
    "RightUpLeg": "right_hip_pitch_link",
    "RightLeg": "right_knee_link",
    "RightToeBase": "right_ankle_roll_sphere_5_link",
    "LeftArm": "left_shoulder_roll_link",
    "LeftForeArm": "left_elbow_link",
    "LeftHandMiddle3": "left_sphere_hand_link",
    "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link",
    "RightHandMiddle3": "right_sphere_hand_link",
    "LeftFoot": "left_ankle_intermediate_1_link",
    "RightFoot": "right_ankle_intermediate_1_link",
}


@dataclass(frozen=True)
class HolosomaClimbPreparation:
    """Typed retarget inputs and problem for a Holosoma MOCAP climbing task."""

    robot: RobotSpec
    motion: MotionSequence
    scene: SceneSpec
    contacts: ContactPlan
    initial_qpos: InitialQposPlan
    problem: RetargetingProblem
    q_init: np.ndarray
    object_poses_mujoco: np.ndarray
    object_sample_points: np.ndarray
    geometry_pairs: tuple[tuple[str, str], ...]
    scale: float
    metadata: dict[str, Any] = field(default_factory=dict)


def default_holosoma_root() -> Path:
    """Return the sibling Holosoma checkout path used by local parity tests."""

    return Path(__file__).resolve().parents[4] / "holosoma"


def mocap_motion_format() -> MotionFormatSpec:
    """Return the Holosoma MOCAP format contract."""

    return MotionFormatSpec(
        name="holosoma_mocap",
        joint_names=MOCAP_DEMO_JOINTS,
        root_joint="Hips",
        contact_joints=("LeftToeBase", "RightToeBase"),
        default_fps=MOCAP_FPS,
        default_height_m=MOCAP_HUMAN_HEIGHT_M,
        description="Holosoma MOCAP climbing joint order.",
    )


def from_mocap_climb_fixture(
    holosoma_root: str | Path | None = None,
    *,
    frame_count: int | None = None,
    ensure_model_assets: bool = False,
    include_object_collision: bool = True,
    solver_backend: str = "cvxpy_clarabel",
    show_progress: bool = False,
) -> HolosomaClimbPreparation:
    """Build the typed retargeting problem for Holosoma's real MOCAP climb fixture."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    fixture_dir = root / "tests" / "fixtures" / "climb_seq_0"
    motion_path = fixture_dir / "mocap_climb_seq_0_joint_positions_f900-3700.npy"
    object_mesh_path = fixture_dir / "multi_boxes.obj"
    object_urdf_path = fixture_dir / "multi_boxes.urdf"
    scene_xml_path = fixture_dir / "g1_29dof_spherehand_w_multi_boxes.xml"
    for path in (motion_path, object_mesh_path, object_urdf_path, scene_xml_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if ensure_model_assets:
        ensure_g1_model_assets(root)

    raw_motion = np.load(motion_path)
    human_joints = raw_motion[::MOCAP_DOWNSAMPLE].astype(np.float64, copy=True)
    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        human_joints = human_joints[:frame_count]
    scale = G1_HEIGHT_M / MOCAP_HUMAN_HEIGHT_M
    object_poses = _dummy_object_poses(len(human_joints))
    human_joints = preprocess_mocap_climb(human_joints, scale=scale)
    object_poses = preprocess_object_poses(object_poses, scale=scale)
    object_poses_mujoco = convert_object_poses_to_mujoco_order(object_poses)
    q_init = compute_climb_q_init(human_joints, object_poses, demo_joints=MOCAP_DEMO_JOINTS, robot_dof=G1_DOF)
    object_samples = sample_multi_boxes_like_holosoma(object_mesh_path, sample_count=MULTI_BOX_SAMPLE_COUNT)
    robot = g1_spherehand_robot(
        root,
        scene_xml_path=scene_xml_path,
        include_object_collision=include_object_collision,
    )
    geometry_pairs = (
        object_non_penetration_geometry_pairs(robot, fixture_dir=fixture_dir)
        if include_object_collision
        else ()
    )
    profile = holosoma_climb_profile(
        qpos_size=robot.qpos_size(has_object=False),
        geometry_pairs=geometry_pairs,
    )
    initial_qpos = holosoma_initial_qpos_plan(
        frame_count=human_joints.shape[0],
        q_init=q_init,
        object_poses_mujoco=object_poses_mujoco,
    )

    root_positions = np.tile(q_init[:3], (human_joints.shape[0], 1))
    root_quaternions = np.tile(q_init[3:7], (human_joints.shape[0], 1))
    root_poses = PoseSequence.from_arrays(
        root_positions,
        root_quaternions,
        fps=MOCAP_FPS,
        quaternion_order=QuaternionOrder.WXYZ,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
    )
    motion = MotionSequence(
        name="holosoma_mocap_climb_seq_0",
        joint_positions=human_joints,
        joint_names=MOCAP_DEMO_JOINTS,
        fps=MOCAP_FPS,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        root_poses=root_poses,
        metadata={
            "source": "holosoma_fixture",
            "source_path": str(motion_path),
            "raw_downsample": MOCAP_DOWNSAMPLE,
            "holosoma_scale": scale,
            "source_height_m": MOCAP_HUMAN_HEIGHT_M,
            "robot_height_m": G1_HEIGHT_M,
        },
    )
    object_trajectory = ObjectTrajectory(
        name="multi_boxes",
        poses=PoseSequence.from_arrays(
            object_poses_mujoco[:, :3],
            object_poses_mujoco[:, 3:],
            fps=MOCAP_FPS,
            quaternion_order=QuaternionOrder.WXYZ,
            frame=FrameConvention.Z_UP_RIGHT_HANDED,
        ),
    )
    scene = SceneSpec.climbing(
        object_spec=ObjectSpec(
            name="multi_boxes",
            mesh_path=object_mesh_path,
            urdf_path=object_urdf_path,
            sample_points=object_samples,
            trajectory=object_trajectory,
            qpos_mode="external",
            metadata={
                "sample_count": MULTI_BOX_SAMPLE_COUNT,
                "sample_seed": MULTI_BOX_SAMPLE_SEED,
                "sample_space": "object_local",
            },
        )
    )
    contacts = foot_sticking_contact_plan(human_joints, demo_joints=MOCAP_DEMO_JOINTS)
    metadata = {
        "source": "holosoma",
        "task_type": "climbing",
        "data_format": "mocap",
        "fixture": str(fixture_dir),
        "include_object_collision": include_object_collision,
        "collision_detection_threshold": COLLISION_DETECTION_THRESHOLD,
        "q_a_init_idx": -7,
    }
    problem = RetargetingProblem(
        name="holosoma_mocap_climb_seq_0",
        task_kind=TaskKind.CLIMBING,
        robot=robot,
        motion=motion,
        scene=scene,
        contacts=contacts,
        initial_qpos=initial_qpos,
        motion_format=mocap_motion_format(),
        variables=QposVariableSpec.holosoma_q_a(-7),
        mesh=InteractionMeshSpec(laplacian_weighting=LaplacianWeighting.UNIFORM),
        solver=SolverSpec(
            backend=solver_backend,
            max_iterations=10,
            first_frame_iterations=50,
            trust_radius=0.2,
            convergence="cost_plateau",
        ),
        objectives=profile.objectives,
        constraints=profile.constraints,
        scale_to_robot=False,
        show_progress=show_progress,
        metadata=metadata,
    )
    return HolosomaClimbPreparation(
        robot=robot,
        motion=motion,
        scene=scene,
        contacts=contacts,
        initial_qpos=initial_qpos,
        problem=problem,
        q_init=q_init,
        object_poses_mujoco=object_poses_mujoco,
        object_sample_points=object_samples,
        geometry_pairs=geometry_pairs,
        scale=scale,
        metadata=metadata,
    )


def g1_spherehand_robot(
    holosoma_root: str | Path | None = None,
    *,
    scene_xml_path: str | Path | None = None,
    include_object_collision: bool = False,
) -> RobotSpec:
    """Return a G1 spherehand robot spec matching Holosoma's MOCAP task."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    robot_dir = root / "src" / "interaction_mesh_retarget" / "robots" / "g1"
    urdf_path = robot_dir / "g1_29dof_spherehand.urdf"
    robot_xml_path = robot_dir / "g1_29dof_spherehand.xml"
    xml_path = Path(scene_xml_path).resolve() if scene_xml_path is not None else robot_xml_path
    for path in (urdf_path, robot_xml_path, xml_path):
        if not path.exists():
            raise FileNotFoundError(path)
    joint_names = tuple(_mjcf_joint_names(robot_xml_path))
    robot_geom_names = tuple(_mjcf_geom_names(robot_xml_path))
    link_names = tuple(dict.fromkeys((*_mjcf_body_names(robot_xml_path), *MOCAP_TO_G1_LINK_MAPPING.values())))
    joint_limits = _apply_manual_qpos_bounds(
        _mjcf_joint_limits(robot_xml_path, joint_names),
        joint_names=joint_names,
    )
    return RobotSpec(
        name="holosoma_g1_29dof_spherehand",
        dof=G1_DOF,
        height_m=G1_HEIGHT_M,
        joint_names=joint_names,
        link_names=link_names,
        contact_links=G1_FOOT_STICKING_LINKS,
        joint_limits=joint_limits,
        default_link_mapping=dict(MOCAP_TO_G1_LINK_MAPPING),
        urdf_path=urdf_path,
        mujoco_xml_path=xml_path if include_object_collision or scene_xml_path is not None else robot_xml_path,
        metadata={
            "source": "holosoma",
            "robot_type": "g1",
            "robot_dof": G1_DOF,
            "uses_spherehand": True,
            "holosoma_robot_geom_names": list(robot_geom_names),
            "scene_xml_path": str(xml_path) if include_object_collision or scene_xml_path is not None else None,
        },
    )


def ensure_g1_model_assets(holosoma_root: str | Path | None = None) -> tuple[Path, ...]:
    """Create Holosoma ``models/g1`` symlinks needed by fixture scene XML files."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    source_robot = root / "src" / "interaction_mesh_retarget" / "robots" / "g1"
    models_g1 = root / "models" / "g1"
    models_g1.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name in ("meshes", "assets"):
        source = source_robot / name
        target = models_g1 / name
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            continue
        target.symlink_to(source, target_is_directory=True)
        created.append(target)
    return tuple(created)


def preprocess_mocap_climb(
    human_joints: np.ndarray,
    *,
    scale: float,
    mat_height: float = MOCAP_MAT_HEIGHT_M,
    demo_joints: tuple[str, ...] = MOCAP_DEMO_JOINTS,
    foot_names: tuple[str, str] = ("LeftToeBase", "RightToeBase"),
) -> np.ndarray:
    """Apply Holosoma's climbing MOCAP height normalization and scaling."""

    joints = np.asarray(human_joints, dtype=np.float64).copy()
    left_idx = demo_joints.index(foot_names[0])
    right_idx = demo_joints.index(foot_names[1])
    z_min = float(joints[:, (left_idx, right_idx), 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    joints[:, :, 2] -= z_min
    return joints * float(scale)


def preprocess_object_poses(object_poses: np.ndarray, *, scale: float) -> np.ndarray:
    """Apply Holosoma's object pose scaling rule to ``[qw, qx, qy, qz, x, y, z]`` poses."""

    poses = np.asarray(object_poses, dtype=np.float64).copy()
    poses[:, -3:-1] *= float(scale)
    z0 = float(poses[0, -1])
    poses[:, -1] = z0 + (poses[:, -1] - z0) * float(scale)
    return poses


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert Holosoma object poses from ``[qw, qx, qy, qz, x, y, z]`` to ``[x, y, z, qw, qx, qy, qz]``."""

    return np.asarray(object_poses, dtype=np.float64)[:, [4, 5, 6, 0, 1, 2, 3]]


def compute_climb_q_init(
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    *,
    demo_joints: tuple[str, ...] = MOCAP_DEMO_JOINTS,
    robot_dof: int = G1_DOF,
    spine_joint_name: str = "Spine1",
) -> np.ndarray:
    """Compute Holosoma's initial floating-base qpos for climbing."""

    _translation, quaternion = transform_from_human_to_world(
        human_joints[0, 0, :],
        object_poses[0],
        np.zeros(3, dtype=np.float64),
    )
    spine_idx = demo_joints.index(spine_joint_name)
    return np.concatenate([human_joints[0, spine_idx], quaternion, np.zeros(robot_dof, dtype=np.float64)])


def transform_from_human_to_world(
    human_initial_root: np.ndarray,
    object_initial_pose: np.ndarray,
    local_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Holosoma's human-local to world transform helper."""

    human_to_object_2d = np.asarray(object_initial_pose, dtype=np.float64)[-3:-1] - np.asarray(
        human_initial_root,
        dtype=np.float64,
    )[:2]
    norm = float(np.linalg.norm(human_to_object_2d))
    if norm <= 1e-12:
        raise ValueError("human root and object origin must not coincide in the horizontal plane")
    x_axis_2d = human_to_object_2d / norm
    x_axis = np.array([x_axis_2d[0], x_axis_2d[1], 0.0], dtype=np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= float(np.linalg.norm(y_axis))
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    return rotation_matrix @ np.asarray(local_translation, dtype=np.float64), quaternion


def foot_sticking_contact_plan(
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...] = MOCAP_DEMO_JOINTS,
    velocity_threshold: float = FOOT_STICKING_VELOCITY_THRESHOLD,
) -> ContactPlan:
    """Extract Holosoma's toe-velocity foot sticking contact plan."""

    left_states, right_states = _foot_sticking_states(
        human_joints,
        demo_joints=demo_joints,
        velocity_threshold=velocity_threshold,
    )
    return ContactPlan(
        tracks=(
            ContactTrack(
                subject="LeftToeBase",
                states=left_states,
                link_names=G1_LEFT_FOOT_STICKING_LINKS,
                active_states=(1,),
                support_states=(1,),
                labels=("air", "sticking"),
                metadata={"source": "holosoma_velocity", "velocity_threshold": velocity_threshold},
            ),
            ContactTrack(
                subject="RightToeBase",
                states=right_states,
                link_names=G1_RIGHT_FOOT_STICKING_LINKS,
                active_states=(1,),
                support_states=(1,),
                labels=("air", "sticking"),
                metadata={"source": "holosoma_velocity", "velocity_threshold": velocity_threshold},
            ),
        ),
        frame_count=int(np.asarray(human_joints).shape[0]),
        support=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3)),
        provenance={"source": "holosoma.extract_foot_sticking_sequence_velocity"},
    )


def sample_multi_boxes_like_holosoma(
    mesh_path: str | Path,
    *,
    sample_count: int = MULTI_BOX_SAMPLE_COUNT,
    seed: int = MULTI_BOX_SAMPLE_SEED,
) -> np.ndarray:
    """Sample the climbing multi-box mesh with Holosoma's weighted surface rule."""

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install retarget[mujoco] or trimesh to sample Holosoma multi-box surfaces") from exc
    mesh = cast(Any, trimesh.load(str(mesh_path), force="mesh"))
    rng = np.random.default_rng(seed)
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    face_centers = triangles.mean(axis=1)
    weights = np.where(face_centers[:, 2] > 0.9, 20.0, 1.0)
    probs = face_areas * weights
    probs = probs / probs.sum()
    sampled_face_indices = rng.choice(len(faces), size=int(sample_count), p=probs)
    samples = np.zeros((int(sample_count), 3), dtype=np.float64)
    for idx, face_idx in enumerate(sampled_face_indices):
        v1, v2, v3 = vertices[faces[face_idx]]
        r1, r2 = rng.random(2)
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        samples[idx] = v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
    return samples


def object_non_penetration_geometry_pairs(
    robot: RobotSpec,
    *,
    fixture_dir: str | Path,
    object_name: str = "multi_boxes",
    include_ground: bool = True,
) -> tuple[tuple[str, str], ...]:
    """Return static geometry-pair candidates for Holosoma object/ground non-penetration."""

    fixture = Path(fixture_dir)
    object_geoms = _included_geom_names(fixture / "box_body.xml", prefix=object_name)
    scene_geoms = object_geoms + (("ground",) if include_ground else ())
    metadata_geoms = robot.metadata.get("holosoma_robot_geom_names")
    robot_geoms = (
        tuple(str(name) for name in metadata_geoms)
        if isinstance(metadata_geoms, (list, tuple))
        else tuple(robot.link_names)
    )
    robot_geoms = tuple(name for name in robot_geoms if name not in scene_geoms and not name.startswith(object_name))
    return tuple((robot_geom, scene_geom) for robot_geom in robot_geoms for scene_geom in scene_geoms)


def holosoma_climb_profile(
    *,
    qpos_size: int,
    geometry_pairs: tuple[tuple[str, str], ...] = (),
) -> OptimizationProfile:
    """Return the typed optimization profile matching Holosoma's climbing defaults."""

    qpos_weights = np.zeros(int(qpos_size), dtype=np.float64)
    for qpos_idx, weight in G1_MANUAL_QPOS_COSTS.items():
        if qpos_idx < qpos_weights.shape[0]:
            qpos_weights[qpos_idx] = float(weight)
    constraints: tuple[ConstraintConfigUnion, ...] = (
        JointLimitsConstraintConfig(),
        TrustRegionConstraintConfig(radius=0.2),
        FootStickingConstraintConfig(tolerance=1e-3),
    )
    if geometry_pairs:
        constraints = (
            *constraints,
            NonPenetrationConstraintConfig(
                sources=("geometry",),
                geometry_source="backend_candidates",
                tolerance=1e-3,
                scene_clearance=1e-3,
                activation_distance=COLLISION_DETECTION_THRESHOLD,
                geometry_pairs=geometry_pairs,
                scene_geometry_keywords=("multi_boxes", "ground"),
                excluded_geometry_keyword_pairs=(("multi_boxes", "ground"),),
            ),
        )
    return OptimizationProfile(
        name="holosoma_climb",
        objectives=(
            LaplacianObjectiveConfig(weight=10.0),
            NominalTrackingObjectiveConfig(
                weight=5.0,
                qpos_indices=G1_NOMINAL_TRACKING_QPOS_INDICES,
                fallback="current",
            ),
            DiagonalRegularizationObjectiveConfig(weight=1.0, qpos_weights=tuple(float(v) for v in qpos_weights)),
            SmoothnessObjectiveConfig(weight=0.2),
        ),
        constraints=constraints,
        metadata={"source": "holosoma"},
    )


def holosoma_initial_qpos_plan(
    *,
    frame_count: int,
    q_init: np.ndarray,
    object_poses_mujoco: np.ndarray,
) -> InitialQposPlan:
    """Return Holosoma's ``q_locked_list`` seed for an original climbing run."""

    q_init_arr = np.asarray(q_init, dtype=np.float64)
    if q_init_arr.shape != (7 + G1_DOF,):
        raise ValueError(f"q_init must have shape ({7 + G1_DOF},)")
    object_poses = np.asarray(object_poses_mujoco, dtype=np.float64)
    if object_poses.shape != (int(frame_count), 7):
        raise ValueError("object_poses_mujoco must have shape (frame_count, 7)")
    qpos = np.zeros((int(frame_count), 7 + G1_DOF), dtype=np.float64)
    qpos[0, :] = q_init_arr
    qpos[:, -7:] = object_poses
    return InitialQposPlan.from_array(
        qpos,
        provenance={"source": "holosoma_q_locked_list", "object_pose_overlay": "last_7_qpos"},
    )


def _dummy_object_poses(frame_count: int) -> np.ndarray:
    poses = np.zeros((int(frame_count), 7), dtype=np.float64)
    poses[:, 0] = 1.0
    return poses


def _foot_sticking_states(
    human_joints: np.ndarray,
    *,
    demo_joints: tuple[str, ...],
    velocity_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(human_joints, dtype=np.float64)
    left_idx = demo_joints.index("LeftToeBase")
    right_idx = demo_joints.index("RightToeBase")
    left_vel = np.linalg.norm(np.diff(joints[:, left_idx, :2], axis=0), axis=1)
    right_vel = np.linalg.norm(np.diff(joints[:, right_idx, :2], axis=0), axis=1)
    left_vel = np.concatenate([[velocity_threshold + 1.0], left_vel])
    right_vel = np.concatenate([[velocity_threshold + 1.0], right_vel])
    return (
        (left_vel <= velocity_threshold).astype(np.int16),
        (right_vel <= velocity_threshold).astype(np.int16),
    )


def _mjcf_joint_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    names = [
        str(elem.attrib["name"])
        for elem in root.iter("joint")
        if elem.attrib.get("type") != "free" and "name" in elem.attrib
    ]
    if len(names) != G1_DOF:
        raise ValueError(f"{xml_path} contains {len(names)} named actuated joints, expected {G1_DOF}")
    return names


def _mjcf_body_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [str(elem.attrib["name"]) for elem in root.iter("body") if "name" in elem.attrib]


def _mjcf_geom_names(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    return [str(elem.attrib["name"]) for elem in root.iter("geom") if "name" in elem.attrib]


def _mjcf_joint_limits(xml_path: Path, joint_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    root = ET.parse(xml_path).getroot()
    valid = set(joint_names)
    out: dict[str, tuple[float, float]] = {}
    for elem in root.iter("joint"):
        name = elem.attrib.get("name")
        raw_range = elem.attrib.get("range")
        if name not in valid or raw_range is None:
            continue
        lower, upper = (float(value) for value in raw_range.split())
        out[str(name)] = (lower, upper)
    return out


def _apply_manual_qpos_bounds(
    joint_limits: Mapping[str, tuple[float, float]],
    *,
    joint_names: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    out = dict(joint_limits)
    for qpos_idx, lower in G1_MANUAL_LOWER_QPOS.items():
        joint_name = _joint_name_for_qpos(qpos_idx, joint_names)
        if joint_name is None:
            continue
        _old_lower, old_upper = out.get(joint_name, (-1e6, 1e6))
        out[joint_name] = (float(lower), old_upper)
    for qpos_idx, upper in G1_MANUAL_UPPER_QPOS.items():
        joint_name = _joint_name_for_qpos(qpos_idx, joint_names)
        if joint_name is None:
            continue
        old_lower, _old_upper = out.get(joint_name, (-1e6, 1e6))
        out[joint_name] = (old_lower, float(upper))
    return out


def _joint_name_for_qpos(qpos_idx: int, joint_names: tuple[str, ...]) -> str | None:
    joint_idx = int(qpos_idx) - 7
    if joint_idx < 0 or joint_idx >= len(joint_names):
        return None
    return joint_names[joint_idx]


def _included_geom_names(xml_path: Path, *, prefix: str) -> tuple[str, ...]:
    if not xml_path.exists():
        return ()
    root = ET.parse(xml_path).getroot()
    return tuple(
        str(elem.attrib["name"])
        for elem in root.iter("geom")
        if "name" in elem.attrib and str(elem.attrib["name"]).startswith(prefix)
    )


__all__ = [
    "COLLISION_DETECTION_THRESHOLD",
    "FOOT_STICKING_VELOCITY_THRESHOLD",
    "G1_DOF",
    "G1_FOOT_STICKING_LINKS",
    "G1_HEIGHT_M",
    "G1_NOMINAL_TRACKING_QPOS_INDICES",
    "MOCAP_DEMO_JOINTS",
    "MOCAP_DOWNSAMPLE",
    "MOCAP_HUMAN_HEIGHT_M",
    "MOCAP_TO_G1_LINK_MAPPING",
    "HolosomaClimbPreparation",
    "compute_climb_q_init",
    "convert_object_poses_to_mujoco_order",
    "default_holosoma_root",
    "ensure_g1_model_assets",
    "foot_sticking_contact_plan",
    "from_mocap_climb_fixture",
    "g1_spherehand_robot",
    "holosoma_climb_profile",
    "holosoma_initial_qpos_plan",
    "mocap_motion_format",
    "object_non_penetration_geometry_pairs",
    "preprocess_mocap_climb",
    "preprocess_object_poses",
    "sample_multi_boxes_like_holosoma",
    "transform_from_human_to_world",
]
