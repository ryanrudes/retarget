"""Holosoma-compatible climb recipe orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core.enums import (
    ConvergenceMode,
    FrameConvention,
    ObjectQposMode,
    ObjectSampleSpace,
    QuaternionOrder,
    SolverBackend,
    TaskKind,
)
from retarget.core.pose import PoseSequence
from retarget.mesh import InteractionMeshSpec, LaplacianWeighting
from retarget.motion.contact import ContactPlan
from retarget.motion.qpos import InitialQposPlan
from retarget.motion.spec import MotionSequence
from retarget.optimization import SolverSpec
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import ObjectSpec, ObjectTrajectory, SceneSpec

from .contacts import foot_sticking_contact_plan
from .geometry import object_non_penetration_geometry_pairs
from .layout import default_holosoma_root, holosoma_climb_layout
from .motion import compute_climb_q_init, mocap_motion_format, preprocess_mocap_climb
from .object import (
    convert_object_poses_to_mujoco_order,
    dummy_object_poses,
    ensure_scaled_multi_boxes_assets,
    object_visual_parts_from_urdf,
    preprocess_object_poses,
    sample_multi_boxes_like_holosoma,
)
from .profile import holosoma_climb_profile
from .robot import ensure_g1_model_assets, g1_spherehand_robot
from .vocabulary import (
    COLLISION_DETECTION_THRESHOLD,
    G1_DOF,
    G1_HEIGHT_M,
    MOCAP_DEMO_JOINTS,
    MOCAP_DOWNSAMPLE,
    MOCAP_FPS,
    MOCAP_HUMAN_HEIGHT_M,
    MULTI_BOX_SAMPLE_COUNT,
    MULTI_BOX_SAMPLE_SEED,
    HolosomaGeometryName,
)


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


@dataclass(frozen=True)
class HolosomaClimbRecipe:
    """Strict typed recipe for Holosoma's MOCAP climbing subset."""

    holosoma_root: str | Path | None = None
    frame_count: int | None = None
    ensure_model_assets: bool = False
    include_object_collision: bool = True
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL
    show_progress: bool = False

    def prepare(self) -> HolosomaClimbPreparation:
        """Load source assets and build typed intermediates plus the problem."""

        return from_mocap_climb_fixture(
            self.holosoma_root,
            frame_count=self.frame_count,
            ensure_model_assets=self.ensure_model_assets,
            include_object_collision=self.include_object_collision,
            solver_backend=self.solver_backend,
            show_progress=self.show_progress,
        )

    def build_problem(self) -> RetargetingProblem:
        """Return the complete Holosoma climb retargeting problem."""

        return self.prepare().problem


def from_mocap_climb_fixture(
    holosoma_root: str | Path | None = None,
    *,
    frame_count: int | None = None,
    ensure_model_assets: bool = False,
    include_object_collision: bool = True,
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL,
    show_progress: bool = False,
) -> HolosomaClimbPreparation:
    """Build the typed retargeting problem for Holosoma's real MOCAP climb fixture."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    layout = holosoma_climb_layout(root)
    if ensure_model_assets:
        ensure_g1_model_assets(root)

    raw_motion = np.load(layout.motion_path)
    human_joints = raw_motion[::MOCAP_DOWNSAMPLE].astype(np.float64, copy=True)
    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        human_joints = human_joints[:frame_count]
    scale = G1_HEIGHT_M / MOCAP_HUMAN_HEIGHT_M
    object_poses = dummy_object_poses(len(human_joints))
    human_joints = preprocess_mocap_climb(human_joints, scale=scale)
    object_poses = preprocess_object_poses(object_poses, scale=scale)
    object_poses_mujoco = convert_object_poses_to_mujoco_order(object_poses)
    q_init = compute_climb_q_init(human_joints, object_poses, demo_joints=MOCAP_DEMO_JOINTS, robot_dof=G1_DOF)
    object_asset_scale = (scale, scale, scale)
    scaled_object_urdf_path, scaled_scene_xml_path = ensure_scaled_multi_boxes_assets(
        fixture_dir=layout.fixture_dir,
        object_urdf_path=layout.object_urdf_path,
        scene_xml_path=layout.scene_xml_path,
        asset_scale=object_asset_scale,
    )
    object_samples_source = sample_multi_boxes_like_holosoma(
        layout.object_mesh_path,
        sample_count=MULTI_BOX_SAMPLE_COUNT,
    )
    object_samples = object_samples_source * np.asarray(object_asset_scale, dtype=np.float64)
    robot = g1_spherehand_robot(
        root,
        scene_xml_path=scaled_scene_xml_path,
        include_object_collision=include_object_collision,
    )
    geometry_pairs = (
        object_non_penetration_geometry_pairs(robot, fixture_dir=layout.fixture_dir)
        if include_object_collision
        else ()
    )
    profile = holosoma_climb_profile(qpos_size=robot.qpos_size(has_object=False), geometry_pairs=geometry_pairs)
    initial_qpos = holosoma_initial_qpos_plan(
        frame_count=human_joints.shape[0],
        q_init=q_init,
        object_poses_mujoco=object_poses_mujoco,
    )

    root_poses = _root_poses_from_q_init(q_init, frame_count=human_joints.shape[0])
    motion = MotionSequence(
        name="holosoma_mocap_climb_seq_0",
        joint_positions=human_joints,
        joint_names=MOCAP_DEMO_JOINTS,
        fps=MOCAP_FPS,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
        root_poses=root_poses,
        source_height_m=MOCAP_HUMAN_HEIGHT_M,
        metadata={
            "source": "holosoma_fixture",
            "source_path": str(layout.motion_path),
            "raw_downsample": MOCAP_DOWNSAMPLE,
            "holosoma_scale": scale,
            "robot_height_m": G1_HEIGHT_M,
        },
    )
    scene = SceneSpec.climbing(
        object_spec=ObjectSpec(
            name=HolosomaGeometryName.MULTI_BOXES.value,
            mesh_path=layout.object_mesh_path,
            urdf_path=scaled_object_urdf_path,
            asset_scale=object_asset_scale,
            visual_parts=object_visual_parts_from_urdf(scaled_object_urdf_path),
            sample_points=object_samples_source,
            sample_space=ObjectSampleSpace.OBJECT_ASSET_LOCAL,
            trajectory=ObjectTrajectory(
                name=HolosomaGeometryName.MULTI_BOXES.value,
                poses=PoseSequence.from_arrays(
                    object_poses_mujoco[:, :3],
                    object_poses_mujoco[:, 3:],
                    fps=MOCAP_FPS,
                    quaternion_order=QuaternionOrder.WXYZ,
                    frame=FrameConvention.Z_UP_RIGHT_HANDED,
                ),
            ),
            qpos_mode=ObjectQposMode.EXTERNAL,
            metadata={"sample_count": MULTI_BOX_SAMPLE_COUNT, "sample_seed": MULTI_BOX_SAMPLE_SEED},
        )
    )
    contacts = foot_sticking_contact_plan(human_joints, demo_joints=MOCAP_DEMO_JOINTS)
    metadata = {
        "source": "holosoma",
        "task_type": "climbing",
        "data_format": "mocap",
        "fixture": str(layout.fixture_dir),
        "asset_scale": list(object_asset_scale),
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
            convergence=ConvergenceMode.NONE,
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


def _root_poses_from_q_init(q_init: np.ndarray, *, frame_count: int) -> PoseSequence:
    root_positions = np.tile(q_init[:3], (int(frame_count), 1))
    root_quaternions = np.tile(q_init[3:7], (int(frame_count), 1))
    return PoseSequence.from_arrays(
        root_positions,
        root_quaternions,
        fps=MOCAP_FPS,
        quaternion_order=QuaternionOrder.WXYZ,
        frame=FrameConvention.Z_UP_RIGHT_HANDED,
    )
