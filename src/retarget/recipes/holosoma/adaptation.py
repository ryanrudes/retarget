"""Robot adaptation for the Holosoma climbing subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from retarget.core.enums import ConvergenceMode, ObjectQposMode, SolverBackend, TaskKind
from retarget.core.pose import PoseSequence
from retarget.mesh import InteractionMeshSpec, LaplacianWeighting
from retarget.motion import InitialQposPlan, MotionSequence
from retarget.observation import SceneObservation
from retarget.optimization import SolverSpec
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline import RetargetingProblem
from retarget.robots.spec import RobotSpec
from retarget.scene import ObjectTrajectory, SceneSpec

from .geometry import object_non_penetration_geometry_pairs
from .layout import default_holosoma_root, holosoma_climb_layout
from .motion import compute_climb_q_init, mocap_motion_format
from .object import ensure_scaled_multi_boxes_assets
from .profile import HolosomaClimbOptimizationPolicy, holosoma_climb_profile
from .vocabulary import (
    MOCAP_TO_ROBOT_ROLE,
    HolosomaContactSubject,
    HolosomaObservationRole,
    HolosomaRobotRole,
)


@dataclass(frozen=True)
class HolosomaClimbRetargetingRecipe:
    """Adapt a climb observation to a Holosoma-compatible robot."""

    holosoma_root: str | Path | None = None
    include_object_collision: bool = True
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL
    show_progress: bool = False
    optimization_policy: HolosomaClimbOptimizationPolicy = field(default_factory=HolosomaClimbOptimizationPolicy)

    def build_problem(
        self,
        observation: SceneObservation,
        robot: RobotSpec,
    ) -> RetargetingProblem:
        """Build the robot-resolved climb problem."""

        observed_object = observation.observed_object(HolosomaObservationRole.CLIMBING_STRUCTURE)
        scale = robot.height_m / float(observation.actor.source_height_m or robot.height_m)
        human_joints = (
            np.stack(
                [track.values for track in observation.actor.joints],
                axis=1,
            )
            * scale
        )
        object_positions = observed_object.pose.positions * scale
        object_quaternions = observed_object.pose.quaternions
        object_poses_mujoco = np.concatenate(
            [object_positions, object_quaternions],
            axis=1,
        )
        q_init = compute_climb_q_init(
            human_joints,
            np.concatenate([object_quaternions, object_positions], axis=1),
            demo_joints=tuple(track.role.value for track in observation.actor.joints),
            robot_dof=robot.dof,
        )
        root = (Path(self.holosoma_root) if self.holosoma_root is not None else default_holosoma_root()).resolve()
        layout = holosoma_climb_layout(root)
        asset_scale = (scale, scale, scale)
        scaled_urdf, scaled_scene_xml = ensure_scaled_multi_boxes_assets(
            fixture_dir=layout.fixture_dir,
            object_urdf_path=layout.object_urdf_path,
            scene_xml_path=layout.scene_xml_path,
            asset_scale=asset_scale,
        )
        problem_robot = robot.model_copy(update={"mujoco_xml_path": scaled_scene_xml})
        geometry_pairs = (
            object_non_penetration_geometry_pairs(
                problem_robot,
                fixture_dir=layout.fixture_dir,
            )
            if self.include_object_collision
            else ()
        )
        profile = holosoma_climb_profile(
            qpos_size=problem_robot.qpos_size(has_object=False),
            geometry_pairs=geometry_pairs,
            policy=self.optimization_policy,
        )
        fps = observation.timeline.nominal_fps
        if fps is None:
            raise ValueError("Holosoma observations require at least two samples")
        joint_names = tuple(track.role.value for track in observation.actor.joints)
        motion = MotionSequence(
            name=observation.name,
            joint_positions=human_joints,
            joint_names=joint_names,
            fps=fps,
            frame=observation.world_frame,
            root_poses=PoseSequence.from_arrays(
                np.tile(q_init[:3], (observation.timeline.sample_count, 1)),
                np.tile(q_init[3:7], (observation.timeline.sample_count, 1)),
                fps=fps,
                frame=observation.world_frame,
            ),
            source_height_m=robot.height_m,
            metadata={"observation": observation.name},
        )
        scene = SceneSpec.climbing(
            object_spec=observed_object.geometry.model_copy(
                update={
                    "urdf_path": scaled_urdf,
                    "asset_scale": asset_scale,
                    "trajectory": ObjectTrajectory(
                        name=observed_object.geometry.name,
                        poses=PoseSequence.from_arrays(
                            object_positions,
                            object_quaternions,
                            fps=fps,
                            quaternion_order=observed_object.pose.quaternion_order,
                            frame=observation.world_frame,
                        ),
                    ),
                    "qpos_mode": ObjectQposMode.EXTERNAL,
                }
            )
        )
        contacts = None
        if observation.contacts is not None:
            contacts = observation.contacts.resolve(
                {
                    HolosomaContactSubject.LEFT_FOOT: problem_robot.links_for_role(HolosomaRobotRole.LEFT_FOOT_CONTACT),
                    HolosomaContactSubject.RIGHT_FOOT: problem_robot.links_for_role(
                        HolosomaRobotRole.RIGHT_FOOT_CONTACT
                    ),
                }
            ).scaled(scale)
        initial_qpos = _initial_qpos(
            observation.timeline.sample_count,
            q_init,
            object_poses_mujoco,
        )
        link_mapping = {joint.value: problem_robot.link_for_role(role) for joint, role in MOCAP_TO_ROBOT_ROLE.items()}
        return RetargetingProblem(
            name=observation.name,
            task_kind=TaskKind.CLIMBING,
            robot=problem_robot,
            motion=motion,
            scene=scene,
            contacts=contacts,
            initial_qpos=initial_qpos,
            motion_format=mocap_motion_format(
                default_fps=fps,
                default_height_m=float(observation.actor.source_height_m or robot.height_m),
            ),
            link_mapping=link_mapping,
            variables=QposVariableSpec.holosoma_q_a(-7),
            mesh=InteractionMeshSpec(laplacian_weighting=LaplacianWeighting.UNIFORM),
            solver=SolverSpec(
                backend=self.solver_backend,
                max_iterations=10,
                first_frame_iterations=50,
                trust_radius=0.2,
                convergence=ConvergenceMode.NONE,
            ),
            objectives=profile.objectives,
            constraints=profile.constraints,
            scale_to_robot=False,
            show_progress=self.show_progress,
            metadata={
                "recipe": "holosoma_climb",
                "include_object_collision": self.include_object_collision,
            },
        )


def _initial_qpos(
    frame_count: int,
    q_init: np.ndarray,
    object_poses_mujoco: np.ndarray,
) -> InitialQposPlan:
    qpos = np.zeros((frame_count, q_init.shape[0]), dtype=np.float64)
    qpos[0] = q_init
    qpos[:, -7:] = object_poses_mujoco
    return InitialQposPlan.from_array(
        qpos,
        provenance={"source": "holosoma_locked_seed"},
    )
