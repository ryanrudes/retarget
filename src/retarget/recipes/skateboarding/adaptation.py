"""Robot adaptation recipe for skateboarding observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.core.enums import (
    HumanoidRobotRole,
    NonPenetrationSource,
    SolverBackend,
    TaskKind,
)
from retarget.core.pose import PoseSequence
from retarget.motion import MotionFormatSpec, MotionSequence
from retarget.observation import SceneObservation
from retarget.optimization.spec import (
    FootStickingConstraintConfig,
    JointLimitsConstraintConfig,
    LinkTrackingObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    SmoothnessObjectiveConfig,
    SolverSpec,
    TrustRegionConstraintConfig,
)
from retarget.pipeline import RetargetingProblem
from retarget.robots.spec import RobotSpec

from .scene import runtime_scene
from .schema import JOINT_TO_ROBOT_ROLE, LINK_TO_ROBOT_ROLE
from .targets import skateboarding_link_targets
from .vocabulary import (
    SkateboardingContactSubject,
    SkateboardingMotionJoint,
)


@dataclass(frozen=True)
class SkateboardingRetargetingRecipe:
    """Adapt a skateboarding observation to a semantic humanoid robot."""

    scale_to_robot: bool = False
    output_fps: float | None = None
    show_progress: bool = False
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL

    def build_problem(
        self,
        observation: SceneObservation,
        robot: RobotSpec,
    ) -> RetargetingProblem:
        """Build the complete robot-resolved skateboarding problem."""

        fps = observation.timeline.nominal_fps
        if fps is None:
            raise ValueError("skateboarding observations require at least two samples")
        motion = _motion_from_observation(observation, fps=fps)
        contacts = (
            observation.contacts.resolve(
                {
                    SkateboardingContactSubject.LEFT_SHOE: (robot.link_for_role(HumanoidRobotRole.LEFT_FOOT),),
                    SkateboardingContactSubject.RIGHT_SHOE: (robot.link_for_role(HumanoidRobotRole.RIGHT_FOOT),),
                }
            )
            if observation.contacts is not None
            else None
        )
        contact_links = tuple(
            dict.fromkeys(
                (
                    robot.link_for_role(HumanoidRobotRole.LEFT_FOOT),
                    robot.link_for_role(HumanoidRobotRole.RIGHT_FOOT),
                )
            )
        )
        return RetargetingProblem(
            name=observation.name,
            task_kind=TaskKind.OBJECT_INTERACTION,
            robot=robot,
            motion=motion,
            scene=runtime_scene(observation, fps=fps),
            contacts=contacts,
            targets=skateboarding_link_targets(observation, robot),
            motion_format=MotionFormatSpec(
                name="skateboarding_observation",
                joint_vocabulary=SkateboardingMotionJoint,
                root_joint=SkateboardingMotionJoint.PELVIS,
                default_fps=fps,
                default_height_m=observation.actor.source_height_m,
            ),
            joint_mapping={joint.value: robot.joint_for_role(role) for joint, role in JOINT_TO_ROBOT_ROLE.items()},
            link_mapping={joint.value: robot.link_for_role(role) for joint, role in LINK_TO_ROBOT_ROLE.items()},
            solver=SolverSpec(
                backend=self.solver_backend,
                max_iterations=10,
                trust_radius=0.2,
            ),
            objectives=(
                LinkTrackingObjectiveConfig(weight=1.0),
                SmoothnessObjectiveConfig(weight=0.2),
                NominalTrackingObjectiveConfig(weight=5.0),
            ),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(),
                FootStickingConstraintConfig(tolerance=1e-3),
                NonPenetrationConstraintConfig(
                    sources=(NonPenetrationSource.SCENE_POINTS,),
                    links=contact_links,
                    scene_clearance=0.015,
                    activation_distance=0.05,
                ),
            ),
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps or fps,
            show_progress=self.show_progress,
            metadata={
                "recipe": "skateboarding",
                "observation": observation.name,
            },
        )


def _motion_from_observation(
    observation: SceneObservation,
    *,
    fps: float,
) -> MotionSequence:
    joint_positions = np.stack(
        [track.values for track in observation.actor.joints],
        axis=1,
    )
    pelvis = observation.actor.joint(SkateboardingMotionJoint.PELVIS).values
    root_quaternions = np.zeros(
        (observation.timeline.sample_count, 4),
        dtype=np.float64,
    )
    root_quaternions[:, 0] = 1.0
    return MotionSequence(
        name=observation.name,
        joint_positions=joint_positions,
        joint_names=tuple(track.role.value for track in observation.actor.joints),
        fps=fps,
        frame=observation.world_frame,
        root_poses=PoseSequence.from_arrays(
            pelvis,
            root_quaternions,
            fps=fps,
            frame=observation.world_frame,
        ),
        source_height_m=observation.actor.source_height_m,
        metadata={"observation": observation.name},
    )
