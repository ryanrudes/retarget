"""Robot adaptation recipe for skateboarding observations."""

from __future__ import annotations

from dataclasses import dataclass

from retarget.core.enums import (
    HumanoidRobotRole,
    NonPenetrationSource,
    SolverBackend,
    TaskKind,
)
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
from retarget.pipeline import JointBinding, LinkBinding, RetargetingProblem
from retarget.robots.spec import RobotSpec

from .scene import runtime_scene
from .schema import JOINT_TO_ROBOT_ROLE, LINK_TO_ROBOT_ROLE
from .targets import skateboarding_link_targets
from .vocabulary import (
    SkateboardingContactSubject,
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
            motion=observation.actor,
            scene=runtime_scene(observation, fps=fps),
            contacts=contacts,
            targets=skateboarding_link_targets(observation, robot),
            joint_bindings=tuple(
                JointBinding(joint, robot.joint_for_role(role))
                for joint, role in JOINT_TO_ROBOT_ROLE.items()
            ),
            link_bindings=tuple(
                LinkBinding(joint, robot.link_for_role(role))
                for joint, role in LINK_TO_ROBOT_ROLE.items()
            ),
            solver=SolverSpec(
                backend=self.solver_backend,
                max_iterations=10,
                trust_radius=0.2,
            ),
            objectives=(
                LinkTrackingObjectiveConfig(weight=1.0),
                SmoothnessObjectiveConfig(weight=0.2),
                NominalTrackingObjectiveConfig(
                    weight=5.0,
                    joints=tuple(
                        robot.joint_for_role(role)
                        for role in (
                            HumanoidRobotRole.LEFT_HIP,
                            HumanoidRobotRole.LEFT_KNEE,
                            HumanoidRobotRole.LEFT_ANKLE,
                            HumanoidRobotRole.RIGHT_HIP,
                            HumanoidRobotRole.RIGHT_KNEE,
                            HumanoidRobotRole.RIGHT_ANKLE,
                            HumanoidRobotRole.TORSO,
                        )
                    ),
                ),
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
            provenance={
                "recipe": "skateboarding",
                "observation": observation.name,
            },
        )
