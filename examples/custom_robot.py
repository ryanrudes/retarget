"""Define a typed robot vocabulary and run it through the experiment API."""

import sys
from pathlib import Path

from retarget import (
    MinimalMotionJoint,
    MotionFileObservationRecipe,
    MotionFormat,
    RetargetingExperiment,
    RobotKind,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
    TaskKind,
    motion_formats,
    robots,
)
from retarget.robots import RobotSpec, RobotVocabulary, SimpleKinematicPoint

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.vocabularies import (  # noqa: E402
    DemoRobotGeometry,
    DemoRobotJoint,
    DemoRobotLink,
    DemoRobotRole,
)


class DemoRobot(RobotKind):
    THREE_POINT_BOT = "three_point_bot"


@robots.register(DemoRobot.THREE_POINT_BOT)
def three_point_bot() -> RobotSpec[
    DemoRobotJoint,
    DemoRobotLink,
    DemoRobotGeometry,
    DemoRobotRole,
]:
    return RobotSpec(
        name=DemoRobot.THREE_POINT_BOT.value,
        height_m=1.0,
        vocabulary=RobotVocabulary(
            joints=DemoRobotJoint,
            links=DemoRobotLink,
            geometries=DemoRobotGeometry,
            roles=DemoRobotRole,
        ),
        joints=tuple(DemoRobotJoint),
        links=tuple(DemoRobotLink),
        contact_links=(DemoRobotLink.LEFT_FOOT, DemoRobotLink.RIGHT_FOOT),
        joint_limits={joint: (-1.0, 1.0) for joint in DemoRobotJoint},
        link_roles={
            DemoRobotRole.PELVIS: DemoRobotLink.PELVIS,
            DemoRobotRole.LEFT_FOOT: DemoRobotLink.LEFT_FOOT,
            DemoRobotRole.RIGHT_FOOT: DemoRobotLink.RIGHT_FOOT,
        },
        simple_kinematics={
            DemoRobotLink.PELVIS: SimpleKinematicPoint(
                joint=DemoRobotJoint.ROOT_SWAY,
                offset=(0.0, 0.0, 0.0),
                axis=(0.0, 1.0, 0.0),
            ),
            DemoRobotLink.LEFT_FOOT: SimpleKinematicPoint(
                joint=DemoRobotJoint.LEFT_LEG,
                offset=(-0.15, 0.0, -0.8),
                axis=(0.0, 0.0, 1.0),
            ),
            DemoRobotLink.RIGHT_FOOT: SimpleKinematicPoint(
                joint=DemoRobotJoint.RIGHT_LEG,
                offset=(0.15, 0.0, -0.8),
                axis=(0.0, 0.0, 1.0),
            ),
        },
    )


motion_format = motion_formats.get(MotionFormat.MINIMAL)
result = RetargetingExperiment(
    observation=MotionFileObservationRecipe(
        path=REPO_ROOT / "tests" / "fixtures" / "minimal_motion.json",
        motion_format=motion_format,
        name="custom_robot",
    ),
    recipe=RoleRetargetingRecipe(
        task_kind=TaskKind.ROBOT_ONLY,
        motion_format=motion_format,
        scene=RobotOnlySceneRecipe(),
        link_roles={
            MinimalMotionJoint.PELVIS: DemoRobotRole.PELVIS,
            MinimalMotionJoint.LEFT_TOE: DemoRobotRole.LEFT_FOOT,
            MinimalMotionJoint.RIGHT_TOE: DemoRobotRole.RIGHT_FOOT,
        },
    ),
    robot=robots.get(DemoRobot.THREE_POINT_BOT),
).run()
result.save_npz("custom_robot.npz")
print(result.qpos.shape)
