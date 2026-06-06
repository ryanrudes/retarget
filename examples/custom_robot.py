"""Register a custom robot spec."""

from retarget import MotionJoint, NameEnum, RobotJoint, RobotLink
from retarget.robots import RobotSpec, robots


class DemoRobot(NameEnum):
    TWO_JOINT_BOT = "two_joint_bot"


class DemoMotionJoint(MotionJoint):
    ROOT = "root"
    LEFT_TOE = "left_toe"


class DemoRobotJoint(RobotJoint):
    HIP = "hip"
    KNEE = "knee"


class DemoRobotLink(RobotLink):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"


@robots.register(DemoRobot.TWO_JOINT_BOT)
def two_joint_bot() -> RobotSpec:
    return RobotSpec(
        name=DemoRobot.TWO_JOINT_BOT,
        dof=2,
        height_m=1.0,
        joint_names=(DemoRobotJoint.HIP, DemoRobotJoint.KNEE),
        link_names=(DemoRobotLink.PELVIS, DemoRobotLink.LEFT_FOOT),
        contact_links=(DemoRobotLink.LEFT_FOOT,),
        joint_limits={DemoRobotJoint.HIP: (-1.0, 1.0), DemoRobotJoint.KNEE: (-2.0, 0.0)},
        default_joint_mapping={DemoMotionJoint.ROOT: DemoRobotJoint.HIP, DemoMotionJoint.LEFT_TOE: DemoRobotJoint.KNEE},
        default_link_mapping={
            DemoMotionJoint.ROOT: DemoRobotLink.PELVIS,
            DemoMotionJoint.LEFT_TOE: DemoRobotLink.LEFT_FOOT,
        },
    )

print(robots.get(DemoRobot.TWO_JOINT_BOT))
