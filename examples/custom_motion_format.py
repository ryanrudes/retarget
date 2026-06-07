"""Register a custom motion format."""

from retarget import MotionJoint, RetargetEnum
from retarget.motion import MotionFormatSpec, motion_formats


class DemoMotionFormat(RetargetEnum):
    MINIMAL_LOWER_BODY = "minimal_lower_body"


class DemoMotionJoint(MotionJoint):
    ROOT = "root"
    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"


@motion_formats.register(DemoMotionFormat.MINIMAL_LOWER_BODY)
def my_format() -> MotionFormatSpec:
    return MotionFormatSpec(
        name=DemoMotionFormat.MINIMAL_LOWER_BODY,
        joint_vocabulary=DemoMotionJoint,
        root_joint=DemoMotionJoint.ROOT,
        default_height_m=1.75,
    )

print(motion_formats.get(DemoMotionFormat.MINIMAL_LOWER_BODY))
