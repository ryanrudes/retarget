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
        joint_names=(DemoMotionJoint.ROOT, DemoMotionJoint.LEFT_TOE, DemoMotionJoint.RIGHT_TOE),
        root_joint=DemoMotionJoint.ROOT,
        contact_joints=(DemoMotionJoint.LEFT_TOE, DemoMotionJoint.RIGHT_TOE),
        default_height_m=1.75,
    )

print(motion_formats.get(DemoMotionFormat.MINIMAL_LOWER_BODY))
