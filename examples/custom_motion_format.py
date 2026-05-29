"""Register a custom motion format."""

from retarget.motion import MotionFormatSpec, motion_formats


@motion_formats.register("my_format")
def my_format() -> MotionFormatSpec:
    return MotionFormatSpec(
        name="my_format",
        joint_names=("root", "left_toe", "right_toe"),
        root_joint="root",
        contact_joints=("left_toe", "right_toe"),
        default_height_m=1.75,
    )

print(motion_formats.get("my_format"))
