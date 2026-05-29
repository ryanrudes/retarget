"""Register a custom robot spec."""

from retarget.robots import RobotSpec, robots


@robots.register("two_joint_bot")
def two_joint_bot() -> RobotSpec:
    return RobotSpec(
        name="two_joint_bot",
        dof=2,
        height_m=1.0,
        joint_names=("hip", "knee"),
        link_names=("pelvis", "left_foot"),
        contact_links=("left_foot",),
        joint_limits={"hip": (-1.0, 1.0), "knee": (-2.0, 0.0)},
        default_joint_mapping={"root": "hip", "left_toe": "knee"},
        default_link_mapping={"root": "pelvis", "left_toe": "left_foot"},
    )

print(robots.get("two_joint_bot"))
