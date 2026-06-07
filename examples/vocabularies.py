"""Typed vocabularies shared by serialized and programmatic robot examples."""

from retarget import RobotGeometry, RobotJoint, RobotLink, RobotRole


class DemoRobotRole(RobotRole):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class DemoRobotJoint(RobotJoint):
    ROOT_SWAY = "root_sway"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"


class DemoRobotLink(RobotLink):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


class DemoRobotGeometry(RobotGeometry):
    """This fixture robot has no model-backed geometry."""
