# Add A Robot

A robot defines concrete enum vocabularies and binds semantic robot roles to
those members.

```python
from retarget import (
    RobotGeometry,
    RobotJoint,
    RobotKind,
    RobotLink,
    RobotRole,
)
from retarget.robots import RobotSpec, RobotVocabulary, robots


class LabRobot(RobotKind):
    BIPED = "lab_biped"


class LabRole(RobotRole):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"


class LabJoint(RobotJoint):
    HIP = "hip"
    KNEE = "knee"


class LabLink(RobotLink):
    PELVIS = "pelvis_link"
    LEFT_FOOT = "left_foot_link"


class LabGeometry(RobotGeometry):
    LEFT_FOOT = "left_foot_geom"


@robots.register(LabRobot.BIPED)
def lab_robot() -> RobotSpec[LabJoint, LabLink, LabGeometry, LabRole]:
    return RobotSpec(
        name="lab_biped",
        height_m=1.0,
        vocabulary=RobotVocabulary(
            joints=LabJoint,
            links=LabLink,
            geometries=LabGeometry,
            roles=LabRole,
        ),
        joints=tuple(LabJoint),
        links=tuple(LabLink),
        geometries=tuple(LabGeometry),
        contact_links=(LabLink.LEFT_FOOT,),
        joint_limits={
            LabJoint.HIP: (-1.0, 1.0),
            LabJoint.KNEE: (-2.0, 0.0),
        },
        joint_roles={LabRole.PELVIS: LabJoint.HIP},
        link_roles={
            LabRole.PELVIS: LabLink.PELVIS,
            LabRole.LEFT_FOOT: LabLink.LEFT_FOOT,
        },
    )
```

`RobotSpec` never normalizes these members to strings. Passing a member from a
different enum class fails even if its value matches.

Nominal tracking selections belong in an optimization profile, not the robot
spec. MuJoCo body aliases remain strings because they are foreign model
identifiers.

## File Specs

Serialized robot specs reference importable enum classes explicitly:

```toml
name = "three_point_bot"
height_m = 1.0
joints = ["root_sway", "left_leg", "right_leg"]
links = ["pelvis", "left_foot", "right_foot"]
contact_links = ["left_foot", "right_foot"]
geometries = []

[vocabulary]
joints = "examples.vocabularies:DemoRobotJoint"
links = "examples.vocabularies:DemoRobotLink"
geometries = "examples.vocabularies:DemoRobotGeometry"
roles = "examples.vocabularies:DemoRobotRole"
```

The strings are deserialized into members of those exact classes. See
`examples/custom_robot.toml` for the complete file and
`examples/custom_robot.py` for the programmatic equivalent.

```python
from retarget.robots import RobotSpec

robot = RobotSpec.load("examples/custom_robot.toml")
```
