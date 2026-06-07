# Add A Robot

`RobotSpec` binds a robot-specific joint and link vocabulary to semantic roles.
Observation and retargeting recipes depend on roles, not spelling conventions
or side-name heuristics.

```python
from retarget import RobotJoint, RobotLink, RobotRole
from retarget.robots import RobotSpec, robots


class LabRole(RobotRole):
    PELVIS = "pelvis"
    LEFT_FOOT = "left_foot"


class LabJoint(RobotJoint):
    HIP = "hip"
    KNEE = "knee"


class LabLink(RobotLink):
    PELVIS = "pelvis_link"
    LEFT_FOOT = "left_foot_link"


@robots.register("lab_robot")
def lab_robot() -> RobotSpec:
    return RobotSpec(
        name="lab_robot",
        dof=2,
        height_m=1.0,
        joint_names=tuple(LabJoint),
        link_names=tuple(LabLink),
        contact_links=(LabLink.LEFT_FOOT,),
        joint_limits={
            LabJoint.HIP: (-1.0, 1.0),
            LabJoint.KNEE: (-2.0, 0.0),
        },
        role_vocabulary=LabRole,
        joint_roles={LabRole.PELVIS: LabJoint.HIP},
        link_roles={
            LabRole.PELVIS: LabLink.PELVIS,
            LabRole.LEFT_FOOT: LabLink.LEFT_FOOT,
        },
    )
```

`RobotSpec` normalizes enum values for serialization while retaining and
validating the declared `role_vocabulary`. A retargeting recipe must pass
members of that exact role enum to `joint_for_role`, `link_for_role`, and
`links_for_role`.

Geometry names, MuJoCo body aliases, asset paths, qpos layout, and nominal joint
selection are explicit fields. `metadata` is provenance-only and must not
control behavior.

## File Specs

TOML and YAML cannot serialize Python enum classes, so file-backed specs use
the standard `HumanoidRobotRole` vocabulary and its string values:

```toml
name = "two_joint_bot"
dof = 2
height_m = 1.0
joint_names = ["hip", "knee"]
link_names = ["pelvis", "left_foot"]
contact_links = ["left_foot"]

[joint_roles]
pelvis = "hip"
left_knee = "knee"

[link_roles]
pelvis = "pelvis"
left_foot = "left_foot"
```

Load file and asset-store specs through the same provider API:

```python
from retarget.robots import robot_providers

robot = robot_providers.get("file").load(
    "two_joint_bot",
    path="examples/custom_robot.toml",
)
```

Use a custom Python `RobotSpec` when the standard humanoid role vocabulary is
not appropriate. Asset paths remain explicit fields regardless of provider.
