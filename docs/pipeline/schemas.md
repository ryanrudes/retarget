# Native Schemas

Define native and semantic vocabularies once:

```python
from retarget import MocapRigidBody, MotionJoint, ObservationRole


class Bodies(MocapRigidBody):
    LEFT_SHOE = "Left_Shoe"
    RIGHT_SHOE = "Right_Shoe"


class PoseJoint(MotionJoint):
    PELVIS = "Pelvis"
    LEFT_ANKLE = "L_Ankle"
    RIGHT_ANKLE = "R_Ankle"


class Landmark(ObservationRole):
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
```

Source schemas are external-format boundaries, so native column names remain
strings while values are typed enum members:

```python
vicon_schema = ViconSourceSchema(
    rigid_bodies={body.value: body for body in Bodies},
    markers={},
)
pose_schema = HumanPoseSourceSchema(
    joint_indices={
        PoseJoint.PELVIS: 0,
        PoseJoint.LEFT_ANKLE: 7,
        PoseJoint.RIGHT_ANKLE: 8,
    }
)
```

Behavior belongs in typed recipe fields. `provenance` is limited to origin,
versions, source files, and processing history.
