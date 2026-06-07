# Custom Capture Schemas

Define native and semantic vocabularies once as enum subclasses.

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

Bind native storage names or array columns with typed source schemas:

```python
from retarget import HumanPoseSourceSchema, ViconSourceSchema

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

A custom `ObservationSource[T]` loads one recording and preserves its native
timeline. A custom `ObservationRecipe` combines sources and returns
`SceneObservation`. Keep clock estimation, interpolation, frame conversion, and
registration inside that recipe or reusable strategies it composes.

Semantic contacts use `ContactSubject`, `ContactPatch`, and `ContactState`
subclasses. They remain robot-independent. The adaptation recipe maps semantic
subjects or observation roles to `RobotRole` members, and `RobotSpec` resolves
those roles to concrete model names.

Behavior belongs in typed fields. `metadata` is reserved for provenance and
diagnostics; it must not carry link mappings, support planes, geometry policies,
or other runtime semantics.
