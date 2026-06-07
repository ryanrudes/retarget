# Motion And Capture Input

There are two input levels:

- native recordings preserve independent clocks and frames;
- `SceneObservation` provides one target-independent timeline and world frame.

`MotionSequence` is created during robot adaptation. It is not the container for
heterogeneous capture or semantic contacts.

## Native Recordings

Use typed sources for capture:

```python
from retarget import (
    GvhmrOutputSource,
    HumanPoseSourceSchema,
    ViconBagSource,
    ViconSourceSchema,
)

vicon = ViconBagSource(
    path="/data/trial",
    schema=ViconSourceSchema(
        rigid_bodies={"Left_Shoe": Body.LEFT_SHOE},
        markers={},
    ),
)
pose = GvhmrOutputSource(
    path="/data/gvhmr/trial",
    schema=HumanPoseSourceSchema(joint_indices=joint_indices),
    fps=59.94,
)
```

Each recording owns timestamps, frame, enum vocabulary, validity, and
provenance. An observation recipe aligns and resamples the sources.

## Registered Motion Files

For an already coherent joint trajectory, use
`MotionFileObservationRecipe`:

```python
from retarget import MotionFileObservationRecipe, MotionFormat, motion_formats

observation_recipe = MotionFileObservationRecipe(
    path="tests/fixtures/minimal_motion.json",
    motion_format=motion_formats.get(MotionFormat.MINIMAL),
)
observation = observation_recipe.observe()
```

Built-in loaders support `.json`, `.npz`, `.npy`, and `.csv`. Formats declare a
`MotionJoint` enum, root joint, coordinate convention, quaternion order, and
default sampling facts.

## In-Memory Recordings

Construct `HumanPoseRecording`, `MocapRecording`, or `VideoRecording` directly
for generated data and tests. Wrap them in `InMemorySource` when a recipe
expects an `ObservationSource[T]`.

Track resampling is type-specific. See
[Coordinate conventions](../coordinate-conventions.md) for clock transforms,
SLERP, and frame conversion.

## Explicit Checkpoints

```python
observation.save_npz("trial_observation.npz")
restored = SceneObservation.load_npz("trial_observation.npz")
```

Checkpointing is optional. Sources and recipes never write an intermediate
artifact automatically.

## Custom Input

- subclass `MotionJoint`, `MocapRigidBody`, `MocapMarker`, or
  `ObservationRole` for the source vocabulary;
- implement `ObservationSource[T]` for a new storage boundary;
- implement `HumanPoseEstimator` for a replaceable video estimator;
- implement `ObservationRecipe` for domain-specific fusion.

See [Add a motion format](../adding-a-motion-format.md) for coherent motion
files and [Native schemas](../pipeline/schemas.md) for native capture.
