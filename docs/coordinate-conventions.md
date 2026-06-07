# Coordinate Conventions

Every native recording owns its coordinate frame and timestamps. Capture sources
must report those facts explicitly; they must not pre-align data or imply that
two recordings share a clock.

`SceneObservation` is the canonical fusion boundary:

- all tracks use one `SampleTimeline`;
- all geometry uses one `world_frame`;
- positions are measured in meters;
- rotations carry an explicit `QuaternionOrder`.

Observation recipes perform temporal and spatial registration before creating
that object. Retargeting recipes can therefore consume an observation without
knowing which sensors produced it.

## Frames

The built-in conventions are:

- `FrameConvention.Z_UP_RIGHT_HANDED`, the internal default;
- `FrameConvention.Y_UP_RIGHT_HANDED`, common for video pose estimators.

The Y-up to Z-up point conversion is `(x, y, z) -> (x, -z, y)`. Use
`convert_points_frame`, `Pose.to_frame`, or `PoseSequence.to_frame`; do not
reproduce axis permutations inside a recipe.

Quaternion storage order and coordinate-frame conversion are independent:

```python
from retarget import QuaternionOrder, reorder_quaternions

wxyz = reorder_quaternions(
    xyzw,
    QuaternionOrder.XYZW,
    QuaternionOrder.WXYZ,
)
```

`reorder_quaternions` operates over the last axis by default and accepts a
custom `axis` for other array layouts.

## Timelines

`SampleTimeline` accepts irregular timestamps. A `ClockTransform` maps one
native clock into observation time with an affine transform:

```python
from retarget import ClockTransform

camera_to_world = ClockTransform(scale=1.0002, offset_s=-0.14)
```

Track classes own resampling behavior:

- `PointTrack`, `MarkerTrack`, and `JointTrack` use linear interpolation;
- `PoseTrack` uses linear translation and quaternion SLERP;
- `CategoricalTrack` uses nearest-neighbor sampling.

Registration strategies return typed `AlignmentReport` objects. Failed quality
gates raise `AlignmentError` carrying the report; zero lag is never assumed as
a fallback.

## Persistence

`SceneObservation.save_npz()` and `SceneObservation.load_npz()` are explicit
inspection checkpoints. Normal experiments remain in memory, and no capture,
registration, or recipe stage writes a cache automatically.
