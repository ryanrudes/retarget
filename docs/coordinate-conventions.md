# Coordinate Conventions

Internal geometry uses explicit `Pose` and `PoseSequence` objects. Quaternions carry their storage order (`wxyz` or `xyzw`) and frame convention. Convert data at the boundary and keep internal code convention-aware.

All retargeting positions are in **meters** in the internal **Z-up right-handed** frame after loaders or adapters run.

Supported frame conventions:

- `z_up_right_handed`: internal default for all retargeting, metrics, visualization, and export code.
- `y_up_right_handed`: common input convention for game and mocap assets.

Use `convert_points_frame`, `Pose.to_frame`, or `PoseSequence.to_frame` when writing loaders or adapters. The built-in `load_motion` boundary converts any registered motion format into internal Z-up coordinates and records `frame_converted_from` / `frame_converted_to` metadata when a conversion occurs.

The Y-up to Z-up conversion maps points as `(x, y, z) -> (x, -z, y)`, preserving right-handed orientation. Quaternion storage order conversion is independent from frame conversion: use `reorder_quaternion` for `wxyz`/`xyzw` storage changes and `Pose.to_frame` for coordinate-frame changes.

Motion root poses follow the same boundary rule as joint positions. Built-in loaders read optional root pose arrays, convert them into `PoseSequence`, and `load_motion` converts them to internal Z-up coordinates together with the joints. Retargeting initializes qpos root translation and quaternion from `MotionSequence.root_poses` when available.

Timing is explicit too. Sequence timestamps start at zero, so `MotionSequence.duration_s` is the endpoint span `(frame_count - 1) / fps`. `PoseSequence.resampled()`, `MotionSequence.resampled()`, and `RetargetingResult.resampled()` interpolate onto a new FPS grid while preserving sequence endpoints. Pose rotations use spherical interpolation; joint positions, qpos arrays, costs, and point clouds use deterministic linear interpolation.

## Motion sync / synced clip

Upstream [motion_sync](https://github.com/ryanrudes/motion_sync) clips keep sensor-native frames until you export for retargeting:

- **Vicon** rigid bodies and markers in a `synced.npz` clip are **Z-up** (meters).
- **Video / SMPL-X** body joints on the same clip are typically **Y-up** (meters).
- Integration adapters (for example `retarget.integrations.motion_sync.skateboarding`) convert packed human motion to Z-up while building `PreparedRetargetInputs`.

See [Custom schemas](ecosystem/custom-schemas.md) for body/marker naming and [Introduction — Step 3](introduction.md#step-3-time-align-and-build-motionsequence) for the full align → convert → pack workflow. Register or adapt a motion format so `load_motion` applies the same Z-up boundary as the rest of the library ([Add a motion format](adding-a-motion-format.md)).
