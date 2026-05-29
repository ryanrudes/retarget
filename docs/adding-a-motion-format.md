# Add A Motion Format

Create a `MotionFormatSpec` with ordered joint names, root joint, optional contact joints, frame convention, and default height or fps. Register it with a decorated factory:

```python
from retarget.motion import MotionFormatSpec, motion_formats

@motion_formats.register("my_format")
def my_format() -> MotionFormatSpec:
    return MotionFormatSpec(
        name="my_format",
        joint_names=("root", "left_toe", "right_toe"),
        root_joint="root",
        contact_joints=("left_toe", "right_toe"),
        frame_convention="y_up_right_handed",
        default_height_m=1.75,
    )
```

Loaders are separate from formats. Add a new suffix loader by implementing `MotionLoader` and registering it in `motion_loaders`.

Registered format `frame_convention` is honored by `load_motion`: loader output is converted to internal `z_up_right_handed` coordinates before retargeting. Custom loaders should return coordinates in the declared format frame, or set `MotionSequence.frame` explicitly when the file carries its own convention.

Root poses are optional but should be loaded when a dataset provides global root translation and orientation. `MotionSequence.root_poses` stores them as a `PoseSequence`, and the retargeter uses those poses to initialize the root qpos. When root poses are absent, the root qpos translation falls back to the format's `root_joint` and the root orientation is identity.

`contact_joints` can be empty for formats without reliable contact labels, or include any number of joints for hands, knees, toes, or dataset-specific markers. `MotionSequence.contacts` stores optional per-frame labels as dictionaries such as `{"left_toe": true}`. When labels are present, contact constraints and metrics use them directly; otherwise the built-in helper infers contacts from contact-joint velocity and filters names that are absent from the loaded motion.

Built-in loaders:

- `.json`: small fixture-style mappings with `joint_positions` plus optional `root_positions`/`root_quaternions`, `contacts`, or `contact_states`.
- `.npy`: raw `(frames, joints, 3)` arrays using the registered format's joint order.
- `.npz`: common array archives with `global_joint_positions`, `joint_positions`, or `joints`; root poses can use `root_positions`/`root_translations` and `root_quaternions`; contact labels can use `contacts`/`contact_states` with `contact_names`.
- `.csv`: wide tables with one row per frame and columns like `Pelvis_x`, `Pelvis_y`, `Pelvis_z`.

CSV files may include `frame`, `time_s`, `fps`, `height_m`, root pose columns named `root_position_x`, `root_position_y`, `root_position_z`, `root_quaternion_w`, `root_quaternion_x`, `root_quaternion_y`, `root_quaternion_z`, and contact columns named `{joint}_contact`, `contact_{joint}`, `{joint}_in_contact`, or `is_contact_{joint}`. When `time_s` is present, FPS is inferred from the average positive time step. If `frame` is present, rows are sorted by frame index before loading.

Example CSV header for the `minimal` format:

```text
frame,time_s,height_m,root_position_x,root_position_y,root_position_z,root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,Pelvis_x,Pelvis_y,Pelvis_z,L_Hip_x,L_Hip_y,L_Hip_z,...
```
