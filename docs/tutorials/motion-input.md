# Motion input

Retargeting starts from a **`MotionSequence`**: named joints, positions shaped `(frames, joints, 3)`, frame rate, and optional root poses or metadata. This tutorial covers how files become that sequence and how to stay aligned with a **motion format** definition.

## Motion formats are contracts

A motion format declares:

- Which joint names appear in files
- How to parse JSON, NPZ, NPY, or CSV into arrays
- Optional metadata keys (for example `height_m` for scaling)

List registered formats:

```bash
uv run retarget doctor
```

The `minimal` format matches the tutorial fixture and the synthetic humanoid mapping. Research formats (for example SMPL-X) register separately and may need optional dependencies.

## Load a fixture file

=== "CLI"

    ```bash
    uv run retarget run \
      --motion tests/fixtures/minimal_motion.json \
      --format minimal \
      --robot synthetic_humanoid \
      --output from_json.npz
    ```

=== "Python"

    ```python
    from retarget.motion import load_motion

    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    print(motion.joint_names)
    print(motion.joint_positions.shape)  # (frames, joints, 3)
    ```

The JSON layout stores `joint_names`, `joint_positions` as nested lists, and `fps`. Optional `height_m` in the file or metadata drives `scale_to_robot` policies in run configs.

## Supported file types

| Extension | Notes |
|-----------|--------|
| `.json` | Human-readable; good for fixtures and small clips |
| `.npz` / `.npy` | Compact; arrays must match the format’s expected keys |
| `.csv` | Wide format: columns `{JointName}_x`, `{JointName}_y`, `{JointName}_z` |

Loaders convert external frame conventions to internal **Z-up right-handed** coordinates and record conversions in motion metadata. If your mocap is Y-up, read [Coordinate conventions](../coordinate-conventions.md) before writing a custom loader.

## Build motion in code

When prototyping, construct arrays directly (see `examples/basic_robot_only.py`):

```python
import numpy as np
from retarget.motion import MotionSequence, motion_formats

fmt = motion_formats.get("minimal")
names = fmt.joint_names
positions = np.zeros((30, len(names), 3))
positions[:, names.index("Pelvis"), 0] = np.linspace(0.0, 0.5, 30)

motion = MotionSequence(
    name="synthetic_walk",
    joint_names=names,
    joint_positions=positions,
    fps=30,
    metadata={"height_m": 1.7},
)
```

Pass this `motion` object to `RetargetingProblem` instead of loading from disk.

## Resampling frame rate

Three related knobs:

1. **`MotionSequence.resampled(fps)`** — resample source joints (and root poses) before optimization.
2. **`RetargetingProblem.output_fps`** — engine resamples motion (and object trajectories) before solving; saved results use that rate.
3. **`RetargetingResult.resampled(fps)`** — resample `qpos` after the fact for export or comparison.

Use `output_fps` in run configs when every experiment in a study should share a time grid.

## Custom formats

To support a new skeleton naming scheme or file layout:

1. Implement a format class and register it on `motion_formats`.
2. Optionally add a `motion_loaders` entry for discovery by extension.

Walk through `examples/custom_motion_format.py` and [Add a motion format](../adding-a-motion-format.md). Keep joint names stable—objectives and foot constraints refer to mapped link names on the robot, not raw mocap labels.

## Checklist before retargeting

- [ ] Every joint name in the file exists in the chosen format.
- [ ] Positions are in meters (or consistently scaled) and roughly match the target robot’s size metadata.
- [ ] `fps` matches the capture rate or your intended playback rate.
- [ ] Root pose arrays, if present, use the quaternion order declared in the format or loader.

## Next steps

- [Your first retarget](your-first-retarget.md) — run and evaluate end to end
- [Scenes and task kinds](scene-tasks.md) — when the motion interacts with objects or terrain
