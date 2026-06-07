# Add A Motion Format

A motion format describes one ordered joint vocabulary and its representation.
Contact semantics do not belong in a motion format; observation recipes produce
`SemanticContactSequence` separately.

Define the vocabulary by subclassing `MotionJoint`:

```python
from retarget import FrameConvention, MotionJoint
from retarget.motion import MotionFormatSpec, motion_formats


class LabJoint(MotionJoint):
    ROOT = "root"
    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"


@motion_formats.register("lab")
def lab_format() -> MotionFormatSpec:
    return MotionFormatSpec(
        name="lab",
        joint_vocabulary=LabJoint,
        root_joint=LabJoint.ROOT,
        frame_convention=FrameConvention.Y_UP_RIGHT_HANDED,
        default_height_m=1.75,
    )
```

`joint_names` is derived from enum declaration order. The root must be a member
of that exact vocabulary.

## Add A Loader

Loaders parse storage. They do not align recordings, infer contacts, or resolve
robot links.

```python
from pathlib import Path

from retarget.motion import MotionFormatSpec, MotionSequence, motion_loaders


@motion_loaders.register(".labmotion")
class LabMotionLoader:
    def load(
        self,
        path: Path,
        spec: MotionFormatSpec,
        *,
        name: str | None = None,
    ) -> MotionSequence:
        data = parse_lab_file(path)
        return MotionSequence(
            name=name or path.stem,
            joint_names=spec.joint_names,
            joint_positions=data.global_joint_positions,
            fps=data.fps,
            frame=spec.frame_convention,
        )
```

`load_motion` converts the loader output to the internal Z-up frame. Load root
translation and orientation into `MotionSequence.root_poses` when available.

Built-in loaders support `.json`, `.npy`, `.npz`, and `.csv`. The array formats
accept global joint positions and optional root-pose arrays. CSV columns use
`{joint}_x`, `{joint}_y`, and `{joint}_z`; optional `time_s` values determine
the sample rate.

For heterogeneous capture, prefer native `HumanPoseRecording` or
`MocapRecording` sources and an `ObservationRecipe`. `MotionSequence` is the
robot-adaptation representation produced after an observation already has one
timeline and world frame.
