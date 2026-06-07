# Add A Motion Format

A motion format owns one ordered `MotionJoint` vocabulary.

```python
from retarget import FrameConvention, MotionFormatKind, MotionJoint
from retarget.motion import MotionFormatSpec, motion_formats


class LabFormat(MotionFormatKind):
    CAPTURE = "lab_capture"


class LabJoint(MotionJoint):
    ROOT = "root"
    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"


motion_formats.register(
    LabFormat.CAPTURE,
    MotionFormatSpec(
        name="lab_capture",
        joint_vocabulary=LabJoint,
        root_joint=LabJoint.ROOT,
        frame_convention=FrameConvention.Y_UP_RIGHT_HANDED,
        default_height_m=1.75,
    ),
)
```

The joint order is `tuple(LabJoint)`. Contact semantics remain separate and are
produced by an observation recipe.

## Add A Loader

Loaders parse one external storage layout. They do not synchronize recordings,
infer contacts, or resolve robot links.

```python
from pathlib import Path

from retarget import MotionLoaderKind, SampleTimeline
from retarget.motion import MotionFormatSpec, MotionSequence, motion_loaders


class LabLoaderKind(MotionLoaderKind):
    LAB = ".labmotion"


@motion_loaders.register(LabLoaderKind.LAB)
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
            joint_vocabulary=spec.joint_vocabulary,
            joints=spec.joints,
            root_joint=spec.root_joint,
            joint_positions=data.global_joint_positions,
            timeline=SampleTimeline.uniform(data.frame_count, data.fps),
            frame=spec.frame_convention,
        )
```

For heterogeneous sensors, load native `HumanPoseRecording` or
`MocapRecording` objects and fuse them in an `ObservationRecipe`.
