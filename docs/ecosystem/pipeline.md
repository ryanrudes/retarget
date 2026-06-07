# End-To-End Pipeline

The skateboarding workflow starts from native Vicon and human-pose recordings.

```python
from retarget import (
    GvhmrOutputSource,
    RetargetingExperiment,
    ViconRecordingSource,
)
from retarget.recipes.skateboarding import (
    GVHMR_SCHEMA,
    VICON_SCHEMA,
    SkateboardingObservationRecipe,
    SkateboardingRetargetingRecipe,
)

observation_recipe = SkateboardingObservationRecipe(
    mocap=ViconRecordingSource(vicon_path, VICON_SCHEMA, name=demo),
    human_pose=GvhmrOutputSource(
        gvhmr_path,
        GVHMR_SCHEMA,
        fps=video_fps,
        name=demo,
    ),
)
experiment = RetargetingExperiment(
    observation=observation_recipe,
    recipe=SkateboardingRetargetingRecipe(),
    robot=robot,
)
result = experiment.run()
```

`SkateboardingObservationRecipe.observe()` performs:

1. source loading on independent native timelines;
2. temporal registration from paired foot-speed cues;
3. timeline overlap selection and type-specific resampling;
4. frame conversion;
5. rigid spatial registration from corresponding foot landmarks;
6. board reconstruction and semantic foot-support classification.

The result is a `SceneObservation`. It can be inspected before adaptation:

```python
observation = experiment.observe()
print(observation.alignment_reports)
print(observation.contacts)
```

`SkateboardingRetargetingRecipe` then resolves semantic landmarks and contacts
through the robot's `HumanoidRobotRole` bindings. The same recipe object is used
by the Python example and the declarative run config.

Raw video can replace an existing pose output:

```python
from retarget import VideoPoseSource

human_pose = VideoPoseSource(video=recording, estimator=estimator)
```

`GvhmrEstimator` is one optional backend. It runs a configured local checkout in
a temporary workspace and returns `HumanPoseRecording`; it does not require a
persistent estimator output directory.
