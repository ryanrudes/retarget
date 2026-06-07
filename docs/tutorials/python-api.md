# Python API

The normal public entry point is `RetargetingExperiment`.

```python
from retarget import HumanoidRobotRole, RetargetingExperiment, TaskKind
from retarget.motion import motion_formats
from retarget.motion.registry import MinimalMotionJoint
from retarget.recipes import (
    MotionFileObservationRecipe,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
)
from retarget.robots import robots

observation = MotionFileObservationRecipe.registered(
    "tests/fixtures/minimal_motion.json",
    "minimal",
)
adaptation = RoleRetargetingRecipe(
    task_kind=TaskKind.ROBOT_ONLY,
    motion_format=motion_formats.get("minimal"),
    scene=RobotOnlySceneRecipe(),
    link_roles={
        MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
        MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
        MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
    },
)
experiment = RetargetingExperiment(
    observation=observation,
    recipe=adaptation,
    robot=robots.get("synthetic_humanoid"),
)
result = experiment.run()
```

## Inspect Between Stages

```python
scene_observation = experiment.observe()
problem = experiment.build_problem()
```

Reuse `scene_observation` with another recipe or robot without repeating capture
processing. Call `save_npz()` only when an explicit checkpoint is useful.

## Native Capture Sources

Built-in sources include:

- `ViconRecordingSource`
- `ViconBagSource`
- `GvhmrOutputSource`
- `MocapArraySource`
- `VideoPoseSource`
- `InMemorySource`

Implement `ObservationSource[T]` for another native format. Implement
`HumanPoseEstimator` to derive a `HumanPoseRecording` from video.

## Run Configs In Python

```python
from retarget.cli.config import RetargetingRunConfig

config = RetargetingRunConfig.load("examples/basic/run_config.toml")
experiment = config.build_experiment()
result = experiment.run()
result.save_npz(config.output)
```

This is the same path used by `retarget run --config`.

## Extension Points

Registries remain available for motion formats, loaders, robot providers,
objectives, constraints, solvers, metrics, exporters, and visualizers. Capture
sources and recipes are ordinary typed protocols; they do not require global
registration for programmatic use.
