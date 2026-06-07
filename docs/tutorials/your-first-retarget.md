# Your First Retarget

This walkthrough uses the bundled fixture and the same experiment architecture
as multimodal capture.

## Run The Declarative Experiment

```bash
uv sync --extra dev
uv run retarget run --config examples/basic/run_config.toml
```

The config creates:

1. `MotionFileObservationRecipe`
2. `RoleRetargetingRecipe`
3. the selected `RobotSpec`
4. `RetargetingExperiment`

The CLI does not build a parallel workflow.

## Inspect The Result

```python
from retarget import RetargetingResult

result = RetargetingResult.load_npz(
    "examples/basic/generated/basic_retarget.npz"
)
print(result.frame_count, result.fps, result.qpos.shape)
```

Evaluate it with:

```bash
uv run retarget evaluate \
  --result examples/basic/generated/basic_retarget.npz
```

## Run The Same Experiment In Python

```python
from retarget import (
    HumanoidRobotRole,
    MinimalMotionJoint,
    MotionFileObservationRecipe,
    MotionFormat,
    RetargetingExperiment,
    Robot,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
    TaskKind,
    motion_formats,
    robots,
)

motion_format = motion_formats.get(MotionFormat.MINIMAL)
experiment = RetargetingExperiment(
    observation=MotionFileObservationRecipe(
        path="tests/fixtures/minimal_motion.json",
        motion_format=motion_format,
    ),
    recipe=RoleRetargetingRecipe(
        task_kind=TaskKind.ROBOT_ONLY,
        motion_format=motion_format,
        scene=RobotOnlySceneRecipe(),
        link_roles={
            MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
            MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
            MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
        },
    ),
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
)

observation = experiment.observe()
problem = experiment.build_problem()
result = experiment.run()
```

The observation can be inspected, saved explicitly, or reused with another
robot before optimization.

## Next Steps

- [Python API](python-api.md)
- [Run configs](run-configs.md)
- [Motion and capture input](motion-input.md)
- [Scenes and task kinds](scene-tasks.md)
