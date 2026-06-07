# retarget

`retarget` is a typed research toolkit for capture processing and robot motion
retargeting. Native sensor streams keep independent clocks and vocabularies until
an `ObservationRecipe` fuses them into a target-independent `SceneObservation`.
A `RetargetingRecipe` then resolves semantic roles through a `RobotSpec` and builds
the optimization problem.

```text
native recordings
-> ObservationRecipe
-> SceneObservation
-> RetargetingRecipe + RobotSpec
-> RetargetingProblem
-> RetargetingResult
```

## Quickstart

```bash
uv sync --extra dev
uv run retarget doctor
uv run retarget run --config examples/basic/run_config.toml
uv run pytest
uv run mypy src/retarget
uv run mkdocs build --strict
```

Python experiments use the same path as run configs:

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
recipe = RoleRetargetingRecipe(
    task_kind=TaskKind.ROBOT_ONLY,
    motion_format=motion_formats.get("minimal"),
    scene=RobotOnlySceneRecipe(),
    link_roles={
        MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
        MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
        MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
    },
)
result = RetargetingExperiment(
    observation=observation,
    recipe=recipe,
    robot=robots.get("synthetic_humanoid"),
).run()
```

Skateboarding composes native Vicon and GVHMR sources directly; Holosoma climbing
uses the same experiment hierarchy as a one-source subset. See `examples/skateboarding/`
and `examples/holosoma/`.

## Optional Dependencies

Core capture models and retargeting remain lightweight. Install only the backends
needed by an experiment:

```bash
uv sync --extra optimize
uv sync --extra mujoco
uv sync --extra viz
uv sync --extra torch --extra smpl
```

## Documentation

The documentation is published at
<https://ryanrudes.github.io/retarget/>. Preview locally with:

```bash
uv sync --extra dev
uv run mkdocs serve
```
