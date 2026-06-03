# retarget

`retarget` is a standalone, typed research toolkit for motion retargeting. It is built as a `uv` project with a composable Python API, an interaction-mesh SQP retargeting core, small fixtures, CLI workflows, and extension points for robots, motion formats, solvers, metrics, exporters, and visualizers.

The implementation is intentionally independent of the `holosoma` reference implementation. The package does not import or vendor it.

## Quickstart

```bash
uv sync --extra dev
uv run retarget doctor
uv run pytest
uv run mkdocs build --strict
```

Python API:

```python
import numpy as np

from retarget import MotionFormat, Retargeter, RetargetingProblem, Robot, SceneSpec, TaskKind
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

joint_names = ("Pelvis", "L_Toe", "R_Toe", "L_Wrist", "R_Wrist")
motion = MotionSequence(
    name="tiny",
    joint_names=joint_names,
    joint_positions=np.zeros((8, len(joint_names), 3)),
    fps=30,
)

problem = RetargetingProblem(
    name="tiny",
    task_kind=TaskKind.ROBOT_ONLY,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
    motion=motion,
    motion_format=motion_formats.get(MotionFormat.MINIMAL),
    scene=SceneSpec.robot_only(),
)
result = Retargeter().run(problem)
result.save_npz("tiny_result.npz")
```

Set `RetargetingProblem.output_fps` or call `result.resampled(fps)` when experiments need a common time grid.

CLI:

```bash
uv run retarget run \
  --motion tests/fixtures/minimal_motion.json \
  --format minimal \
  --robot synthetic_humanoid \
  --output tiny_result.npz
uv run retarget export --result tiny_result.npz --output tracking.npz --format mujoco_npz
```

Config-file CLI:

```bash
uv run retarget run --config examples/run_config.toml
```

Asset manifest workflow:

```bash
uv run retarget assets validate examples/assets_manifest.toml
uv run retarget assets install examples/assets_manifest.toml --store .retarget_assets
```

Robot specs can be loaded from Python registries or external TOML/YAML/JSON files, for example `examples/custom_robot.toml`.

## Documentation

**https://ryanrudes.github.io/retarget/** — built from `docs/` with MkDocs Material (GitHub Pages on `master`).

Local preview (includes live-code Jupyter; not on the hosted site):

```bash
git submodule update --init
uv sync --extra dev
uv run mkdocs serve
```
