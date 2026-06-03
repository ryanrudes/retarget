# Your first retarget

Shorter smoke test: [Quickstart](../quickstart.md).

This tutorial runs a complete retargeting job with the bundled fixture motion, saves a result, and checks it with the evaluation CLI. Everything uses built-in robots and formats—no external assets required.

## Prerequisites

From the repository root:

```bash
uv sync --extra dev
```

Confirm registered components and optional dependencies:

```bash
uv run retarget doctor
```

You should see registries for motion formats, robots, objectives, and constraints. Missing optional packages (MuJoCo, Viser, CVXPY) are fine for this walkthrough; the default solver falls back to the built-in NumPy/SciPy path when CVXPY is not installed.

## Step 1 — Run from the CLI { #step-1--run-from-the-cli }

The smallest motion fixture lives under `tests/fixtures/`. It uses the `minimal` format (eleven named joints in Z-up meters).

```bash
uv run retarget run \
  --motion tests/fixtures/minimal_motion.json \
  --format minimal \
  --robot synthetic_humanoid \
  --task-kind robot_only \
  --output tutorials_first_result.npz
```

What this command does:

1. Loads the motion through the `minimal` format adapter (joint names and positions → internal `MotionSequence`).
2. Builds a `RetargetingProblem` for `robot_only` with the synthetic humanoid and a ground-only scene.
3. Runs the interaction-mesh retargeting engine frame by frame.
4. Writes `tutorials_first_result.npz` with `qpos`, timing, costs, and provenance metadata.

!!! tip "Naming runs"
    Pass `--name my_experiment` to stamp the result metadata. Batch jobs use names to build stable output paths.

## Step 2 — Evaluate the result

Evaluation computes scalar quality metrics (optimization cost, foot sliding proxy, and others). Supply the same robot/scene context when you want contact-aware metrics:

```bash
uv run retarget evaluate \
  --result tutorials_first_result.npz \
  --output tutorials_first_result.metrics.json
```

Open the JSON file or print it in the terminal by omitting `--output`. A `partial` status means some metric plugins warned or failed while others succeeded—common in large batch runs, rare for this fixture.

## Step 3 — Inspect the NPZ (Python)

Load the saved trajectory and read shapes from Python:

```python
from retarget.results import RetargetingResult

result = RetargetingResult.load_npz("tutorials_first_result.npz")
print(result.frame_count, result.fps, result.qpos.shape)
print(result.metadata.get("provenance", {}).get("task_kind"))
```

Key fields are documented in [Result schema](../result-schema.md). Prefer `metadata_json` in raw NPZ inspection; `RetargetingResult` already parses provenance for you.

## Step 4 — Same job in Python { #step-4--same-job-in-python }

The CLI is a thin wrapper around `Retargeter` and `RetargetingProblem`. Equivalent code:

```python
from retarget import Retargeter, RetargetingProblem, SceneSpec, TaskKind
from retarget.motion import load_motion, motion_formats
from retarget.robots import robots

motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
problem = RetargetingProblem(
    name="tutorial",
    task_kind=TaskKind.ROBOT_ONLY,
    robot=robots.get("synthetic_humanoid"),
    motion=motion,
    motion_format=motion_formats.get("minimal"),
    scene=SceneSpec.robot_only(),
)
result = Retargeter().run(problem)
result.save_npz("tutorials_first_result_api.npz")
```

Compare CLI and API outputs with the same evaluation command, swapping the file path.

## What you learned

- **`retarget run`** needs `--motion`, `--format`, `--robot`, and `--output` unless you use a config file.
- Results are **`RetargetingResult`** NPZ files with `qpos` trajectories and rich provenance.
- **`retarget evaluate`** turns a result into an **`EvaluationReport`** JSON file.

## Next steps

- Put the run spec in version control: [Run configs](run-configs.md)
- [Python API](python-api.md) for registries, config load, export from code
- Prepare your own mocap: [Motion input](motion-input.md)
- Add objects or terrain: [Scenes and task kinds](scene-tasks.md)
