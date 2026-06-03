# Run configs

Config files capture everything needed to reproduce a retargeting experiment: motion path, robot, scene, mesh topology, solver settings, objectives, and constraints. The CLI loads TOML, YAML, or JSON and resolves relative paths against the config file’s directory.

## Start from the example

The repository ships `examples/run_config.toml`:

```bash
cd examples
uv run retarget run --config run_config.toml
uv run retarget evaluate --result configured_fixture.npz --config run_config.toml
```

The evaluate step reloads the same scene and metric context as the run, which matters when you set `output_fps` or scene geometry in the config.

## Anatomy of a run config

A minimal config names inputs and outputs:

```toml
name = "my_run"
motion = "../tests/fixtures/minimal_motion.json"
format = "minimal"
robot = "synthetic_humanoid"
task_kind = "robot_only"
output = "my_run.npz"
```

Add tuning sections as experiments grow:

| Key / section | Purpose |
|---------------|---------|
| `show_progress` | Rich per-frame progress bar during optimization (also `retarget run --progress`) |
| `[mesh]` | Interaction mesh topology (`delaunay`, `k_neighbors`, …) |
| `[solver]` | Backend (`auto`, `numpy_least_squares`, `cvxpy_clarabel`), iterations, trust region |
| `[scene]` | Ground grid for robot-only tasks; nested `[scene.object]` / `[scene.terrain]` for other task kinds |
| `[[objectives]]` | Registered objective terms and weights |
| `[[constraints]]` | Registered constraints; optional `parameters = { ... }` |

The full example in `examples/run_config.toml` sets Laplacian and smoothness objectives, joint limits, trust region, and foot-contact constraints:

```toml
scale_to_robot = true
output_fps = 30

[mesh]
topology = "delaunay"
k_neighbors = 4

[solver]
backend = "auto"
max_iterations = 8
trust_radius = 0.2

[[objectives]]
name = "laplacian"
weight = 10.0

[[constraints]]
name = "foot_contact"
parameters = { velocity_threshold = 0.02, tolerance = 0.001 }
```

See [Interaction mesh](../interaction-mesh.md) for how mesh settings affect the optimization geometry.

## Override flags on the CLI

Keep a base config in git and override paths per machine or sweep:

```bash
uv run retarget run \
  --config examples/run_config.toml \
  --motion /data/subject01/walk.json \
  --output /tmp/subject01_walk.npz \
  --name subject01_walk
```

CLI flags win over file values. This pattern is useful in Slurm or Make targets that only change `motion` and `output`.

## Object and terrain scenes in config

For `task_kind = "object_interaction"` or `"climbing"`, add scene blocks. Object meshes can be referenced by path; the CLI samples points when `sample_points` are omitted:

```toml
task_kind = "object_interaction"

[scene.object]
name = "box"
mesh_path = "/path/to/box.obj"
mesh_sample_count = 128
identity_trajectory = true
```

Terrain climbing configs use `[scene.terrain]` similarly. Programmatic equivalents are in [Scenes and task kinds](scene-tasks.md) and `examples/climbing_terrain.py`.

## Custom robots from config

Point `robot` at a registry name **or** provide a spec file:

```toml
robot = { path = "custom_robot.toml" }
```

`examples/custom_robot.toml` shows the layout. External specs work the same in Python via `RobotSpec.load`.

## Validate before a long batch

Check that paths, format names, and constraint references resolve:

```bash
uv run retarget doctor
# Dry-run a single config build in Python:
uv run python -c "
from retarget.cli.config import RetargetingRunConfig
cfg = RetargetingRunConfig.load('examples/run_config.toml')
print(cfg.build_problem().name)
"
```

## Next steps

- Wire configs into directory sweeps: [Batch and metrics](batch-and-metrics.md)
- Register local URDF/MJCF assets: [Assets](../assets.md)
