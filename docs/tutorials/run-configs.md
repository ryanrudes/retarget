# Run Configs

A run config serializes a `RetargetingExperiment`. Observation and adaptation
are separate typed blocks.

```toml
name = "basic"
robot = "synthetic_humanoid"
output = "basic.npz"

[observation]
kind = "motion_file"
path = "../tests/fixtures/minimal_motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "robot_only"

[recipe.link_roles]
Pelvis = "pelvis"
L_Toe = "left_foot"
R_Toe = "right_foot"
```

`[observation]` is deserialized into an `ObservationRecipe`. `[recipe]` is
deserialized into a `RetargetingRecipe`. Robot-role strings validate through the
robot's declared `RobotRole` enum.

## Optimization Fields

Role-mapping recipes expose scene and optimization fields under `[recipe]`:

```toml
[recipe.mesh]
topology = "delaunay"
k_neighbors = 4

[recipe.solver]
backend = "auto"
max_iterations = 8

[[recipe.objectives]]
kind = "laplacian"
weight = 10.0

[[recipe.constraints]]
kind = "joint_limits"
```

Specialized recipes serialize their own public constructor fields:

```toml
[observation]
kind = "skateboarding"
vicon = "/data/vicon/demo"
gvhmr = "/data/gvhmr/demo"
video_fps = 59.942
timeline_selection = "human_pose"
crop_policy = "overlap"

[recipe]
kind = "skateboarding"
output_fps = 30
```

`timeline_selection = "uniform"` requires a `uniform_fps` value in the same
observation block. These fields construct the same `TimelineSelection` and
`CropPolicy` values used by the Python recipe.

No config section implements a hidden synchronization or preparation workflow.

## Providers And Paths

Robot providers are unchanged:

```toml
robot = "g1"
robot_provider = "asset_store"

[robot_options]
store = "../.retarget_assets"
```

Relative source, asset, output, and import paths resolve against the config
file. CLI overrides can replace motion-file paths, output, run name, robot, or
progress settings.

Validate a config without running optimization:

```bash
uv run python -c "
from retarget.cli.config import RetargetingRunConfig
cfg = RetargetingRunConfig.load('examples/basic/run_config.toml')
print(cfg.build_experiment().build_problem().name)
"
```
