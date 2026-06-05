# Scenes and task kinds

`TaskKind` selects which scene geometry the retargeter optimizes against. All three kinds share the same interaction-mesh core; they differ in what points enter the mesh and which constraint profiles are typical.

| Task kind | Scene | Typical use |
|-----------|-------|-------------|
| `robot_only` | Ground plane samples | Locomotion, gestures, no manipulated props |
| `object_interaction` | Ground + moving object samples | Picking, carrying, table contact |
| `climbing` | Ground + terrain/hold samples | Climbing, stepping on structured terrain |

## Robot-only (default)

The fastest path—ground contact and foot constraints without extra geometry:

=== "CLI"

    ```bash
    uv run retarget run --motion tests/fixtures/minimal_motion.json --format minimal \
      --robot synthetic_humanoid --task-kind robot_only --output robot_only.npz
    ```

    Full CLI walkthrough and flags: [Your first retarget — Step 1](your-first-retarget.md#step-1--run-from-the-cli).

=== "Python"

    ```python
    from retarget import SceneSpec, TaskKind

    scene = SceneSpec.robot_only(ground_size=15, ground_range=(-1.0, 1.0))
    # RetargetingProblem(..., task_kind=TaskKind.ROBOT_ONLY, scene=scene)
    ```

    Full problem setup: [Your first retarget — Step 4](your-first-retarget.md#step-4--same-job-in-python). Tune the ground grid with `[scene]` in a run config (`ground_size`, `ground_range`).

## Object interaction

Objects contribute sample points that move with a trajectory. The mesh links human joints, robot-mapped points, object samples, and ground.

Minimal Python setup (static box, identity trajectory)—full script: `examples/object_interaction.py`:

```python
import numpy as np
from retarget import (
    ObjectSpec,
    ObjectTrajectory,
    OptimizationProfile,
    Retargeter,
    RetargetingProblem,
    SceneSpec,
    TaskKind,
)
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

fmt = motion_formats.get("minimal")
motion = MotionSequence(
    name="reach",
    joint_names=fmt.joint_names,
    joint_positions=np.zeros((12, len(fmt.joint_names), 3)),
    fps=30,
    metadata={"height_m": 1.7},
)
box_points = np.array(
    [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)],
    dtype=float,
)
obj = ObjectSpec(
    name="box",
    sample_points=box_points,
    trajectory=ObjectTrajectory.identity(motion.frame_count, fps=motion.fps),
)
profile = OptimizationProfile.object_interaction(
    floor_z=-2.0,
    scene_clearance=0.03,
    links=("left_toe", "right_toe"),
)
problem = RetargetingProblem(
    name="object_interaction",
    task_kind=TaskKind.OBJECT_INTERACTION,
    robot=robots.get("synthetic_humanoid"),
    motion=motion,
    motion_format=fmt,
    scene=SceneSpec.object_interaction(obj),
    constraints=profile.constraints,
)
Retargeter().run(problem).save_npz("object_interaction.npz")
```

For mesh files instead of explicit points, set `mesh_path` in `[scene.object]` and let the CLI sample `mesh_sample_count` points. Trajectories can come from NPZ paths or per-frame positions in the config.

## Climbing / terrain

Terrain supplies hold or surface samples; constraints emphasize clearance and contact on links you name in `OptimizationProfile.climbing`:

```python
import numpy as np
from retarget import (
    OptimizationProfile,
    Retargeter,
    RetargetingProblem,
    SceneSpec,
    TaskKind,
    TerrainSpec,
)
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

fmt = motion_formats.get("minimal")
motion = MotionSequence(
    name="climb",
    joint_names=fmt.joint_names,
    joint_positions=np.zeros((10, len(fmt.joint_names), 3)),
    fps=30,
)
holds = np.array(
    [[-0.35, 0.0, -0.45], [0.35, 0.0, -0.45], [-0.20, 0.0, -0.10], [0.20, 0.0, -0.10]],
    dtype=float,
)
terrain = TerrainSpec(name="blocks", sample_points=holds)
profile = OptimizationProfile.climbing(floor_z=-2.0, scene_clearance=0.025)
problem = RetargetingProblem(
    name="climbing",
    task_kind=TaskKind.CLIMBING,
    robot=robots.get("synthetic_humanoid"),
    motion=motion,
    motion_format=fmt,
    scene=SceneSpec.climbing(terrain=terrain),
    constraints=profile.constraints,
)
Retargeter().run(problem).save_npz("climbing.npz")
```

See `examples/climbing_terrain.py` for a runnable copy.

## Choosing objectives and constraints

`OptimizationProfile` bundles sensible defaults per task kind. You can still append typed objective and constraint config objects on the problem, or add `[[objectives]]` / `[[constraints]]` tables with `kind` fields in a run config.

!!! info "Penetration metrics"
    Evaluation’s `penetration` metric needs the same scene geometry you used at retarget time. Pass `--config` to `retarget evaluate` when comparing object or terrain runs.

## Real assets

Synthetic points are enough to learn the API. For lab robots and scanned objects:

1. Build an [asset manifest](../assets.md) with local paths.
2. Reference `mesh_path` or asset-store robot providers in your run config.

## Next steps

- [Run configs](run-configs.md) — express scenes in TOML
- [Export and view](export-and-view.md) — preview results with `retarget view`
