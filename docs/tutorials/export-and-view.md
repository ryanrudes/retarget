# Export and view

!!! warning "Prerequisite"
    Complete [Your first retarget](your-first-retarget.md) first so you have `tutorials_first_result.npz` on disk.

After retargeting you usually need (1) a quick sanity check of the trajectory and (2) a file downstream simulators or policies can consume. This tutorial covers both using that result. Full exporter and viewer options: [Export and visualization](../export-visualization.md).

## Summarize a result (no extra deps)

`retarget view` defaults to dry-run mode: a Rich table of frames, DOF, warnings, and provenance—no Viser window.

```bash
uv run retarget view --result tutorials_first_result.npz --dry-run
```

Use this in SSH sessions or CI log artifacts.

## Live visualization (optional)

Install the visualization extra and launch the Viser-backed viewer:

```bash
uv sync --extra viz
uv run retarget view --result tutorials_first_result.npz --live
```

In the Viser sidebar, enable **Play** to animate the trajectory, adjust **FPS** for playback speed, and scrub **Frame** while paused. Pass `--playback-fps 30` to set the initial FPS (defaults to the result frame rate).

If Viser is missing, `--live` fails with an import error; `--dry-run` still works.

Python equivalent:

```python
from retarget.results import RetargetingResult
from retarget.visualization import view_result

view_result(RetargetingResult.load_npz("tutorials_first_result.npz"), dry_run=True)
```

## Export MuJoCo-style tracking NPZ

The `mujoco_npz` exporter writes `qpos`, `qvel`, `time_s`, and a strict JSON manifest:

```bash
uv run retarget export \
  --result tutorials_first_result.npz \
  --output tutorials_tracking.npz \
  --format mujoco_npz
```

Resample to a simulator control rate:

```bash
uv run retarget export \
  --result tutorials_first_result.npz \
  --output tutorials_tracking_50hz.npz \
  --format mujoco_npz \
  --output-fps 50
```

### Backend-aware velocities

For robots where `qvel` dimension differs from `qpos` (free joints, ball joints), pass a kinematics backend so velocities match the simulator:

```bash
uv run retarget export \
  --result tutorials_first_result.npz \
  --output tracking.npz \
  --robot synthetic_humanoid \
  --kinematics-backend simple
```

Qvel timing uses forward difference on frame 0, then previous-interval differences—documented in [Export and visualization](../export-visualization.md).

## Python export API

```python
from retarget import ExportFormat
from retarget.export import ExportSpec, export_tracking
from retarget.results import RetargetingResult

result = RetargetingResult.load_npz("tutorials_first_result.npz")
exported = export_tracking(
    result,
    ExportSpec(
        format_name=ExportFormat.MUJOCO_NPZ,
        output_path="tutorials_tracking.npz",
        output_fps=50,
    ),
)
print(exported.frame_count, exported.fps, exported.path)
```

Register custom exporters on the `exporters` registry when your lab needs CSV, HDF5, or environment-specific bundles.

## Inspect the strict exported NPZ

Tracking files expose one JSON manifest and numeric arrays:

```python
import json
import numpy as np

with np.load("tutorials_tracking.npz", allow_pickle=False) as data:
    print(data["qpos"].shape, data["qvel"].shape)
    manifest = json.loads(str(data["manifest_json"]))
    print(manifest["report"]["qvel_scheme"])
```

## End-to-end checklist

| Step | Command / API |
|------|----------------|
| Retarget | `retarget run …` or `Retargeter().run` |
| Metrics | `retarget evaluate --result …` |
| Quick look | `retarget view --dry-run` |
| Simulator handoff | `retarget export --format mujoco_npz` |

## Next steps

- Custom exporters and visualizers: [Export and visualization](../export-visualization.md)
- Asset-backed robots for export: [Assets](../assets.md)
