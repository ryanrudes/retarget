# Export And Visualization

Exporters convert `RetargetingResult` objects into downstream experiment formats. The built-in `mujoco_npz` exporter writes MuJoCo-style qpos/qvel tracking arrays while preserving compatibility keys used by simple tracking scripts. Saved tracking NPZ files include `schema_version` and `metadata_json` keys so experiment metadata can be inspected without loading object arrays.

Tracking metadata records `source_fps`, `output_fps`, source/exported frame counts, duration, whether the result was resampled, the qpos/qvel dimensions, and the qvel scheme. Qvel uses `first_frame_forward_difference_then_previous_interval`: frame 0 stores a forward difference to frame 1 when available, and later frames store the velocity from the previous qpos to the current qpos. Backend-aware exports use the same timing convention through `KinematicsBackend.qpos_to_qvel`.

```bash
retarget export --result result.npz --output tracking.npz --format mujoco_npz --output-fps 50
```

For simulator-correct velocities, pass a robot and registered kinematics backend. This matters for MuJoCo models with free or ball joints, where `qvel` has a different dimension than `qpos`.

```bash
retarget export \
  --result result.npz \
  --output tracking.npz \
  --robot g1_like \
  --kinematics-backend simple
```

Python API:

```python
from retarget.export import ExportSpec, export_tracking
from retarget.kinematics import kinematics_backends
from retarget.results import RetargetingResult
from retarget.robots import robots

result = RetargetingResult.load_npz("result.npz")
backend = kinematics_backends.get("simple")(robots.get("g1_like"))
exported = export_tracking(
    result,
    ExportSpec(output_path="tracking.npz", output_fps=50, kinematics_backend=backend),
)
print(exported.path)
```

Register a custom exporter:

```python
from retarget.export import ExportResult, ExportSpec, exporters

@exporters.register("my_export")
class MyExporter:
    def export(self, result, spec: ExportSpec) -> ExportResult:
        spec.output_path.write_text(result.name)
        return ExportResult(
            format_name=spec.format_name,
            path=spec.output_path,
            frame_count=result.frame_count,
            fps=result.fps,
        )
```

Visualization uses the same extension pattern through `visualizers`. `retarget view` defaults to **dry-run** (Rich summary of frames, DOF, warnings, and provenance) and needs no optional dependencies. Pass `--live` for the Viser-backed viewer after `uv sync --extra viz`. Use the **Play** checkbox and **FPS** control in the Viser sidebar to animate the frame slider; scrub **Frame** manually when playback is paused. Optional `--playback-fps` sets the initial FPS (defaults to the result frame rate). Robot playback in live mode requires URDF-backed geometry from result metadata or `--robot-spec <path>` (for example an asset-store `robot.toml`). Missing URDF assets are reported as setup errors instead of being drawn as primitives.

Playback adapters share a typed `PlaybackData` model:

```python
from retarget.results import RetargetingResult
from retarget.visualization import build_playback_data

result = RetargetingResult.load_npz("result.npz")
playback = build_playback_data(result)
frame = playback.frame(0)
print(frame.root_position)
```

`PlaybackData` exposes root poses, timestamps, qpos arrays, and optional source human joint points. This keeps rendering code thin: custom viewers can focus on their UI or graphics library while reusing the same validated playback arrays.
