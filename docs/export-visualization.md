# Export And Visualization

Exporters convert a `RetargetingResult` at an explicit external-format
boundary. The built-in MuJoCo exporter writes a strict NPZ with
`manifest_json`, `time_s`, `qpos`, and `qvel`.

```bash
retarget export \
  --result result.npz \
  --output tracking.npz \
  --format mujoco_npz \
  --output-fps 50
```

For simulator-correct velocities, provide the robot and kinematics backend:

```bash
retarget export \
  --result result.npz \
  --output tracking.npz \
  --robot g1_like \
  --kinematics-backend mujoco
```

Python API:

```python
from retarget import ExportFormat, Robot
from retarget.export import ExportSpec, export_tracking
from retarget.results import RetargetingResult

result = RetargetingResult.load_npz("result.npz")
exported = export_tracking(
    result,
    ExportSpec(
        format_name=ExportFormat.MUJOCO_NPZ,
        output_path="tracking.npz",
        output_fps=50,
    ),
)
```

Custom exporters use a typed registry key:

```python
from retarget import ExporterKind
from retarget.export import ExportResult, exporters


class LabExport(ExporterKind):
    CSV = "lab_csv"


@exporters.register(LabExport.CSV)
class LabCsvExporter:
    def export(self, result, spec):
        write_csv(spec.output_path, result.qpos)
        return ExportResult(
            format_name=LabExport.CSV,
            path=spec.output_path,
            frame_count=result.frame_count,
            fps=result.fps,
        )
```

`retarget view --dry-run` prints the structured result summary without optional
dependencies. `retarget view --live` uses Viser and the typed playback fields
stored in the result. Robot asset paths and object definitions are not read from
provenance.
