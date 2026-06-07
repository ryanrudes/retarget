import json

import numpy as np
import pytest

from retarget.core.enums import ExporterKind, ExportFormat
from retarget.export import (
    QVEL_SCHEME,
    ExportResult,
    ExportSpec,
    MuJoCoTrackingData,
    build_mujoco_tracking_data,
    export_tracking,
    exporters,
)
from retarget.kinematics import SimpleKinematicsBackend
from retarget.pipeline import Retargeter
from retarget.pipeline.compiled import compile_problem
from tests.typed_fixtures import fixture_problem


class FixtureExportKind(ExporterKind):
    JSON = "fixture_json"


def test_mujoco_export_writes_strict_manifest_and_numeric_arrays(tmp_path):
    result = Retargeter().run(fixture_problem())
    path = tmp_path / "tracking.npz"

    exported = export_tracking(
        result,
        ExportSpec(
            format_name=ExportFormat.MUJOCO_NPZ,
            output_path=path,
            output_fps=60,
        ),
    )

    assert exported.format_name is ExportFormat.MUJOCO_NPZ
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"manifest_json", "time_s", "qpos", "qvel"}
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["schema_version"] == 2
        assert manifest["report"]["qvel_scheme"] == QVEL_SCHEME
        assert manifest["report"]["resampled"] is True
        assert "metadata" not in manifest


def test_tracking_data_round_trip_model_is_structured():
    result = Retargeter().run(fixture_problem())
    tracking = build_mujoco_tracking_data(result)

    assert isinstance(tracking, MuJoCoTrackingData)
    assert tracking.report.qvel_source == "finite_difference"
    assert tracking.report.qpos_dimension == result.qpos.shape[1]
    assert tracking.qvel.shape == tracking.qpos.shape


def test_backend_qvel_is_used_at_external_export_boundary():
    problem = fixture_problem()
    result = Retargeter().run(problem)
    backend = SimpleKinematicsBackend(compile_problem(problem).robot)

    tracking = build_mujoco_tracking_data(
        result,
        kinematics_backend=backend,
    )

    assert tracking.report.qvel_source == "SimpleKinematicsBackend"
    assert tracking.qvel.shape == result.qpos.shape


def test_export_registry_requires_typed_key(tmp_path):
    class JsonExporter:
        def export(self, result, spec):
            spec.output_path.write_text(json.dumps({"name": result.name}))
            return ExportResult(
                format_name=FixtureExportKind.JSON,
                path=spec.output_path,
                frame_count=result.frame_count,
                fps=result.fps,
            )

    exporters.register(FixtureExportKind.JSON, JsonExporter(), replace=True)
    result = Retargeter().run(fixture_problem())
    path = tmp_path / "result.json"

    exported = export_tracking(
        result,
        ExportSpec(format_name=FixtureExportKind.JSON, output_path=path),
    )

    assert exported.path == path
    with pytest.raises(TypeError):
        exporters.get("fixture_json")  # type: ignore[arg-type]
