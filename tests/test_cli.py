import json
import shutil
import subprocess
import sys

import numpy as np


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "retarget.cli.main", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def run_cli_unchecked(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "retarget.cli.main", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_cli_run_evaluate_export_view(tmp_path):
    result = tmp_path / "result.npz"
    report = tmp_path / "report.json"
    export = tmp_path / "tracking.npz"
    run_cli(
        "run",
        "--motion",
        "tests/fixtures/minimal_motion.json",
        "--format",
        "minimal",
        "--robot",
        "synthetic_humanoid",
        "--output",
        str(result),
    )
    assert result.exists()
    run_cli("evaluate", "--result", str(result), "--output", str(report))
    report_data = json.loads(report.read_text())
    assert report_data["source_name"] == "minimal_motion"
    assert report_data["frame_count"] > 0
    assert report_data["metric_units"]["optimization_cost"] == "cost"
    assert report_data["metrics"]["optimization_cost"] >= 0.0
    run_cli(
        "export",
        "--result",
        str(result),
        "--output",
        str(export),
        "--output-fps",
        "50",
        "--robot",
        "synthetic_humanoid",
        "--kinematics-backend",
        "simple",
    )
    assert export.exists()
    export_data = np.load(export, allow_pickle=True)
    assert export_data["metadata"].item()["qvel_source"] == "SimpleKinematicsBackend"
    run_cli("view", "--result", str(result), "--dry-run")


def test_cli_run_from_toml_config(tmp_path):
    motion = tmp_path / "motion.json"
    shutil.copyfile("tests/fixtures/minimal_motion.json", motion)
    config = tmp_path / "run.toml"
    report = tmp_path / "configured.metrics.json"
    config.write_text(
        """
name = "configured"
robot = "synthetic_humanoid"
output = "configured.npz"

[observation]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "robot_only"
output_fps = 60.0

[recipe.mesh]
topology = "k_nearest"
k_neighbors = 2

[recipe.solver]
max_iterations = 4
trust_radius = 0.2

[[recipe.objectives]]
kind = "laplacian"
weight = 8.0

[[recipe.objectives]]
kind = "smoothness"
weight = 0.1
""".strip()
    )
    result = tmp_path / "configured.npz"
    run_cli("run", "--config", str(config))
    assert result.exists()
    result_data = np.load(result, allow_pickle=False)
    result_metadata = json.loads(result_data["metadata_json"].reshape(()).item())
    assert result_metadata["mesh"] == {
        "topology": "k_nearest",
        "k_neighbors": 2,
        "laplacian_weighting": "uniform",
        "laplacian_epsilon": 1e-06,
        "source": "problem",
    }
    run_cli("evaluate", "--result", str(result), "--config", str(config), "--output", str(report))
    report_data = json.loads(report.read_text())
    assert report_data["status"] == "success"
    assert report_data["frame_count"] == 5
    assert report_data["details"]["problem"]["aligned_to_result"] is True
    assert "optimization_cost" in report_data["metrics"]


def test_cli_run_imports_local_extension_from_config(tmp_path):
    motion = tmp_path / "motion.json"
    shutil.copyfile("tests/fixtures/minimal_motion.json", motion)
    (tmp_path / "custom_terms.py").write_text(
        """
from dataclasses import dataclass

import numpy as np

from typing import Literal

from retarget.optimization import ObjectiveConfig, ObjectiveContribution, TermContext, objective_terms


class CliZeroEnergyConfig(ObjectiveConfig):
    kind: Literal["cli_zero_energy"] = "cli_zero_energy"


@objective_terms.register("cli_zero_energy", replace=True)
@dataclass(frozen=True)
class CliZeroEnergy:
    name: str = "cli_zero_energy"
    config_type: type[CliZeroEnergyConfig] = CliZeroEnergyConfig

    def describe(self) -> str:
        return "A CLI-loaded objective."

    def build(self, context: TermContext, _config: CliZeroEnergyConfig) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )
""".strip()
    )
    config = tmp_path / "run.toml"
    config.write_text(
        """
name = "imported_objective"
imports = ["custom_terms.py"]
robot = "synthetic_humanoid"
output = "imported.npz"

[observation]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "robot_only"

[recipe.solver]
max_iterations = 1

[[recipe.objectives]]
kind = "cli_zero_energy"
weight = 0.01
""".strip()
    )
    result = tmp_path / "imported.npz"

    run_cli("run", "--config", str(config))

    assert result.exists()
    result_data = np.load(result, allow_pickle=False)
    result_metadata = json.loads(result_data["metadata_json"].reshape(()).item())
    assert [objective["kind"] for objective in result_metadata["provenance"]["objectives"]] == [
        "cli_zero_energy"
    ]


def test_cli_run_reports_registry_preflight_errors_without_traceback(tmp_path):
    motion = tmp_path / "motion.json"
    shutil.copyfile("tests/fixtures/minimal_motion.json", motion)
    config = tmp_path / "run.toml"
    config.write_text(
        """
name = "bad_extension_config"
robot = "synthetic_humanoid"
output = "bad.npz"

[observation]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "robot_only"

[[recipe.objectives]]
kind = "missing_cli_objective"
""".strip()
    )

    completed = run_cli_unchecked("run", "--config", str(config))

    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "missing_cli_objective" in combined_output
    assert "laplacian" in combined_output
    assert "Traceback" not in combined_output


def test_cli_object_and_climbing_modes(tmp_path):
    for task_kind in ("object_interaction", "climbing"):
        result = tmp_path / f"{task_kind}.npz"
        run_cli(
            "run",
            "--motion",
            "tests/fixtures/minimal_motion.json",
            "--format",
            "minimal",
            "--robot",
            "synthetic_humanoid",
            "--task-kind",
            task_kind,
            "--output",
            str(result),
        )
        assert result.exists()


def test_cli_batch(tmp_path):
    output_dir = tmp_path / "batch"
    run_cli("batch", "--input-dir", "tests/fixtures", "--pattern", "*.json", "--output-dir", str(output_dir))
    assert (output_dir / "minimal_motion.npz").exists()
    manifest_path = output_dir / "batch_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["success_count"] == 1
    assert manifest["skipped_count"] == 0

    run_cli("batch", "--input-dir", "tests/fixtures", "--pattern", "*.json", "--output-dir", str(output_dir))
    resumed = json.loads(manifest_path.read_text())
    assert resumed["success_count"] == 0
    assert resumed["skipped_count"] == 1


def test_cli_evaluate_batch_manifest(tmp_path):
    output_dir = tmp_path / "batch"
    metrics_dir = tmp_path / "metrics"
    summary = tmp_path / "evaluation_manifest.json"
    run_cli("batch", "--input-dir", "tests/fixtures", "--pattern", "*.json", "--output-dir", str(output_dir))

    run_cli(
        "evaluate",
        "--batch-manifest",
        str(output_dir / "batch_manifest.json"),
        "--output-dir",
        str(metrics_dir),
        "--output",
        str(summary),
    )

    assert (metrics_dir / "minimal_motion.metrics.json").exists()
    summary_data = json.loads(summary.read_text())
    assert summary_data["total"] == 1
    assert summary_data["success_count"] == 1
    assert summary_data["records"][0]["metrics"]["optimization_cost"] >= 0.0


def test_cli_batch_preserves_recursive_relative_output_paths(tmp_path):
    input_dir = tmp_path / "motions"
    subject_a = input_dir / "subject_a"
    subject_b = input_dir / "subject_b"
    subject_a.mkdir(parents=True)
    subject_b.mkdir(parents=True)
    shutil.copyfile("tests/fixtures/minimal_motion.json", subject_a / "walk.json")
    shutil.copyfile("tests/fixtures/minimal_motion.json", subject_b / "walk.json")
    output_dir = tmp_path / "batch"

    run_cli("batch", "--input-dir", str(input_dir), "--pattern", "**/*.json", "--output-dir", str(output_dir))

    assert (output_dir / "subject_a" / "walk.npz").exists()
    assert (output_dir / "subject_b" / "walk.npz").exists()
    manifest = json.loads((output_dir / "batch_manifest.json").read_text())
    assert [record["job_id"] for record in manifest["records"]] == [
        "subject_a/walk.json",
        "subject_b/walk.json",
    ]
    assert [record["output"] for record in manifest["records"]] == [
        str(output_dir / "subject_a" / "walk.npz"),
        str(output_dir / "subject_b" / "walk.npz"),
    ]


def test_cli_doctor():
    completed = run_cli("doctor")
    assert "synthetic_humanoid" in completed.stdout
    assert "objective terms" in completed.stdout
    assert "laplacian" in completed.stdout
    assert "exporters" in completed.stdout
    assert "mujoco_npz" in completed.stdout


def test_cli_assets_install_and_validate(tmp_path):
    asset = tmp_path / "asset.txt"
    asset.write_text("asset")
    manifest = tmp_path / "assets.toml"
    manifest.write_text(
        """
[[assets]]
name = "fixture_asset"
kind = "fixture"
source = "asset.txt"
copy = true
destination = "fixture_asset.txt"
license = "Apache-2.0"
""".strip()
    )
    store = tmp_path / "store"

    validate = run_cli("assets", "validate", str(manifest))
    assert "fixture_asset" in validate.stdout
    run_cli("assets", "install", str(manifest), "--store", str(store))

    saved = json.loads((store / "manifest.json").read_text())
    assert saved["records"][0]["name"] == "fixture_asset"
    assert (store / "fixture_asset.txt").read_text() == "asset"


def test_cli_help_lists_workflows():
    completed = run_cli("--help")
    assert "run" in completed.stdout
    assert "assets" in completed.stdout
