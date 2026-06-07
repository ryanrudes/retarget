import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "retarget.cli.main", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def test_cli_run_evaluate_export_and_view_use_strict_manifests(tmp_path):
    result_path = tmp_path / "result.npz"
    report_path = tmp_path / "report.json"
    export_path = tmp_path / "tracking.npz"

    run_cli(
        "run",
        "--motion",
        "tests/fixtures/minimal_motion.json",
        "--format",
        "minimal",
        "--robot",
        "synthetic_humanoid",
        "--output",
        str(result_path),
    )
    run_cli("evaluate", "--result", str(result_path), "--output", str(report_path))
    run_cli(
        "export",
        "--result",
        str(result_path),
        "--output",
        str(export_path),
        "--robot",
        "synthetic_humanoid",
        "--kinematics-backend",
        "simple",
    )
    view = run_cli("view", "--result", str(result_path), "--dry-run")

    with np.load(result_path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["schema_version"] == 2
        assert "metadata_json" not in data
    with np.load(export_path, allow_pickle=False) as data:
        export_manifest = json.loads(str(data["manifest_json"]))
        assert export_manifest["report"]["qvel_source"] == "SimpleKinematicsBackend"
    report = json.loads(report_path.read_text())
    assert report["source_name"] == "minimal_motion"
    assert "frames" in view.stdout.lower()


def test_cli_run_config_builds_same_typed_experiment(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text(
        f"""
name = "configured"
robot = "synthetic_humanoid"
output = "{(tmp_path / "configured.npz").as_posix()}"

[observation]
kind = "motion_file"
path = "{Path("tests/fixtures/minimal_motion.json").resolve().as_posix()}"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "robot_only"

[recipe.link_roles]
Pelvis = "pelvis"
L_Toe = "left_foot"
R_Toe = "right_foot"
""".strip()
    )

    run_cli("run", "--config", str(config))

    with np.load(tmp_path / "configured.npz", allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["run"]["robot_name"] == "synthetic_humanoid"
        assert manifest["human_vocabulary"]["qualname"] == "MinimalMotionJoint"


def test_cli_batch_and_evaluate_manifests(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "motion.json").write_text(
        Path("tests/fixtures/minimal_motion.json").read_text()
    )
    output_dir = tmp_path / "output"

    run_cli(
        "batch",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--pattern",
        "*.json",
    )
    run_cli(
        "evaluate",
        "--batch-manifest",
        str(output_dir / "batch_manifest.json"),
        "--output",
        str(output_dir / "evaluation.json"),
    )

    batch = json.loads((output_dir / "batch_manifest.json").read_text())
    evaluation = json.loads((output_dir / "evaluation.json").read_text())
    assert batch["success_count"] == 1
    assert evaluation["success_count"] == 1


def test_cli_rejects_unknown_serialized_registry_key(tmp_path):
    completed = run_cli(
        "run",
        "--motion",
        "tests/fixtures/minimal_motion.json",
        "--format",
        "not_registered",
        "--output",
        str(tmp_path / "result.npz"),
        check=False,
    )

    assert completed.returncode != 0
    assert "not_registered" in completed.stderr


@pytest.mark.parametrize("command", ("doctor", "assets list"))
def test_cli_read_only_commands(command):
    completed = run_cli(*command.split())
    assert completed.returncode == 0
