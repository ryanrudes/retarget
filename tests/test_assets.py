import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from retarget.assets import AssetInstallManifest, AssetStore
from retarget.core.enums import AssetKind


def test_asset_install_manifest_resolves_and_installs_local_assets(tmp_path):
    source = tmp_path / "source" / "fixture.txt"
    source.parent.mkdir()
    source.write_text("fixture")
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "assets.toml"
    manifest_path.write_text(
        f"""
[[assets]]
name = "tiny_fixture"
kind = "fixture"
source = "source/fixture.txt"
copy = true
destination = "fixtures/tiny_fixture.txt"
sha256 = "{expected_hash}"
license = "Apache-2.0"
notice = "Synthetic fixture created for tests."
""".strip()
    )

    manifest = AssetInstallManifest.load(manifest_path)
    records = AssetStore(tmp_path / "store").install_manifest(manifest)

    assert manifest.assets[0].source == str(source.resolve())
    assert len(records) == 1
    assert records[0].kind == AssetKind.FIXTURE
    assert records[0].path == tmp_path / "store" / "fixtures" / "tiny_fixture.txt"
    assert records[0].path.read_text() == "fixture"
    assert records[0].sha256 == expected_hash
    assert AssetStore(tmp_path / "store").load().find("tiny_fixture") is not None


def test_asset_install_refuses_downloads_without_opt_in(tmp_path):
    manifest_path = tmp_path / "assets.toml"
    manifest_path.write_text(
        """
[[assets]]
name = "remote"
kind = "object"
source = "https://example.invalid/object.zip"
""".strip()
    )

    with pytest.raises(ValueError, match="Refusing to download"):
        AssetStore(tmp_path / "store").install_manifest(manifest_path)


def test_research_asset_manifest_template_is_valid():
    manifest = AssetInstallManifest.load("examples/research_assets_manifest.toml")

    assert {asset.name for asset in manifest.assets} == {
        "g1",
        "t1",
        "interaction_object_set",
        "climbing_hold_set",
    }
    assert all(asset.license for asset in manifest.assets)
    assert all(asset.notice for asset in manifest.assets)


def test_bootstrap_robot_assets_from_local_holosoma_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_bootstrap_module(repo_root)
    holosoma_root = tmp_path / "holosoma"
    g1_dir = holosoma_root / module.HOLOSOMA_G1_DIR
    g1_dir.mkdir(parents=True)
    (holosoma_root / "LICENSE").write_text("Apache-2.0")
    (holosoma_root / "NOTICE").write_text("fixture notice")
    (g1_dir / module.G1_MUJOCO_XML).write_text("<mujoco model='g1_29dof'/>")
    (g1_dir / module.G1_URDF).write_text(_minimal_g1_urdf(module))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "bootstrap_robot_assets.py"),
            "g1",
            "--store",
            str(tmp_path / "store"),
            "--holosoma-root",
            str(holosoma_root),
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    record = AssetStore(tmp_path / "store").load().find("g1")
    assert record is not None
    assert record.kind == AssetKind.ROBOT
    assert (record.path / "robot.toml").exists()
    assert (record.path / "HOLOSOMA_LICENSE").exists()


def _load_bootstrap_module(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "bootstrap_robot_assets_test",
        repo_root / "scripts" / "bootstrap_robot_assets.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_g1_urdf(module) -> str:
    joint_names = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    links = "\n".join(f'  <link name="{name}"/>' for name in ("pelvis", *module.G1_LINK_NAMES))
    joints = "\n".join(
        f"""
  <joint name="{name}" type="revolute">
    <parent link="pelvis"/>
    <child link="{module.G1_LINK_NAMES[index % len(module.G1_LINK_NAMES)]}"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>""".rstrip()
        for index, name in enumerate(joint_names)
    )
    return f"<robot name=\"g1_29dof\">\n{links}\n{joints}\n</robot>\n"
