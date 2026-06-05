import shutil

import numpy as np
import pytest

from retarget.assets import AssetStore
from retarget.cli.config import MotionFileSourceConfig, RetargetingRunConfig
from retarget.core.enums import AssetKind, TaskKind


def test_run_config_resolves_paths_and_builds_problem(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
name: yaml_config
robot: synthetic_humanoid
task_kind: robot_only
output: result.npz
source:
  kind: motion_file
  path: motion.json
  format: minimal
mesh:
  topology: k_nearest
  k_neighbors: 2
variables:
  kind: qpos_slice
  actuated_start_offset: -7
scene:
  ground_size: 3
constraints:
  - kind: joint_limits
  - kind: trust_region
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)
    assert isinstance(config.source, MotionFileSourceConfig)
    assert config.source.path == tmp_path / "motion.json"
    assert config.output == tmp_path / "result.npz"
    assert config.task_kind == TaskKind.ROBOT_ONLY

    problem = config.build_problem()
    assert problem.name == "yaml_config"
    assert problem.mesh.topology == "k_nearest"
    assert problem.mesh.k_neighbors == 2
    assert problem.variables.kind == "qpos_slice"
    assert problem.variables.actuated_start_offset == -7
    resolved_variables = problem.variables.resolve(problem.robot, qpos_size=problem.robot.qpos_size())
    assert resolved_variables.indices[0] == 0
    assert problem.scene.ground_size == 3


def test_run_config_rejects_legacy_top_level_motion_fields(tmp_path):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
name = "legacy"
motion = "motion.json"
format = "minimal"
robot = "synthetic_humanoid"
output = "result.npz"
""".strip()
    )

    with pytest.raises(ValueError) as exc_info:
        RetargetingRunConfig.load(config_path)

    message = str(exc_info.value)
    assert "source" in message
    assert "motion" in message
    assert "format" in message


def test_run_config_rejects_legacy_objective_parameter_blocks(tmp_path):
    config_path = tmp_path / "legacy_parameters.toml"
    config_path.write_text(
        """
robot = "synthetic_humanoid"
output = "result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[[objectives]]
kind = "smoothness"
parameters = { weight = 0.2 }
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)

    with pytest.raises(ValueError) as exc_info:
        config.resolved_objectives()

    assert "parameters" in str(exc_info.value)


def test_run_config_upgrades_motion_contacts_to_typed_contact_plan(tmp_path):
    np.savez(
        tmp_path / "motion.npz",
        joint_positions=np.zeros((2, 3, 3), dtype=np.float64),
        joint_names=np.asarray(("Pelvis", "L_Toe", "R_Toe"), dtype=object),
        fps=np.asarray(30.0),
        contact_states=np.asarray([[True, False], [False, True]], dtype=bool),
        contact_names=np.asarray(("L_Foot", "R_Foot"), dtype=object),
        support_plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        support_plane_origin=np.asarray([0.0, 0.0, 0.2], dtype=np.float64),
        contact_source=np.asarray("motion_sync:foot_support", dtype=object),
        contact_model_fingerprint=np.asarray("model123", dtype=object),
        contact_timeline_fingerprint=np.asarray("clip123", dtype=object),
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
name = "typed_contacts"
robot = "synthetic_humanoid"
output = "result.npz"

[source]
kind = "motion_file"
path = "motion.npz"
format = "minimal"
""".strip()
    )

    problem = RetargetingRunConfig.load(config_path).build_problem()

    assert problem.contacts is not None
    assert problem.contacts.frame(0).active_link_names == ("left_toe",)
    assert problem.contacts.frame(1).active_link_names == ("right_toe",)
    assert problem.contacts.support is not None
    assert np.allclose(problem.contacts.support.origin, [0.0, 0.0, 0.2])
    assert problem.contacts.provenance["contact_source"] == "motion_sync:foot_support"
    assert problem.contacts.provenance["contact_model_fingerprint"] == "model123"


def test_run_config_resolves_relative_import_paths(tmp_path):
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
robot = "synthetic_humanoid"
output = "result.npz"
imports = ["extensions/custom_terms.py", "retarget.motion"]

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)

    assert config.imports == (str(tmp_path / "extensions" / "custom_terms.py"), "retarget.motion")


def test_run_config_imports_dotted_extensions_before_registry_preflight(tmp_path, monkeypatch):
    package_dir = tmp_path / "unit_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        """
from dataclasses import dataclass

import numpy as np

from typing import Literal

from retarget.optimization import ObjectiveConfig, ObjectiveContribution, TermContext, objective_terms


class UnitPluginEnergyConfig(ObjectiveConfig):
    kind: Literal["unit_plugin_energy"] = "unit_plugin_energy"


@objective_terms.register("unit_plugin_energy", replace=True)
@dataclass(frozen=True)
class UnitPluginEnergy:
    name: str = "unit_plugin_energy"
    config_type: type[UnitPluginEnergyConfig] = UnitPluginEnergyConfig

    def describe(self) -> str:
        return "A unit-test extension objective."

    def build(self, context: TermContext, _config: UnitPluginEnergyConfig) -> tuple[ObjectiveContribution, ...]:
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=np.zeros(context.dof, dtype=np.float64),
            ),
        )
""".strip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = RetargetingRunConfig(
        source=MotionFileSourceConfig(path=tmp_path / "missing_motion.json", format_name="minimal"),
        output=tmp_path / "result.npz",
        imports=("unit_plugin",),
        objectives=({"kind": "unit_plugin_energy"},),
    )

    config.validate_registry_references()


def test_run_config_loads_robot_from_file_provider(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    (tmp_path / "robot.toml").write_text(
        """
name = "file_bot"
dof = 1
height_m = 1.0
joint_names = ["joint"]
link_names = ["link"]

[joint_limits]
joint = [-1.0, 1.0]
""".strip()
    )
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
name = "file_robot_config"
robot = "file_bot"
robot_provider = "file"
output = "result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[robot_options]
path = "robot.toml"
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)
    problem = config.build_problem()

    assert config.robot_options["path"] == tmp_path / "robot.toml"
    assert problem.robot.name == "file_bot"
    assert problem.robot.joint_names == ("joint",)


def test_run_config_loads_robot_from_asset_store_provider(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    asset_dir = tmp_path / "asset_robot"
    asset_dir.mkdir()
    (asset_dir / "robot.toml").write_text(
        """
name = "asset_bot"
dof = 1
height_m = 1.0
joint_names = ["joint"]
link_names = ["link"]

[joint_limits]
joint = [-1.0, 1.0]
""".strip()
    )
    AssetStore(tmp_path / "store").import_path(asset_dir, name="asset_bot", kind=AssetKind.ROBOT)
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
name = "asset_robot_config"
robot = "asset_bot"
robot_provider = "asset_store"
output = "result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[robot_options]
store = "store"
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)
    problem = config.build_problem()

    assert config.robot_options["store"] == tmp_path / "store"
    assert problem.robot.name == "asset_bot"


def test_run_config_preflights_registry_references_before_loading_files(tmp_path):
    config = RetargetingRunConfig(
        source=MotionFileSourceConfig(path=tmp_path / "missing_motion.json", format_name="missing_format"),
        output=tmp_path / "result.npz",
        robot="missing_robot",
    )

    with pytest.raises(KeyError) as exc_info:
        config.build_problem()

    message = str(exc_info.value)
    assert "missing_format" in message
    assert "missing_robot" in message
    assert "minimal" in message
    assert "synthetic_humanoid" in message


def test_run_config_preflights_optimization_references_before_loading_files(tmp_path):
    config = RetargetingRunConfig(
        source=MotionFileSourceConfig(path=tmp_path / "missing_motion.json", format_name="minimal"),
        output=tmp_path / "result.npz",
        objectives=({"kind": "missing_cli_objective"},),
    )

    with pytest.raises(KeyError) as exc_info:
        config.build_problem()

    message = str(exc_info.value)
    assert "missing_cli_objective" in message
    assert "laplacian" in message


def test_run_config_loads_object_trajectory_and_sample_points(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    np.save(tmp_path / "object_points.npy", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64))
    np.savez(
        tmp_path / "object_trajectory.npz",
        positions=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64),
        quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float64),
        fps=np.asarray(30.0),
    )
    config_path = tmp_path / "object.toml"
    config_path.write_text(
        """
name = "object_config"
robot = "synthetic_humanoid"
task_kind = "object_interaction"
output = "object_result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[scene.object]
name = "box"
sample_points_path = "object_points.npy"
trajectory_path = "object_trajectory.npz"
""".strip()
    )

    problem = RetargetingRunConfig.load(config_path).build_problem()

    assert problem.task_kind == TaskKind.OBJECT_INTERACTION
    assert problem.scene.object is not None
    assert problem.scene.object.sample_points is not None
    assert problem.scene.object.sample_points.shape == (1, 3)
    assert problem.scene.object.trajectory is not None
    assert problem.scene.object.trajectory.poses.frame_count == problem.motion.frame_count


def test_run_config_loads_terrain_sample_points_from_json(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    (tmp_path / "terrain_points.json").write_text('{"sample_points": [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]}')
    config_path = tmp_path / "climbing.yaml"
    config_path.write_text(
        """
name: climbing_config
robot: synthetic_humanoid
task_kind: climbing
output: climbing_result.npz
source:
  kind: motion_file
  path: motion.json
  format: minimal
scene:
  terrain:
    name: holds
    sample_points_path: terrain_points.json
""".strip()
    )

    problem = RetargetingRunConfig.load(config_path).build_problem()

    assert problem.task_kind == TaskKind.CLIMBING
    assert problem.scene.terrain is not None
    assert problem.scene.terrain.sample_points is not None
    assert np.allclose(problem.scene.terrain.sample_points[:, 2], [1.0, 1.0])


def test_run_config_samples_object_points_from_obj_mesh(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    (tmp_path / "box.obj").write_text(
        """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
""".strip()
    )
    config_path = tmp_path / "object_mesh.toml"
    config_path.write_text(
        """
name = "object_mesh_config"
robot = "synthetic_humanoid"
task_kind = "object_interaction"
output = "object_mesh_result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[scene.object]
name = "box"
mesh_path = "box.obj"
mesh_sample_count = 4
""".strip()
    )

    problem = RetargetingRunConfig.load(config_path).build_problem()

    assert problem.scene.object is not None
    assert problem.scene.object.mesh_path == tmp_path / "box.obj"
    assert problem.scene.object.sample_points is not None
    assert problem.scene.object.sample_points.shape == (4, 3)
    assert np.allclose(problem.scene.object.sample_points[:, 2], 0.0)
