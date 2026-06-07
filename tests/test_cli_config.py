import shutil

import numpy as np
import pytest

from retarget.cli.config import (
    HolosomaClimbAdaptationConfig,
    HolosomaClimbObservationConfig,
    MotionFileObservationConfig,
    RetargetingRunConfig,
    RoleMappingRecipeConfig,
    SkateboardingObservationConfig,
)
from retarget.core.enums import CropPolicy, TaskKind, TimelineSelection
from retarget.pipeline import RetargetingExperiment


def test_run_config_is_a_frontend_over_experiment(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
name: yaml_config
robot: synthetic_humanoid
output: result.npz
observation:
  kind: motion_file
  path: motion.json
  format: minimal
recipe:
  kind: role_mapping
  task_kind: robot_only
  link_roles:
    Pelvis: pelvis
    L_Toe: left_foot
    R_Toe: right_foot
  mesh:
    topology: k_nearest
    k_neighbors: 2
  scene:
    ground_size: 3
""".strip()
    )

    config = RetargetingRunConfig.load(config_path)
    experiment = config.build_experiment()

    assert isinstance(config.observation, MotionFileObservationConfig)
    assert isinstance(config.recipe, RoleMappingRecipeConfig)
    assert config.observation.path == tmp_path / "motion.json"
    assert config.output == tmp_path / "result.npz"
    assert isinstance(experiment, RetargetingExperiment)

    problem = experiment.build_problem()
    assert problem.name == "yaml_config"
    assert problem.task_kind == TaskKind.ROBOT_ONLY
    assert problem.mesh.topology == "k_nearest"
    assert problem.scene.ground_size == 3
    assert problem.link_mapping["L_Toe"] == "left_toe"


def test_run_config_rejects_removed_source_boundary(tmp_path):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        """
robot = "synthetic_humanoid"
output = "result.npz"

[source]
kind = "motion_file"
path = "motion.json"
format = "minimal"
""".strip()
    )

    with pytest.raises(ValueError) as exc_info:
        RetargetingRunConfig.load(config_path)

    assert "observation" in str(exc_info.value)
    assert "source" in str(exc_info.value)


def test_example_configs_deserialize_to_typed_recipes():
    basic = RetargetingRunConfig.load("examples/basic/run_config.toml")
    skate = RetargetingRunConfig.load("examples/skateboarding/run_config.toml")
    holosoma = RetargetingRunConfig.load("examples/holosoma/run_config.toml")

    assert basic.observation.kind.value == "motion_file"
    assert basic.recipe.kind.value == "role_mapping"
    assert skate.observation.kind.value == "skateboarding"
    assert skate.recipe.kind.value == "skateboarding"
    assert holosoma.observation.kind.value == "holosoma_climb"
    assert holosoma.recipe.kind.value == "holosoma_climb"
    assert isinstance(skate.observation, SkateboardingObservationConfig)
    assert skate.observation.timeline_selection == TimelineSelection.HUMAN_POSE
    assert skate.observation.crop_policy == CropPolicy.OVERLAP
    assert isinstance(holosoma.observation, HolosomaClimbObservationConfig)
    assert holosoma.observation.policy.build().object_sample_count == 100
    assert isinstance(holosoma.recipe, HolosomaClimbAdaptationConfig)
    assert holosoma.recipe.policy.build().nominal_qpos_indices == tuple(range(19))


def test_run_config_preflights_references_before_loading(tmp_path):
    config = RetargetingRunConfig(
        observation=MotionFileObservationConfig(
            path=tmp_path / "missing.json",
            format_name="missing_format",
        ),
        recipe=RoleMappingRecipeConfig(),
        output=tmp_path / "result.npz",
        robot="missing_robot",
    )

    with pytest.raises(KeyError) as exc_info:
        config.build_problem()

    message = str(exc_info.value)
    assert "missing_format" in message
    assert "missing_robot" in message


def test_role_config_validates_source_and_robot_vocabularies(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    config = RetargetingRunConfig(
        observation=MotionFileObservationConfig(
            path=tmp_path / "motion.json",
            format_name="minimal",
        ),
        recipe=RoleMappingRecipeConfig(
            joint_roles={"not_a_joint": "left_hip"},
        ),
        output=tmp_path / "result.npz",
    )

    with pytest.raises(ValueError, match="not_a_joint"):
        config.build_problem()

    config = config.model_copy(
        update={
            "recipe": RoleMappingRecipeConfig(
                joint_roles={"L_Hip": "not_a_role"},
            )
        }
    )
    with pytest.raises(ValueError, match="not_a_role"):
        config.build_problem()


def test_run_config_builds_frame_aligned_object_scene(tmp_path):
    shutil.copyfile("tests/fixtures/minimal_motion.json", tmp_path / "motion.json")
    np.save(tmp_path / "points.npy", np.asarray([[1.0, 2.0, 3.0]]))
    config_path = tmp_path / "object.toml"
    config_path.write_text(
        """
name = "object_config"
robot = "synthetic_humanoid"
output = "result.npz"

[observation]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
task_kind = "object_interaction"

[recipe.scene.object]
name = "box"
sample_points_path = "points.npy"
identity_trajectory = true
""".strip()
    )

    problem = RetargetingRunConfig.load(config_path).build_problem()

    assert problem.scene.object is not None
    assert problem.scene.object.sample_points is not None
    assert problem.scene.object.trajectory is not None
    assert problem.scene.object.trajectory.poses.frame_count == problem.motion.frame_count


def test_run_config_resolves_import_and_robot_option_paths(tmp_path):
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
robot = "file_bot"
robot_provider = "file"
output = "result.npz"
imports = ["extensions/plugin.py"]

[robot_options]
path = "robot.toml"

[observation]
kind = "motion_file"
path = "motion.json"
format = "minimal"

[recipe]
kind = "role_mapping"
""".strip()
    )
    (tmp_path / "extensions").mkdir()
    (tmp_path / "extensions" / "plugin.py").write_text("")

    config = RetargetingRunConfig.load(config_path)

    assert config.imports == (str(tmp_path / "extensions" / "plugin.py"),)
    assert config.robot_options["path"] == tmp_path / "robot.toml"
