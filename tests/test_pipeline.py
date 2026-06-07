import json

import numpy as np
import pytest

from retarget.core.enums import (
    NonPenetrationSource,
    RunStatus,
    SceneGeometry,
    SolverBackend,
)
from retarget.kinematics import SimpleKinematicsBackend
from retarget.optimization import (
    GeometryPair,
    NonPenetrationConstraintConfig,
)
from retarget.pipeline import LinkBinding, Retargeter
from retarget.pipeline.compiled import compile_problem
from retarget.results import RetargetingResult
from tests.typed_fixtures import (
    FixtureMotionJoint,
    FixtureRobotGeometry,
    FixtureRobotLink,
    SameValueMotionJoint,
    SameValueRobotLink,
    fixture_problem,
)


class FixtureSceneGeometry(SceneGeometry):
    GROUND = "ground"


def test_compile_problem_resolves_names_once_and_keeps_typed_source():
    problem = fixture_problem()
    compiled = compile_problem(problem)

    assert compiled.source is problem
    assert compiled.joint_mapping == {}
    assert compiled.link_mapping == {
        FixtureMotionJoint.ROOT.value: FixtureRobotLink.PELVIS.value,
        FixtureMotionJoint.LEFT_FOOT.value: FixtureRobotLink.LEFT_FOOT.value,
        FixtureMotionJoint.RIGHT_FOOT.value: FixtureRobotLink.RIGHT_FOOT.value,
    }
    assert compiled.robot.joint_index("left_leg_joint") == 1
    assert compiled.link_name(FixtureRobotLink.LEFT_FOOT) == "left_foot"
    assert compiled.contacts is not None
    assert compiled.contacts.frame(0).active_link_names == ("left_foot",)
    assert compiled.targets is not None
    assert compiled.targets.frame(0).link_names == ("left_foot", "right_foot")


def test_problem_rejects_equal_binding_values_from_wrong_enum_classes():
    problem = fixture_problem()
    with pytest.raises(TypeError):
        problem.model_copy(
            update={
                "link_bindings": (
                    LinkBinding(SameValueMotionJoint.ROOT, FixtureRobotLink.PELVIS),
                )
            }
        )._validate_problem()

    with pytest.raises(TypeError):
        problem.model_copy(
            update={
                "link_bindings": (
                    LinkBinding(FixtureMotionJoint.ROOT, SameValueRobotLink.PELVIS),
                )
            }
        )._validate_problem()


def test_geometry_compilation_contains_only_explicit_constraint_references():
    problem = fixture_problem().model_copy(
        update={
            "constraints": (
                NonPenetrationConstraintConfig(
                    sources=(NonPenetrationSource.GEOMETRY,),
                    geometry_pairs=(
                        GeometryPair(
                            first=FixtureRobotGeometry.LEFT_FOOT,
                            second=FixtureSceneGeometry.GROUND,
                        ),
                    ),
                ),
            )
        }
    )
    problem._validate_problem()

    compiled = compile_problem(problem)

    assert compiled.referenced_geometry_names == ("left_foot_geom", "ground")
    assert "right_foot_geom" not in compiled.referenced_geometry_names


def test_simple_backend_rejects_missing_explicit_fixture_mapping_before_optimization():
    problem = fixture_problem()
    robot = problem.robot.model_copy(
        update={
            "simple_kinematics": {
                link: point
                for link, point in problem.robot.simple_kinematics.items()
                if link is not FixtureRobotLink.LEFT_FOOT
            }
        }
    )
    broken = problem.model_copy(update={"robot": robot})

    with pytest.raises(KeyError, match="left_foot"):
        Retargeter().run(broken)


def test_retargeter_runs_typed_problem_and_emits_structured_reports():
    problem = fixture_problem()

    result = Retargeter().run(problem)

    assert result.status is RunStatus.SUCCESS
    assert result.frame_count == problem.motion.frame_count
    assert result.run is not None
    assert result.run.robot_name == problem.robot.name
    assert result.run.solver.frame_statuses
    assert result.run.solver.requested_backend.resolve() is SolverBackend.NUMPY_LEAST_SQUARES
    assert result.run.solver.resolved_backend.resolve() is SolverBackend.NUMPY_LEAST_SQUARES
    assert result.playback is not None
    assert result.playback.robot.joint_vocabulary.qualname.endswith("FixtureRobotJoint")
    assert result.human_vocabulary is not None
    assert result.human_vocabulary.qualname.endswith("FixtureMotionJoint")


def test_result_checkpoint_is_pickle_free_and_preserves_vocabulary_identity(tmp_path):
    result = Retargeter().run(fixture_problem())
    path = result.save_npz(tmp_path / "result.npz")

    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "manifest_json",
            "qpos",
            "cost",
            "human_joints",
            "robot_link_positions",
        }
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["schema_version"] == 2
        assert manifest["human_vocabulary"]["qualname"].endswith("FixtureMotionJoint")
        assert "metadata" not in manifest

    loaded = RetargetingResult.load_npz(path)
    assert loaded.human_vocabulary == result.human_vocabulary
    assert loaded.playback == result.playback
    assert loaded.run is not None
    assert loaded.run.solver.resolved_backend.resolve() is SolverBackend.NUMPY_LEAST_SQUARES
    assert np.allclose(loaded.qpos, result.qpos)


def test_result_loader_rejects_legacy_object_array_checkpoint(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        qpos=np.zeros((1, 2)),
        metadata=np.asarray({"legacy": True}, dtype=object),
    )

    with pytest.raises(ValueError, match="current RetargetingResult"):
        RetargetingResult.load_npz(path)


def test_problem_provenance_cannot_control_behavior():
    with pytest.raises(ValueError, match="provenance cannot contain behavior"):
        fixture_problem().model_copy(
            update={"provenance": {"solver": "numpy_least_squares"}}
        )._provenance_only({"solver": "numpy_least_squares"})


def test_compiled_simple_backend_uses_explicit_points_not_name_heuristics():
    compiled = compile_problem(fixture_problem())
    backend = SimpleKinematicsBackend(compiled.robot)
    qpos = np.zeros(compiled.robot.qpos_size(), dtype=np.float64)
    qpos[compiled.robot.qpos_layout.joint_start + 1] = 0.5

    positions = backend.link_positions(qpos, ("left_foot",))

    assert np.allclose(positions[0], [-0.15, 0.1, -0.3])
