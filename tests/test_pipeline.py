import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from retarget import (
    ConstraintConfig,
    InteractionMeshBuilder,
    InteractionMeshRetargetingEngine,
    InteractionMeshSpec,
    MeshTopology,
    ObjectiveConfig,
    ObjectSampleSpace,
    ObjectSpec,
    ObjectTrajectory,
    OptimizationProfile,
    Retargeter,
    RetargetingProblem,
    RetargetingResult,
    RunStatus,
    SceneSpec,
    SolverBackend,
    SolverSpec,
    TaskKind,
)
from retarget.core.enums import ConvergenceMode, QuaternionOrder
from retarget.core.pose import PoseSequence
from retarget.kinematics import SimpleKinematicsBackend
from retarget.metrics import PenetrationMetric, evaluate_result, metrics
from retarget.motion import LinkTargetPlan, MotionSequence, load_motion, motion_formats
from retarget.optimization import (
    ConstraintContribution,
    JointLimitsConstraintConfig,
    LinkTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    ObjectiveContribution,
    SelfCollisionConstraintConfig,
    TermContext,
    TrustRegionConstraintConfig,
    constraint_terms,
    objective_terms,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline import engine as pipeline_engine
from retarget.robots import robots


def test_solver_iteration_count_supports_explicit_first_frame_count() -> None:
    solver = SolverSpec(max_iterations=3, first_frame_iterations=7)

    assert pipeline_engine._iteration_count(solver, 0) == 7
    assert pipeline_engine._iteration_count(solver, 1) == 3
    assert pipeline_engine._iteration_count(SolverSpec(max_iterations=3), 0) == 15


def test_solver_convergence_modes_are_explicit() -> None:
    cost_plateau = SolverSpec(convergence=ConvergenceMode.COST_PLATEAU, cost_atol=1e-8, cost_rtol=1e-5)
    step_norm = SolverSpec(convergence=ConvergenceMode.STEP_NORM, tolerance=1e-6)
    no_convergence = SolverSpec(convergence=ConvergenceMode.NONE)

    assert not pipeline_engine._should_stop_solver_iteration(
        cost_plateau,
        solution=np.ones(2, dtype=np.float64),
        cost=1.0,
        previous_cost=np.inf,
    )
    assert pipeline_engine._should_stop_solver_iteration(
        cost_plateau,
        solution=np.ones(2, dtype=np.float64),
        cost=1.0 + 1e-9,
        previous_cost=1.0,
    )
    assert pipeline_engine._should_stop_solver_iteration(
        step_norm,
        solution=np.zeros(2, dtype=np.float64),
        cost=10.0,
        previous_cost=10.0,
    )
    assert not pipeline_engine._should_stop_solver_iteration(
        no_convergence,
        solution=np.zeros(2, dtype=np.float64),
        cost=10.0,
        previous_cost=10.0,
    )


def test_retargeter_runs_minimal_fixture(tmp_path):
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
    )
    result = Retargeter().run(problem)
    assert result.qpos.shape == (motion.frame_count, robot.qpos_size())
    assert np.all(np.isfinite(result.qpos))
    assert result.cost is not None
    assert result.robot_link_positions is not None
    assert result.robot_link_positions.shape == (motion.frame_count, len(robot.link_names), 3)
    assert result.metadata["algorithm"] == "interaction_mesh_sqp"
    assert result.metadata["motion"] == motion.name
    assert result.metadata["motion_format"] == "minimal"
    assert result.metadata["resolved_solver"] in {"numpy_least_squares", "cvxpy_clarabel"}
    assert len(result.metadata["solver_statuses"]) == motion.frame_count
    assert result.metadata["link_mapping"] == {}
    out = result.save_npz(tmp_path / "result.npz")
    assert out.exists()
    loaded = RetargetingResult.load_npz(out)
    assert loaded.robot_link_positions is not None
    assert loaded.robot_link_positions.shape == result.robot_link_positions.shape
    provenance = loaded.metadata["provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["motion"]["name"] == motion.name
    assert provenance["robot"]["name"] == robot.name
    assert provenance["solver"]["backend"] == "auto"
    assert provenance["solver"]["actual_backend"] == result.metadata["resolved_solver"]
    assert len(provenance["solver"]["frame_statuses"]) == motion.frame_count
    assert provenance["mesh"] == {
        "topology": "delaunay",
        "k_neighbors": 4,
        "laplacian_weighting": "uniform",
        "laplacian_epsilon": 1e-06,
        "source": "problem",
    }
    assert result.metadata["mesh"] == provenance["mesh"]
    assert result.metadata["playback"]["robot"]["name"] == robot.name
    assert result.metadata["playback"]["robot"]["link_names"] == list(robot.link_names)
    assert provenance["result"]["frame_count"] == motion.frame_count
    assert [objective["kind"] for objective in provenance["objectives"]] == ["laplacian", "smoothness"]
    report = evaluate_result(result)
    assert "optimization_cost" in report.metrics
    assert report.source_name == "fixture"
    assert report.frame_count == motion.frame_count
    assert report.metric_units["optimization_cost"] == "cost"


def test_object_asset_scale_feeds_environment_and_playback_metadata() -> None:
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    mesh_path = Path("/tmp/box.obj")
    object_spec = ObjectSpec(
        name="scaled_box",
        mesh_path=mesh_path,
        asset_scale=(2.0, 3.0, 4.0),
        visual_parts=[
            {
                "name": "box1",
                "mesh_path": mesh_path,
                "asset_scale": (2.0, 3.0, 4.0),
                "rgba": (0.3, 0.7, 0.9, 0.5),
            }
        ],
        sample_points=np.asarray([[0.5, 1.0, 1.5]], dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="scaled_object",
        task_kind=TaskKind.CLIMBING,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.climbing(object_spec=object_spec),
    )

    assert np.allclose(object_spec.scaled_sample_points(), [[1.0, 3.0, 6.0]])
    assert np.allclose(pipeline_engine._environment_points(problem, None), [[1.0, 3.0, 6.0]])

    playback = pipeline_engine._object_playback_metadata(problem)

    assert playback is not None
    assert playback["asset_scale"] == [2.0, 3.0, 4.0]
    assert playback["sample_points"] == [[1.0, 3.0, 6.0]]
    assert playback["sample_points_space"] == ObjectSampleSpace.OBJECT_LOCAL.value
    assert playback["visual_parts"] == [
        {
            "name": "box1",
            "mesh_path": str(mesh_path),
            "asset_scale": [2.0, 3.0, 4.0],
            "rgba": [0.3, 0.7, 0.9, 0.5],
        }
    ]


def test_object_world_sample_space_is_not_scaled() -> None:
    object_spec = ObjectSpec(
        name="world_points",
        asset_scale=(2.0, 3.0, 4.0),
        sample_points=np.asarray([[0.5, 1.0, 1.5]], dtype=np.float64),
        sample_space=ObjectSampleSpace.WORLD,
    )

    assert np.allclose(object_spec.scaled_sample_points(), [[0.5, 1.0, 1.5]])


def test_scale_to_robot_warns_when_source_height_unknown() -> None:
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal").model_copy(
        update={"source_height_m": None, "metadata": {"height_m": 1.7}},
    )
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="scale_warning",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=None,
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES),
        scale_to_robot=True,
    )

    result = Retargeter().run(problem)

    assert any("scale_to_robot" in warning for warning in result.warnings)
    assert result.metadata["provenance"]["motion_scale_factor"] is None


def test_scale_to_robot_does_not_warn_when_source_height_known() -> None:
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="scale_ok",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES),
        scale_to_robot=True,
    )

    result = Retargeter().run(problem)

    assert not any("scale_to_robot" in warning for warning in result.warnings)
    assert result.metadata["provenance"]["motion_scale_factor"] is not None


def test_engine_advances_progress_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="progress_fixture",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES),
        show_progress=True,
    )
    advances: list[int] = []

    def fake_frame_progress(**_kwargs: object):
        from contextlib import contextmanager

        @contextmanager
        def _manager():
            def advance() -> None:
                advances.append(1)

            yield advance

        return _manager()

    monkeypatch.setattr(pipeline_engine, "frame_progress", fake_frame_progress)

    Retargeter().run(problem)

    assert len(advances) == motion.frame_count


def test_retargeter_output_fps_resamples_motion_before_optimization():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture_60hz",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(max_iterations=1),
        output_fps=60.0,
    )

    result = Retargeter().run(problem)

    assert result.fps == 60.0
    assert result.frame_count == 5
    assert result.human_joints is not None
    assert result.human_joints.shape[0] == 5


def test_evaluation_aligns_problem_to_result_output_fps():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture_60hz_eval",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(max_iterations=1),
        output_fps=60.0,
    )
    result = Retargeter().run(problem)

    report = evaluate_result(result, problem)

    assert report.status == RunStatus.SUCCESS
    assert report.frame_count == 5
    assert not report.warnings
    assert report.details["problem"]["motion_frame_count"] == result.frame_count
    assert report.details["problem"]["aligned_to_result"] is True
    assert "foot_sliding" in report.metrics


def test_problem_mesh_spec_controls_default_engine_topology():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture_k_nearest",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        mesh=InteractionMeshSpec(topology=MeshTopology.K_NEAREST, k_neighbors=2),
        solver=SolverSpec(max_iterations=1),
    )

    result = Retargeter().run(problem)

    assert result.metadata["mesh"] == {
        "topology": "k_nearest",
        "k_neighbors": 2,
        "laplacian_weighting": "uniform",
        "laplacian_epsilon": 1e-06,
        "source": "problem",
    }
    assert result.metadata["provenance"]["mesh"] == result.metadata["mesh"]


def test_custom_engine_mesh_builder_overrides_problem_mesh_spec():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture_engine_mesh",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        mesh=InteractionMeshSpec(topology=MeshTopology.K_NEAREST, k_neighbors=2),
        solver=SolverSpec(max_iterations=1),
    )
    engine = InteractionMeshRetargetingEngine(
        mesh_builder=InteractionMeshBuilder(topology=MeshTopology.COMPLETE, k_neighbors=7)
    )

    result = Retargeter(engine=engine).run(problem)

    assert result.metadata["mesh"] == {
        "topology": "complete",
        "k_neighbors": 7,
        "laplacian_weighting": "uniform",
        "laplacian_epsilon": 1e-06,
        "source": "engine",
    }


def test_retargeter_initializes_root_qpos_from_motion_root_poses():
    source = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    root_positions = np.tile(np.asarray([[0.25, -0.5, 1.25]], dtype=np.float64), (source.frame_count, 1))
    root_quaternions = np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (source.frame_count, 1))
    motion = MotionSequence(
        name="fixture_with_root_pose",
        joint_positions=source.joint_positions,
        joint_names=source.joint_names,
        fps=source.fps,
        frame=source.frame,
        root_poses=PoseSequence.from_arrays(
            root_positions,
            root_quaternions,
            fps=source.fps,
            quaternion_order=QuaternionOrder.XYZW,
            frame=source.frame,
        ),
        metadata=dict(source.metadata),
    )
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture_root_pose",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        solver=SolverSpec(max_iterations=1),
        scale_to_robot=False,
    )

    result = Retargeter().run(problem)

    assert np.allclose(result.qpos[:, 0:3], root_positions)
    assert np.allclose(result.qpos[:, 3:7], [[1.0, 0.0, 0.0, 0.0]] * source.frame_count)


def test_retargeter_can_optimize_root_translation_as_typed_qpos_variable():
    robot = robots.get("synthetic_humanoid")
    backend = SimpleKinematicsBackend(robot)
    qpos = np.zeros(robot.qpos_size(), dtype=np.float64)
    qpos[3] = 1.0
    current, _jacobian = backend.point_jacobians_for_qpos_indices(
        qpos,
        ("left_toe",),
        np.asarray([0], dtype=np.int64),
    )
    motion = MotionSequence(
        name="root_variable",
        joint_names=("Pelvis",),
        joint_positions=np.zeros((1, 1, 3), dtype=np.float64),
        source_height_m=robot.height_m,
    )
    targets = LinkTargetPlan.from_arrays(
        link_names=("left_toe",),
        positions=np.asarray([[current[0] + np.array([0.1, 0.0, 0.0], dtype=np.float64)]], dtype=np.float64),
    )
    problem = RetargetingProblem(
        name="root_variable",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        scene=SceneSpec.robot_only(),
        targets=targets,
        variables=QposVariableSpec.qpos_indices((0,)),
        objectives=(LinkTrackingObjectiveConfig(),),
        constraints=(TrustRegionConstraintConfig(radius=1.0),),
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES, max_iterations=2),
        scale_to_robot=False,
    )

    result = Retargeter().run(problem)

    assert np.allclose(result.qpos[0, 0], 0.1, atol=1e-7)
    assert result.metadata["variables"]["indices"] == [0]


def test_problem_output_fps_resamples_dynamic_object_trajectory():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    object_spec = ObjectSpec(
        name="box",
        trajectory=ObjectTrajectory.identity(motion.frame_count, fps=motion.fps, name="box"),
    )
    problem = RetargetingProblem(
        name="object_60hz",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.object_interaction(object_spec),
        output_fps=60.0,
    )

    prepared = problem.with_output_fps_applied()

    assert prepared.motion.frame_count == 5
    assert prepared.scene.object is not None
    assert prepared.scene.object.trajectory is not None
    assert prepared.scene.object.trajectory.poses.frame_count == 5


def test_object_result_metadata_includes_visualizer_playback_object():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    object_spec = ObjectSpec(
        name="board",
        sample_points=np.asarray([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64),
        trajectory=ObjectTrajectory.identity(motion.frame_count, fps=motion.fps, name="board"),
    )
    problem = RetargetingProblem(
        name="object_playback_metadata",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.object_interaction(object_spec),
        solver=SolverSpec(max_iterations=1),
    )

    result = Retargeter().run(problem)

    playback = result.metadata["playback"]["object"]
    assert playback["name"] == "board"
    assert playback["qpos_slice"] == [
        robot.qpos_layout.object_slice(robot.dof).start,
        robot.qpos_layout.object_slice(robot.dof).stop,
    ]
    assert playback["sample_points"] == [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]


def test_problem_registry_preflight_reports_missing_extension_references():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")

    class MissingObjectiveConfig(ObjectiveConfig):
        kind: Literal["missing_objective"] = "missing_objective"

    class MissingConstraintConfig(ConstraintConfig):
        kind: Literal["missing_constraint"] = "missing_constraint"

    problem = RetargetingProblem(
        name="missing_extensions",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        objectives=(MissingObjectiveConfig(),),
        constraints=(MissingConstraintConfig(),),
        solver=SolverSpec(backend="missing_solver"),
    )

    with pytest.raises(KeyError) as exc_info:
        problem.validate_registry_references()

    message = str(exc_info.value)
    assert "missing_objective" in message
    assert "missing_constraint" in message
    assert "missing_solver" in message
    assert "laplacian" in message
    assert "joint_limits" in message
    assert "numpy_least_squares" in message


def test_problem_can_apply_reusable_optimization_profile():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    base_problem = RetargetingProblem(
        name="profiled",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
    )
    from retarget.optimization import SmoothnessObjectiveConfig

    profile = OptimizationProfile.defaults(name="low_smoothness").with_objective(SmoothnessObjectiveConfig(weight=0.01))

    problem = base_problem.with_optimization_profile(profile)

    assert problem.metadata["optimization_profile"] == "low_smoothness"
    assert [objective.kind for objective in problem.objectives] == ["laplacian", "smoothness"]
    assert problem.objectives[-1].weight == 0.01
    assert base_problem.objectives[-1].weight == 0.2


def test_problem_rejects_behavioral_metadata():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")

    with pytest.raises(ValueError, match="provenance-only"):
        RetargetingProblem(
            name="bad_metadata",
            task_kind=TaskKind.ROBOT_ONLY,
            robot=robot,
            motion=motion,
            scene=SceneSpec.robot_only(),
            metadata={"solver": "numpy_least_squares"},
        )


def test_registered_custom_objective_and_constraint_affect_retargeting():
    class FirstJointTargetConfig(ObjectiveConfig):
        kind: Literal["unit_test_first_joint_target"] = "unit_test_first_joint_target"
        target: float = 0.0

    class FirstJointCapConfig(ConstraintConfig):
        kind: Literal["unit_test_first_joint_cap"] = "unit_test_first_joint_cap"
        upper: float

    class FirstJointTargetObjective:
        name = "unit_test_first_joint_target"
        config_type = FirstJointTargetConfig

        def describe(self) -> str:
            return "Drive the first actuated joint toward a configured target."

        def build(self, context: TermContext, config: FirstJointTargetConfig) -> tuple[ObjectiveContribution, ...]:
            matrix = np.zeros((1, context.dof), dtype=np.float64)
            matrix[0, 0] = 1.0
            target = config.target
            return (ObjectiveContribution(matrix=matrix, target=np.asarray([target - context.current_joints[0]])),)

    class FirstJointCapConstraint:
        name = "unit_test_first_joint_cap"
        config_type = FirstJointCapConfig

        def describe(self) -> str:
            return "Cap the first actuated joint at an absolute upper limit."

        def build(self, context: TermContext, config: FirstJointCapConfig) -> ConstraintContribution:
            limit = config.upper
            lower = np.full(context.dof, -1e6, dtype=np.float64)
            upper = np.full(context.dof, 1e6, dtype=np.float64)
            upper[0] = limit - context.current_joints[0]
            return ConstraintContribution(lower=lower, upper=upper)

    objective_terms.register("unit_test_first_joint_target", FirstJointTargetObjective(), replace=True)
    constraint_terms.register("unit_test_first_joint_cap", FirstJointCapConstraint(), replace=True)
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="custom_terms",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        objectives=(FirstJointTargetConfig(target=0.5),),
        constraints=(FirstJointCapConfig(upper=0.1),),
        solver=SolverSpec(backend=SolverBackend.NUMPY_LEAST_SQUARES, max_iterations=2),
    )

    result = Retargeter().run(problem)

    first_joint = result.qpos[:, robot.qpos_layout.joint_start]
    assert np.allclose(first_joint, 0.1, atol=1e-7)


def test_result_resampled_interpolates_time_aligned_arrays():
    result = RetargetingResult(
        name="linear_result",
        status=RunStatus.SUCCESS,
        qpos=np.asarray([[0.0], [2.0]], dtype=np.float64),
        fps=1.0,
        cost=np.asarray([0.0, 4.0], dtype=np.float64),
        human_joints=np.asarray([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=np.float64),
    )

    resampled = result.resampled(2.0)

    assert resampled.frame_count == 3
    assert np.allclose(resampled.qpos[:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(resampled.cost, [0.0, 2.0, 4.0])
    assert resampled.human_joints is not None
    assert np.allclose(resampled.human_joints[:, 0, 0], [0.0, 1.0, 2.0])


def test_result_npz_includes_versioned_json_metadata(tmp_path):
    result = RetargetingResult(
        name="json_metadata",
        status=RunStatus.SUCCESS,
        qpos=np.asarray([[0.0, 1.0]], dtype=np.float64),
        fps=30.0,
        warnings=("check foot contacts",),
        metadata={"robot": "fixture", "shape": np.asarray([1, 2], dtype=np.int64)},
    )
    path = result.save_npz(tmp_path / "result.npz")

    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(())))
    warnings = json.loads(str(np.asarray(data["warnings_json"]).reshape(())))
    loaded = RetargetingResult.load_npz(path)

    assert int(np.asarray(data["schema_version"]).reshape(())) == 1
    assert metadata == {"robot": "fixture", "shape": [1, 2]}
    assert warnings == ["check foot contacts"]
    assert loaded.metadata == metadata
    assert loaded.warnings == ("check foot contacts",)


def test_result_npz_legacy_object_metadata_requires_explicit_pickle(tmp_path):
    path = tmp_path / "legacy_result.npz"
    np.savez(
        path,
        schema_version=np.asarray(1),
        name="legacy",
        status="success",
        qpos=np.zeros((1, 1), dtype=np.float64),
        fps=np.asarray(30.0),
        metadata=np.asarray({"robot": "legacy_bot"}, dtype=object),
        warnings=np.asarray(("legacy warning",), dtype=object),
    )

    with pytest.raises(ValueError, match="allow_pickle=True"):
        RetargetingResult.load_npz(path)

    loaded = RetargetingResult.load_npz(path, allow_pickle=True)

    assert loaded.metadata == {"robot": "legacy_bot"}
    assert loaded.warnings == ("legacy warning",)


def test_self_collision_constraint_uses_backend_distances():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        constraints=(
            JointLimitsConstraintConfig(),
            SelfCollisionConstraintConfig(minimum_distance=0.3, pairs=(("left_toe", "right_toe"),)),
        ),
        solver=SolverSpec(max_iterations=2),
    )
    qpos = np.zeros(robot.qpos_size())
    qpos[3] = 1.0

    contribution = constraint_terms.get("self_collision").build(
        _term_context(problem, SimpleKinematicsBackend(robot), qpos),
        problem.constraints[1],
    )
    constraints = list(contribution.linear_constraints)

    assert len(constraints) == 1
    assert constraints[0].matrix.shape == (1, robot.dof)
    assert constraints[0].lower is not None
    assert np.isclose(constraints[0].lower[0], 0.06)


def test_scene_non_penetration_uses_object_sample_points():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    object_spec = ObjectSpec(name="fixture", sample_points=np.asarray([[-0.12, 0.0, -1.02]], dtype=np.float64))
    problem = RetargetingProblem(
        name="object_fixture",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.object_interaction(object_spec),
        constraints=(
            NonPenetrationConstraintConfig(links=("left_toe",), scene_clearance=0.05, activation_distance=0.1),
        ),
    )
    qpos = np.zeros(robot.qpos_size())
    qpos[3] = 1.0

    contribution = constraint_terms.get("non_penetration").build(
        _term_context(problem, SimpleKinematicsBackend(robot), qpos),
        problem.constraints[0],
    )
    constraints = list(contribution.linear_constraints)
    scene_constraints = [
        constraint
        for constraint in constraints
        if constraint.lower is not None and np.isclose(constraint.lower[0], 0.05)
    ]

    assert len(scene_constraints) == 1
    assert scene_constraints[0].matrix.shape == (1, robot.dof)


def test_penetration_metric_includes_scene_clearance_violation():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    object_spec = ObjectSpec(name="fixture", sample_points=np.asarray([[-0.12, 0.0, -1.02]], dtype=np.float64))
    problem = RetargetingProblem(
        name="object_fixture",
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.object_interaction(object_spec),
        constraints=(NonPenetrationConstraintConfig(floor_z=-2.0, scene_clearance=0.05),),
    )
    qpos = np.zeros((1, robot.qpos_size()))
    qpos[:, 3] = 1.0
    result = RetargetingResult(name="object_fixture", status=RunStatus.SUCCESS, qpos=qpos, fps=motion.fps)

    assert np.isclose(PenetrationMetric().evaluate(result, problem), 0.05)


def test_retargeter_runs_with_self_collision_constraint():
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
        constraints=(
            JointLimitsConstraintConfig(),
            TrustRegionConstraintConfig(),
            SelfCollisionConstraintConfig(minimum_distance=0.02, pairs=(("left_toe", "right_toe"),)),
        ),
        solver=SolverSpec(max_iterations=2),
    )

    result = Retargeter().run(problem)

    assert result.status == "success"
    assert not result.warnings


def test_evaluation_report_includes_context_and_handles_metric_failures(monkeypatch):
    motion = load_motion("tests/fixtures/minimal_motion.json", "minimal")
    robot = robots.get("synthetic_humanoid")
    problem = RetargetingProblem(
        name="fixture",
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get("minimal"),
        scene=SceneSpec.robot_only(),
    )
    qpos = np.zeros((motion.frame_count, robot.qpos_size()))
    qpos[:, 3] = 1.0
    result = RetargetingResult(name="fixture", status=RunStatus.SUCCESS, qpos=qpos, fps=motion.fps)

    class BrokenMetric:
        def evaluate(self, result, problem=None):
            raise RuntimeError("metric failed")

    monkeypatch.setattr(metrics, "_items", {**metrics._items, "broken_metric": BrokenMetric()})

    report = evaluate_result(result, problem)

    assert report.status == RunStatus.PARTIAL
    assert report.source_name == "fixture"
    assert report.task_kind == "robot_only"
    assert report.robot_name == "synthetic_humanoid"
    assert report.motion_name == motion.name
    assert report.details["problem"]["solver_backend"] == "auto"
    assert "broken_metric" not in report.metrics
    assert any("broken_metric" in warning for warning in report.warnings)


def _term_context(
    problem: RetargetingProblem,
    backend: SimpleKinematicsBackend,
    qpos: np.ndarray,
) -> TermContext:
    dof = problem.robot.dof
    return TermContext(
        problem=problem,
        backend=backend,
        q_current=qpos,
        q_previous=qpos.copy(),
        frame_idx=0,
        contact_frame=None,
        target_frame=None,
        robot_point_names=(),
        robot_points=np.zeros((0, 3), dtype=np.float64),
        robot_jacobians=np.zeros((0, 3, dof), dtype=np.float64),
        environment_points=np.zeros((0, 3), dtype=np.float64),
        adjacency=(),
        target_laplacian=np.zeros((0, 3), dtype=np.float64),
        laplacian_weighting=problem.mesh.laplacian_weighting,
        laplacian_epsilon=problem.mesh.laplacian_epsilon,
        reference_pose=None,
        joint_lower=np.full(dof, -1e6, dtype=np.float64),
        joint_upper=np.full(dof, 1e6, dtype=np.float64),
        current_joints=qpos[problem.robot.qpos_layout.joint_slice(dof)],
    )
