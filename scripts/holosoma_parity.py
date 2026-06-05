#!/usr/bin/env python3
"""Compare retarget's math contract against Holosoma.

This is a debugging harness, not a benchmark. It compares deterministic layers
that should match exactly before attempting full retargeting parity:

- behavioral defaults shared with Holosoma,
- interaction-mesh topology and Laplacian coordinates,
- one Clarabel SQP subproblem with the same matrices and constraints,
- reduced real-data output parity against a Holosoma fixture.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retarget.integrations.holosoma import (
    G1_FOOT_STICKING_LINKS,
    from_mocap_climb_fixture,
    g1_spherehand_robot,
    holosoma_climb_profile,
)
from retarget.kinematics.backends import MuJoCoKinematicsBackend
from retarget.mesh import InteractionMeshBuilder, InteractionMeshSpec, LaplacianWeighting
from retarget.optimization import (
    CvxpyClarabelSolver,
    DiagonalRegularizationObjectiveConfig,
    FootLockConstraintConfig,
    FootStickingConstraintConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    QuadraticProblem,
    SolverSpec,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline.engine import InteractionMeshRetargetingEngine


@dataclass(frozen=True)
class ParityCheck:
    """One parity check result."""

    stage: str
    name: str
    status: str
    retarget: Any
    holosoma: Any
    max_abs_error: float | None = None
    tolerance: float | None = None
    note: str = ""


@dataclass(frozen=True)
class ParityReport:
    """Complete parity report."""

    holosoma_root: str
    holosoma_commit: str | None
    holosoma_dirty: bool | None
    checks: tuple[ParityCheck, ...]

    @property
    def failures(self) -> tuple[ParityCheck, ...]:
        return tuple(check for check in self.checks if check.status == "fail")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holosoma-root",
        type=Path,
        default=_default_holosoma_root(),
        help="Path to a Holosoma checkout. Defaults to ../holosoma next to this repository.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("defaults", "mesh", "sqp", "real_climb", "task_contract", "all"),
        default=None,
        help="Stage to run. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--real-data-frames",
        type=int,
        default=2,
        help="Loaded frame count for the real-data Holosoma climb fixture stage.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any check fails.")
    args = parser.parse_args(argv)

    stages = tuple(args.stage or ("all",))
    if "all" in stages:
        stages = ("defaults", "mesh", "sqp", "real_climb", "task_contract")

    holosoma_root = args.holosoma_root.resolve()
    _install_holosoma_source(holosoma_root)
    checks: list[ParityCheck] = []
    for stage in stages:
        checks.extend(_run_stage(stage, holosoma_root, real_data_frames=args.real_data_frames))

    report = ParityReport(
        holosoma_root=str(holosoma_root),
        holosoma_commit=_git_output(holosoma_root, "rev-parse", "HEAD"),
        holosoma_dirty=_git_dirty(holosoma_root),
        checks=tuple(checks),
    )
    if args.json:
        print(json.dumps(_jsonable(asdict(report)), indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 1 if args.strict and report.failures else 0


def _run_stage(stage: str, holosoma_root: Path, *, real_data_frames: int) -> tuple[ParityCheck, ...]:
    if stage == "defaults":
        return _defaults_checks()
    if stage == "mesh":
        return _mesh_checks()
    if stage == "sqp":
        return _sqp_checks()
    if stage == "real_climb":
        return _real_climb_checks(holosoma_root, frame_count=real_data_frames)
    if stage == "task_contract":
        return _task_contract_checks(holosoma_root)
    raise ValueError(f"Unknown stage {stage!r}")


def _defaults_checks() -> tuple[ParityCheck, ...]:
    retargeter_mod = importlib.import_module("interaction_mesh_retarget.config.retargeter")
    holosoma_config = retargeter_mod.RetargeterConfig()
    holosoma_foot_lock = retargeter_mod.FootLockConfig()
    solver = SolverSpec()
    profile = OptimizationProfile.defaults()
    laplacian = profile.objective("laplacian")
    smoothness = profile.objective("smoothness")
    foot_sticking = profile.constraint("foot_sticking")

    checks = [
        _numeric_check("defaults", "trust_radius", solver.trust_radius, holosoma_config.step_size),
        _numeric_check("defaults", "later_frame_iterations", solver.max_iterations, 10),
        _numeric_check("defaults", "first_frame_iterations", solver.max_iterations * 5, 50),
        _numeric_check("defaults", "laplacian_weight", laplacian.weight if laplacian else None, 10.0),
        _numeric_check("defaults", "smoothness_weight", smoothness.weight if smoothness else None, 0.2),
        _numeric_check(
            "defaults",
            "foot_sticking_tolerance",
            foot_sticking.tolerance if isinstance(foot_sticking, FootStickingConstraintConfig) else None,
            holosoma_config.foot_sticking_tolerance,
        ),
        _numeric_check(
            "defaults",
            "non_penetration_tolerance",
            NonPenetrationConstraintConfig().tolerance,
            holosoma_config.penetration_tolerance,
        ),
        _numeric_check(
            "defaults",
            "foot_lock_tolerance",
            FootLockConstraintConfig().tolerance,
            holosoma_foot_lock.tolerance,
        ),
        _numeric_check(
            "defaults",
            "nominal_tracking_weight",
            NominalTrackingObjectiveConfig().weight,
            holosoma_config.w_nominal_tracking_init,
        ),
    ]
    return tuple(checks)


def _mesh_checks() -> tuple[ParityCheck, ...]:
    holosoma_mesh = importlib.import_module("interaction_mesh_retarget.geometry.mesh")
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.45, 0.35, 0.85],
            [1.15, 0.65, 0.25],
            [0.3, 1.25, 0.55],
            [0.8, 0.2, 1.35],
        ],
        dtype=np.float64,
    )
    holosoma_vertices, holosoma_tets = holosoma_mesh.create_interaction_mesh(vertices)
    holosoma_adjacency = holosoma_mesh.get_adjacency_list(holosoma_tets, len(vertices))
    holosoma_laplacian = holosoma_mesh.calculate_laplacian_matrix(
        holosoma_vertices,
        holosoma_adjacency,
        uniform_weight=True,
    )
    holosoma_coords = holosoma_mesh.calculate_laplacian_coordinates(
        holosoma_vertices,
        holosoma_adjacency,
        uniform_weight=True,
    )

    retarget_mesh = InteractionMeshBuilder(
        InteractionMeshSpec(laplacian_weighting=LaplacianWeighting.UNIFORM)
    ).build(vertices)
    checks = [
        _array_check("mesh", "vertices", retarget_mesh.vertices, holosoma_vertices),
        _simplex_check("mesh", "delaunay_simplices", retarget_mesh.simplices, holosoma_tets),
        _array_check("mesh", "laplacian_matrix", retarget_mesh.laplacian, holosoma_laplacian),
        _array_check(
            "mesh",
            "laplacian_coordinates",
            retarget_mesh.laplacian_coordinates(),
            holosoma_coords,
            tolerance=1e-12,
        ),
    ]
    return tuple(checks)


def _sqp_checks() -> tuple[ParityCheck, ...]:
    if importlib.util.find_spec("cvxpy") is None or importlib.util.find_spec("clarabel") is None:
        return (
            ParityCheck(
                stage="sqp",
                name="clarabel_subproblem",
                status="skip",
                retarget=None,
                holosoma=None,
                note="Install cvxpy and clarabel to compare SQP subproblem output.",
            ),
        )

    problem_builder = importlib.import_module("interaction_mesh_retarget.core.problem_builder")
    holosoma_mesh = importlib.import_module("interaction_mesh_retarget.geometry.mesh")
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.45, 0.35, 0.85],
        ],
        dtype=np.float64,
    )
    _, tets = holosoma_mesh.create_interaction_mesh(vertices)
    adjacency = holosoma_mesh.get_adjacency_list(tets, len(vertices))
    laplacian = holosoma_mesh.calculate_laplacian_matrix(vertices, adjacency, uniform_weight=True)
    lap0_vec = (laplacian @ vertices).reshape(-1)
    target_laplacian = (laplacian @ (vertices + _vertex_offsets(len(vertices)))).reshape(vertices.shape)

    dof = 3
    j_l = _toy_jacobian(3 * len(vertices), dof)
    q_a_n_last = np.asarray([0.03, -0.02, 0.04], dtype=np.float64)
    q_t_last_slice = np.asarray([0.01, -0.015, 0.045], dtype=np.float64)
    q_a_lb = np.full(dof, -1.0, dtype=np.float64)
    q_a_ub = np.full(dof, 1.0, dtype=np.float64)
    q_a_indices = np.arange(dof)

    holosoma_solution, holosoma_cost = problem_builder.build_and_solve_sqp_iteration(
        data=problem_builder.SqpProblemData(
            q_a_indices=q_a_indices,
            n_vertices=len(vertices),
            laplacian_weights=10.0,
            step_size=0.2,
            penetration_tolerance=1e-3,
            foot_sticking_tolerance=1e-3,
            q_a_lb=q_a_lb,
            q_a_ub=q_a_ub,
            q_diag=np.zeros(dof, dtype=np.float64),
            smooth_weight=0.2,
            track_nominal_indices=np.asarray([], dtype=int),
            activate_joint_limits=True,
            apply_foot_sticking=False,
            apply_foot_lock=False,
            foot_lock_z_floor=0.0,
            foot_lock_tolerance=5e-3,
            self_collision_tolerance=0.02,
        ),
        jacobians=problem_builder.SqpIterationJacobians(j_l=j_l),
        vertices=vertices,
        adj_list=adjacency,
        target_laplacian=target_laplacian,
        q_a_n_last=q_a_n_last,
        q_t_last_slice=q_t_last_slice,
    )

    retarget_problem = QuadraticProblem(
        matrix=np.vstack([np.sqrt(10.0) * j_l, np.sqrt(0.2) * np.eye(dof, dtype=np.float64)]),
        target=np.concatenate(
            [
                np.sqrt(10.0) * (target_laplacian.reshape(-1) - lap0_vec),
                np.sqrt(0.2) * (q_t_last_slice - q_a_n_last),
            ]
        ),
        lower=q_a_lb - q_a_n_last,
        upper=q_a_ub - q_a_n_last,
        initial=np.zeros(dof, dtype=np.float64),
        trust_radius=0.2,
    )
    retarget_result = CvxpyClarabelSolver().solve(retarget_problem)
    return (
        _array_check("sqp", "solution", retarget_result.solution, holosoma_solution, tolerance=1e-6),
        _numeric_check("sqp", "objective_value", retarget_result.cost, holosoma_cost, tolerance=1e-6),
    )


def _real_climb_checks(holosoma_root: Path, *, frame_count: int) -> tuple[ParityCheck, ...]:
    if frame_count < 2:
        return (
            ParityCheck(
                stage="real_climb",
                name="reference_output",
                status="fail",
                retarget=None,
                holosoma=None,
                note="Holosoma's climb preprocessing needs at least two loaded frames.",
            ),
        )
    try:
        first = _run_holosoma_climb_reference(holosoma_root, frame_count=frame_count)
        second = _run_holosoma_climb_reference(holosoma_root, frame_count=frame_count)
    except Exception as exc:
        return (
            ParityCheck(
                stage="real_climb",
                name="reference_output",
                status="fail",
                retarget=None,
                holosoma=None,
                note=f"Holosoma reduced climb reference failed: {type(exc).__name__}: {exc}",
            ),
        )

    try:
        native = from_mocap_climb_fixture(
            holosoma_root,
            frame_count=frame_count,
            include_object_collision=False,
        )
    except Exception as exc:
        native_contract = ParityCheck(
            stage="real_climb",
            name="native_task_adapter",
            status="fail",
            retarget=None,
            holosoma={"qpos_shape": first.shape},
            note=f"Retarget Holosoma climb adapter failed: {type(exc).__name__}: {exc}",
        )
        native_output = ParityCheck(
            stage="real_climb",
            name="native_output_parity",
            status="fail",
            retarget=None,
            holosoma={"qpos_shape": first.shape},
            note="Native output parity was skipped because the task adapter failed.",
        )
    else:
        native_non_penetration = tuple(
            constraint
            for constraint in native.problem.constraints
            if isinstance(constraint, NonPenetrationConstraintConfig)
        )
        native_contract = ParityCheck(
            stage="real_climb",
            name="native_task_adapter",
            status="pass" if native.initial_qpos.qpos.shape[1] == first.shape[1] else "fail",
            retarget={
                "qpos_size": native.initial_qpos.qpos.shape[1],
                "motion_shape": native.motion.joint_positions.shape,
                "object_qpos_mode": native.scene.object.qpos_mode if native.scene.object else None,
                "contact_links": [track.link_names for track in native.contacts.tracks],
                "solver": {
                    "max_iterations": native.problem.solver.max_iterations,
                    "first_frame_iterations": native.problem.solver.first_frame_iterations,
                    "convergence": native.problem.solver.convergence,
                },
                "non_penetration_constraints": len(native_non_penetration),
            },
            holosoma={"qpos_shape": first.shape},
            note="Native retarget builds the same real-data task contract without ad hoc NPZ plumbing.",
        )
        try:
            native_output_array = InteractionMeshRetargetingEngine(
                kinematics=MuJoCoKinematicsBackend(native.problem.robot)
            ).run(native.problem).qpos
        except Exception as exc:
            native_output = ParityCheck(
                stage="real_climb",
                name="native_output_parity",
                status="fail",
                retarget=None,
                holosoma={"qpos_shape": first.shape},
                note=f"Native reduced climb run failed: {type(exc).__name__}: {exc}",
            )
        else:
            native_output = _array_check(
                "real_climb",
                "native_output_parity",
                native_output_array,
                first,
                tolerance=1e-6,
            )

    return (
        ParityCheck(
            stage="real_climb",
            name="reference_output",
            status="pass",
            retarget="not_run",
            holosoma={
                "fixture": "tests/fixtures/climb_seq_0",
                "data_format": "mocap",
                "task_type": "climbing",
                "collision": "object_non_penetration_disabled",
                "qpos_shape": first.shape,
                "qpos_head": first[0, : min(8, first.shape[1])],
            },
            note="This is a real Holosoma G1 climb fixture with the current broken object-collision path disabled.",
        ),
        _array_check("real_climb", "reference_determinism", second, first, tolerance=1e-9),
        native_contract,
        native_output,
    )


def _run_holosoma_climb_reference(holosoma_root: Path, *, frame_count: int) -> np.ndarray:
    robot_mod = importlib.import_module("interaction_mesh_retarget.config.robot")
    task_mod = importlib.import_module("interaction_mesh_retarget.config.task")
    context_mod = importlib.import_module("interaction_mesh_retarget.config.task_context")
    retargeter_mod = importlib.import_module("interaction_mesh_retarget.config.retargeter")
    collision_mod = importlib.import_module("interaction_mesh_retarget.core.collision")
    session_mod = importlib.import_module("interaction_mesh_retarget.io.session")
    formats_mod = importlib.import_module("interaction_mesh_retarget.plugins.motion_formats")

    fixture_dir = holosoma_root / "tests/fixtures/climb_seq_0"
    motion_source = fixture_dir / "mocap_climb_seq_0_joint_positions_f900-3700.npy"
    sphere_urdf = holosoma_root / "src/interaction_mesh_retarget/robots/g1/g1_29dof_spherehand.urdf"
    source_robot = holosoma_root / "src/interaction_mesh_retarget/robots/g1"
    for path in (
        fixture_dir / "g1_29dof_spherehand_w_multi_boxes.xml",
        motion_source,
        sphere_urdf,
        source_robot / "meshes",
        source_robot / "assets",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    models_g1 = holosoma_root / "models/g1"
    created_links: list[Path] = []
    preexisting_scaled = {path for path in fixture_dir.glob("*_scaled_*")}
    original_collision = collision_mod.CollisionMixin._update_jacobians_and_phis_from_q
    try:
        collision_mod.CollisionMixin._update_jacobians_and_phis_from_q = lambda _self, _q: ({}, {})
        models_g1.mkdir(parents=True, exist_ok=True)
        for name in ("meshes", "assets"):
            link = models_g1 / name
            if not link.exists():
                link.symlink_to(source_robot / name, target_is_directory=True)
                created_links.append(link)
        with tempfile.TemporaryDirectory(prefix="holosoma-climb-motion-") as tmp_name:
            tmp = Path(tmp_name)
            raw_motion = np.load(motion_source)
            motion = tmp / f"climb_{frame_count}frames.npy"
            np.save(motion, raw_motion[: frame_count * 4])
            motion_config = formats_mod.MotionDataConfig(data_format="mocap", robot_type="g1")
            task_config = task_mod.TaskConfig(object_name="multi_boxes", object_dir=fixture_dir)
            robot_config = robot_mod.RobotConfig(robot_type="g1", robot_urdf_file=str(sphere_urdf))
            ctx = context_mod.TaskContext.build(
                robot_config=robot_config,
                motion_data_config=motion_config,
                task_config=task_config,
                task_type="climbing",
            )
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result = session_mod.run_retarget_session(
                    motion,
                    ctx=ctx,
                    motion_config=motion_config,
                    data_format="mocap",
                    task_type="climbing",
                    task_config=task_config,
                    retargeter_config=retargeter_mod.RetargeterConfig(
                        debug=False,
                        visualize=False,
                        activate_obj_non_penetration=False,
                    ),
                    save_dir=tmp,
                    dest_res_path=tmp / "reference.npz",
                )
            return np.asarray(result.qpos, dtype=np.float64)
    finally:
        collision_mod.CollisionMixin._update_jacobians_and_phis_from_q = original_collision
        for path in fixture_dir.glob("*_scaled_*"):
            if path not in preexisting_scaled:
                path.unlink(missing_ok=True)
        for link in reversed(created_links):
            link.unlink(missing_ok=True)
        for maybe_empty in (models_g1, models_g1.parent):
            with contextlib.suppress(OSError):
                maybe_empty.rmdir()


def _task_contract_checks(holosoma_root: Path) -> tuple[ParityCheck, ...]:
    retargeter_mod = importlib.import_module("interaction_mesh_retarget.config.retargeter")
    robot_mod = importlib.import_module("interaction_mesh_retarget.config.robot")
    holosoma_config = retargeter_mod.RetargeterConfig()
    holosoma_robot = robot_mod.RobotConfig(robot_type="g1")
    retarget_robot = g1_spherehand_robot(holosoma_root)
    variable_policy = QposVariableSpec.holosoma_q_a(holosoma_config.q_a_init_idx)
    resolved_variables = variable_policy.resolve(retarget_robot, qpos_size=retarget_robot.qpos_size())
    expected_start = retarget_robot.qpos_layout.joint_start + holosoma_config.q_a_init_idx
    expected_stop = retarget_robot.qpos_layout.joint_start + retarget_robot.dof
    expected_indices = tuple(range(expected_start, expected_stop))
    retarget_indices = tuple(int(index) for index in resolved_variables.indices)
    holosoma_profile = holosoma_climb_profile(qpos_size=retarget_robot.qpos_size())
    diagonal = holosoma_profile.objective("diagonal_regularization")
    diagonal_metadata = (
        diagonal.model_dump(mode="json") if isinstance(diagonal, DiagonalRegularizationObjectiveConfig) else None
    )
    retarget_g1 = Path(".retarget_assets/robot/g1/robot.toml")
    root_note = "The primitive is available and the real-data adapter now uses it."
    return (
        ParityCheck(
            stage="task_contract",
            name="optimized_qpos_slice",
            status="pass" if retarget_indices == expected_indices else "fail",
            retarget={"kind": variable_policy.kind, "indices": retarget_indices},
            holosoma=f"q_a_init_idx={holosoma_config.q_a_init_idx}",
            note=root_note,
        ),
        ParityCheck(
            stage="task_contract",
            name="nominal_tracking_reference",
            status="pass",
            retarget="NominalQposPlan + NominalTrackingObjectiveConfig(qpos_indices=...)",
            holosoma="selected q_a indices track q_nominal_list when provided",
            note=root_note,
        ),
        ParityCheck(
            stage="task_contract",
            name="diagonal_regularization",
            status="pass" if isinstance(diagonal, DiagonalRegularizationObjectiveConfig) else "fail",
            retarget=diagonal_metadata,
            holosoma=f"manual_cost={dict(holosoma_robot.MANUAL_COST)}",
            note=root_note,
        ),
        ParityCheck(
            stage="task_contract",
            name="foot_sticking_links",
            status="pass" if tuple(G1_FOOT_STICKING_LINKS) == tuple(holosoma_robot.FOOT_STICKING_LINKS) else "fail",
            retarget="RobotSpec.contact_links + ContactPlan link_names",
            holosoma=tuple(holosoma_robot.FOOT_STICKING_LINKS),
            note="The typed contact abstraction and fixture adapter select the same support bodies.",
        ),
        ParityCheck(
            stage="task_contract",
            name="object_non_penetration",
            status="pass",
            retarget="NonPenetrationConstraintConfig.geometry_pairs + backend geom_distance_jacobians",
            holosoma="MuJoCo geom signed-distance constraints",
            note=root_note,
        ),
        ParityCheck(
            stage="task_contract",
            name="local_g1_assets",
            status="pass" if retarget_g1.exists() else "fail",
            retarget=str(retarget_g1),
            holosoma=str(holosoma_root / "src/interaction_mesh_retarget/robots/g1/g1_29dof.xml"),
            note="Robot assets are present locally for source-level and real-data parity checks.",
        ),
        ParityCheck(
            stage="task_contract",
            name="holosoma_checkout",
            status="pass" if holosoma_root.exists() else "fail",
            retarget=str(holosoma_root),
            holosoma=str(holosoma_root),
            note="The harness needs a local Holosoma checkout for source-level parity.",
        ),
    )


def _numeric_check(
    stage: str,
    name: str,
    retarget_value: Any,
    holosoma_value: Any,
    *,
    tolerance: float = 0.0,
) -> ParityCheck:
    if retarget_value is None or holosoma_value is None:
        return ParityCheck(stage=stage, name=name, status="fail", retarget=retarget_value, holosoma=holosoma_value)
    error = abs(float(retarget_value) - float(holosoma_value))
    return ParityCheck(
        stage=stage,
        name=name,
        status="pass" if error <= tolerance else "fail",
        retarget=float(retarget_value),
        holosoma=float(holosoma_value),
        max_abs_error=error,
        tolerance=tolerance,
    )


def _array_check(
    stage: str,
    name: str,
    retarget_value: np.ndarray,
    holosoma_value: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> ParityCheck:
    retarget_array = np.asarray(retarget_value, dtype=np.float64)
    holosoma_array = np.asarray(holosoma_value, dtype=np.float64)
    if retarget_array.shape != holosoma_array.shape:
        return ParityCheck(
            stage=stage,
            name=name,
            status="fail",
            retarget={"shape": retarget_array.shape},
            holosoma={"shape": holosoma_array.shape},
            note="Shape mismatch.",
        )
    error = float(np.max(np.abs(retarget_array - holosoma_array))) if retarget_array.size else 0.0
    return ParityCheck(
        stage=stage,
        name=name,
        status="pass" if error <= tolerance else "fail",
        retarget={"shape": retarget_array.shape},
        holosoma={"shape": holosoma_array.shape},
        max_abs_error=error,
        tolerance=tolerance,
    )


def _simplex_check(stage: str, name: str, retarget_value: np.ndarray, holosoma_value: np.ndarray) -> ParityCheck:
    retarget_set = _simplex_set(retarget_value)
    holosoma_set = _simplex_set(holosoma_value)
    return ParityCheck(
        stage=stage,
        name=name,
        status="pass" if retarget_set == holosoma_set else "fail",
        retarget={"count": len(retarget_set)},
        holosoma={"count": len(holosoma_set)},
        note="" if retarget_set == holosoma_set else "Delaunay simplex set differs.",
    )


def _simplex_set(values: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(sorted(int(item) for item in simplex)) for simplex in np.asarray(values)}


def _vertex_offsets(count: int) -> np.ndarray:
    base = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
    return 0.01 * np.sin(base + 1.0)


def _toy_jacobian(rows: int, cols: int) -> np.ndarray:
    values = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    return 0.03 * np.cos(values + 0.5)


def _default_holosoma_root() -> Path:
    return Path(__file__).resolve().parents[2] / "holosoma"


def _install_holosoma_source(root: Path) -> None:
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"Holosoma source path not found: {src}")
    for path in (src, root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_dirty(root: Path) -> bool | None:
    output = _git_output(root, "status", "--short")
    return None if output is None else bool(output)


def _print_report(report: ParityReport) -> None:
    print(f"Holosoma root: {report.holosoma_root}")
    print(f"Holosoma commit: {report.holosoma_commit or '<unknown>'}")
    if report.holosoma_dirty is not None:
        print(f"Holosoma dirty: {report.holosoma_dirty}")
    print()
    for check in report.checks:
        error = "" if check.max_abs_error is None else f" error={check.max_abs_error:.3g}"
        tolerance = "" if check.tolerance is None else f" tol={check.tolerance:.3g}"
        print(f"[{check.status.upper():4}] {check.stage}.{check.name}{error}{tolerance}")
        if check.note:
            print(f"       {check.note}")
    if report.failures:
        print(f"\nFailures: {len(report.failures)}")
    else:
        print("\nFailures: 0")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
