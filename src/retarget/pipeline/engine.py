"""Interaction-mesh SQP retargeting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from retarget.core.array import FloatArray
from retarget.core.enums import ConvergenceMode, QposVariableKind, QuaternionOrder, RunStatus, SolverKind
from retarget.core.pose import Pose
from retarget.core.protocols import KinematicsBackend
from retarget.kinematics.backends import SimpleKinematicsBackend
from retarget.mesh.interaction import InteractionMeshBuilder, InteractionMeshSpec, LaplacianWeighting
from retarget.motion.contact import ContactPlan
from retarget.motion.qpos import NominalQposFrame, NominalQposPlan
from retarget.motion.spec import MotionSequence
from retarget.motion.targets import LinkTargetPlan
from retarget.optimization.problem import ConstraintContribution, LinearConstraint, QuadraticProblem, TermContext
from retarget.optimization.registry import constraint_terms, objective_terms
from retarget.optimization.solvers import create_solver, resolve_solver_backend_name
from retarget.optimization.spec import SolverSpec
from retarget.optimization.variables import ResolvedQposVariables
from retarget.pipeline.compiled import (
    CompiledContactFrame,
    CompiledRetargetingProblem,
    CompiledTargetFrame,
    compile_problem,
)
from retarget.pipeline.problem import RetargetingProblem
from retarget.pipeline.progress import frame_progress
from retarget.results.spec import (
    MeshRunReport,
    RegistryKeyManifest,
    ResultObjectSpec,
    ResultObjectVisualPart,
    ResultPlaybackSpec,
    ResultRobotSpec,
    RetargetingResult,
    RetargetingRunReport,
    SolverRunReport,
    VocabularyManifest,
)


@dataclass(frozen=True)
class EngineOutput:
    """Raw output from the interaction-mesh engine.

    Attributes:
        qpos (FloatArray): Retargeted generalized coordinates, shape ``(frames, nq)``.
        costs (FloatArray): Per-frame subproblem cost after the last inner iteration.
        iterations (tuple[int, ...]): Inner solver iterations used per frame.
        solver_backend (SolverKind): Resolved typed registry key for the subproblem solver.
        solver_statuses (tuple[str, ...]): Backend status string per frame.
        mesh_spec (InteractionMeshSpec): Mesh topology used when building interaction graphs.
        mesh_source (str): ``"engine"`` when a custom builder was injected; otherwise ``"problem"``.
        robot_link_names (tuple[str, ...]): Link names for optional playback positions.
        robot_link_positions (FloatArray | None): World positions ``(frames, links, 3)`` when computed.
        warnings (tuple[str, ...]): Non-fatal issues encountered during the run.
    """

    qpos: FloatArray
    costs: FloatArray
    iterations: tuple[int, ...]
    solver_backend: SolverKind
    solver_statuses: tuple[str, ...]
    mesh_spec: InteractionMeshSpec
    mesh_source: str
    robot_link_names: tuple[str, ...] = ()
    robot_link_positions: FloatArray | None = None
    warnings: tuple[str, ...] = ()


class InteractionMeshRetargetingEngine:
    """Backend-agnostic interaction-mesh SQP retargeter.

    The engine builds one local quadratic problem per SQP iteration. Its
    decision variable is the actuated-joint increment for the current frame.
    Root and object poses are locked from the motion/scene specification.
    """

    def __init__(
        self,
        *,
        kinematics: KinematicsBackend | None = None,
        mesh_builder: InteractionMeshBuilder | None = None,
    ) -> None:
        self.kinematics: KinematicsBackend | None = kinematics
        self.mesh_builder: InteractionMeshBuilder | None = mesh_builder

    def run(self, problem: RetargetingProblem) -> EngineOutput:
        """Run interaction-mesh retargeting for all frames."""

        problem.validate_registry_references()
        mesh_builder = self.mesh_builder or InteractionMeshBuilder(problem.mesh)
        mesh_source = "engine" if self.mesh_builder is not None else "problem"
        motion = _scale_motion_to_robot(problem)
        scaled_contacts = _scaled_contact_plan(problem)
        scaled_targets = _scaled_target_plan(problem)
        compiled = compile_problem(
            problem.model_copy(
                update={
                    "motion": motion,
                    "contacts": scaled_contacts,
                    "targets": scaled_targets,
                }
            )
        )
        qpos = _initial_qpos(problem, motion)
        solver_backend = resolve_solver_backend_name(problem.solver)
        solver = create_solver(problem.solver)
        backend = self.kinematics or SimpleKinematicsBackend(compiled.robot)
        _validate_backend_problem(backend, compiled)
        backend_limits = backend.joint_limits()
        typed_backend_limits = {
            joint: backend_limits.get(name, problem.robot.joint_limits.get(joint, (-1e6, 1e6)))
            for joint, name in (
                (joint, compiled.joint_name(joint))
                for joint in problem.robot.joints
            )
        }
        variable_set = problem.variables.resolve(
            problem.robot,
            qpos_size=qpos.shape[1],
            joint_limits=typed_backend_limits,
        )
        typed_mapping = problem.link_mapping()
        human_joints = tuple(typed_mapping)
        robot_point_names = tuple(compiled.link_name(link) for link in typed_mapping.values())
        contact_plan = compiled.contacts
        target_plan = compiled.targets
        nominal_qpos_plan = _nominal_qpos_plan_for_problem(problem)
        lower_values, upper_values = _joint_limit_arrays(compiled.robot, backend)
        joint_lower = np.asarray(lower_values, dtype=np.float64)
        joint_upper = np.asarray(upper_values, dtype=np.float64)
        costs = np.zeros(motion.frame_count, dtype=np.float64)
        iterations: list[int] = []
        solver_statuses: list[str] = []
        warnings: list[str] = []
        scale_warning = _scale_to_robot_skipped_warning(problem)
        if scale_warning is not None:
            warnings.append(scale_warning)
        progress_label = problem.progress_description or problem.name

        with frame_progress(
            enabled=problem.show_progress,
            description=progress_label,
            total=motion.frame_count,
        ) as advance_frame:
            for frame_idx in range(motion.frame_count):
                if frame_idx > 0:
                    qpos[frame_idx, variable_set.indices] = qpos[frame_idx - 1, variable_set.indices]
                q_current = qpos[frame_idx].copy()
                reference_pose = _object_reference_pose(problem, frame_idx)
                environment_points = _environment_points(problem, reference_pose)
                human_points = _human_points(motion, human_joints, frame_idx)
                if reference_pose is not None:
                    human_points = reference_pose.inverse_transform_points(human_points)

                mesh = mesh_builder.build(human_points, environment_points)
                target_laplacian = mesh.laplacian_coordinates()
                frame_cost = np.inf
                frame_status = "not_run"
                used_iterations = 0
                iteration_count = _iteration_count(problem.solver, frame_idx)
                q_previous = qpos[max(frame_idx - 1, 0)]
                last_iteration_cost = np.inf

                for iteration_idx in range(iteration_count):
                    quadratic = self._build_subproblem(
                        problem=problem,
                        compiled=compiled,
                        backend=backend,
                        variable_set=variable_set,
                        q_current=q_current,
                        q_previous=q_previous,
                        robot_point_names=robot_point_names,
                        environment_points=environment_points,
                        adjacency=mesh.adjacency,
                        target_laplacian=target_laplacian,
                        laplacian_weighting=mesh.laplacian_weighting,
                        laplacian_epsilon=mesh.laplacian_epsilon,
                        reference_pose=reference_pose,
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                        contact_frame=contact_plan.frame(frame_idx) if contact_plan is not None else None,
                        target_frame=target_plan.frame(frame_idx) if target_plan is not None else None,
                        nominal_qpos_frame=(
                            nominal_qpos_plan.frame(frame_idx) if nominal_qpos_plan is not None else None
                        ),
                        frame_idx=frame_idx,
                    )
                    solved = solver.solve(quadratic)
                    q_current = variable_set.apply_delta(q_current, solved.solution)
                    frame_cost = solved.cost
                    frame_status = solved.status
                    used_iterations = iteration_idx + 1
                    if _should_stop_solver_iteration(
                        problem.solver,
                        solution=solved.solution,
                        cost=frame_cost,
                        previous_cost=last_iteration_cost,
                    ):
                        break
                    last_iteration_cost = frame_cost

                qpos[frame_idx] = q_current
                costs[frame_idx] = frame_cost
                iterations.append(used_iterations)
                solver_statuses.append(frame_status)
                advance_frame()

        robot_link_names, robot_link_positions, playback_warnings = _robot_playback_links(
            compiled=compiled,
            backend=backend,
            qpos=qpos,
            robot_point_names=robot_point_names,
        )
        warnings.extend(playback_warnings)

        return EngineOutput(
            qpos=qpos,
            costs=costs,
            iterations=tuple(iterations),
            solver_backend=solver_backend,
            solver_statuses=tuple(solver_statuses),
            mesh_spec=mesh_builder.spec,
            mesh_source=mesh_source,
            robot_link_names=robot_link_names,
            robot_link_positions=robot_link_positions,
            warnings=tuple(warnings),
        )

    def _build_subproblem(
        self,
        *,
        problem: RetargetingProblem,
        compiled: CompiledRetargetingProblem,
        backend: KinematicsBackend,
        variable_set: ResolvedQposVariables,
        q_current: FloatArray,
        q_previous: FloatArray,
        robot_point_names: tuple[str, ...],
        environment_points: FloatArray,
        adjacency: list[list[int]],
        target_laplacian: FloatArray,
        laplacian_weighting: LaplacianWeighting,
        laplacian_epsilon: float,
        reference_pose: Pose | None,
        joint_lower: FloatArray,
        joint_upper: FloatArray,
        contact_frame: CompiledContactFrame | None,
        target_frame: CompiledTargetFrame | None,
        nominal_qpos_frame: NominalQposFrame | None,
        frame_idx: int,
    ) -> QuadraticProblem:
        robot_points_world, robot_jacobians_world = _point_jacobians_for_variables(
            backend,
            q_current,
            robot_point_names,
            variable_set,
        )
        robot_points = robot_points_world
        robot_jacobians = robot_jacobians_world
        if reference_pose is not None:
            rotation_inv = reference_pose.rotation().as_matrix().T
            robot_points = reference_pose.inverse_transform_points(robot_points_world)
            robot_jacobians = np.einsum("ab,pbc->pac", rotation_inv, robot_jacobians_world)

        joint_slice = problem.robot.qpos_layout.joint_slice(problem.robot.dof)
        current_joints = q_current[joint_slice]
        context = TermContext(
            problem=problem,
            compiled=compiled,
            backend=backend,
            q_current=q_current,
            q_previous=q_previous,
            frame_idx=frame_idx,
            contact_frame=contact_frame,
            target_frame=target_frame,
            robot_point_names=robot_point_names,
            robot_points=robot_points,
            robot_jacobians=robot_jacobians,
            environment_points=environment_points,
            adjacency=tuple(tuple(neighbors) for neighbors in adjacency),
            target_laplacian=target_laplacian,
            laplacian_weighting=laplacian_weighting,
            laplacian_epsilon=laplacian_epsilon,
            reference_pose=reference_pose,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            current_joints=current_joints,
            variable_indices=variable_set.indices,
            current_variables=variable_set.values(q_current),
            variable_lower=variable_set.lower,
            variable_upper=variable_set.upper,
            nominal_qpos_frame=nominal_qpos_frame,
        )
        matrices: list[FloatArray] = []
        targets: list[FloatArray] = []
        for objective in problem.objectives:
            if objective.weight <= 0:
                continue
            term = objective_terms.get(objective.kind)
            for contribution in term.build(context, objective):
                _append_weighted_term(
                    matrices,
                    targets,
                    contribution.matrix,
                    contribution.target,
                    objective.weight,
                )

        matrix = np.vstack(matrices) if matrices else np.zeros((1, context.dof), dtype=np.float64)
        target = np.concatenate(targets) if targets else np.zeros(1, dtype=np.float64)
        constraint_contribution = _build_constraint_contribution(context)
        return QuadraticProblem(
            matrix=matrix,
            target=target,
            lower=constraint_contribution.lower,
            upper=constraint_contribution.upper,
            initial=np.zeros(context.dof, dtype=np.float64),
            trust_radius=constraint_contribution.trust_radius,
            linear_constraints=constraint_contribution.linear_constraints,
        )


def result_from_engine_output(
    problem: RetargetingProblem,
    output: EngineOutput,
    *,
    runtime_s: float,
) -> RetargetingResult:
    """Build a public result object from engine output."""

    return RetargetingResult(
        name=problem.name,
        status=RunStatus.SUCCESS,
        qpos=output.qpos,
        fps=problem.fps,
        cost=output.costs,
        human_joints=_scale_motion_to_robot(problem).joint_positions,
        human_vocabulary=VocabularyManifest.from_members(tuple(problem.motion.joints)),
        robot_link_positions=output.robot_link_positions,
        run=_result_run_report(problem, output, runtime_s=runtime_s),
        playback=_result_playback_spec(problem, output),
        warnings=output.warnings,
        provenance=_result_provenance(problem),
    )


def _iteration_count(solver: SolverSpec, frame_idx: int) -> int:
    if frame_idx == 0:
        return int(solver.first_frame_iterations or (solver.max_iterations * 5))
    return int(solver.max_iterations)


def _should_stop_solver_iteration(
    solver: SolverSpec,
    *,
    solution: FloatArray,
    cost: float,
    previous_cost: float,
) -> bool:
    if solver.convergence == ConvergenceMode.NONE:
        return False
    if solver.convergence == ConvergenceMode.COST_PLATEAU:
        return bool(np.isclose(cost, previous_cost, rtol=solver.cost_rtol, atol=solver.cost_atol))
    return float(np.linalg.norm(solution)) <= solver.tolerance


def _scale_motion_to_robot(problem: RetargetingProblem) -> MotionSequence:
    factor = _motion_scale_factor(problem)
    if factor is None:
        return problem.motion
    return problem.motion.scaled(factor)


def _scaled_contact_plan(problem: RetargetingProblem) -> ContactPlan | None:
    if problem.contacts is None:
        return None
    factor = _motion_scale_factor(problem)
    if factor is None:
        return problem.contacts
    return problem.contacts.scaled(factor)


def _scaled_target_plan(problem: RetargetingProblem) -> LinkTargetPlan | None:
    if problem.targets is None:
        return None
    factor = _motion_scale_factor(problem)
    if factor is None:
        return problem.targets
    return problem.targets.scaled(factor)


def _nominal_qpos_plan_for_problem(problem: RetargetingProblem) -> NominalQposPlan | None:
    return problem.nominal_qpos


def _source_height_m(problem: RetargetingProblem) -> float | None:
    """Return a positive source actor height in meters, if one is available."""

    source_height = problem.motion.source_height_m
    if source_height is None:
        return None
    value = float(source_height)
    if value <= 0:
        return None
    return value


def _motion_scale_factor(problem: RetargetingProblem) -> float | None:
    if not problem.scale_to_robot:
        return None
    source_height = _source_height_m(problem)
    if source_height is None:
        return None
    return problem.robot.height_m / source_height


def _scale_to_robot_skipped_warning(problem: RetargetingProblem) -> str | None:
    if not problem.scale_to_robot or _source_height_m(problem) is not None:
        return None
    return (
        "scale_to_robot is enabled but motion was not scaled: no positive source height. "
        "Set motion.source_height_m or disable scale_to_robot."
    )


def _result_run_report(
    problem: RetargetingProblem,
    output: EngineOutput,
    *,
    runtime_s: float,
) -> RetargetingRunReport:
    return RetargetingRunReport(
        algorithm="interaction_mesh_sqp",
        task_kind=problem.task_kind,
        robot_name=problem.robot.name,
        motion_name=problem.motion.name,
        runtime_s=float(runtime_s),
        input_fps=problem.motion.fps,
        output_fps=problem.fps,
        requested_output_fps=problem.output_fps,
        scale_to_robot=problem.scale_to_robot,
        motion_scale_factor=_motion_scale_factor(problem),
        solver=SolverRunReport(
            requested_backend=RegistryKeyManifest.from_member(problem.solver.backend),
            resolved_backend=RegistryKeyManifest.from_member(output.solver_backend),
            frame_statuses=output.solver_statuses,
            iterations=output.iterations,
        ),
        mesh=MeshRunReport(
            spec=output.mesh_spec,
            custom_builder=output.mesh_source == "engine",
        ),
    )


def _result_playback_spec(
    problem: RetargetingProblem,
    output: EngineOutput,
) -> ResultPlaybackSpec:
    robot = ResultRobotSpec(
        name=problem.robot.name,
        link_names=output.robot_link_names,
        joint_names=tuple(joint.value for joint in problem.robot.joints),
        joint_start=problem.robot.qpos_layout.joint_start,
        joint_vocabulary=cast(
            VocabularyManifest,
            VocabularyManifest.from_members(tuple(problem.robot.joints)),
        ),
        link_vocabulary=cast(
            VocabularyManifest,
            VocabularyManifest.from_members(tuple(problem.robot.links)),
        ),
        urdf_path=problem.robot.urdf_path,
        mujoco_xml_path=problem.robot.mujoco_xml_path,
    )
    object_spec = problem.scene.object
    if object_spec is None:
        return ResultPlaybackSpec(robot=robot)
    object_slice = None
    if problem.scene.has_dynamic_object():
        resolved = problem.robot.qpos_layout.object_slice(problem.robot.dof)
        object_slice = (cast(int, resolved.start), cast(int, resolved.stop))
    return ResultPlaybackSpec(
        robot=robot,
        object=ResultObjectSpec(
            name=object_spec.name,
            sample_points=object_spec.scaled_sample_points(default=_default_object_points()),
            sample_space=object_spec.sample_space,
            qpos_mode=object_spec.qpos_mode,
            qpos_slice=object_slice,
            mesh_path=object_spec.mesh_path,
            asset_scale=object_spec.asset_scale,
            visual_parts=tuple(
                ResultObjectVisualPart(
                    name=part.name,
                    mesh_path=part.mesh_path,
                    asset_scale=part.asset_scale or object_spec.asset_scale,
                    rgba=part.rgba,
                )
                for part in object_spec.visual_parts
            ),
        ),
    )


def _result_provenance(problem: RetargetingProblem) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "problem": dict(problem.provenance),
        "motion": dict(problem.motion.provenance),
        "robot": dict(problem.robot.provenance),
        "scene": dict(problem.scene.provenance),
    }
    if problem.scene.object is not None:
        provenance["object"] = dict(problem.scene.object.provenance)
    if problem.scene.terrain is not None:
        provenance["terrain"] = dict(problem.scene.terrain.provenance)
    if problem.contacts is not None:
        provenance["contacts"] = dict(problem.contacts.provenance)
    if problem.targets is not None:
        provenance["targets"] = dict(problem.targets.provenance)
    return provenance


def _robot_playback_links(
    *,
    compiled: CompiledRetargetingProblem,
    backend: KinematicsBackend,
    qpos: FloatArray,
    robot_point_names: tuple[str, ...],
) -> tuple[tuple[str, ...], FloatArray | None, tuple[str, ...]]:
    link_names = _playback_link_names(compiled, robot_point_names)
    if not link_names:
        return (), None, ()
    try:
        positions = np.asarray([backend.link_positions(frame, link_names) for frame in qpos], dtype=np.float64)
    except (KeyError, RuntimeError, ValueError) as exc:
        return (), None, (f"Could not compute robot playback link positions: {exc}",)
    return link_names, positions, ()


def _playback_link_names(
    compiled: CompiledRetargetingProblem,
    robot_point_names: tuple[str, ...],
) -> tuple[str, ...]:
    if compiled.robot.link_names:
        return compiled.robot.link_names
    if compiled.robot.contact_links:
        return compiled.robot.contact_links
    return tuple(dict.fromkeys(robot_point_names))


def _initial_qpos(problem: RetargetingProblem, motion: MotionSequence) -> FloatArray:
    qpos = np.zeros((motion.frame_count, problem.robot.qpos_size(has_object=problem.scene.has_dynamic_object())))
    if problem.initial_qpos is not None:
        qpos[:] = problem.initial_qpos.qpos
        return qpos
    root_position = slice(*problem.robot.qpos_layout.root_position)
    root_quaternion = slice(*problem.robot.qpos_layout.root_quaternion)
    if motion.root_poses is not None:
        qpos[:, root_position] = motion.root_poses.positions
        qpos[:, root_quaternion] = motion.root_poses.quaternions(QuaternionOrder.WXYZ)
    else:
        qpos[:, root_position] = motion.joint(motion.root_joint)
        qpos[:, root_quaternion] = np.array([1.0, 0.0, 0.0, 0.0])
    if problem.scene.has_dynamic_object() and problem.scene.object is not None and problem.scene.object.trajectory:
        object_slice = problem.robot.qpos_layout.object_slice(problem.robot.dof)
        poses = problem.scene.object.trajectory.poses
        qpos[:, object_slice.start : object_slice.start + 3] = poses.positions
        qpos[:, object_slice.start + 3 : object_slice.stop] = poses.quaternions(QuaternionOrder.WXYZ)
    return qpos


def _object_reference_pose(problem: RetargetingProblem, frame_idx: int) -> Pose | None:
    if problem.scene.object is None or problem.scene.object.trajectory is None:
        return None
    return problem.scene.object.trajectory.poses.poses[frame_idx]


def _environment_points(problem: RetargetingProblem, reference_pose: Pose | None) -> FloatArray:
    if problem.scene.object is not None:
        points = problem.scene.object.scaled_sample_points(default=_default_object_points())
        return np.asarray(points, dtype=np.float64)
    if problem.scene.terrain is not None and problem.scene.terrain.sample_points is not None:
        return problem.scene.terrain.sample_points
    return problem.scene.ground_points()


def _default_object_points() -> FloatArray:
    extent = 0.2
    return np.asarray(
        [
            [x, y, z]
            for x in (-extent, extent)
            for y in (-extent, extent)
            for z in (-extent, extent)
        ],
        dtype=np.float64,
    )


def _joint_limit_arrays(robot: Any, backend: KinematicsBackend) -> tuple[list[float], list[float]]:
    limits = backend.joint_limits()
    lower: list[float] = []
    upper: list[float] = []
    for name in robot.joint_names:
        lo, hi = limits.get(name, robot.joint_limits.get(name, (-1e6, 1e6)))
        lower.append(float(lo))
        upper.append(float(hi))
    return lower, upper


def _human_points(motion: MotionSequence, human_joints: tuple[Any, ...], frame_idx: int) -> FloatArray:
    indices = [motion.joint_index(joint) for joint in human_joints]
    return np.asarray(motion.joint_positions[frame_idx, indices, :], dtype=np.float64)


def _point_jacobians_for_variables(
    backend: KinematicsBackend,
    qpos: FloatArray,
    point_names: tuple[str, ...],
    variable_set: ResolvedQposVariables,
) -> tuple[FloatArray, FloatArray]:
    if not point_names:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3, variable_set.size), dtype=np.float64),
        )
    method = getattr(backend, "point_jacobians_for_qpos_indices", None)
    if callable(method):
        result = method(qpos, point_names, variable_set.indices)
        return cast(tuple[FloatArray, FloatArray], result)
    if variable_set.spec.kind == QposVariableKind.ACTUATED:
        return backend.point_jacobians(qpos, point_names)
    raise TypeError(
        f"{type(backend).__name__} must implement point_jacobians_for_qpos_indices "
        "when RetargetingProblem.variables is not the default actuated policy"
    )


def _validate_backend_problem(
    backend: KinematicsBackend,
    compiled: CompiledRetargetingProblem,
) -> None:
    validator = getattr(backend, "validate_compiled_problem", None)
    if callable(validator):
        validator(compiled)


def _append_weighted_term(
    matrices: list[FloatArray],
    targets: list[FloatArray],
    matrix: FloatArray,
    target: FloatArray,
    weight: float,
) -> None:
    if weight <= 0:
        return
    scale = float(np.sqrt(weight))
    matrices.append(np.asarray(matrix, dtype=np.float64) * scale)
    targets.append(np.asarray(target, dtype=np.float64) * scale)


def _build_constraint_contribution(context: TermContext) -> ConstraintContribution:
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    trust_radius: float | None = None
    linear_constraints: list[LinearConstraint] = []
    for constraint in context.problem.constraints:
        if not constraint.enabled:
            continue
        contribution = constraint_terms.get(constraint.kind).build(context, constraint)
        lower = _merge_lower(lower, contribution.lower)
        upper = _merge_upper(upper, contribution.upper)
        if contribution.trust_radius is not None:
            trust_radius = (
                contribution.trust_radius
                if trust_radius is None
                else min(trust_radius, contribution.trust_radius)
            )
        linear_constraints.extend(contribution.linear_constraints)
    return ConstraintContribution(
        lower=lower,
        upper=upper,
        trust_radius=trust_radius,
        linear_constraints=tuple(linear_constraints),
    )


def _merge_lower(current: FloatArray | None, candidate: FloatArray | None) -> FloatArray | None:
    if candidate is None:
        return current
    if current is None:
        return np.asarray(candidate, dtype=np.float64).copy()
    if current.shape != candidate.shape:
        raise ValueError("constraint lower bounds must have matching shapes")
    return np.maximum(current, candidate)


def _merge_upper(current: FloatArray | None, candidate: FloatArray | None) -> FloatArray | None:
    if candidate is None:
        return current
    if current is None:
        return np.asarray(candidate, dtype=np.float64).copy()
    if current.shape != candidate.shape:
        raise ValueError("constraint upper bounds must have matching shapes")
    return np.minimum(current, candidate)
