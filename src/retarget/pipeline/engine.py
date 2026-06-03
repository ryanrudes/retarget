"""Interaction-mesh SQP retargeting engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core.array import FloatArray
from retarget.core.enums import FrameConvention, QuaternionOrder, RunStatus
from retarget.core.pose import Pose
from retarget.core.protocols import KinematicsBackend
from retarget.kinematics.backends import SimpleKinematicsBackend
from retarget.mesh.interaction import InteractionMeshBuilder, InteractionMeshSpec
from retarget.motion.contact import infer_contact_by_velocity
from retarget.motion.spec import MotionSequence
from retarget.optimization.problem import ConstraintContribution, LinearConstraint, QuadraticProblem, TermContext
from retarget.optimization.registry import constraint_terms, objective_terms
from retarget.optimization.solvers import create_solver, resolve_solver_backend_name
from retarget.pipeline.problem import RetargetingProblem
from retarget.results.spec import RetargetingResult
from retarget.robots.spec import RobotSpec


@dataclass(frozen=True)
class EngineOutput:
    """Raw output from the interaction-mesh engine."""

    qpos: FloatArray
    costs: FloatArray
    iterations: tuple[int, ...]
    solver_backend: str
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
        self.kinematics = kinematics
        self.mesh_builder = mesh_builder

    def run(self, problem: RetargetingProblem) -> EngineOutput:
        """Run interaction-mesh retargeting for all frames."""

        problem.validate_registry_references()
        mesh_builder = self.mesh_builder or InteractionMeshBuilder(problem.mesh)
        mesh_source = "engine" if self.mesh_builder is not None else "problem"
        motion = _scale_motion_to_robot(problem)
        qpos = _initial_qpos(problem, motion)
        solver_backend = resolve_solver_backend_name(problem.solver)
        solver = create_solver(problem.solver)
        backend = self.kinematics or SimpleKinematicsBackend(problem.robot)
        mapping = problem.resolved_link_mapping()
        human_names = tuple(mapping)
        robot_point_names = tuple(mapping.values())
        contacts = infer_contact_by_velocity(
            motion,
            problem.motion_format,
            velocity_threshold=_constraint_parameter(problem, "foot_contact", "velocity_threshold", 0.01),
        )
        joint_slice = problem.robot.qpos_layout.joint_slice(problem.robot.dof)
        lower_values, upper_values = _joint_limit_arrays(problem.robot, backend)
        joint_lower = np.asarray(lower_values, dtype=np.float64)
        joint_upper = np.asarray(upper_values, dtype=np.float64)
        costs = np.zeros(motion.frame_count, dtype=np.float64)
        iterations: list[int] = []
        solver_statuses: list[str] = []
        warnings: list[str] = []

        for frame_idx in range(motion.frame_count):
            if frame_idx > 0:
                qpos[frame_idx, joint_slice] = qpos[frame_idx - 1, joint_slice]
            q_current = qpos[frame_idx].copy()
            reference_pose = _object_reference_pose(problem, frame_idx)
            environment_points = _environment_points(problem, reference_pose)
            human_points = _human_points(motion, human_names, frame_idx)
            if reference_pose is not None:
                human_points = reference_pose.inverse_transform_points(human_points)

            mesh = mesh_builder.build(human_points, environment_points)
            target_laplacian = mesh.laplacian_coordinates()
            frame_cost = np.inf
            frame_status = "not_run"
            used_iterations = 0
            iteration_count = problem.solver.max_iterations * (5 if frame_idx == 0 else 1)
            q_previous = qpos[max(frame_idx - 1, 0)]

            for iteration_idx in range(iteration_count):
                q_joint_current = q_current[joint_slice].copy()
                quadratic = self._build_subproblem(
                    problem=problem,
                    backend=backend,
                    q_current=q_current,
                    q_previous=q_previous,
                    robot_point_names=robot_point_names,
                    environment_points=environment_points,
                    adjacency=mesh.adjacency,
                    target_laplacian=target_laplacian,
                    reference_pose=reference_pose,
                    joint_lower=joint_lower,
                    joint_upper=joint_upper,
                    frame_contacts=contacts[frame_idx] if frame_idx < len(contacts) else {},
                    frame_idx=frame_idx,
                )
                solved = solver.solve(quadratic)
                q_current[joint_slice] = q_joint_current + solved.solution
                frame_cost = solved.cost
                frame_status = solved.status
                used_iterations = iteration_idx + 1
                if float(np.linalg.norm(solved.solution)) <= problem.solver.tolerance:
                    break

            qpos[frame_idx] = q_current
            costs[frame_idx] = frame_cost
            iterations.append(used_iterations)
            solver_statuses.append(frame_status)

        robot_link_names, robot_link_positions, playback_warnings = _robot_playback_links(
            problem=problem,
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
        backend: KinematicsBackend,
        q_current: FloatArray,
        q_previous: FloatArray,
        robot_point_names: tuple[str, ...],
        environment_points: FloatArray,
        adjacency: list[list[int]],
        target_laplacian: FloatArray,
        reference_pose: Pose | None,
        joint_lower: FloatArray,
        joint_upper: FloatArray,
        frame_contacts: dict[str, bool],
        frame_idx: int,
    ) -> QuadraticProblem:
        robot_points_world, robot_jacobians_world = backend.point_jacobians(q_current, robot_point_names)
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
            backend=backend,
            q_current=q_current,
            q_previous=q_previous,
            frame_idx=frame_idx,
            frame_contacts=frame_contacts,
            robot_point_names=robot_point_names,
            robot_points=robot_points,
            robot_jacobians=robot_jacobians,
            environment_points=environment_points,
            adjacency=tuple(tuple(neighbors) for neighbors in adjacency),
            target_laplacian=target_laplacian,
            reference_pose=reference_pose,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            current_joints=current_joints,
        )
        matrices: list[FloatArray] = []
        targets: list[FloatArray] = []
        for objective in problem.objectives:
            if objective.weight <= 0:
                continue
            term = objective_terms.get(objective.name)
            for contribution in term.build(context, objective):
                _append_weighted_term(
                    matrices,
                    targets,
                    contribution.matrix,
                    contribution.target,
                    objective.weight,
                )

        matrix = np.vstack(matrices) if matrices else np.zeros((1, problem.robot.dof), dtype=np.float64)
        target = np.concatenate(targets) if targets else np.zeros(1, dtype=np.float64)
        constraint_contribution = _build_constraint_contribution(context)
        return QuadraticProblem(
            matrix=matrix,
            target=target,
            lower=constraint_contribution.lower,
            upper=constraint_contribution.upper,
            initial=np.zeros(problem.robot.dof, dtype=np.float64),
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
        robot_link_positions=output.robot_link_positions,
        warnings=output.warnings,
        metadata=_result_metadata(problem, output, runtime_s=runtime_s),
    )


def _scale_motion_to_robot(problem: RetargetingProblem) -> MotionSequence:
    factor = _motion_scale_factor(problem)
    if factor is None:
        return problem.motion
    return problem.motion.scaled(factor)


def _motion_scale_factor(problem: RetargetingProblem) -> float | None:
    if not problem.scale_to_robot:
        return None
    source_height = problem.motion.metadata.get("height_m")
    if source_height is None and problem.motion_format is not None:
        source_height = problem.motion_format.default_height_m
    if source_height is None or float(source_height) <= 0:
        return None
    return problem.robot.height_m / float(source_height)


def _result_metadata(problem: RetargetingProblem, output: EngineOutput, *, runtime_s: float) -> dict[str, Any]:
    provenance = _run_provenance(problem, output)
    return {
        "robot": problem.robot.name,
        "motion": problem.motion.name,
        "motion_format": problem.motion_format.name if problem.motion_format is not None else None,
        "task_kind": problem.task_kind.value,
        "runtime_s": runtime_s,
        "solver": problem.solver.backend_name,
        "resolved_solver": output.solver_backend,
        "solver_statuses": output.solver_statuses,
        "mesh": _mesh_metadata(output),
        "playback": _playback_metadata(problem),
        "algorithm": "interaction_mesh_sqp",
        "joint_mapping": problem.resolved_joint_mapping(),
        "link_mapping": problem.resolved_link_mapping(),
        "iterations": output.iterations,
        "provenance": provenance,
    }


def _run_provenance(problem: RetargetingProblem, output: EngineOutput) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": problem.name,
        "task_kind": problem.task_kind.value,
        "input_fps": problem.motion.fps,
        "output_fps": problem.fps,
        "requested_output_fps": problem.output_fps,
        "scale_to_robot": problem.scale_to_robot,
        "motion_scale_factor": _motion_scale_factor(problem),
        "motion": _motion_provenance(problem),
        "robot": _robot_provenance(problem),
        "scene": _scene_provenance(problem),
        "mesh": _mesh_metadata(output),
        "solver": {
            **_jsonable(problem.solver.model_dump(mode="json")),
            "actual_backend": output.solver_backend,
            "frame_statuses": list(output.solver_statuses),
        },
        "resolved_solver": output.solver_backend,
        "solver_statuses": list(output.solver_statuses),
        "objectives": [_jsonable(objective.model_dump(mode="json")) for objective in problem.objectives],
        "constraints": [_jsonable(constraint.model_dump(mode="json")) for constraint in problem.constraints],
        "result": {
            "frame_count": int(output.qpos.shape[0]),
            "qpos_dimension": int(output.qpos.shape[1]),
            "has_cost": output.costs.size > 0,
            "warning_count": len(output.warnings),
        },
        "problem_metadata": _jsonable(problem.metadata),
    }


def _mesh_metadata(output: EngineOutput) -> dict[str, Any]:
    return {
        **_jsonable(output.mesh_spec.model_dump(mode="json")),
        "source": output.mesh_source,
    }


def _playback_metadata(problem: RetargetingProblem) -> dict[str, Any]:
    object_info = _object_playback_metadata(problem)
    playback: dict[str, Any] = {"robot": _robot_playback_metadata(problem)}
    if object_info is not None:
        playback["object"] = object_info
    return playback


def _robot_playback_metadata(problem: RetargetingProblem) -> dict[str, Any]:
    names = _playback_link_names(problem, tuple(dict.fromkeys(problem.resolved_link_mapping().values())))
    return {
        "name": problem.robot.name,
        "link_names": list(names),
        "edges": [list(edge) for edge in _humanoid_link_edges(names)],
        "joint_names": list(problem.robot.joint_names),
        "joint_start": problem.robot.qpos_layout.joint_start,
        "urdf_path": str(problem.robot.urdf_path) if problem.robot.urdf_path is not None else None,
        "mujoco_xml_path": str(problem.robot.mujoco_xml_path) if problem.robot.mujoco_xml_path is not None else None,
    }


def _object_playback_metadata(problem: RetargetingProblem) -> dict[str, Any] | None:
    object_spec = problem.scene.object
    if object_spec is None:
        return None

    sample_points = object_spec.sample_points if object_spec.sample_points is not None else _default_object_points()
    metadata: dict[str, Any] = {
        "name": object_spec.name,
        "mesh_path": str(object_spec.mesh_path) if object_spec.mesh_path is not None else None,
        "sample_points": _jsonable(sample_points),
        "sample_points_space": "object" if problem.scene.has_dynamic_object() else "world",
        "frame_convention": FrameConvention.Z_UP_RIGHT_HANDED.value,
        "quaternion_order": QuaternionOrder.WXYZ.value,
    }
    if problem.scene.has_dynamic_object():
        object_slice = problem.robot.qpos_layout.object_slice(problem.robot.dof)
        metadata["qpos_slice"] = [object_slice.start, object_slice.stop]
    return metadata


def _robot_playback_links(
    *,
    problem: RetargetingProblem,
    backend: KinematicsBackend,
    qpos: FloatArray,
    robot_point_names: tuple[str, ...],
) -> tuple[tuple[str, ...], FloatArray | None, tuple[str, ...]]:
    link_names = _playback_link_names(problem, robot_point_names)
    if not link_names:
        return (), None, ()
    try:
        positions = np.asarray([backend.link_positions(frame, link_names) for frame in qpos], dtype=np.float64)
    except (KeyError, RuntimeError, ValueError) as exc:
        return (), None, (f"Could not compute robot playback link positions: {exc}",)
    return link_names, positions, ()


def _playback_link_names(problem: RetargetingProblem, robot_point_names: tuple[str, ...]) -> tuple[str, ...]:
    if problem.robot.link_names:
        return tuple(problem.robot.link_names)
    if problem.robot.contact_links:
        return tuple(problem.robot.contact_links)
    return tuple(dict.fromkeys(robot_point_names))


def _humanoid_link_edges(link_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    names = set(link_names)

    def first_present(*candidates: str) -> str | None:
        return next((name for name in candidates if name in names), None)

    pelvis = first_present("pelvis", "torso", "root")
    torso = first_present("torso", "spine", "chest")
    head = first_present("head")
    left_foot = first_present("left_foot", "left_toe")
    right_foot = first_present("right_foot", "right_toe")
    left_hand = first_present("left_hand", "left_wrist")
    right_hand = first_present("right_hand", "right_wrist")

    candidate_edges = (
        (pelvis, torso),
        (torso or pelvis, head),
        (pelvis, left_foot),
        (pelvis, right_foot),
        (torso or pelvis, left_hand),
        (torso or pelvis, right_hand),
    )
    edges: list[tuple[str, str]] = []
    for first, second in candidate_edges:
        if first is None or second is None or first == second:
            continue
        edge = (first, second)
        if edge not in edges:
            edges.append(edge)
    if not edges and len(link_names) > 1:
        anchor = link_names[0]
        edges.extend((anchor, name) for name in link_names[1:])
    return tuple(edges)


def _motion_provenance(problem: RetargetingProblem) -> dict[str, Any]:
    return {
        "name": problem.motion.name,
        "format": problem.motion_format.name if problem.motion_format is not None else None,
        "frame_count": problem.motion.frame_count,
        "joint_count": problem.motion.joint_count,
        "fps": problem.motion.fps,
        "frame": problem.motion.frame.value,
        "has_root_poses": problem.motion.root_poses is not None,
        "has_contacts": bool(problem.motion.contacts),
        "metadata": _jsonable(problem.motion.metadata),
    }


def _robot_provenance(problem: RetargetingProblem) -> dict[str, Any]:
    return {
        "name": problem.robot.name,
        "dof": problem.robot.dof,
        "height_m": problem.robot.height_m,
        "qpos_size": problem.robot.qpos_size(has_object=problem.scene.has_dynamic_object()),
        "contact_links": list(problem.robot.contact_links),
    }


def _scene_provenance(problem: RetargetingProblem) -> dict[str, Any]:
    object_info = None
    if problem.scene.object is not None:
        trajectory = problem.scene.object.trajectory
        object_info = {
            "name": problem.scene.object.name,
            "mesh_path": str(problem.scene.object.mesh_path) if problem.scene.object.mesh_path is not None else None,
            "urdf_path": str(problem.scene.object.urdf_path) if problem.scene.object.urdf_path is not None else None,
            "sample_point_count": (
                int(problem.scene.object.sample_points.shape[0])
                if problem.scene.object.sample_points is not None
                else 0
            ),
            "has_trajectory": trajectory is not None,
            "trajectory_frame_count": trajectory.poses.frame_count if trajectory is not None else None,
            "metadata": _jsonable(problem.scene.object.metadata),
        }

    terrain_info = None
    if problem.scene.terrain is not None:
        terrain_info = {
            "name": problem.scene.terrain.name,
            "mesh_path": str(problem.scene.terrain.mesh_path) if problem.scene.terrain.mesh_path is not None else None,
            "sample_point_count": (
                int(problem.scene.terrain.sample_points.shape[0])
                if problem.scene.terrain.sample_points is not None
                else 0
            ),
            "metadata": _jsonable(problem.scene.terrain.metadata),
        }

    return {
        "task_kind": problem.scene.task_kind.value,
        "has_dynamic_object": problem.scene.has_dynamic_object(),
        "object": object_info,
        "terrain": terrain_info,
        "ground_range": list(problem.scene.ground_range),
        "ground_size": problem.scene.ground_size,
        "metadata": _jsonable(problem.scene.metadata),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _initial_qpos(problem: RetargetingProblem, motion: MotionSequence) -> FloatArray:
    qpos = np.zeros((motion.frame_count, problem.robot.qpos_size(has_object=problem.scene.has_dynamic_object())))
    root_position = slice(*problem.robot.qpos_layout.root_position)
    root_quaternion = slice(*problem.robot.qpos_layout.root_quaternion)
    root_name = problem.motion_format.root_joint if problem.motion_format else motion.joint_names[0]
    if motion.root_poses is not None:
        qpos[:, root_position] = motion.root_poses.positions
        qpos[:, root_quaternion] = motion.root_poses.quaternions(QuaternionOrder.WXYZ)
    else:
        qpos[:, root_position] = motion.joint(root_name)
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
        points = problem.scene.object.sample_points
        return np.asarray(points if points is not None else _default_object_points(), dtype=np.float64)
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


def _joint_limit_arrays(robot: RobotSpec, backend: KinematicsBackend) -> tuple[list[float], list[float]]:
    limits = backend.joint_limits()
    lower: list[float] = []
    upper: list[float] = []
    for name in robot.joint_names:
        lo, hi = limits.get(name, robot.joint_limits.get(name, (-1e6, 1e6)))
        lower.append(float(lo))
        upper.append(float(hi))
    return lower, upper


def _human_points(motion: MotionSequence, human_names: tuple[str, ...], frame_idx: int) -> FloatArray:
    indices = [motion.joint_index(name) for name in human_names]
    return np.asarray(motion.joint_positions[frame_idx, indices, :], dtype=np.float64)


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
        contribution = constraint_terms.get(constraint.name).build(context, constraint)
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


def _constraint_parameter(problem: RetargetingProblem, constraint_name: str, parameter: str, default: float) -> float:
    for constraint in problem.constraints:
        if constraint.name == constraint_name:
            return float(constraint.parameters.get(parameter, default))
    return default
