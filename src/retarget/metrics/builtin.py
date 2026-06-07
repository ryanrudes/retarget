"""Built-in result metrics."""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import cast

import numpy as np

from retarget.core.enums import MetricKind, MetricName, RunStatus
from retarget.core.protocols import Metric
from retarget.core.registry import Registry
from retarget.kinematics.backends import SimpleKinematicsBackend
from retarget.optimization.spec import NonPenetrationConstraintConfig
from retarget.pipeline.compiled import compile_problem
from retarget.pipeline.problem import RetargetingProblem
from retarget.results.spec import EvaluationReport, RetargetingResult


def _metric_from_decorator(value: object) -> Metric:
    candidate = value
    if isclass(value) or not isinstance(value, Metric):
        if not callable(value):
            raise TypeError("metric registrations must implement Metric or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, Metric):
        raise TypeError("metric registrations must implement Metric")
    return candidate


metrics: Registry[MetricKind, Metric] = Registry(
    "metric",
    MetricKind,
    decorator_transform=_metric_from_decorator,
)
"""Registry of built-in and user-registered result metrics keyed by :class:`~retarget.core.enums.MetricName`."""

METRIC_UNITS = {
    MetricName.OPTIMIZATION_COST.value: "cost",
    MetricName.FOOT_SLIDING.value: "m/s",
    MetricName.CONTACT_PRESERVATION.value: "fraction",
    MetricName.PENETRATION.value: "m",
}


class OptimizationCostMetric:
    """Mean optimization cost."""

    name = MetricName.OPTIMIZATION_COST

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float:
        """Return the mean per-frame optimization cost.

        Args:
            result: Retargeted trajectory to score.
            problem: Unused; present for the :class:`~retarget.core.protocols.Metric` protocol.

        Returns:
            Mean of ``result.cost``, or ``0.0`` when cost is absent.
        """

        if result.cost is None:
            return 0.0
        return float(np.mean(result.cost))


class FootSlidingMetric:
    """Mean stance-foot xy speed during explicitly planned contact."""

    name = MetricName.FOOT_SLIDING

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float:
        """Return mean stance-foot horizontal speed while in contact.

        Args:
            result: Retargeted trajectory to score.
            problem: When provided with a ``ContactPlan``, uses its robot-resolved
                contact links; otherwise falls back to root ``qpos`` xy velocity.

        Returns:
            Mean sliding speed in m/s.
        """

        if result.frame_count < 2:
            return 0.0
        if problem is not None and problem.contacts is not None and problem.robot.contact_links:
            positions = _contact_link_positions(result, problem)
            contacts = _human_contact_mask(problem)
            sliding: list[float] = []
            frame_count = min(result.frame_count, positions.shape[0], contacts.shape[0])
            for frame_idx in range(1, frame_count):
                active_links = contacts[frame_idx] & contacts[frame_idx - 1]
                if np.any(active_links):
                    deltas = positions[frame_idx, active_links, :2] - positions[frame_idx - 1, active_links, :2]
                    sliding.extend((np.linalg.norm(deltas, axis=1) * result.fps).tolist())
            return float(np.mean(sliding)) if sliding else 0.0
        xy_velocity = np.linalg.norm(np.diff(result.qpos[:, :2], axis=0), axis=1) * result.fps
        return float(np.mean(xy_velocity))


class ContactPreservationMetric:
    """Fraction of contact labels preserved by the retargeted contact links."""

    name = MetricName.CONTACT_PRESERVATION

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float:
        """Return the fraction of frames with matching human/robot contact labels.

        Args:
            result: Retargeted trajectory to score.
            problem: When provided with a ``ContactPlan``, compares its contact
                labels to robot link contacts.

        Returns:
            Fraction in ``[0, 1]``; ``1.0`` when human joints are unavailable.
        """

        if result.human_joints is None:
            return 1.0
        if problem is not None and problem.contacts is not None and problem.robot.contact_links:
            human_contacts = _human_contact_mask(problem)
            robot_contacts = _robot_contact_mask(problem, result=result)
            if human_contacts.shape == robot_contacts.shape and human_contacts.size:
                return float(np.mean(human_contacts == robot_contacts))
        finite_human = np.all(np.isfinite(result.human_joints), axis=(1, 2))
        finite_robot = np.all(np.isfinite(result.qpos), axis=1)
        return float(np.mean(finite_human & finite_robot))


class PenetrationMetric:
    """Ground and scene clearance violation depth for configured contact links."""

    name = MetricName.PENETRATION

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float:
        """Return maximum ground or scene penetration depth for contact links.

        Args:
            result: Retargeted trajectory to score.
            problem: Required for scene-aware clearance; uses typed ``non_penetration`` config.

        Returns:
            Maximum violation depth in meters, or ``0.0`` without a problem.
        """

        if problem is not None and problem.robot.contact_links:
            positions = _contact_link_positions(result, problem)
            ground_penetration = _ground_penetration_depth(positions, problem)
            scene_penetration = _scene_penetration_depth(positions, problem)
            return max(ground_penetration, scene_penetration)
        return 0.0


metrics.register(MetricName.OPTIMIZATION_COST, OptimizationCostMetric())
metrics.register(MetricName.FOOT_SLIDING, FootSlidingMetric())
metrics.register(MetricName.CONTACT_PRESERVATION, ContactPreservationMetric())
metrics.register(MetricName.PENETRATION, PenetrationMetric())


def evaluate_result(result: RetargetingResult, problem: RetargetingProblem | None = None) -> EvaluationReport:
    """Evaluate all built-in metrics."""

    metric_problem = _problem_aligned_to_result(result, problem)
    values: dict[str, float] = {}
    warnings = list(result.warnings)
    status = result.status
    for name, metric in metrics.items():
        try:
            value = float(metric.evaluate(result, metric_problem))
        except Exception as exc:
            status = _partial_status(status)
            warnings.append(f"metric {name!r} failed: {type(exc).__name__}: {exc}")
            continue
        if not np.isfinite(value):
            status = _partial_status(status)
            warnings.append(f"metric {name!r} returned a non-finite value")
            continue
        values[name] = value

    return EvaluationReport(
        status=status,
        source_name=result.name,
        frame_count=result.frame_count,
        qpos_dimension=int(result.qpos.shape[1]),
        fps=result.fps,
        task_kind=(
            metric_problem.task_kind.value
            if metric_problem is not None
            else result.run.task_kind.value if result.run is not None else None
        ),
        robot_name=(
            metric_problem.robot.name
            if metric_problem is not None
            else result.run.robot_name if result.run is not None else None
        ),
        motion_name=(
            metric_problem.motion.name
            if metric_problem is not None
            else result.run.motion_name if result.run is not None else None
        ),
        metrics=values,
        metric_units={name: METRIC_UNITS.get(name, "unitless") for name in values},
        details=_evaluation_details(result, metric_problem, original_problem=problem),
        warnings=tuple(warnings),
    )


def _contact_link_positions(result: RetargetingResult, problem: RetargetingProblem) -> np.ndarray:
    compiled = compile_problem(problem)
    backend = SimpleKinematicsBackend(compiled.robot)
    links = tuple(compiled.link_name(link) for link in problem.robot.contact_links)
    return np.stack(
        [backend.link_positions(qpos, links) for qpos in result.qpos],
        axis=0,
    )


def _partial_status(status: RunStatus) -> RunStatus:
    return RunStatus.PARTIAL if status == RunStatus.SUCCESS else status


def _problem_aligned_to_result(
    result: RetargetingResult,
    problem: RetargetingProblem | None,
) -> RetargetingProblem | None:
    if problem is None:
        return None
    if abs(problem.fps - result.fps) <= 1e-9 and problem.motion.frame_count == result.frame_count:
        return problem
    return problem.model_copy(update={"output_fps": result.fps}).with_output_fps_applied()


def _evaluation_details(
    result: RetargetingResult,
    problem: RetargetingProblem | None,
    *,
    original_problem: RetargetingProblem | None = None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "result": {
            "qpos_shape": list(result.qpos.shape),
            "has_cost": result.cost is not None,
            "has_human_joints": result.human_joints is not None,
            "warning_count": len(result.warnings),
        }
    }
    if problem is not None:
        details["problem"] = {
            "task_kind": problem.task_kind.value,
            "robot": problem.robot.name,
            "motion": problem.motion.name,
            "motion_frame_count": problem.motion.frame_count,
            "contact_links": [link.value for link in problem.robot.contact_links],
            "contacts": _contact_plan_details(problem),
            "objectives": [objective.kind.value for objective in problem.objectives],
            "constraints": [constraint.kind.value for constraint in problem.constraints if constraint.enabled],
            "solver_backend": problem.solver.backend.value,
            "aligned_to_result": _problem_was_aligned(original_problem, problem),
        }
    return details


def _problem_was_aligned(
    original_problem: RetargetingProblem | None,
    metric_problem: RetargetingProblem,
) -> bool:
    if original_problem is None:
        return False
    return (
        original_problem.motion.frame_count != metric_problem.motion.frame_count
        or abs(original_problem.fps - metric_problem.fps) > 1e-9
    )


def _scene_penetration_depth(link_positions: np.ndarray, problem: RetargetingProblem) -> float:
    scene_points = _scene_points(problem)
    if scene_points is None or scene_points.size == 0:
        return 0.0
    config = _non_penetration_config(problem)
    clearance = config.scene_clearance if config is not None else 0.0
    if clearance <= 0:
        return 0.0
    max_violation = 0.0
    for frame_idx, frame_positions in enumerate(link_positions):
        if not _scene_frame_available(problem, frame_idx):
            break
        positions = _positions_in_scene_frame(frame_positions, problem, frame_idx)
        distances = np.linalg.norm(positions[:, None, :] - scene_points[None, :, :], axis=2)
        max_violation = max(max_violation, float(max(0.0, clearance - np.min(distances))))
    return max_violation


def _ground_penetration_depth(link_positions: np.ndarray, problem: RetargetingProblem) -> float:
    if problem.contacts is not None and problem.contacts.support is not None:
        clearance = problem.contacts.support.clearance(link_positions)
        return float(max(0.0, -float(np.min(clearance))))
    config = _non_penetration_config(problem)
    floor_z = config.floor_z if config is not None else 0.0
    return float(max(0.0, floor_z - np.min(link_positions[:, :, 2])))


def _scene_frame_available(problem: RetargetingProblem, frame_idx: int) -> bool:
    if problem.scene.object is None or problem.scene.object.trajectory is None:
        return True
    return frame_idx < problem.scene.object.trajectory.poses.frame_count


def _positions_in_scene_frame(positions: np.ndarray, problem: RetargetingProblem, frame_idx: int) -> np.ndarray:
    if problem.scene.object is None or problem.scene.object.trajectory is None:
        return positions
    return problem.scene.object.trajectory.poses.poses[frame_idx].inverse_transform_points(positions)


def _scene_points(problem: RetargetingProblem) -> np.ndarray | None:
    if problem.scene.object is not None:
        if problem.scene.object.sample_points is not None:
            return np.asarray(problem.scene.object.sample_points, dtype=np.float64)
        return np.asarray(
            [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)],
            dtype=np.float64,
        )
    if problem.scene.terrain is not None and problem.scene.terrain.sample_points is not None:
        return np.asarray(problem.scene.terrain.sample_points, dtype=np.float64)
    return None


def _human_contact_mask(problem: RetargetingProblem) -> np.ndarray:
    if problem.contacts is None:
        raise ValueError("contact-aware metrics require an explicit ContactPlan")
    return _contact_plan_to_mask(problem)


def _robot_contact_mask(problem: RetargetingProblem, result: RetargetingResult) -> np.ndarray:
    positions = _contact_link_positions(result, problem)
    if len(positions) == 1:
        speeds = np.zeros((1, len(problem.robot.contact_links)), dtype=np.float64)
    else:
        speeds = np.linalg.norm(np.gradient(positions, 1.0 / result.fps, axis=0), axis=2)
    threshold = 0.01
    return speeds <= threshold


def _contact_plan_to_mask(problem: RetargetingProblem) -> np.ndarray:
    assert problem.contacts is not None
    frame_count = cast(int, problem.contacts.frame_count)
    mask = np.zeros((frame_count, len(problem.robot.contact_links)), dtype=np.bool_)
    link_index = {link: idx for idx, link in enumerate(problem.robot.contact_links)}
    for track in problem.contacts.tracks:
        mapped_indices = tuple(link_index[link] for link in track.links if link in link_index)
        indices = mapped_indices
        if not indices:
            continue
        columns = list(indices)
        mask[:, columns] = mask[:, columns] | track.active_mask[:, None]
    return mask


def _contact_plan_details(problem: RetargetingProblem) -> dict[str, object] | None:
    if problem.contacts is None:
        return None
    frame_count = cast(int, problem.contacts.frame_count)
    return {
        "frame_count": frame_count,
        "subjects": [track.subject.value for track in problem.contacts.tracks],
        "track_count": len(problem.contacts.tracks),
        "has_support": problem.contacts.support is not None,
    }


def _non_penetration_config(problem: RetargetingProblem) -> NonPenetrationConstraintConfig | None:
    for constraint in problem.constraints:
        if isinstance(constraint, NonPenetrationConstraintConfig) and constraint.enabled:
            return constraint
    return None
