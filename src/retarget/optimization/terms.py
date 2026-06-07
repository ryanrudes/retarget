"""Composable objective and constraint term descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from retarget.core.array import FloatArray
from retarget.core.enums import Constraint, NominalFallback, NonPenetrationSource, Objective
from retarget.kinematics.types import GeometryDistanceJacobian
from retarget.mesh.interaction import laplacian_matrix
from retarget.motion.support import SupportPlane
from retarget.optimization.problem import (
    ConstraintContribution,
    LinearConstraint,
    ObjectiveContribution,
    TermContext,
)
from retarget.optimization.registry import constraint_terms, objective_terms
from retarget.optimization.spec import (
    DiagonalRegularizationObjectiveConfig,
    FootLockConstraintConfig,
    FootStickingConstraintConfig,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    LinkTrackingObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    SelfCollisionConstraintConfig,
    SmoothnessObjectiveConfig,
    TrustRegionConstraintConfig,
)


@dataclass(frozen=True)
class LaplacianObjective:
    """Least-squares term matching interaction-mesh Laplacian coordinates."""

    weight: float = 10.0
    kind = Objective.LAPLACIAN
    config_type: type[LaplacianObjectiveConfig] = LaplacianObjectiveConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Preserve local spatial relationships between body and environment points."

    def build(self, context: TermContext, _config: LaplacianObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        """Linearize interaction-mesh Laplacian preservation."""

        vertices = np.vstack([context.robot_points, context.environment_points])
        laplacian = laplacian_matrix(
            vertices,
            [list(neighbors) for neighbors in context.adjacency],
            weighting=context.laplacian_weighting,
            epsilon=context.laplacian_epsilon,
        )
        current_laplacian = laplacian @ vertices
        point_jacobian = np.zeros((3 * len(vertices), context.dof), dtype=np.float64)
        for point_idx in range(len(context.robot_point_names)):
            point_jacobian[3 * point_idx : 3 * (point_idx + 1), :] = context.robot_jacobians[point_idx]
        deformation = sparse.kron(sparse.csr_matrix(laplacian), sparse.eye(3, format="csr"), format="csr")
        laplacian_jacobian = cast(NDArray[np.float64], deformation @ point_jacobian)
        return (
            ObjectiveContribution(
                matrix=laplacian_jacobian,
                target=context.target_laplacian.reshape(-1) - current_laplacian.reshape(-1),
            ),
        )


@dataclass(frozen=True)
class SmoothnessObjective:
    """Penalize actuated-joint changes between consecutive frames."""

    weight: float = 0.2
    kind = Objective.SMOOTHNESS
    config_type: type[SmoothnessObjectiveConfig] = SmoothnessObjectiveConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Reduce discontinuities between adjacent retargeted frames."

    def build(self, context: TermContext, _config: SmoothnessObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        """Penalize changes from the previous frame's active qpos variables."""

        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=context.q_previous[context.active_qpos_indices] - context.current_variable_values,
            ),
        )


@dataclass(frozen=True)
class LinkTrackingObjective:
    """Track named robot links to per-frame world-space target positions."""

    weight: float = 1.0
    kind = Objective.LINK_TRACKING
    config_type: type[LinkTrackingObjectiveConfig] = LinkTrackingObjectiveConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Track robot links to typed target positions."

    def build(self, context: TermContext, config: LinkTrackingObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        """Build link-position tracking rows from the current target frame."""

        if context.target_frame is None:
            return ()
        names = context.target_frame.link_names
        if not names:
            return ()
        positions = context.target_frame.positions
        weights = context.target_frame.weights * config.weight_scale
        current_positions, current_jacobians = _point_jacobians_for_context(context, names)
        matrix = current_jacobians.reshape(3 * len(names), context.dof)
        target = (positions - current_positions).reshape(-1)
        scales = np.repeat(np.sqrt(weights), 3)
        return (
            ObjectiveContribution(
                matrix=matrix * scales[:, None],
                target=target * scales,
            ),
        )


@dataclass(frozen=True)
class DiagonalRegularizationObjective:
    """Penalize selected qpos variables toward zero."""

    weight: float = 1.0
    kind = Objective.DIAGONAL_REGULARIZATION
    config_type: type[DiagonalRegularizationObjectiveConfig] = DiagonalRegularizationObjectiveConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Apply diagonal qpos regularization over the active variable set."

    def build(
        self,
        context: TermContext,
        config: DiagonalRegularizationObjectiveConfig,
    ) -> tuple[ObjectiveContribution, ...]:
        """Build diagonal rows aligned to active qpos variables."""

        weights = _diagonal_weights(context, config)
        active_cols = np.flatnonzero(weights > 0.0)
        if active_cols.size == 0:
            return ()
        matrix = np.zeros((active_cols.size, context.dof), dtype=np.float64)
        target = np.zeros(active_cols.size, dtype=np.float64)
        for row, col in enumerate(active_cols):
            scale = float(np.sqrt(weights[col]))
            matrix[row, int(col)] = scale
            target[row] = -scale * context.current_variable_values[int(col)]
        return (ObjectiveContribution(matrix=matrix, target=target),)


@dataclass(frozen=True)
class NominalTrackingObjective:
    """Pull selected joints toward the robot's nominal posture."""

    weight: float = 5.0
    kind = Objective.NOMINAL_TRACKING
    config_type: type[NominalTrackingObjectiveConfig] = NominalTrackingObjectiveConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Keep selected joints close to a nominal solution."

    def build(self, context: TermContext, config: NominalTrackingObjectiveConfig) -> tuple[ObjectiveContribution, ...]:
        """Track selected qpos coordinates toward configured nominal values."""

        rows = _nominal_qpos_rows(context, config)
        if not rows:
            return ()
        matrix = np.zeros((len(rows), context.dof), dtype=np.float64)
        target = np.zeros(len(rows), dtype=np.float64)
        for row, (col, target_value, weight) in enumerate(rows):
            scale = float(np.sqrt(weight))
            matrix[row, col] = scale
            target[row] = scale * (target_value - context.current_variable_values[col])
        return (
            ObjectiveContribution(
                matrix=matrix,
                target=target,
            ),
        )


@dataclass(frozen=True)
class JointLimitConstraint:
    """Box limits on actuated joint increments from current ``qpos``."""

    kind = Constraint.JOINT_LIMITS
    config_type: type[JointLimitsConstraintConfig] = JointLimitsConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Clamp or constrain actuated joints within configured limits."

    def build(self, context: TermContext, _config: JointLimitsConstraintConfig) -> ConstraintContribution:
        """Constrain actuated joint increments so qpos remains inside limits."""

        return ConstraintContribution(
            lower=context.variable_lower_bounds - context.current_variable_values,
            upper=context.variable_upper_bounds - context.current_variable_values,
        )


@dataclass(frozen=True)
class TrustRegionConstraint:
    """Cap the Euclidean norm of each SQP joint update."""

    radius: float = 0.2
    kind = Constraint.TRUST_REGION
    config_type: type[TrustRegionConstraintConfig] = TrustRegionConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Bound each optimizer update to keep linearization valid."

    def build(self, context: TermContext, config: TrustRegionConstraintConfig) -> ConstraintContribution:
        """Constrain the optimizer step radius."""

        return ConstraintContribution(trust_radius=config.radius or context.problem.solver.trust_radius)


@dataclass(frozen=True)
class FootStickingConstraint:
    """Lock active support links in the support tangent plane."""

    tolerance: float = 1e-3
    kind = Constraint.FOOT_STICKING
    config_type: type[FootStickingConstraintConfig] = FootStickingConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Preserve active support-link tangent position during detected contact."

    def build(self, context: TermContext, config: FootStickingConstraintConfig) -> ConstraintContribution:
        """Constrain active contact links near their previous tangent-plane position."""

        constraints = foot_sticking_constraints(context=context, config=config)
        return ConstraintContribution(linear_constraints=tuple(constraints))


@dataclass(frozen=True)
class FootLockConstraint:
    """Pin contact links to a support plane during contact or configured frame windows."""

    tolerance: float = 5e-3
    kind = Constraint.FOOT_LOCK
    config_type: type[FootLockConstraintConfig] = FootLockConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Pin feet to support height during detected or configured lock windows."

    def build(self, context: TermContext, config: FootLockConstraintConfig) -> ConstraintContribution:
        """Constrain configured contact links to support height."""

        return ConstraintContribution(linear_constraints=tuple(foot_lock_constraints(context=context, config=config)))


@dataclass(frozen=True)
class NonPenetrationConstraint:
    """Separate contact links from ground and sampled scene geometry."""

    tolerance: float = 1e-3
    kind = Constraint.NON_PENETRATION
    config_type: type[NonPenetrationConstraintConfig] = NonPenetrationConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Maintain separation between robot collision geometry and scene geometry."

    def build(self, context: TermContext, config: NonPenetrationConstraintConfig) -> ConstraintContribution:
        """Constrain robot contacts away from ground and sampled scene geometry."""

        constraints: list[LinearConstraint] = []
        if NonPenetrationSource.SUPPORT in config.sources:
            constraints.extend(ground_non_penetration_constraints(context=context, config=config))
        if NonPenetrationSource.SCENE_POINTS in config.sources:
            constraints.extend(scene_non_penetration_constraints(context=context, config=config))
        if NonPenetrationSource.GEOMETRY in config.sources:
            constraints.extend(geometry_non_penetration_constraints(context=context, config=config))
        return ConstraintContribution(linear_constraints=tuple(constraints))


@dataclass(frozen=True)
class SelfCollisionConstraint:
    """Maintain minimum separation between configured body pairs."""

    tolerance: float = 0.02
    kind = Constraint.SELF_COLLISION
    config_type: type[SelfCollisionConstraintConfig] = SelfCollisionConstraintConfig

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Maintain distance between configured robot body pairs."

    def build(self, context: TermContext, config: SelfCollisionConstraintConfig) -> ConstraintContribution:
        """Constrain configured collision pairs to stay separated."""

        constraints = self_collision_constraints(context=context, config=config)
        return ConstraintContribution(linear_constraints=tuple(constraints))


def foot_sticking_constraints(*, context: TermContext, config: FootStickingConstraintConfig) -> list[LinearConstraint]:
    """Build stance-contact constraints from the typed contact frame."""

    active_links = _active_contact_links(context)
    if not active_links:
        return []
    current_positions, current_jacobians = _point_jacobians_for_context(context, active_links)
    previous_positions = context.backend.link_positions(context.q_previous, active_links)
    constraints: list[LinearConstraint] = []
    tangent_basis = _support_tangent_basis(context.contact_frame.support if context.contact_frame is not None else None)
    for idx in range(len(active_links)):
        if tangent_basis is None:
            matrix = current_jacobians[idx, :2, :]
            delta = previous_positions[idx, :2] - current_positions[idx, :2]
        else:
            matrix = tangent_basis @ current_jacobians[idx]
            delta = tangent_basis @ (previous_positions[idx] - current_positions[idx])
        constraints.append(
            LinearConstraint(
                matrix=matrix,
                lower=delta - config.tolerance,
                upper=delta + config.tolerance,
            )
        )
    return constraints


def foot_lock_constraints(*, context: TermContext, config: FootLockConstraintConfig) -> list[LinearConstraint]:
    """Build support-height constraints for configured lock windows."""

    active_links = _support_contact_links(context)
    if not active_links and config.windows:
        active_links = _window_locked_links(context, config)
    if not active_links:
        return []
    current_positions, current_jacobians = _point_jacobians_for_context(context, active_links)
    support = context.contact_frame.support if context.contact_frame is not None else None
    if support is not None:
        return [
            _support_plane_constraint(
                support=support,
                position=current_positions[idx],
                jacobian=current_jacobians[idx],
                lower=-config.tolerance,
                upper=config.tolerance,
            )
            for idx in range(len(active_links))
        ]
    return [
        LinearConstraint(
            matrix=current_jacobians[idx, 2:3, :],
            lower=np.asarray([config.z_floor - current_positions[idx, 2] - config.tolerance], dtype=np.float64),
            upper=np.asarray([config.z_floor - current_positions[idx, 2] + config.tolerance], dtype=np.float64),
        )
        for idx in range(len(active_links))
    ]


def ground_non_penetration_constraints(
    *,
    context: TermContext,
    config: NonPenetrationConstraintConfig,
) -> list[LinearConstraint]:
    """Build floor non-penetration constraints for contact links."""

    link_names = _non_penetration_links(context, config)
    if not link_names:
        return []
    positions, jacobians = _point_jacobians_for_context(context, link_names)
    support = context.contact_frame.support if context.contact_frame is not None else None
    if support is not None:
        return [
            _support_plane_constraint(
                support=support,
                position=positions[idx],
                jacobian=jacobians[idx],
                lower=config.tolerance,
                upper=None,
            )
            for idx in range(len(link_names))
        ]
    return [
        LinearConstraint(
            matrix=jacobians[idx, 2:3, :],
            lower=np.asarray([config.floor_z + config.tolerance - positions[idx, 2]], dtype=np.float64),
            upper=None,
        )
        for idx in range(len(link_names))
    ]


def scene_non_penetration_constraints(
    *,
    context: TermContext,
    config: NonPenetrationConstraintConfig,
) -> list[LinearConstraint]:
    """Build sampled scene non-penetration constraints."""

    scene_points = _collision_scene_points(context)
    if scene_points is None or scene_points.size == 0:
        return []
    link_names = _non_penetration_links(context, config)
    if not link_names:
        return []

    clearance = config.scene_clearance
    activation_distance = config.activation_distance or clearance

    link_positions_world, link_jacobians_world = _point_jacobians_for_context(context, link_names)
    link_positions = link_positions_world
    link_jacobians = link_jacobians_world
    if context.reference_pose is not None and context.problem.scene.object is not None:
        rotation_inv = context.reference_pose.rotation().as_matrix().T
        link_positions = context.reference_pose.inverse_transform_points(link_positions_world)
        link_jacobians = np.einsum("ab,pbc->pac", rotation_inv, link_jacobians_world)

    constraints: list[LinearConstraint] = []
    for link_idx, link_position in enumerate(link_positions):
        delta = link_position[None, :] - scene_points
        distances = np.linalg.norm(delta, axis=1)
        active_indices = np.flatnonzero(distances <= activation_distance)
        if len(active_indices) == 0:
            closest = int(np.argmin(distances))
            active_indices = np.asarray([closest]) if distances[closest] < clearance else np.asarray([], dtype=int)
        for scene_idx in active_indices:
            distance = float(distances[scene_idx])
            if distance >= clearance:
                continue
            normal = delta[scene_idx] / distance if distance > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            constraints.append(
                LinearConstraint(
                    matrix=(normal @ link_jacobians[link_idx]).reshape(1, -1),
                    lower=np.asarray([clearance - distance], dtype=np.float64),
                    upper=None,
                )
            )
    return constraints


def geometry_non_penetration_constraints(
    *,
    context: TermContext,
    config: NonPenetrationConstraintConfig,
) -> list[LinearConstraint]:
    """Build backend geometry non-penetration constraints."""

    max_distance = config.activation_distance or config.scene_clearance
    if not config.geometry_pairs:
        return []
    distances = _geom_distance_jacobians_for_context(
        context,
        geom_pairs=tuple(
            (
                context.compiled.geometry_name(geometry.first),
                context.compiled.geometry_name(geometry.second),
            )
            for geometry in config.geometry_pairs
        ),
        max_distance=max_distance,
    )
    constraints: list[LinearConstraint] = []
    for row in distances:
        distance = row.distance.distance
        if distance > max_distance:
            continue
        constraints.append(
            LinearConstraint(
                matrix=row.jacobian.reshape(1, -1),
                lower=np.asarray([-distance - config.tolerance], dtype=np.float64),
                upper=None,
            )
        )
    return constraints


def self_collision_constraints(
    *,
    context: TermContext,
    config: SelfCollisionConstraintConfig,
) -> list[LinearConstraint]:
    """Build linearized self-collision constraints."""

    if config.windows is not None and not any(start <= context.frame_idx <= end for start, end in config.windows):
        return []
    candidates = _geom_distance_jacobians_for_context(
        context,
        geom_pairs=tuple(
            (
                context.compiled.geometry_name(pair.first),
                context.compiled.geometry_name(pair.second),
            )
            for pair in config.pairs
        ),
        max_distance=config.margin or config.minimum_distance,
    )
    if not candidates:
        return []

    constraints: list[LinearConstraint] = []
    for candidate in candidates:
        if float(np.linalg.norm(candidate.distance.normal_from_first_to_second)) <= 1e-12:
            continue
        constraints.append(
            LinearConstraint(
                matrix=candidate.jacobian.reshape(1, -1),
                lower=np.asarray([config.minimum_distance - candidate.distance.distance], dtype=np.float64),
                upper=None,
            )
        )
    return constraints


def _active_contact_links(context: TermContext) -> tuple[str, ...]:
    if context.contact_frame is None:
        return ()
    return context.contact_frame.active_link_names


def _support_contact_links(context: TermContext) -> tuple[str, ...]:
    if context.contact_frame is None:
        return ()
    return context.contact_frame.support_link_names


def _support_plane_constraint(
    *,
    support: SupportPlane,
    position: FloatArray,
    jacobian: FloatArray,
    lower: float | None,
    upper: float | None,
) -> LinearConstraint:
    clearance = float(support.clearance(position))
    row = (support.normal @ jacobian).reshape(1, -1)
    return LinearConstraint(
        matrix=row,
        lower=None if lower is None else np.asarray([lower - clearance], dtype=np.float64),
        upper=None if upper is None else np.asarray([upper - clearance], dtype=np.float64),
    )


def _support_tangent_basis(support: SupportPlane | None) -> FloatArray | None:
    if support is None:
        return None
    normal = support.normal
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, normal))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_a = seed - float(np.dot(seed, normal)) * normal
    tangent_a /= float(np.linalg.norm(tangent_a))
    tangent_b = np.cross(normal, tangent_a)
    tangent_b /= float(np.linalg.norm(tangent_b))
    return np.stack([tangent_a, tangent_b], axis=0)


def _window_locked_links(context: TermContext, config: FootLockConstraintConfig) -> tuple[str, ...]:
    links: list[str] = []
    for window in config.windows:
        if not any(start <= context.frame_idx <= end for start, end in window.ranges):
            continue
        if window.link is not None:
            links.append(context.compiled.link_name(window.link))
            continue
        if context.contact_frame is None or window.subject is None:
            continue
        for track in context.contact_frame.tracks:
            if track.subject == window.subject.value:
                links.extend(track.link_names)
    return tuple(dict.fromkeys(links))


def _non_penetration_links(
    context: TermContext,
    config: NonPenetrationConstraintConfig,
) -> tuple[str, ...]:
    links = [context.compiled.link_name(link) for link in config.links]
    if config.subjects and context.contact_frame is not None:
        selected_subjects = {subject.value for subject in config.subjects}
        links.extend(
            link
            for track in context.contact_frame.tracks
            if track.subject in selected_subjects
            for link in track.link_names
        )
    return tuple(dict.fromkeys(links))


def _collision_scene_points(context: TermContext) -> FloatArray | None:
    if context.problem.scene.object is not None:
        return np.asarray(
            context.problem.scene.object.sample_points
            if context.problem.scene.object.sample_points is not None
            else _default_object_points(),
            dtype=np.float64,
        )
    if context.problem.scene.terrain is not None and context.problem.scene.terrain.sample_points is not None:
        return context.problem.scene.terrain.sample_points
    return None


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


def _point_jacobians_for_context(
    context: TermContext,
    point_names: tuple[str, ...],
) -> tuple[FloatArray, FloatArray]:
    method = getattr(context.backend, "point_jacobians_for_qpos_indices", None)
    if callable(method):
        result = method(context.q_current, point_names, context.active_qpos_indices)
        return cast(tuple[FloatArray, FloatArray], result)
    if _uses_default_actuated_variables(context):
        return context.backend.point_jacobians(context.q_current, point_names)
    raise TypeError(
        f"{type(context.backend).__name__} must implement point_jacobians_for_qpos_indices "
        "when the optimization variable set is not actuated joints"
    )


def _geom_distance_jacobians_for_context(
    context: TermContext,
    *,
    geom_pairs: tuple[tuple[str, str], ...],
    max_distance: float,
) -> tuple[GeometryDistanceJacobian, ...]:
    method = getattr(context.backend, "geom_distance_jacobians", None)
    if callable(method):
        result = method(
            context.q_current,
            context.active_qpos_indices,
            geom_pairs,
            max_distance=max_distance,
        )
        return cast(tuple[GeometryDistanceJacobian, ...], result)
    if not _uses_default_actuated_variables(context):
        raise TypeError(
            f"{type(context.backend).__name__} must implement geom_distance_jacobians "
            "when the optimization variable set is not actuated joints"
        )
    distances = context.backend.geom_distances(context.q_current, geom_pairs, max_distance=max_distance)
    if not distances:
        return ()
    names = tuple(dict.fromkeys(name for distance in distances for name in (distance.first, distance.second)))
    _positions, jacobians = context.backend.point_jacobians(context.q_current, names)
    jacobian_by_name = dict(zip(names, jacobians, strict=True))
    rows: list[GeometryDistanceJacobian] = []
    for distance in distances:
        row = distance.normal_from_first_to_second @ (
            jacobian_by_name[distance.second] - jacobian_by_name[distance.first]
        )
        rows.append(GeometryDistanceJacobian(distance=distance, jacobian=row))
    return tuple(rows)


def _uses_default_actuated_variables(context: TermContext) -> bool:
    start = context.problem.robot.qpos_layout.joint_start
    default = np.arange(start, start + context.problem.robot.dof, dtype=np.int64)
    return bool(np.array_equal(context.active_qpos_indices, default))


def _diagonal_weights(context: TermContext, config: DiagonalRegularizationObjectiveConfig) -> FloatArray:
    weights = np.zeros(context.dof, dtype=np.float64)
    if config.variable_weights:
        variable_weights = np.asarray(config.variable_weights, dtype=np.float64)
        if variable_weights.shape != (context.dof,):
            raise ValueError(f"variable_weights must have shape ({context.dof},)")
        weights = np.maximum(weights, variable_weights)
    if config.qpos_weights:
        qpos_weights = np.asarray(config.qpos_weights, dtype=np.float64)
        for col, qpos_idx in enumerate(context.active_qpos_indices):
            index = int(qpos_idx)
            if index < qpos_weights.shape[0]:
                weights[col] = max(weights[col], float(qpos_weights[index]))
    for qpos_idx, weight in config.qpos_weight_overrides.items():
        matches = np.flatnonzero(context.active_qpos_indices == int(qpos_idx))
        for matched_col in matches:
            column = int(matched_col)
            weights[column] = max(weights[column], float(weight))
    return weights


def _nominal_qpos_rows(
    context: TermContext,
    config: NominalTrackingObjectiveConfig,
) -> list[tuple[int, float, float]]:
    selected = _selected_nominal_qpos_indices(context, config)
    qpos_to_col = {int(qpos_idx): col for col, qpos_idx in enumerate(context.active_qpos_indices)}
    rows: list[tuple[int, float, float]] = []
    frame = context.nominal_qpos_frame
    if not selected and frame is not None:
        selected = tuple(int(index) for index in context.active_qpos_indices)
    for qpos_idx in selected:
        col = qpos_to_col.get(qpos_idx)
        if col is None:
            continue
        if frame is None:
            target_value = (
                context.current_variable_values[col] if config.fallback == NominalFallback.CURRENT else 0.0
            )
            rows.append((col, float(target_value), 1.0))
            continue
        if qpos_idx >= frame.plan.qpos_size or not frame.active_at(qpos_idx):
            continue
        rows.append((col, float(frame.qpos[qpos_idx]), float(frame.weights[qpos_idx])))
    return rows


def _selected_nominal_qpos_indices(
    context: TermContext,
    config: NominalTrackingObjectiveConfig,
) -> tuple[int, ...]:
    selected: list[int] = list(config.qpos_indices)
    joints = config.joints
    joint_start = context.problem.robot.qpos_layout.joint_start
    for joint in joints:
        selected.append(joint_start + context.problem.robot.joint_index(joint))
    return tuple(dict.fromkeys(selected))


objective_terms.register(Objective.LAPLACIAN, LaplacianObjective())
objective_terms.register(Objective.LINK_TRACKING, LinkTrackingObjective())
objective_terms.register(Objective.SMOOTHNESS, SmoothnessObjective())
objective_terms.register(Objective.NOMINAL_TRACKING, NominalTrackingObjective())
objective_terms.register(Objective.DIAGONAL_REGULARIZATION, DiagonalRegularizationObjective())

constraint_terms.register(Constraint.JOINT_LIMITS, JointLimitConstraint())
constraint_terms.register(Constraint.TRUST_REGION, TrustRegionConstraint())
constraint_terms.register(Constraint.FOOT_STICKING, FootStickingConstraint())
constraint_terms.register(Constraint.FOOT_LOCK, FootLockConstraint())
constraint_terms.register(Constraint.NON_PENETRATION, NonPenetrationConstraint())
constraint_terms.register(Constraint.SELF_COLLISION, SelfCollisionConstraint())
