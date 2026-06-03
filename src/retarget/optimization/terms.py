"""Composable objective and constraint term descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from retarget.core.array import FloatArray
from retarget.core.enums import Constraint, Objective
from retarget.mesh.interaction import laplacian_matrix
from retarget.optimization.problem import (
    ConstraintContribution,
    LinearConstraint,
    ObjectiveContribution,
    TermContext,
)
from retarget.optimization.registry import constraint_terms, objective_terms
from retarget.optimization.spec import ConstraintSpec, ObjectiveSpec


@dataclass(frozen=True)
class LaplacianObjective:
    """Least-squares term matching interaction-mesh Laplacian coordinates."""

    weight: float = 10.0
    name: str = "laplacian"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Preserve local spatial relationships between body and environment points."

    def build(self, context: TermContext, _spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        """Linearize interaction-mesh Laplacian preservation."""

        vertices = np.vstack([context.robot_points, context.environment_points])
        laplacian = laplacian_matrix(vertices, [list(neighbors) for neighbors in context.adjacency])
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
    name: str = "smoothness"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Reduce discontinuities between adjacent retargeted frames."

    def build(self, context: TermContext, _spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        """Penalize changes from the previous frame's actuated joints."""

        joint_slice = context.problem.robot.qpos_layout.joint_slice(context.dof)
        return (
            ObjectiveContribution(
                matrix=np.eye(context.dof, dtype=np.float64),
                target=context.q_previous[joint_slice] - context.current_joints,
            ),
        )


@dataclass(frozen=True)
class LinkTrackingObjective:
    """Track named robot links to per-frame world-space target positions."""

    weight: float = 1.0
    name: str = "link_tracking"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Track robot links to motion-provided target positions."

    def build(self, context: TermContext, spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        """Build link-position tracking rows from ``motion.metadata['link_targets']``."""

        targets = _link_targets_for_frame(context, spec)
        if targets is None:
            return ()
        names, positions, weights = targets
        current_positions, current_jacobians = context.backend.point_jacobians(context.q_current, names)
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
class NominalTrackingObjective:
    """Pull selected joints toward the robot's nominal posture."""

    weight: float = 5.0
    name: str = "nominal_tracking"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Keep selected joints close to a nominal solution."

    def build(self, context: TermContext, _spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]:
        """Track configured nominal joints toward zero in the fixture nominal space."""

        nominal_indices = tuple(
            context.problem.robot.joint_index(name) for name in context.problem.robot.nominal_tracking_joints
        )
        if not nominal_indices:
            return ()
        nominal_matrix = np.zeros((len(nominal_indices), context.dof), dtype=np.float64)
        for row, joint_idx in enumerate(nominal_indices):
            nominal_matrix[row, joint_idx] = 1.0
        return (
            ObjectiveContribution(
                matrix=nominal_matrix,
                target=-context.current_joints[list(nominal_indices)],
            ),
        )


@dataclass(frozen=True)
class JointLimitConstraint:
    """Box limits on actuated joint increments from current ``qpos``."""

    name: str = "joint_limits"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Clamp or constrain actuated joints within configured limits."

    def build(self, context: TermContext, _spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain actuated joint increments so qpos remains inside limits."""

        return ConstraintContribution(
            lower=context.joint_lower - context.current_joints,
            upper=context.joint_upper - context.current_joints,
        )


@dataclass(frozen=True)
class TrustRegionConstraint:
    """Cap the Euclidean norm of each SQP joint update."""

    radius: float = 0.2
    name: str = "trust_region"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Bound each optimizer update to keep linearization valid."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain the optimizer step radius."""

        return ConstraintContribution(trust_radius=_parameter(spec, "radius", context.problem.solver.trust_radius))


@dataclass(frozen=True)
class FootContactConstraint:
    """Lock stance feet in xy when velocity-based contact is active."""

    tolerance: float = 1e-3
    name: str = "foot_contact"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Preserve stance-foot xy position during detected contact."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain active contact links near their previous xy position."""

        constraints = foot_contact_constraints(context=context, spec=spec)
        return ConstraintContribution(linear_constraints=tuple(constraints))


@dataclass(frozen=True)
class FootLockConstraint:
    """Pin contact links to a floor height during configured frame windows."""

    tolerance: float = 5e-3
    name: str = "foot_lock"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Pin feet to a configured floor height during explicit frame windows."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain configured contact links to floor height during lock windows."""

        return ConstraintContribution(linear_constraints=tuple(foot_lock_constraints(context=context, spec=spec)))


@dataclass(frozen=True)
class NonPenetrationConstraint:
    """Separate contact links from ground and sampled scene geometry."""

    tolerance: float = 1e-3
    name: str = "non_penetration"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Maintain separation between robot collision geometry and scene geometry."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain robot contacts away from ground and sampled scene geometry."""

        constraints = [
            *ground_non_penetration_constraints(context=context, spec=spec),
            *scene_non_penetration_constraints(context=context, spec=spec),
        ]
        return ConstraintContribution(linear_constraints=tuple(constraints))


@dataclass(frozen=True)
class SelfCollisionConstraint:
    """Maintain minimum separation between configured body pairs."""

    tolerance: float = 0.02
    name: str = "self_collision"

    def describe(self) -> str:
        """Return a short label for logs and diagnostics."""

        return "Maintain distance between configured robot body pairs."

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution:
        """Constrain configured collision pairs to stay separated."""

        return ConstraintContribution(linear_constraints=tuple(self_collision_constraints(context=context, spec=spec)))


def foot_contact_constraints(*, context: TermContext, spec: ConstraintSpec) -> list[LinearConstraint]:
    """Build stance-contact constraints from explicit or inferred frame contacts."""

    if not context.frame_contacts or not context.problem.robot.contact_links:
        return []
    tolerance = _parameter(spec, "tolerance", 1e-3)
    active_links = tuple(
        link
        for motion_joint, active in context.frame_contacts.items()
        if active
        for link in _contact_links(context, motion_joint)
    )
    if not active_links:
        return []
    current_positions, current_jacobians = context.backend.point_jacobians(context.q_current, active_links)
    previous_positions = context.backend.link_positions(context.q_previous, active_links)
    constraints: list[LinearConstraint] = []
    for idx in range(len(active_links)):
        delta_xy = previous_positions[idx, :2] - current_positions[idx, :2]
        constraints.append(
            LinearConstraint(
                matrix=current_jacobians[idx, :2, :],
                lower=delta_xy - tolerance,
                upper=delta_xy + tolerance,
            )
        )
    return constraints


def foot_lock_constraints(*, context: TermContext, spec: ConstraintSpec) -> list[LinearConstraint]:
    """Build floor-height constraints for configured lock windows."""

    windows = _windows(spec)
    if not windows:
        return []
    z_floor = _parameter(spec, "z_floor", 0.0)
    tolerance = _parameter(spec, "tolerance", 5e-3)
    active_links = tuple(
        link for link in context.problem.robot.contact_links if _link_locked(link, windows, context.frame_idx)
    )
    if not active_links:
        return []
    current_positions, current_jacobians = context.backend.point_jacobians(context.q_current, active_links)
    return [
        LinearConstraint(
            matrix=current_jacobians[idx, 2:3, :],
            lower=np.asarray([z_floor - current_positions[idx, 2] - tolerance], dtype=np.float64),
            upper=np.asarray([z_floor - current_positions[idx, 2] + tolerance], dtype=np.float64),
        )
        for idx in range(len(active_links))
    ]


def ground_non_penetration_constraints(*, context: TermContext, spec: ConstraintSpec) -> list[LinearConstraint]:
    """Build floor non-penetration constraints for contact links."""

    if not context.problem.robot.contact_links:
        return []
    tolerance = _parameter(spec, "tolerance", 1e-3)
    floor_z = _parameter(spec, "floor_z", 0.0)
    positions, jacobians = context.backend.point_jacobians(context.q_current, context.problem.robot.contact_links)
    return [
        LinearConstraint(
            matrix=jacobians[idx, 2:3, :],
            lower=np.asarray([floor_z + tolerance - positions[idx, 2]], dtype=np.float64),
            upper=None,
        )
        for idx in range(len(context.problem.robot.contact_links))
    ]


def scene_non_penetration_constraints(*, context: TermContext, spec: ConstraintSpec) -> list[LinearConstraint]:
    """Build sampled scene non-penetration constraints."""

    scene_points = _collision_scene_points(context)
    if scene_points is None or scene_points.size == 0:
        return []
    link_names = _non_penetration_links(context, spec)
    if not link_names:
        return []

    clearance = _parameter(spec, "scene_clearance", 0.02)
    activation_distance = _parameter(spec, "activation_distance", clearance)
    if clearance <= 0 or activation_distance <= 0:
        return []

    link_positions_world, link_jacobians_world = context.backend.point_jacobians(context.q_current, link_names)
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


def self_collision_constraints(*, context: TermContext, spec: ConstraintSpec) -> list[LinearConstraint]:
    """Build linearized self-collision constraints."""

    default_distance = _parameter(spec, "tolerance", 0.02)
    margin_distance = _parameter(spec, "margin", default_distance)
    minimum_distance = _parameter(spec, "minimum_distance", margin_distance)
    if minimum_distance <= 0:
        return []
    candidates = context.backend.collision_candidates(
        context.q_current,
        margin=minimum_distance,
        geom_pairs=_pairs(spec),
    )
    if not candidates:
        return []

    names = tuple(dict.fromkeys(name for candidate in candidates for name in (candidate.first, candidate.second)))
    _positions, jacobians = context.backend.point_jacobians(context.q_current, names)
    jacobian_by_name = dict(zip(names, jacobians, strict=True))

    constraints: list[LinearConstraint] = []
    for candidate in candidates:
        normal = candidate.normal_from_first_to_second
        if float(np.linalg.norm(normal)) <= 1e-12:
            continue
        row = normal @ (jacobian_by_name[candidate.second] - jacobian_by_name[candidate.first])
        constraints.append(
            LinearConstraint(
                matrix=row.reshape(1, -1),
                lower=np.asarray([minimum_distance - candidate.distance], dtype=np.float64),
                upper=None,
            )
        )
    return constraints


def _link_targets_for_frame(
    context: TermContext,
    spec: ObjectiveSpec,
) -> tuple[tuple[str, ...], FloatArray, FloatArray] | None:
    raw = context.problem.motion.metadata.get("link_targets")
    if not isinstance(raw, dict):
        return None
    names = _link_target_names(raw)
    positions = np.asarray(raw.get("positions"), dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] != len(names) or positions.shape[2] != 3:
        raise ValueError("motion metadata link_targets.positions must have shape (frames, links, 3)")
    if context.frame_idx >= positions.shape[0]:
        return None

    weights = _link_target_weights(raw.get("weights"), positions.shape[:2])
    masks = _link_target_masks(raw.get("masks"), positions.shape[:2])
    frame_positions = positions[context.frame_idx]
    frame_weights = weights[context.frame_idx] * float(spec.parameters.get("weight_scale", 1.0))
    finite = np.isfinite(frame_positions).all(axis=1)
    active = masks[context.frame_idx] & finite & (frame_weights > 0.0)
    if not np.any(active):
        return None
    active_indices = np.flatnonzero(active)
    active_names = tuple(names[int(idx)] for idx in active_indices)
    return active_names, frame_positions[active_indices], frame_weights[active_indices]


def _link_target_names(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw.get("names", raw.get("link_names"))
    if value is None:
        raise ValueError("motion metadata link_targets requires names")
    array = np.asarray(value)
    names = (str(array.reshape(()).item()),) if array.shape == () else tuple(str(name) for name in array.tolist())
    if not names:
        raise ValueError("motion metadata link_targets.names must not be empty")
    return names


def _link_target_weights(value: Any, shape: tuple[int, int]) -> FloatArray:
    frames, links = shape
    if value is None:
        return np.ones((frames, links), dtype=np.float64)
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape == ():
        return np.full((frames, links), float(weights), dtype=np.float64)
    if weights.shape == (links,):
        return np.tile(weights.reshape(1, links), (frames, 1))
    if weights.shape == (frames, links):
        return weights
    raise ValueError("motion metadata link_targets.weights must be scalar, (links,), or (frames, links)")


def _link_target_masks(value: Any, shape: tuple[int, int]) -> NDArray[np.bool_]:
    frames, links = shape
    if value is None:
        return np.ones((frames, links), dtype=bool)
    masks = np.asarray(value, dtype=bool)
    if masks.shape == (links,):
        return np.tile(masks.reshape(1, links), (frames, 1))
    if masks.shape == (frames, links):
        return masks
    raise ValueError("motion metadata link_targets.masks must have shape (links,) or (frames, links)")


def _parameter(spec: ConstraintSpec, name: str, default: float) -> float:
    return float(spec.parameters.get(name, default))


def _contact_links(context: TermContext, motion_joint: str) -> tuple[str, ...]:
    lower = motion_joint.lower()
    if "left" in lower or lower.startswith("l_"):
        return tuple(
            link
            for link in context.problem.robot.contact_links
            if "left" in link.lower() or link.lower().startswith("l_")
        )
    if "right" in lower or lower.startswith("r_"):
        return tuple(
            link
            for link in context.problem.robot.contact_links
            if "right" in link.lower() or link.lower().startswith("r_")
        )
    return context.problem.robot.contact_links


def _windows(spec: ConstraintSpec) -> dict[str, tuple[tuple[int, int], ...]]:
    raw_windows = spec.parameters.get("windows", {})
    if not isinstance(raw_windows, dict):
        return {}
    return {
        str(name): tuple((int(start), int(end)) for start, end in ranges)
        for name, ranges in raw_windows.items()
    }


def _link_locked(link_name: str, windows: dict[str, tuple[tuple[int, int], ...]], frame_idx: int) -> bool:
    lower = link_name.lower()
    for key, ranges in windows.items():
        key_lower = key.lower()
        if key_lower not in lower and not lower.startswith(key_lower[:1]):
            continue
        if any(start <= frame_idx <= end for start, end in ranges):
            return True
    return False


def _non_penetration_links(context: TermContext, spec: ConstraintSpec) -> tuple[str, ...]:
    configured = _names(spec, "links", "link_names", "body_names")
    if configured:
        return configured
    if context.problem.robot.contact_links:
        return context.problem.robot.contact_links
    return tuple(dict.fromkeys(context.problem.resolved_link_mapping().values()))


def _collision_scene_points(context: TermContext) -> FloatArray | None:
    if context.problem.scene.object is not None:
        return np.asarray(
            context.problem.scene.object.sample_points
            if context.problem.scene.object.sample_points is not None
            else _default_object_points(),
            dtype=np.float64,
        )
    if context.problem.scene.terrain is not None and context.problem.scene.terrain.sample_points is not None:
        return np.asarray(context.problem.scene.terrain.sample_points, dtype=np.float64)
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


def _names(spec: ConstraintSpec, *parameters: str) -> tuple[str, ...]:
    for parameter in parameters:
        raw_names = spec.parameters.get(parameter)
        if raw_names is None:
            continue
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError(f"{spec.name}.{parameter} must be a sequence of names")
        return tuple(str(name) for name in raw_names)
    return ()


def _pairs(spec: ConstraintSpec) -> tuple[tuple[str, str], ...] | None:
    raw_pairs: Any = spec.parameters.get("geom_pairs")
    if raw_pairs is None:
        raw_pairs = spec.parameters.get("pairs")
    if raw_pairs is None:
        return None
    if not isinstance(raw_pairs, (list, tuple)):
        raise ValueError(f"{spec.name}.geom_pairs must be a sequence of name pairs")
    pairs: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError(f"{spec.name}.geom_pairs entries must contain exactly two names")
        first, second = raw_pair
        pairs.append((str(first), str(second)))
    return tuple(pairs)


objective_terms.register(Objective.LAPLACIAN, LaplacianObjective())
objective_terms.register(Objective.LINK_TRACKING, LinkTrackingObjective())
objective_terms.register(Objective.SMOOTHNESS, SmoothnessObjective())
objective_terms.register(Objective.NOMINAL_TRACKING, NominalTrackingObjective())

constraint_terms.register(Constraint.JOINT_LIMITS, JointLimitConstraint())
constraint_terms.register(Constraint.TRUST_REGION, TrustRegionConstraint())
constraint_terms.register(Constraint.FOOT_CONTACT, FootContactConstraint())
constraint_terms.register(Constraint.FOOT_LOCK, FootLockConstraint())
constraint_terms.register(Constraint.NON_PENETRATION, NonPenetrationConstraint())
constraint_terms.register(Constraint.SELF_COLLISION, SelfCollisionConstraint())
