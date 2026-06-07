"""Quadratic problem representation used by retargeting solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from retarget.core.pose import Pose
    from retarget.core.protocols import KinematicsBackend
    from retarget.mesh import LaplacianWeighting
    from retarget.motion.qpos import NominalQposFrame
    from retarget.pipeline.compiled import CompiledContactFrame, CompiledRetargetingProblem, CompiledTargetFrame
    from retarget.pipeline.problem import AnyRetargetingProblem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LinearConstraint:
    """Linear inequality or equality constraint on the solver variable.

    The constraint is expressed as ``lower <= matrix @ x <= upper``. Set
    matching lower and upper values to represent equality constraints.
    """

    matrix: FloatArray
    lower: FloatArray | None = None
    upper: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("linear constraint matrix must be 2D")
        rows = self.matrix.shape[0]
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and value.shape != (rows,):
                raise ValueError(f"{name} must have shape ({rows},)")


@dataclass(frozen=True)
class ObjectiveContribution:
    """Least-squares contribution from one objective term.

    The contribution is expressed as ``||matrix @ dq - target||^2`` where
    ``dq`` is the current subproblem's actuated-joint increment.
    """

    matrix: FloatArray
    target: FloatArray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("objective contribution matrix must be 2D")
        if self.target.ndim != 1:
            raise ValueError("objective contribution target must be 1D")
        if self.matrix.shape[0] != self.target.shape[0]:
            raise ValueError("objective contribution rows must match target length")


@dataclass(frozen=True)
class ConstraintContribution:
    """Bounds and linear constraints from one constraint term."""

    lower: FloatArray | None = None
    upper: FloatArray | None = None
    trust_radius: float | None = None
    linear_constraints: tuple[LinearConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.lower is not None and self.lower.ndim != 1:
            raise ValueError("constraint contribution lower must be 1D")
        if self.upper is not None and self.upper.ndim != 1:
            raise ValueError("constraint contribution upper must be 1D")
        if self.lower is not None and self.upper is not None and self.lower.shape != self.upper.shape:
            raise ValueError("constraint contribution lower and upper must have matching shapes")
        if self.trust_radius is not None and self.trust_radius <= 0:
            raise ValueError("constraint contribution trust_radius must be positive")


@dataclass(frozen=True)
class TermContext:
    """Per-frame data available to objective and constraint terms.

    Terms operate on the actuated-joint increment for the current SQP
    subproblem. All point arrays and Jacobians are already represented in the
    local frame relevant to the current task, such as the object frame for
    dynamic object interaction.

    Attributes:
        problem (RetargetingProblem): Parent run specification.
        backend (KinematicsBackend): Kinematics provider for positions and Jacobians.
        q_current (FloatArray): Full ``qpos`` vector for the frame being optimized.
        q_previous (FloatArray): ``qpos`` from the prior frame (warm start for smoothness).
        frame_idx (int): Zero-based index into the motion sequence.
        contact_frame (ContactFrame | None): Typed contact view for the current frame, when configured.
        target_frame (TargetFrame | None): Typed link-target view for the current frame, when configured.
        robot_point_names (tuple[str, ...]): Link or joint names used for mesh matching.
        robot_points (FloatArray): Robot match points in the task-local frame, shape ``(P, 3)``.
        robot_jacobians (FloatArray): Position Jacobians w.r.t. active variables, shape ``(P, 3, dof)``.
        environment_points (FloatArray): Scene or object sample points, shape ``(E, 3)``.
        adjacency (tuple[tuple[int, ...], ...]): Mesh neighbor indices per vertex.
        target_laplacian (FloatArray): Desired Laplacian coordinates for the interaction mesh.
        laplacian_weighting (LaplacianWeighting): Neighbor weighting used by the current mesh builder.
        laplacian_epsilon (float): Epsilon used by distance-weighted Laplacians.
        reference_pose (Pose | None): Object pose for dynamic scenes; ``None`` in world frame.
        joint_lower (FloatArray): Actuated joint lower limits for the current robot.
        joint_upper (FloatArray): Actuated joint upper limits for the current robot.
        current_joints (FloatArray): Actuated joint values extracted from ``q_current``.
        variable_indices (NDArray[np.int64] | None): Qpos indices for active optimization variables.
        current_variables (FloatArray | None): Active qpos values extracted from ``q_current``.
        variable_lower (FloatArray | None): Absolute lower bounds for active qpos variables.
        variable_upper (FloatArray | None): Absolute upper bounds for active qpos variables.
        nominal_qpos_frame (NominalQposFrame | None): Optional nominal qpos target view.
    """

    problem: AnyRetargetingProblem
    compiled: CompiledRetargetingProblem
    backend: KinematicsBackend
    q_current: FloatArray
    q_previous: FloatArray
    frame_idx: int
    contact_frame: CompiledContactFrame | None
    target_frame: CompiledTargetFrame | None
    robot_point_names: tuple[str, ...]
    robot_points: FloatArray
    robot_jacobians: FloatArray
    environment_points: FloatArray
    adjacency: tuple[tuple[int, ...], ...]
    target_laplacian: FloatArray
    laplacian_weighting: LaplacianWeighting
    laplacian_epsilon: float
    reference_pose: Pose | None
    joint_lower: FloatArray
    joint_upper: FloatArray
    current_joints: FloatArray
    variable_indices: NDArray[np.int64] | None = None
    current_variables: FloatArray | None = None
    variable_lower: FloatArray | None = None
    variable_upper: FloatArray | None = None
    nominal_qpos_frame: NominalQposFrame | None = None

    @property
    def dof(self) -> int:
        """Number of coordinates in the subproblem variable."""

        return int(self.current_variable_values.shape[0])

    @property
    def active_qpos_indices(self) -> NDArray[np.int64]:
        """Qpos indices represented by the current subproblem variable."""

        if self.variable_indices is not None:
            return np.asarray(self.variable_indices, dtype=np.int64)
        start = self.problem.robot.qpos_layout.joint_start
        return np.arange(start, start + self.problem.robot.dof, dtype=np.int64)

    @property
    def current_variable_values(self) -> FloatArray:
        """Current values of active qpos variables."""

        if self.current_variables is not None:
            return np.asarray(self.current_variables, dtype=np.float64)
        return np.asarray(self.current_joints, dtype=np.float64)

    @property
    def variable_lower_bounds(self) -> FloatArray:
        """Absolute lower bounds for active qpos variables."""

        if self.variable_lower is not None:
            return np.asarray(self.variable_lower, dtype=np.float64)
        return np.asarray(self.joint_lower, dtype=np.float64)

    @property
    def variable_upper_bounds(self) -> FloatArray:
        """Absolute upper bounds for active qpos variables."""

        if self.variable_upper is not None:
            return np.asarray(self.variable_upper, dtype=np.float64)
        return np.asarray(self.joint_upper, dtype=np.float64)


@dataclass(frozen=True)
class QuadraticProblem:
    """Least-squares problem with optional bounds, linear constraints, and trust radius.

    The objective is ``minimize ||A x - b||^2``.

    Attributes:
        matrix (FloatArray): Least-squares design matrix ``A``, shape ``(m, n)``.
        target (FloatArray): Right-hand side ``b``, shape ``(m,)``.
        lower (FloatArray | None): Per-variable lower bounds on ``x``.
        upper (FloatArray | None): Per-variable upper bounds on ``x``.
        initial (FloatArray | None): Trust-region center for ``x`` when a radius is set.
        trust_radius (float | None): Maximum Euclidean norm of ``x - initial``.
        linear_constraints (tuple[LinearConstraint, ...]): Additional linear inequalities on ``x``.
    """

    matrix: FloatArray
    target: FloatArray
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    initial: FloatArray | None = None
    trust_radius: float | None = None
    linear_constraints: tuple[LinearConstraint, ...] = ()

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        if self.target.ndim != 1:
            raise ValueError("target must be 1D")
        if self.matrix.shape[0] != self.target.shape[0]:
            raise ValueError("matrix rows must match target length")
        n = self.matrix.shape[1]
        for name, value in (("lower", self.lower), ("upper", self.upper), ("initial", self.initial)):
            if value is not None and value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if self.trust_radius is not None and self.trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        for constraint in self.linear_constraints:
            if constraint.matrix.shape[1] != n:
                raise ValueError("linear constraint column count must match problem variables")

    def objective_value(self, solution: FloatArray) -> float:
        """Return squared residual cost for a candidate solution."""

        residual = self.matrix @ solution - self.target
        return float(residual @ residual)


@dataclass(frozen=True)
class SolverResult:
    """Solver output.

    Attributes:
        solution (FloatArray): Optimized decision vector for the subproblem.
        cost (float): Squared residual ``||A x - b||^2`` at ``solution``.
        status (str): Backend-specific status string (for example ``"optimal"``).
        iterations (int): Reported inner iteration count; defaults to ``1`` when not tracked.
    """

    solution: FloatArray
    cost: float
    status: str
    iterations: int = 1

    @classmethod
    def from_solution(cls, problem: QuadraticProblem, solution: FloatArray, *, status: str) -> SolverResult:
        """Create a result and compute squared residual cost."""

        solution_array = np.asarray(solution, dtype=np.float64)
        return cls(solution=solution_array, cost=problem.objective_value(solution_array), status=status)
