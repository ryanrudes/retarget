"""Extension point protocols."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from retarget.core.enums import ConstraintKind, MetricKind, ObjectiveKind
    from retarget.export.spec import ExportResult, ExportSpec
    from retarget.kinematics.types import GeometryDistance, GeometryDistanceJacobian
    from retarget.motion.spec import MotionFormatSpec, MotionSequence
    from retarget.optimization.problem import (
        ConstraintContribution,
        ObjectiveContribution,
        QuadraticProblem,
        SolverResult,
        TermContext,
    )
    from retarget.optimization.spec import ConstraintConfig, ObjectiveConfig
    from retarget.pipeline.problem import RetargetingProblem
    from retarget.results.spec import RetargetingResult
    from retarget.robots.spec import RobotSpec


@runtime_checkable
class MotionLoader(Protocol):
    """Load a motion file into a `MotionSequence`."""

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence:
        """Load motion from disk.

        Args:
            path (Path): Source file path.
            spec (MotionFormatSpec): Format descriptor for parsing.
            name (str | None): Optional clip name override.

        Returns:
            MotionSequence: Parsed motion sequence.
        """
        ...


@runtime_checkable
class RobotProvider(Protocol):
    """Resolve robot specs from names or external asset locations."""

    def load(self, name: str, **kwargs: Any) -> RobotSpec:
        """Resolve a robot specification.

        Args:
            name (str): Registry key or asset identifier.
            **kwargs (Any): Provider-specific options (paths, manifests, etc.).

        Returns:
            RobotSpec: Loaded robot model description.
        """
        ...


ObjectiveConfigT = TypeVar("ObjectiveConfigT", bound="ObjectiveConfig")
ConstraintConfigT = TypeVar("ConstraintConfigT", bound="ConstraintConfig")


@runtime_checkable
class ObjectiveTerm(Protocol[ObjectiveConfigT]):
    """Optimization objective term."""

    @property
    def kind(self) -> ObjectiveKind:
        """Typed registry key for this objective term."""
        ...

    @property
    def config_type(self) -> type[ObjectiveConfigT]:
        """Typed config model consumed by this term."""
        ...

    def describe(self) -> str:
        """Return a short human-readable summary of the term."""
        ...

    def build(self, context: TermContext, config: ObjectiveConfigT) -> tuple[ObjectiveContribution, ...]:
        """Build objective contributions for one optimization step.

        Args:
            context (TermContext): Shared kinematics and trajectory state.
            config (ObjectiveConfig): Term configuration from the problem spec.

        Returns:
            tuple[ObjectiveContribution, ...]: One or more stacked objective blocks.
        """
        ...


@runtime_checkable
class ConstraintTerm(Protocol[ConstraintConfigT]):
    """Optimization constraint term."""

    @property
    def kind(self) -> ConstraintKind:
        """Typed registry key for this constraint term."""
        ...

    @property
    def config_type(self) -> type[ConstraintConfigT]:
        """Typed config model consumed by this term."""
        ...

    def describe(self) -> str:
        """Return a short human-readable summary of the term."""
        ...

    def build(self, context: TermContext, config: ConstraintConfigT) -> ConstraintContribution:
        """Build a constraint contribution for one optimization step.

        Args:
            context (TermContext): Shared kinematics and trajectory state.
            config (ConstraintConfig): Term configuration from the problem spec.

        Returns:
            ConstraintContribution: Linearized inequality or equality block.
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Solve a quadratic retargeting subproblem."""

    def solve(self, problem: QuadraticProblem) -> SolverResult:
        """Solve a quadratic subproblem.

        Args:
            problem (QuadraticProblem): Assembled least-squares or conic problem.

        Returns:
            SolverResult: Primal solution and solver diagnostics.
        """
        ...


@runtime_checkable
class KinematicsBackend(Protocol):
    """Robot kinematics operations needed by retargeting and metrics."""

    def forward_kinematics(
        self,
        qpos: NDArray[np.float64],
        link_names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        """Compute link poses for the given configuration.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            link_names (tuple[str, ...]): Links to evaluate.

        Returns:
            NDArray[np.float64]: Flattened pose array for the requested links.
        """
        ...

    def link_positions(
        self,
        qpos: NDArray[np.float64],
        link_names: tuple[str, ...],
    ) -> NDArray[np.float64]:
        """Return world-frame link origins.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            link_names (tuple[str, ...]): Links to evaluate.

        Returns:
            NDArray[np.float64]: Positions with shape ``(len(link_names), 3)``.
        """
        ...

    def body_jacobians(
        self,
        qpos: NDArray[np.float64],
        body_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return position, rotation, and full spatial Jacobians for bodies.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            body_names (tuple[str, ...]): Bodies to differentiate.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
                Position Jacobians, rotation Jacobians, and stacked spatial Jacobians.
        """
        ...

    def body_jacobians_for_qpos_indices(
        self,
        qpos: NDArray[np.float64],
        body_names: tuple[str, ...],
        qpos_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return body Jacobians with columns selected by qpos indices.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            body_names (tuple[str, ...]): Bodies to differentiate.
            qpos_indices (NDArray[np.int64]): Qpos coordinates used as optimizer variables.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
                Positions, translational Jacobians, and rotational Jacobians.
        """
        ...

    def point_jacobians(
        self,
        qpos: NDArray[np.float64],
        point_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return Jacobians for named kinematic points.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            point_names (tuple[str, ...]): Named points on the model.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]:
                Position Jacobians and optional auxiliary Jacobians.
        """
        ...

    def point_jacobians_for_qpos_indices(
        self,
        qpos: NDArray[np.float64],
        point_names: tuple[str, ...],
        qpos_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return point Jacobians with columns selected by qpos indices.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            point_names (tuple[str, ...]): Named points on the model.
            qpos_indices (NDArray[np.int64]): Qpos coordinates used as optimizer variables.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: Positions and qpos-index Jacobians.
        """
        ...

    def qpos_to_qvel(
        self,
        qpos: NDArray[np.float64],
        previous_qpos: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Finite-difference generalized velocity from two poses.

        Args:
            qpos (NDArray[np.float64]): Current configuration.
            previous_qpos (NDArray[np.float64]): Previous configuration.
            dt (float): Time step in seconds.

        Returns:
            NDArray[np.float64]: Generalized velocity matching ``qpos`` layout.
        """
        ...

    def integrate_qvel(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate generalized velocity for one time step.

        Args:
            qpos (NDArray[np.float64]): Starting configuration.
            qvel (NDArray[np.float64]): Generalized velocity.
            dt (float): Time step in seconds.

        Returns:
            NDArray[np.float64]: Integrated configuration.
        """
        ...

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return per-joint position limits.

        Returns:
            dict[str, tuple[float, float]]: Mapping from joint name to ``(low, high)``.
        """
        ...

    def geom_distances(
        self,
        qpos: NDArray[np.float64],
        geom_pairs: tuple[tuple[str, str], ...],
        *,
        max_distance: float = np.inf,
    ) -> tuple[GeometryDistance, ...]:
        """Measure distances between geometry pairs.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            geom_pairs (tuple[tuple[str, str], ...]): Explicit pairs to evaluate.
            max_distance (float): Ignore pairs farther than this threshold.

        Returns:
            tuple[GeometryDistance, ...]: Signed distances and contact normals per pair.
        """
        ...

    def collision_candidates(
        self,
        qpos: NDArray[np.float64],
        *,
        margin: float = 0.0,
        geom_pairs: tuple[tuple[str, str], ...],
    ) -> tuple[GeometryDistance, ...]:
        """Return geometry pairs within a collision margin.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            margin (float): Distance threshold for candidate inclusion.
            geom_pairs (tuple[tuple[str, str], ...]): Explicit pairs to scan.

        Returns:
            tuple[GeometryDistance, ...]: Near-contact pairs suitable for constraints.
        """
        ...

    def geom_distance_jacobians(
        self,
        qpos: NDArray[np.float64],
        qpos_indices: NDArray[np.int64],
        geom_pairs: tuple[tuple[str, str], ...],
        *,
        max_distance: float = np.inf,
    ) -> tuple[GeometryDistanceJacobian, ...]:
        """Return linearized geometry distances for qpos-index variables.

        Args:
            qpos (NDArray[np.float64]): Generalized coordinates.
            qpos_indices (NDArray[np.int64]): Qpos coordinates used as optimizer variables.
            geom_pairs (tuple[tuple[str, str], ...]): Explicit pairs to evaluate.
            max_distance (float): Ignore pairs farther than this threshold.

        Returns:
            tuple[GeometryDistanceJacobian, ...]: Distance rows suitable for linear constraints.
        """
        ...


@runtime_checkable
class Visualizer(Protocol):
    """Render or summarize a retargeting result."""

    def view(self, result: RetargetingResult) -> None:
        """Open an interactive view or print a summary.

        Args:
            result (RetargetingResult): Retargeting output to display.
        """
        ...


@runtime_checkable
class Exporter(Protocol):
    """Export a retargeting result to an external workflow format."""

    def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult:
        """Write retargeting output to an external format.

        Args:
            result (RetargetingResult): Solved trajectory and structured reports.
            spec (ExportSpec): Target format and destination options.

        Returns:
            ExportResult: Paths and status for written artifacts.
        """
        ...


@runtime_checkable
class Metric(Protocol):
    """Evaluate one metric."""

    @property
    def name(self) -> MetricKind:
        """Registry key for this metric."""
        ...

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float:
        """Compute the metric for a retargeting run.

        Args:
            result (RetargetingResult): Solved trajectory and diagnostics.
            problem (RetargetingProblem | None): Original problem, when needed for context.

        Returns:
            float: Scalar metric value (lower is typically better).
        """
        ...
