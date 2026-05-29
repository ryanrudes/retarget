"""Extension point protocols."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from retarget.export.spec import ExportResult, ExportSpec
    from retarget.kinematics.types import GeometryDistance
    from retarget.motion.spec import MotionFormatSpec, MotionSequence
    from retarget.optimization.problem import (
        ConstraintContribution,
        ObjectiveContribution,
        QuadraticProblem,
        SolverResult,
        TermContext,
    )
    from retarget.optimization.spec import ConstraintSpec, ObjectiveSpec
    from retarget.pipeline.problem import RetargetingProblem
    from retarget.results.spec import RetargetingResult
    from retarget.robots.spec import RobotSpec


@runtime_checkable
class MotionLoader(Protocol):
    """Load a motion file into a `MotionSequence`."""

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence: ...


@runtime_checkable
class RobotProvider(Protocol):
    """Resolve robot specs from names or external asset locations."""

    def load(self, name: str, **kwargs: Any) -> RobotSpec: ...


@runtime_checkable
class ObjectiveTerm(Protocol):
    """Optimization objective term."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str: ...

    def build(self, context: TermContext, spec: ObjectiveSpec) -> tuple[ObjectiveContribution, ...]: ...


@runtime_checkable
class ConstraintTerm(Protocol):
    """Optimization constraint term."""

    @property
    def name(self) -> str: ...

    def describe(self) -> str: ...

    def build(self, context: TermContext, spec: ConstraintSpec) -> ConstraintContribution: ...


@runtime_checkable
class Solver(Protocol):
    """Solve a quadratic retargeting subproblem."""

    def solve(self, problem: QuadraticProblem) -> SolverResult: ...


@runtime_checkable
class KinematicsBackend(Protocol):
    """Robot kinematics operations needed by retargeting and metrics."""

    def forward_kinematics(
        self,
        qpos: NDArray[np.float64],
        link_names: tuple[str, ...],
    ) -> NDArray[np.float64]: ...

    def link_positions(self, qpos: NDArray[np.float64], link_names: tuple[str, ...]) -> NDArray[np.float64]: ...

    def body_jacobians(
        self,
        qpos: NDArray[np.float64],
        body_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

    def point_jacobians(
        self,
        qpos: NDArray[np.float64],
        point_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def qpos_to_qvel(
        self,
        qpos: NDArray[np.float64],
        previous_qpos: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]: ...

    def integrate_qvel(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]: ...

    def joint_limits(self) -> dict[str, tuple[float, float]]: ...

    def geom_distances(
        self,
        qpos: NDArray[np.float64],
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
        *,
        max_distance: float = np.inf,
    ) -> tuple[GeometryDistance, ...]: ...

    def collision_candidates(
        self,
        qpos: NDArray[np.float64],
        *,
        margin: float = 0.0,
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> tuple[GeometryDistance, ...]: ...


@runtime_checkable
class Visualizer(Protocol):
    """Render or summarize a retargeting result."""

    def view(self, result: RetargetingResult) -> None: ...


@runtime_checkable
class Exporter(Protocol):
    """Export a retargeting result to an external workflow format."""

    def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult: ...


@runtime_checkable
class Metric(Protocol):
    """Evaluate one metric."""

    @property
    def name(self) -> str: ...

    def evaluate(self, result: RetargetingResult, problem: RetargetingProblem | None = None) -> float: ...
