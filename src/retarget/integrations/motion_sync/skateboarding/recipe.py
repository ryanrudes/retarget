"""Retargeting recipe for skateboarding synchronized clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retarget.core.enums import NonPenetrationSource, SolverBackend, TaskKind
from retarget.optimization.spec import (
    FootStickingConstraintConfig,
    JointLimitsConstraintConfig,
    LinkTrackingObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    SmoothnessObjectiveConfig,
    SolverSpec,
    TrustRegionConstraintConfig,
)
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import RobotSpec

from .source import SkateboardingClipSource


@dataclass(frozen=True)
class SkateboardingRetargetingRecipe:
    """Build the standard skateboarding retargeting problem from a typed source."""

    source: SkateboardingClipSource
    scale_to_robot: bool = False
    output_fps: float | None = None
    show_progress: bool = False
    solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL

    @classmethod
    def from_clip(
        cls,
        synced_path: Path,
        *,
        name: str = "",
        max_frames: int | None = None,
        height_m: float | None = None,
        force_contacts: bool = False,
        save_contact_layer: bool = False,
        scale_to_robot: bool = False,
        output_fps: float | None = None,
        show_progress: bool = False,
        solver_backend: SolverBackend = SolverBackend.CVXPY_CLARABEL,
    ) -> SkateboardingRetargetingRecipe:
        """Construct the recipe from a synced clip path."""

        return cls(
            source=SkateboardingClipSource(
                synced_path=synced_path,
                name=name,
                max_frames=max_frames,
                height_m=height_m,
                force_contacts=force_contacts,
                save_contact_layer=save_contact_layer,
            ),
            scale_to_robot=scale_to_robot,
            output_fps=output_fps,
            show_progress=show_progress,
            solver_backend=solver_backend,
        )

    def build_problem(self, robot: RobotSpec) -> RetargetingProblem:
        """Return the complete skateboarding retargeting problem for ``robot``."""

        prepared = self.source.prepare(robot)
        return prepared.build_problem(
            name=self.source.name or prepared.motion.name,
            task_kind=TaskKind.OBJECT_INTERACTION,
            robot=robot,
            solver=SolverSpec(backend=self.solver_backend, max_iterations=10, trust_radius=0.2),
            objectives=(
                LinkTrackingObjectiveConfig(weight=1.0),
                SmoothnessObjectiveConfig(weight=0.2),
                NominalTrackingObjectiveConfig(weight=5.0),
            ),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(),
                FootStickingConstraintConfig(tolerance=1e-3),
                NonPenetrationConstraintConfig(
                    sources=(NonPenetrationSource.SCENE_POINTS,),
                    links=robot.contact_links,
                    scene_clearance=0.015,
                    activation_distance=0.05,
                ),
            ),
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps or prepared.motion.fps,
            show_progress=self.show_progress,
            metadata={"example": "skateboarding"},
        )
