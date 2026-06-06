"""Reusable source and recipe abstractions for typed retargeting workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from retarget.core.enums import TaskKind
from retarget.mesh import InteractionMeshSpec
from retarget.motion.contact import ContactPlan
from retarget.motion.qpos import InitialQposPlan, NominalQposPlan
from retarget.motion.spec import MotionFormatSpec, MotionSequence
from retarget.motion.targets import LinkTargetPlan
from retarget.optimization.spec import ConstraintConfig, ObjectiveConfig, OptimizationProfile, SolverSpec
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import SceneSpec


@dataclass(frozen=True)
class PreparedRetargetingInputs:
    """Typed intermediate objects loaded from an external capture source."""

    motion: MotionSequence
    scene: SceneSpec
    contacts: ContactPlan | None = None
    targets: LinkTargetPlan | None = None
    initial_qpos: InitialQposPlan | None = None
    nominal_qpos: NominalQposPlan | None = None
    motion_format: MotionFormatSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_problem(
        self,
        *,
        name: str | None,
        task_kind: TaskKind,
        robot: RobotSpec,
        mesh: InteractionMeshSpec | None = None,
        solver: SolverSpec | None = None,
        variables: QposVariableSpec | None = None,
        objectives: Sequence[ObjectiveConfig] | None = None,
        constraints: Sequence[ConstraintConfig] | None = None,
        joint_mapping: dict[str, str] | None = None,
        scale_to_robot: bool = True,
        output_fps: float | None = None,
        show_progress: bool = False,
        progress_description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RetargetingProblem:
        """Build a complete `RetargetingProblem` from the prepared source bundle."""

        defaults = OptimizationProfile.defaults()
        return RetargetingProblem(
            name=name or self.motion.name,
            task_kind=task_kind,
            robot=robot,
            motion=self.motion,
            scene=self.scene,
            contacts=self.contacts,
            targets=self.targets,
            initial_qpos=self.initial_qpos,
            nominal_qpos=self.nominal_qpos,
            motion_format=self.motion_format,
            joint_mapping=joint_mapping,
            mesh=mesh or InteractionMeshSpec(),
            solver=solver or SolverSpec(),
            variables=variables or QposVariableSpec.actuated(),
            objectives=tuple(objectives) if objectives is not None else defaults.objectives,
            constraints=tuple(constraints) if constraints is not None else defaults.constraints,
            scale_to_robot=scale_to_robot,
            output_fps=output_fps,
            show_progress=show_progress,
            progress_description=progress_description,
            metadata={**self.metadata, **(metadata or {})},
        )


class RetargetingSource(Protocol):
    """External data source that can load typed retargeting intermediates."""

    def prepare(self, robot: RobotSpec) -> PreparedRetargetingInputs:
        """Load source data and return typed retargeting inputs."""


class RetargetingRecipe(Protocol):
    """Reusable workflow that builds a full retargeting problem."""

    def build_problem(self, robot: RobotSpec) -> RetargetingProblem:
        """Return a complete `RetargetingProblem` for ``robot``."""
