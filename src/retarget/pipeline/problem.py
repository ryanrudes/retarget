"""Top-level retargeting problem specification."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retarget.core.enums import TaskKind
from retarget.mesh import InteractionMeshSpec
from retarget.motion.spec import MotionFormatSpec, MotionSequence
from retarget.optimization.spec import ConstraintSpec, ObjectiveSpec, OptimizationProfile, SolverSpec
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import SceneSpec


class RetargetingProblem(BaseModel):
    """Complete run specification for retargeting.

    Attributes:
        name (str): Human-readable run identifier.
        task_kind (TaskKind): High-level workflow (robot-only, object interaction, climbing).
        robot (RobotSpec): Target robot model, limits, and default mappings.
        motion (MotionSequence): Source human joint trajectory in world space.
        scene (SceneSpec): Ground, terrain, and optional manipulated object.
        motion_format (MotionFormatSpec | None): Format metadata for contact inference and scaling.
        joint_mapping (dict[str, str] | None): Motion-joint to robot-joint map; ``None`` uses robot defaults.
        mesh (InteractionMeshSpec): Interaction-mesh topology for Laplacian objectives.
        solver (SolverSpec): Backend selection and SQP subproblem solver options.
        objectives (tuple[ObjectiveSpec, ...]): Weighted least-squares terms applied each frame.
        constraints (tuple[ConstraintSpec, ...]): Bounds and linear constraints merged per subproblem.
        scale_to_robot (bool): Rescale motion to ``robot.height_m`` when format height is known.
        output_fps (float | None): Resample motion and scene to this rate before retargeting; ``None`` keeps motion fps.
        metadata (dict[str, Any]): Opaque key-value tags stored on results and manifests.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task_kind: TaskKind
    robot: RobotSpec
    motion: MotionSequence
    scene: SceneSpec
    motion_format: MotionFormatSpec | None = None
    joint_mapping: dict[str, str] | None = None
    mesh: InteractionMeshSpec = Field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    objectives: tuple[ObjectiveSpec, ...] = OptimizationProfile.defaults().objectives
    constraints: tuple[ConstraintSpec, ...] = OptimizationProfile.defaults().constraints
    scale_to_robot: bool = True
    output_fps: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_problem(self) -> RetargetingProblem:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.scene.task_kind != self.task_kind:
            raise ValueError("scene.task_kind must match problem.task_kind")
        if self.output_fps is not None and self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        mapping = self.resolved_joint_mapping()
        unknown_human = set(mapping) - set(self.motion.joint_names)
        if unknown_human:
            raise ValueError(f"joint_mapping references unknown motion joints: {sorted(unknown_human)}")
        unknown_robot = set(mapping.values()) - set(self.robot.joint_names)
        if unknown_robot:
            raise ValueError(f"joint_mapping references unknown robot joints: {sorted(unknown_robot)}")
        link_mapping = self.resolved_link_mapping()
        unknown_link_human = set(link_mapping) - set(self.motion.joint_names)
        if unknown_link_human:
            raise ValueError(f"link mapping references unknown motion joints: {sorted(unknown_link_human)}")
        valid_link_targets = set(self.robot.link_names) | set(self.robot.joint_names)
        unknown_link_robot = set(link_mapping.values()) - valid_link_targets
        if unknown_link_robot:
            raise ValueError(f"link mapping references unknown robot links: {sorted(unknown_link_robot)}")
        if self.scene.has_dynamic_object():
            trajectory = self.scene.object.trajectory if self.scene.object else None
            if trajectory is not None and trajectory.poses.frame_count != self.motion.frame_count:
                raise ValueError("object trajectory frame count must match motion")
        return self

    def resolved_joint_mapping(self) -> dict[str, str]:
        """Return explicit mapping or robot defaults filtered to available motion joints."""

        if self.joint_mapping is not None:
            return dict(self.joint_mapping)
        return {
            human: robot_joint
            for human, robot_joint in self.robot.default_joint_mapping.items()
            if human in self.motion.joint_names and robot_joint in self.robot.joint_names
        }

    def resolved_link_mapping(self) -> dict[str, str]:
        """Return motion-joint to robot-link mapping for interaction-mesh matching."""

        if self.robot.default_link_mapping:
            valid_link_targets = set(self.robot.link_names) | set(self.robot.joint_names)
            return {
                human: link_name
                for human, link_name in self.robot.default_link_mapping.items()
                if human in self.motion.joint_names and link_name in valid_link_targets
            }
        return self.resolved_joint_mapping()

    def validate_registry_references(self) -> None:
        """Validate registered optimization references used by this problem."""

        from retarget.optimization.validation import validate_optimization_references

        validate_optimization_references(
            solver=self.solver,
            objectives=self.objectives,
            constraints=self.constraints,
        )

    def with_optimization_profile(self, profile: OptimizationProfile) -> RetargetingProblem:
        """Return a copy using the objectives and constraints from `profile`."""

        return RetargetingProblem(
            name=self.name,
            task_kind=self.task_kind,
            robot=self.robot,
            motion=self.motion,
            scene=self.scene,
            motion_format=self.motion_format,
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            objectives=profile.objectives,
            constraints=profile.constraints,
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps,
            metadata={**self.metadata, "optimization_profile": profile.name},
        )

    @property
    def fps(self) -> float:
        """Effective playback and result sampling rate in Hz.

        Returns ``output_fps`` when set; otherwise the motion sequence's native ``fps``.
        """

        return float(self.output_fps or self.motion.fps)

    def with_output_fps_applied(self) -> RetargetingProblem:
        """Return a problem whose motion and dynamic scene data match `output_fps`."""

        if self.output_fps is None or abs(float(self.output_fps) - self.motion.fps) <= 1e-9:
            return self
        fps = float(self.output_fps)
        return RetargetingProblem(
            name=self.name,
            task_kind=self.task_kind,
            robot=self.robot,
            motion=self.motion.resampled(fps),
            scene=self.scene.resampled(fps),
            motion_format=self.motion_format,
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            objectives=self.objectives,
            constraints=self.constraints,
            scale_to_robot=self.scale_to_robot,
            output_fps=fps,
            metadata=dict(self.metadata),
        )
