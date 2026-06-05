"""Top-level retargeting problem specification."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from retarget.core.enums import TaskKind
from retarget.mesh import InteractionMeshSpec
from retarget.motion.contact import ContactPlan
from retarget.motion.qpos import InitialQposPlan, NominalQposPlan
from retarget.motion.spec import MotionFormatSpec, MotionSequence
from retarget.motion.targets import LinkTargetPlan
from retarget.optimization.spec import ConstraintConfig, ObjectiveConfig, OptimizationProfile, SolverSpec
from retarget.optimization.variables import QposVariableSpec
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
        contacts (ContactPlan | None): Optional typed contact states and support geometry.
        targets (LinkTargetPlan | None): Optional typed link-tracking targets.
        initial_qpos (InitialQposPlan | None): Optional full-qpos seeds for each frame.
        nominal_qpos (NominalQposPlan | None): Optional frame-aligned nominal robot qpos trajectory.
        motion_format (MotionFormatSpec | None): Format metadata for contact inference and scaling.
        joint_mapping (dict[str, str] | None): Motion-joint to robot-joint map; ``None`` uses robot defaults.
        mesh (InteractionMeshSpec): Interaction-mesh topology for Laplacian objectives.
        solver (SolverSpec): Backend selection and SQP subproblem solver options.
        objectives (tuple[ObjectiveConfig, ...]): Weighted least-squares terms applied each frame.
        constraints (tuple[ConstraintConfig, ...]): Bounds and linear constraints merged per subproblem.
        scale_to_robot (bool): Rescale motion to ``robot.height_m`` when source height is known;
            emits a run warning when enabled but ``height_m`` / ``default_height_m`` is missing.
        output_fps (float | None): Resample motion and scene to this rate before retargeting; ``None`` keeps motion fps.
        show_progress (bool): When ``True``, show a Rich progress bar during per-frame optimization.
        progress_description (str | None): Progress bar label; defaults to :attr:`name`.
        metadata (dict[str, Any]): Opaque key-value tags stored on results and manifests.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task_kind: TaskKind
    robot: RobotSpec
    motion: MotionSequence
    scene: SceneSpec
    contacts: ContactPlan | None = None
    targets: LinkTargetPlan | None = None
    initial_qpos: InitialQposPlan | None = None
    nominal_qpos: NominalQposPlan | None = None
    motion_format: MotionFormatSpec | None = None
    joint_mapping: dict[str, str] | None = None
    mesh: InteractionMeshSpec = Field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    variables: QposVariableSpec = Field(default_factory=QposVariableSpec.actuated)
    objectives: tuple[ObjectiveConfig, ...] = OptimizationProfile.defaults().objectives
    constraints: tuple[ConstraintConfig, ...] = OptimizationProfile.defaults().constraints
    scale_to_robot: bool = True
    output_fps: float | None = None
    show_progress: bool = False
    progress_description: str | None = None
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
        valid_link_targets = set(self.robot.link_names) | set(self.robot.joint_names) | set(self.robot.contact_links)
        unknown_link_robot = set(link_mapping.values()) - valid_link_targets
        if unknown_link_robot:
            raise ValueError(f"link mapping references unknown robot links: {sorted(unknown_link_robot)}")
        trajectory = self.scene.object.trajectory if self.scene.object else None
        if trajectory is not None and trajectory.poses.frame_count != self.motion.frame_count:
            raise ValueError("object trajectory frame count must match motion")
        if self.contacts is not None and self.contacts.frame_count != self.motion.frame_count:
            raise ValueError("contacts frame count must match motion")
        if self.targets is not None and self.targets.frame_count != self.motion.frame_count:
            raise ValueError("targets frame count must match motion")
        expected_qpos_size = self.robot.qpos_size(has_object=self.scene.has_dynamic_object())
        if self.initial_qpos is not None:
            if self.initial_qpos.frame_count != self.motion.frame_count:
                raise ValueError("initial_qpos frame count must match motion")
            if self.initial_qpos.qpos_size != expected_qpos_size:
                raise ValueError(
                    f"initial_qpos qpos_size must be {expected_qpos_size} for robot {self.robot.name!r}"
                )
        if self.nominal_qpos is not None:
            if self.nominal_qpos.frame_count != self.motion.frame_count:
                raise ValueError("nominal_qpos frame count must match motion")
            if self.nominal_qpos.qpos_size != expected_qpos_size:
                raise ValueError(
                    f"nominal_qpos qpos_size must be {expected_qpos_size} for robot {self.robot.name!r}"
                )
        self.variables.resolve(
            self.robot,
            qpos_size=self.robot.qpos_size(has_object=self.scene.has_dynamic_object()),
            joint_limits=self.robot.joint_limits,
        )
        if self.targets is not None:
            valid_targets = set(self.robot.link_names) | set(self.robot.joint_names) | set(self.robot.contact_links)
            unknown_targets = {track.link_name for track in self.targets.tracks} - valid_targets
            if unknown_targets:
                raise ValueError(f"targets reference unknown robot links: {sorted(unknown_targets)}")
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
            valid_link_targets = (
                set(self.robot.link_names)
                | set(self.robot.joint_names)
                | set(self.robot.contact_links)
            )
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
            contacts=self.contacts,
            targets=self.targets,
            initial_qpos=self.initial_qpos,
            nominal_qpos=self.nominal_qpos,
            motion_format=self.motion_format,
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            variables=self.variables,
            objectives=profile.objectives,
            constraints=profile.constraints,
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps,
            show_progress=self.show_progress,
            progress_description=self.progress_description,
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
            contacts=self.contacts.resampled(self.motion.fps, fps) if self.contacts is not None else None,
            targets=self.targets.resampled(self.motion.fps, fps) if self.targets is not None else None,
            initial_qpos=(
                self.initial_qpos.resampled(self.motion.fps, fps)
                if self.initial_qpos is not None
                else None
            ),
            nominal_qpos=(
                self.nominal_qpos.resampled(self.motion.fps, fps)
                if self.nominal_qpos is not None
                else None
            ),
            motion_format=self.motion_format,
            joint_mapping=self.joint_mapping,
            mesh=self.mesh,
            solver=self.solver,
            variables=self.variables,
            objectives=self.objectives,
            constraints=self.constraints,
            scale_to_robot=self.scale_to_robot,
            output_fps=fps,
            show_progress=self.show_progress,
            progress_description=self.progress_description,
            metadata=dict(self.metadata),
        )
