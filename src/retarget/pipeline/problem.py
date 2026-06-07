"""Strongly typed top-level retargeting problem specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeVar

from retarget.core.enums import (
    MotionJoint,
    RobotGeometry,
    RobotJoint,
    RobotLink,
    RobotRole,
    TaskKind,
)
from retarget.mesh import InteractionMeshSpec
from retarget.motion.contact import ContactPlan
from retarget.motion.qpos import InitialQposPlan, NominalQposPlan
from retarget.motion.spec import MotionSequence
from retarget.motion.targets import LinkTargetPlan
from retarget.optimization.spec import (
    ConstraintConfig,
    FootLockConstraintConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    ObjectiveConfig,
    OptimizationProfile,
    SelfCollisionConstraintConfig,
    SolverSpec,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.robots.spec import RobotSpec
from retarget.scene.spec import SceneSpec

MotionJointT = TypeVar("MotionJointT", bound=MotionJoint, default=MotionJoint)
RobotJointT = TypeVar("RobotJointT", bound=RobotJoint, default=RobotJoint)
RobotLinkT = TypeVar("RobotLinkT", bound=RobotLink, default=RobotLink)
RobotGeometryT = TypeVar("RobotGeometryT", bound=RobotGeometry, default=RobotGeometry)
RobotRoleT = TypeVar("RobotRoleT", bound=RobotRole, default=RobotRole)


@dataclass(frozen=True)
class JointBinding(Generic[MotionJointT, RobotJointT]):
    """Explicit source-motion joint to robot-joint binding."""

    source: MotionJointT
    target: RobotJointT

    def __post_init__(self) -> None:
        if not isinstance(self.source, MotionJoint):
            raise TypeError("joint binding source must be a MotionJoint member")
        if not isinstance(self.target, RobotJoint):
            raise TypeError("joint binding target must be a RobotJoint member")


@dataclass(frozen=True)
class LinkBinding(Generic[MotionJointT, RobotLinkT]):
    """Explicit source-motion joint to robot-link binding."""

    source: MotionJointT
    target: RobotLinkT

    def __post_init__(self) -> None:
        if not isinstance(self.source, MotionJoint):
            raise TypeError("link binding source must be a MotionJoint member")
        if not isinstance(self.target, RobotLink):
            raise TypeError("link binding target must be a RobotLink member")


class RetargetingProblem(
    BaseModel,
    Generic[MotionJointT, RobotJointT, RobotLinkT, RobotGeometryT, RobotRoleT],
):
    """Complete typed run specification before backend compilation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    task_kind: TaskKind
    robot: RobotSpec[RobotJointT, RobotLinkT, RobotGeometryT, RobotRoleT]
    motion: MotionSequence[MotionJointT]
    scene: SceneSpec
    contacts: ContactPlan[Any, Any, Any, RobotLinkT] | None = None
    targets: LinkTargetPlan[RobotLinkT] | None = None
    initial_qpos: InitialQposPlan | None = None
    nominal_qpos: NominalQposPlan | None = None
    joint_bindings: tuple[JointBinding[MotionJointT, RobotJointT], ...] = ()
    link_bindings: tuple[LinkBinding[MotionJointT, RobotLinkT], ...] = ()
    mesh: InteractionMeshSpec = Field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    variables: QposVariableSpec = Field(default_factory=QposVariableSpec.actuated)
    objectives: tuple[ObjectiveConfig, ...] = OptimizationProfile.defaults().objectives
    constraints: tuple[ConstraintConfig, ...] = OptimizationProfile.defaults().constraints
    scale_to_robot: bool = True
    output_fps: float | None = None
    show_progress: bool = False
    progress_description: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def _provenance_only(cls, value: dict[str, Any]) -> dict[str, Any]:
        blocked = {
            "constraints",
            "contacts",
            "initial_qpos",
            "joint_bindings",
            "link_bindings",
            "mesh",
            "motion",
            "nominal_qpos",
            "objectives",
            "output_fps",
            "robot",
            "scene",
            "solver",
            "targets",
            "variables",
        }
        present = sorted(blocked & set(value))
        if present:
            raise ValueError("RetargetingProblem provenance cannot contain behavior: " + ", ".join(present))
        return dict(value)

    @model_validator(mode="after")
    def _validate_problem(
        self,
    ) -> RetargetingProblem[MotionJointT, RobotJointT, RobotLinkT, RobotGeometryT, RobotRoleT]:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.scene.task_kind != self.task_kind:
            raise ValueError("scene.task_kind must match problem.task_kind")
        if self.output_fps is not None and self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
        for joint_binding in self.joint_bindings:
            if not isinstance(joint_binding.source, self.motion.joint_vocabulary):
                raise TypeError("joint binding source uses the wrong motion vocabulary")
            if not isinstance(joint_binding.target, self.robot.vocabulary.joints):
                raise TypeError("joint binding target uses the wrong robot-joint vocabulary")
            if joint_binding.source not in self.motion.joints or joint_binding.target not in self.robot.joints:
                raise ValueError("joint binding references an undeclared joint")
        for link_binding in self.link_bindings:
            if not isinstance(link_binding.source, self.motion.joint_vocabulary):
                raise TypeError("link binding source uses the wrong motion vocabulary")
            if not isinstance(link_binding.target, self.robot.vocabulary.links):
                raise TypeError("link binding target uses the wrong robot-link vocabulary")
            if link_binding.source not in self.motion.joints or link_binding.target not in self.robot.links:
                raise ValueError("link binding references an undeclared member")
        if len({binding.source for binding in self.joint_bindings}) != len(self.joint_bindings):
            raise ValueError("joint binding sources must be unique")
        if len({binding.source for binding in self.link_bindings}) != len(self.link_bindings):
            raise ValueError("link binding sources must be unique")
        trajectory = self.scene.object.trajectory if self.scene.object else None
        if trajectory is not None and trajectory.poses.frame_count != self.motion.frame_count:
            raise ValueError("object trajectory frame count must match motion")
        if self.contacts is not None:
            if self.contacts.frame_count != self.motion.frame_count:
                raise ValueError("contacts frame count must match motion")
            if any(
                not isinstance(link, self.robot.vocabulary.links)
                for track in self.contacts.tracks
                for link in track.links
            ):
                raise TypeError("contact plan uses the wrong robot-link vocabulary")
        self._validate_typed_optimization_selections()
        if self.targets is not None:
            if self.targets.frame_count != self.motion.frame_count:
                raise ValueError("targets frame count must match motion")
            if any(not isinstance(link, self.robot.vocabulary.links) for link in self.targets.links):
                raise TypeError("target plan uses the wrong robot-link vocabulary")
            unknown_targets = set(self.targets.links) - set(self.robot.links)
            if unknown_targets:
                raise ValueError(f"targets reference undeclared robot links: {_values(unknown_targets)}")
        expected_qpos_size = self.robot.qpos_size(has_object=self.scene.has_dynamic_object())
        for label, plan in (("initial_qpos", self.initial_qpos), ("nominal_qpos", self.nominal_qpos)):
            if plan is None:
                continue
            if plan.frame_count != self.motion.frame_count:
                raise ValueError(f"{label} frame count must match motion")
            if plan.qpos_size != expected_qpos_size:
                raise ValueError(f"{label} qpos_size must be {expected_qpos_size} for robot {self.robot.name!r}")
        self.variables.resolve(
            self.robot,
            qpos_size=expected_qpos_size,
            joint_limits=self.robot.joint_limits,
        )
        return self

    def _validate_typed_optimization_selections(self) -> None:
        contact_subject_type = (
            type(self.contacts.tracks[0].subject)
            if self.contacts is not None and self.contacts.tracks
            else None
        )
        for objective in self.objectives:
            if isinstance(objective, NominalTrackingObjectiveConfig) and any(
                not isinstance(joint, self.robot.vocabulary.joints)
                for joint in objective.joints
            ):
                raise TypeError("nominal tracking joints use the wrong robot-joint vocabulary")
        for constraint in self.constraints:
            if isinstance(constraint, FootLockConstraintConfig):
                for window in constraint.windows:
                    if window.link is not None and not isinstance(window.link, self.robot.vocabulary.links):
                        raise TypeError("foot-lock link uses the wrong robot-link vocabulary")
                    if window.subject is not None and (
                        contact_subject_type is None or type(window.subject) is not contact_subject_type
                    ):
                        raise TypeError("foot-lock subject uses the wrong contact-subject vocabulary")
            if isinstance(constraint, NonPenetrationConstraintConfig):
                if any(not isinstance(link, self.robot.vocabulary.links) for link in constraint.links):
                    raise TypeError("non-penetration links use the wrong robot-link vocabulary")
                if constraint.subjects and (
                    contact_subject_type is None
                    or any(type(subject) is not contact_subject_type for subject in constraint.subjects)
                ):
                    raise TypeError("non-penetration subjects use the wrong contact-subject vocabulary")
                self._validate_geometry_pairs(constraint.geometry_pairs)
            elif isinstance(constraint, SelfCollisionConstraintConfig):
                self._validate_geometry_pairs(constraint.pairs)

    def _validate_geometry_pairs(self, pairs: tuple[Any, ...]) -> None:
        for pair in pairs:
            for geometry in (pair.first, pair.second):
                if isinstance(geometry, RobotGeometry):
                    if not isinstance(geometry, self.robot.vocabulary.geometries):
                        raise TypeError("constraint geometry uses the wrong robot-geometry vocabulary")
                    if geometry not in self.robot.geometries:
                        raise ValueError("constraint references an undeclared robot geometry")

    def joint_mapping(self) -> dict[MotionJointT, RobotJointT]:
        """Return the typed motion-joint to robot-joint mapping."""

        return {binding.source: binding.target for binding in self.joint_bindings}

    def link_mapping(self) -> dict[MotionJointT, RobotLinkT]:
        """Return the typed motion-joint to robot-link mapping."""

        return {binding.source: binding.target for binding in self.link_bindings}

    def validate_registry_references(self) -> None:
        """Validate registered optimization references used by this problem."""

        from retarget.optimization.validation import validate_optimization_references

        validate_optimization_references(
            solver=self.solver,
            objectives=self.objectives,
            constraints=self.constraints,
        )

    def with_optimization_profile(
        self,
        profile: OptimizationProfile,
    ) -> RetargetingProblem[MotionJointT, RobotJointT, RobotLinkT, RobotGeometryT, RobotRoleT]:
        """Return a copy using one reusable optimization profile."""

        return self.model_copy(
            update={
                "objectives": profile.objectives,
                "constraints": profile.constraints,
                "provenance": {**self.provenance, "optimization_profile": profile.name},
            }
        )

    @property
    def fps(self) -> float:
        """Effective playback and result sampling rate."""

        return float(self.output_fps or self.motion.fps)

    def with_output_fps_applied(
        self,
    ) -> RetargetingProblem[MotionJointT, RobotJointT, RobotLinkT, RobotGeometryT, RobotRoleT]:
        """Return a problem whose dynamic data matches ``output_fps``."""

        if self.output_fps is None or abs(float(self.output_fps) - self.motion.fps) <= 1e-9:
            return self
        fps = float(self.output_fps)
        return self.model_copy(
            update={
                "motion": self.motion.resampled(fps),
                "scene": self.scene.resampled(fps),
                "contacts": self.contacts.resampled(self.motion.fps, fps) if self.contacts is not None else None,
                "targets": self.targets.resampled(self.motion.fps, fps) if self.targets is not None else None,
                "initial_qpos": (
                    self.initial_qpos.resampled(self.motion.fps, fps) if self.initial_qpos is not None else None
                ),
                "nominal_qpos": (
                    self.nominal_qpos.resampled(self.motion.fps, fps) if self.nominal_qpos is not None else None
                ),
                "output_fps": fps,
            }
        )


def _values(values: Any) -> list[str]:
    return sorted(value.value for value in values)


AnyRetargetingProblem = RetargetingProblem[Any, Any, Any, Any, Any]
