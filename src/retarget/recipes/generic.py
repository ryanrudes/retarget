"""Generic typed recipes for registered motion files and configured runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from retarget.capture import HumanPoseRecording, JointTrack, PoseTrack, SampleTimeline
from retarget.core.enums import MotionJoint, RobotRole, TaskKind
from retarget.core.pose import PoseSequence
from retarget.mesh import InteractionMeshSpec
from retarget.motion import MotionFormatSpec, MotionSequence, load_motion, motion_formats
from retarget.observation import SceneObservation
from retarget.optimization.spec import (
    ConstraintConfig,
    ObjectiveConfig,
    OptimizationProfile,
    SolverSpec,
)
from retarget.optimization.variables import QposVariableSpec
from retarget.pipeline import RetargetingProblem, SceneRecipe
from retarget.robots.spec import RobotSpec
from retarget.scene import SceneSpec, TerrainSpec


@dataclass(frozen=True)
class MotionFileObservationRecipe:
    """Load a registered motion file into a typed scene observation."""

    path: Path
    motion_format: MotionFormatSpec
    name: str | None = None
    max_frames: int | None = None
    source_height_m: float | None = None

    @classmethod
    def registered(
        cls,
        path: str | Path,
        format_name: str,
        *,
        name: str | None = None,
        max_frames: int | None = None,
        source_height_m: float | None = None,
    ) -> MotionFileObservationRecipe:
        """Construct from a registered motion-format key."""

        return cls(
            path=Path(path),
            motion_format=motion_formats.get(format_name),
            name=name,
            max_frames=max_frames,
            source_height_m=source_height_m,
        )

    def observe(self) -> SceneObservation:
        """Load the file without creating an intermediate checkpoint."""

        motion = load_motion(self.path, self.motion_format.name, name=self.name)
        if self.max_frames is not None:
            if self.max_frames <= 0:
                raise ValueError("max_frames must be positive")
            frame_count = min(self.max_frames, motion.frame_count)
            motion = motion.model_copy(
                update={
                    "joint_positions": motion.joint_positions[:frame_count],
                    "root_poses": (
                        PoseSequence(
                            poses=motion.root_poses.poses[:frame_count],
                            fps=motion.root_poses.fps,
                        )
                        if motion.root_poses is not None
                        else None
                    ),
                }
            )
        if self.source_height_m is not None:
            if self.source_height_m <= 0.0:
                raise ValueError("source_height_m must be positive")
            motion = motion.model_copy(update={"source_height_m": self.source_height_m})
        timeline = SampleTimeline.uniform(
            motion.frame_count,
            motion.fps,
            clock=f"motion:{motion.name}",
        )
        root_pose = None
        if motion.root_poses is not None:
            root_pose = PoseTrack(
                role=self.motion_format.root_joint,
                positions=motion.root_poses.positions,
                quaternions=motion.root_poses.quaternions(),
            )
        actor = HumanPoseRecording(
            name=motion.name,
            timeline=timeline,
            frame=motion.frame,
            joints=tuple(
                JointTrack(role=role, values=motion.joint_positions[:, index, :])
                for index, role in enumerate(self.motion_format.joint_vocabulary)
            ),
            root_pose=root_pose,
            source_height_m=motion.source_height_m,
            provenance={"source_path": str(self.path), "format": self.motion_format.name},
        )
        return SceneObservation(
            name=motion.name,
            timeline=timeline,
            world_frame=motion.frame,
            actor=actor,
            metadata={"source_path": str(self.path), "format": self.motion_format.name},
        )


@dataclass(frozen=True)
class RobotOnlySceneRecipe:
    """Build a flat-ground robot-only scene."""

    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    ground_range: tuple[float, float] = (-1.0, 1.0)
    ground_size: int = 15

    def build_scene(self, observation: SceneObservation) -> SceneSpec:
        """Build the scene independently of the target robot."""

        del observation
        return SceneSpec(
            task_kind=TaskKind.ROBOT_ONLY,
            terrain=self.terrain,
            ground_range=self.ground_range,
            ground_size=self.ground_size,
        )


@dataclass(frozen=True)
class StaticSceneRecipe:
    """Return a predeclared target-independent scene specification."""

    scene: SceneSpec

    def build_scene(self, observation: SceneObservation) -> SceneSpec:
        """Validate the scene timeline and return the configured scene."""

        object_spec = self.scene.object
        if (
            object_spec is not None
            and object_spec.trajectory is not None
            and object_spec.trajectory.poses.frame_count != observation.timeline.sample_count
        ):
            raise ValueError(f"object {object_spec.name!r} trajectory length does not match the observation timeline")
        return self.scene


@dataclass(frozen=True)
class RoleRetargetingRecipe:
    """Resolve typed motion joints through semantic robot roles."""

    task_kind: TaskKind
    motion_format: MotionFormatSpec
    scene: SceneRecipe
    joint_roles: dict[MotionJoint, RobotRole] = field(default_factory=dict)
    link_roles: dict[MotionJoint, RobotRole] = field(default_factory=dict)
    mesh: InteractionMeshSpec = field(default_factory=InteractionMeshSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    variables: QposVariableSpec = field(default_factory=QposVariableSpec.actuated)
    objectives: tuple[ObjectiveConfig, ...] = OptimizationProfile.defaults().objectives
    constraints: tuple[ConstraintConfig, ...] = OptimizationProfile.defaults().constraints
    scale_to_robot: bool = True
    output_fps: float | None = None
    show_progress: bool = False
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vocabulary = self.motion_format.joint_vocabulary
        invalid_joints = [joint for joint in (*self.joint_roles, *self.link_roles) if not isinstance(joint, vocabulary)]
        if invalid_joints:
            raise TypeError(
                f"recipe joints must be members of {vocabulary.__name__}: {[joint.value for joint in invalid_joints]}"
            )

    def build_problem(
        self,
        observation: SceneObservation,
        robot: RobotSpec,
    ) -> RetargetingProblem:
        """Build the generic problem through explicit semantic role bindings."""

        expected_roles = tuple(self.motion_format.joint_vocabulary)
        if observation.actor.joint_roles != expected_roles:
            raise ValueError(f"observation actor vocabulary does not match {self.motion_format.name!r}")
        for role in (*self.joint_roles.values(), *self.link_roles.values()):
            if not isinstance(role, robot.role_vocabulary):
                raise TypeError(f"recipe robot roles must be members of {robot.role_vocabulary.__name__}")
        motion = _motion_from_observation(observation)
        return RetargetingProblem(
            name=self.name or observation.name,
            task_kind=self.task_kind,
            robot=robot,
            motion=motion,
            scene=self.scene.build_scene(observation),
            motion_format=self.motion_format,
            joint_mapping={source.value: robot.joint_for_role(target) for source, target in self.joint_roles.items()},
            link_mapping={source.value: robot.link_for_role(target) for source, target in self.link_roles.items()},
            mesh=self.mesh,
            solver=self.solver,
            variables=self.variables,
            objectives=self.objectives,
            constraints=self.constraints,
            scale_to_robot=self.scale_to_robot,
            output_fps=self.output_fps,
            show_progress=self.show_progress,
            metadata=dict(self.metadata),
        )


def _motion_from_observation(observation: SceneObservation) -> MotionSequence:
    fps = observation.timeline.nominal_fps
    if fps is None:
        raise ValueError("retargeting requires an observation with at least two samples")
    root_poses = None
    if observation.actor.root_pose is not None:
        root_poses = PoseSequence.from_arrays(
            observation.actor.root_pose.positions,
            observation.actor.root_pose.quaternions,
            fps=fps,
            quaternion_order=observation.actor.root_pose.quaternion_order,
            frame=observation.world_frame,
        )
    return MotionSequence(
        name=observation.name,
        joint_positions=np.stack(
            [track.values for track in observation.actor.joints],
            axis=1,
        ),
        joint_names=tuple(role.value for role in observation.actor.joint_roles),
        fps=fps,
        frame=observation.world_frame,
        root_poses=root_poses,
        source_height_m=observation.actor.source_height_m,
        metadata={"observation": observation.name},
    )
