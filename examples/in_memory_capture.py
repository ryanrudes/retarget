"""Retarget a fully in-memory typed scene observation."""

import numpy as np

from retarget import (
    FrameConvention,
    HumanoidRobotRole,
    MinimalMotionJoint,
    MotionFormat,
    MotionSequence,
    RetargetingExperiment,
    Robot,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
    SampleTimeline,
    SceneObservation,
    TaskKind,
    motion_formats,
    robots,
)

timeline = SampleTimeline.uniform(3, 30.0, clock="in_memory")
joints = tuple(MinimalMotionJoint)
positions = np.zeros((timeline.sample_count, len(joints), 3), dtype=np.float64)
positions[:, joints.index(MinimalMotionJoint.LEFT_TOE)] = (-0.15, 0.10, -0.85)
positions[:, joints.index(MinimalMotionJoint.RIGHT_TOE)] = (0.15, 0.10, -0.85)
motion = MotionSequence(
    name="in_memory",
    joint_vocabulary=MinimalMotionJoint,
    joints=joints,
    root_joint=MinimalMotionJoint.PELVIS,
    joint_positions=positions,
    timeline=timeline,
    frame=FrameConvention.Z_UP_RIGHT_HANDED,
    source_height_m=1.7,
)
observation = SceneObservation(
    name=motion.name,
    timeline=timeline,
    world_frame=motion.frame,
    actor=motion,
)
result = RetargetingExperiment(
    observation=observation,
    recipe=RoleRetargetingRecipe(
        task_kind=TaskKind.ROBOT_ONLY,
        motion_format=motion_formats.get(MotionFormat.MINIMAL),
        scene=RobotOnlySceneRecipe(),
        link_roles={
            MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
            MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
            MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
        },
    ),
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
).run()
result.save_npz("in_memory_capture.npz")
print(result.qpos.shape)
