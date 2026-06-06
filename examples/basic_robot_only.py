"""Basic robot-only retargeting with the typed Python API."""

from __future__ import annotations

import numpy as np

from retarget import MotionFormat, Retargeter, RetargetingProblem, Robot, SceneSpec, TaskKind
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

joint_names = motion_formats.get(MotionFormat.MINIMAL).joint_names
positions = np.zeros((20, len(joint_names), 3), dtype=float)
positions[:, joint_names.index("Pelvis"), 0] = np.linspace(0.0, 0.2, 20)
positions[:, joint_names.index("L_Toe"), 2] = -0.8
positions[:, joint_names.index("R_Toe"), 2] = -0.8

motion = MotionSequence(
    name="basic",
    joint_names=joint_names,
    joint_positions=positions,
    fps=30,
    source_height_m=1.7,
)
problem = RetargetingProblem(
    name="basic",
    task_kind=TaskKind.ROBOT_ONLY,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
    motion=motion,
    motion_format=motion_formats.get(MotionFormat.MINIMAL),
    scene=SceneSpec.robot_only(),
)
result = Retargeter().run(problem)
result.save_npz("basic_robot_only.npz")
print(result.qpos.shape)
