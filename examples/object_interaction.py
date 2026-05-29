"""Object-interaction example with an identity object trajectory."""

from __future__ import annotations

import numpy as np

from retarget import ConstraintSpec, ObjectSpec, ObjectTrajectory, Retargeter, RetargetingProblem, SceneSpec, TaskKind
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

fmt = motion_formats.get("minimal")
motion = MotionSequence(
    name="object_interaction",
    joint_names=fmt.joint_names,
    joint_positions=np.zeros((12, len(fmt.joint_names), 3)),
    metadata={"height_m": 1.7},
)
object_spec = ObjectSpec(
    name="box",
    sample_points=np.asarray(
        [
            [x, y, z]
            for x in (-0.2, 0.2)
            for y in (-0.2, 0.2)
            for z in (-0.2, 0.2)
        ],
        dtype=np.float64,
    ),
    trajectory=ObjectTrajectory.identity(motion.frame_count, fps=motion.fps, name="box"),
)
problem = RetargetingProblem(
    name="object_interaction",
    task_kind=TaskKind.OBJECT_INTERACTION,
    robot=robots.get("synthetic_humanoid"),
    motion=motion,
    motion_format=fmt,
    scene=SceneSpec.object_interaction(object_spec),
    constraints=(
        ConstraintSpec(name="joint_limits"),
        ConstraintSpec(name="trust_region"),
        ConstraintSpec(
            name="non_penetration",
            parameters={"floor_z": -2.0, "scene_clearance": 0.03, "links": ("left_toe", "right_toe")},
        ),
    ),
)
Retargeter().run(problem).save_npz("object_interaction.npz")
