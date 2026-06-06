"""Climbing/terrain workflow with synthetic terrain metadata."""

from __future__ import annotations

import numpy as np

from retarget import (
    MotionFormat,
    OptimizationProfile,
    Retargeter,
    RetargetingProblem,
    Robot,
    SceneSpec,
    TaskKind,
    TerrainSpec,
)
from retarget.motion import MotionSequence, motion_formats
from retarget.robots import robots

fmt = motion_formats.get(MotionFormat.MINIMAL)
motion = MotionSequence(
    name="climbing",
    joint_names=fmt.joint_names,
    joint_positions=np.zeros((10, len(fmt.joint_names), 3)),
    source_height_m=1.7,
)
problem = RetargetingProblem(
    name="climbing",
    task_kind=TaskKind.CLIMBING,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
    motion=motion,
    motion_format=fmt,
    scene=SceneSpec.climbing(
        terrain=TerrainSpec(
            name="blocks",
            sample_points=np.asarray(
                [
                    [-0.35, 0.0, -0.45],
                    [0.35, 0.0, -0.45],
                    [-0.20, 0.0, -0.10],
                    [0.20, 0.0, -0.10],
                ],
                dtype=np.float64,
            ),
        )
    ),
    constraints=OptimizationProfile.climbing(floor_z=-2.0, scene_clearance=0.025).constraints,
)
Retargeter().run(problem).save_npz("climbing.npz")
