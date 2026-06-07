"""Climbing workflow through a typed observation and adaptation recipe."""

from pathlib import Path

import numpy as np

from retarget import (
    HumanoidRobotRole,
    MinimalMotionJoint,
    MotionFileObservationRecipe,
    MotionFormat,
    OptimizationProfile,
    RetargetingExperiment,
    Robot,
    RoleRetargetingRecipe,
    SceneSpec,
    StaticSceneRecipe,
    TaskKind,
    TerrainSpec,
    motion_formats,
    robots,
)

repo_root = Path(__file__).resolve().parents[1]
motion_format = motion_formats.get(MotionFormat.MINIMAL)
observation = MotionFileObservationRecipe(
    path=repo_root / "tests" / "fixtures" / "minimal_motion.json",
    motion_format=motion_format,
    name="climbing",
)
scene = SceneSpec.climbing(
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
)
recipe = RoleRetargetingRecipe(
    task_kind=TaskKind.CLIMBING,
    motion_format=motion_format,
    scene=StaticSceneRecipe(scene),
    link_roles={
        MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
        MinimalMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
        MinimalMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
        MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
        MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
    },
    constraints=OptimizationProfile.climbing(
        floor_z=-2.0,
        scene_clearance=0.025,
    ).constraints,
)
RetargetingExperiment(
    observation=observation,
    recipe=recipe,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
).run().save_npz("climbing.npz")
