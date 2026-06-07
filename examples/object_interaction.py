"""Object interaction through the unified experiment API."""

from pathlib import Path

import numpy as np

from retarget import (
    HumanoidRobotRole,
    JointLimitsConstraintConfig,
    MinimalMotionJoint,
    MotionFileObservationRecipe,
    MotionFormat,
    NonPenetrationConstraintConfig,
    NonPenetrationSource,
    ObjectSpec,
    ObjectTrajectory,
    RetargetingExperiment,
    Robot,
    RoleRetargetingRecipe,
    SceneSpec,
    StaticSceneRecipe,
    TaskKind,
    TrustRegionConstraintConfig,
    motion_formats,
    robots,
)
from retarget.robots.registry import SyntheticLink

repo_root = Path(__file__).resolve().parents[1]
motion_format = motion_formats.get(MotionFormat.MINIMAL)
observation = MotionFileObservationRecipe(
    path=repo_root / "tests" / "fixtures" / "minimal_motion.json",
    motion_format=motion_format,
    name="object_interaction",
)
object_spec = ObjectSpec(
    name="box",
    sample_points=np.asarray(
        [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)],
        dtype=np.float64,
    ),
    trajectory=ObjectTrajectory.identity(3, fps=30.0, name="box"),
)
recipe = RoleRetargetingRecipe(
    task_kind=TaskKind.OBJECT_INTERACTION,
    motion_format=motion_format,
    scene=StaticSceneRecipe(SceneSpec.object_interaction(object_spec)),
    link_roles={
        MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
        MinimalMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
        MinimalMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
        MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
        MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
    },
    constraints=(
        JointLimitsConstraintConfig(),
        TrustRegionConstraintConfig(),
        NonPenetrationConstraintConfig(
            sources=(NonPenetrationSource.SCENE_POINTS,),
            links=(SyntheticLink.LEFT_TOE, SyntheticLink.RIGHT_TOE),
            floor_z=-2.0,
            scene_clearance=0.03,
        ),
    ),
)
RetargetingExperiment(
    observation=observation,
    recipe=recipe,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
).run().save_npz("object_interaction.npz")
