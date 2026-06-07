"""Basic robot-only retargeting through the experiment API."""

from pathlib import Path

from retarget import (
    HumanoidRobotRole,
    MinimalMotionJoint,
    MotionFileObservationRecipe,
    MotionFormat,
    RetargetingExperiment,
    Robot,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
    TaskKind,
    motion_formats,
    robots,
)

repo_root = Path(__file__).resolve().parents[1]
motion_format = motion_formats.get(MotionFormat.MINIMAL)
observation = MotionFileObservationRecipe(
    path=repo_root / "tests" / "fixtures" / "minimal_motion.json",
    motion_format=motion_format,
    name="basic",
)
recipe = RoleRetargetingRecipe(
    task_kind=TaskKind.ROBOT_ONLY,
    motion_format=motion_format,
    scene=RobotOnlySceneRecipe(),
    link_roles={
        MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
        MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
        MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
    },
)
result = RetargetingExperiment(
    observation=observation,
    recipe=recipe,
    robot=robots.get(Robot.SYNTHETIC_HUMANOID),
).run()
result.save_npz("basic_robot_only.npz")
print(result.qpos.shape)
