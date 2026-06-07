"""Retarget the bundled minimal fixture through the public experiment API."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from retarget import (
    FootStickingConstraintConfig,
    HumanoidRobotRole,
    InteractionMeshRetargetingEngine,
    InteractionMeshSpec,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    MeshTopology,
    Retargeter,
    RetargetingExperiment,
    SmoothnessObjectiveConfig,
    SolverBackend,
    SolverSpec,
    TaskKind,
    TrustRegionConstraintConfig,
)
from retarget.kinematics.backends import (
    MuJoCoKinematicsBackend,
    SimpleKinematicsBackend,
)
from retarget.motion import motion_formats
from retarget.motion.registry import MinimalMotionJoint
from retarget.pipeline.compiled import compile_robot
from retarget.recipes import (
    MotionFileObservationRecipe,
    RobotOnlySceneRecipe,
    RoleRetargetingRecipe,
)
from retarget.robots import robots
from retarget.robots.spec import RobotSpec
from retarget.visualization import view_result

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION = REPO_ROOT / "tests" / "fixtures" / "minimal_motion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--format", default="minimal")
    parser.add_argument("--name", default="basic")
    parser.add_argument("--work-dir", type=Path, default=EXAMPLE_DIR / "generated")
    parser.add_argument("--robot", default="synthetic_humanoid")
    parser.add_argument(
        "--kinematics",
        choices=("auto", "mujoco", "simple"),
        default="auto",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--height-m", type=float)
    parser.add_argument("--scale-to-robot", action="store_true")
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    output = (
        args.output
        or args.work_dir.expanduser().resolve() / f"{args.name}_retarget.npz"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = robots.get_serialized(args.robot)
    backend = _kinematics_backend(robot, args.kinematics, console)
    experiment = RetargetingExperiment(
        observation=MotionFileObservationRecipe.registered(
            args.motion.expanduser().resolve(),
            motion_formats.key_from_serialized(args.format),
            name=args.name,
            max_frames=args.max_frames,
            source_height_m=args.height_m,
        ),
        recipe=RoleRetargetingRecipe(
            name=args.name,
            task_kind=TaskKind.ROBOT_ONLY,
            motion_format=motion_formats.get_serialized(args.format),
            scene=RobotOnlySceneRecipe(
                ground_size=7,
                ground_range=(-0.75, 0.75),
            ),
            joint_roles={
                MinimalMotionJoint.LEFT_HIP: HumanoidRobotRole.LEFT_HIP,
                MinimalMotionJoint.LEFT_KNEE: HumanoidRobotRole.LEFT_KNEE,
                MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_ANKLE,
                MinimalMotionJoint.RIGHT_HIP: HumanoidRobotRole.RIGHT_HIP,
                MinimalMotionJoint.RIGHT_KNEE: HumanoidRobotRole.RIGHT_KNEE,
                MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_ANKLE,
                MinimalMotionJoint.SPINE: HumanoidRobotRole.TORSO,
                MinimalMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
                MinimalMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
            },
            link_roles={
                MinimalMotionJoint.PELVIS: HumanoidRobotRole.PELVIS,
                MinimalMotionJoint.LEFT_HIP: HumanoidRobotRole.LEFT_HIP,
                MinimalMotionJoint.LEFT_KNEE: HumanoidRobotRole.LEFT_KNEE,
                MinimalMotionJoint.LEFT_TOE: HumanoidRobotRole.LEFT_FOOT,
                MinimalMotionJoint.RIGHT_HIP: HumanoidRobotRole.RIGHT_HIP,
                MinimalMotionJoint.RIGHT_KNEE: HumanoidRobotRole.RIGHT_KNEE,
                MinimalMotionJoint.RIGHT_TOE: HumanoidRobotRole.RIGHT_FOOT,
                MinimalMotionJoint.SPINE: HumanoidRobotRole.TORSO,
                MinimalMotionJoint.LEFT_WRIST: HumanoidRobotRole.LEFT_HAND,
                MinimalMotionJoint.RIGHT_WRIST: HumanoidRobotRole.RIGHT_HAND,
            },
            mesh=InteractionMeshSpec(topology=MeshTopology.DELAUNAY, k_neighbors=4),
            solver=SolverSpec(
                backend=SolverBackend.AUTO,
                max_iterations=8,
                trust_radius=0.2,
                tolerance=1e-6,
            ),
            objectives=(
                LaplacianObjectiveConfig(weight=10.0),
                SmoothnessObjectiveConfig(weight=0.2),
            ),
            constraints=(
                JointLimitsConstraintConfig(),
                TrustRegionConstraintConfig(),
                FootStickingConstraintConfig(tolerance=1e-3),
            ),
            scale_to_robot=args.scale_to_robot,
            output_fps=args.output_fps,
            show_progress=args.progress,
            provenance={"example": "basic"},
        ),
        robot=robot,
        retargeter=Retargeter(
            engine=InteractionMeshRetargetingEngine(kinematics=backend)
        ),
    )
    result = experiment.run()
    result.save_npz(output)
    console.print(f"Saved [bold]{result.name}[/bold] to [cyan]{output}[/cyan]")
    if args.dry_run or not args.live:
        view_result(result, dry_run=True, robot_spec=robot)
    if args.live:
        view_result(result, dry_run=False, robot_spec=robot)


def _kinematics_backend(
    robot: RobotSpec,
    mode: str,
    console: Console,
) -> MuJoCoKinematicsBackend | SimpleKinematicsBackend:
    if mode == "simple":
        if not robot.simple_kinematics:
            raise RuntimeError(
                f"Robot {robot.name!r} does not declare simple kinematics"
            )
        return SimpleKinematicsBackend(compile_robot(robot))
    if robot.mujoco_xml_path is not None and robot.mujoco_xml_path.exists():
        return MuJoCoKinematicsBackend(compile_robot(robot))
    if mode == "auto" and robot.simple_kinematics:
        console.print("[yellow]Using the robot's declared simple kinematics.[/yellow]")
        return SimpleKinematicsBackend(compile_robot(robot))
    raise RuntimeError(f"MuJoCo XML path is missing for robot {robot.name!r}")


if __name__ == "__main__":
    main()
