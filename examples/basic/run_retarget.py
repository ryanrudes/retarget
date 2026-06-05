"""Retarget the bundled minimal fixture onto the synthetic humanoid."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from retarget import (
    FootStickingConstraintConfig,
    InteractionMeshSpec,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    Retargeter,
    RetargetingProblem,
    SceneSpec,
    SmoothnessObjectiveConfig,
    SolverBackend,
    SolverSpec,
    TaskKind,
    TerrainSpec,
    TrustRegionConstraintConfig,
    motion_formats,
)
from retarget.kinematics.backends import MuJoCoKinematicsBackend, SimpleKinematicsBackend
from retarget.motion import load_motion
from retarget.motion.spec import MotionSequence
from retarget.pipeline.engine import InteractionMeshRetargetingEngine
from retarget.robots import robots
from retarget.robots.spec import RobotSpec
from retarget.visualization import view_result

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION = REPO_ROOT / "tests" / "fixtures" / "minimal_motion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOTION,
        help="Path to a motion file loadable through --format.",
    )
    parser.add_argument("--format", default="minimal", help="Registered motion format name.")
    parser.add_argument("--name", default="basic", help="Run name stamped into the result metadata.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=EXAMPLE_DIR / "generated",
        help="Directory for retargeting outputs.",
    )
    parser.add_argument("--robot", default="synthetic_humanoid", help="Built-in robot registry name.")
    parser.add_argument(
        "--kinematics",
        choices=("auto", "mujoco", "simple"),
        default="auto",
        help="Kinematics backend. auto uses MuJoCo when available.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame cap for smoke runs.")
    parser.add_argument("--height-m", type=float, help="Source human height in meters.")
    parser.add_argument("--scale-to-robot", action="store_true", help="Scale source motion to the robot height.")
    parser.add_argument("--output-fps", type=float, help="Output frame rate. Defaults to the source motion fps.")
    parser.add_argument("--output", type=Path, help="Result path. Defaults to <work-dir>/<name>_retarget.npz.")
    parser.add_argument("--live", action="store_true", help="Open the URDF-backed live visualizer after solving.")
    parser.add_argument("--dry-run", action="store_true", help="Print the result summary after solving.")
    parser.add_argument("--progress", action="store_true", help="Show a Rich per-frame progress bar while retargeting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    output_dir = args.work_dir.expanduser().resolve()
    output = (args.output or (output_dir / f"{args.name}_retarget.npz")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = _load_robot(args.robot)
    backend = _kinematics_backend(robot, args.kinematics, console)
    problem = _build_problem(args, robot)
    result = Retargeter(engine=InteractionMeshRetargetingEngine(kinematics=backend)).run(problem)
    result.save_npz(output)
    console.print(f"Saved [bold]{result.name}[/bold] to [cyan]{output}[/cyan]")
    if args.dry_run or not args.live:
        view_result(result, dry_run=True, robot_spec=robot)
    if args.live:
        view_result(result, dry_run=False, robot_spec=robot)


def _load_robot(name: str) -> RobotSpec:
    return robots.get(name)


def _kinematics_backend(
    robot: RobotSpec,
    mode: str,
    console: Console,
) -> MuJoCoKinematicsBackend | SimpleKinematicsBackend:
    if mode == "simple":
        return SimpleKinematicsBackend(robot)
    if robot.mujoco_xml_path is None or not robot.mujoco_xml_path.exists():
        if mode == "mujoco":
            raise RuntimeError(f"MuJoCo XML path is missing for robot {robot.name!r}")
        console.print("[yellow]MuJoCo XML missing; using simple kinematics for this run.[/yellow]")
        return SimpleKinematicsBackend(robot)
    try:
        return MuJoCoKinematicsBackend(robot)
    except RuntimeError:
        if mode == "mujoco":
            raise
        console.print("[yellow]MuJoCo is not installed; using simple kinematics for this run.[/yellow]")
        return SimpleKinematicsBackend(robot)


def _load_motion(args: argparse.Namespace) -> MotionSequence:
    motion = load_motion(args.motion.expanduser().resolve(), args.format, name=args.name)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        end = min(args.max_frames, motion.frame_count)
        motion = motion.with_positions(motion.joint_positions[:end])
        if motion.contacts:
            motion = motion.model_copy(update={"contacts": motion.contacts[:end]})
    if args.height_m is not None:
        motion = motion.model_copy(update={"metadata": {**motion.metadata, "height_m": args.height_m}})
    return motion


def _build_problem(args: argparse.Namespace, robot: RobotSpec) -> RetargetingProblem:
    motion = _load_motion(args)
    return RetargetingProblem(
        name=args.name,
        task_kind=TaskKind.ROBOT_ONLY,
        robot=robot,
        motion=motion,
        motion_format=motion_formats.get(args.format),
        scene=SceneSpec(
            task_kind=TaskKind.ROBOT_ONLY,
            terrain=TerrainSpec(),
            ground_size=7,
            ground_range=(-0.75, 0.75),
        ),
        mesh=InteractionMeshSpec(topology="delaunay", k_neighbors=4),
        solver=SolverSpec(backend=SolverBackend.AUTO, max_iterations=8, trust_radius=0.2, tolerance=1e-6),
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
        output_fps=args.output_fps or motion.fps,
        show_progress=args.progress,
        metadata={"example": "basic"},
    )


if __name__ == "__main__":
    main()
