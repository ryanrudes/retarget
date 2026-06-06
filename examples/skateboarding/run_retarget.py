"""Retarget a motion_sync skateboarding clip onto the Unitree G1 model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from retarget import (
    Retargeter,
    RetargetingProblem,
)
from retarget.integrations.motion_sync.skateboarding import DEFAULT_DEMO, SkateboardingRetargetingRecipe
from retarget.kinematics.backends import MuJoCoKinematicsBackend, SimpleKinematicsBackend
from retarget.pipeline.engine import InteractionMeshRetargetingEngine
from retarget.robots import robot_providers
from retarget.robots.spec import RobotSpec
from retarget.visualization import view_result

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default=DEFAULT_DEMO, help="Demo id under --synced-root.")
    parser.add_argument("--synced", type=Path, help="Path to synced.npz or a motion_sync demo directory.")
    parser.add_argument(
        "--synced-root",
        type=Path,
        default=REPO_ROOT / "motion_sync_output" / "synced",
        help="Directory containing motion_sync synced demo folders.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=EXAMPLE_DIR / "generated",
        help="Directory for retargeting outputs.",
    )
    parser.add_argument("--store", type=Path, default=REPO_ROOT / ".retarget_assets", help="Asset store root.")
    parser.add_argument("--robot", default="g1", help="Robot asset name in the asset store.")
    parser.add_argument(
        "--kinematics",
        choices=("auto", "mujoco", "simple"),
        default="auto",
        help="Kinematics backend. auto uses MuJoCo when available.",
    )
    parser.add_argument("--max-frames", type=int, help="Optional frame cap for smoke runs.")
    parser.add_argument("--download-assets", action="store_true", help="Bootstrap missing robot assets automatically.")
    parser.add_argument("--force-contacts", action="store_true", help="Re-run foot-support detection.")
    parser.add_argument("--height-m", type=float, help="Source human height in meters.")
    parser.add_argument("--scale-to-robot", action="store_true", help="Scale source motion to the robot height.")
    parser.add_argument("--output", type=Path, help="Result path. Defaults to <work-dir>/<demo>_retarget.npz.")
    parser.add_argument("--live", action="store_true", help="Open the URDF-backed live visualizer after solving.")
    parser.add_argument("--dry-run", action="store_true", help="Print the result summary after solving.")
    parser.add_argument("--progress", action="store_true", help="Show a Rich per-frame progress bar while retargeting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    output_dir = args.work_dir.expanduser().resolve()
    output = (args.output or (output_dir / f"{args.demo}_retarget.npz")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = _load_robot(args, console)
    backend = _kinematics_backend(robot, args.kinematics, console)
    problem = _build_problem(args, robot)
    result = Retargeter(engine=InteractionMeshRetargetingEngine(kinematics=backend)).run(problem)
    result.save_npz(output)
    console.print(f"Saved [bold]{result.name}[/bold] to [cyan]{output}[/cyan]")
    if args.dry_run or not args.live:
        view_result(result, dry_run=True, robot_spec=robot)
    if args.live:
        view_result(result, dry_run=False, robot_spec=robot)


def _load_robot(args: argparse.Namespace, console: Console) -> RobotSpec:
    try:
        return robot_providers.get("asset_store").load(args.robot, store=args.store.expanduser())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if not args.download_assets:
            raise RuntimeError(
                f"Robot asset {args.robot!r} is not available in {args.store}. "
                "Run uv run python scripts/bootstrap_robot_assets.py g1 --store .retarget_assets, "
                "or pass --download-assets."
            ) from exc
    bootstrap = REPO_ROOT / "scripts" / "bootstrap_robot_assets.py"
    console.print(f"Bootstrapping robot asset [bold]{args.robot}[/bold] into [cyan]{args.store}[/cyan]...")
    subprocess.run(
        [sys.executable, str(bootstrap), args.robot, "--store", str(args.store.expanduser())],
        check=True,
    )
    return robot_providers.get("asset_store").load(args.robot, store=args.store.expanduser())


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


def _build_problem(args: argparse.Namespace, robot: RobotSpec) -> RetargetingProblem:
    synced = (
        args.synced.expanduser().resolve()
        if args.synced
        else (args.synced_root.expanduser() / args.demo).resolve()
    )
    recipe = SkateboardingRetargetingRecipe.from_clip(
        synced,
        name=args.demo,
        max_frames=args.max_frames,
        height_m=args.height_m,
        force_contacts=args.force_contacts,
        scale_to_robot=args.scale_to_robot,
        show_progress=args.progress,
    )
    return recipe.build_problem(robot)


if __name__ == "__main__":
    main()
