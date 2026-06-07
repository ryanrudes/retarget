"""Fuse native skateboarding capture and retarget it in one experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from retarget import (
    GvhmrOutputSource,
    InteractionMeshRetargetingEngine,
    Retargeter,
    RetargetingExperiment,
    RobotProviderName,
    ViconRecordingSource,
)
from retarget.kinematics.backends import MuJoCoKinematicsBackend
from retarget.pipeline.compiled import compile_robot
from retarget.recipes.skateboarding import (
    DEFAULT_DEMO,
    GVHMR_SCHEMA,
    VICON_SCHEMA,
    SkateboardingObservationRecipe,
    SkateboardingRetargetingRecipe,
)
from retarget.robots import robot_providers
from retarget.robots.spec import RobotSpec
from retarget.visualization import view_result

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default=DEFAULT_DEMO)
    parser.add_argument(
        "--vicon-root",
        type=Path,
        default=REPO_ROOT / "capture_data" / "vicon",
    )
    parser.add_argument(
        "--gvhmr-root",
        type=Path,
        default=REPO_ROOT / "capture_data" / "gvhmr",
    )
    parser.add_argument("--video-fps", type=float, default=59.942)
    parser.add_argument("--height-m", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--work-dir", type=Path, default=EXAMPLE_DIR / "generated")
    parser.add_argument(
        "--store",
        type=Path,
        default=REPO_ROOT / ".retarget_assets",
    )
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--scale-to-robot", action="store_true")
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
        or args.work_dir.expanduser().resolve()
        / f"{args.demo}_retarget.npz"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = _load_robot(args, console)
    backend = _kinematics_backend(robot)
    observation_recipe = SkateboardingObservationRecipe(
        mocap=ViconRecordingSource(
            args.vicon_root.expanduser() / args.demo,
            VICON_SCHEMA,
            name=args.demo,
        ),
        human_pose=GvhmrOutputSource(
            args.gvhmr_root.expanduser() / args.demo,
            GVHMR_SCHEMA,
            fps=args.video_fps,
            name=args.demo,
            source_height_m=args.height_m,
        ),
        max_frames=args.max_frames,
    )
    experiment = RetargetingExperiment(
        observation=observation_recipe,
        recipe=SkateboardingRetargetingRecipe(
            scale_to_robot=args.scale_to_robot,
            show_progress=args.progress,
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


def _load_robot(args: argparse.Namespace, console: Console) -> RobotSpec:
    try:
        return robot_providers.get(RobotProviderName.ASSET_STORE).load(
            args.robot, store=args.store.expanduser()
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if not args.download_assets:
            raise RuntimeError(
                f"Robot asset {args.robot!r} is unavailable in {args.store}; "
                "run scripts/bootstrap_robot_assets.py or pass --download-assets"
            ) from exc
    console.print(f"Bootstrapping [bold]{args.robot}[/bold] into {args.store}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "bootstrap_robot_assets.py"),
            args.robot,
            "--store",
            str(args.store.expanduser()),
        ],
        check=True,
    )
    return robot_providers.get(RobotProviderName.ASSET_STORE).load(
        args.robot, store=args.store.expanduser()
    )


def _kinematics_backend(
    robot: RobotSpec,
) -> MuJoCoKinematicsBackend:
    if robot.mujoco_xml_path is None or not robot.mujoco_xml_path.exists():
        raise RuntimeError(
            f"MuJoCo XML path is missing for robot {robot.name!r}"
        )
    return MuJoCoKinematicsBackend(compile_robot(robot))


if __name__ == "__main__":
    main()
