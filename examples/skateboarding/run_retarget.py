"""Retarget a prepared skateboarding clip onto the Unitree G1 model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from prepare_clip import DEFAULT_DEMO, prepare_clip
from rich.console import Console

from retarget import (
    Constraint,
    ConstraintSpec,
    ContactPlan,
    Objective,
    ObjectiveSpec,
    ObjectSpec,
    ObjectTrajectory,
    PoseSequence,
    Retargeter,
    RetargetingProblem,
    SceneSpec,
    SolverBackend,
    SolverSpec,
    SupportPlane,
    TaskKind,
    motion_formats,
)
from retarget.kinematics.backends import MuJoCoKinematicsBackend, SimpleKinematicsBackend
from retarget.motion import load_motion
from retarget.pipeline.engine import InteractionMeshRetargetingEngine
from retarget.robots import robot_providers
from retarget.robots.spec import RobotSpec
from retarget.visualization import view_result

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default=DEFAULT_DEMO, help="Demo id under --synced-root and --work-dir.")
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
        help="Directory for prepared inputs and retargeting outputs.",
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
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild prepared clip assets.")
    parser.add_argument(
        "--force-contacts",
        action="store_true",
        help="Re-run foot-support detection during preparation.",
    )
    parser.add_argument("--height-m", type=float, help="Source human height in meters forwarded to preparation.")
    parser.add_argument("--scale-to-robot", action="store_true", help="Scale source motion to the robot height.")
    parser.add_argument("--output", type=Path, help="Result path. Defaults to <work-dir>/<demo>/<demo>_retarget.npz.")
    parser.add_argument("--live", action="store_true", help="Open the URDF-backed live visualizer after solving.")
    parser.add_argument("--dry-run", action="store_true", help="Print the result summary after solving.")
    parser.add_argument("--progress", action="store_true", help="Show a Rich per-frame progress bar while retargeting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    output_dir = (args.work_dir.expanduser() / args.demo).resolve()
    output = (args.output or (output_dir / f"{args.demo}_retarget.npz")).resolve()
    _prepare_if_needed(args, output_dir, console)
    robot = _load_robot(args, console)
    backend = _kinematics_backend(robot, args.kinematics, console)
    problem = _build_problem(
        args,
        output_dir,
        robot,
    )
    result = Retargeter(engine=InteractionMeshRetargetingEngine(kinematics=backend)).run(problem)
    result.save_npz(output)
    console.print(f"Saved [bold]{result.name}[/bold] to [cyan]{output}[/cyan]")
    if args.dry_run or not args.live:
        view_result(result, dry_run=True, robot_spec=robot)
    if args.live:
        view_result(result, dry_run=False, robot_spec=robot)


def _prepare_if_needed(args: argparse.Namespace, output_dir: Path, console: Console) -> None:
    motion_path = output_dir / "skate_motion.npz"
    board_path = output_dir / "board_trajectory.npz"
    deck_path = output_dir / "deck_samples.npy"
    if not args.force_prepare and motion_path.exists() and board_path.exists() and deck_path.exists():
        return
    synced = (args.synced_root.expanduser() / args.demo).resolve()
    console.print(f"Preparing synced clip [cyan]{synced}[/cyan]...")
    prepare_clip(
        synced,
        output_dir,
        name=args.demo,
        max_frames=args.max_frames,
        height_m=args.height_m,
        force_contacts=args.force_contacts,
    )


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


def _build_problem(
    args: argparse.Namespace,
    output_dir: Path,
    robot: RobotSpec,
) -> RetargetingProblem:
    motion = load_motion(output_dir / "skate_motion.npz", "smplx", name=args.demo)
    contacts = _contact_plan_from_motion(motion, robot)
    board = np.load(output_dir / "board_trajectory.npz", allow_pickle=True)
    deck_samples = np.asarray(np.load(output_dir / "deck_samples.npy"), dtype=np.float64)
    object_trajectory = ObjectTrajectory(
        name="skateboard",
        poses=PoseSequence.from_arrays(
            np.asarray(board["positions"], dtype=np.float64),
            np.asarray(board["quaternions"], dtype=np.float64),
            fps=float(np.asarray(board["fps"]).reshape(())),
        ),
    )
    scene = SceneSpec(
        task_kind=TaskKind.OBJECT_INTERACTION,
        object=ObjectSpec(name="skateboard", sample_points=deck_samples, trajectory=object_trajectory),
        ground_range=(-3.0, 3.0),
        ground_size=15,
    )
    constraints = [
        ConstraintSpec(name=Constraint.JOINT_LIMITS),
        ConstraintSpec(name=Constraint.TRUST_REGION),
    ]
    constraints.extend(
        [
            ConstraintSpec(name=Constraint.FOOT_CONTACT, parameters={"tolerance": 2e-3}),
            ConstraintSpec(
                name=Constraint.NON_PENETRATION,
                parameters={
                    "links": list(robot.contact_links),
                    "scene_clearance": 0.015,
                    "activation_distance": 0.05,
                },
            ),
        ]
    )
    return RetargetingProblem(
        name=args.demo,
        task_kind=TaskKind.OBJECT_INTERACTION,
        robot=robot,
        motion=motion,
        contacts=contacts,
        motion_format=motion_formats.get("smplx"),
        scene=scene,
        solver=SolverSpec(backend=SolverBackend.CVXPY_CLARABEL, max_iterations=6, trust_radius=0.12),
        objectives=(
            ObjectiveSpec(name=Objective.LINK_TRACKING, weight=1.0),
            ObjectiveSpec(name=Objective.SMOOTHNESS, weight=0.12),
            ObjectiveSpec(name=Objective.NOMINAL_TRACKING, weight=0.05),
        ),
        constraints=tuple(constraints),
        scale_to_robot=args.scale_to_robot,
        output_fps=motion.fps,
        show_progress=args.progress,
        metadata={"example": "skateboarding", "prepared_dir": str(output_dir)},
    )


def _contact_plan_from_motion(motion: object, robot: RobotSpec) -> ContactPlan | None:
    contacts = getattr(motion, "contacts", ())
    if not contacts:
        return None
    support = _support_plane_from_motion(motion)
    subjects = tuple(dict.fromkeys(name for frame in contacts for name in frame))
    link_mapping = {subject: _links_for_subject(subject, robot.contact_links) for subject in subjects}
    metadata = getattr(motion, "metadata", {})
    provenance = metadata.get("contact_provenance", {}) if isinstance(metadata, dict) else {}
    return ContactPlan.from_binary_contacts(
        contacts,
        link_mapping=link_mapping,
        support=support,
        provenance={
            "source": "prepared_skate_motion",
            **(dict(provenance) if isinstance(provenance, dict) else {}),
        },
    )


def _support_plane_from_motion(motion: object) -> SupportPlane | None:
    metadata = getattr(motion, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("support_plane")
    if not isinstance(raw, dict):
        return None
    return SupportPlane(
        normal=np.asarray(raw["normal"], dtype=np.float64),
        origin=np.asarray(raw["origin"], dtype=np.float64),
        up_axis=int(raw.get("up_axis", 2)),
    )


def _links_for_subject(subject: str, contact_links: tuple[str, ...]) -> tuple[str, ...]:
    lower = subject.lower()
    if "left" in lower or lower.startswith(("l_", "l-")):
        return tuple(link for link in contact_links if "left" in link.lower() or link.lower().startswith(("l_", "l-")))
    if "right" in lower or lower.startswith(("r_", "r-")):
        return tuple(
            link for link in contact_links if "right" in link.lower() or link.lower().startswith(("r_", "r-"))
        )
    return contact_links


if __name__ == "__main__":
    main()
