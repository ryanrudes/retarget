"""Install robot model assets into a local retarget asset store.

The default target is the Unitree G1 model used by the skateboarding research
example. Assets are copied from Holosoma into the ignored ``.retarget_assets``
directory, a validated ``robot.toml`` is generated from the URDF, and the asset
store manifest is updated so run configs can use ``robot_provider = "asset_store"``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from rich.console import Console

from retarget import AssetKind, AssetStore, RobotSpec
from retarget.kinematics.mujoco_xml import strip_floor_contact_pairs

HOLOSOMA_REPO = "https://github.com/amazon-far/holosoma.git"
HOLOSOMA_REF = "main"
HOLOSOMA_G1_DIRS = (
    Path("src/holosoma/holosoma/data/robots/g1"),
    Path("src/interaction_mesh_retarget/robots/g1"),
)
HOLOSOMA_G1_DIR = HOLOSOMA_G1_DIRS[0]
G1_URDF = "g1_29dof.urdf"
G1_MUJOCO_XML = "g1_29dof.xml"
G1_HEIGHT_M = 1.32

G1_LINK_NAMES = (
    "pelvis",
    "torso_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)

G1_CONTACT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")

G1_NOMINAL_TRACKING_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_JOINT_ROLES = {
    "left_hip": "left_hip_pitch_joint",
    "left_knee": "left_knee_joint",
    "left_ankle": "left_ankle_pitch_joint",
    "right_hip": "right_hip_pitch_joint",
    "right_knee": "right_knee_joint",
    "right_ankle": "right_ankle_pitch_joint",
    "torso": "waist_yaw_joint",
    "left_hand": "left_wrist_roll_joint",
    "right_hand": "right_wrist_roll_joint",
}

G1_LINK_ROLES = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "head": "torso_link",
    "left_hip": "left_hip_pitch_link",
    "left_knee": "left_knee_link",
    "left_ankle": "left_ankle_pitch_link",
    "left_foot": "left_ankle_roll_link",
    "right_hip": "right_hip_pitch_link",
    "right_knee": "right_knee_link",
    "right_ankle": "right_ankle_pitch_link",
    "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_roll_link",
    "right_hand": "right_wrist_roll_link",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="?", default="g1", choices=("g1",), help="Robot model to install.")
    parser.add_argument("--store", type=Path, default=Path(".retarget_assets"), help="Asset store root.")
    parser.add_argument("--repo", default=HOLOSOMA_REPO, help="Holosoma git repository URL.")
    parser.add_argument("--ref", default=HOLOSOMA_REF, help="Holosoma branch or tag to clone.")
    parser.add_argument("--holosoma-root", type=Path, help="Use an existing Holosoma checkout instead of cloning.")
    parser.add_argument("--force", action="store_true", help="Replace an existing copied robot asset directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    source_root = _holosoma_root(args, console)
    source_dir = _source_g1_dir(source_root)

    target_dir = args.store.resolve() / "robot" / "g1"
    _copy_robot_asset(source_dir, target_dir, source_root=source_root, force=args.force)
    spec_path = _write_g1_robot_spec(target_dir, source_repo=args.repo, source_ref=args.ref)
    spec = RobotSpec.load(spec_path)
    record = AssetStore(args.store).import_path(
        target_dir,
        name="g1",
        kind=AssetKind.ROBOT,
        copy=False,
        license="Apache-2.0",
        notice="Robot assets copied from amazon-far/holosoma; see HOLOSOMA_LICENSE and HOLOSOMA_NOTICE.",
        metadata={
            "source_repository": args.repo,
            "source_ref": args.ref,
            "source_path": str(source_dir.relative_to(source_root)),
            "urdf_path": str(spec.urdf_path),
            "mujoco_xml_path": str(spec.mujoco_xml_path),
        },
    )
    console.print(f"Installed [bold]{spec.name}[/bold] ({spec.dof} dof) at [cyan]{record.path}[/cyan]")
    console.print(f"Robot spec: [cyan]{spec_path}[/cyan]")


def _holosoma_root(args: argparse.Namespace, console: Console) -> Path:
    if args.holosoma_root is not None:
        return args.holosoma_root.expanduser().resolve()
    clone_root = args.store.resolve() / "downloads" / "holosoma"
    if clone_root.exists() and not args.force:
        return clone_root
    if clone_root.exists():
        shutil.rmtree(clone_root)
    clone_root.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Cloning Holosoma robot assets from [cyan]{args.repo}[/cyan]...")
    _sparse_clone(args.repo, args.ref, clone_root)
    return clone_root


def _sparse_clone(repo: str, ref: str, target: Path) -> None:
    clone_cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        "--branch",
        ref,
        repo,
        str(target),
    ]
    try:
        subprocess.run(clone_cmd, check=True)
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "set", *(str(path) for path in HOLOSOMA_G1_DIRS)],
            check=True,
        )
        _source_g1_dir(target)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo, str(target)], check=True)


def _source_g1_dir(source_root: Path) -> Path:
    for relative in HOLOSOMA_G1_DIRS:
        candidate = source_root / relative
        if candidate.exists():
            return candidate
    searched = ", ".join(str(source_root / relative) for relative in HOLOSOMA_G1_DIRS)
    raise FileNotFoundError(f"Holosoma G1 asset directory not found; searched: {searched}")


def _copy_robot_asset(source_dir: Path, target_dir: Path, *, source_root: Path, force: bool) -> None:
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)
    for source_name, target_name in (("LICENSE", "HOLOSOMA_LICENSE"), ("NOTICE", "HOLOSOMA_NOTICE")):
        source = source_root / source_name
        if source.exists():
            shutil.copy2(source, target_dir / target_name)
    mujoco_xml = target_dir / G1_MUJOCO_XML
    if mujoco_xml.exists():
        strip_floor_contact_pairs(mujoco_xml)


def _write_g1_robot_spec(asset_dir: Path, *, source_repo: str, source_ref: str) -> Path:
    urdf_path = asset_dir / G1_URDF
    xml_path = asset_dir / G1_MUJOCO_XML
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    joint_names, joint_limits = _parse_urdf_joints(urdf_path)
    link_names = _validate_links(urdf_path, G1_LINK_NAMES)
    spec_path = asset_dir / "robot.toml"
    spec_path.write_text(
        _g1_toml(
            joint_names=joint_names,
            link_names=link_names,
            joint_limits=joint_limits,
            source_repo=source_repo,
            source_ref=source_ref,
        )
    )
    RobotSpec.load(spec_path)
    return spec_path


def _parse_urdf_joints(urdf_path: Path) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    root = ET.parse(urdf_path).getroot()
    joint_names: list[str] = []
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        name = joint.attrib["name"]
        joint_names.append(name)
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            joint_limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return tuple(joint_names), joint_limits


def _validate_links(urdf_path: Path, link_names: tuple[str, ...]) -> tuple[str, ...]:
    root = ET.parse(urdf_path).getroot()
    urdf_links = {link.attrib["name"] for link in root.findall("link")}
    missing = sorted(set(link_names) - urdf_links)
    if missing:
        raise ValueError(f"Configured G1 links are missing from {urdf_path}: {missing}")
    return link_names


def _g1_toml(
    *,
    joint_names: tuple[str, ...],
    link_names: tuple[str, ...],
    joint_limits: dict[str, tuple[float, float]],
    source_repo: str,
    source_ref: str,
) -> str:
    lines: list[str] = [
        'name = "g1"',
        f"dof = {len(joint_names)}",
        f"height_m = {G1_HEIGHT_M}",
        f"urdf_path = {json.dumps(G1_URDF)}",
        f"mujoco_xml_path = {json.dumps(G1_MUJOCO_XML)}",
        "",
        f"joint_names = {_toml_array(joint_names)}",
        f"link_names = {_toml_array(link_names)}",
        f"contact_links = {_toml_array(G1_CONTACT_LINKS)}",
        f"nominal_tracking_joints = {_toml_array(G1_NOMINAL_TRACKING_JOINTS)}",
        "",
        "[joint_limits]",
    ]
    for name in joint_names:
        lower, upper = joint_limits[name]
        lines.append(f"{name} = [{lower:.12g}, {upper:.12g}]")
    lines.extend(["", "[joint_roles]"])
    lines.extend(
        f"{role} = {json.dumps(joint)}"
        for role, joint in G1_JOINT_ROLES.items()
    )
    lines.extend(["", "[link_roles]"])
    lines.extend(
        f"{role} = {json.dumps(link)}"
        for role, link in G1_LINK_ROLES.items()
    )
    lines.extend(
        [
            "",
            "[metadata]",
            'asset_source = "holosoma"',
            f"source_repository = {json.dumps(source_repo)}",
            f"source_ref = {json.dumps(source_ref)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _toml_array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        Console(stderr=True).print(f"[red]robot asset bootstrap failed:[/red] {exc}")
        sys.exit(1)
