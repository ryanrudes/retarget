"""Filesystem layout discovery for Holosoma fixtures and assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HolosomaClimbLayout:
    """Resolved file layout for the MOCAP climbing fixture."""

    fixture_dir: Path
    motion_path: Path
    object_mesh_path: Path
    object_urdf_path: Path
    scene_xml_path: Path


def default_holosoma_root() -> Path:
    """Return the sibling Holosoma checkout path used by local parity tests."""

    return Path(__file__).resolve().parents[5] / "holosoma"


def holosoma_climb_layout(root: Path) -> HolosomaClimbLayout:
    """Resolve the climb fixture across known Holosoma checkout layouts."""

    fixture_candidates = (
        root / "src" / "holosoma_retargeting" / "holosoma_retargeting" / "demo_data" / "climb" / "mocap_climb_seq_0",
        root / "demo_data" / "climb" / "mocap_climb_seq_0",
        root / "tests" / "fixtures" / "climb_seq_0",
    )
    fixture_dir = next((path for path in fixture_candidates if path.exists()), None)
    if fixture_dir is None:
        candidates = "\n".join(str(path) for path in fixture_candidates)
        raise FileNotFoundError(f"Could not find Holosoma climb fixture. Checked:\n{candidates}")

    motion_files = sorted(fixture_dir.glob("mocap_climb_seq_0_joint_positions*.npy"))
    if not motion_files:
        raise FileNotFoundError(f"No mocap_climb_seq_0_joint_positions*.npy file found in {fixture_dir}")

    layout = HolosomaClimbLayout(
        fixture_dir=fixture_dir,
        motion_path=motion_files[0],
        object_mesh_path=fixture_dir / "multi_boxes.obj",
        object_urdf_path=fixture_dir / "multi_boxes.urdf",
        scene_xml_path=fixture_dir / "g1_29dof_spherehand_w_multi_boxes.xml",
    )
    for path in (layout.object_mesh_path, layout.object_urdf_path, layout.scene_xml_path):
        if not path.exists():
            raise FileNotFoundError(path)
    return layout


def holosoma_g1_robot_dir(root: Path) -> Path:
    """Resolve the G1 spherehand asset directory across known layouts."""

    candidates = (
        root / "src" / "holosoma_retargeting" / "holosoma_retargeting" / "models" / "g1",
        root / "models" / "g1",
        root / "src" / "interaction_mesh_retarget" / "robots" / "g1",
    )
    for candidate in candidates:
        if (candidate / "g1_29dof_spherehand.urdf").exists() and (candidate / "g1_29dof_spherehand.xml").exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Holosoma G1 spherehand assets. Checked:\n{checked}")
