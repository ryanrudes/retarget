"""G1 spherehand robot specification for Holosoma-compatible recipes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from retarget.robots.spec import RobotSpec, RobotVocabulary

from .layout import default_holosoma_root, holosoma_g1_robot_dir
from .vocabulary import (
    G1_FOOT_STICKING_LINKS,
    G1_LEFT_FOOT_STICKING_LINKS,
    G1_RIGHT_FOOT_STICKING_LINKS,
    G1SpherehandGeometry,
    G1SpherehandJoint,
    G1SpherehandLink,
    HolosomaRobotRole,
)

_G1_EXPECTED_DOF = 29
_G1_MANUAL_LOWER_QPOS = {
    3: -1.0,
    4: -1.0,
    5: -1.0,
    6: -1.0,
    20: -0.3,
    21: -0.1,
    26: -0.1,
    27: -0.1,
    28: -0.05,
    33: -0.1,
    34: -0.1,
    35: -0.05,
}
_G1_MANUAL_UPPER_QPOS = {
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    20: 0.3,
    25: 1.4,
    26: 0.2,
    27: 0.3,
    28: 0.05,
    32: 1.4,
    33: 0.2,
    34: 0.3,
    35: 0.05,
}


def g1_spherehand_robot(
    holosoma_root: str | Path | None = None,
    *,
    scene_xml_path: str | Path | None = None,
    include_object_collision: bool = False,
    height_m: float = 1.32,
) -> RobotSpec[G1SpherehandJoint, G1SpherehandLink, G1SpherehandGeometry, HolosomaRobotRole]:
    """Return a G1 spherehand robot spec matching Holosoma's MOCAP task."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    robot_dir = holosoma_g1_robot_dir(root)
    urdf_path = robot_dir / "g1_29dof_spherehand.urdf"
    robot_xml_path = robot_dir / "g1_29dof_spherehand.xml"
    xml_path = Path(scene_xml_path).resolve() if scene_xml_path is not None else robot_xml_path
    for path in (urdf_path, robot_xml_path, xml_path):
        if not path.exists():
            raise FileNotFoundError(path)
    joint_names = tuple(mjcf_joint_names(robot_xml_path))
    _validate_model_vocabulary(
        "joints",
        joint_names,
        tuple(joint.value for joint in G1SpherehandJoint),
        ordered=True,
    )
    _validate_model_vocabulary(
        "bodies",
        tuple(mjcf_body_names(robot_xml_path)),
        tuple(link.value for link in G1SpherehandLink),
    )
    _validate_model_vocabulary(
        "geometries",
        tuple(mjcf_geom_names(robot_xml_path)),
        tuple(geometry.value for geometry in G1SpherehandGeometry),
    )
    joint_limits_by_name = apply_manual_qpos_bounds(
        mjcf_joint_limits(robot_xml_path, joint_names),
        joint_names=joint_names,
    )
    return RobotSpec(
        name="holosoma_g1_29dof_spherehand",
        height_m=height_m,
        vocabulary=RobotVocabulary(
            joints=G1SpherehandJoint,
            links=G1SpherehandLink,
            geometries=G1SpherehandGeometry,
            roles=HolosomaRobotRole,
        ),
        joints=tuple(G1SpherehandJoint),
        links=tuple(G1SpherehandLink),
        contact_links=G1_FOOT_STICKING_LINKS,
        joint_limits={G1SpherehandJoint(name): bounds for name, bounds in joint_limits_by_name.items()},
        link_roles={
            HolosomaRobotRole.PELVIS: G1SpherehandLink.PELVIS_CONTOUR,
            HolosomaRobotRole.LEFT_HIP: G1SpherehandLink.LEFT_HIP_PITCH,
            HolosomaRobotRole.LEFT_KNEE: G1SpherehandLink.LEFT_KNEE,
            HolosomaRobotRole.LEFT_TOE: G1SpherehandLink.LEFT_ANKLE_ROLL_SPHERE_5,
            HolosomaRobotRole.RIGHT_HIP: G1SpherehandLink.RIGHT_HIP_PITCH,
            HolosomaRobotRole.RIGHT_KNEE: G1SpherehandLink.RIGHT_KNEE,
            HolosomaRobotRole.RIGHT_TOE: G1SpherehandLink.RIGHT_ANKLE_ROLL_SPHERE_5,
            HolosomaRobotRole.LEFT_SHOULDER: G1SpherehandLink.LEFT_SHOULDER_ROLL,
            HolosomaRobotRole.LEFT_ELBOW: G1SpherehandLink.LEFT_ELBOW,
            HolosomaRobotRole.LEFT_HAND: G1SpherehandLink.LEFT_SPHERE_HAND,
            HolosomaRobotRole.RIGHT_SHOULDER: G1SpherehandLink.RIGHT_SHOULDER_ROLL,
            HolosomaRobotRole.RIGHT_ELBOW: G1SpherehandLink.RIGHT_ELBOW,
            HolosomaRobotRole.RIGHT_HAND: G1SpherehandLink.RIGHT_SPHERE_HAND,
            HolosomaRobotRole.LEFT_FOOT: G1SpherehandLink.LEFT_ANKLE_INTERMEDIATE_1,
            HolosomaRobotRole.RIGHT_FOOT: G1SpherehandLink.RIGHT_ANKLE_INTERMEDIATE_1,
        },
        link_groups={
            HolosomaRobotRole.LEFT_FOOT_CONTACT: G1_LEFT_FOOT_STICKING_LINKS,
            HolosomaRobotRole.RIGHT_FOOT_CONTACT: G1_RIGHT_FOOT_STICKING_LINKS,
        },
        mujoco_body_aliases={
            G1SpherehandLink.PELVIS_CONTOUR: G1SpherehandLink.PELVIS.value,
            G1SpherehandLink.HEAD: G1SpherehandLink.TORSO.value,
        },
        geometries=tuple(G1SpherehandGeometry),
        urdf_path=urdf_path,
        mujoco_xml_path=xml_path if include_object_collision or scene_xml_path is not None else robot_xml_path,
        provenance={
            "source": "holosoma",
        }
    )


def _validate_model_vocabulary(
    label: str,
    actual: tuple[str, ...],
    expected: tuple[str, ...],
    *,
    ordered: bool = False,
) -> None:
    matches = actual == expected if ordered else set(actual) == set(expected)
    if not matches:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"G1 spherehand {label} differ from the built-in typed vocabulary; "
            f"missing={missing}, unexpected={unexpected}"
        )


def ensure_g1_model_assets(holosoma_root: str | Path | None = None) -> tuple[Path, ...]:
    """Create Holosoma ``models/g1`` symlinks needed by fixture scene XML files."""

    root = Path(holosoma_root) if holosoma_root is not None else default_holosoma_root()
    root = root.resolve()
    source_robot = holosoma_g1_robot_dir(root)
    if not (root / "tests" / "fixtures" / "climb_seq_0").exists():
        return ()
    models_g1 = root / "models" / "g1"
    models_g1.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name in ("meshes", "assets"):
        source = source_robot / name
        target = models_g1 / name
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            continue
        target.symlink_to(source, target_is_directory=True)
        created.append(target)
    return tuple(created)


def mjcf_joint_names(xml_path: Path) -> list[str]:
    """Return actuated joint names from a G1 MJCF file."""

    root = ET.parse(xml_path).getroot()
    names = [
        str(elem.attrib["name"])
        for elem in root.iter("joint")
        if elem.attrib.get("type") != "free" and "name" in elem.attrib
    ]
    if len(names) != _G1_EXPECTED_DOF:
        raise ValueError(f"{xml_path} contains {len(names)} named actuated joints, expected {_G1_EXPECTED_DOF}")
    return names


def mjcf_body_names(xml_path: Path) -> list[str]:
    """Return body names from a MJCF file."""

    root = ET.parse(xml_path).getroot()
    return [str(elem.attrib["name"]) for elem in root.iter("body") if "name" in elem.attrib]


def mjcf_geom_names(xml_path: Path) -> list[str]:
    """Return geometry names from a MJCF file."""

    root = ET.parse(xml_path).getroot()
    return [str(elem.attrib["name"]) for elem in root.iter("geom") if "name" in elem.attrib]


def mjcf_joint_limits(xml_path: Path, joint_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Return joint limits keyed by joint name from MJCF ranges."""

    root = ET.parse(xml_path).getroot()
    valid = set(joint_names)
    out: dict[str, tuple[float, float]] = {}
    for elem in root.iter("joint"):
        name = elem.attrib.get("name")
        raw_range = elem.attrib.get("range")
        if name not in valid or raw_range is None:
            continue
        lower, upper = (float(value) for value in raw_range.split())
        out[str(name)] = (lower, upper)
    return out


def apply_manual_qpos_bounds(
    joint_limits: Mapping[str, tuple[float, float]],
    *,
    joint_names: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    """Apply Holosoma's manual qpos bounds to named joint limits."""

    out = dict(joint_limits)
    for qpos_idx, lower in _G1_MANUAL_LOWER_QPOS.items():
        joint_name = joint_name_for_qpos(qpos_idx, joint_names)
        if joint_name is None:
            continue
        _old_lower, old_upper = out.get(joint_name, (-1e6, 1e6))
        out[joint_name] = (float(lower), old_upper)
    for qpos_idx, upper in _G1_MANUAL_UPPER_QPOS.items():
        joint_name = joint_name_for_qpos(qpos_idx, joint_names)
        if joint_name is None:
            continue
        old_lower, _old_upper = out.get(joint_name, (-1e6, 1e6))
        out[joint_name] = (old_lower, float(upper))
    return out


def joint_name_for_qpos(qpos_idx: int, joint_names: tuple[str, ...]) -> str | None:
    """Return the actuated joint name for a full-qpos index."""

    joint_idx = int(qpos_idx) - 7
    if joint_idx < 0 or joint_idx >= len(joint_names):
        return None
    return joint_names[joint_idx]
