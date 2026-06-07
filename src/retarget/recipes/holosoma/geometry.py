"""Geometry-pair policy for Holosoma-compatible object collision."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from retarget.robots.spec import RobotSpec

from .vocabulary import HolosomaGeometryName


def object_non_penetration_geometry_pairs(
    robot: RobotSpec,
    *,
    fixture_dir: str | Path,
    object_name: HolosomaGeometryName = HolosomaGeometryName.MULTI_BOXES,
    include_ground: bool = True,
) -> tuple[tuple[str, str], ...]:
    """Return static geometry-pair candidates for Holosoma object/ground non-penetration."""

    fixture = Path(fixture_dir)
    object_geoms = included_geom_names(fixture / "box_body.xml", prefix=object_name.value)
    ground_geoms = (HolosomaGeometryName.GROUND.value,) if include_ground else ()
    scene_geoms = object_geoms + ground_geoms
    robot_geoms = robot.geometry_names or tuple(robot.link_names)
    robot_geoms = tuple(
        name for name in robot_geoms if name not in scene_geoms and not name.startswith(object_name.value)
    )
    return tuple((robot_geom, scene_geom) for robot_geom in robot_geoms for scene_geom in scene_geoms)


def included_geom_names(xml_path: Path, *, prefix: str) -> tuple[str, ...]:
    """Return included scene geometry names matching ``prefix``."""

    if not xml_path.exists():
        return ()
    root = ET.parse(xml_path).getroot()
    return tuple(
        str(elem.attrib["name"])
        for elem in root.iter("geom")
        if "name" in elem.attrib and str(elem.attrib["name"]).startswith(prefix)
    )
