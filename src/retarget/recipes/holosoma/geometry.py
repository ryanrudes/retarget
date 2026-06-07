"""Geometry-pair policy for Holosoma-compatible object collision."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from retarget.optimization.spec import GeometryPair
from retarget.robots.spec import AnyRobotSpec

from .vocabulary import HolosomaGeometryName


def object_non_penetration_geometry_pairs(
    robot: AnyRobotSpec,
    *,
    fixture_dir: str | Path,
    include_ground: bool = True,
) -> tuple[GeometryPair, ...]:
    """Return static geometry-pair candidates for Holosoma object/ground non-penetration."""

    fixture = Path(fixture_dir)
    object_names = included_geom_names(fixture / "box_body.xml")
    expected_names = tuple(
        geometry.value
        for geometry in (
            HolosomaGeometryName.MULTI_BOX_1,
            HolosomaGeometryName.MULTI_BOX_2,
            HolosomaGeometryName.MULTI_BOX_3,
        )
    )
    if object_names != expected_names:
        raise ValueError(
            "Holosoma box geometry differs from the built-in typed vocabulary: "
            f"expected {expected_names}, received {object_names}"
        )
    scene_geometries: tuple[HolosomaGeometryName, ...] = (
        HolosomaGeometryName.MULTI_BOX_1,
        HolosomaGeometryName.MULTI_BOX_2,
        HolosomaGeometryName.MULTI_BOX_3,
    )
    if include_ground:
        scene_geometries = (*scene_geometries, HolosomaGeometryName.GROUND)
    return tuple(
        GeometryPair(first=robot_geometry, second=scene_geometry)
        for robot_geometry in robot.geometries
        for scene_geometry in scene_geometries
    )


def included_geom_names(xml_path: Path) -> tuple[str, ...]:
    """Return every explicitly named geometry in an included scene body."""

    if not xml_path.exists():
        return ()
    root = ET.parse(xml_path).getroot()
    return tuple(
        str(elem.attrib["name"])
        for elem in root.iter("geom")
        if "name" in elem.attrib
    )
