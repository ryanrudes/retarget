"""Helpers for preparing and resolving external MuJoCo models."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path


def strip_floor_contact_pairs(xml_path: Path) -> bool:
    """Remove foot-floor contact pairs when the floor geom is defined in a scene file.

    Holosoma ships ``g1_29dof.xml`` with ``<pair geom2="floor" .../>`` entries, but the
    ``floor`` geom only exists in ``scenes/scene_g1_29dof_wbt_plane.xml``. Kinematics-only
    loads use the robot XML directly, so those pairs must be removed.

    Returns:
        True if the file was modified.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    contact = root.find("contact")
    if contact is None:
        return False

    removed = False
    for pair in list(contact.findall("pair")):
        if pair.attrib.get("geom1") == "floor" or pair.attrib.get("geom2") == "floor":
            contact.remove(pair)
            removed = True

    if not list(contact):
        root.remove(contact)
        removed = True

    if not removed:
        return False

    tree.write(xml_path, encoding="unicode", xml_declaration=True)
    return True


def resolve_mujoco_body_name(
    mujoco: object,
    model: object,
    name: str,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Map a robot link name to a MuJoCo body name when they differ."""

    body_obj = mujoco.mjtObj.mjOBJ_BODY  # type: ignore[attr-defined]
    if mujoco.mj_name2id(model, body_obj, name) >= 0:  # type: ignore[attr-defined]
        return name
    alias = (aliases or {}).get(name)
    if alias is not None and mujoco.mj_name2id(model, body_obj, alias) >= 0:  # type: ignore[attr-defined]
        return alias
    return None


def build_mujoco_body_name_map(
    mujoco: object,
    model: object,
    names: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return link/body names that resolve in the loaded MuJoCo model."""

    mapping: dict[str, str] = {}
    for name in names:
        if not name or name in mapping:
            continue
        resolved = resolve_mujoco_body_name(mujoco, model, name, aliases)
        if resolved is not None:
            mapping[name] = resolved
    return mapping
