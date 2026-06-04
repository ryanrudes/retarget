"""Tests for MuJoCo robot XML helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from retarget.kinematics.mujoco_xml import (
    build_mujoco_body_name_map,
    strip_floor_contact_pairs,
)

pytest.importorskip("mujoco")


def test_strip_floor_contact_pairs_allows_kinematics_load(tmp_path: Path) -> None:
    import mujoco

    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
  <worldbody>
    <body name="root">
      <freejoint name="root"/>
      <geom name="foot_collision" type="sphere" size="0.05" pos="0 0 0"/>
    </body>
  </worldbody>
  <contact>
    <pair name="foot_floor" geom1="foot_collision" geom2="floor"/>
  </contact>
</mujoco>
"""
    )
    with pytest.raises(ValueError, match="floor"):
        mujoco.MjModel.from_xml_path(str(xml_path))

    assert strip_floor_contact_pairs(xml_path) is True
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nbody >= 1
    assert strip_floor_contact_pairs(xml_path) is False


def test_build_mujoco_body_name_map_resolves_g1_aliases(tmp_path: Path) -> None:
    import mujoco

    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
  <worldbody>
    <body name="pelvis">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <freejoint name="root"/>
      <body name="torso_link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist" type="hinge" axis="0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mapping = build_mujoco_body_name_map(
        mujoco,
        model,
        ("pelvis_contour_link", "head_link", "torso_link"),
    )
    assert mapping["pelvis_contour_link"] == "pelvis"
    assert mapping["head_link"] == "torso_link"
    assert mapping["torso_link"] == "torso_link"
