"""Object assets and trajectories for Holosoma-compatible recipes."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import numpy as np

from retarget.scene.spec import ObjectVisualPart


def preprocess_object_poses(object_poses: np.ndarray, *, scale: float) -> np.ndarray:
    """Apply Holosoma's object pose scaling rule to ``[qw, qx, qy, qz, x, y, z]`` poses."""

    poses = np.asarray(object_poses, dtype=np.float64).copy()
    poses[:, -3:-1] *= float(scale)
    z0 = float(poses[0, -1])
    poses[:, -1] = z0 + (poses[:, -1] - z0) * float(scale)
    return poses


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert Holosoma object poses from ``[qw, qx, qy, qz, x, y, z]`` to ``[x, y, z, qw, qx, qy, qz]``."""

    return np.asarray(object_poses, dtype=np.float64)[:, [4, 5, 6, 0, 1, 2, 3]]


def dummy_object_poses(frame_count: int) -> np.ndarray:
    """Return identity Holosoma object poses in ``[qw, qx, qy, qz, x, y, z]`` order."""

    poses = np.zeros((int(frame_count), 7), dtype=np.float64)
    poses[:, 0] = 1.0
    return poses


def sample_multi_boxes_like_holosoma(
    mesh_path: str | Path,
    *,
    sample_count: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Sample the climbing multi-box mesh with Holosoma's weighted surface rule."""

    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install retarget[mujoco] or trimesh to sample Holosoma multi-box surfaces") from exc
    mesh = cast(Any, trimesh.load(str(mesh_path), force="mesh"))
    rng = np.random.default_rng(seed)
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    face_centers = triangles.mean(axis=1)
    weights = np.where(face_centers[:, 2] > 0.9, 20.0, 1.0)
    probs = face_areas * weights
    probs = probs / probs.sum()
    sampled_face_indices = rng.choice(len(faces), size=int(sample_count), p=probs)
    samples = np.zeros((int(sample_count), 3), dtype=np.float64)
    for idx, face_idx in enumerate(sampled_face_indices):
        v1, v2, v3 = vertices[faces[face_idx]]
        r1, r2 = rng.random(2)
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        samples[idx] = v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
    return samples


def ensure_scaled_multi_boxes_assets(
    *,
    fixture_dir: str | Path,
    object_urdf_path: str | Path,
    scene_xml_path: str | Path,
    asset_scale: tuple[float, float, float],
) -> tuple[Path, Path]:
    """Create Holosoma-style scaled multi-box URDF and scene XML assets."""

    fixture = Path(fixture_dir)
    source_urdf = Path(object_urdf_path)
    source_scene_xml = Path(scene_xml_path)
    source_box_assets = fixture / "box_assets.xml"
    for path in (source_urdf, source_scene_xml, source_box_assets):
        if not path.exists():
            raise FileNotFoundError(path)

    sx, sy, sz = (float(v) for v in asset_scale)
    suffix = f"{sx:.2f}_{sy:.2f}_{sz:.2f}"
    scaled_urdf = source_urdf.with_name(f"{source_urdf.stem}_scaled_{suffix}{source_urdf.suffix}")
    scaled_box_assets = source_box_assets.with_name(
        f"{source_box_assets.stem}_scaled_{suffix}{source_box_assets.suffix}"
    )
    scaled_scene_xml = source_scene_xml.with_name(f"{source_scene_xml.stem}_scaled_{suffix}{source_scene_xml.suffix}")

    _write_xml_with_replaced_scale(source_urdf, scaled_urdf, asset_scale=asset_scale)
    _write_xml_with_replaced_scale(source_box_assets, scaled_box_assets, asset_scale=asset_scale)
    _write_scene_with_box_asset_include(source_scene_xml, scaled_scene_xml, include_name=scaled_box_assets.name)
    return scaled_urdf, scaled_scene_xml


def object_visual_parts_from_urdf(urdf_path: str | Path) -> tuple[ObjectVisualPart, ...]:
    """Read visual mesh parts from a Holosoma multi-box URDF."""

    path = Path(urdf_path)
    if not path.exists():
        return ()
    root = ET.parse(path).getroot()
    parts: list[ObjectVisualPart] = []
    for link in root.iter("link"):
        link_name = link.attrib.get("name", "object")
        for visual_idx, visual in enumerate(link.findall("visual")):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh_path = (path.parent / filename).resolve()
            scale = cast(tuple[float, float, float] | None, _xml_float_tuple(mesh.attrib.get("scale"), expected=3))
            rgba = _visual_rgba(visual)
            part_name = Path(filename).stem or f"{link_name}_{visual_idx}"
            parts.append(ObjectVisualPart(name=part_name, mesh_path=mesh_path, asset_scale=scale, rgba=rgba))
    return tuple(parts)


def _write_xml_with_replaced_scale(source: Path, destination: Path, *, asset_scale: tuple[float, float, float]) -> None:
    sx, sy, sz = (float(v) for v in asset_scale)
    content = source.read_text()
    replacement = f'scale="{sx} {sy} {sz}"'
    content = re.sub(r'scale="[^"]*"', replacement, content)
    if destination.exists() and destination.read_text() == content:
        return
    destination.write_text(content)


def _write_scene_with_box_asset_include(source: Path, destination: Path, *, include_name: str) -> None:
    content = source.read_text()
    content = re.sub(r'file="box_assets\.xml"', f'file="{include_name}"', content)
    if destination.exists() and destination.read_text() == content:
        return
    destination.write_text(content)


def _visual_rgba(visual: ET.Element) -> tuple[float, float, float, float] | None:
    color = visual.find("./material/color")
    if color is None:
        return None
    return cast(tuple[float, float, float, float] | None, _xml_float_tuple(color.attrib.get("rgba"), expected=4))


def _xml_float_tuple(value: str | None, *, expected: int) -> tuple[float, ...] | None:
    if not value:
        return None
    values = tuple(float(item) for item in value.split())
    if len(values) != expected:
        raise ValueError(f"expected {expected} float values, got {len(values)}")
    return values
