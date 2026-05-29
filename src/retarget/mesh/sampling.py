"""Deterministic point sampling for scene meshes."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retarget.core.array import FloatArray, as_float_array


def sample_mesh_points(path: str | Path, *, count: int = 128) -> FloatArray:
    """Return deterministic surface samples for a mesh file.

    OBJ files are parsed without optional dependencies. Other mesh formats use
    `trimesh` when it is installed.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        vertices, faces = _load_obj(mesh_path)
    else:
        vertices, faces = _load_with_trimesh(mesh_path)
    return _sample_geometry(vertices, faces, count=count)


def _sample_geometry(vertices: FloatArray, faces: NDArray[np.int_], *, count: int) -> FloatArray:
    points = as_float_array(vertices, shape_tail=(3,), name="vertices")
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("mesh vertices must have shape (N, 3) with N > 0")
    triangles = np.asarray(faces, dtype=int)
    if triangles.size == 0:
        return _sample_rows(points, count=count)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh faces must have shape (N, 3)")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("mesh faces reference missing vertices")
    return _sample_triangles(points, triangles, count=count)


def _load_obj(path: Path) -> tuple[FloatArray, NDArray[np.int_]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex line in {path} has fewer than 3 coordinates")
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[0] == "f":
            indices = [_parse_obj_face_index(token, vertex_count=len(vertices)) for token in fields[1:]]
            if len(indices) < 3:
                continue
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=int).reshape((-1, 3))


def _parse_obj_face_index(token: str, *, vertex_count: int) -> int:
    raw_index = int(token.split("/", 1)[0])
    if raw_index == 0:
        raise ValueError("OBJ face indices are 1-based; 0 is invalid")
    if raw_index < 0:
        return vertex_count + raw_index
    return raw_index - 1


def _load_with_trimesh(path: Path) -> tuple[FloatArray, NDArray[np.int_]]:
    try:
        trimesh = importlib.import_module("trimesh")
    except ImportError as exc:
        raise ImportError(
            f"Sampling {path.suffix or 'this mesh'} files requires the optional trimesh dependency"
        ) from exc
    loaded: Any = trimesh.load_mesh(str(path), process=False)
    mesh = _as_trimesh_geometry(loaded)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=int).reshape((-1, 3))


def _as_trimesh_geometry(loaded: Any) -> Any:
    if hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
        return loaded
    if hasattr(loaded, "dump"):
        dumped = loaded.dump(concatenate=True)
        if hasattr(dumped, "vertices") and hasattr(dumped, "faces"):
            return dumped
    raise ValueError("loaded mesh does not expose vertices and triangular faces")


def _sample_rows(points: FloatArray, *, count: int) -> FloatArray:
    if len(points) >= count:
        indices = np.linspace(0, len(points) - 1, num=count, dtype=int)
    else:
        indices = np.arange(count, dtype=int) % len(points)
    return np.asarray(points[indices], dtype=np.float64)


def _sample_triangles(vertices: FloatArray, faces: NDArray[np.int_], *, count: int) -> FloatArray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    positive = areas > 0.0
    if not np.any(positive):
        return _sample_rows(vertices, count=count)

    triangles = triangles[positive]
    areas = areas[positive]
    cumulative = np.cumsum(areas)
    targets = (np.arange(count, dtype=np.float64) + 0.5) * (float(cumulative[-1]) / count)
    face_indices = np.searchsorted(cumulative, targets, side="right")
    face_indices = np.clip(face_indices, 0, len(triangles) - 1)

    # Irrational strides give deterministic coverage without introducing RNG state.
    u = np.mod(np.arange(count, dtype=np.float64) * 0.7548776662466927, 1.0)
    v = np.mod(np.arange(count, dtype=np.float64) * 0.5698402909980532, 1.0)
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]

    selected = triangles[face_indices]
    a = selected[:, 0]
    samples = a + u[:, None] * (selected[:, 1] - a) + v[:, None] * (selected[:, 2] - a)
    return np.asarray(samples, dtype=np.float64)
