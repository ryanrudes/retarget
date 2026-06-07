"""Interaction mesh construction and Laplacian geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.spatial import Delaunay

from retarget.core.array import FloatArray, as_float_array


class MeshTopology(StrEnum):
    """Interaction mesh topology policy.

    Attributes:
        DELAUNAY (str): Delaunay triangulation over mesh sites.
        CHAIN (str): Open chain along ordered sites.
        COMPLETE (str): Fully connected graph.
        K_NEAREST (str): k-nearest-neighbor edges (uses ``k_neighbors``).
    """

    DELAUNAY = "delaunay"
    CHAIN = "chain"
    COMPLETE = "complete"
    K_NEAREST = "k_nearest"


class LaplacianWeighting(StrEnum):
    """Neighbor weighting policy for Laplacian coordinates.

    Attributes:
        UNIFORM (str): Average all neighbors equally. This matches Holosoma's default.
        INVERSE_DISTANCE (str): Weight closer neighbors more strongly.
    """

    UNIFORM = "uniform"
    INVERSE_DISTANCE = "inverse_distance"


class InteractionMeshSpec(BaseModel):
    """Configuration for interaction mesh construction.

    Attributes:
        topology (MeshTopology): Graph construction policy.
        k_neighbors (int): Neighbor count when ``topology`` is ``K_NEAREST`` or Delaunay fallback.
        laplacian_weighting (LaplacianWeighting): Neighbor weighting used by Laplacian objectives.
        laplacian_epsilon (float): Small positive value used by distance-weighted Laplacians.
    """

    model_config = ConfigDict(extra="forbid")

    topology: MeshTopology = MeshTopology.DELAUNAY
    k_neighbors: int = 4
    laplacian_weighting: LaplacianWeighting = LaplacianWeighting.UNIFORM
    laplacian_epsilon: float = 1e-6

    @field_validator("k_neighbors")
    @classmethod
    def _positive_neighbors(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("k_neighbors must be positive")
        return value

    @field_validator("laplacian_epsilon")
    @classmethod
    def _positive_epsilon(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("laplacian_epsilon must be positive")
        return float(value)


@dataclass(frozen=True)
class InteractionMesh:
    """A point set plus simplex connectivity.

    Attributes:
        vertices (FloatArray): Mesh sites with shape ``(V, 3)``.
        simplices (NDArray[np.int_]): Edge or triangle vertex indices into ``vertices``.
    """

    vertices: FloatArray
    simplices: NDArray[np.int_]
    laplacian_weighting: LaplacianWeighting = LaplacianWeighting.UNIFORM
    laplacian_epsilon: float = 1e-6

    @property
    def adjacency(self) -> list[list[int]]:
        """Adjacency list induced by simplices."""

        return adjacency_from_simplices(self.simplices, len(self.vertices))

    @property
    def laplacian(self) -> FloatArray:
        """Dense row-normalized Laplacian matrix."""

        return laplacian_matrix(
            self.vertices,
            self.adjacency,
            weighting=self.laplacian_weighting,
            epsilon=self.laplacian_epsilon,
        )

    def laplacian_coordinates(self) -> FloatArray:
        """Laplacian coordinates for all vertices."""

        return laplacian_coordinates(
            self.vertices,
            self.adjacency,
            weighting=self.laplacian_weighting,
            epsilon=self.laplacian_epsilon,
        )


class InteractionMeshBuilder:
    """Build an interaction mesh from human and object/terrain points."""

    def __init__(
        self,
        spec: InteractionMeshSpec | None = None,
        *,
        topology: MeshTopology | str | None = None,
        k_neighbors: int | None = None,
    ) -> None:
        """Configure mesh topology for subsequent :meth:`build` calls.

        Args:
            spec (InteractionMeshSpec | None): Full mesh configuration; mutually exclusive with keyword overrides.
            topology (MeshTopology | str | None): Override topology when ``spec`` is omitted.
            k_neighbors (int | None): Neighbor count for ``K_NEAREST`` topology.
        Raises:
            ValueError: If ``spec`` is combined with explicit mesh keyword options.
        """
        if spec is not None and (topology is not None or k_neighbors is not None):
            raise ValueError("Pass either spec or explicit mesh options, not both")
        if spec is not None:
            self.spec = spec
            return
        resolved_topology = MeshTopology.DELAUNAY
        if topology is not None:
            resolved_topology = MeshTopology(topology)
        self.spec = InteractionMeshSpec(
            topology=resolved_topology,
            k_neighbors=k_neighbors if k_neighbors is not None else 4,
        )

    def build(self, human_points: FloatArray, environment_points: FloatArray | None = None) -> InteractionMesh:
        """Build an interaction mesh from human and optional environment points.

        Args:
            human_points (FloatArray): Body or joint sample sites with shape ``(N, 3)``.
            environment_points (FloatArray | None): Object or terrain sites stacked after human points.

        Returns:
            InteractionMesh: Combined vertex set and simplex connectivity.

        Raises:
            ValueError: If point arrays have invalid rank or shape.
        """

        human = as_float_array(human_points, shape_tail=(3,), name="human_points")
        if human.ndim != 2:
            raise ValueError("human_points must have shape (N, 3)")
        if environment_points is None:
            vertices = human
        else:
            env = as_float_array(environment_points, shape_tail=(3,), name="environment_points")
            if env.ndim != 2:
                raise ValueError("environment_points must have shape (N, 3)")
            vertices = np.vstack([human, env])

        simplices = _simplices_for_topology(vertices, self.spec)
        return InteractionMesh(
            vertices=vertices,
            simplices=simplices,
            laplacian_weighting=self.spec.laplacian_weighting,
            laplacian_epsilon=self.spec.laplacian_epsilon,
        )


def _chain_simplices(n_vertices: int) -> NDArray[np.int_]:
    return np.asarray([[i, i + 1] for i in range(max(n_vertices - 1, 0))], dtype=int)


def _simplices_for_topology(vertices: FloatArray, spec: InteractionMeshSpec) -> NDArray[np.int_]:
    n_vertices = len(vertices)
    if n_vertices < 2:
        return np.zeros((0, 2), dtype=int)
    if spec.topology == MeshTopology.CHAIN:
        return _chain_simplices(n_vertices)
    if spec.topology == MeshTopology.COMPLETE:
        return _complete_graph_simplices(n_vertices)
    if spec.topology == MeshTopology.K_NEAREST:
        return _k_nearest_simplices(vertices, spec.k_neighbors)
    if n_vertices >= 4:
        try:
            return np.asarray(Delaunay(vertices).simplices, dtype=int)
        except Exception:
            return _k_nearest_simplices(vertices, min(spec.k_neighbors, n_vertices - 1))
    return _complete_graph_simplices(n_vertices)


def _complete_graph_simplices(n_vertices: int) -> NDArray[np.int_]:
    return np.asarray(
        [[src, dst] for src in range(n_vertices) for dst in range(src + 1, n_vertices)],
        dtype=int,
    )


def _k_nearest_simplices(vertices: FloatArray, k_neighbors: int) -> NDArray[np.int_]:
    n_vertices = len(vertices)
    k = min(k_neighbors, max(n_vertices - 1, 0))
    if k == 0:
        return np.zeros((0, 2), dtype=int)
    edges: set[tuple[int, int]] = set()
    for src in range(n_vertices):
        distances = np.linalg.norm(vertices - vertices[src], axis=1)
        order = np.argsort(distances)
        for dst in order[1 : k + 1]:
            a, b = sorted((src, int(dst)))
            edges.add((a, b))
    return np.asarray(sorted(edges), dtype=int)


def adjacency_from_simplices(simplices: NDArray[np.int_], n_vertices: int) -> list[list[int]]:
    """Build adjacency from any simplex arity."""

    adjacency = [set[int]() for _ in range(n_vertices)]
    for simplex in simplices:
        values = [int(v) for v in simplex if 0 <= int(v) < n_vertices]
        for i, src in enumerate(values):
            for dst in values[i + 1 :]:
                adjacency[src].add(dst)
                adjacency[dst].add(src)
    return [sorted(values) for values in adjacency]


def laplacian_matrix(
    vertices: FloatArray,
    adjacency: list[list[int]],
    *,
    weighting: LaplacianWeighting | str = LaplacianWeighting.UNIFORM,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Return a dense row-normalized Laplacian matrix."""

    weighting_policy = LaplacianWeighting(weighting)
    n_vertices = len(vertices)
    if len(adjacency) != n_vertices:
        raise ValueError("adjacency length must match vertices")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    matrix = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for i, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        matrix[i, i] = 1.0
        if weighting_policy == LaplacianWeighting.UNIFORM:
            weights = np.full(len(neighbors), 1.0 / len(neighbors), dtype=np.float64)
        else:
            distances = np.linalg.norm(vertices[neighbors] - vertices[i], axis=1)
            weights = 1.0 / (distances + epsilon)
            weights = weights / weights.sum()
        for neighbor, weight in zip(neighbors, weights, strict=True):
            matrix[i, neighbor] = -float(weight)
    return matrix


def laplacian_coordinates(
    vertices: FloatArray,
    adjacency: list[list[int]],
    *,
    weighting: LaplacianWeighting | str = LaplacianWeighting.UNIFORM,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Return Laplacian coordinates for vertices."""

    matrix = laplacian_matrix(vertices, adjacency, weighting=weighting, epsilon=epsilon)
    return matrix @ vertices
