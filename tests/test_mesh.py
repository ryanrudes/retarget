import numpy as np
import pytest

from retarget.mesh import (
    InteractionMeshBuilder,
    InteractionMeshSpec,
    LaplacianWeighting,
    MeshTopology,
    adjacency_from_simplices,
    laplacian_coordinates,
    laplacian_matrix,
    sample_mesh_points,
)


def test_adjacency_and_laplacian():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    simplices = np.array([[0, 1, 2]])
    adjacency = adjacency_from_simplices(simplices, 3)
    assert adjacency == [[1, 2], [0, 2], [0, 1]]
    matrix = laplacian_matrix(vertices, adjacency)
    coords = laplacian_coordinates(vertices, adjacency)
    assert matrix.shape == (3, 3)
    assert coords.shape == vertices.shape


def test_laplacian_defaults_to_holosoma_uniform_neighbor_average():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    adjacency = [[1, 2], [0, 2], [0, 1]]

    matrix = laplacian_matrix(vertices, adjacency)
    coords = laplacian_coordinates(vertices, adjacency)

    assert np.allclose(matrix[0], [1.0, -0.5, -0.5])
    assert np.allclose(coords, matrix @ vertices)


def test_laplacian_keeps_explicit_inverse_distance_mode():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    adjacency = [[1, 2], [0, 2], [0, 1]]

    matrix = laplacian_matrix(vertices, adjacency, weighting=LaplacianWeighting.INVERSE_DISTANCE)

    assert not np.allclose(matrix[0], [1.0, -0.5, -0.5])
    assert np.allclose(matrix.sum(axis=1), 0.0)


def test_interaction_mesh_builder_chain_fallback():
    mesh = InteractionMeshBuilder(topology=MeshTopology.CHAIN).build(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    assert mesh.vertices.shape == (2, 3)
    assert mesh.simplices.shape == (1, 2)


def test_interaction_mesh_complete_topology():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    mesh = InteractionMeshBuilder(topology=MeshTopology.COMPLETE).build(points)

    assert mesh.simplices.shape == (6, 2)
    assert mesh.adjacency == [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]


def test_interaction_mesh_k_nearest_topology_is_deterministic():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    mesh = InteractionMeshBuilder(InteractionMeshSpec(topology=MeshTopology.K_NEAREST, k_neighbors=1)).build(points)

    assert mesh.simplices.tolist() == [[0, 1], [1, 2], [2, 3]]


def test_interaction_mesh_rejects_mixed_spec_and_options():
    with pytest.raises(ValueError, match="Pass either spec"):
        InteractionMeshBuilder(InteractionMeshSpec(), topology=MeshTopology.CHAIN)


def test_sample_mesh_points_from_obj_surface_is_deterministic(tmp_path):
    mesh_path = tmp_path / "triangle.obj"
    mesh_path.write_text(
        """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
""".strip()
    )

    first = sample_mesh_points(mesh_path, count=5)
    second = sample_mesh_points(mesh_path, count=5)

    assert first.shape == (5, 3)
    assert np.all(np.isfinite(first))
    assert np.allclose(first[:, 2], 0.0)
    assert np.allclose(first, second)


def test_sample_mesh_points_from_obj_vertices_when_faces_are_absent(tmp_path):
    mesh_path = tmp_path / "points.obj"
    mesh_path.write_text(
        """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
""".strip()
    )

    points = sample_mesh_points(mesh_path, count=5)

    assert points.shape == (5, 3)
    assert points.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
