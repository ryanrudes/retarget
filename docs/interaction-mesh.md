# Interaction Mesh

The default retargeter preserves local geometry through an interaction mesh built from mapped human joints plus object, terrain, or ground sample points.

The mesh topology defines which vertex pairs contribute to the **laplacian** objective: local Laplacian coordinates on the source human+scene point set are matched during optimization, so connectivity directly affects how strongly neighborhoods are preserved. See [Adding objectives and constraints](adding-objectives-constraints.md) (`laplacian` term).

When a CLI run spec provides only `mesh_path` for an object or terrain, `retarget.mesh.sample_mesh_points` produces deterministic scene samples before the interaction mesh is built. In run configs, `mesh_path` lives under `[scene.object]` or `[scene.terrain]` (along with `mesh_sample_count`). This keeps experiments reproducible while letting asset-heavy runs stay outside the package.

`InteractionMeshBuilder` supports explicit topology policies:

- `delaunay`: Delaunay simplices when possible, with deterministic fallback. Fewer than four vertices uses a **complete** graph; Delaunay failure falls back to **k_nearest** with `k_neighbors` capped by `n_vertices - 1`.
- `chain`: consecutive points only; useful for tiny fixtures.
- `complete`: every point pair is connected; useful for small diagnostic cases.
- `k_nearest`: each point connects to its nearest neighbors; useful for deterministic large point sets.

```python
from retarget.mesh import InteractionMeshBuilder, InteractionMeshSpec, MeshTopology

builder = InteractionMeshBuilder(
    InteractionMeshSpec(topology=MeshTopology.K_NEAREST, k_neighbors=6)
)
mesh = builder.build(human_points, environment_points)
```

CLI and Python runs can set topology directly on the run spec:

```toml
[mesh]
topology = "k_nearest"
k_neighbors = 6
```

```python
from retarget import InteractionMeshSpec, MeshTopology

problem = problem.model_copy(
    update={"mesh": InteractionMeshSpec(topology=MeshTopology.K_NEAREST, k_neighbors=6)}
)
```

Result metadata records the actual topology and whether it came from the problem spec or a custom engine override. To share one mesh policy across many programmatic runs, pass a custom engine:

```python
from retarget.mesh import InteractionMeshBuilder, MeshTopology
from retarget.pipeline import InteractionMeshRetargetingEngine, Retargeter

engine = InteractionMeshRetargetingEngine(
    mesh_builder=InteractionMeshBuilder(topology=MeshTopology.K_NEAREST, k_neighbors=4)
)
result = Retargeter(engine=engine).run(problem)
```

## See also

- [API models](api/models.md) — `InteractionMeshSpec`, `MeshTopology`, scene object/terrain fields
- [Adding objectives and constraints](adding-objectives-constraints.md) — `laplacian` objective and scene sampling
- [Run configs](tutorials/run-configs.md) — `[mesh]` and `mesh_path` on scene entries
- [Scenes and task kinds](tutorials/scene-tasks.md) — object interaction and terrain geometry
