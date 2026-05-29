# Interaction Mesh

The default retargeter preserves local geometry through an interaction mesh built from mapped human joints plus object, terrain, or ground sample points.

When a CLI run spec provides only `mesh_path` for an object or terrain, `retarget.mesh.sample_mesh_points` produces deterministic scene samples before the interaction mesh is built. This keeps experiments reproducible while letting asset-heavy runs stay outside the package.

`InteractionMeshBuilder` supports explicit topology policies:

- `delaunay`: Delaunay simplices when possible, with deterministic fallback.
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
