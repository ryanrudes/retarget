# Workspace Setup

The repository is self-contained. No git submodules or sibling Python packages
are required for tests, docs, capture fusion, or contact classification.

```bash
git clone https://github.com/ryanrudes/retarget.git
cd retarget
uv sync --extra dev
uv run pytest
uv run mkdocs build --strict
```

Optional experiment assets remain local:

- robot URDF/MJCF files in an `AssetStore`
- native Vicon arrays or a custom ROS bag source
- GVHMR outputs or a local GVHMR checkout used by `GvhmrEstimator`
- Holosoma fixtures when running parity asset tests

Install backend extras only where needed:

```bash
uv sync --extra optimize --extra mujoco
uv sync --extra viz
uv sync --extra torch --extra smpl
```

Large recordings and estimator weights are data dependencies, not package
dependencies. Point source objects at their paths; recipes keep all intermediate
state in memory unless `SceneObservation.save_npz()` is called explicitly.
