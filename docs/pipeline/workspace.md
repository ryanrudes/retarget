# Workspace Setup

The repository is self-contained. No git submodules, synchronization package,
or contact-classification package is required.

```bash
git clone https://github.com/ryanrudes/retarget.git
cd retarget
uv sync --extra dev
uv run pytest
uv run mkdocs build --strict
```

Optional robot assets, recordings, and estimator weights remain local data
dependencies. Install only the backend extras needed by an experiment:

```bash
uv sync --extra optimize --extra mujoco
uv sync --extra viz
uv sync --extra torch --extra smpl
```

Intermediate observation persistence is explicit through
`SceneObservation.save_npz()`.
