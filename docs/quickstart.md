# Quickstart

After this smoke test, read [Introduction](introduction.md) for a full research-style walkthrough (sensors → `MotionSequence` → scene → robot). For shorter hands-on lessons, see [Tutorials](tutorials/index.md).

```bash
uv sync --extra dev
uv run retarget doctor
uv run retarget run --motion tests/fixtures/minimal_motion.json --format minimal --robot synthetic_humanoid --output result.npz
uv run retarget evaluate --result result.npz
```

Motion inputs can be JSON, NPY, NPZ, or wide CSV files. CSV columns use the registered format's joint names with `_x`, `_y`, and `_z` suffixes.

For repeatable experiments, put the run in a TOML or YAML config:

```bash
uv run retarget run --config examples/run_config.toml   # writes examples/configured_fixture.npz
uv run retarget evaluate --result examples/configured_fixture.npz --config examples/run_config.toml
```

For a full capture-to-robot path (Vicon, GVHMR, foot support, fuse, retarget), see [Research ecosystem](ecosystem/index.md) and `examples/skateboarding/`.

Verify local changes with the same gates used by CI:

```bash
git submodule update --init   # vendor motion_sync + contact_detection for API docs
uv run ruff check src tests examples docs
uv run mypy src
uv run pytest
uv run mkdocs build --strict
```

Sibling-repo layout and optional editable installs: [Workspace setup](ecosystem/workspace-setup.md).

Pytest runs a fixed set of example scripts from a temporary working directory (`tests/test_examples.py`: `basic_robot_only.py`, `batch_and_evaluate.py`, `climbing_terrain.py`, `custom_motion_format.py`, `custom_objective.py`, `custom_robot.py`, `object_interaction.py`, `skateboarding/run_retarget.py`). Keep those scripts self-contained and free of repo-local output assumptions.
