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
uv run retarget run --config examples/run_config.toml
uv run retarget evaluate --result examples/configured_fixture.npz --config examples/run_config.toml
```

Verify local changes with the same gates used by CI:

```bash
uv run ruff check src tests examples
uv run mypy src
uv run pytest
uv run mkdocs build --strict
```

The pytest suite executes every script in `examples/` from a temporary working directory, so examples must remain self-contained and free of repo-local output assumptions.
