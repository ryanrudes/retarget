# Quickstart

```bash
uv sync --extra dev
uv run retarget doctor
uv run retarget run --config examples/basic/run_config.toml
uv run retarget evaluate --result examples/basic/generated/basic_retarget.npz
```

The config constructs a `MotionFileObservationRecipe`, a
`RoleRetargetingRecipe`, and a `RetargetingExperiment`.

For native multimodal capture:

```bash
uv run python examples/skateboarding/run_retarget.py \
  --demo pushoff5_twoshoes \
  --max-frames 20 \
  --download-assets
```

That command loads native Vicon and GVHMR recordings directly, performs temporal
and spatial registration in memory, derives semantic contacts, adapts them to
the robot, and retargets. It does not consume or create a synchronized stage
file.

Raw video can be composed with any `HumanPoseEstimator` through
`VideoPoseSource`; `GvhmrEstimator` is the optional concrete local-checkout
backend.

Verify changes with:

```bash
uv run ruff check src tests examples scripts docs/hooks
uv run mypy src/retarget
uv run pytest
uv run mkdocs build --strict
```

Continue with [Introduction](introduction.md), [Capture to retarget](pipeline/index.md),
and [Architecture](architecture.md).
