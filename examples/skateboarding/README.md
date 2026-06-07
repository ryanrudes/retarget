# Humanoid Skateboarding Retargeting

This example runs native Vicon and GVHMR recordings through one in-memory
`RetargetingExperiment`:

```text
ViconRecordingSource + GvhmrOutputSource
-> SkateboardingObservationRecipe
-> SceneObservation
-> SkateboardingRetargetingRecipe
-> RetargetingResult
```

No synchronization command or synchronized archive is required. Contact
classification produces robot-independent semantic tracks; the adaptation
recipe resolves those tracks through robot roles.

## Setup

```bash
uv sync --extra dev
uv run python scripts/bootstrap_robot_assets.py g1 --store .retarget_assets
```

## Run

Pass the native recording locations directly:

```bash
uv run python examples/skateboarding/run_retarget.py \
  --demo pushoff5_twoshoes \
  --vicon-root /path/to/vicon-recordings \
  --gvhmr-root /path/to/gvhmr-output
```

The declarative equivalent uses `run_config.toml`. It is deserialized into the
same two recipes and experiment:

```bash
uv run retarget run --config examples/skateboarding/run_config.toml
```

For MuJoCo-backed kinematics:

```bash
uv sync --extra mujoco
uv run python examples/skateboarding/run_retarget.py \
  --demo pushoff5_twoshoes \
  --vicon-root /path/to/vicon-recordings \
  --gvhmr-root /path/to/gvhmr-output \
  --kinematics mujoco
```

Results are written under `examples/skateboarding/generated/`.
