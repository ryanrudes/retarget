# Humanoid Skateboarding Retargeting

Runnable code for the research skateboarding walkthrough. This example starts from
`motion_sync` synced clips, uses `contact_detection` foot-support labels as
retargeter hints, and renders the retargeted humanoid with a real URDF-backed G1
model.

Results are written under `examples/skateboarding/generated/` and are ignored by git.

| Script | Purpose |
|--------|---------|
| `run_retarget.py` | Load a synced `motion_sync` clip, build typed contacts and link targets, solve, and save the result |
| `run_config.toml` | CLI equivalent using the same `motion_sync_skateboarding` source |

## Setup

```bash
uv sync --extra dev
git submodule update --init
uv run python scripts/bootstrap_robot_assets.py g1 --store .retarget_assets
```

The bootstrap script copies Holosoma's G1 URDF/MJCF assets into
`.retarget_assets/robot/g1`, writes `.retarget_assets/robot/g1/robot.toml`, and
registers the asset-store manifest.

## Run

```bash
uv run python examples/skateboarding/run_retarget.py --demo pushoff5_twoshoes --download-assets
```

For MuJoCo-backed kinematics, install the optional stack:

```bash
uv sync --extra mujoco
uv run python examples/skateboarding/run_retarget.py --demo pushoff5_twoshoes --kinematics mujoco
```

## View

```bash
uv sync --extra viz
uv run retarget view \
  --result examples/skateboarding/generated/pushoff5_twoshoes/pushoff5_twoshoes_retarget.npz \
  --live \
  --robot-spec .retarget_assets/robot/g1/robot.toml
```

Live robot playback requires a valid URDF. Missing robot assets are treated as a
setup error rather than drawn as point or line primitives.
