# End-to-end pipeline (skateboarding)

Reference path from raw capture to `retarget run`. Commands assume the [workspace layout](workspace-setup.md).

## Phase A — motion_sync ingest and sync

```bash
cd ~/GitHub/motion_sync
uv sync
uv pip install -e ../event_detection

# 1. Vicon bags → vicon.npz
./scripts/convert_bags.bash data/bags output/vicon_tables

# 2. GVHMR (external) → output/gvhmr/<demo>/hmr4d_results.pt

# 3. SMPL-X FK → joints.npy, vertices.npy
./scripts/run_smplx_fkin.bash output/gvhmr

# 4. Time sync → output/synced/<demo>/synced.npz
./scripts/sync.bash
# optional: ./scripts/sync.bash --plot

# 5. Foot support on the clip (persisted on synced.npz)
uv run motion-sync detect foot-support output/synced/<demo> --plot
```

Inspect in Python:

```python
from motion_sync import SyncClip
from motion_sync.schemas.skateboarding import SKATE_SESSION, SKATE_FOOT_SUPPORT

clip = SyncClip.load("output/synced/<demo>", session=SKATE_SESSION)
foot = clip.contact(SKATE_FOOT_SUPPORT)
stance = foot.stance_matrix()  # (frames, 2) for retarget L_Foot / R_Foot
```

## Phase B — retarget adapter and run

Foot-support should be on the clip before retargeting (Phase A step 5). The skateboarding adapter refreshes stale `SKATE_FOOT_SUPPORT` labels in memory; pass `--save-contact-layer` to `run_retarget.py` if you want to persist a freshly detected layer back to `synced.npz`.

```bash
cd ~/GitHub/retarget
uv sync
git submodule update --init
```

The adapter returns:

| Object | Contents |
|--------|----------|
| `MotionSequence` | Z-up SMPL-X core joints, root poses, fps, and provenance |
| `SceneSpec` | Board object trajectory and deck sample points |
| `ContactPlan` | Foot-support states, link mapping, and support plane |
| `LinkTargetPlan` | Named robot link targets for mocap shoes, SMPL-X lower body, and torso proxy |

`retarget.integrations.motion_sync.skateboarding` uses `clip.core_joint_positions()`, `clip.contact(SKATE_FOOT_SUPPORT).stance_matrix()`, Vicon shoe poses from `clip.body(Bodies.LEFT_SHOE / RIGHT_SHOE)`, and board poses from `clip.body(Bodies.SKATEBOARD)`.

## Phase C — retarget

Install the G1 robot assets once, then run the programmatic MuJoCo-aware example. Example demo: `pushoff5_twoshoes`.

```bash
cd ~/GitHub/retarget
uv run python scripts/bootstrap_robot_assets.py g1 --store .retarget_assets
uv run python examples/skateboarding/run_retarget.py --demo pushoff5_twoshoes
uv run retarget evaluate \
  --result examples/skateboarding/generated/pushoff5_twoshoes/pushoff5_twoshoes_retarget.npz \
  --config examples/skateboarding/run_config.toml
uv run retarget view \
  --result examples/skateboarding/generated/pushoff5_twoshoes/pushoff5_twoshoes_retarget.npz \
  --live \
  --robot-spec .retarget_assets/robot/g1/robot.toml
```

Tune objectives and constraints in the TOML ([Run configs](../tutorials/run-configs.md), [Introduction](../introduction.md)).

## Checklist

| Step | Artifact | Common failure |
|------|----------|----------------|
| FK before sync | `joints.npy` in GVHMR folder | Sync refuses to run |
| Sync quality | `lag`, `corr` in metadata | Wrong foot alignment in fuse |
| Detect | `contact__foot_support__*` keys in `synced.npz` | Stale layer after re-sync—re-run `detect --force` |
| Adapter | `PreparedRetargetInputs` frame counts align | Trim/crop mismatch |
| Retarget | `[source] kind = "motion_sync_skateboarding"` in config | Missing synced clip or stale contact layer |
