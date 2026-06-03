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

## Phase B — Fuse into retarget inputs

```bash
cd ~/GitHub/retarget
uv sync
uv pip install -e ../motion_sync
uv pip install -e ../event_detection

uv run python examples/skateboarding/fuse_unified.py \
  --synced ../motion_sync/output/synced/<demo> \
  --output examples/skateboarding/data/<demo>
```

Writes:

| File | Contents |
|------|----------|
| `skate_motion.npz` | Z-up SMPL-X core joints, `contact_states`, `fps` |
| `board_trajectory.npz` | Board positions + quaternions (wxyz) |
| `deck_samples.npy` | Deck sample points in object frame |

`fuse_unified.py` uses `clip.core_joint_positions()`, `clip.contact(SKATE_FOOT_SUPPORT).stance_matrix()`, and Vicon board poses from `clip.body(Bodies.SKATEBOARD)`.

## Phase C — retarget

```bash
uv run retarget run --config examples/skateboarding/run_config_pushoff5.toml
uv run retarget view --result skateboarding_retarget.npz --dry-run
```

Tune objectives and constraints in the TOML ([Run configs](../tutorials/run-configs.md), [Introduction](../introduction.md)).

## Checklist

| Step | Artifact | Common failure |
|------|----------|----------------|
| FK before sync | `joints.npy` in GVHMR folder | Sync refuses to run |
| Sync quality | `lag`, `corr` in metadata | Wrong foot alignment in fuse |
| Detect | `contact__foot_support__*` keys in `synced.npz` | Stale layer after re-sync—re-run `detect --force` |
| Fuse | `skate_motion.npz` frame count = board trajectory | Trim/crop mismatch |
| Retarget | `format = "smplx"` in config | Joint order / contact name mismatch |
