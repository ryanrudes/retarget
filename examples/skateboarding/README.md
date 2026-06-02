# Skateboarding research example

Runnable code for the [Introduction](../../docs/introduction.md) walkthrough (moving skateboard + foot contacts + human motion).

| Script | Purpose |
|--------|---------|
| `_synthetic.py` | Synthetic motion, deck samples, and board trajectory (no external data) |
| `fuse_motion.py` | Export `data/skate_motion.npz`, `data/deck_samples.npy`, `data/board_trajectory.npz` |
| `probe_mapping.py` | Print `resolved_link_mapping()` for the fixture robot |
| `run_retarget.py` | Full Python API retarget → `skateboarding_retarget.npz` |
| `run_config.toml` | Same job via CLI (run `fuse_motion.py` first) |

```bash
# From repository root
uv run python examples/skateboarding/run_retarget.py
uv run retarget evaluate --result skateboarding_retarget.npz
uv run retarget view --result skateboarding_retarget.npz --dry-run
uv sync --extra viz
uv run retarget view --result skateboarding_retarget.npz --live

# CLI path (writes fixtures into examples/skateboarding/data/)
cd examples/skateboarding
uv run python fuse_motion.py
uv run retarget run --config run_config.toml
uv run retarget view --result skateboarding_retarget.npz --live
```

The fixture uses the `minimal` motion format and `synthetic_humanoid` so it runs without lab assets. For SMPL-X + `g1_like`, swap `format` / `robot` in `run_config.toml` and replace `data/skate_motion.npz` with your fused export.

The default profile is intentionally small and robust: Laplacian + smoothness objectives with joint-limit and trust-region constraints. Add foot-contact, non-penetration, or self-collision constraints in `run_config.toml` when you are ready to tune the synthetic clip or run against real assets.
