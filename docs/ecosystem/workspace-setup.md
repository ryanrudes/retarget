# Workspace setup

## Clone layout

**Docs and API reference (retarget repo):** clone with submodules so mkdocstrings can import `motion_sync` and `contact_detection`:

```bash
cd ~/GitHub
git clone --recurse-submodules https://github.com/ryanrudes/retarget.git
# existing clone without submodules:
cd retarget && git submodule update --init
```

Submodules live at `vendor/motion_sync` and `vendor/event_detection` (GitHub remote: [contact_detection](https://github.com/ryanrudes/contact_detection)).

**Pipeline development:** clone **motion_sync** and **contact_detection** as siblings when you need editable installs, large `output/` trees, and SMPL-X data outside the retarget tree:

```bash
cd ~/GitHub
git clone https://github.com/ryanrudes/motion_sync.git
git clone https://github.com/ryanrudes/contact_detection.git event_detection   # local folder name is arbitrary
```

Keep **the same demo name** across `output/vicon_tables/<demo>`, `output/gvhmr/<demo>`, and `output/synced/<demo>` inside `motion_sync`.

## Editable installs

Use one virtual environment per repo, or a shared env with all three editable (sibling layout):

```bash
# motion_sync
cd ~/GitHub/motion_sync
uv sync
uv pip install -e ../event_detection   # foot-support detector for `motion-sync detect`

# retarget
cd ~/GitHub/retarget
uv sync --extra dev
uv pip install -e ../motion_sync
uv pip install -e ../event_detection   # only if you run contact_detection CLI/tests
```

Verify:

```bash
uv run motion-sync --help
uv run retarget doctor
python -c "from motion_sync import SyncClip; from contact_detection import classify_foot_support_states"
```

## Building this documentation site

API pages for **motion_sync** and **contact_detection** use mkdocstrings with paths under `vendor/` (git submodules). From the **retarget** repo root:

Published docs: **https://ryanrudes.github.io/retarget/** (built on each push to `master`).

```bash
git submodule update --init
uv sync --extra dev
uv run mkdocs serve
```

If submodules are missing, narrative docs still build; API pages for those packages fail to import until `git submodule update --init` succeeds.

For local development you may point mkdocstrings at sibling clones instead (`../motion_sync`, `../event_detection/src`) by editing `paths` in `mkdocs.yml`; the committed config uses `vendor/*`.

## Environment variables

| Variable | Used by |
|----------|---------|
| `SMPL-X` weights under `motion_sync/data/smplx_models/` | `motion-sync fkin` |
| GVHMR outputs in `motion_sync/output/gvhmr/<demo>/` | sync pipeline |

`retarget` does not read GVHMR or Vicon paths directly for the skateboarding walkthrough—only fused NPZ exports.
