# Troubleshooting

Run `retarget doctor` to see registered motion formats, loaders, robots, kinematics backends, objective terms, constraint terms, solver factories, metrics, exporters, visualizers, and optional dependency availability. Install feature extras only when needed, for example `uv sync --extra optimize --extra mujoco`.

Run configs are preflighted against the extension registries before motions or assets are loaded. If a CLI run reports an unknown objective, constraint, solver, robot, or motion format, add the extension module to the run spec's `imports` list or import/register it in the process that builds the `RetargetingProblem`, then rerun `retarget doctor` to confirm the name appears.

## Live code, MkDocs serve, and Jupyter

Live-code cells need a local docs server and Jupyter. See [Interactive playground](interactive-playground.md) for setup (`uv sync --extra dev`, `uv run mkdocs serve`, optional `./scripts/docs-jupyter.sh`).

- **Live switch shows "not installed"** — install the dev extra: `uv sync --extra dev`.
- **"Not running" on GitHub Pages** — live code is local-only; use `mkdocs serve` on your machine.
- **Run fails after editing a snippet** — reset the cell or restart Jupyter if the kernel is stuck.
- **Symbol hovers missing in code blocks** — run `mkdocs build` (or restart `mkdocs serve`) so hooks regenerate `api-symbols.json`; use qualified names when a short token is ambiguous.

## Docs build and submodules

API pages for `motion_sync` and `contact_detection` import vendor packages from submodules:

```bash
git submodule update --init
uv sync --extra dev
uv run mkdocs build
```

If mkdocstrings cannot import a sibling package, confirm `vendor/motion_sync` and `vendor/event_detection` exist and match [Workspace setup](ecosystem/workspace-setup.md).

## Batch resume, `--force`, and SMPL-X robots

Batch runs write `batch_manifest.json` and resume by default: completed outputs are skipped, failures are retried. Pass `--force` to rerun every job. See [Batch and evaluation](batch-evaluation.md).

SMPL-X motions need a real humanoid target. Pass `--robot <name>` or set `robot` in a shared `--config`; without that, batch may default to `synthetic_humanoid`, which is usually wrong for SMPL-X input.

## Export, view, and the viz extra

`retarget view` defaults to **dry-run** (summary only). Live Viser playback needs `uv sync --extra viz` and `--live`. Pass `--robot-spec <path>` when playback should load URDF/MJCF from a `RobotSpec` file. Details: [Export and visualization](export-visualization.md).

## motion_sync and contact_detection imports

- **Python package names:** `motion_sync`, `contact_detection` (not the local clone folder name).
- **Submodule paths in retarget:** `vendor/motion_sync`, `vendor/event_detection` (remote: [contact_detection](https://github.com/ryanrudes/contact_detection)).
- **Editable sibling clones:** `../motion_sync` and `../event_detection` work for pipeline development; keep demo names aligned across `output/vicon_tables`, `output/gvhmr`, and `output/synced`.

Full wiring: [Ecosystem workspace setup](ecosystem/workspace-setup.md).
