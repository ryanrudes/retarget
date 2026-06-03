# Tutorials

These guides walk through retarget end to end: install, run, inspect results, tune configs, and scale up to batch jobs. They assume you have cloned the repository and can run commands from the repo root.

If you only need a one-liner to verify the install, start with [Quickstart](../quickstart.md). If you are planning a project with heterogeneous sensors (mocap, video body, contacts, props), read [Introduction](../introduction.md) first—it walks through a complete research use case. Use the pages below for shorter, focused lessons on each subsystem.

## Suggested order

| Step | Tutorial | You will |
|------|----------|----------|
| 1 | [Your first retarget](your-first-retarget.md) | Run the CLI and Python API, inspect a `.npz` result, evaluate quality |
| 2 | [Run configs](run-configs.md) | Drive repeatable experiments from TOML/YAML/JSON |
| 3 | [Motion input](motion-input.md) | Load JSON, NPZ, NPY, and CSV; match joint names to a format |
| 4 | [Scenes and task kinds](scene-tasks.md) | Robot-only, object interaction, and climbing workflows |
| 5 | [Python API](python-api.md) | Build `RetargetingProblem` objects without the CLI |
| 6 | [Batch and metrics](batch-and-metrics.md) | Process many motions and summarize evaluation reports |
| 7 | [Export and view](export-and-view.md) | Ship tracking NPZ files and preview results |

## Live code in the browser

Several tutorials include runnable Python blocks. To execute them in the docs UI:

1. Start MkDocs: `uv sync --extra dev` then `uv run mkdocs serve`
2. Start Jupyter: `./scripts/docs-jupyter.sh` (second terminal)
3. Flip **Live** in the page header (left of the search bar)

The [hosted site](https://ryanrudes.github.io/retarget/) is read-only: no Live switch or Jupyter kernel.

See [Interactive playground](../interactive-playground.md) for troubleshooting.

## Reference vs tutorials

| Need | Go to |
|------|--------|
| Mental model and engine flow | [Architecture](../architecture.md) |
| Frames, units, resampling | [Coordinate conventions](../coordinate-conventions.md) |
| NPZ fields and provenance | [Result schema](../result-schema.md) |
| Batch manifests, resume, evaluation metrics | [Batch and evaluation](../batch-evaluation.md) |
| Exporters, viewers, tracking NPZ layout | [Export and visualization](../export-visualization.md) |
| Asset manifests and stores | [Assets](../assets.md) |
| Add robots, formats, objectives | [Extending](../extending.md) |
| API signatures | [API](../api/index.md) |
