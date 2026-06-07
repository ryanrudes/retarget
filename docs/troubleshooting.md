# Troubleshooting

Run `retarget doctor` to inspect registered formats, robots, backends, terms,
solvers, metrics, exporters, and visualizers. Run configs are preflighted
against these registries before capture or assets are loaded.

## Alignment Failures

Temporal and spatial quality gates raise `AlignmentError` with an
`AlignmentReport`. Inspect the report instead of forcing zero lag:

```python
from retarget import AlignmentError

try:
    observation = recipe.observe()
except AlignmentError as error:
    print(error.report)
```

Check cue validity, source units, coordinate frames, and overlap before
loosening thresholds.

## Optional Capture Backends

- ROS 2 bags require `uv sync --extra capture-ros`.
- Video decoding requires `uv sync --extra capture-video`.
- The `GvhmrEstimator` backend requires a configured external checkout plus
  `uv sync --extra capture-gvhmr`.

`GvhmrEstimator` runs the configured command in a temporary workspace. Missing
output is an estimator error; it does not leave or reuse an implicit cache.

## Robot Roles

Missing role errors mean the recipe requested a semantic role that the selected
`RobotSpec` does not bind. Add the role to `joint_roles`, `link_roles`, or
`link_groups`. Do not patch the recipe with concrete link names or side-name
heuristics.

## Batch Resume

Batch runs write `batch_manifest.json` and resume by default. Completed outputs
are skipped and failures are retried. Pass `--force` to rerun every job.

## Export And Visualization

`retarget view` defaults to a dry-run summary. Live Viser playback requires
`uv sync --extra viz` and `--live`. Provide `--robot-spec` when playback needs
URDF or MJCF assets.

## Docs And Live Code

Use `uv sync --extra dev` and `uv run mkdocs serve`. The live-code switch works
only on the local server. Run `mkdocs build` or restart the server after changing
API hooks so `api-symbols.json` is regenerated.
