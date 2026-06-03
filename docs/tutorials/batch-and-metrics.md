# Batch and metrics

When you have many motion files—subjects, clips, or augmentation variants—use **`retarget batch`** to retarget them with shared settings and **`retarget evaluate`** to score every output. Both commands write JSON manifests so runs are resumable and auditable.

For manifest fields and resume semantics, see [Batch and Evaluation](../batch-evaluation.md).

## Prepare a motion directory

Copy or symlink motions into one folder. This tutorial reuses the JSON fixture twice to simulate two clips:

```bash
mkdir -p tutorials_motions
cp tests/fixtures/minimal_motion.json tutorials_motions/clip_a.json
cp tests/fixtures/minimal_motion.json tutorials_motions/clip_b.json
```

## Run a batch

```bash
uv run retarget batch \
  --input-dir tutorials_motions \
  --pattern "*.json" \
  --format minimal \
  --robot synthetic_humanoid \
  --output-dir tutorials_batch_results
```

Or share a run config across every file:

```bash
uv run retarget batch \
  --input-dir tutorials_motions \
  --pattern "*.json" \
  --config examples/run_config.toml \
  --output-dir tutorials_batch_results
```

Outputs mirror input paths: `tutorials_batch_results/clip_a.npz`, etc. After the run, open `tutorials_batch_results/batch_manifest.json` for per-job status, timestamps, and errors.

### Resume and parallel workers

- **Resume** — Re-run the same command; existing NPZ files are `skipped` unless you pass `--force`.
- **Parallelism** — `--max-workers 4` uses a process pool. Custom problem builders must be picklable when workers > 1.

Nested globs preserve subfolders:

```bash
uv run retarget batch \
  --input-dir datasets/motions \
  --pattern "**/*.json" \
  --output-dir datasets/results
```

`motions/subject_a/walk.json` → `results/subject_a/walk.npz`.

## Evaluate one result

```bash
uv run retarget evaluate \
  --result tutorials_batch_results/clip_a.npz \
  --output tutorials_batch_results/clip_a.metrics.json
```

With a run config (recommended for object/terrain scenes):

```bash
uv run retarget evaluate \
  --result tutorials_batch_results/clip_a.npz \
  --config examples/run_config.toml \
  --output tutorials_batch_results/clip_a.metrics.json
```

Metric names, definitions, and partial-report behavior: [Batch and evaluation](../batch-evaluation.md).

## Evaluate an entire batch

```bash
uv run retarget evaluate \
  --batch-manifest tutorials_batch_results/batch_manifest.json \
  --output-dir tutorials_batch_results/metrics \
  --output tutorials_batch_results/evaluation_manifest.json
```

Per-result JSON files land beside the manifest layout, for example `metrics/clip_a.metrics.json`. The evaluation manifest aggregates success, partial, skipped, and failed counts.

## Programmatic batch (optional)

`examples/batch_and_evaluate.py` calls `main([...])` with the same flags—useful in integration tests or driver scripts:

```python
from pathlib import Path
from retarget.cli.main import main

root = Path(__file__).resolve().parents[1]
out = Path("batch_results")
main(["batch", "--input-dir", str(root / "tests/fixtures"),
      "--pattern", "*.json", "--output-dir", str(out)])
main(["evaluate", "--batch-manifest", str(out / "batch_manifest.json"),
      "--output-dir", str(out / "metrics"),
      "--output", str(out / "evaluation_manifest.json")])
```

## Workflow tips

1. **Version the run config** next to your dataset README.
2. **Evaluate with the same config** you used to retarget when metrics depend on scene geometry or `output_fps`.
3. **Archive manifests** with results—they record git hashes, worker ids, and failure messages.
4. **Inspect partial reports** before deleting “bad” NPZ files—a single metric plugin may have failed while the trajectory is usable.

## Next steps

- [Export and view](export-and-view.md) — ship tracking files to simulators
- [Troubleshooting](../troubleshooting.md) — common batch failures
