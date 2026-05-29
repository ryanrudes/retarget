# Batch And Evaluation

Use:

```bash
retarget batch --input-dir motions --pattern "*.npz" --format smplx --output-dir results
retarget evaluate --result results/example.npz --output results/example.metrics.json
```

Recursive patterns preserve input-relative output paths, so duplicate filenames in different subjects or trials do not collide:

```bash
retarget batch --input-dir motions --pattern "**/*.npz" --config run.toml --output-dir results
```

An input such as `motions/subject_a/walk.npz` writes `results/subject_a/walk.npz`. The batch job id uses the same input-relative path.

Batch runs write `batch_manifest.json` after every completed job. The manifest records:

- `total`, `success_count`, `skipped_count`, and `failed_count`
- one record per input motion, with output path, status, timestamps, error type, and worker metadata
- the input directory, output directory, and glob pattern used for the run

Rerunning the same command resumes the batch: existing outputs are marked `skipped`, missing or failed outputs are retried, and `--force` reruns everything. `--max-workers` uses a process pool; custom workers must therefore be pickleable when parallelism is enabled.

Evaluate every successful or skipped result in a batch manifest with:

```bash
retarget evaluate \
  --batch-manifest results/batch_manifest.json \
  --output-dir results/metrics \
  --output results/evaluation_manifest.json
```

Per-result reports keep the result-relative layout, for example `results/subject_a/walk.npz` writes `results/metrics/subject_a/walk.metrics.json`. The summary output is an `EvaluationManifest` with success, partial, skipped, and failed counts plus the metrics recorded for each result.

Built-in evaluation reports:

- `optimization_cost`: mean per-frame solver objective.
- `foot_sliding`: stance-contact xy speed when a `RetargetingProblem` is supplied; otherwise a root-motion proxy.
- `contact_preservation`: agreement between explicit or inferred source contacts and retargeted contact-link contacts.
- `penetration`: ground penetration depth plus object/terrain sample-point clearance violations when a `RetargetingProblem` is supplied.

Reports include result identity, frame count, qpos dimension, fps, metric units, result/problem details, and warnings. If a registered metric plugin fails or returns a non-finite value, evaluation records a warning and marks the report `partial` instead of stopping a batch run.

Asset-backed projects can register stricter metrics that use simulator collision geometry or logged contact states.
