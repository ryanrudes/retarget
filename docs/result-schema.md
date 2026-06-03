# Result Schema

`RetargetingResult` stores:

- `schema_version`: result NPZ schema version.
- `name`: run identifier (from the problem or CLI config).
- `status`: `RunStatus` (`success`, `partial`, `failed`, …) for the retargeting run.
- `qpos`: `(frames, nq)` robot trajectory.
- `fps`: result frame rate.
- `cost`: optional optimization cost; length **1** (single aggregate) or **per frame** matching `qpos`.
- `human_joints`: optional source motion positions.
- `metadata`: robot, task, solver, and mapping information.
- `metadata_json` and `warnings_json` in saved NPZ files for inspectable metadata without loading pickled object arrays.

See [API models](api/models.md) for field types and validators.

Use `RetargetingResult.resampled(fps)` when comparing or exporting runs on a common time grid. When `RetargetingProblem.output_fps` is set, `Retargeter` resamples the source motion and any dynamic object trajectory before optimization, so the saved result frame count matches the requested output rate.

Evaluation aligns a supplied `RetargetingProblem` to the result time grid before computing contact and scene metrics. This keeps CLI reports correct when `retarget run --config ...` used `output_fps` and `retarget evaluate --config ...` later reloads the original run spec.

Result metadata includes a `provenance` object with the run name, task kind, input/output FPS, scale policy, motion summary, robot summary, scene summary, interaction mesh topology, requested solver config, actual solver backend, per-frame solver statuses, objective specs, constraint specs, result dimensions, and user-supplied problem metadata. Saved NPZ files also keep legacy `metadata` and `warnings` object-array keys for compatibility, but `RetargetingResult.load_npz()` reads with `allow_pickle=False` by default and prefers the JSON keys. Pass `allow_pickle=True` only when loading trusted legacy files that do not contain JSON metadata.

`EvaluationReport` stores:

- `status`: overall evaluation outcome (`success` or `partial` when individual metrics fail).
- `source_name`, `frame_count`, `qpos_dimension`, and `fps` for joining reports back to results.
- `task_kind`, `robot_name`, and `motion_name` when evaluation used a `RetargetingProblem`.
- `metrics`: scalar metric values (access as `report.metrics["metric_name"]`, not a bare `metrics` dict on the result).
- `metric_units`: display units such as `m`, `m/s`, `fraction`, or `cost`.
- `details`: machine-readable result/problem context.
- `warnings`: result warnings plus any metric failures.

Metric failures produce a `partial` report instead of aborting evaluation, which keeps long batch analyses resumable while making failed metric plugins visible.

`EvaluationManifest` stores batch evaluation summaries:

- `total`, `success_count`, `partial_count`, `skipped_count`, and `failed_count`.
- one `EvaluationRecord` per batch output, with result path, report path, status, source name, frame count, metric values, and warnings.
