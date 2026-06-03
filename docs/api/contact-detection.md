# contact_detection API

**Full package index:** `import contact_detection` — algorithms only; no clip I/O.

Foot-support **classification**, **quiet-interval** detection, and **generic contact** intervals on marker or body time series. Persistence and clip wiring live in **motion_sync** (`SyncClip.detect`, contact layers on `synced.npz`).

Requires the [contact_detection](https://github.com/ryanrudes/contact_detection) package on `PYTHONPATH` (`import contact_detection`). In this repo it is vendored at `vendor/event_detection` (submodule path; GitHub repo name is **contact_detection**). A sibling clone folder name is arbitrary—often `event_detection`. See [Workspace setup](../ecosystem/workspace-setup.md).

## Quiet detection

Detect low-activity windows on scalar, vector, or quaternion signals (smoothing, hysteresis, minimum duration). Use before support fitting or contact scoring when the scene should be still.

::: contact_detection.detect_quiet_intervals

::: contact_detection.QuietDetectionConfig

::: contact_detection.QuietDetectionResult

::: contact_detection.compute_quiet_activity_and_spread

## Generic contact intervals

Fit a support surface (plane or heightmap), score proximity and motion relative to it, and emit contact intervals. Distinct from per-foot skate **foot support** (below).

::: contact_detection.detect_contact_intervals

::: contact_detection.ContactDetectionConfig

::: contact_detection.SupportModel

::: contact_detection.PlaneSupportModel

## Support surface fitting

Bootstrap and refine floor or heightmap models from quiet windows and marker clouds. Lower-level building blocks for custom contact pipelines; `detect_contact_intervals` wraps most of this workflow.

::: contact_detection.SupportDetectionConfig

::: contact_detection.bootstrap_support_surface

::: contact_detection.find_support_candidates

::: contact_detection.filter_support_candidates

::: contact_detection.fit_support_model_from_candidates

::: contact_detection.fit_best_support_surface

::: contact_detection.fit_plane_svd

::: contact_detection.HeightmapSupportModel

::: contact_detection.LocalPercentileHeightmap

::: contact_detection.FloorModel

::: contact_detection.SupportCandidate

::: contact_detection.SupportCandidateSet

::: contact_detection.compute_support_relative_features

## Foot support

Per-foot **air / ground / skateboard** classification from synced mocap (and optional board body). Register and persist via motion_sync; call these functions directly when building custom pipelines.

::: contact_detection.classify_foot_support_states

::: contact_detection.FootSupportConfig

::: contact_detection.FootSupportState

::: contact_detection.FootSupportClassification

## Contact scoring

Feature extraction, hysteresis scoring, and offset estimation used by generic contact detection. Enums configure support-model and quiet-signal modes.

::: contact_detection.ContactDetectionResult

::: contact_detection.score_contact_features

::: contact_detection.score_hysteresis_mask

::: contact_detection.estimate_contact_offset

::: contact_detection.SupportModelType

::: contact_detection.VectorQuietMode

::: contact_detection.QuietSignalType

## Interval utilities

Convert between boolean masks and `(start, end)` intervals, summarize runs, and clean masks temporally. `motion_sync.intervals.intervals_from_mask` duplicates the mask→interval helper so clips do not require this package at read time. Use **contact_detection** when running detectors; use **motion_sync** tracks for `.intervals(state)` on loaded clips.

::: contact_detection.intervals_from_mask

::: contact_detection.intervals_by_state

::: contact_detection.summarize_intervals

::: contact_detection.mask_from_intervals

::: contact_detection.IntervalSummary

::: contact_detection.clean_mask_by_time

## Signal / time helpers

Shared smoothing, window statistics, quaternion metrics, and enum normalization used by quiet detection and contact scoring.

??? note "Utilities"

    ::: contact_detection.quaternion_angular_speed

    ::: contact_detection.quaternion_local_spread

    ::: contact_detection.quaternion_standardize_xyzw

    ::: contact_detection.time_gaussian_smooth

    ::: contact_detection.time_window_component_range

    ::: contact_detection.time_window_range

    ::: contact_detection.time_window_rms

    ::: contact_detection.time_window_std

    ::: contact_detection.local_polynomial_derivative

    ::: contact_detection.normalize_enum

## CLI diagnostics

The contact_detection repo’s `main.py` discovers `synced.npz` trials and plots classifications. Prefer `motion-sync detect foot-support` to persist layers on the clip.

## See also

- [End-to-end pipeline (skateboarding)](../ecosystem/pipeline.md) — `motion-sync detect foot-support` after sync
- [motion_sync API](motion-sync.md) — `SyncClip.detect`, `ContactLayer`, foot-support contact types
