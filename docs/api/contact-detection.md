# contact_detection API

**Full package index:** `import contact_detection` — algorithms only; no clip I/O.

Foot-support **classification**, **quiet-interval** detection, and **generic contact** intervals on marker or body time series. Persistence and clip wiring live in **motion_sync** (`SyncClip.detect`, contact layers on `synced.npz`).

Requires the [contact_detection](https://github.com/ryanrudes/contact_detection) package on `PYTHONPATH` (`import contact_detection`). In this repo it is vendored at `vendor/event_detection` (submodule path; GitHub repo name is **contact_detection**). A sibling clone folder name is arbitrary—often `event_detection`. See [Workspace setup](../ecosystem/workspace-setup.md).

## Quiet detection

Detect low-activity windows on scalar, vector, or quaternion signals (smoothing, hysteresis, minimum duration). Use before support fitting or contact scoring when the scene should be still.

::: contact_detection.detect_quiet_intervals

::: contact_detection.QuietDetectionConfig

::: contact_detection.QuietDetectionResult

## Generic contact intervals

Fit a support surface (plane or heightmap), score proximity and motion relative to it, and emit contact intervals. Distinct from per-foot skate **foot support** (below).

::: contact_detection.detect_contact_intervals

::: contact_detection.ContactDetectionConfig

::: contact_detection.SupportModel

::: contact_detection.PlaneSupportModel

## Foot support

Per-foot **air / ground / skateboard** classification from synced mocap (and optional board body). Register and persist via motion_sync; call these functions directly when building custom pipelines.

::: contact_detection.classify_foot_support_states

::: contact_detection.FootSupportConfig

::: contact_detection.FootSupportState

::: contact_detection.FootSupportClassification

## Intervals (also in motion_sync)

`motion_sync.intervals.intervals_from_mask` duplicates the mask→interval helper so clips do not require this package at read time. Use **contact_detection** when running detectors; use **motion_sync** tracks for `.intervals(state)` on loaded clips.

::: contact_detection.intervals.intervals_from_mask

## CLI diagnostics

The contact_detection repo’s `main.py` discovers `synced.npz` trials and plots classifications. Prefer `motion-sync detect foot-support` to persist layers on the clip.

## See also

- [End-to-end pipeline (skateboarding)](../ecosystem/pipeline.md) — `motion-sync detect foot-support` after sync
- [motion_sync API](motion-sync.md) — `SyncClip.detect`, `ContactLayer`, foot-support contact types
