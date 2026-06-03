# contact_detection API

Foot-support **classification algorithms** (air / ground / skateboard). Persistence and clip wiring live in **motion_sync** (`SyncClip.detect`, contact layers on `synced.npz`).

Requires [event_detection](https://github.com/ryanrudes/event_detection) on `PYTHONPATH`. See [Workspace setup](../ecosystem/workspace-setup.md).

## Primary entry points

::: contact_detection.classify_foot_support_states

::: contact_detection.FootSupportConfig

::: contact_detection.FootSupportState

::: contact_detection.FootSupportClassification

## Intervals (also in motion_sync)

`motion_sync.intervals.intervals_from_mask` duplicates the mask→interval helper so clips do not require this package at read time. Use **contact_detection** when running detectors; use **motion_sync** tracks for `.intervals(state)` on loaded clips.

::: contact_detection.intervals.intervals_from_mask

## CLI diagnostics

`event_detection` repo `main.py` discovers `synced.npz` trials and plots classifications. Prefer `motion-sync detect foot-support` to persist layers on the clip.
