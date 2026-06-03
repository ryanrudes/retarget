# API overview

Generated from docstrings in the installed package. Use the sidebar sections below, or search (++ctrl+k++ / ++cmd+k++) for a symbol name.

!!! tip "Where to start"
    Most workflows use [`Retargeter`][retarget.pipeline.retargeter.Retargeter],
    [`RetargetingProblem`][retarget.pipeline.problem.RetargetingProblem], and
    [`RetargetingResult`][retarget.results.spec.RetargetingResult] — see [Pipeline](pipeline.md).

!!! tip "Code-block symbol hovers"
    Fenced-code hovers resolve `retarget`, `motion_sync`, and `contact_detection` symbols when `api-symbols.json` is built (`mkdocs build`, or restart `mkdocs serve` after hook changes). Qualified names (for example `RetargetingResult.load_npz`) are always the safest choice when a short token is ambiguous.

| Section | Contents |
|---------|----------|
| [motion_sync](motion-sync.md) | `SyncClip`, sessions, mocap/video/contact schemas (sibling repo) |
| [motion_sync (full)](motion-sync-reference.md) | Full package tree (`show_submodules`) for discoverability |
| [contact_detection](contact-detection.md) | Foot-support classification algorithms (sibling repo) |
| [Pipeline](pipeline.md) | `Retargeter`, problem spec, engine, batch runner |
| [Data models](models.md) | Results, scene, motion, robot specs |
| [Protocols](protocols.md) | Extension interfaces (backends, terms, exporters) |
| [Optimization](optimization.md) | Objectives, constraints, solvers |
| [Registries](registries.md) | Plugin registries for robots, formats, metrics |
| [Enums](enums.md) | `TaskKind`, registry keys, frames, run status |
| [Asset store](asset-store.md) | `AssetStore`, manifests, install requirements |
| [Export types](export-api.md) | `ExportSpec`, `ExportResult`, `Exporter` |
| [Pose and timing](pose.md) | `Pose`, `PoseSequence`, resampling helpers |
| [Full reference](reference.md) | Complete `retarget` package index |

## Public exports

Summary of symbols re-exported from `retarget`:

::: retarget
    options:
      members: false
      show_root_heading: false
      summary: true
