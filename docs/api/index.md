# API overview

Generated from docstrings in the installed package. Use the sidebar sections below, or search (++ctrl+k++ / ++cmd+k++) for a symbol name.

!!! tip "Where to start"
    Most workflows use [`Retargeter`][retarget.pipeline.retargeter.Retargeter],
    [`RetargetingProblem`][retarget.pipeline.problem.RetargetingProblem], and
    [`RetargetingResult`][retarget.results.spec.RetargetingResult] — see [Pipeline](pipeline.md).

| Section | Contents |
|---------|----------|
| [Pipeline](pipeline.md) | `Retargeter`, problem spec, engine, batch runner |
| [Data models](models.md) | Results, scene, motion, robot specs |
| [Protocols](protocols.md) | Extension interfaces (backends, terms, exporters) |
| [Optimization](optimization.md) | Objectives, constraints, solvers |
| [Registries](registries.md) | Plugin registries for robots, formats, metrics |
| [Full reference](reference.md) | Complete `retarget` package index |

## Public exports

Summary of symbols re-exported from `retarget`:

::: retarget
    options:
      members: false
      show_root_heading: false
      summary: true
