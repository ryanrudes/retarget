# API Overview

Generated from docstrings in the installed package. Search for a symbol name or
use the sections below.

!!! tip "Where to start"
    Capture workflows compose [`ObservationRecipe`][retarget.pipeline.recipe.ObservationRecipe],
    [`RetargetingRecipe`][retarget.pipeline.recipe.RetargetingRecipe], and
    [`RetargetingExperiment`][retarget.pipeline.experiment.RetargetingExperiment].

!!! tip "Code-block symbol hovers"
    Fenced-code hovers resolve `retarget` symbols after `mkdocs build` or a
    restart of `mkdocs serve`. Qualified names are safest when a short symbol is
    ambiguous.

| Section | Contents |
|---------|----------|
| [Capture](capture.md) | Native recordings, tracks, sources, estimators, alignment |
| [Recipes](recipes.md) | Observation and robot-adaptation recipes |
| [Pipeline](pipeline.md) | Experiment orchestration, problems, retargeter, batch runner |
| [Data models](models.md) | Observations, results, scene, motion, robot specs |
| [Protocols](protocols.md) | Extension interfaces |
| [Optimization](optimization.md) | Objectives, constraints, solvers |
| [Registries](registries.md) | Plugin registries for robots, formats, metrics |
| [Enums](enums.md) | Typed vocabularies and registry keys |
| [Asset store](asset-store.md) | Assets, manifests, install requirements |
| [Export types](export-api.md) | Export specs, results, and protocol |
| [Pose and timing](pose.md) | Poses, quaternion order, resampling |
| [Full reference](reference.md) | Complete `retarget` package index |

## Public Exports

::: retarget
    options:
      members: false
      show_root_heading: false
      summary: true
