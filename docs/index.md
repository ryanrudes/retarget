---
hide:
  - toc
---

# retarget

`retarget` is a standalone research toolkit for motion retargeting: typed specs, explicit coordinate conventions, a high-level runner, CLI workflows, extension registries, metrics, exporters, and visualization hooks.

The implementation is independent of the holosoma reference repository.

!!! tip "First time here?"
    Jump to [Quickstart](quickstart.md) for install, a one-command run, and CI checks. Read [Introduction](introduction.md) for a complete research use case (what to build upstream vs. what `retarget` consumes). Follow the [Tutorials](tutorials/index.md) for shorter step-by-step lessons. Skim [Architecture](architecture.md) when you want the mental model. With Jupyter running locally, flip **Live** in the page header (left of the search bar), then use the icons on Python tabs to run code — see [Interactive playground](interactive-playground.md).

!!! note "Live code (local only)"
    Runnable blocks and Jupyter work only with local `uv run mkdocs serve`. The [hosted docs](https://ryanrudes.github.io/retarget/) are read-only (no Live switch or kernel).

=== "CLI"

    ```bash
    uv sync --extra dev
    uv run retarget doctor
    uv run retarget run \
      --motion tests/fixtures/minimal_motion.json \
      --format minimal \
      --robot synthetic_humanoid \
      --output result.npz
    ```

=== "Python"

    ```python
    from retarget import TaskKind

    [task.value for task in TaskKind]
    ```

    For a full run, build a `RetargetingProblem` and call `Retargeter().run(problem)` — see [Quickstart](quickstart.md).

!!! info "Asset manifests"
    Register local robot, object, terrain, and fixture assets without vendoring upstream datasets into the package. See [Assets](assets.md).

## Where to go next

| Goal | Page |
|------|------|
| Run something in 60 seconds | [Quickstart](quickstart.md) |
| Map a real project (sensors → problem) | [Introduction](introduction.md) |
| Learn by doing (guided paths) | [Tutorials](tutorials/index.md) |
| Understand the pipeline (in-repo engine and types) | [Architecture](architecture.md) |
| End-to-end lab workflow (capture → clip → retarget) | [Ecosystem](ecosystem/index.md) · [Pipeline](ecosystem/pipeline.md) |
| Frames, units, conventions | [Coordinate Conventions](coordinate-conventions.md) |
| Extend the toolkit | [Extending](extending.md) |
| API reference | [API](api/index.md) |
| motion_sync API (vendored submodule) | [motion_sync](api/motion-sync.md) |

Press ++ctrl+f++ (or ++cmd+f++) to search — results are highlighted and shareable. Use the **back-to-top** button after long pages; the header tucks away while you scroll.
