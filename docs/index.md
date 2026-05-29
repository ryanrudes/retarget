---
hide:
  - toc
---

# retarget

`retarget` is a standalone research toolkit for motion retargeting: typed specs, explicit coordinate conventions, a high-level runner, CLI workflows, extension registries, metrics, exporters, and visualization hooks.

The implementation is independent of the holosoma reference repository.

!!! tip "First time here?"
    Jump to [Quickstart](quickstart.md) for install, a one-command run, and CI checks. Skim [Architecture](architecture.md) when you want the mental model. With Jupyter running locally, flip **Live** on (bottom-right), then use the icons on the Python tab to run code.

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
| Understand the pipeline | [Architecture](architecture.md) |
| Frames, units, conventions | [Coordinate Conventions](coordinate-conventions.md) |
| Extend the toolkit | [Extending](adding-a-robot.md) |
| API reference | [API](api/index.md) |

Press ++ctrl+f++ (or ++cmd+f++) to search — results are highlighted and shareable. Use the **back-to-top** button after long pages; the header tucks away while you scroll.
