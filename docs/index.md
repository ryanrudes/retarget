---
hide:
  - toc
---

# retarget

`retarget` is a typed research toolkit for processing heterogeneous capture and
retargeting the resulting scene observation to robots. Native recordings,
alignment, semantic contacts, robot adaptation, optimization, export, and
visualization use one public abstraction hierarchy.

!!! tip "First time here?"
    Start with [Quickstart](quickstart.md), then read
    [Introduction](introduction.md) for the complete capture-to-retarget flow.
    [Architecture](architecture.md) explains the domain boundaries.

!!! note "Live code (local only)"
    Runnable blocks require local `uv run mkdocs serve`. Hosted documentation is
    read-only.

=== "CLI"

    ```bash
    uv sync --extra dev
    uv run retarget doctor
    uv run retarget run --config examples/basic/run_config.toml
    ```

=== "Python"

    ```python
    from retarget import RetargetingExperiment

    result = RetargetingExperiment(
        observation=observation_recipe,
        recipe=retargeting_recipe,
        robot=robot,
    ).run()
    ```

Run configs deserialize into the same observation recipe, retargeting recipe,
robot spec, and `RetargetingExperiment`; they are not a second workflow.

## Where To Go Next

| Goal | Page |
|------|------|
| Run a small experiment | [Quickstart](quickstart.md) |
| Process heterogeneous sensors | [Introduction](introduction.md) |
| Understand the abstraction hierarchy | [Architecture](architecture.md) |
| Follow the capture-to-retarget workflow | [Ecosystem](ecosystem/index.md) |
| Work with frames and clocks | [Coordinate Conventions](coordinate-conventions.md) |
| Extend formats, robots, or recipes | [Extending](extending.md) |
| Browse the API | [API](api/index.md) |
