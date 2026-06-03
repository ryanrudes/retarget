"""Rich progress helpers for retargeting runs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


FrameAdvance = Callable[[], None]


@contextmanager
def frame_progress(
    *,
    enabled: bool,
    description: str,
    total: int,
    console: Console | None = None,
) -> Iterator[FrameAdvance]:
    """Yield a callback that advances a Rich frame task by one step.

    When ``enabled`` is false or ``total`` is zero, the callback is a no-op.
    """

    if not enabled or total <= 0:

        def noop() -> None:
            return

        yield noop
        return

    from rich.console import Console as RichConsole
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console or RichConsole(),
        transient=True,
    )
    task_id = progress.add_task(description, total=total)

    def advance() -> None:
        progress.advance(task_id)

    with progress:
        yield advance
