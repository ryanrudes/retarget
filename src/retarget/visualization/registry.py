"""Visualization extension registries.

``visualizers`` maps visualizer names (for example ``"dry_run"``, ``"viser"``) to
:class:`~retarget.core.protocols.Visualizer` implementations. Use
:meth:`~retarget.core.registry.Registry.get` to resolve a visualizer and
:meth:`~retarget.core.registry.Registry.register` to add one.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import cast

from retarget.core.enums import VisualizerKind
from retarget.core.protocols import Visualizer
from retarget.core.registry import Registry


def _visualizer_from_decorator(value: object) -> Visualizer:
    candidate = value
    if isclass(value) or not isinstance(value, Visualizer):
        if not callable(value):
            raise TypeError("visualizer registrations must implement Visualizer or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, Visualizer):
        raise TypeError("visualizer registrations must implement Visualizer")
    return candidate


visualizers: Registry[VisualizerKind, Visualizer] = Registry(
    "visualizer",
    VisualizerKind,
    decorator_transform=_visualizer_from_decorator,
)

__all__ = ["visualizers"]
