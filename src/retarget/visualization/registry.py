"""Visualization extension registries."""

from __future__ import annotations

from retarget.core.protocols import Visualizer
from retarget.core.registry import Registry

visualizers: Registry[Visualizer] = Registry("visualizer")

__all__ = ["visualizers"]
