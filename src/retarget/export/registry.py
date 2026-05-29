"""Export registries."""

from __future__ import annotations

from retarget.core.protocols import Exporter
from retarget.core.registry import Registry

exporters: Registry[Exporter] = Registry("exporter")

__all__ = ["exporters"]
