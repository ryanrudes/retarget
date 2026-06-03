"""Export registries.

``exporters`` maps registered format names (for example ``"mujoco_npz"``) to
:class:`~retarget.core.protocols.Exporter` implementations. Use
:meth:`~retarget.core.registry.Registry.get` to resolve an exporter and
:meth:`~retarget.core.registry.Registry.register` to add one.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from typing import cast

from retarget.core.protocols import Exporter
from retarget.core.registry import Registry


def _exporter_from_decorator(value: object) -> Exporter:
    candidate = value
    if isclass(value) or not isinstance(value, Exporter):
        if not callable(value):
            raise TypeError("exporter registrations must implement Exporter or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, Exporter):
        raise TypeError("exporter registrations must implement Exporter")
    return candidate


exporters: Registry[Exporter] = Registry("exporter", decorator_transform=_exporter_from_decorator)

__all__ = ["exporters"]
