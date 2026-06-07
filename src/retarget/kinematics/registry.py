"""Kinematics backend registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from retarget.core.enums import KinematicsKind
from retarget.core.protocols import KinematicsBackend
from retarget.core.registry import Registry

if TYPE_CHECKING:
    from retarget.pipeline.compiled import CompiledRobotSpec
else:
    CompiledRobotSpec = Any

KinematicsBackendFactory = Callable[[CompiledRobotSpec], KinematicsBackend]
"""Callable that constructs a :class:`~retarget.core.protocols.KinematicsBackend` for a robot spec."""

kinematics_backends: Registry[KinematicsKind, KinematicsBackendFactory] = Registry(
    "kinematics backend",
    KinematicsKind,
)
"""Registry of kinematics backend factories keyed by backend name."""

__all__ = ["KinematicsBackendFactory", "kinematics_backends"]
