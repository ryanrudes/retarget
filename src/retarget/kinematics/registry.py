"""Kinematics backend registries."""

from __future__ import annotations

from collections.abc import Callable

from retarget.core.protocols import KinematicsBackend
from retarget.core.registry import Registry
from retarget.robots.spec import RobotSpec

KinematicsBackendFactory = Callable[[RobotSpec], KinematicsBackend]
"""Callable that constructs a :class:`~retarget.core.protocols.KinematicsBackend` for a robot spec."""

kinematics_backends: Registry[KinematicsBackendFactory] = Registry("kinematics backend")
"""Registry of kinematics backend factories keyed by backend name."""

__all__ = ["KinematicsBackendFactory", "kinematics_backends"]
