"""Kinematics backend registries."""

from __future__ import annotations

from collections.abc import Callable

from retarget.core.protocols import KinematicsBackend
from retarget.core.registry import Registry
from retarget.robots.spec import RobotSpec

KinematicsBackendFactory = Callable[[RobotSpec], KinematicsBackend]

kinematics_backends: Registry[KinematicsBackendFactory] = Registry("kinematics backend")

__all__ = ["KinematicsBackendFactory", "kinematics_backends"]
