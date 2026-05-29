"""Stable enums used by public configuration and result objects."""

from __future__ import annotations

from enum import StrEnum


class TaskKind(StrEnum):
    """Supported high-level retargeting workflows."""

    ROBOT_ONLY = "robot_only"
    OBJECT_INTERACTION = "object_interaction"
    CLIMBING = "climbing"


class FrameConvention(StrEnum):
    """Coordinate frame conventions understood by the package."""

    Z_UP_RIGHT_HANDED = "z_up_right_handed"
    Y_UP_RIGHT_HANDED = "y_up_right_handed"


class QuaternionOrder(StrEnum):
    """Quaternion storage order."""

    WXYZ = "wxyz"
    XYZW = "xyzw"


class SolverBackend(StrEnum):
    """Optimization backend choices."""

    AUTO = "auto"
    NUMPY_LEAST_SQUARES = "numpy_least_squares"
    CVXPY_CLARABEL = "cvxpy_clarabel"


class ContactMode(StrEnum):
    """How contact constraints are inferred or supplied."""

    DISABLED = "disabled"
    VELOCITY = "velocity"
    HEIGHT = "height"
    EXPLICIT_WINDOWS = "explicit_windows"


class AssetKind(StrEnum):
    """Types of assets tracked by an asset manifest."""

    ROBOT = "robot"
    OBJECT = "object"
    TERRAIN = "terrain"
    MOTION = "motion"
    FIXTURE = "fixture"


class RunStatus(StrEnum):
    """Retargeting or evaluation lifecycle state."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class MetricName(StrEnum):
    """Built-in evaluation metric names."""

    OPTIMIZATION_COST = "optimization_cost"
    FOOT_SLIDING = "foot_sliding"
    CONTACT_PRESERVATION = "contact_preservation"
    PENETRATION = "penetration"
