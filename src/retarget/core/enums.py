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


class Robot(StrEnum):
    """Built-in robot registry keys."""

    SYNTHETIC_HUMANOID = "synthetic_humanoid"
    G1_LIKE = "g1_like"
    T1_LIKE = "t1_like"


class RobotProviderName(StrEnum):
    """Built-in robot provider registry keys."""

    REGISTRY = "registry"
    FILE = "file"
    ASSET_STORE = "asset_store"


class MotionFormat(StrEnum):
    """Built-in motion format registry keys."""

    MINIMAL = "minimal"
    SMPLH = "smplh"
    LAFAN = "lafan"
    MOCAP = "mocap"
    SMPLX = "smplx"


class MotionLoaderSuffix(StrEnum):
    """Built-in motion loader suffix registry keys."""

    JSON = ".json"
    CSV = ".csv"
    NPY = ".npy"
    NPZ = ".npz"


class Objective(StrEnum):
    """Built-in objective term registry keys."""

    LAPLACIAN = "laplacian"
    SMOOTHNESS = "smoothness"
    NOMINAL_TRACKING = "nominal_tracking"


class Constraint(StrEnum):
    """Built-in constraint term registry keys."""

    JOINT_LIMITS = "joint_limits"
    TRUST_REGION = "trust_region"
    FOOT_CONTACT = "foot_contact"
    FOOT_LOCK = "foot_lock"
    NON_PENETRATION = "non_penetration"
    SELF_COLLISION = "self_collision"


class ExportFormat(StrEnum):
    """Built-in exporter registry keys."""

    MUJOCO_NPZ = "mujoco_npz"


class VisualizerName(StrEnum):
    """Built-in visualizer registry keys."""

    DRY_RUN = "dry_run"
    VISER = "viser"


class KinematicsBackendName(StrEnum):
    """Built-in kinematics backend registry keys."""

    SIMPLE = "simple"
    MUJOCO = "mujoco"


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
