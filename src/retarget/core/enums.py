"""Stable enums used by public configuration and result objects."""

from __future__ import annotations

from enum import StrEnum


class TaskKind(StrEnum):
    """Supported high-level retargeting workflows.

    Attributes:
        ROBOT_ONLY (str): Retarget humanoid motion without scene props (``"robot_only"``).
        OBJECT_INTERACTION (str): Retarget with manipulated objects and contact (``"object_interaction"``).
        CLIMBING (str): Climbing / terrain interaction workflow (``"climbing"``).
    """

    ROBOT_ONLY = "robot_only"
    OBJECT_INTERACTION = "object_interaction"
    CLIMBING = "climbing"


class FrameConvention(StrEnum):
    """Coordinate frame conventions understood by the package.

    Attributes:
        Z_UP_RIGHT_HANDED (str): Z-up, right-handed world frame.
        Y_UP_RIGHT_HANDED (str): Y-up, right-handed world frame (common in video/SMPL).
    """

    Z_UP_RIGHT_HANDED = "z_up_right_handed"
    Y_UP_RIGHT_HANDED = "y_up_right_handed"


class QuaternionOrder(StrEnum):
    """Quaternion storage order.

    Attributes:
        WXYZ (str): Scalar-first ``(w, x, y, z)`` layout.
        XYZW (str): Scalar-last ``(x, y, z, w)`` layout (common in graphics APIs).
    """

    WXYZ = "wxyz"
    XYZW = "xyzw"


class SolverBackend(StrEnum):
    """Optimization backend choices.

    Attributes:
        AUTO (str): Pick an available backend at runtime.
        NUMPY_LEAST_SQUARES (str): SciPy/NumPy least-squares solver.
        CVXPY_CLARABEL (str): CVXPY with Clarabel conic solver.
    """

    AUTO = "auto"
    NUMPY_LEAST_SQUARES = "numpy_least_squares"
    CVXPY_CLARABEL = "cvxpy_clarabel"


class Robot(StrEnum):
    """Built-in robot registry keys.

    Attributes:
        SYNTHETIC_HUMANOID (str): Lightweight test humanoid.
        G1_LIKE (str): Unitree G1-style humanoid preset.
        T1_LIKE (str): Booster T1-style humanoid preset.
    """

    SYNTHETIC_HUMANOID = "synthetic_humanoid"
    G1_LIKE = "g1_like"
    T1_LIKE = "t1_like"


class RobotProviderName(StrEnum):
    """Built-in robot provider registry keys.

    Attributes:
        REGISTRY (str): Load from built-in registry entries.
        FILE (str): Load robot spec from a file path.
        ASSET_STORE (str): Resolve robot via the asset store manifest.
    """

    REGISTRY = "registry"
    FILE = "file"
    ASSET_STORE = "asset_store"


class MotionFormat(StrEnum):
    """Built-in motion format registry keys.

    Attributes:
        MINIMAL (str): Minimal joint-name + qpos arrays.
        SMPLH (str): SMPL-H pose parameters.
        LAFAN (str): LaFAN1-style skeleton dumps.
        MOCAP (str): Generic mocap tables.
        SMPLX (str): SMPL-X body pose and shape.
    """

    MINIMAL = "minimal"
    SMPLH = "smplh"
    LAFAN = "lafan"
    MOCAP = "mocap"
    SMPLX = "smplx"


class MotionLoaderSuffix(StrEnum):
    """Built-in motion loader suffix registry keys.

    Attributes:
        JSON (str): ``.json`` loader.
        CSV (str): ``.csv`` loader.
        NPY (str): ``.npy`` loader.
        NPZ (str): ``.npz`` loader.
    """

    JSON = ".json"
    CSV = ".csv"
    NPY = ".npy"
    NPZ = ".npz"


class Objective(StrEnum):
    """Built-in objective term registry keys.

    Attributes:
        LAPLACIAN (str): Interaction-mesh Laplacian deformation objective.
        LINK_TRACKING (str): Track named robot links to per-frame world-space target points.
        SMOOTHNESS (str): Temporal smoothness on generalized coordinates.
        NOMINAL_TRACKING (str): Track a nominal pose trajectory.
        DIAGONAL_REGULARIZATION (str): Penalize selected qpos coordinates toward zero.
    """

    LAPLACIAN = "laplacian"
    LINK_TRACKING = "link_tracking"
    SMOOTHNESS = "smoothness"
    NOMINAL_TRACKING = "nominal_tracking"
    DIAGONAL_REGULARIZATION = "diagonal_regularization"


class Constraint(StrEnum):
    """Built-in constraint term registry keys.

    Attributes:
        JOINT_LIMITS (str): Enforce joint limits.
        TRUST_REGION (str): Limit per-step configuration change.
        FOOT_STICKING (str): Foot support consistency constraints.
        FOOT_LOCK (str): Lock feet during planted phases.
        NON_PENETRATION (str): Prevent interpenetration with scene geometry.
        SELF_COLLISION (str): Self-collision avoidance.
    """

    JOINT_LIMITS = "joint_limits"
    TRUST_REGION = "trust_region"
    FOOT_STICKING = "foot_sticking"
    FOOT_LOCK = "foot_lock"
    NON_PENETRATION = "non_penetration"
    SELF_COLLISION = "self_collision"


class ExportFormat(StrEnum):
    """Built-in exporter registry keys.

    Attributes:
        MUJOCO_NPZ (str): MuJoCo-compatible NPZ export bundle.
    """

    MUJOCO_NPZ = "mujoco_npz"


class VisualizerName(StrEnum):
    """Built-in visualizer registry keys.

    Attributes:
        DRY_RUN (str): Print summary without opening a viewer.
        VISER (str): Interactive Viser web visualizer.
    """

    DRY_RUN = "dry_run"
    VISER = "viser"


class KinematicsBackendName(StrEnum):
    """Built-in kinematics backend registry keys.

    Attributes:
        SIMPLE (str): Lightweight analytic / table backend.
        MUJOCO (str): MuJoCo model backend.
    """

    SIMPLE = "simple"
    MUJOCO = "mujoco"


class ContactMode(StrEnum):
    """How contact constraints are inferred or supplied.

    Attributes:
        DISABLED (str): Do not infer or apply contact constraints.
        VELOCITY (str): Infer contact from foot velocity thresholds.
        HEIGHT (str): Infer contact from height above support.
        EXPLICIT_WINDOWS (str): Use user-supplied contact time windows.
    """

    DISABLED = "disabled"
    VELOCITY = "velocity"
    HEIGHT = "height"
    EXPLICIT_WINDOWS = "explicit_windows"


class AssetKind(StrEnum):
    """Types of assets tracked by an asset manifest.

    Attributes:
        ROBOT (str): Robot model assets.
        OBJECT (str): Manipulated object meshes or trajectories.
        TERRAIN (str): Terrain / environment geometry.
        MOTION (str): Source motion clips.
        FIXTURE (str): Scene fixtures and props.
    """

    ROBOT = "robot"
    OBJECT = "object"
    TERRAIN = "terrain"
    MOTION = "motion"
    FIXTURE = "fixture"


class RunStatus(StrEnum):
    """Retargeting or evaluation lifecycle state.

    Attributes:
        SUCCESS (str): Completed successfully.
        PARTIAL (str): Completed with recoverable issues.
        FAILED (str): Failed before producing a usable result.
        SKIPPED (str): Skipped (for example missing inputs).
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class MetricName(StrEnum):
    """Built-in evaluation metric names.

    Attributes:
        OPTIMIZATION_COST (str): Final optimization cost.
        FOOT_SLIDING (str): Foot sliding during contact.
        CONTACT_PRESERVATION (str): Agreement with input contact signal.
        PENETRATION (str): Mesh penetration depth metric.
    """

    OPTIMIZATION_COST = "optimization_cost"
    FOOT_SLIDING = "foot_sliding"
    CONTACT_PRESERVATION = "contact_preservation"
    PENETRATION = "penetration"
