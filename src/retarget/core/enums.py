"""Stable enums used by public configuration and result objects."""

from __future__ import annotations

from enum import StrEnum


class RetargetEnum(StrEnum):
    """Base class for string-valued retarget enums."""


class NameEnum(RetargetEnum):
    """Base class for user-defined symbolic vocabularies."""


class RobotJoint(NameEnum):
    """Base enum for robot joint names."""


class RobotLink(NameEnum):
    """Base enum for robot link names."""


class RobotGeometry(NameEnum):
    """Base enum for robot collision and visual geometry names."""


class SceneGeometry(NameEnum):
    """Base enum for target-independent scene geometry groups."""


class MotionJoint(NameEnum):
    """Base enum for source motion joint names."""


class MocapRigidBody(NameEnum):
    """Base enum for native motion-capture rigid-body names."""


class MocapMarker(NameEnum):
    """Base enum for native motion-capture marker names."""


class ObservationRole(NameEnum):
    """Base enum for target-independent semantic observation roles."""


class RobotRole(NameEnum):
    """Base enum for semantic roles resolved by a robot specification."""


class HumanoidRobotRole(RobotRole):
    """Standard semantic roles exposed by humanoid robot specs."""

    PELVIS = "pelvis"
    TORSO = "torso"
    HEAD = "head"
    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE = "left_ankle"
    LEFT_FOOT = "left_foot"
    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE = "right_ankle"
    RIGHT_FOOT = "right_foot"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class ContactSubject(NameEnum):
    """Base enum for contact subjects such as feet, shoes, or hands."""


class ContactState(NameEnum):
    """Base enum for symbolic contact state labels."""


class ContactPatch(NameEnum):
    """Base enum for contact patch labels."""


class RegistryKind(RetargetEnum):
    """Base class for extensible registry-key vocabularies."""


class ObjectiveKind(RegistryKind):
    """Base enum for objective registrations."""


class ConstraintKind(RegistryKind):
    """Base enum for constraint registrations."""


class SolverKind(RegistryKind):
    """Base enum for solver registrations."""


class RobotKind(RegistryKind):
    """Base enum for robot registrations."""


class RobotProviderKind(RegistryKind):
    """Base enum for robot-provider registrations."""


class MotionFormatKind(RegistryKind):
    """Base enum for motion-format registrations."""


class MotionLoaderKind(RegistryKind):
    """Base enum for motion-loader registrations."""


class ObservationRecipeKindBase(RegistryKind):
    """Base enum for observation-recipe registrations."""


class RetargetingRecipeKindBase(RegistryKind):
    """Base enum for retargeting-recipe registrations."""


class ExporterKind(RegistryKind):
    """Base enum for exporter registrations."""


class VisualizerKind(RegistryKind):
    """Base enum for visualizer registrations."""


class KinematicsKind(RegistryKind):
    """Base enum for kinematics-backend registrations."""


class MetricKind(RegistryKind):
    """Base enum for metric registrations."""


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


class SolverBackend(SolverKind):
    """Optimization backend choices.

    Attributes:
        AUTO (str): Pick an available backend at runtime.
        NUMPY_LEAST_SQUARES (str): SciPy/NumPy least-squares solver.
        CVXPY_CLARABEL (str): CVXPY with Clarabel conic solver.
    """

    AUTO = "auto"
    NUMPY_LEAST_SQUARES = "numpy_least_squares"
    CVXPY_CLARABEL = "cvxpy_clarabel"


class ConvergenceMode(RetargetEnum):
    """SQP inner-loop convergence policy."""

    STEP_NORM = "step_norm"
    COST_PLATEAU = "cost_plateau"
    NONE = "none"


class Robot(RobotKind):
    """Built-in robot registry keys.

    Attributes:
        SYNTHETIC_HUMANOID (str): Lightweight test humanoid.
        G1_LIKE (str): Unitree G1-style humanoid preset.
        T1_LIKE (str): Booster T1-style humanoid preset.
    """

    SYNTHETIC_HUMANOID = "synthetic_humanoid"
    G1_LIKE = "g1_like"
    T1_LIKE = "t1_like"


class RobotProviderName(RobotProviderKind):
    """Built-in robot provider registry keys.

    Attributes:
        REGISTRY (str): Load from built-in registry entries.
        FILE (str): Load robot spec from a file path.
        ASSET_STORE (str): Resolve robot via the asset store manifest.
    """

    REGISTRY = "registry"
    FILE = "file"
    ASSET_STORE = "asset_store"
    HOLOSOMA = "holosoma"


class ObservationRecipeKind(ObservationRecipeKindBase):
    """Built-in observation-recipe kinds for declarative run configs."""

    MOTION_FILE = "motion_file"
    SKATEBOARDING = "skateboarding"
    HOLOSOMA_CLIMB = "holosoma_climb"


class RetargetingRecipeKind(RetargetingRecipeKindBase):
    """Built-in robot-adaptation recipe kinds for declarative run configs."""

    ROLE_MAPPING = "role_mapping"
    SKATEBOARDING = "skateboarding"
    HOLOSOMA_CLIMB = "holosoma_climb"


class TimelineSelection(RetargetEnum):
    """Native timeline selected as the observation sampling grid."""

    HUMAN_POSE = "human_pose"
    MOCAP = "mocap"
    UNIFORM = "uniform"


class CropPolicy(RetargetEnum):
    """How a shared observation timeline handles source support."""

    OVERLAP = "overlap"
    FULL = "full"


class MotionFormat(MotionFormatKind):
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


class MotionLoaderSuffix(MotionLoaderKind):
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


class Objective(ObjectiveKind):
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


class Constraint(ConstraintKind):
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


class ExportFormat(ExporterKind):
    """Built-in exporter registry keys.

    Attributes:
        MUJOCO_NPZ (str): MuJoCo-compatible NPZ export bundle.
    """

    MUJOCO_NPZ = "mujoco_npz"


class VisualizerName(VisualizerKind):
    """Built-in visualizer registry keys.

    Attributes:
        DRY_RUN (str): Print summary without opening a viewer.
        VISER (str): Interactive Viser web visualizer.
    """

    DRY_RUN = "dry_run"
    VISER = "viser"


class KinematicsBackendName(KinematicsKind):
    """Built-in kinematics backend registry keys.

    Attributes:
        SIMPLE (str): Lightweight analytic / table backend.
        MUJOCO (str): MuJoCo model backend.
    """

    SIMPLE = "simple"
    MUJOCO = "mujoco"


class ObjectQposMode(RetargetEnum):
    """How a dynamic object trajectory participates in qpos."""

    APPENDED = "appended"
    EXTERNAL = "external"


class ObjectSampleSpace(RetargetEnum):
    """Coordinate space for object sample points."""

    OBJECT_LOCAL = "object_local"
    OBJECT_ASSET_LOCAL = "object_asset_local"
    SCALED_OBJECT_LOCAL = "scaled_object_local"
    WORLD = "world"


class QposVariableKind(RetargetEnum):
    """Which qpos coordinates the optimizer may change."""

    ACTUATED = "actuated"
    QPOS_SLICE = "qpos_slice"
    QPOS_INDICES = "qpos_indices"


class NonPenetrationSource(RetargetEnum):
    """Sources used by the non-penetration constraint."""

    SUPPORT = "support"
    SCENE_POINTS = "scene_points"
    GEOMETRY = "geometry"


class NominalFallback(RetargetEnum):
    """Fallback target for nominal qpos tracking."""

    ZERO = "zero"
    CURRENT = "current"


class ContactLayerKind(RetargetEnum):
    """Persisted contact layer storage kind."""

    BINARY = "binary"
    CATEGORICAL = "categorical"


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


class MetricName(MetricKind):
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
