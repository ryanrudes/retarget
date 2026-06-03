"""Built-in robot registry.

``robots`` maps built-in robot keys (for example ``"synthetic_humanoid"``) to
:class:`~retarget.robots.spec.RobotSpec` presets.

``robot_providers`` maps provider names (``"registry"``, ``"file"``, ``"asset_store"``) to
:class:`~retarget.core.protocols.RobotProvider` implementations that resolve specs from
registry keys, files, or the asset store.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from pathlib import Path
from typing import Any, cast

from retarget.assets import AssetStore
from retarget.core.enums import AssetKind, Robot, RobotProviderName
from retarget.core.protocols import RobotProvider
from retarget.core.registry import Registry
from retarget.robots.spec import RobotSpec


def _robot_spec_from_decorator(value: object) -> RobotSpec:
    candidate = value() if callable(value) and not isinstance(value, RobotSpec) else value
    if not isinstance(candidate, RobotSpec):
        raise TypeError("robot decorators must return RobotSpec")
    return candidate


def _robot_provider_from_decorator(value: object) -> RobotProvider:
    candidate = value
    if isclass(value) or not isinstance(value, RobotProvider):
        if not callable(value):
            raise TypeError("robot provider registrations must implement RobotProvider or be zero-argument factories")
        candidate = cast(Callable[[], object], value)()
    if not isinstance(candidate, RobotProvider):
        raise TypeError("robot provider registrations must implement RobotProvider")
    return candidate


robots: Registry[RobotSpec] = Registry("robot", decorator_transform=_robot_spec_from_decorator)
robot_providers: Registry[RobotProvider] = Registry(
    "robot provider",
    decorator_transform=_robot_provider_from_decorator,
)


class RegistryRobotProvider:
    """Resolve robot specs from the built-in robot registry."""

    def load(self, name: str, **kwargs: Any) -> RobotSpec:
        """Load a preset robot spec registered in ``robots``.

        Args:
            name (str): Built-in robot registry key (for example ``"synthetic_humanoid"``).
            **kwargs (Any): Ignored; raises if any extra options are passed.

        Returns:
            RobotSpec: Registered robot description.

        Raises:
            ValueError: If unsupported keyword options are supplied.
            KeyError: If ``name`` is not registered in ``robots``.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"RegistryRobotProvider does not accept extra options: {unknown}")
        return robots.get(name)


class FileRobotProvider:
    """Load robot specs from explicit TOML/YAML/JSON files."""

    def load(self, name: str, **kwargs: Any) -> RobotSpec:
        """Load a robot spec from a file path.

        Args:
            name (str): Registry alias or filesystem path to the spec file.
            **kwargs (Any): Provider options; supports ``path`` to override ``name`` as the file
                location.

        Returns:
            RobotSpec: Parsed spec with relative asset paths resolved against the spec directory.
            When ``name`` differs from the file path and the spec's own ``name``, the returned
            spec is copied with ``name`` set to the requested alias.

        Raises:
            ValueError: If unsupported keyword options are supplied.
            FileNotFoundError: If the spec file does not exist.
        """
        path_value = kwargs.pop("path", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"FileRobotProvider does not accept options: {unknown}")
        path = Path(path_value) if path_value is not None else Path(name)
        spec = RobotSpec.load(path)
        return spec if name == str(path) or name == spec.name else spec.model_copy(update={"name": name})


class AssetStoreRobotProvider:
    """Load robot specs from robot assets tracked in an `AssetStore`.

    Attributes:
        SPEC_FILENAMES (tuple[str, ...]): Default robot spec filenames searched inside a robot
            asset directory when ``spec_filename`` is not provided.
    """

    SPEC_FILENAMES: tuple[str, ...] = ("robot.toml", "robot.yaml", "robot.yml", "robot.json")

    def load(self, name: str, **kwargs: Any) -> RobotSpec:
        """Load a robot spec from a robot asset registered in an :class:`~retarget.assets.store.AssetStore`.

        Args:
            name (str): Asset manifest name with kind :attr:`~retarget.core.enums.AssetKind.ROBOT`.
            **kwargs (Any): Provider options. ``store`` (default ``".retarget_assets"``) selects the
                asset store root. ``spec_filename`` overrides :attr:`SPEC_FILENAMES` lookup.

        Returns:
            RobotSpec: Parsed spec with relative paths resolved against the spec file directory.

        Raises:
            ValueError: If the asset kind is not ``robot`` or unsupported options are passed.
            KeyError: If ``name`` is not present in the store manifest.
            FileNotFoundError: If no spec file is found under the asset path.
        """
        store = AssetStore(kwargs.pop("store", ".retarget_assets"))
        spec_filename = str(kwargs.pop("spec_filename", ""))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"AssetStoreRobotProvider does not accept options: {unknown}")
        record = store.load().find(name)
        if record is None:
            raise KeyError(f"Robot asset {name!r} is not registered in {store.root}")
        if record.kind != AssetKind.ROBOT:
            raise ValueError(f"Asset {name!r} has kind {record.kind.value!r}, expected robot")
        spec_path = _robot_spec_path(record.path, spec_filename)
        return RobotSpec.load(spec_path)


def _robot_spec_path(asset_path: Path, spec_filename: str = "") -> Path:
    if asset_path.is_file():
        return asset_path
    if spec_filename:
        candidate = asset_path / spec_filename
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    for filename in AssetStoreRobotProvider.SPEC_FILENAMES:
        candidate = asset_path / filename
        if candidate.exists():
            return candidate
    expected = ", ".join(AssetStoreRobotProvider.SPEC_FILENAMES)
    raise FileNotFoundError(f"No robot spec found in {asset_path}; expected one of: {expected}")


SYNTHETIC_JOINTS = (
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "spine_yaw",
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
)

G1_LIKE_JOINTS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

T1_LIKE_JOINTS = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "torso_yaw",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist",
)

HUMANOID_LINKS = (
    "pelvis",
    "torso",
    "head",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
)

HUMANOID_LINK_MAPPING = {
    "Pelvis": "pelvis",
    "Spine": "torso",
    "Head": "head",
    "L_Toe": "left_foot",
    "R_Toe": "right_foot",
    "L_Wrist": "left_hand",
    "R_Wrist": "right_hand",
}

HUMANOID_JOINT_MAPPING = {
    "L_Hip": "left_hip_pitch",
    "L_Knee": "left_knee",
    "L_Toe": "left_ankle_pitch",
    "R_Hip": "right_hip_pitch",
    "R_Knee": "right_knee",
    "R_Toe": "right_ankle_pitch",
    "Spine": "waist_yaw",
    "L_Wrist": "left_elbow",
    "R_Wrist": "right_elbow",
}


def _humanoid_limits(joint_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for name in joint_names:
        if "knee" in name:
            limits[name] = (-0.1, 2.7)
        elif "ankle" in name:
            limits[name] = (-1.2, 1.2)
        elif "hip_roll" in name or "shoulder_roll" in name:
            limits[name] = (-1.0, 1.0)
        elif "waist" in name or "torso" in name:
            limits[name] = (-1.5, 1.5)
        elif "wrist" in name:
            limits[name] = (-2.0, 2.0)
        else:
            limits[name] = (-3.14, 3.14)
    return limits

robots.register(
    Robot.SYNTHETIC_HUMANOID,
    RobotSpec(
        name=Robot.SYNTHETIC_HUMANOID.value,
        dof=len(SYNTHETIC_JOINTS),
        height_m=1.2,
        joint_names=SYNTHETIC_JOINTS,
        link_names=(
            "pelvis",
            "left_toe",
            "right_toe",
            "left_hand",
            "right_hand",
        ),
        contact_links=("left_toe", "right_toe"),
        nominal_tracking_joints=SYNTHETIC_JOINTS[:7],
        joint_limits={name: (-2.0, 2.0) for name in SYNTHETIC_JOINTS},
        default_joint_mapping={
            "L_Hip": "left_hip_pitch",
            "L_Knee": "left_knee",
            "L_Toe": "left_ankle",
            "R_Hip": "right_hip_pitch",
            "R_Knee": "right_knee",
            "R_Toe": "right_ankle",
            "Spine": "spine_yaw",
            "L_Wrist": "left_elbow",
            "R_Wrist": "right_elbow",
        },
        default_link_mapping={
            "Pelvis": "pelvis",
            "L_Toe": "left_toe",
            "R_Toe": "right_toe",
            "L_Wrist": "left_hand",
            "R_Wrist": "right_hand",
        },
        metadata={"fixture": True},
    ),
)

robots.register(
    Robot.G1_LIKE,
    RobotSpec(
        name=Robot.G1_LIKE.value,
        dof=len(G1_LIKE_JOINTS),
        height_m=1.32,
        joint_names=G1_LIKE_JOINTS,
        link_names=HUMANOID_LINKS,
        contact_links=("left_foot", "right_foot"),
        nominal_tracking_joints=G1_LIKE_JOINTS[:19],
        joint_limits=_humanoid_limits(G1_LIKE_JOINTS),
        default_joint_mapping=HUMANOID_JOINT_MAPPING,
        default_link_mapping=HUMANOID_LINK_MAPPING,
        metadata={
            "asset_required": True,
            "description": (
                "G1-like humanoid template; attach local URDF/MJCF assets through a file or asset-store spec."
            ),
        },
    ),
)

robots.register(
    Robot.T1_LIKE,
    RobotSpec(
        name=Robot.T1_LIKE.value,
        dof=len(T1_LIKE_JOINTS),
        height_m=1.2,
        joint_names=T1_LIKE_JOINTS,
        link_names=HUMANOID_LINKS,
        contact_links=("left_foot", "right_foot"),
        nominal_tracking_joints=(
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
            "torso_yaw",
            "left_shoulder_pitch",
            "left_elbow",
            "right_shoulder_pitch",
            "right_elbow",
        ),
        joint_limits=_humanoid_limits(T1_LIKE_JOINTS),
        default_joint_mapping=HUMANOID_JOINT_MAPPING | {"Spine": "torso_yaw"},
        default_link_mapping=HUMANOID_LINK_MAPPING,
        metadata={
            "asset_required": True,
            "description": (
                "T1-like humanoid template; attach local URDF/MJCF assets through a file or asset-store spec."
            ),
        },
    ),
)

robot_providers.register(RobotProviderName.REGISTRY, RegistryRobotProvider())
robot_providers.register(RobotProviderName.FILE, FileRobotProvider())
robot_providers.register(RobotProviderName.ASSET_STORE, AssetStoreRobotProvider())
