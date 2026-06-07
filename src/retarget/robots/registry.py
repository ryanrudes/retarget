"""Built-in robot vocabularies, specifications, and providers."""

from __future__ import annotations

from collections.abc import Callable
from inspect import isclass
from pathlib import Path
from typing import Any, TypeVar, cast

from retarget.assets import AssetStore
from retarget.core.enums import (
    AssetKind,
    HumanoidRobotRole,
    Robot,
    RobotGeometry,
    RobotJoint,
    RobotKind,
    RobotLink,
    RobotProviderKind,
    RobotProviderName,
)
from retarget.core.protocols import RobotProvider
from retarget.core.registry import Registry
from retarget.robots.spec import AnyRobotSpec, RobotSpec, RobotVocabulary, SimpleKinematicPoint


class SyntheticJoint(RobotJoint):
    """Actuated joints for the deterministic synthetic humanoid."""

    LEFT_HIP_PITCH = "left_hip_pitch"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_HIP_PITCH = "right_hip_pitch"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE = "right_ankle"
    SPINE_YAW = "spine_yaw"
    LEFT_SHOULDER = "left_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_SHOULDER = "right_shoulder"
    RIGHT_ELBOW = "right_elbow"


class SyntheticLink(RobotLink):
    """Links for the deterministic synthetic humanoid."""

    PELVIS = "pelvis"
    TORSO = "torso"
    LEFT_HIP = "left_hip"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE = "left_ankle"
    LEFT_TOE = "left_toe"
    RIGHT_HIP = "right_hip"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE = "right_ankle"
    RIGHT_TOE = "right_toe"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


class SyntheticGeometry(RobotGeometry):
    """Synthetic fixture has no model-backed geometry."""


class G1LikeJoint(RobotJoint):
    """G1-like actuated-joint vocabulary."""

    LEFT_HIP_PITCH = "left_hip_pitch"
    LEFT_HIP_ROLL = "left_hip_roll"
    LEFT_HIP_YAW = "left_hip_yaw"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE_PITCH = "left_ankle_pitch"
    LEFT_ANKLE_ROLL = "left_ankle_roll"
    RIGHT_HIP_PITCH = "right_hip_pitch"
    RIGHT_HIP_ROLL = "right_hip_roll"
    RIGHT_HIP_YAW = "right_hip_yaw"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch"
    RIGHT_ANKLE_ROLL = "right_ankle_roll"
    WAIST_YAW = "waist_yaw"
    WAIST_ROLL = "waist_roll"
    WAIST_PITCH = "waist_pitch"
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw"
    LEFT_ELBOW = "left_elbow"
    LEFT_WRIST_ROLL = "left_wrist_roll"
    LEFT_WRIST_PITCH = "left_wrist_pitch"
    LEFT_WRIST_YAW = "left_wrist_yaw"
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_WRIST_ROLL = "right_wrist_roll"
    RIGHT_WRIST_PITCH = "right_wrist_pitch"
    RIGHT_WRIST_YAW = "right_wrist_yaw"


class T1LikeJoint(RobotJoint):
    """T1-like actuated-joint vocabulary."""

    LEFT_HIP_PITCH = "left_hip_pitch"
    LEFT_HIP_ROLL = "left_hip_roll"
    LEFT_HIP_YAW = "left_hip_yaw"
    LEFT_KNEE = "left_knee"
    LEFT_ANKLE_PITCH = "left_ankle_pitch"
    LEFT_ANKLE_ROLL = "left_ankle_roll"
    RIGHT_HIP_PITCH = "right_hip_pitch"
    RIGHT_HIP_ROLL = "right_hip_roll"
    RIGHT_HIP_YAW = "right_hip_yaw"
    RIGHT_KNEE = "right_knee"
    RIGHT_ANKLE_PITCH = "right_ankle_pitch"
    RIGHT_ANKLE_ROLL = "right_ankle_roll"
    TORSO_YAW = "torso_yaw"
    LEFT_SHOULDER_PITCH = "left_shoulder_pitch"
    LEFT_SHOULDER_ROLL = "left_shoulder_roll"
    LEFT_SHOULDER_YAW = "left_shoulder_yaw"
    LEFT_ELBOW = "left_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_SHOULDER_PITCH = "right_shoulder_pitch"
    RIGHT_SHOULDER_ROLL = "right_shoulder_roll"
    RIGHT_SHOULDER_YAW = "right_shoulder_yaw"
    RIGHT_ELBOW = "right_elbow"
    RIGHT_WRIST = "right_wrist"


class HumanoidLink(RobotLink):
    """Shared link vocabulary for built-in humanoid presets."""

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


class HumanoidGeometry(RobotGeometry):
    """Template robots leave geometry to model-backed specifications."""


JointT = TypeVar("JointT", bound=RobotJoint)
LinkT = TypeVar("LinkT", bound=RobotLink)


def _robot_spec_from_decorator(value: object) -> AnyRobotSpec:
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


robots: Registry[RobotKind, AnyRobotSpec] = Registry(
    "robot",
    RobotKind,
    decorator_transform=_robot_spec_from_decorator,
)
robot_providers: Registry[RobotProviderKind, RobotProvider] = Registry(
    "robot provider",
    RobotProviderKind,
    decorator_transform=_robot_provider_from_decorator,
)


class RegistryRobotProvider:
    """Resolve robot specs from the robot registry."""

    def load(self, name: str, **kwargs: Any) -> AnyRobotSpec:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"RegistryRobotProvider does not accept extra options: {unknown}")
        return robots.get_serialized(name)


class FileRobotProvider:
    """Load a robot spec from TOML, YAML, or JSON."""

    def load(self, name: str, **kwargs: Any) -> AnyRobotSpec:
        path_value = kwargs.pop("path", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"FileRobotProvider does not accept options: {unknown}")
        path = Path(path_value) if path_value is not None else Path(name)
        spec = RobotSpec.load(path)
        return spec if name == str(path) or name == spec.name else spec.model_copy(update={"name": name})


class AssetStoreRobotProvider:
    """Load a robot spec from an asset-store record."""

    SPEC_FILENAMES: tuple[str, ...] = ("robot.toml", "robot.yaml", "robot.yml", "robot.json")

    def load(self, name: str, **kwargs: Any) -> AnyRobotSpec:
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
        return RobotSpec.load(_robot_spec_path(record.path, spec_filename))


class HolosomaRobotProvider:
    """Construct the typed G1 spherehand spec from a Holosoma checkout."""

    def load(self, name: str, **kwargs: Any) -> AnyRobotSpec:
        from retarget.recipes.holosoma import g1_spherehand_robot

        holosoma_root = kwargs.pop("holosoma_root", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"HolosomaRobotProvider does not accept options: {unknown}")
        spec = g1_spherehand_robot(holosoma_root)
        return spec if name == spec.name else spec.model_copy(update={"name": name})


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


def _explicit_limits(
    joints: tuple[JointT, ...],
    *,
    knees: tuple[JointT, ...],
    ankles: tuple[JointT, ...],
    rolls: tuple[JointT, ...],
    torso: tuple[JointT, ...],
    wrists: tuple[JointT, ...],
) -> dict[JointT, tuple[float, float]]:
    limits = {joint: (-3.14, 3.14) for joint in joints}
    limits.update({joint: (-0.1, 2.7) for joint in knees})
    limits.update({joint: (-1.2, 1.2) for joint in ankles})
    limits.update({joint: (-1.0, 1.0) for joint in rolls})
    limits.update({joint: (-1.5, 1.5) for joint in torso})
    limits.update({joint: (-2.0, 2.0) for joint in wrists})
    return limits


def _humanoid_fixture_points(
    *,
    height: float,
    pelvis: LinkT,
    torso: LinkT,
    head: LinkT | None,
    left_hip: LinkT,
    left_knee: LinkT,
    left_ankle: LinkT,
    left_foot: LinkT,
    right_hip: LinkT,
    right_knee: LinkT,
    right_ankle: LinkT,
    right_foot: LinkT,
    left_hand: LinkT,
    right_hand: LinkT,
    torso_joint: JointT,
    left_hip_joint: JointT,
    left_knee_joint: JointT,
    left_ankle_joint: JointT,
    right_hip_joint: JointT,
    right_knee_joint: JointT,
    right_ankle_joint: JointT,
    left_hand_joint: JointT,
    right_hand_joint: JointT,
) -> dict[LinkT, SimpleKinematicPoint[JointT]]:
    points: dict[LinkT, SimpleKinematicPoint[JointT]] = {
        pelvis: _point(torso_joint, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.05),
        torso: _point(torso_joint, (0.0, 0.0, 0.15 * height), (0.0, 1.0, 0.0), 0.05),
        left_hip: _point(left_hip_joint, (-0.12, 0.0, -0.10 * height), (0.0, 1.0, 0.0), 0.05),
        left_knee: _point(left_knee_joint, (-0.12, 0.0, -0.45 * height), (0.0, 0.0, -1.0), 0.06),
        left_ankle: _point(left_ankle_joint, (-0.12, 0.0, -0.82 * height), (1.0, 0.0, 0.0), 0.08),
        left_foot: _point(left_ankle_joint, (-0.12, 0.08, -0.85 * height), (1.0, 0.0, 0.0), 0.08),
        right_hip: _point(right_hip_joint, (0.12, 0.0, -0.10 * height), (0.0, 1.0, 0.0), 0.05),
        right_knee: _point(right_knee_joint, (0.12, 0.0, -0.45 * height), (0.0, 0.0, -1.0), 0.06),
        right_ankle: _point(right_ankle_joint, (0.12, 0.0, -0.82 * height), (1.0, 0.0, 0.0), 0.08),
        right_foot: _point(right_ankle_joint, (0.12, 0.08, -0.85 * height), (1.0, 0.0, 0.0), 0.08),
        left_hand: _point(left_hand_joint, (-0.32, 0.0, -0.10 * height), (1.0, 0.0, 0.0), 0.10),
        right_hand: _point(right_hand_joint, (0.32, 0.0, -0.10 * height), (1.0, 0.0, 0.0), 0.10),
    }
    if head is not None:
        points[head] = _point(torso_joint, (0.0, 0.0, 0.30 * height), (0.0, 1.0, 0.0), 0.05)
    return points


def _point(
    joint: JointT,
    offset: tuple[float, float, float],
    axis: tuple[float, float, float],
    scale: float,
) -> SimpleKinematicPoint[JointT]:
    return SimpleKinematicPoint(joint=joint, offset=offset, axis=axis, scale=scale)


def _synthetic_spec() -> RobotSpec[SyntheticJoint, SyntheticLink, SyntheticGeometry, HumanoidRobotRole]:
    points = _humanoid_fixture_points(
        height=1.2,
        pelvis=SyntheticLink.PELVIS,
        torso=SyntheticLink.TORSO,
        head=None,
        left_hip=SyntheticLink.LEFT_HIP,
        left_knee=SyntheticLink.LEFT_KNEE,
        left_ankle=SyntheticLink.LEFT_ANKLE,
        left_foot=SyntheticLink.LEFT_TOE,
        right_hip=SyntheticLink.RIGHT_HIP,
        right_knee=SyntheticLink.RIGHT_KNEE,
        right_ankle=SyntheticLink.RIGHT_ANKLE,
        right_foot=SyntheticLink.RIGHT_TOE,
        left_hand=SyntheticLink.LEFT_HAND,
        right_hand=SyntheticLink.RIGHT_HAND,
        torso_joint=SyntheticJoint.SPINE_YAW,
        left_hip_joint=SyntheticJoint.LEFT_HIP_PITCH,
        left_knee_joint=SyntheticJoint.LEFT_KNEE,
        left_ankle_joint=SyntheticJoint.LEFT_ANKLE,
        right_hip_joint=SyntheticJoint.RIGHT_HIP_PITCH,
        right_knee_joint=SyntheticJoint.RIGHT_KNEE,
        right_ankle_joint=SyntheticJoint.RIGHT_ANKLE,
        left_hand_joint=SyntheticJoint.LEFT_ELBOW,
        right_hand_joint=SyntheticJoint.RIGHT_ELBOW,
    )
    return RobotSpec(
        name=Robot.SYNTHETIC_HUMANOID.value,
        height_m=1.2,
        vocabulary=RobotVocabulary(
            joints=SyntheticJoint,
            links=SyntheticLink,
            geometries=SyntheticGeometry,
            roles=HumanoidRobotRole,
        ),
        joints=tuple(SyntheticJoint),
        links=tuple(SyntheticLink),
        contact_links=(SyntheticLink.LEFT_TOE, SyntheticLink.RIGHT_TOE),
        joint_limits={joint: (-2.0, 2.0) for joint in SyntheticJoint},
        joint_roles={
            HumanoidRobotRole.LEFT_HIP: SyntheticJoint.LEFT_HIP_PITCH,
            HumanoidRobotRole.LEFT_KNEE: SyntheticJoint.LEFT_KNEE,
            HumanoidRobotRole.LEFT_ANKLE: SyntheticJoint.LEFT_ANKLE,
            HumanoidRobotRole.RIGHT_HIP: SyntheticJoint.RIGHT_HIP_PITCH,
            HumanoidRobotRole.RIGHT_KNEE: SyntheticJoint.RIGHT_KNEE,
            HumanoidRobotRole.RIGHT_ANKLE: SyntheticJoint.RIGHT_ANKLE,
            HumanoidRobotRole.TORSO: SyntheticJoint.SPINE_YAW,
            HumanoidRobotRole.LEFT_HAND: SyntheticJoint.LEFT_ELBOW,
            HumanoidRobotRole.RIGHT_HAND: SyntheticJoint.RIGHT_ELBOW,
        },
        link_roles={
            HumanoidRobotRole.PELVIS: SyntheticLink.PELVIS,
            HumanoidRobotRole.TORSO: SyntheticLink.TORSO,
            HumanoidRobotRole.LEFT_HIP: SyntheticLink.LEFT_HIP,
            HumanoidRobotRole.LEFT_KNEE: SyntheticLink.LEFT_KNEE,
            HumanoidRobotRole.LEFT_ANKLE: SyntheticLink.LEFT_ANKLE,
            HumanoidRobotRole.LEFT_FOOT: SyntheticLink.LEFT_TOE,
            HumanoidRobotRole.RIGHT_HIP: SyntheticLink.RIGHT_HIP,
            HumanoidRobotRole.RIGHT_KNEE: SyntheticLink.RIGHT_KNEE,
            HumanoidRobotRole.RIGHT_ANKLE: SyntheticLink.RIGHT_ANKLE,
            HumanoidRobotRole.RIGHT_FOOT: SyntheticLink.RIGHT_TOE,
            HumanoidRobotRole.LEFT_HAND: SyntheticLink.LEFT_HAND,
            HumanoidRobotRole.RIGHT_HAND: SyntheticLink.RIGHT_HAND,
        },
        simple_kinematics=points,
        provenance={"fixture": True},
    )


def _g1_like_spec() -> RobotSpec[G1LikeJoint, HumanoidLink, HumanoidGeometry, HumanoidRobotRole]:
    height = 1.32
    return RobotSpec(
        name=Robot.G1_LIKE.value,
        height_m=height,
        vocabulary=RobotVocabulary(
            joints=G1LikeJoint,
            links=HumanoidLink,
            geometries=HumanoidGeometry,
            roles=HumanoidRobotRole,
        ),
        joints=tuple(G1LikeJoint),
        links=tuple(HumanoidLink),
        contact_links=(HumanoidLink.LEFT_FOOT, HumanoidLink.RIGHT_FOOT),
        joint_limits=_explicit_limits(
            tuple(G1LikeJoint),
            knees=(G1LikeJoint.LEFT_KNEE, G1LikeJoint.RIGHT_KNEE),
            ankles=(
                G1LikeJoint.LEFT_ANKLE_PITCH,
                G1LikeJoint.LEFT_ANKLE_ROLL,
                G1LikeJoint.RIGHT_ANKLE_PITCH,
                G1LikeJoint.RIGHT_ANKLE_ROLL,
            ),
            rolls=(
                G1LikeJoint.LEFT_HIP_ROLL,
                G1LikeJoint.RIGHT_HIP_ROLL,
                G1LikeJoint.LEFT_SHOULDER_ROLL,
                G1LikeJoint.RIGHT_SHOULDER_ROLL,
            ),
            torso=(G1LikeJoint.WAIST_YAW, G1LikeJoint.WAIST_ROLL, G1LikeJoint.WAIST_PITCH),
            wrists=(
                G1LikeJoint.LEFT_WRIST_ROLL,
                G1LikeJoint.LEFT_WRIST_PITCH,
                G1LikeJoint.LEFT_WRIST_YAW,
                G1LikeJoint.RIGHT_WRIST_ROLL,
                G1LikeJoint.RIGHT_WRIST_PITCH,
                G1LikeJoint.RIGHT_WRIST_YAW,
            ),
        ),
        joint_roles=_g1_joint_roles(),
        link_roles=_humanoid_link_roles(),
        simple_kinematics=_g1_fixture_points(height),
        provenance={
            "asset_required": True,
            "description": "G1-like template; attach local URDF or MJCF assets with a typed spec.",
        },
    )


def _t1_like_spec() -> RobotSpec[T1LikeJoint, HumanoidLink, HumanoidGeometry, HumanoidRobotRole]:
    height = 1.2
    return RobotSpec(
        name=Robot.T1_LIKE.value,
        height_m=height,
        vocabulary=RobotVocabulary(
            joints=T1LikeJoint,
            links=HumanoidLink,
            geometries=HumanoidGeometry,
            roles=HumanoidRobotRole,
        ),
        joints=tuple(T1LikeJoint),
        links=tuple(HumanoidLink),
        contact_links=(HumanoidLink.LEFT_FOOT, HumanoidLink.RIGHT_FOOT),
        joint_limits=_explicit_limits(
            tuple(T1LikeJoint),
            knees=(T1LikeJoint.LEFT_KNEE, T1LikeJoint.RIGHT_KNEE),
            ankles=(
                T1LikeJoint.LEFT_ANKLE_PITCH,
                T1LikeJoint.LEFT_ANKLE_ROLL,
                T1LikeJoint.RIGHT_ANKLE_PITCH,
                T1LikeJoint.RIGHT_ANKLE_ROLL,
            ),
            rolls=(
                T1LikeJoint.LEFT_HIP_ROLL,
                T1LikeJoint.RIGHT_HIP_ROLL,
                T1LikeJoint.LEFT_SHOULDER_ROLL,
                T1LikeJoint.RIGHT_SHOULDER_ROLL,
            ),
            torso=(T1LikeJoint.TORSO_YAW,),
            wrists=(T1LikeJoint.LEFT_WRIST, T1LikeJoint.RIGHT_WRIST),
        ),
        joint_roles=_t1_joint_roles(),
        link_roles=_humanoid_link_roles(),
        simple_kinematics=_t1_fixture_points(height),
        provenance={
            "asset_required": True,
            "description": "T1-like template; attach local URDF or MJCF assets with a typed spec.",
        },
    )


def _humanoid_link_roles() -> dict[HumanoidRobotRole, HumanoidLink]:
    return {
        HumanoidRobotRole.PELVIS: HumanoidLink.PELVIS,
        HumanoidRobotRole.TORSO: HumanoidLink.TORSO,
        HumanoidRobotRole.HEAD: HumanoidLink.HEAD,
        HumanoidRobotRole.LEFT_HIP: HumanoidLink.LEFT_HIP,
        HumanoidRobotRole.LEFT_KNEE: HumanoidLink.LEFT_KNEE,
        HumanoidRobotRole.LEFT_ANKLE: HumanoidLink.LEFT_ANKLE,
        HumanoidRobotRole.LEFT_FOOT: HumanoidLink.LEFT_FOOT,
        HumanoidRobotRole.RIGHT_HIP: HumanoidLink.RIGHT_HIP,
        HumanoidRobotRole.RIGHT_KNEE: HumanoidLink.RIGHT_KNEE,
        HumanoidRobotRole.RIGHT_ANKLE: HumanoidLink.RIGHT_ANKLE,
        HumanoidRobotRole.RIGHT_FOOT: HumanoidLink.RIGHT_FOOT,
        HumanoidRobotRole.LEFT_HAND: HumanoidLink.LEFT_HAND,
        HumanoidRobotRole.RIGHT_HAND: HumanoidLink.RIGHT_HAND,
    }


def _g1_joint_roles() -> dict[HumanoidRobotRole, G1LikeJoint]:
    return {
        HumanoidRobotRole.LEFT_HIP: G1LikeJoint.LEFT_HIP_PITCH,
        HumanoidRobotRole.LEFT_KNEE: G1LikeJoint.LEFT_KNEE,
        HumanoidRobotRole.LEFT_ANKLE: G1LikeJoint.LEFT_ANKLE_PITCH,
        HumanoidRobotRole.RIGHT_HIP: G1LikeJoint.RIGHT_HIP_PITCH,
        HumanoidRobotRole.RIGHT_KNEE: G1LikeJoint.RIGHT_KNEE,
        HumanoidRobotRole.RIGHT_ANKLE: G1LikeJoint.RIGHT_ANKLE_PITCH,
        HumanoidRobotRole.TORSO: G1LikeJoint.WAIST_YAW,
        HumanoidRobotRole.LEFT_HAND: G1LikeJoint.LEFT_ELBOW,
        HumanoidRobotRole.RIGHT_HAND: G1LikeJoint.RIGHT_ELBOW,
    }


def _t1_joint_roles() -> dict[HumanoidRobotRole, T1LikeJoint]:
    return {
        HumanoidRobotRole.LEFT_HIP: T1LikeJoint.LEFT_HIP_PITCH,
        HumanoidRobotRole.LEFT_KNEE: T1LikeJoint.LEFT_KNEE,
        HumanoidRobotRole.LEFT_ANKLE: T1LikeJoint.LEFT_ANKLE_PITCH,
        HumanoidRobotRole.RIGHT_HIP: T1LikeJoint.RIGHT_HIP_PITCH,
        HumanoidRobotRole.RIGHT_KNEE: T1LikeJoint.RIGHT_KNEE,
        HumanoidRobotRole.RIGHT_ANKLE: T1LikeJoint.RIGHT_ANKLE_PITCH,
        HumanoidRobotRole.TORSO: T1LikeJoint.TORSO_YAW,
        HumanoidRobotRole.LEFT_HAND: T1LikeJoint.LEFT_ELBOW,
        HumanoidRobotRole.RIGHT_HAND: T1LikeJoint.RIGHT_ELBOW,
    }


def _g1_fixture_points(height: float) -> dict[HumanoidLink, SimpleKinematicPoint[G1LikeJoint]]:
    return _humanoid_fixture_points(
        height=height,
        pelvis=HumanoidLink.PELVIS,
        torso=HumanoidLink.TORSO,
        head=HumanoidLink.HEAD,
        left_hip=HumanoidLink.LEFT_HIP,
        left_knee=HumanoidLink.LEFT_KNEE,
        left_ankle=HumanoidLink.LEFT_ANKLE,
        left_foot=HumanoidLink.LEFT_FOOT,
        right_hip=HumanoidLink.RIGHT_HIP,
        right_knee=HumanoidLink.RIGHT_KNEE,
        right_ankle=HumanoidLink.RIGHT_ANKLE,
        right_foot=HumanoidLink.RIGHT_FOOT,
        left_hand=HumanoidLink.LEFT_HAND,
        right_hand=HumanoidLink.RIGHT_HAND,
        torso_joint=G1LikeJoint.WAIST_YAW,
        left_hip_joint=G1LikeJoint.LEFT_HIP_PITCH,
        left_knee_joint=G1LikeJoint.LEFT_KNEE,
        left_ankle_joint=G1LikeJoint.LEFT_ANKLE_PITCH,
        right_hip_joint=G1LikeJoint.RIGHT_HIP_PITCH,
        right_knee_joint=G1LikeJoint.RIGHT_KNEE,
        right_ankle_joint=G1LikeJoint.RIGHT_ANKLE_PITCH,
        left_hand_joint=G1LikeJoint.LEFT_ELBOW,
        right_hand_joint=G1LikeJoint.RIGHT_ELBOW,
    )


def _t1_fixture_points(height: float) -> dict[HumanoidLink, SimpleKinematicPoint[T1LikeJoint]]:
    return _humanoid_fixture_points(
        height=height,
        pelvis=HumanoidLink.PELVIS,
        torso=HumanoidLink.TORSO,
        head=HumanoidLink.HEAD,
        left_hip=HumanoidLink.LEFT_HIP,
        left_knee=HumanoidLink.LEFT_KNEE,
        left_ankle=HumanoidLink.LEFT_ANKLE,
        left_foot=HumanoidLink.LEFT_FOOT,
        right_hip=HumanoidLink.RIGHT_HIP,
        right_knee=HumanoidLink.RIGHT_KNEE,
        right_ankle=HumanoidLink.RIGHT_ANKLE,
        right_foot=HumanoidLink.RIGHT_FOOT,
        left_hand=HumanoidLink.LEFT_HAND,
        right_hand=HumanoidLink.RIGHT_HAND,
        torso_joint=T1LikeJoint.TORSO_YAW,
        left_hip_joint=T1LikeJoint.LEFT_HIP_PITCH,
        left_knee_joint=T1LikeJoint.LEFT_KNEE,
        left_ankle_joint=T1LikeJoint.LEFT_ANKLE_PITCH,
        right_hip_joint=T1LikeJoint.RIGHT_HIP_PITCH,
        right_knee_joint=T1LikeJoint.RIGHT_KNEE,
        right_ankle_joint=T1LikeJoint.RIGHT_ANKLE_PITCH,
        left_hand_joint=T1LikeJoint.LEFT_ELBOW,
        right_hand_joint=T1LikeJoint.RIGHT_ELBOW,
    )


robots.register(Robot.SYNTHETIC_HUMANOID, _synthetic_spec())
robots.register(Robot.G1_LIKE, _g1_like_spec())
robots.register(Robot.T1_LIKE, _t1_like_spec())

robot_providers.register(RobotProviderName.REGISTRY, RegistryRobotProvider())
robot_providers.register(RobotProviderName.FILE, FileRobotProvider())
robot_providers.register(RobotProviderName.ASSET_STORE, AssetStoreRobotProvider())
robot_providers.register(RobotProviderName.HOLOSOMA, HolosomaRobotProvider())
