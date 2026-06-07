"""Strongly typed robot vocabularies and specifications."""

from __future__ import annotations

import importlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypeVar

from retarget.core.enums import RobotGeometry, RobotJoint, RobotLink, RobotRole

JointT = TypeVar("JointT", bound=RobotJoint, default=RobotJoint)
LinkT = TypeVar("LinkT", bound=RobotLink, default=RobotLink)
GeometryT = TypeVar("GeometryT", bound=RobotGeometry, default=RobotGeometry)
RoleT = TypeVar("RoleT", bound=RobotRole, default=RobotRole)


class JointLimit(BaseModel, Generic[JointT]):
    """Position bounds for one typed actuated joint."""

    joint: JointT
    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate_bounds(self) -> JointLimit[JointT]:
        if self.upper < self.lower:
            raise ValueError("upper must be >= lower")
        return self


class QposLayout(BaseModel):
    """Convention for serialized qpos vectors."""

    root_position: tuple[int, int] = (0, 3)
    root_quaternion: tuple[int, int] = (3, 7)
    joint_start: int = 7
    object_pose_size: int = 7

    def qpos_size(self, robot_dof: int, *, has_object: bool = False) -> int:
        """Return qpos size for this layout."""

        return self.joint_start + robot_dof + (self.object_pose_size if has_object else 0)

    def joint_slice(self, robot_dof: int) -> slice:
        """Return the slice containing actuated robot joints."""

        return slice(self.joint_start, self.joint_start + robot_dof)

    def object_slice(self, robot_dof: int) -> slice:
        """Return the slice containing an appended object pose."""

        start = self.joint_start + robot_dof
        return slice(start, start + self.object_pose_size)


class RobotVocabulary(BaseModel, Generic[JointT, LinkT, GeometryT, RoleT]):
    """Concrete enum classes used by one robot specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    joints: type[JointT]
    links: type[LinkT]
    geometries: type[GeometryT]
    roles: type[RoleT]

    @model_validator(mode="after")
    def _validate_types(self) -> RobotVocabulary[JointT, LinkT, GeometryT, RoleT]:
        for enum_type, base, label in (
            (self.joints, RobotJoint, "joints"),
            (self.links, RobotLink, "links"),
            (self.geometries, RobotGeometry, "geometries"),
            (self.roles, RobotRole, "roles"),
        ):
            if not isinstance(enum_type, type) or not issubclass(enum_type, base):
                raise TypeError(f"robot vocabulary {label} must subclass {base.__name__}")
        return self


class SimpleKinematicPoint(BaseModel, Generic[JointT]):
    """Explicit fixture kinematics for one robot link."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    joint: JointT
    offset: tuple[float, float, float]
    axis: tuple[float, float, float]
    scale: float = 1.0

    @model_validator(mode="after")
    def _validate_point(self) -> SimpleKinematicPoint[JointT]:
        if np.linalg.norm(np.asarray(self.axis, dtype=np.float64)) <= 1e-12:
            raise ValueError("simple kinematic point axis must be non-zero")
        if self.scale <= 0.0:
            raise ValueError("simple kinematic point scale must be positive")
        return self


class RobotSpec(BaseModel, Generic[JointT, LinkT, GeometryT, RoleT]):
    """Robot description that preserves enum identity until backend compilation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    height_m: float
    vocabulary: RobotVocabulary[JointT, LinkT, GeometryT, RoleT]
    joints: tuple[JointT, ...]
    links: tuple[LinkT, ...]
    contact_links: tuple[LinkT, ...] = ()
    geometries: tuple[GeometryT, ...] = ()
    joint_limits: dict[JointT, tuple[float, float]] = Field(default_factory=dict)
    joint_roles: dict[RoleT, JointT] = Field(default_factory=dict)
    link_roles: dict[RoleT, LinkT] = Field(default_factory=dict)
    link_groups: dict[RoleT, tuple[LinkT, ...]] = Field(default_factory=dict)
    mujoco_body_aliases: dict[LinkT, str] = Field(default_factory=dict)
    simple_kinematics: dict[LinkT, SimpleKinematicPoint[JointT]] = Field(default_factory=dict)
    urdf_path: Path | None = None
    mujoco_xml_path: Path | None = None
    qpos_layout: QposLayout = Field(default_factory=QposLayout)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_serialized_members(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        vocabulary = normalized.get("vocabulary")
        if isinstance(vocabulary, dict):
            vocabulary = RobotVocabulary(
                joints=_enum_type(vocabulary["joints"], RobotJoint),
                links=_enum_type(vocabulary["links"], RobotLink),
                geometries=_enum_type(vocabulary["geometries"], RobotGeometry),
                roles=_enum_type(vocabulary["roles"], RobotRole),
            )
            normalized["vocabulary"] = vocabulary
        if not isinstance(vocabulary, RobotVocabulary):
            return normalized
        normalized["joints"] = _members(normalized.get("joints", ()), vocabulary.joints)
        normalized["links"] = _members(normalized.get("links", ()), vocabulary.links)
        normalized["contact_links"] = _members(normalized.get("contact_links", ()), vocabulary.links)
        normalized["geometries"] = _members(normalized.get("geometries", ()), vocabulary.geometries)
        normalized["joint_limits"] = _typed_mapping(
            normalized.get("joint_limits", {}),
            vocabulary.joints,
            lambda bounds: (float(bounds[0]), float(bounds[1])),
        )
        normalized["joint_roles"] = _typed_mapping(
            normalized.get("joint_roles", {}),
            vocabulary.roles,
            lambda member: _member(member, vocabulary.joints),
        )
        normalized["link_roles"] = _typed_mapping(
            normalized.get("link_roles", {}),
            vocabulary.roles,
            lambda member: _member(member, vocabulary.links),
        )
        normalized["link_groups"] = _typed_mapping(
            normalized.get("link_groups", {}),
            vocabulary.roles,
            lambda members: _members(members, vocabulary.links),
        )
        normalized["mujoco_body_aliases"] = _typed_mapping(
            normalized.get("mujoco_body_aliases", {}),
            vocabulary.links,
            str,
        )
        simple: dict[RobotLink, SimpleKinematicPoint[Any]] = {}
        for link, point in dict(normalized.get("simple_kinematics", {})).items():
            typed_link = _member(link, vocabulary.links)
            payload = dict(point)
            payload["joint"] = _member(payload["joint"], vocabulary.joints)
            simple[typed_link] = SimpleKinematicPoint.model_validate(payload)
        normalized["simple_kinematics"] = simple
        for path_field in ("urdf_path", "mujoco_xml_path"):
            if normalized.get(path_field) in ("", None):
                normalized[path_field] = None
            elif not isinstance(normalized[path_field], Path):
                normalized[path_field] = Path(normalized[path_field])
        return normalized

    @model_validator(mode="after")
    def _validate_robot(self) -> RobotSpec[JointT, LinkT, GeometryT, RoleT]:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        if not self.joints:
            raise ValueError("robot must declare at least one actuated joint")
        _require_members(self.joints, self.vocabulary.joints, "joints")
        _require_members(self.links, self.vocabulary.links, "links")
        _require_members(self.contact_links, self.vocabulary.links, "contact_links")
        _require_members(self.geometries, self.vocabulary.geometries, "geometries")
        _require_unique(self.joints, "joints")
        _require_unique(self.links, "links")
        _require_unique(self.contact_links, "contact_links")
        _require_unique(self.geometries, "geometries")
        _require_members(self.joint_limits, self.vocabulary.joints, "joint limit keys")
        _require_members(self.joint_roles, self.vocabulary.roles, "joint role keys")
        _require_members(self.joint_roles.values(), self.vocabulary.joints, "joint role targets")
        _require_members(self.link_roles, self.vocabulary.roles, "link role keys")
        _require_members(self.link_roles.values(), self.vocabulary.links, "link role targets")
        _require_members(self.link_groups, self.vocabulary.roles, "link group keys")
        _require_members(
            (link for links in self.link_groups.values() for link in links),
            self.vocabulary.links,
            "link group targets",
        )
        _require_members(self.mujoco_body_aliases, self.vocabulary.links, "MuJoCo body alias keys")
        _require_members(self.simple_kinematics, self.vocabulary.links, "simple kinematic point keys")
        _require_subset(self.joint_limits, self.joints, "joint_limits")
        _require_subset(self.contact_links, self.links, "contact_links")
        _require_subset(self.joint_roles, tuple(self.vocabulary.roles), "joint role keys")
        _require_subset(self.link_roles, tuple(self.vocabulary.roles), "link role keys")
        _require_subset(self.link_groups, tuple(self.vocabulary.roles), "link group keys")
        _require_subset(self.joint_roles.values(), self.joints, "joint role targets")
        _require_subset(self.link_roles.values(), self.links, "link role targets")
        _require_subset(
            (link for links in self.link_groups.values() for link in links),
            self.links,
            "link group targets",
        )
        _require_subset(self.mujoco_body_aliases, self.links, "MuJoCo body aliases")
        _require_subset(self.simple_kinematics, self.links, "simple kinematic points")
        for link, point in self.simple_kinematics.items():
            if not isinstance(point.joint, self.vocabulary.joints):
                raise TypeError(f"simple kinematic point {link.value!r} uses the wrong joint vocabulary")
        return self

    @property
    def dof(self) -> int:
        """Number of actuated joints."""

        return len(self.joints)

    def joint_index(self, joint: JointT) -> int:
        """Return the actuated index for a typed joint."""

        _require_member(joint, self.vocabulary.joints, "joint")
        try:
            return self.joints.index(joint)
        except ValueError as exc:
            raise KeyError(f"Unknown robot joint {joint.value!r} for {self.name!r}") from exc

    def joint_for_role(self, role: RoleT) -> JointT:
        """Resolve a semantic robot role to an actuated joint."""

        _require_member(role, self.vocabulary.roles, "role")
        try:
            return self.joint_roles[role]
        except KeyError as exc:
            raise KeyError(f"Robot {self.name!r} has no joint binding for role {role.value!r}") from exc

    def link_for_role(self, role: RoleT) -> LinkT:
        """Resolve a semantic robot role to a link."""

        _require_member(role, self.vocabulary.roles, "role")
        try:
            return self.link_roles[role]
        except KeyError as exc:
            raise KeyError(f"Robot {self.name!r} has no link binding for role {role.value!r}") from exc

    def links_for_role(self, role: RoleT) -> tuple[LinkT, ...]:
        """Resolve a semantic robot role to one or more links."""

        _require_member(role, self.vocabulary.roles, "role")
        if role in self.link_groups:
            return self.link_groups[role]
        return (self.link_for_role(role),)

    def qpos_size(self, *, has_object: bool = False) -> int:
        """Return qpos size for this robot and object setting."""

        return self.qpos_layout.qpos_size(self.dof, has_object=has_object)

    @classmethod
    def load(cls, path: str | Path) -> RobotSpec[Any, Any, Any, Any]:
        """Load a robot spec whose config names concrete enum vocabulary classes."""

        spec_path = Path(path)
        return cls.model_validate(_load_mapping(spec_path)).resolve_paths(spec_path.parent)

    def resolve_paths(self, base_dir: str | Path) -> RobotSpec[JointT, LinkT, GeometryT, RoleT]:
        """Return a copy with relative asset paths resolved against ``base_dir``."""

        base = Path(base_dir)
        return self.model_copy(
            update={
                "urdf_path": _resolve_relative(self.urdf_path, base),
                "mujoco_xml_path": _resolve_relative(self.mujoco_xml_path, base),
            }
        )

    def save_json(self, path: str | Path) -> Path:
        """Save a JSON representation with explicit vocabulary references."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude={"vocabulary"})
        payload["vocabulary"] = {
            "joints": _enum_reference(self.vocabulary.joints),
            "links": _enum_reference(self.vocabulary.links),
            "geometries": _enum_reference(self.vocabulary.geometries),
            "roles": _enum_reference(self.vocabulary.roles),
        }
        output.write_text(json.dumps(payload, indent=2))
        return output

    def limits_array(self) -> tuple[list[float], list[float]]:
        """Return joint limits in joint order."""

        lower: list[float] = []
        upper: list[float] = []
        for joint in self.joints:
            lo, hi = self.joint_limits.get(joint, (-1e6, 1e6))
            lower.append(float(lo))
            upper.append(float(hi))
        return lower, upper


def _member(value: Any, enum_type: type[StrEnum]) -> Any:
    if isinstance(value, StrEnum):
        if not isinstance(value, enum_type):
            raise TypeError(
                f"expected a {enum_type.__name__} member, received {type(value).__name__}.{value.name}"
            )
        return value
    return enum_type(str(value))


def _members(values: Any, enum_type: type[StrEnum]) -> tuple[Any, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, str | StrEnum):
        values = (values,)
    return tuple(_member(value, enum_type) for value in values)


def _typed_mapping(values: Any, key_type: type[StrEnum], convert_value: Any) -> dict[Any, Any]:
    if values in (None, ""):
        return {}
    if not isinstance(values, dict):
        raise TypeError("typed mappings must be dictionaries")
    return {_member(key, key_type): convert_value(value) for key, value in values.items()}


def _require_member(value: StrEnum, enum_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{label} must be a {enum_type.__name__} member")


def _require_members(values: Any, enum_type: type[StrEnum], label: str) -> None:
    if not all(isinstance(value, enum_type) for value in values):
        raise TypeError(f"{label} must use {enum_type.__name__}")


def _require_unique(values: Any, label: str) -> None:
    sequence = tuple(values)
    if len(set(sequence)) != len(sequence):
        raise ValueError(f"{label} must be unique")


def _require_subset(values: Any, allowed: Any, label: str) -> None:
    unknown = set(values) - set(allowed)
    if unknown:
        rendered = sorted(value.value for value in unknown)
        raise ValueError(f"{label} reference unknown members: {rendered}")


def _enum_type(reference: Any, base: type[StrEnum]) -> type[Any]:
    enum_type: Any
    if isinstance(reference, type):
        enum_type = reference
    else:
        module_name, separator, qualname = str(reference).partition(":")
        if not separator:
            raise ValueError("enum vocabulary references must use 'module:qualname'")
        enum_type = importlib.import_module(module_name)
        for segment in qualname.split("."):
            enum_type = getattr(enum_type, segment)
    if not isinstance(enum_type, type) or not issubclass(enum_type, base):
        raise TypeError(f"{reference!r} does not resolve to a {base.__name__} subclass")
    return enum_type


def _enum_reference(enum_type: type[StrEnum]) -> str:
    return f"{enum_type.__module__}:{enum_type.__qualname__}"


def _resolve_relative(path: Path | None, base_dir: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return dict(tomllib.loads(path.read_text()))
    if suffix in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    if suffix == ".json":
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping at the document root")
        return dict(loaded)
    raise ValueError(f"Unsupported robot spec suffix {suffix!r}; expected .toml, .yaml, .yml, or .json")


AnyRobotSpec = RobotSpec[Any, Any, Any, Any]
