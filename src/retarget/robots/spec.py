"""Robot configuration models."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JointLimit(BaseModel):
    """Named joint position bounds.

    Attributes:
        name (str): Joint name matching :attr:`RobotSpec.joint_names`.
        lower (float): Lower position bound in radians or meters.
        upper (float): Upper position bound in radians or meters.
    """

    name: str
    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate_bounds(self) -> JointLimit:
        if self.upper < self.lower:
            raise ValueError("upper must be >= lower")
        return self


class QposLayout(BaseModel):
    """Convention for serialized qpos vectors.

    Attributes:
        root_position (tuple[int, int]): Half-open slice ``(start, stop)`` for root translation.
        root_quaternion (tuple[int, int]): Half-open slice for root orientation (wxyz).
        joint_start (int): Index where actuated robot joints begin in ``qpos``.
        object_pose_size (int): Number of scalars reserved for object pose (position + quaternion).
    """

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
        """Return the slice containing object pose, if present."""

        start = self.joint_start + robot_dof
        return slice(start, start + self.object_pose_size)


class RobotSpec(BaseModel):
    """Validated robot description used by retargeting.

    Attributes:
        name (str): Robot identifier used in problems and results.
        dof (int): Number of actuated degrees of freedom.
        height_m (float): Nominal standing height in meters (for scaling heuristics).
        joint_names (tuple[str, ...]): Actuated joint names in qpos order.
        link_names (tuple[str, ...]): Named links for kinematics and contact (default empty).
        contact_links (tuple[str, ...]): Links used for foot or support contact (default empty).
        nominal_tracking_joints (tuple[str, ...]): Joints tracked by nominal-pose objectives.
        joint_limits (dict[str, tuple[float, float]]): Per-joint ``(lower, upper)`` bounds.
        default_joint_mapping (dict[str, str]): Source-joint to robot-joint name map.
        default_link_mapping (dict[str, str]): Source-link to robot link/joint name map.
        urdf_path (Path | None): Optional URDF used for visualization or kinematics.
        mujoco_xml_path (Path | None): Optional MuJoCo XML model path.
        qpos_layout (QposLayout): Layout of root, joints, and optional object pose in ``qpos``.
        metadata (dict[str, Any]): Free-form robot metadata (asset hints, descriptions, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    dof: int
    height_m: float
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...] = ()
    contact_links: tuple[str, ...] = ()
    nominal_tracking_joints: tuple[str, ...] = ()
    joint_limits: dict[str, tuple[float, float]] = Field(default_factory=dict)
    default_joint_mapping: dict[str, str] = Field(default_factory=dict)
    default_link_mapping: dict[str, str] = Field(default_factory=dict)
    urdf_path: Path | None = None
    mujoco_xml_path: Path | None = None
    qpos_layout: QposLayout = Field(default_factory=QposLayout)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("urdf_path", "mujoco_xml_path", mode="before")
    @classmethod
    def _path_or_none(cls, value: Any) -> Path | None:
        return None if value in (None, "") else Path(value)

    @model_validator(mode="after")
    def _validate_robot(self) -> RobotSpec:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.dof <= 0:
            raise ValueError("dof must be positive")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        if len(self.joint_names) != self.dof:
            raise ValueError("joint_names length must match dof")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        unknown_limits = set(self.joint_limits) - set(self.joint_names)
        if unknown_limits:
            raise ValueError(f"joint_limits reference unknown joints: {sorted(unknown_limits)}")
        unknown_nominal = set(self.nominal_tracking_joints) - set(self.joint_names)
        if unknown_nominal:
            raise ValueError(f"nominal_tracking_joints reference unknown joints: {sorted(unknown_nominal)}")
        unknown_mapping_targets = set(self.default_joint_mapping.values()) - set(self.joint_names)
        if unknown_mapping_targets:
            raise ValueError(f"default_joint_mapping targets unknown joints: {sorted(unknown_mapping_targets)}")
        valid_link_targets = set(self.link_names) | set(self.joint_names)
        unknown_link_targets = set(self.default_link_mapping.values()) - valid_link_targets
        if unknown_link_targets:
            raise ValueError(f"default_link_mapping targets unknown links: {sorted(unknown_link_targets)}")
        return self

    def joint_index(self, name: str) -> int:
        """Return actuated joint index."""

        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown robot joint {name!r} for {self.name!r}") from exc

    def qpos_size(self, *, has_object: bool = False) -> int:
        """Return qpos size for this robot and object setting."""

        return self.qpos_layout.qpos_size(self.dof, has_object=has_object)

    @classmethod
    def load(cls, path: str | Path) -> RobotSpec:
        """Load a robot spec from TOML, YAML, or JSON."""

        spec_path = Path(path)
        return cls.model_validate(_load_mapping(spec_path)).resolve_paths(spec_path.parent)

    def resolve_paths(self, base_dir: str | Path) -> RobotSpec:
        """Return a copy with relative asset paths resolved against `base_dir`."""

        base = Path(base_dir)
        return self.model_copy(
            update={
                "urdf_path": _resolve_relative(self.urdf_path, base),
                "mujoco_xml_path": _resolve_relative(self.mujoco_xml_path, base),
            }
        )

    def save_json(self, path: str | Path) -> Path:
        """Save this robot spec as JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output

    def limits_array(self) -> tuple[list[float], list[float]]:
        """Return joint limits in joint order, defaulting to unbounded finite research-safe values."""

        lower: list[float] = []
        upper: list[float] = []
        for name in self.joint_names:
            lo, hi = self.joint_limits.get(name, (-1e6, 1e6))
            lower.append(float(lo))
            upper.append(float(hi))
        return lower, upper


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
