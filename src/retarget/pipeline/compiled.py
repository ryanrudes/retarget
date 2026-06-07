"""Private compilation boundary from typed domain objects to backend names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from retarget.core.enums import RobotGeometry, RobotJoint, RobotLink, SceneGeometry
from retarget.motion.support import SupportPlane
from retarget.optimization.spec import GeometryPair, NonPenetrationConstraintConfig, SelfCollisionConstraintConfig
from retarget.pipeline.problem import RetargetingProblem
from retarget.robots.spec import AnyRobotSpec, QposLayout

EnumIdentity = tuple[type[StrEnum], str]


@dataclass(frozen=True)
class CompiledSimpleKinematicPoint:
    """String/index form of one explicit fixture kinematic point."""

    joint_index: int
    offset: np.ndarray
    axis: np.ndarray
    scale: float


@dataclass(frozen=True)
class CompiledRobotSpec:
    """Backend-facing robot model names and indices."""

    name: str
    height_m: float
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    contact_links: tuple[str, ...]
    geometry_names: tuple[str, ...]
    joint_limits: dict[str, tuple[float, float]]
    mujoco_body_aliases: dict[str, str]
    simple_kinematics: dict[str, CompiledSimpleKinematicPoint]
    urdf_path: Any
    mujoco_xml_path: Any
    qpos_layout: QposLayout

    @property
    def dof(self) -> int:
        """Number of actuated joints."""

        return len(self.joint_names)

    def joint_index(self, name: str) -> int:
        """Return the compiled actuated-joint index."""

        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown compiled robot joint {name!r}") from exc

    def qpos_size(self, *, has_object: bool = False) -> int:
        """Return qpos size for this robot and object setting."""

        return self.qpos_layout.qpos_size(self.dof, has_object=has_object)

    def limits_array(self) -> tuple[list[float], list[float]]:
        """Return joint limits in compiled joint order."""

        lower: list[float] = []
        upper: list[float] = []
        for name in self.joint_names:
            lo, hi = self.joint_limits.get(name, (-1e6, 1e6))
            lower.append(float(lo))
            upper.append(float(hi))
        return lower, upper


@dataclass(frozen=True)
class CompiledContactTrack:
    """Solver-facing masks and link names for one semantic contact track."""

    subject: str
    link_names: tuple[str, ...]
    active_mask: np.ndarray
    support_mask: np.ndarray

    def active_at(self, frame_idx: int) -> bool:
        return bool(self.active_mask[frame_idx])

    def support_at(self, frame_idx: int) -> bool:
        return bool(self.support_mask[frame_idx])


@dataclass(frozen=True)
class CompiledContactFrame:
    """Per-frame solver-facing contact view."""

    frame_idx: int
    tracks: tuple[CompiledContactTrack, ...]
    support: SupportPlane | None

    @property
    def active_link_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                link
                for track in self.tracks
                if track.active_at(self.frame_idx)
                for link in track.link_names
            )
        )

    @property
    def support_link_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                link
                for track in self.tracks
                if track.support_at(self.frame_idx)
                for link in track.link_names
            )
        )


@dataclass(frozen=True)
class CompiledContactPlan:
    """Solver-facing contact plan."""

    tracks: tuple[CompiledContactTrack, ...]
    frame_count: int
    support: SupportPlane | None

    def frame(self, frame_idx: int) -> CompiledContactFrame:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(frame_idx)
        return CompiledContactFrame(frame_idx, self.tracks, self.support)


@dataclass(frozen=True)
class CompiledLinkTargetTrack:
    """Solver-facing target track."""

    link_name: str
    positions: np.ndarray
    weights: np.ndarray
    active_mask: np.ndarray

    def active_at(self, frame_idx: int) -> bool:
        return bool(
            self.active_mask[frame_idx]
            and np.isfinite(self.positions[frame_idx]).all()
            and np.isfinite(self.weights[frame_idx])
            and self.weights[frame_idx] > 0.0
        )


@dataclass(frozen=True)
class CompiledTargetFrame:
    """Per-frame solver-facing target samples."""

    frame_idx: int
    tracks: tuple[CompiledLinkTargetTrack, ...]

    @property
    def _active(self) -> tuple[CompiledLinkTargetTrack, ...]:
        return tuple(track for track in self.tracks if track.active_at(self.frame_idx))

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(track.link_name for track in self._active)

    @property
    def positions(self) -> np.ndarray:
        active = self._active
        return (
            np.asarray([track.positions[self.frame_idx] for track in active], dtype=np.float64)
            if active
            else np.zeros((0, 3), dtype=np.float64)
        )

    @property
    def weights(self) -> np.ndarray:
        return np.asarray([track.weights[self.frame_idx] for track in self._active], dtype=np.float64)


@dataclass(frozen=True)
class CompiledLinkTargetPlan:
    """Solver-facing target plan."""

    tracks: tuple[CompiledLinkTargetTrack, ...]
    frame_count: int

    def frame(self, frame_idx: int) -> CompiledTargetFrame:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(frame_idx)
        return CompiledTargetFrame(frame_idx, self.tracks)


@dataclass(frozen=True)
class CompiledRetargetingProblem:
    """Private backend representation compiled from a typed problem."""

    source: RetargetingProblem[Any, Any, Any, Any, Any]
    robot: CompiledRobotSpec
    joint_mapping: dict[str, str]
    link_mapping: dict[str, str]
    link_names: dict[EnumIdentity, str]
    joint_names: dict[EnumIdentity, str]
    geometry_names: dict[EnumIdentity, str]
    geometry_references: tuple[str, ...]
    contacts: CompiledContactPlan | None
    targets: CompiledLinkTargetPlan | None

    @property
    def referenced_link_names(self) -> tuple[str, ...]:
        """Every backend link referenced by mappings, contacts, or targets."""

        names = list(self.link_mapping.values())
        if self.contacts is not None:
            names.extend(link for track in self.contacts.tracks for link in track.link_names)
        if self.targets is not None:
            names.extend(track.link_name for track in self.targets.tracks)
        return tuple(dict.fromkeys(names))

    @property
    def referenced_geometry_names(self) -> tuple[str, ...]:
        """Every backend geometry referenced by an optimization policy."""

        return self.geometry_references

    def link_name(self, link: RobotLink) -> str:
        """Return the compiled model name for a typed robot link."""

        try:
            return self.link_names[_enum_identity(link)]
        except KeyError as exc:
            raise TypeError("robot link uses a vocabulary different from the compiled robot") from exc

    def joint_name(self, joint: RobotJoint) -> str:
        """Return the compiled model name for a typed robot joint."""

        try:
            return self.joint_names[_enum_identity(joint)]
        except KeyError as exc:
            raise TypeError("robot joint uses a vocabulary different from the compiled robot") from exc

    def geometry_name(self, geometry: RobotGeometry | SceneGeometry) -> str:
        """Return the compiled model name for a typed robot geometry."""

        try:
            return self.geometry_names[_enum_identity(geometry)]
        except KeyError as exc:
            raise TypeError("robot geometry uses a vocabulary different from the compiled robot") from exc


def compile_problem(
    problem: RetargetingProblem[Any, Any, Any, Any, Any],
) -> CompiledRetargetingProblem:
    """Compile enum-bearing domain objects into one validated backend representation."""

    robot = problem.robot
    joint_names = {_enum_identity(joint): joint.value for joint in robot.joints}
    link_names = {_enum_identity(link): link.value for link in robot.links}
    geometry_names: dict[EnumIdentity, str] = {
        _enum_identity(geometry): geometry.value for geometry in robot.geometries
    }
    geometry_references: list[str] = []
    for config in problem.constraints:
        pairs: tuple[GeometryPair, ...] = ()
        if isinstance(config, NonPenetrationConstraintConfig):
            pairs = config.geometry_pairs
        elif isinstance(config, SelfCollisionConstraintConfig):
            pairs = config.pairs
        for pair in pairs:
            geometry_names.setdefault(_enum_identity(pair.first), pair.first.value)
            geometry_names.setdefault(_enum_identity(pair.second), pair.second.value)
            geometry_references.extend((pair.first.value, pair.second.value))
    compiled_robot = compile_robot(robot)
    contacts = None
    if problem.contacts is not None:
        contacts = CompiledContactPlan(
            tracks=tuple(
                CompiledContactTrack(
                    subject=track.subject.value,
                    link_names=tuple(link_names[_enum_identity(link)] for link in track.links),
                    active_mask=track.active_mask,
                    support_mask=track.support_mask,
                )
                for track in problem.contacts.tracks
            ),
            frame_count=_frame_count(problem.contacts.frame_count),
            support=problem.contacts.support,
        )
    targets = None
    if problem.targets is not None:
        targets = CompiledLinkTargetPlan(
            tracks=tuple(
                CompiledLinkTargetTrack(
                    link_name=link_names[_enum_identity(track.link)],
                    positions=track.positions,
                    weights=np.asarray(track.weights, dtype=np.float64),
                    active_mask=np.asarray(track.active_mask, dtype=bool),
                )
                for track in problem.targets.tracks
            ),
            frame_count=_frame_count(problem.targets.frame_count),
        )
    return CompiledRetargetingProblem(
        source=problem,
        robot=compiled_robot,
        joint_mapping={
            binding.source.value: joint_names[_enum_identity(binding.target)]
            for binding in problem.joint_bindings
        },
        link_mapping={
            binding.source.value: link_names[_enum_identity(binding.target)]
            for binding in problem.link_bindings
        },
        link_names=link_names,
        joint_names=joint_names,
        geometry_names=geometry_names,
        geometry_references=tuple(dict.fromkeys(geometry_references)),
        contacts=contacts,
        targets=targets,
    )


def compile_robot(robot: AnyRobotSpec) -> CompiledRobotSpec:
    """Compile one typed robot specification for a backend or exporter."""

    joint_names = {_enum_identity(joint): joint.value for joint in robot.joints}
    link_names = {_enum_identity(link): link.value for link in robot.links}
    return CompiledRobotSpec(
        name=robot.name,
        height_m=robot.height_m,
        joint_names=tuple(joint.value for joint in robot.joints),
        link_names=tuple(link.value for link in robot.links),
        contact_links=tuple(link_names[_enum_identity(link)] for link in robot.contact_links),
        geometry_names=tuple(geometry.value for geometry in robot.geometries),
        joint_limits={
            joint_names[_enum_identity(joint)]: bounds
            for joint, bounds in robot.joint_limits.items()
        },
        mujoco_body_aliases={
            link_names[_enum_identity(link)]: alias
            for link, alias in robot.mujoco_body_aliases.items()
        },
        simple_kinematics={
            link_names[_enum_identity(link)]: CompiledSimpleKinematicPoint(
                joint_index=robot.joint_index(point.joint),
                offset=np.asarray(point.offset, dtype=np.float64),
                axis=_unit(np.asarray(point.axis, dtype=np.float64)),
                scale=float(point.scale),
            )
            for link, point in robot.simple_kinematics.items()
        },
        urdf_path=robot.urdf_path,
        mujoco_xml_path=robot.mujoco_xml_path,
        qpos_layout=robot.qpos_layout,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("kinematic axis must be non-zero")
    return vector / norm


def _enum_identity(member: StrEnum) -> EnumIdentity:
    return type(member), member.value


def _frame_count(value: int | None) -> int:
    if value is None:
        raise ValueError("compiled plans require a concrete frame count")
    return value


__all__ = [
    "CompiledContactFrame",
    "CompiledContactPlan",
    "CompiledLinkTargetPlan",
    "CompiledRetargetingProblem",
    "CompiledRobotSpec",
    "CompiledTargetFrame",
    "compile_problem",
    "compile_robot",
]
