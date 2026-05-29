"""Kinematics backend implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from retarget.kinematics.registry import kinematics_backends
from retarget.kinematics.types import GeometryDistance
from retarget.robots.spec import RobotSpec


class SimpleKinematicsBackend:
    """A deterministic fixture backend for tests and examples.

    It is not a physics model; it provides stable link-like positions from qpos so
    metrics and examples can run without MuJoCo assets.
    """

    def __init__(self, robot: RobotSpec) -> None:
        self.robot = robot

    def forward_kinematics(self, qpos: NDArray[np.float64], link_names: tuple[str, ...]) -> NDArray[np.float64]:
        """Return link positions for the fixture model."""

        return self.link_positions(qpos, link_names)

    def link_positions(self, qpos: NDArray[np.float64], link_names: tuple[str, ...]) -> NDArray[np.float64]:
        """Return deterministic link-like point positions."""

        positions, _ = self.point_jacobians(qpos, link_names)
        return positions

    def body_jacobians(
        self,
        qpos: NDArray[np.float64],
        body_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return fixture body positions, translational Jacobians, and zero rotational Jacobians."""

        positions, translational = self.point_jacobians(qpos, body_names)
        rotational = np.zeros_like(translational)
        return positions, translational, rotational

    def point_jacobians(
        self,
        qpos: NDArray[np.float64],
        point_names: tuple[str, ...],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return point positions and dense Jacobians with respect to actuated joints."""

        q = np.asarray(qpos, dtype=np.float64)
        root = q[:3]
        joints = q[self.robot.qpos_layout.joint_slice(self.robot.dof)]
        positions = []
        jacobians = np.zeros((len(point_names), 3, self.robot.dof), dtype=np.float64)
        for i, name in enumerate(point_names):
            joint_index = self._point_to_joint_index(name, fallback=i)
            value = joints[joint_index] if len(joints) else 0.0
            axis = self._joint_axis(joint_index)
            scale = self._joint_scale(name)
            positions.append(root + self._point_offset(name) + scale * value * axis)
            jacobians[i, :, joint_index] = scale * axis
        return np.asarray(positions, dtype=np.float64), jacobians

    def qpos_to_qvel(
        self,
        qpos: NDArray[np.float64],
        previous_qpos: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Convert two fixture qpos vectors to a finite-difference velocity."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        q = np.asarray(qpos, dtype=np.float64)
        previous = np.asarray(previous_qpos, dtype=np.float64)
        if q.shape != previous.shape:
            raise ValueError("qpos and previous_qpos must have the same shape")
        return (q - previous) / dt

    def integrate_qvel(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate a fixture velocity with Euler addition and normalize the root quaternion."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        q = np.asarray(qpos, dtype=np.float64)
        velocity = np.asarray(qvel, dtype=np.float64)
        if q.shape != velocity.shape:
            raise ValueError("qpos and qvel must have the same shape for SimpleKinematicsBackend")
        integrated = q + velocity * dt
        root_quat = self.robot.qpos_layout.root_quaternion
        quat = integrated[root_quat[0] : root_quat[1]]
        norm = float(np.linalg.norm(quat))
        if quat.shape == (4,) and norm > 0:
            integrated[root_quat[0] : root_quat[1]] = quat / norm
        return integrated

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return configured joint limits."""

        return dict(self.robot.joint_limits)

    def geom_distances(
        self,
        qpos: NDArray[np.float64],
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
        *,
        max_distance: float = np.inf,
    ) -> tuple[GeometryDistance, ...]:
        """Return point-distance approximations for fixture geometry pairs."""

        pairs = geom_pairs if geom_pairs is not None else self._default_geom_pairs()
        if not pairs:
            return ()
        names = tuple(dict.fromkeys(name for pair in pairs for name in pair))
        positions = dict(zip(names, self.link_positions(qpos, names), strict=True))
        distances: list[GeometryDistance] = []
        for first, second in pairs:
            first_point = positions[first]
            second_point = positions[second]
            delta = second_point - first_point
            distance = float(np.linalg.norm(delta))
            if distance > max_distance:
                continue
            normal = delta / distance if distance > 1e-12 else np.zeros(3, dtype=np.float64)
            distances.append(
                GeometryDistance(
                    first=first,
                    second=second,
                    distance=distance,
                    point_on_first=first_point,
                    point_on_second=second_point,
                    normal_from_first_to_second=normal,
                )
            )
        return tuple(distances)

    def collision_candidates(
        self,
        qpos: NDArray[np.float64],
        *,
        margin: float = 0.0,
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> tuple[GeometryDistance, ...]:
        """Return fixture geometry pairs no farther apart than `margin`."""

        return self.geom_distances(qpos, geom_pairs, max_distance=margin)

    def _point_to_joint_index(self, point_name: str, *, fallback: int) -> int:
        normalized = point_name.lower().replace("-", "_")
        aliases: tuple[str, ...]
        if "left" in normalized and ("toe" in normalized or "foot" in normalized or "ankle" in normalized):
            aliases = ("left_ankle", "left_knee", "left_hip")
        elif "right" in normalized and ("toe" in normalized or "foot" in normalized or "ankle" in normalized):
            aliases = ("right_ankle", "right_knee", "right_hip")
        elif "left" in normalized and ("hand" in normalized or "wrist" in normalized):
            aliases = ("left_elbow", "left_shoulder")
        elif "right" in normalized and ("hand" in normalized or "wrist" in normalized):
            aliases = ("right_elbow", "right_shoulder")
        elif "pelvis" in normalized or "torso" in normalized or "spine" in normalized:
            aliases = ("spine", "waist", "hip")
        else:
            aliases = (normalized,)

        for alias in aliases:
            for index, joint_name in enumerate(self.robot.joint_names):
                if alias in joint_name.lower():
                    return index
        return fallback % self.robot.dof

    def _joint_axis(self, joint_index: int) -> NDArray[np.float64]:
        axes = (
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        return axes[joint_index % len(axes)]

    def _joint_scale(self, point_name: str) -> float:
        lower = point_name.lower()
        if "toe" in lower or "foot" in lower:
            return 0.08
        if "hand" in lower or "wrist" in lower:
            return 0.10
        return 0.05

    def _point_offset(self, point_name: str) -> NDArray[np.float64]:
        lower = point_name.lower()
        side = -1.0 if "left" in lower or lower.startswith("l_") else 1.0
        if "toe" in lower or "foot" in lower or "ankle" in lower:
            return np.array([0.12 * side, 0.0, -0.85 * self.robot.height_m], dtype=np.float64)
        if "hand" in lower or "wrist" in lower:
            return np.array([0.32 * side, 0.0, -0.10 * self.robot.height_m], dtype=np.float64)
        if "head" in lower:
            return np.array([0.0, 0.0, 0.30 * self.robot.height_m], dtype=np.float64)
        if "pelvis" in lower or "torso" in lower or "spine" in lower:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return np.array([0.06 * side, 0.0, -0.15 * self.robot.height_m], dtype=np.float64)

    def _default_geom_pairs(self) -> tuple[tuple[str, str], ...]:
        names = self.robot.link_names or self.robot.contact_links
        return tuple((first, second) for idx, first in enumerate(names) for second in names[idx + 1 :])


class MuJoCoKinematicsBackend:
    """MuJoCo-backed kinematics adapter.

    This class intentionally keeps a small surface. More advanced operations
    such as contact Jacobians and qvel transforms should be added here rather
    than leaking MuJoCo objects through the public pipeline.
    """

    def __init__(self, robot: RobotSpec, xml_path: str | Path | None = None) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install retarget[mujoco] to use MuJoCoKinematicsBackend") from exc

        path = Path(xml_path or robot.mujoco_xml_path or "")
        if not path:
            raise ValueError("A MuJoCo XML path is required")
        self._mujoco = mujoco
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

    def forward_kinematics(self, qpos: NDArray[np.float64], link_names: tuple[str, ...]) -> NDArray[np.float64]:
        """Return MuJoCo body positions for named links."""

        return self.link_positions(qpos, link_names)

    def link_positions(self, qpos: NDArray[np.float64], link_names: tuple[str, ...]) -> NDArray[np.float64]:
        """Return MuJoCo body positions for named links."""

        mujoco = self._mujoco
        self._set_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)
        positions = []
        for link_name in link_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id < 0:
                raise KeyError(f"Body {link_name!r} not found in MuJoCo model")
            positions.append(self.data.xpos[body_id].copy())
        return np.asarray(positions, dtype=np.float64)

    def point_jacobians(
        self,
        qpos: NDArray[np.float64],
        point_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return body positions and MuJoCo Jacobians with respect to actuated joints."""

        positions, translational, _rotational = self.body_jacobians(qpos, point_names)
        return positions, translational

    def body_jacobians(
        self,
        qpos: NDArray[np.float64],
        body_names: tuple[str, ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return body positions plus translational and rotational Jacobians."""

        mujoco = self._mujoco
        self._set_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)
        transform = self._qdot_to_qvel_transform()
        translational = np.zeros((len(body_names), 3, self.robot.dof), dtype=np.float64)
        rotational = np.zeros((len(body_names), 3, self.robot.dof), dtype=np.float64)
        positions = np.zeros((len(body_names), 3), dtype=np.float64)
        qpos_joint_slice = self.robot.qpos_layout.joint_slice(self.robot.dof)
        for idx, body_name in enumerate(body_names):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise KeyError(f"Body {body_name!r} not found in MuJoCo model")
            positions[idx] = self.data.xpos[body_id]
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            translational[idx] = (jacp @ transform)[:, qpos_joint_slice]
            rotational[idx] = (jacr @ transform)[:, qpos_joint_slice]
        return positions, translational, rotational

    def qpos_to_qvel(
        self,
        qpos: NDArray[np.float64],
        previous_qpos: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Use MuJoCo's position differencing to compute generalized velocity."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        q = np.asarray(qpos, dtype=np.float64)
        previous = np.asarray(previous_qpos, dtype=np.float64)
        if q.shape[0] < self.model.nq or previous.shape[0] < self.model.nq:
            raise ValueError(f"qpos vectors must have at least {self.model.nq} elements")
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        self._mujoco.mj_differentiatePos(self.model, qvel, dt, previous[: self.model.nq], q[: self.model.nq])
        return qvel

    def integrate_qvel(
        self,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate a generalized velocity through MuJoCo and preserve any trailing object qpos."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        q = np.asarray(qpos, dtype=np.float64)
        velocity = np.asarray(qvel, dtype=np.float64)
        if q.shape[0] < self.model.nq:
            raise ValueError(f"qpos has length {q.shape[0]}, but MuJoCo model requires {self.model.nq}")
        if velocity.shape[0] < self.model.nv:
            raise ValueError(f"qvel has length {velocity.shape[0]}, but MuJoCo model requires {self.model.nv}")
        integrated = q.copy()
        model_qpos = integrated[: self.model.nq].copy()
        self._mujoco.mj_integratePos(self.model, model_qpos, velocity[: self.model.nv], dt)
        integrated[: self.model.nq] = model_qpos
        return integrated

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return MuJoCo joint ranges with explicit `RobotSpec` overrides applied."""

        limits = self._mujoco_joint_limits()
        limits.update(self.robot.joint_limits)
        return limits

    def geom_distances(
        self,
        qpos: NDArray[np.float64],
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
        *,
        max_distance: float = np.inf,
    ) -> tuple[GeometryDistance, ...]:
        """Return MuJoCo geom distances for named geom pairs."""

        mujoco = self._mujoco
        self._set_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)
        pairs = geom_pairs if geom_pairs is not None else self._default_geom_pairs()
        distances: list[GeometryDistance] = []
        for first, second in pairs:
            first_id = self._geom_id(first)
            second_id = self._geom_id(second)
            distance, first_point, second_point = self._geom_distance(first_id, second_id, max_distance=max_distance)
            if distance > max_distance:
                continue
            delta = second_point - first_point
            norm = float(np.linalg.norm(delta))
            normal = delta / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
            distances.append(
                GeometryDistance(
                    first=first,
                    second=second,
                    distance=distance,
                    point_on_first=first_point,
                    point_on_second=second_point,
                    normal_from_first_to_second=normal,
                )
            )
        return tuple(distances)

    def collision_candidates(
        self,
        qpos: NDArray[np.float64],
        *,
        margin: float = 0.0,
        geom_pairs: tuple[tuple[str, str], ...] | None = None,
    ) -> tuple[GeometryDistance, ...]:
        """Return MuJoCo geom pairs within a collision margin."""

        return self.geom_distances(qpos, geom_pairs, max_distance=margin)

    def _set_qpos(self, qpos: NDArray[np.float64]) -> None:
        q = np.asarray(qpos, dtype=np.float64)
        if q.shape[0] < self.model.nq:
            raise ValueError(f"qpos has length {q.shape[0]}, but MuJoCo model requires {self.model.nq}")
        self.data.qpos[:] = q[: self.model.nq]

    def _geom_id(self, geom_name: str) -> int:
        mujoco = self._mujoco
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise KeyError(f"Geom {geom_name!r} not found in MuJoCo model")
        return int(geom_id)

    def _geom_name(self, geom_id: int) -> str:
        mujoco = self._mujoco
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return str(name) if name else f"geom_{geom_id}"

    def _joint_name(self, joint_id: int) -> str:
        mujoco = self._mujoco
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        return str(name) if name else f"joint_{joint_id}"

    def _mujoco_joint_limits(self) -> dict[str, tuple[float, float]]:
        mujoco = self._mujoco
        limits: dict[str, tuple[float, float]] = {}
        qpos_to_robot_joint = {
            self.robot.qpos_layout.joint_start + idx: name for idx, name in enumerate(self.robot.joint_names)
        }
        for joint_id in range(self.model.njnt):
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            lower, upper = np.asarray(self.model.jnt_range[joint_id], dtype=np.float64)
            model_name = self._joint_name(joint_id)
            robot_name: str | None
            if model_name in self.robot.joint_names:
                robot_name = model_name
            else:
                robot_name = qpos_to_robot_joint.get(int(self.model.jnt_qposadr[joint_id]))
            if robot_name is not None:
                limits[robot_name] = (float(lower), float(upper))
        return limits

    def _default_geom_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self._geom_name(first), self._geom_name(second))
            for first in range(self.model.ngeom)
            for second in range(first + 1, self.model.ngeom)
            if self.model.geom_bodyid[first] != self.model.geom_bodyid[second]
        )

    def _geom_distance(
        self,
        first_id: int,
        second_id: int,
        *,
        max_distance: float,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        mujoco = self._mujoco
        if hasattr(mujoco, "mj_geomDistance"):
            fromto = np.zeros(6, dtype=np.float64)
            distance_limit = max_distance if np.isfinite(max_distance) else 1.0e6
            distance = float(mujoco.mj_geomDistance(self.model, self.data, first_id, second_id, distance_limit, fromto))
            return distance, fromto[:3].copy(), fromto[3:].copy()
        first_point = np.asarray(self.data.geom_xpos[first_id], dtype=np.float64).copy()
        second_point = np.asarray(self.data.geom_xpos[second_id], dtype=np.float64).copy()
        return float(np.linalg.norm(second_point - first_point)), first_point, second_point

    def _qdot_to_qvel_transform(self) -> NDArray[np.float64]:
        mujoco = self._mujoco
        transform = np.zeros((self.model.nv, self.model.nq), dtype=np.float64)
        for joint_idx in range(self.model.njnt):
            joint_type = self.model.jnt_type[joint_idx]
            qadr = int(self.model.jnt_qposadr[joint_idx])
            dadr = int(self.model.jnt_dofadr[joint_idx])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                transform[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)
                transform[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = self._quat_qdot_to_angular_velocity_matrix(
                    self.data.qpos[qadr + 3 : qadr + 7]
                )
            elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                transform[dadr, qadr] = 1.0
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                transform[dadr : dadr + 3, qadr : qadr + 4] = self._quat_qdot_to_angular_velocity_matrix(
                    self.data.qpos[qadr : qadr + 4]
                )
        return transform

    @staticmethod
    def _quat_qdot_to_angular_velocity_matrix(quaternion: NDArray[np.float64]) -> NDArray[np.float64]:
        qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
        return 2.0 * np.array(
            [
                [-qx, qw, qz, -qy],
                [-qy, -qz, qw, qx],
                [-qz, qy, -qx, qw],
            ],
            dtype=np.float64,
        )


kinematics_backends.register("simple", lambda robot: SimpleKinematicsBackend(robot))
kinematics_backends.register("mujoco", lambda robot: MuJoCoKinematicsBackend(robot))
