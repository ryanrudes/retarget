"""Typed optimization variable selection for retargeting problems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from retarget.core.array import FloatArray
from retarget.robots.spec import RobotSpec

QposVariableKind = Literal["actuated", "qpos_slice", "qpos_indices"]


@dataclass(frozen=True)
class ResolvedQposVariables:
    """Resolved qpos coordinates used as one SQP subproblem variable.

    The solver still optimizes an increment vector, but the columns can now
    refer to any qpos coordinate, not only actuated joints. This is the
    abstraction needed for floating-base ``q_a`` style solves.
    """

    spec: QposVariableSpec
    indices: NDArray[np.int64]
    lower: FloatArray
    upper: FloatArray
    quaternion_slice: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if indices.ndim != 1:
            raise ValueError("variable indices must be 1D")
        if len(set(int(index) for index in indices)) != indices.shape[0]:
            raise ValueError("variable indices must be unique")
        if lower.shape != indices.shape or upper.shape != indices.shape:
            raise ValueError("variable bounds must match variable indices")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        """Number of optimization variables."""

        return int(self.indices.shape[0])

    def values(self, qpos: FloatArray) -> FloatArray:
        """Return this variable set's current qpos values."""

        q = np.asarray(qpos, dtype=np.float64)
        if q.ndim != 1:
            raise ValueError("qpos must be 1D")
        if np.any(self.indices >= q.shape[0]):
            raise ValueError("variable indices exceed qpos length")
        return q[self.indices].copy()

    def apply_delta(self, qpos: FloatArray, delta: FloatArray) -> FloatArray:
        """Return a copy of ``qpos`` after applying one solver increment."""

        q = np.asarray(qpos, dtype=np.float64).copy()
        step = np.asarray(delta, dtype=np.float64)
        if step.shape != (self.size,):
            raise ValueError(f"delta must have shape ({self.size},)")
        if np.any(self.indices >= q.shape[0]):
            raise ValueError("variable indices exceed qpos length")
        q[self.indices] = q[self.indices] + step
        if self.spec.normalize_quaternion and self.quaternion_slice is not None:
            start, stop = self.quaternion_slice
            quat = q[start:stop]
            norm = float(np.linalg.norm(quat))
            if quat.shape == (4,) and norm > 0.0:
                q[start:stop] = quat / norm
        return q

    def metadata(self) -> dict[str, object]:
        """JSON-safe provenance for result metadata."""

        return {
            **self.spec.model_dump(mode="json"),
            "indices": [int(index) for index in self.indices],
            "size": self.size,
        }


class QposVariableSpec(BaseModel):
    """Configuration describing which qpos coordinates the optimizer may change.

    ``actuated`` preserves the package default: only robot joints are decision
    variables. ``qpos_slice`` and ``qpos_indices`` allow floating-base,
    Holosoma-style, or otherwise custom variable sets without changing terms.
    """

    model_config = ConfigDict(extra="forbid")

    kind: QposVariableKind = "actuated"
    start: int | None = None
    stop: int | None = None
    indices: tuple[int, ...] = ()
    actuated_start_offset: int | None = None
    unbounded_limit: float = 1e6
    normalize_quaternion: bool = True

    @classmethod
    def actuated(cls) -> Self:
        """Return the default actuated-joint variable policy."""

        return cls(kind="actuated")

    @classmethod
    def qpos_slice(
        cls,
        *,
        start: int,
        stop: int | None = None,
        normalize_quaternion: bool = True,
    ) -> Self:
        """Return a contiguous qpos variable policy."""

        return cls(kind="qpos_slice", start=start, stop=stop, normalize_quaternion=normalize_quaternion)

    @classmethod
    def qpos_indices(cls, indices: tuple[int, ...], *, normalize_quaternion: bool = True) -> Self:
        """Return an explicit qpos-index variable policy."""

        return cls(kind="qpos_indices", indices=indices, normalize_quaternion=normalize_quaternion)

    @classmethod
    def from_actuated_start_offset(cls, offset: int, *, stop: int | None = None) -> Self:
        """Return a slice starting relative to ``robot.qpos_layout.joint_start``.

        Holosoma's ``q_a_init_idx=-7`` is represented by
        ``from_actuated_start_offset(-7)`` for the default floating-base qpos
        layout, giving a variable slice from root position through actuated
        joints.
        """

        return cls(kind="qpos_slice", actuated_start_offset=offset, stop=stop)

    @classmethod
    def holosoma_q_a(cls, q_a_init_idx: int = -7) -> Self:
        """Return the variable policy matching Holosoma's default ``q_a`` slice."""

        return cls.from_actuated_start_offset(q_a_init_idx)

    @model_validator(mode="after")
    def _validate_policy(self) -> QposVariableSpec:
        if self.kind == "actuated":
            has_extra_fields = (
                self.start is not None
                or self.stop is not None
                or bool(self.indices)
                or self.actuated_start_offset is not None
            )
            if has_extra_fields:
                raise ValueError("actuated variables do not accept start, stop, indices, or actuated_start_offset")
        elif self.kind == "qpos_slice":
            if self.indices:
                raise ValueError("qpos_slice variables do not accept explicit indices")
            if self.start is None and self.actuated_start_offset is None:
                raise ValueError("qpos_slice variables require start or actuated_start_offset")
            if self.start is not None and self.actuated_start_offset is not None:
                raise ValueError("qpos_slice accepts only one of start or actuated_start_offset")
            if self.start is not None and self.start < 0:
                raise ValueError("qpos_slice start must be non-negative")
            if self.stop is not None and self.stop <= 0:
                raise ValueError("qpos_slice stop must be positive")
            if self.start is not None and self.stop is not None and self.stop <= self.start:
                raise ValueError("qpos_slice stop must be greater than start")
        else:
            if self.start is not None or self.stop is not None or self.actuated_start_offset is not None:
                raise ValueError("qpos_indices variables do not accept start, stop, or actuated_start_offset")
            if not self.indices:
                raise ValueError("qpos_indices variables require at least one index")
            if any(index < 0 for index in self.indices):
                raise ValueError("qpos_indices entries must be non-negative")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("qpos_indices entries must be unique")
        return self

    @field_validator("unbounded_limit")
    @classmethod
    def _positive_limit(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("unbounded_limit must be positive")
        return float(value)

    def resolve(
        self,
        robot: RobotSpec,
        *,
        qpos_size: int,
        joint_limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> ResolvedQposVariables:
        """Resolve this policy against a robot and concrete qpos width."""

        indices = self._indices(robot, qpos_size=qpos_size)
        lower, upper = _limits_for_indices(
            robot,
            indices,
            joint_limits=joint_limits,
            unbounded_limit=self.unbounded_limit,
        )
        return ResolvedQposVariables(
            spec=self,
            indices=indices,
            lower=lower,
            upper=upper,
            quaternion_slice=robot.qpos_layout.root_quaternion,
        )

    def _indices(self, robot: RobotSpec, *, qpos_size: int) -> NDArray[np.int64]:
        layout = robot.qpos_layout
        joint_stop = layout.joint_start + robot.dof
        raw: Iterable[int]
        if self.kind == "actuated":
            raw = range(layout.joint_start, joint_stop)
        elif self.kind == "qpos_slice":
            start = (
                layout.joint_start + self.actuated_start_offset
                if self.actuated_start_offset is not None
                else self.start
            )
            stop = joint_stop if self.stop is None else self.stop
            if start is None:
                raise ValueError("qpos_slice start could not be resolved")
            if start < 0 or stop <= start:
                raise ValueError("resolved qpos_slice bounds are invalid")
            raw = range(start, stop)
        else:
            raw = self.indices
        indices = np.asarray(tuple(raw), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("variable policy resolved to no qpos indices")
        if int(indices.max()) >= qpos_size:
            raise ValueError(
                f"variable policy references qpos index {int(indices.max())}, but qpos size is {qpos_size}"
            )
        return indices


def _limits_for_indices(
    robot: RobotSpec,
    indices: NDArray[np.int64],
    *,
    joint_limits: Mapping[str, tuple[float, float]] | None,
    unbounded_limit: float,
) -> tuple[FloatArray, FloatArray]:
    lower = np.full(indices.shape, -unbounded_limit, dtype=np.float64)
    upper = np.full(indices.shape, unbounded_limit, dtype=np.float64)
    limits = dict(joint_limits or {})
    layout = robot.qpos_layout
    joint_start = layout.joint_start
    joint_stop = joint_start + robot.dof
    quat_start, quat_stop = layout.root_quaternion
    for col, qpos_idx in enumerate(indices):
        index = int(qpos_idx)
        if quat_start <= index < quat_stop:
            lower[col] = -1.0
            upper[col] = 1.0
        elif joint_start <= index < joint_stop:
            joint_name = robot.joint_names[index - joint_start]
            lo, hi = limits.get(joint_name, robot.joint_limits.get(joint_name, (-unbounded_limit, unbounded_limit)))
            lower[col] = float(lo)
            upper[col] = float(hi)
    return lower, upper


__all__ = ["QposVariableKind", "QposVariableSpec", "ResolvedQposVariables"]
