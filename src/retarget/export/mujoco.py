"""Export retargeted qpos to MuJoCo-style tracking datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.protocols import KinematicsBackend
from retarget.export.registry import exporters
from retarget.export.spec import ExportResult, ExportSpec
from retarget.results.spec import RetargetingResult

QVEL_SCHEME = "first_frame_forward_difference_then_previous_interval"


class MuJoCoTrackingData(BaseModel):
    """Qpos/qvel tracking arrays ready for downstream MuJoCo experiments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    source_name: str
    qpos: FloatArray
    qvel: FloatArray
    time_s: FloatArray
    fps: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("qpos", "qvel", mode="before")
    @classmethod
    def _validate_matrix(cls, value: Any) -> FloatArray:
        arr = as_float_array(value)
        if arr.ndim != 2:
            raise ValueError("tracking arrays must be 2D")
        return arr

    @field_validator("time_s", mode="before")
    @classmethod
    def _validate_time(cls, value: Any) -> FloatArray:
        arr = as_float_array(value)
        if arr.ndim != 1:
            raise ValueError("time_s must be 1D")
        return arr

    @property
    def frame_count(self) -> int:
        """Number of tracking frames."""

        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        """Time between the first and last tracking sample."""

        return float(self.time_s[-1]) if self.frame_count else 0.0

    @model_validator(mode="after")
    def _validate_lengths(self) -> MuJoCoTrackingData:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.qpos.shape[0] != self.qvel.shape[0]:
            raise ValueError("qpos and qvel must have the same frame count")
        if self.time_s.shape != (self.qpos.shape[0],):
            raise ValueError("time_s length must match qpos frames")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        return self

    def save_npz(self, path: str | Path) -> Path:
        """Save in a compatibility-friendly NPZ schema."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            schema_version=np.asarray(self.schema_version),
            fps=np.asarray(self.fps),
            time_s=self.time_s,
            qpos=self.qpos,
            qvel=self.qvel,
            joint_pos=self.qpos,
            joint_vel=self.qvel,
            source_name=self.source_name,
            metadata_json=json.dumps(self.metadata, default=_json_default),
            metadata=np.asarray(self.metadata, dtype=object),
        )
        return output


class MuJoCoTrackingExporter:
    """Export `RetargetingResult` objects to MuJoCo-style tracking NPZ files."""

    format_name = "mujoco_npz"

    def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult:
        """Build tracking arrays and save them to disk."""

        tracking = build_mujoco_tracking_data(
            result,
            output_fps=spec.output_fps,
            kinematics_backend=spec.kinematics_backend,
            metadata=spec.metadata,
        )
        path = tracking.save_npz(spec.output_path)
        return ExportResult(
            format_name=self.format_name,
            path=path,
            frame_count=tracking.frame_count,
            fps=tracking.fps,
            metadata={
                "source_name": result.name,
                "qpos_dimension": tracking.qpos.shape[1],
                "qvel_dimension": tracking.qvel.shape[1],
                **tracking.metadata,
            },
        )


def build_mujoco_tracking_data(
    result: RetargetingResult,
    *,
    output_fps: int | None = None,
    kinematics_backend: KinematicsBackend | None = None,
    metadata: dict[str, Any] | None = None,
) -> MuJoCoTrackingData:
    """Build qpos/qvel arrays for downstream tracking experiments."""

    fps = float(output_fps) if output_fps is not None else float(result.fps)
    if fps <= 0:
        raise ValueError("output_fps must be positive")
    qpos = result.resampled(fps).qpos if not np.isclose(fps, result.fps) else result.qpos.copy()
    if kinematics_backend is None:
        qvel = _finite_difference_qvel(qpos, fps)
        qvel_source = "finite_difference"
    else:
        qvel = _kinematics_qvel(qpos, fps, kinematics_backend)
        qvel_source = type(kinematics_backend).__name__
    time_s = np.arange(qpos.shape[0], dtype=np.float64) / fps
    duration_s = float(time_s[-1]) if len(time_s) else 0.0
    return MuJoCoTrackingData(
        source_name=result.name,
        qpos=qpos,
        qvel=qvel,
        time_s=time_s,
        fps=float(fps),
        metadata={
            "result_status": result.status.value,
            "qvel_source": qvel_source,
            "qvel_scheme": QVEL_SCHEME,
            "source_fps": result.fps,
            "output_fps": float(fps),
            "source_frame_count": result.frame_count,
            "frame_count": int(qpos.shape[0]),
            "duration_s": duration_s,
            "resampled": not np.isclose(fps, result.fps),
            "qpos_dimension": qpos.shape[1],
            "qvel_dimension": qvel.shape[1],
            **(metadata or {}),
        },
    )


def export_tracking(result: RetargetingResult, spec: ExportSpec) -> ExportResult:
    """Export using a registered exporter."""

    return exporters.get(spec.format_name).export(result, spec)


def export_tracking_npz(result: RetargetingResult, output_path: str | Path, *, output_fps: int | None = None) -> Path:
    """Compatibility wrapper for MuJoCo tracking NPZ export."""

    export = export_tracking(
        result,
        ExportSpec(format_name="mujoco_npz", output_path=output_path, output_fps=output_fps),
    )
    return export.path


def _finite_difference_qvel(qpos: np.ndarray, fps: float) -> np.ndarray:
    if qpos.shape[0] <= 1:
        return np.zeros_like(qpos)
    qvel = np.zeros_like(qpos, dtype=np.float64)
    qvel[0] = (qpos[1] - qpos[0]) * fps
    qvel[1:] = np.diff(qpos, axis=0) * fps
    return qvel


def _kinematics_qvel(qpos: np.ndarray, fps: float, backend: KinematicsBackend) -> np.ndarray:
    dt = 1.0 / fps
    if qpos.shape[0] == 1:
        return _as_qvel_matrix((backend.qpos_to_qvel(qpos[0], qpos[0], dt),))

    velocities = [backend.qpos_to_qvel(qpos[1], qpos[0], dt)]
    velocities.extend(backend.qpos_to_qvel(qpos[frame], qpos[frame - 1], dt) for frame in range(1, qpos.shape[0]))
    return _as_qvel_matrix(tuple(velocities))


def _as_qvel_matrix(vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    qvel = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    if not qvel:
        raise ValueError("at least one qvel vector is required")
    expected_shape = qvel[0].shape
    if len(expected_shape) != 1:
        raise ValueError("kinematics_backend.qpos_to_qvel must return a 1D array")
    for frame, vector in enumerate(qvel):
        if vector.shape != expected_shape:
            raise ValueError(
                "kinematics_backend.qpos_to_qvel returned inconsistent shapes: "
                f"frame 0 has {expected_shape}, frame {frame} has {vector.shape}"
            )
    return np.vstack(qvel)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


exporters.register(MuJoCoTrackingExporter.format_name, MuJoCoTrackingExporter())
