"""Export retargeted qpos to MuJoCo-style tracking datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import ExportFormat
from retarget.core.protocols import KinematicsBackend
from retarget.export.registry import exporters
from retarget.export.spec import ExportResult, ExportSpec
from retarget.results.spec import RetargetingResult

# Qvel policy label stored in the structured report: frame 0 uses forward difference to frame 1;
# frames 1..N-1 use the interval from the previous qpos sample.
QVEL_SCHEME = "first_frame_forward_difference_then_previous_interval"


class MuJoCoTrackingReport(BaseModel):
    """Structured details of MuJoCo tracking conversion."""

    qvel_source: str
    qvel_scheme: str
    source_fps: float
    output_fps: float
    source_frame_count: int
    frame_count: int
    duration_s: float
    resampled: bool
    qpos_dimension: int
    qvel_dimension: int


class MuJoCoTrackingData(BaseModel):
    """Qpos/qvel tracking arrays ready for downstream MuJoCo experiments.

    Attributes:
        schema_version (int): NPZ schema version written by :meth:`save_npz` (default ``1``).
        source_name (str): Retargeting result or clip name carried into the export.
        qpos (FloatArray): Generalized positions with shape ``(frames, nq)``.
        qvel (FloatArray): Generalized velocities with shape ``(frames, nv)``, aligned with ``qpos``.
        time_s (FloatArray): Sample times in seconds with shape ``(frames,)``.
        fps (float): Playback frame rate used to build ``time_s`` and qvel.
        report (MuJoCoTrackingReport): Structured conversion report.
        provenance (dict[str, Any]): Origin information supplied by the caller.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 2
    source_name: str
    qpos: FloatArray
    qvel: FloatArray
    time_s: FloatArray
    fps: float
    report: MuJoCoTrackingReport
    provenance: dict[str, Any] = Field(default_factory=dict)

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
        """Save the current strict MuJoCo tracking NPZ schema."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            manifest_json=json.dumps(
                self.model_dump(mode="json", exclude={"qpos", "qvel", "time_s"}),
                default=_json_default,
            ),
            time_s=self.time_s,
            qpos=self.qpos,
            qvel=self.qvel,
        )
        return output


class MuJoCoTrackingExporter:
    """Export `RetargetingResult` objects to MuJoCo-style tracking NPZ files."""

    format_name = ExportFormat.MUJOCO_NPZ

    def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult:
        """Build tracking arrays and save them to disk."""

        tracking = build_mujoco_tracking_data(
            result,
            output_fps=spec.output_fps,
            kinematics_backend=spec.kinematics_backend,
            provenance=spec.provenance,
        )
        path = tracking.save_npz(spec.output_path)
        return ExportResult(
            format_name=self.format_name,
            path=path,
            frame_count=tracking.frame_count,
            fps=tracking.fps,
            qpos_dimension=tracking.report.qpos_dimension,
            qvel_dimension=tracking.report.qvel_dimension,
            provenance={"source_name": result.name, **tracking.provenance},
        )


def build_mujoco_tracking_data(
    result: RetargetingResult,
    *,
    output_fps: int | None = None,
    kinematics_backend: KinematicsBackend | None = None,
    provenance: dict[str, Any] | None = None,
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
        report=MuJoCoTrackingReport(
            qvel_source=qvel_source,
            qvel_scheme=QVEL_SCHEME,
            source_fps=result.fps,
            output_fps=float(fps),
            source_frame_count=result.frame_count,
            frame_count=int(qpos.shape[0]),
            duration_s=duration_s,
            resampled=not np.isclose(fps, result.fps),
            qpos_dimension=qpos.shape[1],
            qvel_dimension=qvel.shape[1],
        ),
        provenance={"result_status": result.status.value, **(provenance or {})},
    )


def export_tracking(result: RetargetingResult, spec: ExportSpec) -> ExportResult:
    """Export using a registered exporter."""

    return exporters.get(spec.format_name).export(result, spec)


def export_tracking_npz(result: RetargetingResult, output_path: str | Path, *, output_fps: int | None = None) -> Path:
    """Export a retargeting result to a MuJoCo-style tracking ``.npz`` file.

    Convenience wrapper around :func:`export_tracking` with format ``"mujoco_npz"``.
    The NPZ contains ``qpos``, ``qvel``, ``time_s``, and a strict JSON manifest
    compatible with downstream MuJoCo tracking experiments.

    Args:
        result (RetargetingResult): Solved retargeting output to export.
        output_path (str | Path): Destination ``.npz`` path (parent directories are created).
        output_fps (int | None): Optional resample rate before export; uses ``result.fps`` when omitted.

    Returns:
        Path: Resolved path of the written NPZ file.
    """

    export = export_tracking(
        result,
        ExportSpec(format_name=ExportFormat.MUJOCO_NPZ, output_path=output_path, output_fps=output_fps),
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


exporters.register(ExportFormat.MUJOCO_NPZ, MuJoCoTrackingExporter())
