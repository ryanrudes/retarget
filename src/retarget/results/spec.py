"""Result and report models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import RunStatus
from retarget.core.timing import resample_linear


class RetargetingResult(BaseModel):
    """Serialized retargeting output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    name: str
    status: RunStatus
    qpos: FloatArray
    fps: float = 30.0
    cost: FloatArray | None = None
    human_joints: FloatArray | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("qpos", mode="before")
    @classmethod
    def _validate_qpos(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="qpos")
        if arr.ndim != 2:
            raise ValueError("qpos must have shape (frames, nq)")
        return arr

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="cost")
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("cost must be 1D")
        return arr

    @field_validator("human_joints", mode="before")
    @classmethod
    def _validate_human_joints(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="human_joints")
        if arr.ndim != 3:
            raise ValueError("human_joints must have shape (frames, joints, 3)")
        return arr

    @model_validator(mode="after")
    def _validate_lengths(self) -> RetargetingResult:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.qpos.shape[0] == 0:
            raise ValueError("qpos must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.cost is not None and len(self.cost) not in (1, self.qpos.shape[0]):
            raise ValueError("cost length must be 1 or match qpos frames")
        if self.human_joints is not None and self.human_joints.shape[0] != self.qpos.shape[0]:
            raise ValueError("human_joints frame count must match qpos")
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return int(self.qpos.shape[0])

    def resampled(self, fps: float, *, name: str | None = None) -> RetargetingResult:
        """Return this result sampled on a new FPS grid."""

        metadata = dict(self.metadata)
        if not np.isclose(float(fps), self.fps):
            metadata["resampled_from_fps"] = self.fps
            metadata["resampled_to_fps"] = float(fps)
        cost = None
        if self.cost is not None:
            cost = (
                resample_linear(self.cost, self.fps, fps)
                if len(self.cost) == self.frame_count
                else np.asarray(self.cost, dtype=np.float64).copy()
            )
        human_joints = (
            resample_linear(self.human_joints, self.fps, fps) if self.human_joints is not None else None
        )
        return RetargetingResult(
            schema_version=self.schema_version,
            name=name or self.name,
            status=self.status,
            qpos=resample_linear(self.qpos, self.fps, fps),
            fps=fps,
            cost=cost,
            human_joints=human_joints,
            warnings=self.warnings,
            metadata=metadata,
        )

    def save_npz(self, path: str | Path) -> Path:
        """Save result to `.npz`."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": np.asarray(self.schema_version),
            "name": self.name,
            "status": self.status.value,
            "qpos": self.qpos,
            "fps": np.asarray(self.fps),
            "metadata_json": json.dumps(self.metadata, default=_json_default),
            "metadata": np.asarray(self.metadata, dtype=object),
            "warnings_json": json.dumps(self.warnings),
            "warnings": np.asarray(self.warnings, dtype=object),
        }
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.human_joints is not None:
            payload["human_joints"] = self.human_joints
        np.savez(output, **payload)
        return output

    @classmethod
    def load_npz(cls, path: str | Path) -> RetargetingResult:
        """Load a `.npz` result."""

        data = np.load(path, allow_pickle=True)
        metadata = _load_metadata(data)
        warnings = _load_warnings(data)
        return cls(
            schema_version=int(np.asarray(data["schema_version"]).reshape(())) if "schema_version" in data else 1,
            name=str(data["name"]) if "name" in data else Path(path).stem,
            status=RunStatus(str(data["status"])) if "status" in data else RunStatus.SUCCESS,
            qpos=data["qpos"],
            fps=float(np.asarray(data["fps"]).reshape(())) if "fps" in data else 30.0,
            cost=data.get("cost", None),
            human_joints=data.get("human_joints", None),
            metadata=metadata,
            warnings=warnings,
        )


class EvaluationReport(BaseModel):
    """Evaluation metrics for a result."""

    status: RunStatus = RunStatus.SUCCESS
    source_name: str | None = None
    frame_count: int | None = None
    qpos_dimension: int | None = None
    fps: float | None = None
    task_kind: str | None = None
    robot_name: str | None = None
    motion_name: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_units: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_report(self) -> EvaluationReport:
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("frame_count must be positive when provided")
        if self.qpos_dimension is not None and self.qpos_dimension <= 0:
            raise ValueError("qpos_dimension must be positive when provided")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive when provided")
        return self

    def save_json(self, path: str | Path) -> Path:
        """Save report as JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output


class EvaluationRecord(BaseModel):
    """Recorded outcome for evaluating one retargeting result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_path: Path
    status: RunStatus
    report_path: Path | None = None
    job_id: str | None = None
    source_name: str | None = None
    frame_count: int | None = None
    message: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class EvaluationManifest(BaseModel):
    """Manifest written by batch evaluation runs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    created_at: datetime
    updated_at: datetime
    total: int = 0
    success_count: int = 0
    partial_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    records: tuple[EvaluationRecord, ...] = ()

    @classmethod
    def from_records(cls, records: Iterable[EvaluationRecord]) -> EvaluationManifest:
        """Build a summary manifest from per-result records."""

        ordered_records = tuple(records)
        now = datetime.now(UTC)
        return cls(
            created_at=now,
            updated_at=now,
            total=len(ordered_records),
            success_count=sum(record.status == RunStatus.SUCCESS for record in ordered_records),
            partial_count=sum(record.status == RunStatus.PARTIAL for record in ordered_records),
            skipped_count=sum(record.status == RunStatus.SKIPPED for record in ordered_records),
            failed_count=sum(record.status == RunStatus.FAILED for record in ordered_records),
            records=ordered_records,
        )

    @classmethod
    def load(cls, path: str | Path) -> EvaluationManifest:
        """Load a batch evaluation manifest JSON file."""

        return cls.model_validate_json(Path(path).read_text())

    def save_json(self, path: str | Path) -> Path:
        """Save manifest JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output


def _load_metadata(data: Any) -> dict[str, Any]:
    if "metadata_json" in data:
        loaded = json.loads(str(np.asarray(data["metadata_json"]).reshape(())))
        if not isinstance(loaded, dict):
            raise ValueError("metadata_json must contain a JSON object")
        return loaded
    if "metadata" in data:
        loaded = data["metadata"].item()
        if not isinstance(loaded, dict):
            raise ValueError("metadata must contain a mapping")
        return loaded
    return {}


def _load_warnings(data: Any) -> tuple[str, ...]:
    if "warnings_json" in data:
        loaded = json.loads(str(np.asarray(data["warnings_json"]).reshape(())))
        if not isinstance(loaded, list):
            raise ValueError("warnings_json must contain a JSON list")
        return tuple(str(value) for value in loaded)
    if "warnings" in data:
        return tuple(str(v) for v in data["warnings"])
    return ()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)
