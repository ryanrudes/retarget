"""Typed export specifications and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.enums import ExporterKind, ExportFormat
from retarget.core.protocols import KinematicsBackend


class ExportSpec(BaseModel):
    """Configuration for exporting a retargeting result.

    Attributes:
        format_name (str): Registered export format key (default ``"mujoco_npz"``).
        output_path (Path): Destination file path for exported data.
        output_fps (int | None): Optional resample rate before export; uses result fps when omitted.
        kinematics_backend (KinematicsBackend | None): Optional backend for analytic qvel; finite
            differences when omitted.
        provenance (dict[str, Any]): Origin information copied into the export.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    format_name: ExporterKind = ExportFormat.MUJOCO_NPZ
    output_path: Path
    output_fps: int | None = None
    kinematics_backend: KinematicsBackend | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


class ExportResult(BaseModel):
    """Summary of an export operation.

    Attributes:
        format_name (str): Export format key used for the operation.
        path (Path): Written output file path.
        frame_count (int): Number of exported frames.
        fps (float): Exported playback frame rate.
        qpos_dimension (int | None): Exported generalized-position width.
        qvel_dimension (int | None): Exported generalized-velocity width.
        provenance (dict[str, Any]): Origin information for the export.
    """

    format_name: ExporterKind
    path: Path
    frame_count: int
    fps: float
    qpos_dimension: int | None = None
    qvel_dimension: int | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


__all__ = ["ExportResult", "ExportSpec"]
