"""Typed export specifications and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.protocols import KinematicsBackend


class ExportSpec(BaseModel):
    """Configuration for exporting a retargeting result.

    Attributes:
        format_name (str): Registered export format key (default ``"mujoco_npz"``).
        output_path (Path): Destination file path for exported data.
        output_fps (int | None): Optional resample rate before export; uses result fps when omitted.
        kinematics_backend (KinematicsBackend | None): Optional backend for analytic qvel; finite
            differences when omitted.
        metadata (dict[str, Any]): Extra key-value metadata merged into export output.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    format_name: str = "mujoco_npz"
    output_path: Path
    output_fps: int | None = None
    kinematics_backend: KinematicsBackend | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

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
        metadata (dict[str, Any]): Summary metadata (dimensions, source name, etc.).
    """

    format_name: str
    path: Path
    frame_count: int
    fps: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)


__all__ = ["ExportResult", "ExportSpec"]
