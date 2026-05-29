"""Typed export specifications and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retarget.core.protocols import KinematicsBackend


class ExportSpec(BaseModel):
    """Configuration for exporting a retargeting result."""

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
    """Summary of an export operation."""

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
