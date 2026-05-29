"""Optimization configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from retarget.core.enums import SolverBackend


class SolverSpec(BaseModel):
    """Solver selection and common options."""

    backend: SolverBackend | str = SolverBackend.AUTO
    max_iterations: int = 10
    trust_radius: float = 0.2
    tolerance: float = 1e-6
    verbose: bool = False

    @property
    def backend_name(self) -> str:
        """Registry key for the selected solver backend."""

        return self.backend.value if isinstance(self.backend, SolverBackend) else self.backend

    @field_validator("max_iterations")
    @classmethod
    def _positive_iterations(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_iterations must be positive")
        return value

    @field_validator("trust_radius", "tolerance")
    @classmethod
    def _positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return float(value)


class ObjectiveSpec(BaseModel):
    """Declarative objective term configuration."""

    name: str
    weight: float = 1.0
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("weight")
    @classmethod
    def _non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("weight must be non-negative")
        return float(value)


class ConstraintSpec(BaseModel):
    """Declarative constraint term configuration."""

    name: str
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
