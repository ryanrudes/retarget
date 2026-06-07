"""Optimization profile for Holosoma-compatible climbing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.core.enums import GeometrySource, NominalFallback, NonPenetrationSource
from retarget.optimization import (
    ConstraintConfigUnion,
    DiagonalRegularizationObjectiveConfig,
    FootStickingConstraintConfig,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    SmoothnessObjectiveConfig,
    TrustRegionConstraintConfig,
)

from .vocabulary import HolosomaGeometryName


@dataclass(frozen=True)
class HolosomaClimbOptimizationPolicy:
    """Optimization choices specific to the Holosoma climbing experiment."""

    collision_activation_distance: float = 0.1
    qpos_regularization: tuple[tuple[int, float], ...] = ((19, 0.2), (20, 0.2))
    nominal_qpos_indices: tuple[int, ...] = tuple(range(19))

    def __post_init__(self) -> None:
        if self.collision_activation_distance <= 0.0:
            raise ValueError("collision_activation_distance must be positive")
        if any(index < 0 or weight < 0.0 for index, weight in self.qpos_regularization):
            raise ValueError("qpos regularization indices and weights must be non-negative")
        if len({index for index, _weight in self.qpos_regularization}) != len(self.qpos_regularization):
            raise ValueError("qpos regularization indices must be unique")
        if any(index < 0 for index in self.nominal_qpos_indices):
            raise ValueError("nominal qpos indices must be non-negative")
        if len(set(self.nominal_qpos_indices)) != len(self.nominal_qpos_indices):
            raise ValueError("nominal qpos indices must be unique")


def holosoma_climb_profile(
    *,
    qpos_size: int,
    geometry_pairs: tuple[tuple[str, str], ...] = (),
    policy: HolosomaClimbOptimizationPolicy | None = None,
) -> OptimizationProfile:
    """Return the typed optimization profile matching Holosoma's climbing defaults."""

    policy = policy or HolosomaClimbOptimizationPolicy()
    qpos_weights = np.zeros(int(qpos_size), dtype=np.float64)
    for qpos_idx, weight in policy.qpos_regularization:
        if qpos_idx < qpos_weights.shape[0]:
            qpos_weights[qpos_idx] = float(weight)
    constraints: tuple[ConstraintConfigUnion, ...] = (
        JointLimitsConstraintConfig(),
        TrustRegionConstraintConfig(radius=0.2),
        FootStickingConstraintConfig(tolerance=1e-3),
    )
    if geometry_pairs:
        scene_keywords = (HolosomaGeometryName.MULTI_BOXES.value, HolosomaGeometryName.GROUND.value)
        constraints = (
            *constraints,
            NonPenetrationConstraintConfig(
                sources=(NonPenetrationSource.GEOMETRY,),
                geometry_source=GeometrySource.BACKEND_CANDIDATES,
                tolerance=1e-3,
                scene_clearance=1e-3,
                activation_distance=policy.collision_activation_distance,
                geometry_pairs=geometry_pairs,
                scene_geometry_keywords=scene_keywords,
                excluded_geometry_keyword_pairs=(scene_keywords,),
            ),
        )
    return OptimizationProfile(
        name="holosoma_climb",
        objectives=(
            LaplacianObjectiveConfig(weight=10.0),
            NominalTrackingObjectiveConfig(
                weight=5.0,
                qpos_indices=policy.nominal_qpos_indices,
                fallback=NominalFallback.CURRENT,
            ),
            DiagonalRegularizationObjectiveConfig(weight=1.0, qpos_weights=tuple(float(v) for v in qpos_weights)),
            SmoothnessObjectiveConfig(weight=0.2),
        ),
        constraints=constraints,
        metadata={"source": "holosoma"},
    )
