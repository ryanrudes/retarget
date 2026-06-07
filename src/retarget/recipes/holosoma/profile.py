"""Optimization profile for Holosoma-compatible climbing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.core.enums import NominalFallback, NonPenetrationSource
from retarget.optimization import (
    ConstraintConfigUnion,
    DiagonalRegularizationObjectiveConfig,
    FootStickingConstraintConfig,
    GeometryPair,
    JointLimitsConstraintConfig,
    LaplacianObjectiveConfig,
    NominalTrackingObjectiveConfig,
    NonPenetrationConstraintConfig,
    OptimizationProfile,
    SmoothnessObjectiveConfig,
    TrustRegionConstraintConfig,
)


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
    geometry_pairs: tuple[GeometryPair, ...] = (),
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
        constraints = (
            *constraints,
            NonPenetrationConstraintConfig(
                sources=(NonPenetrationSource.GEOMETRY,),
                tolerance=1e-3,
                scene_clearance=1e-3,
                activation_distance=policy.collision_activation_distance,
                geometry_pairs=geometry_pairs,
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
        provenance={"source": "holosoma"},
    )
