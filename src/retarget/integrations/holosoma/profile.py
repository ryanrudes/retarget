"""Optimization profile for Holosoma-compatible climbing."""

from __future__ import annotations

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

from .vocabulary import (
    COLLISION_DETECTION_THRESHOLD,
    G1_MANUAL_QPOS_COSTS,
    G1_NOMINAL_TRACKING_QPOS_INDICES,
    HolosomaGeometryName,
)


def holosoma_climb_profile(
    *,
    qpos_size: int,
    geometry_pairs: tuple[tuple[str, str], ...] = (),
) -> OptimizationProfile:
    """Return the typed optimization profile matching Holosoma's climbing defaults."""

    qpos_weights = np.zeros(int(qpos_size), dtype=np.float64)
    for qpos_idx, weight in G1_MANUAL_QPOS_COSTS.items():
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
                activation_distance=COLLISION_DETECTION_THRESHOLD,
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
                qpos_indices=G1_NOMINAL_TRACKING_QPOS_INDICES,
                fallback=NominalFallback.CURRENT,
            ),
            DiagonalRegularizationObjectiveConfig(weight=1.0, qpos_weights=tuple(float(v) for v in qpos_weights)),
            SmoothnessObjectiveConfig(weight=0.2),
        ),
        constraints=constraints,
        metadata={"source": "holosoma"},
    )
