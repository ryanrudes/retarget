"""Temporal and spatial registration for native recordings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from retarget.capture.timeline import ClockTransform, SampleTimeline
from retarget.core.array import FloatArray


@dataclass(frozen=True)
class AlignmentReport:
    """Quality report for one temporal or spatial registration."""

    strategy: str
    accepted: bool
    score: float
    sample_count: int
    overlap_s: float | None = None
    clock_transform: ClockTransform | None = None
    rms_error: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class AlignmentError(ValueError):
    """Registration failure carrying the rejected quality report."""

    def __init__(self, message: str, report: AlignmentReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class TemporalRegistrationConfig:
    """Quality and search policy for offset registration."""

    max_abs_offset_s: float
    min_overlap_s: float = 0.5
    minimum_score: float = 0.2
    coarse_steps: int = 1001
    activity_weighted: bool = True

    def __post_init__(self) -> None:
        if self.max_abs_offset_s <= 0.0:
            raise ValueError("max_abs_offset_s must be positive")
        if self.min_overlap_s <= 0.0:
            raise ValueError("min_overlap_s must be positive")
        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [-1, 1]")
        if self.coarse_steps < 3:
            raise ValueError("coarse_steps must be at least 3")


def estimate_clock_offset(
    reference_timeline: SampleTimeline,
    reference_signal: FloatArray,
    moving_timeline: SampleTimeline,
    moving_signal: FloatArray,
    config: TemporalRegistrationConfig,
    *,
    target_clock: str = "observation",
) -> AlignmentReport:
    """Estimate an offset mapping ``moving_timeline`` into ``reference_timeline``."""

    reference = _signal(reference_signal, reference_timeline.sample_count, name="reference_signal")
    moving = _signal(moving_signal, moving_timeline.sample_count, name="moving_signal")
    if reference.shape[1] != moving.shape[1]:
        raise ValueError("alignment signals must have the same component count")

    def score(offset_s: float) -> tuple[float, int, float]:
        shifted = moving_timeline.timestamps + offset_s
        start = max(reference_timeline.start_s, float(shifted[0]))
        end = min(reference_timeline.end_s, float(shifted[-1]))
        overlap = end - start
        if overlap < config.min_overlap_s:
            return float("-inf"), 0, overlap
        dt = min(
            _median_dt(reference_timeline.timestamps),
            _median_dt(moving_timeline.timestamps),
        )
        times = np.arange(start, end + 0.5 * dt, dt, dtype=np.float64)
        ref = interp1d(
            reference_timeline.timestamps,
            reference,
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )(times)
        mov = interp1d(
            shifted,
            moving,
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )(times)
        finite = np.isfinite(ref).all(axis=1) & np.isfinite(mov).all(axis=1)
        if np.count_nonzero(finite) < 3:
            return float("-inf"), int(np.count_nonzero(finite)), overlap
        return _correlation(ref[finite], mov[finite], activity_weighted=config.activity_weighted), int(
            np.count_nonzero(finite)
        ), overlap

    offsets = np.linspace(-config.max_abs_offset_s, config.max_abs_offset_s, config.coarse_steps)
    scores = np.asarray([score(float(offset))[0] for offset in offsets], dtype=np.float64)
    if not np.isfinite(scores).any():
        report = AlignmentReport(
            strategy="signal_offset",
            accepted=False,
            score=float("-inf"),
            sample_count=0,
            diagnostics={"reason": "no valid overlap"},
        )
        raise AlignmentError("temporal registration has no valid overlap", report)
    best_index = int(np.nanargmax(scores))
    spacing = float(offsets[1] - offsets[0])
    lower = max(-config.max_abs_offset_s, float(offsets[best_index] - spacing))
    upper = min(config.max_abs_offset_s, float(offsets[best_index] + spacing))
    refined = minimize_scalar(lambda value: -score(float(value))[0], bounds=(lower, upper), method="bounded")
    best_offset = float(refined.x) if refined.success else float(offsets[best_index])
    best_score, sample_count, overlap_s = score(best_offset)
    transform = ClockTransform(
        offset_s=best_offset,
        source_clock=moving_timeline.clock,
        target_clock=target_clock,
    )
    accepted = bool(np.isfinite(best_score) and best_score >= config.minimum_score)
    report = AlignmentReport(
        strategy="signal_offset",
        accepted=accepted,
        score=float(best_score),
        sample_count=sample_count,
        overlap_s=float(overlap_s),
        clock_transform=transform,
        diagnostics={
            "search_bounds_s": (-config.max_abs_offset_s, config.max_abs_offset_s),
            "coarse_best_offset_s": float(offsets[best_index]),
        },
    )
    if not accepted:
        raise AlignmentError(
            f"temporal registration score {best_score:.3f} is below {config.minimum_score:.3f}",
            report,
        )
    return report


@dataclass(frozen=True)
class RigidTransform:
    """Rigid point transform ``target = source @ rotation.T + translation``."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rigid transform requires rotation (3, 3) and translation (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("rotation must be right-handed")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points: FloatArray) -> FloatArray:
        """Transform points with trailing dimension 3."""

        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3")
        return np.asarray(values @ self.rotation.T + self.translation, dtype=np.float64)


def register_rigid_points(
    source: FloatArray,
    target: FloatArray,
    *,
    maximum_rms_error: float | None = None,
) -> tuple[RigidTransform, AlignmentReport]:
    """Estimate a Kabsch rigid transform from corresponding finite points."""

    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source and target must both have shape (points, 3)")
    finite = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
    source_points = source_points[finite]
    target_points = target_points[finite]
    if source_points.shape[0] < 3:
        report = AlignmentReport(
            strategy="rigid_point_registration",
            accepted=False,
            score=0.0,
            sample_count=int(source_points.shape[0]),
            diagnostics={"reason": "fewer than three finite correspondences"},
        )
        raise AlignmentError("rigid registration requires at least three finite correspondences", report)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = RigidTransform(rotation=rotation, translation=translation)
    residuals = transform.apply(source_points) - target_points
    rms_error = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    accepted = maximum_rms_error is None or rms_error <= maximum_rms_error
    report = AlignmentReport(
        strategy="rigid_point_registration",
        accepted=accepted,
        score=1.0 / (1.0 + rms_error),
        sample_count=int(source_points.shape[0]),
        rms_error=rms_error,
    )
    if not accepted:
        raise AlignmentError(
            f"rigid registration RMS error {rms_error:.6f} exceeds {maximum_rms_error:.6f}",
            report,
        )
    return transform, report


def _signal(value: FloatArray, sample_count: int, *, name: str) -> FloatArray:
    signal = np.asarray(value, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2 or signal.shape[0] != sample_count:
        raise ValueError(f"{name} must have shape (timeline samples, components)")
    return signal


def _median_dt(timestamps: FloatArray) -> float:
    if timestamps.size < 2:
        raise ValueError("alignment timelines need at least two samples")
    return float(np.median(np.diff(timestamps)))


def _correlation(reference: FloatArray, moving: FloatArray, *, activity_weighted: bool) -> float:
    if activity_weighted:
        activity = np.minimum(np.linalg.norm(reference, axis=1), np.linalg.norm(moving, axis=1))
        floor = float(np.quantile(activity, 0.12))
        weights = np.clip(activity - floor, 0.0, None)
    else:
        weights = np.ones(reference.shape[0], dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        return float("-inf")
    scores: list[float] = []
    for component in range(reference.shape[1]):
        left = reference[:, component]
        right = moving[:, component]
        left_centered = left - float(np.sum(weights * left) / weight_sum)
        right_centered = right - float(np.sum(weights * right) / weight_sum)
        denominator = float(
            np.sqrt(
                np.sum(weights * left_centered * left_centered)
                * np.sum(weights * right_centered * right_centered)
            )
        )
        if denominator <= 1e-18:
            return float("-inf")
        scores.append(float(np.sum(weights * left_centered * right_centered) / denominator))
    return float(np.mean(scores))
