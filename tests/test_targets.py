import numpy as np
import pytest

from retarget.core.enums import FrameConvention
from retarget.motion import LinkTargetPlan, LinkTargetTrack, NominalQposPlan


def test_link_target_plan_frame_filters_inactive_and_nan_samples() -> None:
    plan = LinkTargetPlan.from_arrays(
        link_names=("left_toe", "right_toe", "torso"),
        positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        weights=np.asarray([[4.0, 3.0, 0.0], [4.0, 3.0, 2.0]], dtype=np.float64),
        active_mask=np.asarray([[True, True, True], [False, True, True]], dtype=bool),
        provenance={"source": "unit"},
    )

    frame0 = plan.frame(0)
    frame1 = plan.frame(1)

    assert plan.link_names == ("left_toe", "right_toe", "torso")
    assert frame0.link_names == ("left_toe",)
    assert np.allclose(frame0.positions, [[0.0, 0.0, 0.0]])
    assert np.allclose(frame0.weights, [4.0])
    assert frame1.link_names == ("right_toe", "torso")
    assert plan.provenance["source"] == "unit"


def test_link_target_plan_validation_rejects_bad_shapes_and_duplicates() -> None:
    with pytest.raises(ValueError, match="shape"):
        LinkTargetTrack(link_name="toe", positions=np.zeros((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="unique"):
        LinkTargetPlan(
            tracks=(
                LinkTargetTrack(link_name="toe", positions=np.zeros((2, 3), dtype=np.float64)),
                LinkTargetTrack(link_name="toe", positions=np.zeros((2, 3), dtype=np.float64)),
            )
        )
    with pytest.raises(ValueError, match="weights"):
        LinkTargetPlan.from_arrays(
            link_names=("toe",),
            positions=np.zeros((2, 1, 3), dtype=np.float64),
            weights=np.ones((3, 1), dtype=np.float64),
        )


def test_link_target_plan_scales_resamples_and_converts_frames() -> None:
    plan = LinkTargetPlan.from_arrays(
        link_names=("toe",),
        positions=np.asarray([[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]]], dtype=np.float64),
        weights=np.asarray([[1.0], [2.0]], dtype=np.float64),
    )

    scaled = plan.scaled(2.0)
    resampled = plan.resampled(source_fps=1.0, target_fps=2.0)
    converted = plan.to_frame(FrameConvention.Y_UP_RIGHT_HANDED, FrameConvention.Z_UP_RIGHT_HANDED)

    assert np.allclose(scaled.tracks[0].positions, [[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    assert resampled.frame_count == 3
    assert np.allclose(resampled.tracks[0].positions[:, 0], [1.0, 1.0, 2.0])
    assert np.allclose(converted.tracks[0].positions[0], [1.0, -3.0, 2.0])


def test_nominal_qpos_plan_validates_resamples_and_filters_inactive_values() -> None:
    plan = NominalQposPlan.from_array(
        np.asarray([[0.0, 1.0, np.nan], [0.5, 1.5, 2.5]], dtype=np.float64),
        weights=np.asarray([[1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float64),
        active_mask=np.asarray([[True, True, True], [True, False, True]], dtype=bool),
        provenance={"source": "unit"},
    )

    frame0 = plan.frame(0)
    frame1 = plan.frame(1)
    resampled = plan.resampled(source_fps=1.0, target_fps=2.0)

    assert frame0.active_at(0)
    assert not frame0.active_at(1)
    assert not frame0.active_at(2)
    assert frame1.active_at(2)
    assert not frame1.active_at(1)
    assert resampled.frame_count == 3
    assert np.allclose(resampled.qpos[:, 0], [0.0, 0.0, 0.5])
    assert resampled.provenance["resampled_to_fps"] == 2.0
