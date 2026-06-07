import numpy as np
import pytest

from retarget.motion import LinkTargetPlan, LinkTargetTrack
from tests.typed_fixtures import (
    FixtureRobotLink,
    SameValueRobotLink,
    fixture_targets,
)


def test_target_plan_preserves_typed_links_and_active_samples():
    plan = fixture_targets()
    frame = plan.frame(1)

    assert plan.links == (FixtureRobotLink.LEFT_FOOT, FixtureRobotLink.RIGHT_FOOT)
    assert frame.links == plan.links
    assert frame.positions.shape == (2, 3)
    assert np.allclose(frame.weights, 1.0)


def test_target_plan_rejects_equal_values_from_different_vocabularies():
    positions = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        LinkTargetPlan(
            tracks=(
                LinkTargetTrack(link=FixtureRobotLink.PELVIS, positions=positions),
                LinkTargetTrack(link=SameValueRobotLink.PELVIS, positions=positions),
            )
        )


def test_target_plan_dense_constructor_and_resampling_keep_enum_identity():
    plan = LinkTargetPlan.from_arrays(
        links=(FixtureRobotLink.LEFT_FOOT, FixtureRobotLink.RIGHT_FOOT),
        positions=np.zeros((3, 2, 3), dtype=np.float64),
    )
    resampled = plan.resampled(30.0, 60.0)

    assert resampled.frame_count == 5
    assert resampled.links == plan.links


def test_inactive_target_is_absent_from_frame():
    track = LinkTargetTrack(
        link=FixtureRobotLink.LEFT_FOOT,
        positions=np.zeros((2, 3), dtype=np.float64),
        active_mask=np.asarray([True, False]),
    )
    plan = LinkTargetPlan(tracks=(track,))

    assert plan.frame(0).links == (FixtureRobotLink.LEFT_FOOT,)
    assert plan.frame(1).links == ()
