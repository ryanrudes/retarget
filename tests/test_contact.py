import pytest

from retarget.motion import ContactPlan, ContactTrack
from tests.typed_fixtures import (
    FixtureContactPatch,
    FixtureContactState,
    FixtureContactSubject,
    FixtureRobotLink,
    SameValueContactSubject,
    SameValueRobotLink,
    fixture_contacts,
)


def test_contact_plan_preserves_typed_subject_state_patch_and_links():
    plan = fixture_contacts()
    frame = plan.frame(0)

    assert plan.tracks[0].subject is FixtureContactSubject.LEFT_FOOT
    assert plan.tracks[0].patch is FixtureContactPatch.LEFT_SOLE
    assert plan.tracks[0].states[0] is FixtureContactState.SUPPORT
    assert frame.active_links == (FixtureRobotLink.LEFT_FOOT,)
    assert frame.support_links == (FixtureRobotLink.LEFT_FOOT,)


def test_contact_plan_rejects_equal_values_from_different_vocabularies():
    with pytest.raises(TypeError):
        ContactPlan(
            tracks=(
                ContactTrack(
                    subject=FixtureContactSubject.LEFT_FOOT,
                    states=(FixtureContactState.SUPPORT,),
                    links=(FixtureRobotLink.LEFT_FOOT,),
                ),
                ContactTrack(
                    subject=SameValueContactSubject.LEFT_FOOT,
                    states=(FixtureContactState.SUPPORT,),
                    links=(FixtureRobotLink.RIGHT_FOOT,),
                ),
            )
        )

    with pytest.raises(TypeError):
        ContactPlan(
            tracks=(
                ContactTrack(
                    subject=FixtureContactSubject.LEFT_FOOT,
                    states=(FixtureContactState.SUPPORT,),
                    links=(FixtureRobotLink.LEFT_FOOT,),
                ),
                ContactTrack(
                    subject=FixtureContactSubject.RIGHT_FOOT,
                    states=(FixtureContactState.SUPPORT,),
                    links=(SameValueRobotLink.PELVIS,),
                ),
            )
        )


def test_contact_resampling_uses_nearest_state_without_string_encoding():
    plan = fixture_contacts(frame_count=3)
    resampled = plan.resampled(30.0, 60.0)

    assert resampled.frame_count == 5
    assert all(isinstance(state, FixtureContactState) for track in resampled.tracks for state in track.states)
