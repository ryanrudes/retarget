import numpy as np
import pytest

from retarget.capture import PointTrack, SampleTimeline
from retarget.core.enums import ContactPatch, ContactState, ContactSubject
from retarget.motion import SupportPlane
from retarget.observation import SemanticContactSequence, SemanticContactTrack
from retarget.observation.support import (
    FootSupportClassificationConfig,
    FootSupportStates,
    classify_foot_support,
)


class Subject(ContactSubject):
    LEFT = "left_foot"
    RIGHT = "right_foot"


class State(ContactState):
    AIR = "air"
    GROUND = "ground"
    OBJECT = "object"


class Patch(ContactPatch):
    LEFT_SOLE = "left_sole"
    RIGHT_SOLE = "right_sole"


def test_semantic_contacts_are_target_independent_until_resolved():
    timeline = SampleTimeline.uniform(3, 30.0, clock="observation")
    sequence = SemanticContactSequence(
        timeline=timeline,
        tracks=(
            SemanticContactTrack(
                subject=Subject.LEFT,
                patch=Patch.LEFT_SOLE,
                states=(State.AIR, State.GROUND, State.OBJECT),
                active_states=(State.GROUND, State.OBJECT),
                support_states=(State.GROUND,),
            ),
        ),
        support=SupportPlane(
            normal=np.asarray([0.0, 0.0, 1.0]),
            origin=np.asarray([0.0, 0.0, 0.1]),
        ),
    )

    assert not hasattr(sequence.tracks[0], "link_names")
    plan = sequence.resolve({Subject.LEFT: ("left_toe",)})

    assert plan.frame(0).active_link_names == ()
    assert plan.frame(1).active_link_names == ("left_toe",)
    assert plan.frame(1).support_link_names == ("left_toe",)
    assert plan.frame(2).support_link_names == ()


def test_semantic_contact_resolution_requires_explicit_subject_mapping():
    timeline = SampleTimeline.uniform(2, 30.0)
    sequence = SemanticContactSequence(
        timeline=timeline,
        tracks=(
            SemanticContactTrack(
                subject=Subject.LEFT,
                states=(State.AIR, State.GROUND),
                active_states=(State.GROUND,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="left_foot"):
        sequence.resolve({})


def test_foot_support_classifier_returns_typed_semantic_sequence():
    timeline = SampleTimeline.uniform(4, 10.0, clock="observation")
    board = np.asarray([[0.0, 0.0, 0.1]] * 4)
    left = np.asarray([[0.0, 0.0, 0.11]] * 4)
    right = np.asarray([[0.2, 0.0, 0.0]] * 4)

    result = classify_foot_support(
        timeline=timeline,
        left_foot=PointTrack(role=Subject.LEFT, values=left),
        right_foot=PointTrack(role=Subject.RIGHT, values=right),
        observed_object=PointTrack(role=Subject.LEFT, values=board),
        left_subject=Subject.LEFT,
        right_subject=Subject.RIGHT,
        left_patch=Patch.LEFT_SOLE,
        right_patch=Patch.RIGHT_SOLE,
        states=FootSupportStates(
            air=State.AIR,
            ground=State.GROUND,
            observed_object=State.OBJECT,
        ),
        config=FootSupportClassificationConfig(
            ground_height_percentile=0.0,
            ground_clearance_m=0.03,
            ground_speed_mps=0.05,
            object_horizontal_distance_m=0.1,
            object_height_min_m=0.0,
            object_height_max_m=0.03,
            object_relative_speed_mps=0.05,
        ),
    )

    assert result.tracks[0].states == (State.OBJECT,) * 4
    assert result.tracks[1].states == (State.GROUND,) * 4
