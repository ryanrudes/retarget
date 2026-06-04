import numpy as np

from retarget.motion import (
    ContactPlan,
    ContactTrack,
    MotionFormatSpec,
    MotionSequence,
    SupportPlane,
    infer_contact_by_velocity,
)


def test_infer_contact_by_velocity_marks_stationary_toe():
    motion = MotionSequence(
        name="contact",
        joint_names=("Pelvis", "L_Toe", "R_Toe"),
        joint_positions=np.asarray(
            [
                [[0.0, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
                [[0.1, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.2, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        fps=30.0,
    )
    motion_format = MotionFormatSpec(
        name="fixture",
        joint_names=motion.joint_names,
        root_joint="Pelvis",
        contact_joints=("L_Toe", "R_Toe"),
    )
    contacts = infer_contact_by_velocity(motion, motion_format, velocity_threshold=0.05)
    assert all(frame["L_Toe"] for frame in contacts)
    assert not any(frame["R_Toe"] for frame in contacts)


def test_motion_format_accepts_no_contact_joints():
    motion = MotionSequence.zeros("no_contact", ("root", "hand"), 2)
    motion_format = MotionFormatSpec(name="no_contact", joint_names=motion.joint_names, root_joint="root")

    contacts = infer_contact_by_velocity(motion, motion_format)

    assert contacts == ({}, {})


def test_infer_contact_supports_more_than_two_contact_joints():
    motion = MotionSequence(
        name="multi_contact",
        joint_names=("root", "left_toe", "right_toe", "left_hand"),
        joint_positions=np.asarray(
            [
                [[0.0, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.3, 0.0, 1.0]],
                [[0.1, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.3, 0.0, 1.0]],
                [[0.2, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [-0.3, 0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
        fps=30.0,
    )
    motion_format = MotionFormatSpec(
        name="multi_contact",
        joint_names=motion.joint_names,
        root_joint="root",
        contact_joints=("left_toe", "right_toe", "left_hand"),
    )

    contacts = infer_contact_by_velocity(motion, motion_format, velocity_threshold=0.05)

    assert all(frame["left_toe"] for frame in contacts)
    assert not any(frame["right_toe"] for frame in contacts)
    assert all(frame["left_hand"] for frame in contacts)


def test_explicit_motion_contacts_override_velocity_inference():
    motion = MotionSequence(
        name="explicit_contact",
        joint_names=("root", "left_toe", "right_toe"),
        joint_positions=np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.1, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        fps=30.0,
        contacts=(
            {"left_toe": True, "right_toe": False, "ignored": True},
            {"left_toe": False, "right_toe": True, "ignored": True},
        ),
    )
    motion_format = MotionFormatSpec(
        name="explicit_contact",
        joint_names=motion.joint_names,
        root_joint="root",
        contact_joints=("left_toe", "right_toe"),
    )

    contacts = infer_contact_by_velocity(motion, motion_format, velocity_threshold=0.05)

    assert contacts == (
        {"left_toe": True, "right_toe": False},
        {"left_toe": False, "right_toe": True},
    )


def test_motion_contacts_resample_with_nearest_neighbor_states():
    motion = MotionSequence(
        name="contact_resample",
        joint_names=("root", "toe"),
        joint_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        fps=1.0,
        contacts={"toe": [True, False]},
    )

    resampled = motion.resampled(2.0)

    assert resampled.contacts == ({"toe": True}, {"toe": True}, {"toe": False})


def test_contact_plan_validates_tracks_and_exposes_frame_view():
    support = SupportPlane(normal=np.array([0.0, 0.0, 2.0]), origin=np.array([0.0, 0.0, 0.1]))
    plan = ContactPlan(
        tracks=(
            ContactTrack("left_foot", np.array([0, 1, 2]), link_names=("left_toe",), labels=("air", "ground", "board")),
            ContactTrack("right_foot", np.array([1, 0, 0]), link_names=("right_toe",)),
        ),
        support=support,
        provenance={"source": "test"},
    )

    frame = plan.frame(1)

    assert plan.frame_count == 3
    assert frame.active_link_names == ("left_toe",)
    assert frame.support_link_names == ("left_toe",)
    assert frame.support is support
    assert frame.as_contact_dict() == {"left_foot": True, "right_foot": False}


def test_contact_plan_resamples_and_scales_support_plane():
    plan = ContactPlan(
        tracks=(ContactTrack("toe", np.array([0, 1]), link_names=("left_toe",)),),
        support=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), origin=np.array([0.0, 0.0, 0.5])),
    )

    resampled = plan.resampled(1.0, 2.0)
    scaled = resampled.scaled(2.0)

    assert resampled.frame_count == 3
    assert resampled.tracks[0].states.tolist() == [0, 0, 1]
    assert np.allclose(scaled.support.origin, [0.0, 0.0, 1.0])


def test_contact_plan_from_binary_contacts_maps_links():
    plan = ContactPlan.from_binary_contacts(
        ({"L_Foot": True, "R_Foot": False}, {"L_Foot": False, "R_Foot": True}),
        link_mapping={"L_Foot": "left_toe", "R_Foot": ("right_toe",)},
        support=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3)),
        provenance={"source": "legacy"},
    )

    assert tuple(track.subject for track in plan.tracks) == ("L_Foot", "R_Foot")
    assert plan.tracks[0].link_names == ("left_toe",)
    assert plan.frame(0).active_link_names == ("left_toe",)
    assert plan.frame(1).active_link_names == ("right_toe",)
