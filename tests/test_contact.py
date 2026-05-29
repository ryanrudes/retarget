import numpy as np

from retarget.motion import MotionFormatSpec, MotionSequence, infer_contact_by_velocity


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
