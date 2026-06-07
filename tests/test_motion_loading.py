import json

import numpy as np
import pytest

from retarget.capture import SampleTimeline
from retarget.core.enums import FrameConvention, MotionFormat, QuaternionOrder
from retarget.core.pose import PoseSequence
from retarget.motion import MinimalMotionJoint, load_motion, motion_formats
from retarget.motion.spec import MotionSequence
from tests.typed_fixtures import (
    FixtureMotionJoint,
    SameValueMotionJoint,
    fixture_motion,
)


def test_json_loader_returns_typed_motion_sequence():
    motion = load_motion("tests/fixtures/minimal_motion.json", MotionFormat.MINIMAL)

    assert motion.joints == tuple(MinimalMotionJoint)
    assert motion.root_joint is MinimalMotionJoint.PELVIS
    assert motion.timeline.sample_count == 3
    assert motion.frame is FrameConvention.Z_UP_RIGHT_HANDED


def test_programmatic_loader_rejects_raw_registry_string():
    with pytest.raises(TypeError):
        load_motion("tests/fixtures/minimal_motion.json", "minimal")  # type: ignore[arg-type]


def test_json_external_joint_strings_reconstruct_declared_vocabulary(tmp_path):
    spec = motion_formats.get(MotionFormat.MINIMAL)
    path = tmp_path / "motion.json"
    path.write_text(
        json.dumps(
            {
                "joints": [joint.value for joint in spec.joints],
                "joint_positions": np.zeros((2, len(spec.joints), 3)).tolist(),
                "fps": 20.0,
            }
        )
    )

    motion = load_motion(path, spec)

    assert all(type(joint) is MinimalMotionJoint for joint in motion.joints)
    assert motion.fps == pytest.approx(20.0)


def test_npz_loader_is_pickle_free_and_rejects_embedded_target_arrays(tmp_path):
    spec = motion_formats.get(MotionFormat.MINIMAL)
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        joint_positions=np.zeros((2, len(spec.joints), 3)),
        joints=np.asarray([joint.value for joint in spec.joints], dtype=np.str_),
        link_target_names=np.asarray(["left_foot"], dtype=np.str_),
    )

    with pytest.raises(ValueError, match="LinkTargetPlan"):
        load_motion(path, spec)


def test_motion_sequence_resampling_preserves_vocabulary_and_provenance():
    motion = fixture_motion()
    resampled = motion.resampled(60.0)

    assert resampled.joints == motion.joints
    assert resampled.joint_vocabulary is FixtureMotionJoint
    assert resampled.frame_count == 5
    assert resampled.provenance["source"] == "unit_test"
    assert resampled.provenance["resampled_from_fps"] == pytest.approx(30.0)


def test_motion_sequence_rejects_equal_joint_from_wrong_vocabulary():
    motion = fixture_motion()
    with pytest.raises(TypeError):
        MotionSequence(
            name=motion.name,
            joint_vocabulary=FixtureMotionJoint,
            joints=(SameValueMotionJoint.ROOT, *motion.joints[1:]),
            root_joint=FixtureMotionJoint.ROOT,
            joint_positions=motion.joint_positions,
            timeline=motion.timeline,
        )


def test_motion_frame_conversion_and_root_pose_share_frame():
    timeline = SampleTimeline.uniform(2, 30.0)
    motion = MotionSequence(
        name="frame",
        joint_vocabulary=FixtureMotionJoint,
        joints=tuple(FixtureMotionJoint),
        root_joint=FixtureMotionJoint.ROOT,
        joint_positions=np.asarray(
            [
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 2.0, 3.0], [0.0, 2.0, 3.0], [0.0, 2.0, 3.0]],
            ]
        ),
        timeline=timeline,
        frame=FrameConvention.Y_UP_RIGHT_HANDED,
        root_poses=PoseSequence.from_arrays(
            np.zeros((2, 3)),
            np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
            fps=30.0,
            quaternion_order=QuaternionOrder.WXYZ,
            frame=FrameConvention.Y_UP_RIGHT_HANDED,
        ),
    )

    converted = motion.to_frame(FrameConvention.Z_UP_RIGHT_HANDED)

    assert converted.frame is FrameConvention.Z_UP_RIGHT_HANDED
    assert converted.root_poses is not None
    assert converted.root_poses.frame is converted.frame


def test_motion_duration_uses_timeline_endpoints():
    motion = fixture_motion(frame_count=3)
    assert motion.duration_s == pytest.approx(2.0 / 30.0)
