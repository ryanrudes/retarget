from pathlib import Path

import numpy as np
import pytest

from retarget.core.enums import FrameConvention, MotionJoint, QuaternionOrder
from retarget.core.pose import PoseSequence
from retarget.motion import MotionFormatSpec, MotionSequence, load_motion, motion_formats


class UnitMotionJoint(MotionJoint):
    ROOT = "root"
    LEFT_TOE = "left_toe"
    RIGHT_TOE = "right_toe"


def test_json_motion_loader():
    motion = load_motion(Path("tests/fixtures/minimal_motion.json"), "minimal")
    assert motion.frame_count == 3
    assert motion.joint_count == 11
    assert motion.source_height_m == 1.7
    assert "height_m" not in motion.metadata


def test_json_motion_loader_reads_root_poses_and_converts_frame(tmp_path):
    motion_formats.register(
        "unit_test_root_y_up",
        MotionFormatSpec(
            name="unit_test_root_y_up",
            joint_vocabulary=UnitMotionJoint,
            root_joint=UnitMotionJoint.ROOT,
            quaternion_order=QuaternionOrder.XYZW,
            frame_convention=FrameConvention.Y_UP_RIGHT_HANDED,
        ),
        replace=True,
    )
    path = tmp_path / "motion.json"
    path.write_text(
        """
{
  "name": "root_pose_motion",
  "joint_names": ["root", "left_toe", "right_toe"],
  "joint_positions": [
    [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  ],
  "root_positions": [[1.0, 2.0, 3.0]],
  "root_quaternions": [[0.0, 0.0, 0.0, 1.0]]
}
""".strip()
    )

    motion = load_motion(path, "unit_test_root_y_up")

    assert motion.root_poses is not None
    assert motion.root_poses.frame == FrameConvention.Z_UP_RIGHT_HANDED
    assert np.allclose(motion.root_poses.positions, [[1.0, -3.0, 2.0]])
    assert np.allclose(motion.root_poses.quaternions(QuaternionOrder.WXYZ), [[1.0, 0.0, 0.0, 0.0]])


def test_npz_motion_loader_reads_root_poses(tmp_path):
    spec = motion_formats.get("minimal")
    path = tmp_path / "motion.npz"
    positions = np.zeros((2, len(spec.joint_names), 3), dtype=np.float64)
    np.savez(
        path,
        joint_positions=positions,
        joint_names=np.asarray(spec.joint_names),
        fps=np.asarray(20.0),
        root_positions=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        root_quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        root_quaternion_order=np.asarray("wxyz"),
    )

    motion = load_motion(path, "minimal")

    assert motion.root_poses is not None
    assert motion.root_poses.fps == 20.0
    assert np.allclose(motion.root_poses.positions[:, 0], [1.0, 2.0])


def test_npz_motion_loader_rejects_legacy_link_target_arrays(tmp_path):
    spec = motion_formats.get("minimal")
    path = tmp_path / "motion_with_targets.npz"
    positions = np.zeros((2, len(spec.joint_names), 3), dtype=np.float64)
    np.savez(
        path,
        joint_positions=positions,
        joint_names=np.asarray(spec.joint_names),
        link_target_names=np.asarray(["left_foot", "right_foot"], dtype=object),
        link_target_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]],
            ],
            dtype=np.float64,
        ),
        link_target_weights=np.asarray([[10.0, 1.0], [8.0, 2.0]], dtype=np.float64),
        link_target_masks=np.asarray([[True, False], [True, True]], dtype=bool),
    )

    with pytest.raises(ValueError, match="link_target_"):
        load_motion(path, "minimal")


def test_csv_motion_loader_infers_fps_and_reorders_by_frame(tmp_path):
    spec = motion_formats.get("minimal")
    path = tmp_path / "motion.csv"
    columns = ["frame", "time_s", "height_m"]
    for joint in spec.joint_names:
        columns.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    rows = []
    for frame, time_s, pelvis_x in ((1, 0.5, 2.0), (0, 0.0, 1.0)):
        row = {column: "0.0" for column in columns}
        row["frame"] = str(frame)
        row["time_s"] = str(time_s)
        row["height_m"] = "1.8"
        row["Pelvis_x"] = str(pelvis_x)
        rows.append(row)
    path.write_text(",".join(columns) + "\n" + "\n".join(",".join(row[column] for column in columns) for row in rows))

    motion = load_motion(path, "minimal")

    assert motion.frame_count == 2
    assert motion.joint_names == spec.joint_names
    assert motion.source_height_m == 1.8
    assert "height_m" not in motion.metadata
    assert motion.fps == 2.0
    assert motion.joint("Pelvis")[0, 0] == 1.0
    assert motion.joint("Pelvis")[1, 0] == 2.0


def test_csv_motion_loader_reads_root_pose_columns(tmp_path):
    spec = motion_formats.get("minimal")
    path = tmp_path / "motion.csv"
    columns = [
        "frame",
        "root_position_x",
        "root_position_y",
        "root_position_z",
        "root_quaternion_w",
        "root_quaternion_x",
        "root_quaternion_y",
        "root_quaternion_z",
    ]
    for joint in spec.joint_names:
        columns.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    rows = []
    for frame, root_x in ((0, "1.0"), (1, "2.0")):
        row = {column: "0.0" for column in columns}
        row["frame"] = str(frame)
        row["root_position_x"] = root_x
        row["root_quaternion_w"] = "1.0"
        rows.append(row)
    path.write_text(",".join(columns) + "\n" + "\n".join(",".join(row[column] for column in columns) for row in rows))

    motion = load_motion(path, "minimal")

    assert motion.root_poses is not None
    assert np.allclose(motion.root_poses.positions[:, 0], [1.0, 2.0])
    assert np.allclose(motion.root_poses.quaternions(), [[1.0, 0.0, 0.0, 0.0]] * 2)


def test_csv_motion_loader_reports_missing_coordinate(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("frame,Pelvis_x\n0,0.0\n")

    with pytest.raises(KeyError, match="Missing CSV column"):
        load_motion(path, "minimal")


def test_load_motion_converts_registered_format_frame_to_internal_z_up(tmp_path):
    motion_formats.register(
        "unit_test_y_up",
        MotionFormatSpec(
            name="unit_test_y_up",
            joint_vocabulary=UnitMotionJoint,
            root_joint=UnitMotionJoint.ROOT,
            frame_convention=FrameConvention.Y_UP_RIGHT_HANDED,
        ),
        replace=True,
    )
    path = tmp_path / "motion.json"
    path.write_text(
        """
{
  "name": "y_up_motion",
  "joint_names": ["root", "left_toe", "right_toe"],
  "joint_positions": [
    [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  ]
}
""".strip()
    )

    motion = load_motion(path, "unit_test_y_up")

    assert motion.frame == FrameConvention.Z_UP_RIGHT_HANDED
    assert motion.metadata["frame_converted_from"] == "y_up_right_handed"
    assert np.allclose(motion.joint("root"), [[1.0, -3.0, 2.0]])


def test_motion_sequence_resampled_interpolates_joint_positions():
    motion = MotionSequence(
        name="linear",
        joint_names=("root", "hand"),
        joint_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        fps=1.0,
        source_height_m=1.8,
        metadata={"source": "unit_test"},
    )

    resampled = motion.resampled(2.0)

    assert resampled.frame_count == 3
    assert resampled.fps == 2.0
    assert resampled.source_height_m == 1.8
    assert resampled.metadata["source"] == "unit_test"
    assert resampled.metadata["resampled_from_fps"] == 1.0
    assert np.allclose(resampled.joint("root")[:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(resampled.joint("hand")[:, 1], [2.0, 3.0, 4.0])


def test_motion_sequence_rejects_behavioral_metadata():
    with pytest.raises(ValueError, match="provenance-only"):
        MotionSequence(
            name="bad_metadata",
            joint_names=("root",),
            joint_positions=np.zeros((1, 1, 3)),
            metadata={"source_height_m": 1.8},
        )


def test_motion_sequence_duration_matches_endpoint_time_grid():
    motion = MotionSequence.zeros("duration", ("root",), frame_count=3, fps=30.0)
    single_frame = MotionSequence.zeros("single", ("root",), frame_count=1, fps=30.0)

    assert np.isclose(motion.duration_s, 2.0 / 30.0)
    assert single_frame.duration_s == 0.0


def test_motion_sequence_scaled_also_scales_root_pose_translations():
    motion = MotionSequence(
        name="root_pose_scale",
        joint_names=("root",),
        joint_positions=np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64),
        root_poses=PoseSequence.from_arrays([[2.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]),
    )

    scaled = motion.scaled(0.5)

    assert scaled.root_poses is not None
    assert np.allclose(scaled.joint("root"), [[0.5, 0.0, 0.0]])
    assert np.allclose(scaled.root_poses.positions, [[1.0, 0.0, 0.0]])
