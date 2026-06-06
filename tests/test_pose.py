import numpy as np
import pytest

from retarget.core.enums import FrameConvention, QuaternionOrder
from retarget.core.pose import Pose, PoseSequence, convert_points_frame, reorder_quaternion, reorder_quaternions


def test_quaternion_reorder_round_trip():
    wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    xyzw = reorder_quaternion(wxyz, QuaternionOrder.WXYZ, QuaternionOrder.XYZW)
    assert np.allclose(xyzw, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(reorder_quaternion(xyzw, QuaternionOrder.XYZW, QuaternionOrder.WXYZ), wxyz)


def test_quaternions_reorder_vectorized_last_axis():
    wxyz = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]],
        ],
        dtype=np.float64,
    )

    xyzw = reorder_quaternions(wxyz, QuaternionOrder.WXYZ, QuaternionOrder.XYZW)

    assert xyzw.shape == wxyz.shape
    assert np.allclose(xyzw[..., 0], wxyz[..., 1])
    assert np.allclose(xyzw[..., 1], wxyz[..., 2])
    assert np.allclose(xyzw[..., 2], wxyz[..., 3])
    assert np.allclose(xyzw[..., 3], wxyz[..., 0])
    assert np.allclose(reorder_quaternions(xyzw, QuaternionOrder.XYZW, QuaternionOrder.WXYZ), wxyz)


def test_quaternions_reorder_custom_axis_and_copy_behavior():
    wxyz = np.asarray(
        [
            [[1.0, 0.5], [0.0, 0.1], [0.0, 0.2], [0.0, 0.3]],
            [[0.4, 0.8], [0.5, 0.9], [0.6, 1.0], [0.7, 1.1]],
        ],
        dtype=np.float64,
    )

    xyzw = reorder_quaternions(wxyz, QuaternionOrder.WXYZ, QuaternionOrder.XYZW, axis=1)
    same = reorder_quaternions(wxyz, QuaternionOrder.WXYZ, QuaternionOrder.WXYZ, axis=1)
    same[0, 0, 0] = 99.0

    assert xyzw.shape == wxyz.shape
    assert np.allclose(xyzw[:, 0, :], wxyz[:, 1, :])
    assert np.allclose(xyzw[:, 3, :], wxyz[:, 0, :])
    assert wxyz[0, 0, 0] == 1.0


def test_quaternions_reorder_rejects_non_quaternion_axis():
    with pytest.raises(ValueError, match="axis must have length 4"):
        reorder_quaternions(np.zeros((2, 3)), QuaternionOrder.WXYZ, QuaternionOrder.XYZW)


def test_pose_transform_inverse():
    pose = Pose(translation=[1.0, 2.0, 3.0], quaternion=[1.0, 0.0, 0.0, 0.0])
    point = np.array([[0.5, 0.0, 0.0]])
    world = pose.transform_points(point)
    assert np.allclose(world, [[1.5, 2.0, 3.0]])
    assert np.allclose(pose.inverse_transform_points(world), point)


def test_pose_sequence_arrays():
    seq = PoseSequence.identity(2)
    assert seq.positions.shape == (2, 3)
    assert seq.quaternions().shape == (2, 4)


def test_frame_conversion_round_trip_points_and_pose_sequence():
    points_y_up = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    points_z_up = convert_points_frame(
        points_y_up,
        FrameConvention.Y_UP_RIGHT_HANDED,
        FrameConvention.Z_UP_RIGHT_HANDED,
    )

    assert np.allclose(points_z_up, [[[1.0, -3.0, 2.0]]])
    assert np.allclose(
        convert_points_frame(points_z_up, FrameConvention.Z_UP_RIGHT_HANDED, FrameConvention.Y_UP_RIGHT_HANDED),
        points_y_up,
    )

    seq_y_up = PoseSequence.identity(1, frame=FrameConvention.Y_UP_RIGHT_HANDED)
    seq_z_up = seq_y_up.to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
    assert seq_z_up.frame == FrameConvention.Z_UP_RIGHT_HANDED
    assert np.allclose(seq_z_up.positions, np.zeros((1, 3)))


def test_pose_to_frame_preserves_transform_under_coordinate_conversion():
    pose_y_up = Pose(
        translation=[1.0, 2.0, 3.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        frame=FrameConvention.Y_UP_RIGHT_HANDED,
    )
    local_y_up = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)

    world_y_up = pose_y_up.transform_points(local_y_up)
    pose_z_up = pose_y_up.to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
    local_z_up = convert_points_frame(
        local_y_up,
        FrameConvention.Y_UP_RIGHT_HANDED,
        FrameConvention.Z_UP_RIGHT_HANDED,
    )
    world_z_up = pose_z_up.transform_points(local_z_up)

    assert pose_z_up.frame == FrameConvention.Z_UP_RIGHT_HANDED
    assert np.allclose(
        world_z_up,
        convert_points_frame(world_y_up, FrameConvention.Y_UP_RIGHT_HANDED, FrameConvention.Z_UP_RIGHT_HANDED),
    )


def test_pose_sequence_resampled_interpolates_translation_and_rotation():
    poses = PoseSequence.from_arrays(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        fps=1.0,
    )

    resampled = poses.resampled(2.0)

    assert resampled.frame_count == 3
    assert resampled.fps == 2.0
    assert np.allclose(resampled.positions[:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(resampled.poses[1].rotation().apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
