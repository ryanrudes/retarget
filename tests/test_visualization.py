import numpy as np
import pytest

from retarget.core.enums import (
    FrameConvention,
    ObjectQposMode,
    ObjectSampleSpace,
    QuaternionOrder,
    RunStatus,
)
from retarget.pipeline import Retargeter
from retarget.results import (
    ResultObjectSpec,
    ResultPlaybackSpec,
    RetargetingResult,
)
from retarget.visualization.playback import build_playback_data
from tests.typed_fixtures import fixture_problem


def test_playback_uses_structured_robot_spec_from_result():
    result = Retargeter().run(fixture_problem())

    playback = build_playback_data(result)

    assert playback.robot is not None
    assert playback.robot.name == "fixture_robot"
    assert playback.robot.link_names == result.playback.robot.link_names
    assert playback.robot.joint_names == result.playback.robot.joint_names
    assert playback.frame_count == result.frame_count


def test_playback_can_override_external_model_paths_with_typed_robot_spec(tmp_path):
    problem = fixture_problem()
    robot = problem.robot.model_copy(
        update={
            "urdf_path": tmp_path / "robot.urdf",
            "mujoco_xml_path": tmp_path / "robot.xml",
        }
    )
    result = Retargeter().run(problem.model_copy(update={"robot": robot}))

    playback = build_playback_data(result, robot_spec=robot)

    assert playback.robot is not None
    assert playback.robot.urdf_path == robot.urdf_path
    assert playback.robot.mujoco_xml_path == robot.mujoco_xml_path


def test_object_playback_uses_typed_object_definition_not_provenance():
    base = Retargeter().run(fixture_problem())
    object_points = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    object_start = base.qpos.shape[1]
    qpos = np.pad(base.qpos, ((0, 0), (0, 7)))
    qpos[:, object_start : object_start + 3] = (1.0, 2.0, 3.0)
    qpos[:, object_start + 3] = 1.0
    result = base.model_copy(
        update={
            "qpos": qpos,
            "playback": ResultPlaybackSpec(
                robot=base.playback.robot,
                object=ResultObjectSpec(
                    name="fixture_object",
                    sample_points=object_points,
                    sample_space=ObjectSampleSpace.OBJECT_LOCAL,
                    qpos_mode=ObjectQposMode.APPENDED,
                    frame_convention=FrameConvention.Z_UP_RIGHT_HANDED,
                    quaternion_order=QuaternionOrder.WXYZ,
                    qpos_slice=(object_start, object_start + 7),
                ),
            ),
            "provenance": {"source": "unit_test"},
        }
    )

    playback = build_playback_data(result)

    assert playback.object is not None
    assert np.allclose(
        playback.object.world_points[0],
        object_points + np.asarray([1.0, 2.0, 3.0]),
    )


def test_playback_rejects_missing_typed_link_name_manifest():
    result = Retargeter().run(fixture_problem())
    broken_robot = result.playback.robot.model_copy(update={"link_names": ()})
    broken = result.model_copy(
        update={"playback": result.playback.model_copy(update={"robot": broken_robot})}
    )

    with pytest.raises(ValueError, match="link names"):
        build_playback_data(broken)


def test_result_with_no_link_positions_has_no_robot_playback():
    result = RetargetingResult(
        name="minimal",
        status=RunStatus.SUCCESS,
        qpos=np.asarray([[0.0, 0.0, 0.0]]),
    )

    assert build_playback_data(result).robot is None
