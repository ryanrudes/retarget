import numpy as np
from rich.console import Console

from retarget.core.enums import RunStatus
from retarget.results import RetargetingResult
from retarget.visualization import DryRunVisualizer, build_playback_data
from retarget.visualization.viewers import _populate_viser_scene


def test_build_playback_data_uses_result_root_pose_and_human_points():
    result = RetargetingResult(
        name="playback",
        status=RunStatus.SUCCESS,
        qpos=np.asarray(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        human_joints=np.zeros((2, 3, 3), dtype=np.float64),
        fps=20.0,
    )

    playback = build_playback_data(result)
    frame = playback.frame(1)

    assert playback.frame_count == 2
    assert np.allclose(playback.root_positions[0], [1.0, 2.0, 3.0])
    assert np.allclose(frame.root_quaternion, [0.0, 1.0, 0.0, 0.0])
    assert frame.human_points is not None
    assert np.isclose(frame.time_s, 0.05)


def test_build_playback_data_uses_object_playback_metadata():
    result = RetargetingResult(
        name="object_playback",
        status=RunStatus.SUCCESS,
        qpos=np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        fps=20.0,
        metadata={
            "playback": {
                "object": {
                    "name": "skateboard",
                    "sample_points": [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
                    "qpos_slice": [7, 14],
                }
            }
        },
    )

    playback = build_playback_data(result)

    assert playback.object is not None
    assert playback.object.name == "skateboard"
    assert playback.object.point_count == 2
    assert np.allclose(playback.object.positions[:, 0], [0.25, 0.50])
    assert np.allclose(playback.frame(1).object_points[:, 0], [0.40, 0.60])


def test_build_playback_data_uses_robot_link_positions():
    result = RetargetingResult(
        name="robot_playback",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((2, 7), dtype=np.float64),
        robot_link_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, -0.8]],
                [[0.1, 0.0, 0.0], [0.1, 0.0, 1.0], [0.2, 0.0, -0.8]],
            ],
            dtype=np.float64,
        ),
        fps=20.0,
        metadata={
            "playback": {
                "robot": {
                    "name": "humanoid",
                    "link_names": ["pelvis", "head", "left_foot"],
                    "edges": [["pelvis", "head"], ["pelvis", "left_foot"]],
                }
            }
        },
    )

    playback = build_playback_data(result)
    frame = playback.frame(1)

    assert playback.robot is not None
    assert playback.robot.name == "humanoid"
    assert playback.robot.link_count == 3
    assert playback.robot.edge_count == 2
    assert frame.robot_points is not None
    assert np.allclose(frame.robot_points[:, 0], [0.1, 0.1, 0.2])
    assert frame.robot_segments is not None
    assert frame.robot_segments.shape == (2, 2, 3)


def test_dry_run_visualizer_prints_playback_summary():
    console = Console(record=True, force_terminal=False, width=100)
    result = RetargetingResult(
        name="summary",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((3, 5), dtype=np.float64),
        fps=30.0,
    )

    DryRunVisualizer(console=console).view(result)

    text = console.export_text()
    assert "summary" in text
    assert "Frames" in text
    assert "3" in text


def test_populate_viser_scene_uses_scene_methods():
    result = RetargetingResult(
        name="viser",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((2, 7), dtype=np.float64),
        human_joints=np.zeros((2, 2, 3), dtype=np.float64),
        robot_link_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.1, 0.0, -0.8]],
                [[0.1, 0.0, 0.0], [0.1, 0.0, 1.0], [0.2, 0.0, -0.8]],
            ],
            dtype=np.float64,
        ),
        fps=10.0,
        metadata={
            "playback": {
                "robot": {
                    "name": "humanoid",
                    "link_names": ["pelvis", "head", "left_foot"],
                    "edges": [["pelvis", "head"], ["pelvis", "left_foot"]],
                },
                "object": {
                    "name": "skateboard",
                    "sample_points": [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
                    "qpos_slice": [7, 14],
                }
            }
        },
    )
    playback = build_playback_data(result)
    server = _FakeViserServer()

    _populate_viser_scene(server, playback)

    assert ("point_cloud", "/retarget/root_path") in server.scene.calls
    assert ("frame", "/retarget/root") in server.scene.calls
    assert ("point_cloud", "/retarget/human_points/frame_0000") in server.scene.calls
    assert ("point_cloud", "/retarget/robot/links/frame_0000") in server.scene.calls
    assert ("line_segments", "/retarget/robot/segments/frame_0000") in server.scene.calls
    assert ("point_cloud", "/retarget/object/skateboard/samples/frame_0000") in server.scene.calls
    assert ("point_cloud", "/retarget/object/skateboard/path") in server.scene.calls
    assert ("frame", "/retarget/object/skateboard") in server.scene.calls
    assert server.gui.slider_value == 0


class _FakeViserServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.gui = _FakeGui()


class _FakeScene:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_point_cloud(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("point_cloud", name))
        return object()

    def add_frame(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("frame", name))
        return object()

    def add_line_segments(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("line_segments", name))
        return object()


class _FakeGui:
    def __init__(self) -> None:
        self.slider_value: int | None = None

    def add_slider(self, _name: str, **kwargs: object) -> object:
        slider = _FakeSlider(int(kwargs["initial_value"]))
        self.slider_value = slider.value
        return slider


class _FakeSlider:
    def __init__(self, value: int) -> None:
        self.value = value
