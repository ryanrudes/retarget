import numpy as np
from rich.console import Console

from retarget.core.enums import RunStatus
from retarget.results import RetargetingResult
from retarget.robots import RobotSpec
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


def test_build_playback_data_can_use_robot_spec_for_model_metadata(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='fixture'/>")
    robot = RobotSpec(
        name="fixture_robot",
        dof=2,
        height_m=1.0,
        joint_names=("joint_a", "joint_b"),
        link_names=("base", "tool"),
        urdf_path=urdf_path,
    )
    result = RetargetingResult(
        name="robot_model_playback",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((1, 9), dtype=np.float64),
        robot_link_positions=np.zeros((1, 2, 3), dtype=np.float64),
        fps=20.0,
    )

    playback = build_playback_data(result, robot_spec=robot)

    assert playback.robot is not None
    assert playback.robot.name == "fixture_robot"
    assert playback.robot.joint_names == ("joint_a", "joint_b")
    assert playback.robot.joint_start == 7
    assert playback.robot.urdf_path == urdf_path
    assert np.allclose(playback.robot.joint_configuration(np.arange(9, dtype=np.float64)), [7.0, 8.0])


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


def test_populate_viser_scene_uses_model_oriented_scene_methods_by_default():
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

    assert ("grid", "/retarget/floor") in server.scene.calls
    assert ("frame", "/retarget/robot") in server.scene.calls
    assert ("icosphere", "/retarget/robot_primitive/pelvis") in server.scene.calls
    assert ("box", "/retarget/robot_primitive/limb_00") in server.scene.calls
    assert ("frame", "/retarget/object/skateboard") in server.scene.calls
    assert ("box", "/retarget/object/skateboard/body") in server.scene.calls
    assert all(call[0] != "point_cloud" for call in server.scene.calls)
    assert all(call[0] != "line_segments" for call in server.scene.calls)
    assert server.gui.slider_value == 0


def test_populate_viser_scene_can_show_diagnostics():
    result = RetargetingResult(
        name="viser_diagnostics",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((2, 7), dtype=np.float64),
        human_joints=np.zeros((2, 2, 3), dtype=np.float64),
        robot_link_positions=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.1, 0.0, 0.0], [0.1, 0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
        fps=10.0,
        metadata={
            "playback": {
                "robot": {
                    "name": "humanoid",
                    "link_names": ["pelvis", "head"],
                    "edges": [["pelvis", "head"]],
                }
            }
        },
    )
    playback = build_playback_data(result)
    server = _FakeViserServer()

    _populate_viser_scene(server, playback, show_diagnostics=True)

    assert ("point_cloud", "/retarget/diagnostics/root_path") in server.scene.calls
    assert ("point_cloud", "/retarget/diagnostics/human_points") in server.scene.calls
    assert ("point_cloud", "/retarget/diagnostics/robot_links") in server.scene.calls
    assert ("line_segments", "/retarget/diagnostics/robot_segments") in server.scene.calls


class _FakeViserServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.gui = _FakeGui()


class _FakeScene:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_grid(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("grid", name))
        return _FakeHandle()

    def add_point_cloud(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("point_cloud", name))
        return _FakeHandle()

    def add_frame(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("frame", name))
        return _FakeHandle()

    def add_icosphere(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("icosphere", name))
        return _FakeHandle()

    def add_box(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("box", name))
        return _FakeHandle()

    def add_line_segments(self, name: str, **_kwargs: object) -> object:
        colors = np.asarray(_kwargs["colors"])
        points = np.asarray(_kwargs["points"])
        assert points.ndim == 3
        assert colors.shape in {points.shape, (3,)}
        self.calls.append(("line_segments", name))
        return _FakeHandle()


class _FakeHandle:
    position: object = None
    wxyz: object = None
    points: object = None
    dimensions: object = None


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
