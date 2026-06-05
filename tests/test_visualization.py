import numpy as np
import pytest
from rich.console import Console

from retarget.core.enums import RunStatus
from retarget.results import RetargetingResult
from retarget.robots import RobotSpec
from retarget.visualization import DryRunVisualizer, build_playback_data
from retarget.visualization.playback import PlaybackObject, PlaybackObjectVisualPart
from retarget.visualization.viewers import (
    _PlaybackGuiHandles,
    _populate_viser_scene,
    _run_viser_playback_loop,
    _try_add_object_mesh,
)


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


def test_build_playback_data_uses_object_visual_parts(tmp_path) -> None:
    mesh_path = tmp_path / "box.obj"
    mesh_path.write_text("")
    result = RetargetingResult(
        name="object_parts",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((1, 7), dtype=np.float64),
        metadata={
            "playback": {
                "object": {
                    "name": "multi_boxes",
                    "sample_points": [[0.0, 0.0, 0.0]],
                    "visual_parts": [
                        {
                            "name": "box1",
                            "mesh_path": str(mesh_path),
                            "asset_scale": [2.0, 3.0, 4.0],
                            "rgba": [0.3, 0.7, 0.9, 0.5],
                        }
                    ],
                }
            }
        },
    )

    playback = build_playback_data(result)

    assert playback.object is not None
    assert len(playback.object.visual_parts) == 1
    assert playback.object.visual_parts[0].mesh_path == mesh_path
    assert playback.object.visual_parts[0].asset_scale == (2.0, 3.0, 4.0)
    assert playback.object.visual_parts[0].rgba == pytest.approx((0.3, 0.7, 0.9, 0.5))


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
                }
            }
        },
    )

    playback = build_playback_data(result)
    frame = playback.frame(1)

    assert playback.robot is not None
    assert playback.robot.name == "humanoid"
    assert playback.robot.link_count == 3
    assert frame.robot_points is not None
    assert np.allclose(frame.robot_points[:, 0], [0.1, 0.1, 0.2])


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
        qpos=np.zeros((2, 14), dtype=np.float64),
        human_joints=np.zeros((2, 2, 3), dtype=np.float64),
        fps=10.0,
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
    server = _FakeViserServer()

    _populate_viser_scene(server, playback)

    assert ("grid", "/retarget/floor") in server.scene.calls
    assert ("frame", "/retarget/object/skateboard") in server.scene.calls
    assert ("box", "/retarget/object/skateboard/body") in server.scene.calls
    assert all(call[1] != "/retarget/robot" for call in server.scene.calls)
    assert all(call[0] != "point_cloud" for call in server.scene.calls)
    assert server.gui.slider_value == 0
    assert server.gui.playing is not None
    assert server.gui.fps_value == 10.0


def test_populate_viser_scene_uses_playback_fps_override():
    result = RetargetingResult(
        name="viser_fps",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((3, 5), dtype=np.float64),
        fps=24.0,
    )
    playback = build_playback_data(result)
    server = _FakeViserServer()

    _populate_viser_scene(server, playback, playback_fps=60.0)

    assert server.gui.fps_value == 60.0


def test_object_mesh_rendering_applies_asset_scale(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
    obj = PlaybackObject(
        name="scaled_box",
        local_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        world_points=np.zeros((1, 1, 3), dtype=np.float64),
        positions=np.zeros((1, 3), dtype=np.float64),
        quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        mesh_path=mesh_path,
        asset_scale=(2.0, 3.0, 4.0),
    )
    scene = _FakeScene()

    _try_add_object_mesh(scene, obj, "scaled_box")

    assert ("mesh", "/retarget/object/scaled_box/mesh") in scene.calls
    assert scene.mesh_bounds is not None
    assert np.allclose(scene.mesh_bounds[0], [-1.0, -1.5, -2.0])
    assert np.allclose(scene.mesh_bounds[1], [1.0, 1.5, 2.0])


def test_object_visual_part_rendering_uses_color_and_scale(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
    obj = PlaybackObject(
        name="multi_boxes",
        local_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        world_points=np.zeros((1, 1, 3), dtype=np.float64),
        positions=np.zeros((1, 3), dtype=np.float64),
        quaternions=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        visual_parts=(
            PlaybackObjectVisualPart(
                name="box1",
                mesh_path=mesh_path,
                asset_scale=(2.0, 3.0, 4.0),
                rgba=(0.3, 0.7, 0.9, 0.5),
            ),
        ),
    )
    scene = _FakeScene()

    _try_add_object_mesh(scene, obj, "multi_boxes")

    assert ("mesh_simple", "/retarget/object/multi_boxes/parts/box1") in scene.calls
    assert scene.mesh_bounds is not None
    assert np.allclose(scene.mesh_bounds[0], [-1.0, -1.5, -2.0])
    assert np.allclose(scene.mesh_bounds[1], [1.0, 1.5, 2.0])
    assert scene.mesh_color == (76, 178, 230)
    assert scene.mesh_opacity == pytest.approx(0.5)


def test_run_viser_playback_loop_advances_frames_when_playing(monkeypatch: pytest.MonkeyPatch) -> None:
    result = RetargetingResult(
        name="loop",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((4, 5), dtype=np.float64),
        fps=100.0,
    )
    playback = build_playback_data(result)
    slider = _FakeSlider(0)
    playing = _FakeCheckbox(True)
    gui = _PlaybackGuiHandles(
        frame_slider=slider,
        playing=playing,
        fps=_FakeNumber(100.0),
        update_frame=lambda index: setattr(slider, "last_updated", index),
    )

    def stop_after_one_tick(_duration: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("retarget.visualization.viewers.time.sleep", stop_after_one_tick)

    _run_viser_playback_loop(playback, gui)

    assert slider.value == 1
    assert slider.last_updated == 1


def test_populate_viser_scene_requires_urdf_for_robot_playback():
    result = RetargetingResult(
        name="viser_robot_requires_urdf",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((2, 7), dtype=np.float64),
        robot_link_positions=np.zeros((2, 1, 3), dtype=np.float64),
        fps=10.0,
        metadata={"playback": {"robot": {"name": "humanoid", "link_names": ["pelvis"]}}},
    )
    playback = build_playback_data(result)

    with pytest.raises(RuntimeError, match="URDF-backed"):
        _populate_viser_scene(_FakeViserServer(), playback)


def test_populate_viser_scene_can_show_diagnostics():
    result = RetargetingResult(
        name="viser_diagnostics",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((2, 7), dtype=np.float64),
        human_joints=np.zeros((2, 2, 3), dtype=np.float64),
        fps=10.0,
    )
    playback = build_playback_data(result)
    server = _FakeViserServer()

    _populate_viser_scene(server, playback, show_diagnostics=True)

    assert ("point_cloud", "/retarget/diagnostics/root_path") in server.scene.calls
    assert ("point_cloud", "/retarget/diagnostics/human_points") in server.scene.calls


class _FakeViserServer:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.gui = _FakeGui()


class _FakeScene:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.mesh_bounds: np.ndarray | None = None
        self.mesh_color: tuple[int, int, int] | None = None
        self.mesh_opacity: float | None = None

    def add_grid(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("grid", name))
        return _FakeHandle()

    def add_point_cloud(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("point_cloud", name))
        return _FakeHandle()

    def add_frame(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("frame", name))
        return _FakeHandle()

    def add_box(self, name: str, **_kwargs: object) -> object:
        self.calls.append(("box", name))
        return _FakeHandle()

    def add_mesh_trimesh(self, name: str, mesh: object, **_kwargs: object) -> object:
        self.calls.append(("mesh", name))
        self.mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        return _FakeHandle()

    def add_mesh_simple(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        **kwargs: object,
    ) -> object:
        del faces
        self.calls.append(("mesh_simple", name))
        self.mesh_bounds = np.vstack([vertices.min(axis=0), vertices.max(axis=0)])
        color = kwargs.get("color")
        self.mesh_color = color if isinstance(color, tuple) else None
        opacity = kwargs.get("opacity")
        self.mesh_opacity = float(opacity) if opacity is not None else None
        return _FakeHandle()


class _FakeHandle:
    position: object = None
    wxyz: object = None
    points: object = None
    dimensions: object = None


class _FakeCheckbox:
    def __init__(self, value: bool) -> None:
        self.value = value


class _FakeNumber:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeSlider:
    def __init__(self, value: int) -> None:
        self.value = value
        self.last_updated: int | None = None

    def on_update(self, callback: object) -> None:
        self._callback = callback


class _FakeGui:
    def __init__(self) -> None:
        self.slider_value: int | None = None
        self.playing: _FakeCheckbox | None = None
        self.fps_value: float | None = None

    def add_slider(self, _name: str, **kwargs: object) -> object:
        slider = _FakeSlider(int(kwargs["initial_value"]))
        self.slider_value = slider.value
        return slider

    def add_checkbox(self, _name: str, **kwargs: object) -> _FakeCheckbox:
        playing = _FakeCheckbox(bool(kwargs["initial_value"]))
        self.playing = playing
        return playing

    def add_number(self, _name: str, **kwargs: object) -> _FakeNumber:
        number = _FakeNumber(float(kwargs["initial_value"]))
        self.fps_value = number.value
        return number
