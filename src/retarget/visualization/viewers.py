"""Visualization adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from retarget.core.enums import VisualizerName
from retarget.results.spec import RetargetingResult
from retarget.robots import RobotSpec
from retarget.visualization.playback import (
    PlaybackData,
    PlaybackFrame,
    PlaybackObject,
    PlaybackRobot,
    build_playback_data,
)
from retarget.visualization.registry import visualizers


class DryRunVisualizer:
    """Print a concise result summary."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def view(self, result: RetargetingResult) -> None:
        playback = build_playback_data(result)
        table = Table(title=result.name)
        table.add_column("Frames", justify="right")
        table.add_column("nq", justify="right")
        table.add_column("fps", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Robot")
        table.add_column("Object")
        table.add_column("Status")
        robot_label = f"{playback.robot.name} ({playback.robot.link_count} links)" if playback.robot is not None else ""
        object_label = (
            f"{playback.object.name} ({playback.object.point_count} pts)" if playback.object is not None else ""
        )
        table.add_row(
            str(playback.frame_count),
            str(playback.qpos.shape[1]),
            f"{playback.fps:g}",
            f"{playback.duration_s:.3f}s",
            robot_label,
            object_label,
            result.status.value,
        )
        self.console.print(table)


class ViserVisualizer:
    """Viser-backed playback visualizer."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        block: bool = True,
        robot_spec: RobotSpec | None = None,
        show_diagnostics: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.block = block
        self.robot_spec = robot_spec
        self.show_diagnostics = show_diagnostics

    def view(self, result: RetargetingResult) -> None:
        try:
            import viser
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install retarget[viz] to use ViserVisualizer") from exc
        playback = build_playback_data(result, robot_spec=self.robot_spec)
        server = viser.ViserServer(host=self.host, port=self.port)
        _populate_viser_scene(server, playback, show_diagnostics=self.show_diagnostics)
        if self.block:  # pragma: no cover - requires interactive optional dependency
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                return


@dataclass
class _RobotSceneHandles:
    base_frame: Any | None = None
    urdf: Any | None = None
    urdf_joint_names: tuple[str, ...] = ()
    primitive_links: tuple[Any, ...] = ()
    primitive_segments: tuple[tuple[Any, int, int], ...] = ()
    diagnostic_points: Any | None = None
    diagnostic_segments: Any | None = None


@dataclass
class _ObjectSceneHandles:
    frame: Any | None = None
    body: Any | None = None
    diagnostic_points: Any | None = None


@dataclass
class _DiagnosticSceneHandles:
    root_path: Any | None = None
    human_points: Any | None = None


def _populate_viser_scene(server: Any, playback: PlaybackData, *, show_diagnostics: bool = False) -> None:
    scene = server.scene
    _add_scene_context(scene)
    frame = playback.frame(0)
    robot_handles = _add_robot_scene(server, playback.robot, frame, show_diagnostics=show_diagnostics)
    object_handles = _add_object_scene(scene, playback.object, frame, show_diagnostics=show_diagnostics)
    diagnostics = _add_diagnostics(scene, playback, frame) if show_diagnostics else _DiagnosticSceneHandles()
    if hasattr(server, "gui"):
        slider = _call_if_present(
            server.gui,
            "add_slider",
            "Frame",
            min=0,
            max=playback.frame_count - 1,
            step=1,
            initial_value=0,
        )
        if slider is not None:
            slider.value = 0
            _attach_slider_callback(
                slider,
                playback,
                robot_handles=robot_handles,
                object_handles=object_handles,
                diagnostics=diagnostics,
            )


def _add_scene_context(scene: Any) -> None:
    _call_if_present(
        scene,
        "add_grid",
        "/retarget/floor",
        width=4.0,
        height=4.0,
        plane="xy",
        cell_size=0.2,
        section_size=1.0,
        cell_color=(70, 74, 86),
        section_color=(100, 105, 120),
        position=(0.0, 0.0, 0.0),
    )


def _add_robot_scene(
    server: Any,
    robot: PlaybackRobot | None,
    frame: PlaybackFrame,
    *,
    show_diagnostics: bool,
) -> _RobotSceneHandles:
    if robot is None or frame.robot_points is None:
        return _RobotSceneHandles()
    scene = server.scene
    base_frame = _call_if_present(
        scene,
        "add_frame",
        "/retarget/robot",
        position=frame.root_position,
        wxyz=frame.root_quaternion,
        show_axes=False,
    )
    urdf = _try_add_urdf_robot(server, robot)
    if urdf is not None:
        _update_urdf_robot(robot, urdf, frame)
        return _RobotSceneHandles(
            base_frame=base_frame,
            urdf=urdf,
            urdf_joint_names=tuple(str(name) for name in urdf.get_actuated_joint_names()),
        )

    primitive_links = tuple(
        _call_if_present(
            scene,
            "add_icosphere",
            f"/retarget/robot_primitive/{_scene_name_component(name)}",
            radius=_link_radius(name, robot),
            color=(205, 209, 218),
            position=position,
            subdivisions=2,
        )
        for name, position in zip(robot.link_names, frame.robot_points, strict=True)
    )
    primitive_segments: list[tuple[Any, int, int]] = []
    if frame.robot_segments is not None:
        for edge_index, (first, second) in enumerate(robot.edges):
            first_point = frame.robot_points[first]
            second_point = frame.robot_points[second]
            midpoint, wxyz, length = _segment_pose(first_point, second_point)
            handle = _call_if_present(
                scene,
                "add_box",
                f"/retarget/robot_primitive/limb_{edge_index:02d}",
                color=(174, 181, 195),
                dimensions=(0.045, 0.045, max(length, 0.01)),
                position=midpoint,
                wxyz=wxyz,
                material="standard",
                flat_shading=False,
            )
            primitive_segments.append((handle, first, second))
    handles = _RobotSceneHandles(
        base_frame=base_frame,
        primitive_links=primitive_links,
        primitive_segments=tuple(primitive_segments),
    )
    if show_diagnostics:
        handles.diagnostic_points = _add_robot_point_diagnostics(scene, frame)
        handles.diagnostic_segments = _add_robot_segment_diagnostics(scene, frame)
    return handles


def _try_add_urdf_robot(server: Any, robot: PlaybackRobot) -> Any | None:
    if robot.urdf_path is None:
        return None
    if not robot.urdf_path.exists():
        return None
    try:
        from viser.extras import ViserUrdf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install retarget[viz] to render URDF-backed robots with Viser") from exc
    try:
        return ViserUrdf(
            server,
            robot.urdf_path,
            root_node_name="/retarget/robot",
            mesh_color_override=None,
            load_meshes=True,
            load_collision_meshes=False,
        )
    except Exception as exc:  # pragma: no cover - depends on user URDF assets
        raise RuntimeError(f"Could not load URDF robot model from {robot.urdf_path}") from exc


def _update_urdf_robot(robot: PlaybackRobot, urdf: Any, frame: PlaybackFrame) -> None:
    urdf_joint_names = tuple(str(name) for name in urdf.get_actuated_joint_names())
    robot_values = robot.joint_configuration(frame.qpos)
    robot_by_name = dict(zip(robot.joint_names, robot_values, strict=False))
    configuration = np.asarray([robot_by_name.get(name, 0.0) for name in urdf_joint_names], dtype=np.float64)
    urdf.update_cfg(configuration)


def _add_object_scene(
    scene: Any,
    obj: PlaybackObject | None,
    frame: PlaybackFrame,
    *,
    show_diagnostics: bool,
) -> _ObjectSceneHandles:
    if obj is None:
        return _ObjectSceneHandles()
    object_name = _scene_name_component(obj.name)
    object_frame = _call_if_present(
        scene,
        "add_frame",
        f"/retarget/object/{object_name}",
        position=obj.positions[0],
        wxyz=obj.quaternions[0],
        show_axes=False,
    )
    body = _try_add_object_mesh(scene, obj, object_name)
    if body is None and obj.local_points.size:
        local_center, dimensions = _object_box_from_points(obj.local_points)
        body = _call_if_present(
            scene,
            "add_box",
            f"/retarget/object/{object_name}/body",
            color=(72, 76, 88),
            dimensions=dimensions,
            position=local_center,
            material="standard",
            flat_shading=False,
        )
    diagnostic_points = None
    if show_diagnostics and frame.object_points is not None:
        diagnostic_points = _add_object_point_diagnostics(scene, obj, object_name, frame)
    return _ObjectSceneHandles(frame=object_frame, body=body, diagnostic_points=diagnostic_points)


def _try_add_object_mesh(scene: Any, obj: PlaybackObject, object_name: str) -> Any | None:
    if obj.mesh_path is None or not obj.mesh_path.exists():
        return None
    add_mesh = getattr(scene, "add_mesh_trimesh", None)
    if add_mesh is None:
        return None
    try:
        import trimesh
    except ImportError:  # pragma: no cover - optional dependency
        return None
    loaded = trimesh.load_mesh(str(obj.mesh_path), process=False)
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        return None
    return add_mesh(
        f"/retarget/object/{object_name}/mesh",
        loaded,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _add_diagnostics(scene: Any, playback: PlaybackData, frame: PlaybackFrame) -> _DiagnosticSceneHandles:
    root_color = np.tile(np.asarray([[80, 120, 190]], dtype=np.uint8), (playback.frame_count, 1))
    root_path = _call_if_present(
        scene,
        "add_point_cloud",
        "/retarget/diagnostics/root_path",
        points=playback.root_positions,
        colors=root_color,
        point_size=0.018,
    )
    human_points = None
    if frame.human_points is not None:
        human_color = np.tile(np.asarray([[220, 150, 60]], dtype=np.uint8), (frame.human_points.shape[0], 1))
        human_points = _call_if_present(
            scene,
            "add_point_cloud",
            "/retarget/diagnostics/human_points",
            points=frame.human_points,
            colors=human_color,
            point_size=0.015,
        )
    return _DiagnosticSceneHandles(root_path=root_path, human_points=human_points)


def _add_robot_point_diagnostics(scene: Any, frame: PlaybackFrame) -> Any | None:
    if frame.robot_points is None:
        return None
    colors = np.tile(np.asarray([[120, 165, 220]], dtype=np.uint8), (frame.robot_points.shape[0], 1))
    return _call_if_present(
        scene,
        "add_point_cloud",
        "/retarget/diagnostics/robot_links",
        points=frame.robot_points,
        colors=colors,
        point_size=0.018,
    )


def _add_robot_segment_diagnostics(scene: Any, frame: PlaybackFrame) -> Any | None:
    if frame.robot_segments is None:
        return None
    return _call_if_present(
        scene,
        "add_line_segments",
        "/retarget/diagnostics/robot_segments",
        points=frame.robot_segments,
        colors=np.asarray([130, 170, 220], dtype=np.uint8),
        line_width=1.0,
    )


def _add_object_point_diagnostics(
    scene: Any,
    obj: PlaybackObject,
    object_name: str,
    frame: PlaybackFrame,
) -> Any | None:
    if frame.object_points is None:
        return None
    colors = np.tile(np.asarray([[90, 160, 150]], dtype=np.uint8), (frame.object_points.shape[0], 1))
    path_colors = np.tile(np.asarray([[120, 110, 145]], dtype=np.uint8), (obj.positions.shape[0], 1))
    _call_if_present(
        scene,
        "add_point_cloud",
        f"/retarget/diagnostics/object/{object_name}/path",
        points=obj.positions,
        colors=path_colors,
        point_size=0.012,
    )
    return _call_if_present(
        scene,
        "add_point_cloud",
        f"/retarget/diagnostics/object/{object_name}/samples",
        points=frame.object_points,
        colors=colors,
        point_size=0.016,
    )


def _link_radius(name: str, robot: PlaybackRobot) -> float:
    lower = name.lower()
    if "head" in lower:
        return 0.065
    if "foot" in lower or "toe" in lower or "hand" in lower or "wrist" in lower:
        return 0.04
    if robot.link_count <= 5:
        return 0.045
    return 0.05


def _segment_pose(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    start = np.asarray(first, dtype=np.float64)
    stop = np.asarray(second, dtype=np.float64)
    delta = stop - start
    length = float(np.linalg.norm(delta))
    midpoint = (start + stop) / 2.0
    if length <= 1e-12:
        return midpoint, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), 0.0
    return midpoint, _quaternion_from_z_axis(delta / length), length


def _quaternion_from_z_axis(direction: NDArray[np.float64]) -> NDArray[np.float64]:
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    dot = float(np.clip(np.dot(z_axis, unit), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    axis = np.cross(z_axis, unit)
    scale = np.sqrt((1.0 + dot) * 2.0)
    quaternion = np.asarray([scale * 0.5, axis[0] / scale, axis[1] / scale, axis[2] / scale], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _object_box_from_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2.0
    dimensions = np.maximum(upper - lower, np.asarray([0.02, 0.02, 0.035], dtype=np.float64))
    return center, dimensions


def _call_if_present(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return method(*args, **kwargs)


def _attach_slider_callback(
    slider: Any,
    playback: PlaybackData,
    *,
    robot_handles: _RobotSceneHandles,
    object_handles: _ObjectSceneHandles,
    diagnostics: _DiagnosticSceneHandles,
) -> None:
    on_update = getattr(slider, "on_update", None)
    if not callable(on_update):
        return

    def update_frame() -> None:
        frame = playback.frame(int(slider.value))
        _update_robot_scene(playback.robot, robot_handles, frame)
        _update_object_scene(playback.object, object_handles, frame)
        _update_diagnostics(diagnostics, frame)

    def _on_slider_update(_event: Any) -> None:
        update_frame()

    on_update(_on_slider_update)


def _update_robot_scene(
    robot: PlaybackRobot | None,
    handles: _RobotSceneHandles,
    frame: PlaybackFrame,
) -> None:
    _set_if_present(handles.base_frame, "position", frame.root_position)
    _set_if_present(handles.base_frame, "wxyz", frame.root_quaternion)
    if robot is None:
        return
    if handles.urdf is not None:
        _update_urdf_robot(robot, handles.urdf, frame)
        return
    if frame.robot_points is not None:
        for handle, position in zip(handles.primitive_links, frame.robot_points, strict=False):
            _set_if_present(handle, "position", position)
    for handle, first, second in handles.primitive_segments:
        if frame.robot_points is None:
            continue
        midpoint, wxyz, length = _segment_pose(frame.robot_points[first], frame.robot_points[second])
        _set_if_present(handle, "position", midpoint)
        _set_if_present(handle, "wxyz", wxyz)
        _set_if_present(handle, "dimensions", (0.045, 0.045, max(length, 0.01)))
    if frame.robot_points is not None:
        _set_if_present(handles.diagnostic_points, "points", frame.robot_points)
    if frame.robot_segments is not None:
        _set_if_present(handles.diagnostic_segments, "points", frame.robot_segments)


def _update_object_scene(
    obj: PlaybackObject | None,
    handles: _ObjectSceneHandles,
    frame: PlaybackFrame,
) -> None:
    if obj is None:
        return
    _set_if_present(handles.frame, "position", obj.positions[frame.index])
    _set_if_present(handles.frame, "wxyz", obj.quaternions[frame.index])
    if frame.object_points is not None:
        _set_if_present(handles.diagnostic_points, "points", frame.object_points)


def _update_diagnostics(handles: _DiagnosticSceneHandles, frame: PlaybackFrame) -> None:
    if frame.human_points is not None:
        _set_if_present(handles.human_points, "points", frame.human_points)


def _set_if_present(target: Any, attribute: str, value: Any) -> None:
    if target is not None and hasattr(target, attribute):
        setattr(target, attribute, value)


def _scene_name_component(value: str) -> str:
    return value.strip().replace("/", "_") or "object"


def view_result(
    result: RetargetingResult,
    *,
    dry_run: bool = True,
    robot_spec: RobotSpec | None = None,
    show_diagnostics: bool = False,
) -> None:
    """View or summarize a result."""

    if dry_run:
        visualizer = visualizers.get(VisualizerName.DRY_RUN)
    else:
        visualizer = ViserVisualizer(robot_spec=robot_spec, show_diagnostics=show_diagnostics)
    visualizer.view(result)


visualizers.register(VisualizerName.DRY_RUN, DryRunVisualizer())
visualizers.register(VisualizerName.VISER, ViserVisualizer())
