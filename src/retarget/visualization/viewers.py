"""Visualization adapters."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from retarget.core.enums import VisualizerName
from retarget.results.spec import RetargetingResult
from retarget.visualization.playback import PlaybackData, build_playback_data
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

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8080, block: bool = True) -> None:
        self.host = host
        self.port = port
        self.block = block

    def view(self, result: RetargetingResult) -> None:
        try:
            import viser
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install retarget[viz] to use ViserVisualizer") from exc
        playback = build_playback_data(result)
        server = viser.ViserServer(host=self.host, port=self.port)
        _populate_viser_scene(server, playback)
        if self.block:  # pragma: no cover - requires interactive optional dependency
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                return


def _populate_viser_scene(server: Any, playback: PlaybackData) -> None:
    scene = server.scene
    root_color = np.tile(np.asarray([[40, 120, 240]], dtype=np.uint8), (playback.frame_count, 1))
    _call_if_present(
        scene,
        "add_point_cloud",
        "/retarget/root_path",
        points=playback.root_positions,
        colors=root_color,
        point_size=0.025,
    )
    frame = playback.frame(0)
    root_handle = _call_if_present(
        scene,
        "add_frame",
        "/retarget/root",
        position=frame.root_position,
        wxyz=frame.root_quaternion,
        axes_length=0.2,
        axes_radius=0.01,
    )
    human_handle = None
    if frame.human_points is not None:
        human_color = np.tile(np.asarray([[240, 160, 40]], dtype=np.uint8), (frame.human_points.shape[0], 1))
        human_handle = _call_if_present(
            scene,
            "add_point_cloud",
            "/retarget/human_points/frame_0000",
            points=frame.human_points,
            colors=human_color,
            point_size=0.02,
        )
    robot_links_handle = None
    robot_segments_handle = None
    if playback.robot is not None and frame.robot_points is not None:
        robot_color = np.tile(np.asarray([[80, 170, 255]], dtype=np.uint8), (frame.robot_points.shape[0], 1))
        robot_links_handle = _call_if_present(
            scene,
            "add_point_cloud",
            "/retarget/robot/links/frame_0000",
            points=frame.robot_points,
            colors=robot_color,
            point_size=0.035,
        )
        if frame.robot_segments is not None:
            segment_color = np.tile(np.asarray([[110, 210, 255]], dtype=np.uint8), (frame.robot_segments.shape[0], 1))
            robot_segments_handle = _call_if_present(
                scene,
                "add_line_segments",
                "/retarget/robot/segments/frame_0000",
                points=frame.robot_segments,
                colors=segment_color,
                line_width=2.0,
            )
    object_samples_handle = None
    object_frame_handle = None
    if playback.object is not None and frame.object_points is not None:
        object_name = _scene_name_component(playback.object.name)
        object_color = np.tile(np.asarray([[90, 220, 180]], dtype=np.uint8), (frame.object_points.shape[0], 1))
        path_color = np.tile(np.asarray([[180, 140, 255]], dtype=np.uint8), (playback.object.positions.shape[0], 1))
        object_samples_handle = _call_if_present(
            scene,
            "add_point_cloud",
            f"/retarget/object/{object_name}/samples/frame_0000",
            points=frame.object_points,
            colors=object_color,
            point_size=0.025,
        )
        _call_if_present(
            scene,
            "add_point_cloud",
            f"/retarget/object/{object_name}/path",
            points=playback.object.positions,
            colors=path_color,
            point_size=0.018,
        )
        object_frame_handle = _call_if_present(
            scene,
            "add_frame",
            f"/retarget/object/{object_name}",
            position=playback.object.positions[0],
            wxyz=playback.object.quaternions[0],
            axes_length=0.18,
            axes_radius=0.008,
        )
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
                root_handle=root_handle,
                human_handle=human_handle,
                robot_links_handle=robot_links_handle,
                robot_segments_handle=robot_segments_handle,
                object_samples_handle=object_samples_handle,
                object_frame_handle=object_frame_handle,
            )


def _call_if_present(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return method(*args, **kwargs)


def _attach_slider_callback(
    slider: Any,
    playback: PlaybackData,
    *,
    root_handle: Any,
    human_handle: Any,
    robot_links_handle: Any,
    robot_segments_handle: Any,
    object_samples_handle: Any,
    object_frame_handle: Any,
) -> None:
    on_update = getattr(slider, "on_update", None)
    if not callable(on_update):
        return

    def update_frame() -> None:
        frame = playback.frame(int(slider.value))
        _set_if_present(root_handle, "position", frame.root_position)
        _set_if_present(root_handle, "wxyz", frame.root_quaternion)
        if frame.human_points is not None:
            _set_if_present(human_handle, "points", frame.human_points)
        if frame.robot_points is not None:
            _set_if_present(robot_links_handle, "points", frame.robot_points)
        if frame.robot_segments is not None:
            _set_if_present(robot_segments_handle, "points", frame.robot_segments)
        if playback.object is not None:
            _set_if_present(object_frame_handle, "position", playback.object.positions[frame.index])
            _set_if_present(object_frame_handle, "wxyz", playback.object.quaternions[frame.index])
        if frame.object_points is not None:
            _set_if_present(object_samples_handle, "points", frame.object_points)

    def _on_slider_update(_event: Any) -> None:
        update_frame()

    on_update(_on_slider_update)


def _set_if_present(target: Any, attribute: str, value: Any) -> None:
    if target is not None and hasattr(target, attribute):
        setattr(target, attribute, value)


def _scene_name_component(value: str) -> str:
    return value.strip().replace("/", "_") or "object"


def view_result(result: RetargetingResult, *, dry_run: bool = True) -> None:
    """View or summarize a result."""

    visualizer = visualizers.get(VisualizerName.DRY_RUN) if dry_run else visualizers.get(VisualizerName.VISER)
    visualizer.view(result)


visualizers.register(VisualizerName.DRY_RUN, DryRunVisualizer())
visualizers.register(VisualizerName.VISER, ViserVisualizer())
