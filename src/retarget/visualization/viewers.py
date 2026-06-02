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
        table.add_column("Object")
        table.add_column("Status")
        object_label = (
            f"{playback.object.name} ({playback.object.point_count} pts)" if playback.object is not None else ""
        )
        table.add_row(
            str(playback.frame_count),
            str(playback.qpos.shape[1]),
            f"{playback.fps:g}",
            f"{playback.duration_s:.3f}s",
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
    _call_if_present(
        scene,
        "add_frame",
        "/retarget/root",
        position=frame.root_position,
        wxyz=frame.root_quaternion,
        axes_length=0.2,
        axes_radius=0.01,
    )
    if frame.human_points is not None:
        human_color = np.tile(np.asarray([[240, 160, 40]], dtype=np.uint8), (frame.human_points.shape[0], 1))
        _call_if_present(
            scene,
            "add_point_cloud",
            "/retarget/human_points/frame_0000",
            points=frame.human_points,
            colors=human_color,
            point_size=0.02,
        )
    if playback.object is not None and frame.object_points is not None:
        object_name = _scene_name_component(playback.object.name)
        object_color = np.tile(np.asarray([[90, 220, 180]], dtype=np.uint8), (frame.object_points.shape[0], 1))
        path_color = np.tile(np.asarray([[180, 140, 255]], dtype=np.uint8), (playback.object.positions.shape[0], 1))
        _call_if_present(
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
        _call_if_present(
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


def _call_if_present(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return method(*args, **kwargs)


def _scene_name_component(value: str) -> str:
    return value.strip().replace("/", "_") or "object"


def view_result(result: RetargetingResult, *, dry_run: bool = True) -> None:
    """View or summarize a result."""

    visualizer = visualizers.get(VisualizerName.DRY_RUN) if dry_run else visualizers.get(VisualizerName.VISER)
    visualizer.view(result)


visualizers.register(VisualizerName.DRY_RUN, DryRunVisualizer())
visualizers.register(VisualizerName.VISER, ViserVisualizer())
