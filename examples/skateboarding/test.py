import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from contact_detection import BodyContactSurface, FootSupportClassification
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from motion_sync import SyncClip
from motion_sync.contact_metadata import stamp_detection_metadata
from motion_sync.contacts.foot_support import layer_from_foot_classification
from motion_sync.schemas.skateboarding import SKATE_FOOT_SUPPORT, SKATE_SESSION, Bodies
from motion_sync.surface_schema import BodyMarkerPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

AIR_COLOR = "white"
GROUND_COLOR = "green"
BOARD_COLOR = "blue"

MARKER_VIEW_MARGIN = 0.08
MIN_HALF_SPAN_M = 0.05
PLANE_HALF_WIDTH_M = 0.12
PLANE_HALF_LENGTH_M = 0.14
VICON_UP_AXIS = 2

def _atomic_save_synced_clip(clip: SyncClip, demo_path: str | Path) -> Path:
    """Write synced.npz via a temp file so interrupted saves do not corrupt the archive."""
    from motion_sync._storage import synced_dataset_path, write_synced_clip

    dest = synced_dataset_path(Path(demo_path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_synced_clip(clip, tmp)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _finite_point(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _markers_at_frame(marker_trajs: dict[object, np.ndarray], frame: int) -> np.ndarray:
    """Return marker positions in the clip world frame."""
    points: list[np.ndarray] = []
    for traj in marker_trajs.values():
        point = np.asarray(traj[frame], dtype=np.float64)
        if _finite_point(point):
            points.append(point)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def _markers_centered_at_shoe(
    center: np.ndarray,
    marker_trajs: dict[object, np.ndarray],
    frame: int,
) -> np.ndarray:
    """World markers translated to the shoe origin (world axes, not body-rotated)."""
    world = _markers_at_frame(marker_trajs, frame)
    if not world.size:
        return world
    return world - center


def _compile_sole_surface(
    sole: BodyMarkerPatch,
    marker_trajs: dict[object, np.ndarray],
    shoe_track: object,
    frame: int,
) -> BodyContactSurface | None:
    """Compile ``T_body_contact`` once from marker samples at a calibration frame."""
    try:
        return sole.compile_at_frame(marker_trajs, frame, shoe_track)
    except ValueError:
        return None


def _fixed_view_half_span(
    shoe_positions: np.ndarray,
    marker_trajs: dict[object, np.ndarray],
    *,
    margin_frac: float = MARKER_VIEW_MARGIN,
    min_half_span_m: float = MIN_HALF_SPAN_M,
) -> float:
    """Cube half-span (meters) that fits all shoe markers in the body frame."""
    relative: list[np.ndarray] = []
    for frame in range(shoe_positions.shape[0]):
        center = shoe_positions[frame]
        relative.extend(_markers_centered_at_shoe(center, marker_trajs, frame))
    if not relative:
        return max(min_half_span_m, 0.1)

    stack = np.vstack(relative)
    half_span = float(np.max(np.abs(stack)))
    return max(half_span, min_half_span_m) * (1.0 + margin_frac)


def _ground_below_view_faces(
    center: np.ndarray,
    half_span: float,
    classification: FootSupportClassification,
) -> list[np.ndarray] | None:
    """Fill the view cube under the plane floor fit from sole markers."""
    if classification.floor_normal is None or classification.floor_origin is None:
        return None
    lo = center - half_span
    hi = center + half_span
    z_bot = float(lo[VICON_UP_AXIS])
    z_hi = float(hi[VICON_UP_AXIS])
    corners_xy = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
        dtype=np.float64,
    )
    normal = np.asarray(classification.floor_normal, dtype=np.float64)
    origin = np.asarray(classification.floor_origin, dtype=np.float64)
    normal_up = float(normal[VICON_UP_AXIS])
    if abs(normal_up) <= 1e-12:
        return None
    delta_x = corners_xy[:, 0] - origin[0]
    delta_y = corners_xy[:, 1] - origin[1]
    z_on_plane = origin[VICON_UP_AXIS] - (normal[0] * delta_x + normal[1] * delta_y) / normal_up
    top = np.column_stack([corners_xy, z_on_plane])
    if float(np.min(top[:, VICON_UP_AXIS])) > z_hi + 1e-9 or float(
        np.max(top[:, VICON_UP_AXIS])
    ) < z_bot - 1e-9:
        return None
    bot = np.column_stack([corners_xy, np.full(4, z_bot, dtype=np.float64)])
    return [
        bot,
        top,
        np.stack([bot[0], bot[1], top[1], top[0]]),
        np.stack([bot[1], bot[2], top[2], top[1]]),
        np.stack([bot[2], bot[3], top[3], top[2]]),
        np.stack([bot[3], bot[0], top[0], top[3]]),
    ]


def _configure_marker_ax_draw_order(
    marker_ax: object,
    *,
    ground_patch: Poly3DCollection,
    plane_patch: Poly3DCollection,
    marker_scatter: object,
    shoe_scatter: object,
) -> None:
    """Keep ground behind shoe plane and markers regardless of view angle."""
    marker_ax.computed_zorder = False
    ground_patch.set_zorder(1)
    ground_patch.set_zsort("min")
    plane_patch.set_zorder(2)
    plane_patch.set_zsort("max")
    marker_scatter.set_zorder(3)
    marker_scatter.set_depthshade(False)
    shoe_scatter.set_zorder(4)
    shoe_scatter.set_depthshade(False)


def _apply_focused_world_limits(ax, center: np.ndarray, half_span: float) -> None:
    """Fixed-size axis window centered on a world-frame point (true Vicon coordinates)."""
    lo = center - half_span
    hi = center + half_span
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    ax.set_box_aspect((1, 1, 1))


def _first_frame_with_markers(
    shoe_positions: np.ndarray,
    marker_trajs: dict[object, np.ndarray],
) -> int:
    for frame in range(shoe_positions.shape[0]):
        if _markers_centered_at_shoe(shoe_positions[frame], marker_trajs, frame).size:
            return frame
    return 0


def _figure_section_legend(fig: plt.Figure) -> None:
    """Figure legend grouped by contact shading, height traces, and 3D elements."""
    section_specs: list[tuple[str, list[tuple[object, str]]]] = [
        (
            "Contact state",
            [
                (Patch(facecolor=AIR_COLOR, edgecolor="none", alpha=0.3), "Air"),
                (Patch(facecolor=GROUND_COLOR, edgecolor="none", alpha=0.3), "Ground"),
                (Patch(facecolor=BOARD_COLOR, edgecolor="none", alpha=0.3), "Board"),
            ],
        ),
        (
            "Height (global z)",
            [
                (Line2D([0], [0], color="C0", linewidth=1.5), "Shoe"),
                (
                    Line2D([0], [0], color="black", linewidth=1.5, linestyle="--"),
                    "Ground (under foot)",
                ),
                (Line2D([0], [0], color="gray", linewidth=1.5), "Board"),
            ],
        ),
        (
            "3D view",
            [
                (
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        color="C0",
                        markersize=7,
                    ),
                    "Shoe markers",
                ),
                (
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="*",
                        color="black",
                        markersize=11,
                    ),
                    "Shoe pose",
                ),
                (
                    Patch(facecolor="C1", edgecolor="C1", alpha=0.8),
                    "Sole plane (heel, toes)",
                ),
                (
                    Patch(facecolor=GROUND_COLOR, edgecolor="none", alpha=0.3),
                    "Ground (inferred floor)",
                ),
            ],
        ),
    ]

    handles: list[object] = []
    labels: list[str] = []
    section_titles: list[str] = []
    for section_index, (title, entries) in enumerate(section_specs):
        if section_index > 0:
            handles.append(Line2D([0], [0], linestyle="none", marker="none"))
            labels.append("")
        handles.append(Line2D([0], [0], linestyle="none", marker="none"))
        labels.append(title)
        section_titles.append(title)
        for handle, label in entries:
            handles.append(handle)
            labels.append(label)

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        if text.get_text() in section_titles:
            text.set_fontweight("bold")
            text.set_horizontalalignment("left")


def _shade_support(
    ax,
    classification: FootSupportClassification,
    foot_name: str,
) -> None:
    intervals = classification.intervals[foot_name]
    for state_name, color in (
        ("air", AIR_COLOR),
        ("ground", GROUND_COLOR),
        ("skateboard", BOARD_COLOR),
    ):
        for start, end in intervals[state_name]:
            ax.axvspan(start, end, color=color, alpha=0.3)


@dataclass
class FootMarkerView:
    label: str
    shoe_track: object
    marker_trajs: dict[object, np.ndarray]
    sole: BodyMarkerPatch
    sole_surface: BodyContactSurface | None
    classification: FootSupportClassification
    time_ax: object
    marker_ax: object
    view_half_span: float
    marker_scatter: object
    shoe_scatter: object
    plane_patch: Poly3DCollection
    ground_patch: Poly3DCollection
    time_cursor: object


def _setup_foot_marker_view(
    fig: plt.Figure,
    gridspec: GridSpec,
    *,
    row: int,
    label: str,
    shoe_track: object,
    marker_trajs: dict[object, np.ndarray],
    sole: BodyMarkerPatch,
    classification: FootSupportClassification,
    foot_name: str,
    time_s: np.ndarray,
    board_z: np.ndarray,
    sharex: object | None,
) -> FootMarkerView:
    time_ax = fig.add_subplot(gridspec[row, 0], sharex=sharex)
    marker_ax = fig.add_subplot(gridspec[row, 1], projection="3d")

    shoe_z = np.asarray(
        classification.features[foot_name]["sole_height"],
        dtype=np.float64,
    )
    ground_z = np.asarray(
        classification.features[foot_name]["floor_height_at_sole"],
        dtype=np.float64,
    )

    _shade_support(time_ax, classification, foot_name)
    time_ax.plot(time_s, shoe_z, color="C0")
    time_ax.plot(time_s, ground_z, color="black", linestyle="--")
    time_ax.plot(time_s, board_z, color="gray")
    time_ax.set_ylabel("z (m)")
    time_ax.set_ylim(
        float(np.nanmin(np.concatenate([shoe_z, ground_z, board_z])) - 0.03),
        float(np.nanmax(np.concatenate([shoe_z, ground_z, board_z])) + 0.03),
    )
    time_ax.set_title(f"{label} foot support")
    if row == 1:
        time_ax.set_xlabel("Time (s)")

    marker_ax.set_xlabel(r"$x$ (m)")
    marker_ax.set_ylabel(r"$y$ (m)")
    marker_ax.set_zlabel(r"$z$ (m)")
    marker_ax.set_proj_type("persp")

    ground_patch = Poly3DCollection(
        [],
        facecolor=GROUND_COLOR,
        edgecolor="none",
        linewidths=0,
        alpha=0.3,
        zsort="min",
    )
    marker_ax.add_collection3d(ground_patch)

    view_half_span = _fixed_view_half_span(shoe_track.positions, marker_trajs)
    start_frame = _first_frame_with_markers(shoe_track.positions, marker_trajs)
    shoe_center = np.asarray(shoe_track.positions[start_frame], dtype=np.float64)
    _apply_focused_world_limits(marker_ax, shoe_center, view_half_span)

    frame0_points = _markers_at_frame(marker_trajs, start_frame)
    marker_scatter = marker_ax.scatter(
        frame0_points[:, 0] if frame0_points.size else [shoe_center[0]],
        frame0_points[:, 1] if frame0_points.size else [shoe_center[1]],
        frame0_points[:, 2] if frame0_points.size else [shoe_center[2]],
        s=55,
        c="C0",
        depthshade=True,
    )
    shoe_scatter = marker_ax.scatter(
        [shoe_center[0]],
        [shoe_center[1]],
        [shoe_center[2]],
        s=140,
        c="black",
        marker="*",
        zorder=5,
    )
    sole_surface = _compile_sole_surface(sole, marker_trajs, shoe_track, start_frame)
    plane_corners = (
        sole.plane_corners_world_at_frame(
            sole_surface,
            shoe_track,
            start_frame,
            half_width=PLANE_HALF_WIDTH_M,
            half_length=PLANE_HALF_LENGTH_M,
        )
        if sole_surface is not None
        else None
    )
    if plane_corners is None:
        plane_corners = np.zeros((4, 3), dtype=np.float64)
        plane_patch = Poly3DCollection([plane_corners], visible=False)
    else:
        plane_patch = Poly3DCollection(
            [plane_corners],
            alpha=0.8,
            facecolor="C1",
            edgecolor="C1",
            linewidths=0.8,
        )
    marker_ax.add_collection3d(plane_patch)
    _configure_marker_ax_draw_order(
        marker_ax,
        ground_patch=ground_patch,
        plane_patch=plane_patch,
        marker_scatter=marker_scatter,
        shoe_scatter=shoe_scatter,
    )
    time_cursor = time_ax.axvline(time_s[start_frame], color="k", lw=1, zorder=5)
    marker_ax.set_title(f"{label} shoe markers (world)  t={time_s[start_frame]:.3f} s")

    foot_view = FootMarkerView(
        label=label,
        shoe_track=shoe_track,
        marker_trajs=marker_trajs,
        sole=sole,
        sole_surface=sole_surface,
        classification=classification,
        time_ax=time_ax,
        marker_ax=marker_ax,
        view_half_span=view_half_span,
        marker_scatter=marker_scatter,
        shoe_scatter=shoe_scatter,
        plane_patch=plane_patch,
        ground_patch=ground_patch,
        time_cursor=time_cursor,
    )
    ground_faces = _ground_below_view_faces(
        shoe_center, view_half_span, classification
    )
    if ground_faces is not None:
        ground_patch.set_verts(ground_faces)
    return foot_view


def _update_foot_marker_view(view: FootMarkerView, frame: int, time_s: np.ndarray) -> None:
    shoe_center = np.asarray(view.shoe_track.positions[frame], dtype=np.float64)
    world_points = _markers_at_frame(view.marker_trajs, frame)
    if world_points.size:
        view.marker_scatter._offsets3d = (
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
        )
    view.shoe_scatter._offsets3d = ([shoe_center[0]], [shoe_center[1]], [shoe_center[2]])

    plane_corners = (
        view.sole.plane_corners_world_at_frame(
            view.sole_surface,
            view.shoe_track,
            frame,
            half_width=PLANE_HALF_WIDTH_M,
            half_length=PLANE_HALF_LENGTH_M,
        )
        if view.sole_surface is not None
        else None
    )
    if plane_corners is None:
        view.plane_patch.set_visible(False)
    else:
        view.plane_patch.set_verts([plane_corners])
        view.plane_patch.set_visible(True)

    _apply_focused_world_limits(view.marker_ax, shoe_center, view.view_half_span)
    ground_faces = _ground_below_view_faces(
        shoe_center, view.view_half_span, view.classification
    )
    if ground_faces is None:
        view.ground_patch.set_visible(False)
    else:
        view.ground_patch.set_verts(ground_faces)
        view.ground_patch.set_visible(True)
    view.time_cursor.set_xdata([time_s[frame], time_s[frame]])
    view.marker_ax.set_title(f"{view.label} shoe markers (world)  t={time_s[frame]:.3f} s")


demo_path = "/Users/ryanrudes/GitHub/retarget/motion_sync_output/synced/pushoff7_twoshoes"
clip = SyncClip.load(demo_path, session=SKATE_SESSION)

foot_support_classification = SKATE_FOOT_SUPPORT.run_classification(clip)
if not clip.contact_is_fresh(SKATE_FOOT_SUPPORT):
    layer = layer_from_foot_classification(
        foot_support_classification, SKATE_FOOT_SUPPORT.layer_id
    )
    layer = stamp_detection_metadata(
        layer, clip, config=SKATE_FOOT_SUPPORT.default_config()
    )
    clip = clip.attach_contact(layer)
    _atomic_save_synced_clip(clip, demo_path)

left_shoe = clip.body(Bodies.LEFT_SHOE)
right_shoe = clip.body(Bodies.RIGHT_SHOE)
board = clip.body(Bodies.SKATEBOARD)

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(
    nrows=2,
    ncols=2,
    width_ratios=[1, 1.2],
    height_ratios=[1, 1],
    figure=fig,
)

board_z = board.positions[:, VICON_UP_AXIS]
left_view = _setup_foot_marker_view(
    fig,
    gs,
    row=0,
    label="Left",
    shoe_track=left_shoe,
    marker_trajs=clip.markers_for_body(Bodies.LEFT_SHOE),
    sole=SKATE_FOOT_SUPPORT.sole_patches.left,
    classification=foot_support_classification,
    foot_name=Bodies.LEFT_SHOE.value,
    time_s=clip.time_s,
    board_z=board_z,
    sharex=None,
)
right_view = _setup_foot_marker_view(
    fig,
    gs,
    row=1,
    label="Right",
    shoe_track=right_shoe,
    marker_trajs=clip.markers_for_body(Bodies.RIGHT_SHOE),
    sole=SKATE_FOOT_SUPPORT.sole_patches.right,
    classification=foot_support_classification,
    foot_name=Bodies.RIGHT_SHOE.value,
    time_s=clip.time_s,
    board_z=board_z,
    sharex=left_view.time_ax,
)
left_view.time_ax.set_xlim(0, clip.duration_s)

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
_figure_section_legend(fig)
plt.ion()
fig.show()

n_frames = left_shoe.positions.shape[0]
frame_pause_s = 1.0 / 30.0
try:
    while plt.fignum_exists(fig.number):
        for i in range(0, n_frames, 3):
            if not plt.fignum_exists(fig.number):
                break
            _update_foot_marker_view(left_view, i, clip.time_s)
            _update_foot_marker_view(right_view, i, clip.time_s)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(frame_pause_s)
finally:
    plt.ioff()

plt.show(block=True)
