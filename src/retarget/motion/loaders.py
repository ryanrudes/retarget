"""Motion file loaders."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core.enums import FrameConvention, MotionLoaderSuffix, QuaternionOrder
from retarget.core.pose import PoseSequence
from retarget.motion.registry import motion_formats, motion_loaders
from retarget.motion.spec import MotionFormatSpec, MotionSequence


class JsonMotionLoader:
    """Load a small JSON fixture motion."""

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence:
        """Load a JSON fixture motion file.

        Args:
            path (Path): ``.json`` file containing ``joint_positions`` and optional metadata.
            spec (MotionFormatSpec): Default joint names, fps, and frame convention.
            name (str | None): Override sequence name; uses the file ``name`` field or stem when omitted.

        Returns:
            MotionSequence: Parsed motion sequence.

        Raises:
            KeyError: If required keys such as ``joint_positions`` are missing.
            ValueError: If arrays fail shape or validation checks.
        """

        data = json.loads(path.read_text())
        joint_names = tuple(data.get("joint_names", spec.joint_names))
        positions = np.asarray(data["joint_positions"], dtype=np.float64)
        metadata = dict(data.get("metadata", {}))
        source_height_m = _optional_height_from_mapping(data)
        fps = float(data.get("fps", spec.default_fps))
        frame = _frame_from_mapping(data, spec.frame_convention)
        return MotionSequence(
            name=name or data.get("name") or path.stem,
            joint_positions=positions,
            joint_names=joint_names,
            fps=fps,
            frame=frame,
            root_poses=_root_poses_from_mapping(data, spec, fps=fps, frame=frame),
            source_height_m=source_height_m,
            metadata=metadata,
        )


class NpyMotionLoader:
    """Load ``(T, J, 3)`` positions from ``.npy``."""

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence:
        """Load raw joint positions from a NumPy ``.npy`` file.

        Args:
            path (Path): ``.npy`` array with shape ``(frames, joints, 3)``.
            spec (MotionFormatSpec): Supplies ``joint_names``, ``default_fps``, and ``frame_convention``.
            name (str | None): Override sequence name; defaults to the file stem.

        Returns:
            MotionSequence: Motion built from the array and format defaults.

        Raises:
            ValueError: If the loaded array fails ``MotionSequence`` validation.
        """

        return MotionSequence(
            name=name or path.stem,
            joint_positions=np.load(path),
            joint_names=spec.joint_names,
            fps=spec.default_fps,
            frame=spec.frame_convention,
        )


class NpzMotionLoader:
    """Load global joint positions from ``.npz`` archives."""

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence:
        """Load joint positions and optional sidecar fields from ``.npz``.

        Args:
            path (Path): Archive containing ``global_joint_positions``, ``joint_positions``, or ``joints``.
            spec (MotionFormatSpec): Default joint names, fps, and frame convention.
            name (str | None): Override sequence name; defaults to the file stem.

        Returns:
            MotionSequence: Parsed motion with optional root poses and height metadata.

        Raises:
            KeyError: If no recognized position array key is present.
            ValueError: If arrays fail validation.
        """

        data = np.load(path, allow_pickle=True)
        positions = _first_present(data, "global_joint_positions", "joint_positions", "joints")
        joint_names_raw: Any = data.get("joint_names", spec.joint_names)
        joint_names = tuple(str(v) for v in joint_names_raw)
        metadata: dict[str, Any] = {}
        source_height_m = None
        if "height" in data:
            source_height_m = float(np.asarray(data["height"]).reshape(()))
        if "height_m" in data:
            source_height_m = float(np.asarray(data["height_m"]).reshape(()))
        _reject_legacy_link_targets(data)
        fps = float(np.asarray(data["fps"]).reshape(())) if "fps" in data else spec.default_fps
        frame = _frame_from_mapping(data, spec.frame_convention)
        return MotionSequence(
            name=name or path.stem,
            joint_positions=positions,
            joint_names=joint_names,
            fps=fps,
            frame=frame,
            root_poses=_root_poses_from_mapping(data, spec, fps=fps, frame=frame),
            source_height_m=source_height_m,
            metadata=metadata,
        )


class CsvMotionLoader:
    """Load wide CSV files with one row per frame and ``{joint}_{axis}`` columns.

    Attributes:
        TIME_COLUMNS (tuple[str, ...]): Normalized column names tried for per-row timestamps.
        FRAME_COLUMNS (tuple[str, ...]): Normalized column names tried for frame indices.
        AXES (tuple[str, ...]): Coordinate suffixes appended to joint names (``x``, ``y``, ``z``).
    """

    TIME_COLUMNS = ("time_s", "time", "timestamp")
    FRAME_COLUMNS = ("frame", "frame_idx", "frame_index")
    AXES = ("x", "y", "z")

    def load(self, path: Path, spec: MotionFormatSpec, *, name: str | None = None) -> MotionSequence:
        """Load a wide CSV motion table.

        Each row is one frame. Joint coordinates are read from ``{joint}_{axis}`` columns
        (also ``.`` and ``:`` separators). Optional root-pose columns are detected.

        Args:
            path (Path): CSV file with a header row.
            spec (MotionFormatSpec): Joint order, contact joints, quaternion order, and defaults.
            name (str | None): Override sequence name; defaults to the file stem.

        Returns:
            MotionSequence: Parsed motion sequence.

        Raises:
            ValueError: If the file is empty or required coordinate columns are missing.
            KeyError: If a joint coordinate column cannot be resolved.
        """

        rows = list(csv.DictReader(path.read_text().splitlines()))
        if not rows:
            raise ValueError(f"{path} does not contain any motion rows")
        rows = _sort_csv_rows(rows)
        normalized_rows = [{_normalize_column(key): value for key, value in row.items()} for row in rows]
        positions = np.zeros((len(normalized_rows), len(spec.joint_names), 3), dtype=np.float64)
        for frame_idx, row in enumerate(normalized_rows):
            for joint_idx, joint_name in enumerate(spec.joint_names):
                for axis_idx, axis in enumerate(self.AXES):
                    positions[frame_idx, joint_idx, axis_idx] = _csv_float(
                        row,
                        _coordinate_column_candidates(joint_name, axis),
                    )
        metadata: dict[str, Any] = {}
        source_height_m = _optional_csv_float(normalized_rows[0], ("height_m", "height"))
        fps = _csv_fps(normalized_rows, spec.default_fps)
        return MotionSequence(
            name=name or path.stem,
            joint_positions=positions,
            joint_names=spec.joint_names,
            fps=fps,
            frame=spec.frame_convention,
            root_poses=_csv_root_poses(normalized_rows, spec, fps=fps),
            source_height_m=source_height_m,
            metadata=metadata,
        )


def _first_present(data: Any, *keys: str) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise KeyError(f"Expected one of {keys} in motion file")


def _reject_legacy_link_targets(data: Any) -> None:
    keys = {
        "link_target_names",
        "link_target_positions",
        "link_target_weights",
        "link_target_masks",
        "link_target_source",
    }
    present = sorted(key for key in keys if key in data)
    if present:
        raise ValueError(
            "NPZ link_target_* arrays are no longer loaded into MotionSequence. "
            "Use retarget.motion.LinkTargetPlan or a typed integration source instead. "
            f"Found: {', '.join(present)}"
        )


def _optional_height_from_mapping(data: dict[str, Any]) -> float | None:
    value = data.get("source_height_m", data.get("height_m", data.get("height")))
    return None if value is None else float(value)


def _root_poses_from_mapping(
    data: Any,
    spec: MotionFormatSpec,
    *,
    fps: float,
    frame: FrameConvention,
) -> PoseSequence | None:
    raw_poses = _optional_present(data, "root_poses")
    if raw_poses is not None:
        if isinstance(raw_poses, PoseSequence):
            return raw_poses
        if isinstance(raw_poses, np.ndarray) and raw_poses.dtype == object:
            raw_poses = raw_poses.tolist()
        if isinstance(raw_poses, dict):
            positions = _optional_present(
                raw_poses,
                "positions",
                "translations",
                "root_positions",
                "root_translations",
            )
            quaternions = _optional_present(raw_poses, "quaternions", "root_quaternions", "root_rotations")
            order = _quaternion_order_from_mapping(raw_poses, spec.quaternion_order)
            return _pose_sequence_from_arrays(positions, quaternions, fps=fps, frame=frame, quaternion_order=order)
        if isinstance(raw_poses, list | tuple):
            positions = []
            quaternions = []
            orders: list[QuaternionOrder] = []
            for item in raw_poses:
                if not isinstance(item, dict):
                    raise ValueError("root_poses entries must be mappings")
                position = _optional_present(item, "translation", "position", "root_position", "root_translation")
                quaternion = _optional_present(item, "quaternion", "root_quaternion", "root_rotation")
                if position is None or quaternion is None:
                    raise ValueError("each root_poses entry must contain a translation/position and quaternion")
                positions.append(position)
                quaternions.append(quaternion)
                orders.append(_quaternion_order_from_mapping(item, spec.quaternion_order))
            if len(set(orders)) > 1:
                raise ValueError("all root_poses entries must use the same quaternion order")
            order = orders[0] if orders else spec.quaternion_order
            return _pose_sequence_from_arrays(
                positions,
                quaternions,
                fps=fps,
                frame=frame,
                quaternion_order=order,
            )
        raise ValueError("root_poses must be a mapping or a sequence of mappings")

    positions = _optional_present(data, "root_positions", "root_translations", "root_position", "root_translation")
    quaternions = _optional_present(data, "root_quaternions", "root_rotations", "root_quaternion", "root_rotation")
    if positions is None and quaternions is None:
        return None
    order = _quaternion_order_from_mapping(data, spec.quaternion_order)
    return _pose_sequence_from_arrays(positions, quaternions, fps=fps, frame=frame, quaternion_order=order)


def _pose_sequence_from_arrays(
    positions: Any,
    quaternions: Any,
    *,
    fps: float,
    frame: FrameConvention,
    quaternion_order: QuaternionOrder,
) -> PoseSequence:
    if positions is None or quaternions is None:
        raise ValueError("root poses require both positions/translations and quaternions")
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(quaternions, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)
    return PoseSequence.from_arrays(
        pos,
        quat,
        fps=fps,
        quaternion_order=quaternion_order,
        frame=frame,
    )


def _optional_present(data: Any, *keys: str) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _quaternion_order_from_mapping(data: Any, default: QuaternionOrder) -> QuaternionOrder:
    for key in ("root_quaternion_order", "quaternion_order"):
        if key not in data:
            continue
        return QuaternionOrder(_scalar_string(data[key]))
    return default


def _frame_from_mapping(data: Any, default: FrameConvention) -> FrameConvention:
    for key in ("frame_convention", "frame"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, FrameConvention):
            return value
        return FrameConvention(_scalar_string(value))
    return default


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        scalar = array.reshape(()).item()
        if isinstance(scalar, bytes):
            scalar = scalar.decode()
        return str(scalar)
    return str(value)


def _sort_csv_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    frame_key = _first_csv_key(rows[0], CsvMotionLoader.FRAME_COLUMNS)
    if frame_key is None:
        return rows
    return sorted(rows, key=lambda row: int(float(row[frame_key])))


def _csv_fps(rows: list[dict[str, str]], default_fps: float) -> float:
    fps = _optional_csv_float(rows[0], ("fps",))
    if fps is not None:
        return fps
    time_key = _first_csv_key(rows[0], CsvMotionLoader.TIME_COLUMNS)
    if time_key is None or len(rows) < 2:
        return default_fps
    times = np.asarray([float(row[time_key]) for row in rows], dtype=np.float64)
    deltas = np.diff(times)
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        return default_fps
    return float(1.0 / np.mean(positive))


def _coordinate_column_candidates(joint_name: str, axis: str) -> tuple[str, ...]:
    return (
        _normalize_column(f"{joint_name}_{axis}"),
        _normalize_column(f"{joint_name}.{axis}"),
        _normalize_column(f"{joint_name}:{axis}"),
        _normalize_column(f"{joint_name} {axis}"),
    )


def _csv_float(row: dict[str, str], keys: tuple[str, ...]) -> float:
    value = _optional_csv_float(row, keys)
    if value is None:
        raise KeyError(f"Missing CSV column; expected one of {keys}")
    return value


def _optional_csv_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return None


def _csv_root_poses(rows: list[dict[str, str]], spec: MotionFormatSpec, *, fps: float) -> PoseSequence | None:
    position_keys = [
        _first_csv_key(rows[0], _root_position_column_candidates(axis))
        for axis in CsvMotionLoader.AXES
    ]
    components = ("w", "x", "y", "z") if spec.quaternion_order == QuaternionOrder.WXYZ else ("x", "y", "z", "w")
    quaternion_keys = [
        _first_csv_key(rows[0], _root_quaternion_column_candidates(component))
        for component in components
    ]
    has_position = any(key is not None for key in position_keys)
    has_quaternion = any(key is not None for key in quaternion_keys)
    if not has_position and not has_quaternion:
        return None
    if any(key is None for key in position_keys) or any(key is None for key in quaternion_keys):
        raise ValueError("CSV root pose columns must include all root position axes and quaternion components")
    positions = np.asarray(
        [[float(row[key]) for key in position_keys if key is not None] for row in rows],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [[float(row[key]) for key in quaternion_keys if key is not None] for row in rows],
        dtype=np.float64,
    )
    return PoseSequence.from_arrays(
        positions,
        quaternions,
        fps=fps,
        quaternion_order=spec.quaternion_order,
        frame=spec.frame_convention,
    )


def _root_position_column_candidates(axis: str) -> tuple[str, ...]:
    return (
        _normalize_column(f"root_position_{axis}"),
        _normalize_column(f"root_translation_{axis}"),
        _normalize_column(f"root_pose_position_{axis}"),
        _normalize_column(f"root_pose_translation_{axis}"),
    )


def _root_quaternion_column_candidates(component: str) -> tuple[str, ...]:
    return (
        _normalize_column(f"root_quaternion_{component}"),
        _normalize_column(f"root_rotation_{component}"),
        _normalize_column(f"root_q{component}"),
        _normalize_column(f"root_pose_quaternion_{component}"),
    )


def _first_csv_key(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    normalized = {_normalize_column(key): key for key in row}
    for key in keys:
        found = normalized.get(_normalize_column(key))
        if found is not None:
            return found
    return None


def _normalize_column(value: str | None) -> str:
    if value is None:
        return ""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    return normalized.strip("_")


motion_loaders.register(MotionLoaderSuffix.JSON, JsonMotionLoader())
motion_loaders.register(MotionLoaderSuffix.CSV, CsvMotionLoader())
motion_loaders.register(MotionLoaderSuffix.NPY, NpyMotionLoader())
motion_loaders.register(MotionLoaderSuffix.NPZ, NpzMotionLoader())


def load_motion(path: str | Path, format_name: str, *, name: str | None = None) -> MotionSequence:
    """Load a motion sequence using a registered format and suffix loader.

    The returned sequence is converted to :attr:`~retarget.core.enums.FrameConvention.Z_UP_RIGHT_HANDED`.

    Args:
        path (str | Path): Motion file path; the suffix selects the loader.
        format_name (str): Registered motion format name (a :class:`~retarget.core.enums.MotionFormat` value).
        name (str | None): Optional sequence name; defaults to the file stem.

    Returns:
        MotionSequence: Parsed motion in Z-up world coordinates.

    Raises:
        KeyError: If the format name or file suffix is not registered.
        ValueError: If the file content fails validation.
    """

    motion_path = Path(path)
    spec = motion_formats.get(format_name)
    loader = motion_loaders.get(motion_path.suffix.lower())
    return loader.load(motion_path, spec, name=name).to_frame(FrameConvention.Z_UP_RIGHT_HANDED)
