import json

import numpy as np

from retarget.core.enums import RunStatus
from retarget.export import (
    QVEL_SCHEME,
    ExportResult,
    ExportSpec,
    build_mujoco_tracking_data,
    export_tracking,
    export_tracking_npz,
    exporters,
)
from retarget.results import RetargetingResult


def test_mujoco_tracking_export_resamples_and_saves_npz(tmp_path):
    result = RetargetingResult(
        name="linear",
        status=RunStatus.SUCCESS,
        qpos=np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64),
        fps=2.0,
    )
    path = tmp_path / "tracking.npz"

    exported = export_tracking(result, ExportSpec(output_path=path, output_fps=4))

    assert exported.path == path
    assert exported.frame_count == 3
    data = np.load(path, allow_pickle=True)
    assert data["qpos"].shape == (3, 2)
    assert data["qvel"].shape == (3, 2)
    assert np.allclose(data["time_s"], np.array([0.0, 0.25, 0.5]))
    assert np.array_equal(data["joint_pos"], data["qpos"])
    assert np.array_equal(data["joint_vel"], data["qvel"])
    assert int(np.asarray(data["schema_version"]).reshape(())) == 1
    no_pickle_data = np.load(path, allow_pickle=False)
    metadata = json.loads(no_pickle_data["metadata_json"].reshape(()).item())
    assert metadata["result_status"] == "success"
    assert metadata["qvel_scheme"] == QVEL_SCHEME
    assert metadata["source_fps"] == 2.0
    assert metadata["output_fps"] == 4.0
    assert metadata["source_frame_count"] == 2
    assert metadata["frame_count"] == 3
    assert metadata["duration_s"] == 0.5
    assert metadata["resampled"] is True
    assert metadata["qpos_dimension"] == 2


def test_mujoco_tracking_finite_difference_uses_previous_interval_scheme():
    result = RetargetingResult(
        name="nonlinear",
        status=RunStatus.SUCCESS,
        qpos=np.array([[0.0], [1.0], [3.0]], dtype=np.float64),
        fps=1.0,
    )

    tracking = build_mujoco_tracking_data(result)

    assert np.array_equal(tracking.qvel, np.array([[1.0], [1.0], [2.0]], dtype=np.float64))
    assert tracking.metadata["qvel_scheme"] == QVEL_SCHEME
    assert tracking.metadata["duration_s"] == 2.0
    assert tracking.duration_s == 2.0


def test_mujoco_tracking_single_frame_velocity_is_zero():
    result = RetargetingResult(
        name="single",
        status=RunStatus.SUCCESS,
        qpos=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        fps=30.0,
    )

    tracking = build_mujoco_tracking_data(result)

    assert tracking.frame_count == 1
    assert np.array_equal(tracking.qvel, np.zeros_like(tracking.qpos))


def test_mujoco_tracking_uses_backend_qvel_shape(tmp_path):
    result = RetargetingResult(
        name="manifold",
        status=RunStatus.SUCCESS,
        qpos=np.array([[0.0, 0.0, 9.0], [1.0, 2.0, 8.0]], dtype=np.float64),
        fps=10.0,
    )
    path = tmp_path / "tracking.npz"

    exported = export_tracking(
        result,
        ExportSpec(output_path=path, kinematics_backend=_TwoDofVelocityBackend()),
    )
    data = np.load(path, allow_pickle=True)

    assert exported.metadata["qpos_dimension"] == 3
    assert exported.metadata["qvel_dimension"] == 2
    assert data["qpos"].shape == (2, 3)
    assert data["qvel"].shape == (2, 2)
    assert np.allclose(data["qvel"], np.array([[10.0, 20.0], [10.0, 20.0]]))
    metadata = json.loads(data["metadata_json"].reshape(()).item())
    assert metadata["qvel_scheme"] == QVEL_SCHEME
    assert metadata["source_fps"] == 10.0


def test_backend_qvel_single_frame_uses_backend_dimension():
    result = RetargetingResult(
        name="single_manifold",
        status=RunStatus.SUCCESS,
        qpos=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        fps=30.0,
    )

    tracking = build_mujoco_tracking_data(result, kinematics_backend=_TwoDofVelocityBackend())

    assert tracking.qvel.shape == (1, 2)
    assert np.array_equal(tracking.qvel, np.zeros((1, 2), dtype=np.float64))


def test_export_registry_accepts_custom_exporter(tmp_path):
    class MarkerExporter:
        def export(self, result: RetargetingResult, spec: ExportSpec) -> ExportResult:
            spec.output_path.write_text(result.name)
            return ExportResult(format_name=spec.format_name, path=spec.output_path, frame_count=1, fps=result.fps)

    exporters.register("unit_test_marker", MarkerExporter(), replace=True)
    result = RetargetingResult(
        name="marker",
        status=RunStatus.SUCCESS,
        qpos=np.zeros((1, 1), dtype=np.float64),
        fps=10.0,
    )
    path = tmp_path / "marker.txt"

    exported = export_tracking(result, ExportSpec(format_name="unit_test_marker", output_path=path))

    assert exported.path.read_text() == "marker"
    assert export_tracking_npz(result, tmp_path / "compat.npz").exists()


class _TwoDofVelocityBackend:
    def forward_kinematics(self, qpos, link_names):
        raise AssertionError("not used")

    def link_positions(self, qpos, link_names):
        raise AssertionError("not used")

    def body_jacobians(self, qpos, body_names):
        raise AssertionError("not used")

    def point_jacobians(self, qpos, point_names):
        raise AssertionError("not used")

    def qpos_to_qvel(self, qpos, previous_qpos, dt):
        return (np.asarray(qpos[:2], dtype=np.float64) - np.asarray(previous_qpos[:2], dtype=np.float64)) / dt

    def integrate_qvel(self, qpos, qvel, dt):
        raise AssertionError("not used")

    def joint_limits(self):
        raise AssertionError("not used")

    def geom_distances(self, qpos, geom_pairs=None, *, max_distance=np.inf):
        raise AssertionError("not used")

    def collision_candidates(self, qpos, *, margin=0.0, geom_pairs=None):
        raise AssertionError("not used")
