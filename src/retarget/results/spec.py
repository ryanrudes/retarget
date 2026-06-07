"""Result and report models."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retarget.core.array import FloatArray, as_float_array
from retarget.core.enums import (
    FrameConvention,
    ObjectQposMode,
    ObjectSampleSpace,
    QuaternionOrder,
    RunStatus,
    SolverKind,
    TaskKind,
)
from retarget.core.timing import resample_linear
from retarget.mesh.interaction import InteractionMeshSpec


class VocabularyManifest(BaseModel):
    """Qualified identity and ordered values for one enum vocabulary."""

    module: str
    qualname: str
    values: tuple[str, ...]

    @classmethod
    def from_members(cls, members: tuple[StrEnum, ...]) -> VocabularyManifest | None:
        """Build a manifest from members of one concrete enum class."""

        if not members:
            return None
        vocabulary = type(members[0])
        if not all(type(member) is vocabulary for member in members):
            raise TypeError("vocabulary manifest members must use one enum class")
        return cls(
            module=vocabulary.__module__,
            qualname=vocabulary.__qualname__,
            values=tuple(member.value for member in members),
        )


class RegistryKeyManifest(BaseModel):
    """Qualified identity for one extensible registry enum member."""

    module: str
    qualname: str
    name: str
    value: str

    @classmethod
    def from_member(cls, member: StrEnum) -> RegistryKeyManifest:
        """Preserve the concrete enum class and member identity."""

        return cls(
            module=type(member).__module__,
            qualname=type(member).__qualname__,
            name=member.name,
            value=member.value,
        )

    def resolve(self, base: type[SolverKind] = SolverKind) -> SolverKind:
        """Resolve the recorded member and validate its registry-key base."""

        target: object = importlib.import_module(self.module)
        for part in self.qualname.split("."):
            target = getattr(target, part)
        if not isinstance(target, type) or not issubclass(target, base):
            raise TypeError(f"{self.module}:{self.qualname} is not a {base.__name__} vocabulary")
        member = target[self.name]
        if member.value != self.value:
            raise ValueError("serialized registry member value does not match its vocabulary")
        return member


class SolverRunReport(BaseModel):
    """Typed solver execution report."""

    requested_backend: RegistryKeyManifest
    resolved_backend: RegistryKeyManifest
    frame_statuses: tuple[str, ...]
    iterations: tuple[int, ...]


class MeshRunReport(BaseModel):
    """Interaction-mesh configuration used by one run."""

    spec: InteractionMeshSpec
    custom_builder: bool = False


class ResamplingReport(BaseModel):
    """Explicit result resampling operation."""

    source_fps: float
    target_fps: float


class RetargetingRunReport(BaseModel):
    """Structured behavioral report emitted by the retargeting engine."""

    algorithm: str
    task_kind: TaskKind
    robot_name: str
    motion_name: str
    runtime_s: float
    input_fps: float
    output_fps: float
    requested_output_fps: float | None = None
    scale_to_robot: bool
    motion_scale_factor: float | None = None
    solver: SolverRunReport
    mesh: MeshRunReport
    resampling: ResamplingReport | None = None


class ResultRobotSpec(BaseModel):
    """Backend-facing robot information required for playback."""

    name: str
    link_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_start: int
    joint_vocabulary: VocabularyManifest
    link_vocabulary: VocabularyManifest
    urdf_path: Path | None = None
    mujoco_xml_path: Path | None = None


class ResultObjectVisualPart(BaseModel):
    """One object visual part persisted for playback."""

    name: str
    mesh_path: Path
    asset_scale: tuple[float, float, float]
    rgba: tuple[float, float, float, float] | None = None


class ResultObjectSpec(BaseModel):
    """Typed object definition required for result playback."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    sample_points: FloatArray
    sample_space: ObjectSampleSpace
    qpos_mode: ObjectQposMode
    frame_convention: FrameConvention = FrameConvention.Z_UP_RIGHT_HANDED
    quaternion_order: QuaternionOrder = QuaternionOrder.WXYZ
    qpos_slice: tuple[int, int] | None = None
    mesh_path: Path | None = None
    asset_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    visual_parts: tuple[ResultObjectVisualPart, ...] = ()

    @field_validator("sample_points", mode="before")
    @classmethod
    def _validate_sample_points(cls, value: Any) -> FloatArray:
        points = as_float_array(value, shape_tail=(3,), name="sample_points")
        if points.ndim != 2:
            raise ValueError("sample_points must have shape (points, 3)")
        return points


class ResultPlaybackSpec(BaseModel):
    """Typed robot and object playback definition."""

    robot: ResultRobotSpec
    object: ResultObjectSpec | None = None


class RetargetingResult(BaseModel):
    """Strict pickle-free retargeting checkpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 2
    name: str
    status: RunStatus
    qpos: FloatArray
    fps: float = 30.0
    cost: FloatArray | None = None
    human_joints: FloatArray | None = None
    human_vocabulary: VocabularyManifest | None = None
    robot_link_positions: FloatArray | None = None
    run: RetargetingRunReport | None = None
    playback: ResultPlaybackSpec | None = None
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("qpos", mode="before")
    @classmethod
    def _validate_qpos(cls, value: Any) -> FloatArray:
        arr = as_float_array(value, name="qpos")
        if arr.ndim != 2:
            raise ValueError("qpos must have shape (frames, nq)")
        return arr

    @field_validator("cost", mode="before")
    @classmethod
    def _validate_cost(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, name="cost")
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError("cost must be 1D")
        return arr

    @field_validator("human_joints", mode="before")
    @classmethod
    def _validate_human_joints(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="human_joints")
        if arr.ndim != 3:
            raise ValueError("human_joints must have shape (frames, joints, 3)")
        return arr

    @field_validator("robot_link_positions", mode="before")
    @classmethod
    def _validate_robot_link_positions(cls, value: Any) -> FloatArray | None:
        if value is None:
            return None
        arr = as_float_array(value, shape_tail=(3,), name="robot_link_positions")
        if arr.ndim != 3:
            raise ValueError("robot_link_positions must have shape (frames, links, 3)")
        return arr

    @model_validator(mode="after")
    def _validate_lengths(self) -> RetargetingResult:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.qpos.shape[0] == 0:
            raise ValueError("qpos must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.cost is not None and len(self.cost) not in (1, self.qpos.shape[0]):
            raise ValueError("cost length must be 1 or match qpos frames")
        if self.human_joints is not None and self.human_joints.shape[0] != self.qpos.shape[0]:
            raise ValueError("human_joints frame count must match qpos")
        if self.robot_link_positions is not None and self.robot_link_positions.shape[0] != self.qpos.shape[0]:
            raise ValueError("robot_link_positions frame count must match qpos")
        return self

    @property
    def frame_count(self) -> int:
        """Number of frames."""

        return int(self.qpos.shape[0])

    def resampled(self, fps: float, *, name: str | None = None) -> RetargetingResult:
        """Return this result sampled on a new FPS grid."""

        run = self.run
        if run is not None and not np.isclose(float(fps), self.fps):
            run = run.model_copy(
                update={
                    "output_fps": float(fps),
                    "resampling": ResamplingReport(source_fps=self.fps, target_fps=float(fps)),
                }
            )
        cost = None
        if self.cost is not None:
            cost = (
                resample_linear(self.cost, self.fps, fps)
                if len(self.cost) == self.frame_count
                else np.asarray(self.cost, dtype=np.float64).copy()
            )
        human_joints = (
            resample_linear(self.human_joints, self.fps, fps) if self.human_joints is not None else None
        )
        robot_link_positions = (
            resample_linear(self.robot_link_positions, self.fps, fps)
            if self.robot_link_positions is not None
            else None
        )
        return RetargetingResult(
            schema_version=self.schema_version,
            name=name or self.name,
            status=self.status,
            qpos=resample_linear(self.qpos, self.fps, fps),
            fps=fps,
            cost=cost,
            human_joints=human_joints,
            human_vocabulary=self.human_vocabulary,
            robot_link_positions=robot_link_positions,
            run=run,
            playback=self.playback,
            warnings=self.warnings,
            provenance=dict(self.provenance),
        )

    def save_npz(self, path: str | Path) -> Path:
        """Save result to `.npz`."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.model_dump(
            mode="json",
            exclude={
                "qpos": True,
                "cost": True,
                "human_joints": True,
                "robot_link_positions": True,
                "playback": {"object": {"sample_points": True}},
            },
        )
        payload: dict[str, Any] = {
            "manifest_json": json.dumps(manifest, default=_json_default),
            "qpos": self.qpos,
        }
        if self.cost is not None:
            payload["cost"] = self.cost
        if self.human_joints is not None:
            payload["human_joints"] = self.human_joints
        if self.robot_link_positions is not None:
            payload["robot_link_positions"] = self.robot_link_positions
        if self.playback is not None and self.playback.object is not None:
            payload["object_sample_points"] = self.playback.object.sample_points
        np.savez(output, **payload)
        return output

    @classmethod
    def load_npz(cls, path: str | Path) -> RetargetingResult:
        """Load the current strict `.npz` checkpoint schema."""

        with np.load(path, allow_pickle=False) as data:
            if "manifest_json" not in data or "qpos" not in data:
                raise ValueError("not a current RetargetingResult checkpoint")
            manifest = json.loads(str(np.asarray(data["manifest_json"]).reshape(())))
            if not isinstance(manifest, dict):
                raise ValueError("manifest_json must contain a JSON object")
            manifest.update(
                qpos=data["qpos"],
                cost=data.get("cost", None),
                human_joints=data.get("human_joints", None),
                robot_link_positions=data.get("robot_link_positions", None),
            )
            playback = manifest.get("playback")
            if isinstance(playback, dict) and isinstance(playback.get("object"), dict):
                if "object_sample_points" not in data:
                    raise ValueError("result object manifest requires object_sample_points")
                playback["object"]["sample_points"] = data["object_sample_points"]
            return cls.model_validate(manifest)


class EvaluationReport(BaseModel):
    """Evaluation metrics for a result.

    Attributes:
        status (RunStatus): Overall evaluation outcome.
        source_name (str | None): Human-readable name of the evaluated result or clip.
        frame_count (int | None): Number of frames in the evaluated motion.
        qpos_dimension (int | None): Robot ``qpos`` width.
        fps (float | None): Sampling rate of the evaluated result.
        task_kind (str | None): Scene task kind string when available.
        robot_name (str | None): Robot preset or asset name.
        motion_name (str | None): Source human motion name.
        metrics (dict[str, float]): Scalar metric values keyed by name.
        metric_units (dict[str, str]): Unit strings aligned with ``metrics`` keys.
        details (dict[str, Any]): Structured diagnostic payloads (curves, thresholds, …).
        warnings (tuple[str, ...]): Non-fatal evaluation messages.
    """

    status: RunStatus = RunStatus.SUCCESS
    source_name: str | None = None
    frame_count: int | None = None
    qpos_dimension: int | None = None
    fps: float | None = None
    task_kind: str | None = None
    robot_name: str | None = None
    motion_name: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_units: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_report(self) -> EvaluationReport:
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("frame_count must be positive when provided")
        if self.qpos_dimension is not None and self.qpos_dimension <= 0:
            raise ValueError("qpos_dimension must be positive when provided")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive when provided")
        return self

    def save_json(self, path: str | Path) -> Path:
        """Save report as JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output


class EvaluationRecord(BaseModel):
    """Recorded outcome for evaluating one retargeting result.

    Attributes:
        result_path (Path): Path to the evaluated ``.npz`` retargeting output.
        status (RunStatus): Per-result evaluation outcome.
        report_path (Path | None): Optional JSON report written for this result.
        job_id (str | None): Batch or scheduler job identifier.
        source_name (str | None): Human-readable clip or run name.
        frame_count (int | None): Frame count observed during evaluation.
        message (str): Error or skip explanation when ``status`` is not success.
        metrics (dict[str, float]): Scalar metrics copied from the report.
        warnings (tuple[str, ...]): Warnings surfaced during evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_path: Path
    status: RunStatus
    report_path: Path | None = None
    job_id: str | None = None
    source_name: str | None = None
    frame_count: int | None = None
    message: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class EvaluationManifest(BaseModel):
    """Manifest written by batch evaluation runs.

    Attributes:
        schema_version (int): On-disk manifest format version.
        created_at (datetime): UTC timestamp when the manifest was first created.
        updated_at (datetime): UTC timestamp of the most recent update.
        total (int): Number of records in ``records``.
        success_count (int): Records with :attr:`~retarget.core.enums.RunStatus.SUCCESS`.
        partial_count (int): Records with :attr:`~retarget.core.enums.RunStatus.PARTIAL`.
        skipped_count (int): Records with :attr:`~retarget.core.enums.RunStatus.SKIPPED`.
        failed_count (int): Records with :attr:`~retarget.core.enums.RunStatus.FAILED`.
        records (tuple[EvaluationRecord, ...]): Per-result evaluation entries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    created_at: datetime
    updated_at: datetime
    total: int = 0
    success_count: int = 0
    partial_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    records: tuple[EvaluationRecord, ...] = ()

    @classmethod
    def from_records(cls, records: Iterable[EvaluationRecord]) -> EvaluationManifest:
        """Build a summary manifest from per-result records."""

        ordered_records = tuple(records)
        now = datetime.now(UTC)
        return cls(
            created_at=now,
            updated_at=now,
            total=len(ordered_records),
            success_count=sum(record.status == RunStatus.SUCCESS for record in ordered_records),
            partial_count=sum(record.status == RunStatus.PARTIAL for record in ordered_records),
            skipped_count=sum(record.status == RunStatus.SKIPPED for record in ordered_records),
            failed_count=sum(record.status == RunStatus.FAILED for record in ordered_records),
            records=ordered_records,
        )

    @classmethod
    def load(cls, path: str | Path) -> EvaluationManifest:
        """Load a batch evaluation manifest JSON file."""

        return cls.model_validate_json(Path(path).read_text())

    def save_json(self, path: str | Path) -> Path:
        """Save manifest JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return str(value)
