"""Resumable batch execution for retargeting workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from retarget.core.enums import MotionFormatKind, RobotKind, RunStatus

BatchWorker = Callable[["BatchJob"], Mapping[str, Any] | None]
"""Callable invoked for each :class:`BatchJob`; return value is stored as run provenance."""


class BatchJob(BaseModel):
    """One retargeting job in a batch run.

    Attributes:
        id (str): Stable job identifier used in manifests and resume logic.
        motion (Path): Input motion artifact path for the worker.
        output (Path): Expected output artifact path; existing files may be skipped.
        name (str | None): Optional display name; defaults to ``id`` in callers.
        motion_format (MotionFormatKind | None): Typed source motion format override.
        robot (RobotKind | None): Typed robot registry override.
        config_path (Path | None): Optional declarative experiment config.
        provenance (dict[str, Any]): Origin and scheduling history for the job.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    motion: Path
    output: Path
    name: str | None = None
    motion_format: MotionFormatKind | None = None
    robot: RobotKind | None = None
    config_path: Path | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class BatchRunRecord(BaseModel):
    """Recorded outcome for one batch job.

    Attributes:
        job_id (str): Identifier matching :attr:`BatchJob.id`.
        motion (Path): Input path from the job definition.
        output (Path): Output path from the job definition.
        status (RunStatus): Terminal status (success, skipped, or failed).
        started_at (datetime): UTC timestamp when execution began.
        finished_at (datetime): UTC timestamp when execution completed.
        message (str): Human-readable summary or error text.
        error_type (str | None): Exception class name when ``status`` is failed.
        provenance (dict[str, Any]): Worker-returned processing history or resume information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    motion: Path
    output: Path
    status: RunStatus
    started_at: datetime
    finished_at: datetime
    message: str = ""
    error_type: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Wall-clock duration in seconds."""

        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class BatchManifest(BaseModel):
    """Manifest written by resumable batch runs.

    Attributes:
        schema_version (int): Manifest format version for forward compatibility.
        created_at (datetime): UTC time when the batch run first started.
        updated_at (datetime): UTC time of the most recent manifest write.
        input_dir (Path | None): Root directory scanned for input motions, if known.
        output_dir (Path | None): Root directory for written outputs, if known.
        pattern (str): Glob or filter pattern used to build the job list.
        total (int): Number of jobs in the batch definition.
        success_count (int): Jobs that completed with :attr:`~RunStatus.SUCCESS`.
        skipped_count (int): Jobs skipped because outputs already existed.
        failed_count (int): Jobs that raised during execution.
        records (tuple[BatchRunRecord, ...]): Per-job outcomes in submission order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: int = 1
    created_at: datetime
    updated_at: datetime
    input_dir: Path | None = None
    output_dir: Path | None = None
    pattern: str = ""
    total: int = 0
    success_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    records: tuple[BatchRunRecord, ...] = ()

    @classmethod
    def from_records(
        cls,
        *,
        records: Iterable[BatchRunRecord],
        total: int,
        created_at: datetime,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        pattern: str = "",
    ) -> BatchManifest:
        """Build a manifest and derive summary counts from records."""

        ordered_records = tuple(records)
        return cls(
            created_at=created_at,
            updated_at=_utc_now(),
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            total=total,
            success_count=sum(record.status == RunStatus.SUCCESS for record in ordered_records),
            skipped_count=sum(record.status == RunStatus.SKIPPED for record in ordered_records),
            failed_count=sum(record.status == RunStatus.FAILED for record in ordered_records),
            records=ordered_records,
        )

    @classmethod
    def load(cls, path: str | Path) -> BatchManifest:
        """Load a manifest JSON file."""

        return cls.model_validate_json(Path(path).read_text())

    def save_json(self, path: str | Path) -> Path:
        """Save manifest JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2))
        return output

    def records_by_id(self) -> dict[str, BatchRunRecord]:
        """Return records keyed by job id."""

        return {record.job_id: record for record in self.records}


class BatchRunner:
    """Execute jobs with resume-aware manifest updates.

    The worker must be pickleable when `max_workers` is greater than one.
    """

    def run(
        self,
        jobs: Iterable[BatchJob],
        worker: BatchWorker,
        *,
        manifest_path: str | Path,
        max_workers: int = 1,
        force: bool = False,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        pattern: str = "",
    ) -> BatchManifest:
        """Run jobs, skip existing outputs, and update a manifest incrementally.

        Loads an existing manifest at ``manifest_path`` when present so reruns can
        resume. Jobs whose ``output`` already exists are marked skipped unless
        ``force`` is true. After each completed (or skipped) job the manifest is
        rewritten so partial progress survives interruption.

        Args:
            jobs: Iterable of retargeting jobs to execute.
            worker: Pickleable callable invoked per pending job; must accept a
                :class:`BatchJob` and return optional provenance on success.
            manifest_path: JSON manifest path updated after each job finishes.
            max_workers: Process pool size; ``1`` runs jobs sequentially in-process.
            force: When true, rerun jobs even if ``output`` already exists.
            input_dir: Optional batch input root stored on the manifest.
            output_dir: Optional batch output root stored on the manifest.
            pattern: Optional glob or filter label stored on the manifest.

        Returns:
            Final manifest aggregating all job records and summary counts.

        Raises:
            ValueError: If ``max_workers`` is less than 1.
        """

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        manifest_file = Path(manifest_path)
        jobs_tuple = tuple(jobs)
        previous = BatchManifest.load(manifest_file) if manifest_file.exists() else None
        previous_records = previous.records_by_id() if previous is not None else {}
        created_at = previous.created_at if previous is not None else _utc_now()
        records_by_id: dict[str, BatchRunRecord] = {}
        pending: list[BatchJob] = []

        for job in jobs_tuple:
            previous_record = previous_records.get(job.id)
            if not force and job.output.exists():
                records_by_id[job.id] = _skipped_record(job, previous_record)
            else:
                pending.append(job)

        manifest = _write_manifest(
            manifest_file,
            jobs=jobs_tuple,
            records_by_id=records_by_id,
            created_at=created_at,
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
        )

        if max_workers == 1:
            for job in pending:
                records_by_id[job.id] = _execute_batch_job(job, worker)
                manifest = _write_manifest(
                    manifest_file,
                    jobs=jobs_tuple,
                    records_by_id=records_by_id,
                    created_at=created_at,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    pattern=pattern,
                )
            return manifest

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_execute_batch_job, job, worker): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                records_by_id[job.id] = future.result()
                manifest = _write_manifest(
                    manifest_file,
                    jobs=jobs_tuple,
                    records_by_id=records_by_id,
                    created_at=created_at,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    pattern=pattern,
                )
        return manifest


def _execute_batch_job(job: BatchJob, worker: BatchWorker) -> BatchRunRecord:
    started_at = _utc_now()
    try:
        provenance = dict(worker(job) or {})
    except Exception as exc:  # pragma: no cover - exact worker failures are caller-specific
        finished_at = _utc_now()
        return BatchRunRecord(
            job_id=job.id,
            motion=job.motion,
            output=job.output,
            status=RunStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            message=str(exc),
            error_type=exc.__class__.__name__,
        )
    finished_at = _utc_now()
    return BatchRunRecord(
        job_id=job.id,
        motion=job.motion,
        output=job.output,
        status=RunStatus.SUCCESS,
        started_at=started_at,
        finished_at=finished_at,
        provenance=provenance,
    )


def _skipped_record(job: BatchJob, previous: BatchRunRecord | None) -> BatchRunRecord:
    now = _utc_now()
    provenance: dict[str, Any] = {}
    if previous is not None:
        provenance["previous_status"] = previous.status.value
    return BatchRunRecord(
        job_id=job.id,
        motion=job.motion,
        output=job.output,
        status=RunStatus.SKIPPED,
        started_at=now,
        finished_at=now,
        message="output exists; pass force=True or --force to rerun",
        provenance=provenance,
    )


def _write_manifest(
    path: Path,
    *,
    jobs: tuple[BatchJob, ...],
    records_by_id: dict[str, BatchRunRecord],
    created_at: datetime,
    input_dir: Path | None,
    output_dir: Path | None,
    pattern: str,
) -> BatchManifest:
    manifest = BatchManifest.from_records(
        records=(records_by_id[job.id] for job in jobs if job.id in records_by_id),
        total=len(jobs),
        created_at=created_at,
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
    )
    manifest.save_json(path)
    return manifest


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = ["BatchJob", "BatchManifest", "BatchRunRecord", "BatchRunner", "BatchWorker"]
