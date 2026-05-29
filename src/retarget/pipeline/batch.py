"""Resumable batch execution for retargeting workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from retarget.core.enums import RunStatus

BatchWorker = Callable[["BatchJob"], Mapping[str, Any] | None]


class BatchJob(BaseModel):
    """One retargeting job in a batch run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    motion: Path
    output: Path
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchRunRecord(BaseModel):
    """Recorded outcome for one batch job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    motion: Path
    output: Path
    status: RunStatus
    started_at: datetime
    finished_at: datetime
    message: str = ""
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Wall-clock duration in seconds."""

        return max(0.0, (self.finished_at - self.started_at).total_seconds())


class BatchManifest(BaseModel):
    """Manifest written by resumable batch runs."""

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
        """Run jobs, skip existing successful outputs, and write a manifest incrementally."""

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
        metadata = dict(worker(job) or {})
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
        metadata=metadata,
    )


def _skipped_record(job: BatchJob, previous: BatchRunRecord | None) -> BatchRunRecord:
    now = _utc_now()
    metadata: dict[str, Any] = {}
    if previous is not None:
        metadata["previous_status"] = previous.status.value
    return BatchRunRecord(
        job_id=job.id,
        motion=job.motion,
        output=job.output,
        status=RunStatus.SKIPPED,
        started_at=now,
        finished_at=now,
        message="output exists; pass force=True or --force to rerun",
        metadata=metadata,
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
