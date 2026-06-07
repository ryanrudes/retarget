from retarget.pipeline import BatchJob, BatchManifest, BatchRunner


def _write_batch_output(job: BatchJob) -> dict[str, object]:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    job.output.write_text(job.id)
    return {"job_id": job.id}


def _fail_batch_output(job: BatchJob) -> dict[str, object]:
    raise RuntimeError(f"failed {job.id}")


def test_batch_runner_process_pool_and_resume(tmp_path):
    jobs = tuple(
        BatchJob(
            id=f"motion_{idx}.json",
            motion=tmp_path / f"motion_{idx}.json",
            output=tmp_path / "out" / f"motion_{idx}.npz",
        )
        for idx in range(2)
    )
    manifest_path = tmp_path / "out" / "batch_manifest.json"

    manifest = BatchRunner().run(jobs, _write_batch_output, manifest_path=manifest_path, max_workers=2)

    assert manifest.success_count == 2
    assert manifest.failed_count == 0
    assert BatchManifest.load(manifest_path).records[0].provenance["job_id"] == "motion_0.json"

    resumed = BatchRunner().run(jobs, _write_batch_output, manifest_path=manifest_path)

    assert resumed.success_count == 0
    assert resumed.skipped_count == 2
    assert all(record.output.exists() for record in resumed.records)


def test_batch_runner_records_failures(tmp_path):
    job = BatchJob(id="bad.json", motion=tmp_path / "bad.json", output=tmp_path / "bad.npz")
    manifest = BatchRunner().run((job,), _fail_batch_output, manifest_path=tmp_path / "manifest.json")

    assert manifest.failed_count == 1
    assert manifest.records[0].status == "failed"
    assert manifest.records[0].error_type == "RuntimeError"
