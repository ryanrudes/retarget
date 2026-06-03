"""Command line interface for retarget."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from retarget.assets import AssetInstallManifest, AssetStore
from retarget.cli.config import RetargetingRunConfig
from retarget.core.enums import AssetKind, RunStatus, TaskKind
from retarget.core.protocols import KinematicsBackend
from retarget.export import ExportSpec, export_tracking, exporters
from retarget.kinematics import kinematics_backends
from retarget.metrics import evaluate_result, metrics
from retarget.motion import motion_formats, motion_loaders
from retarget.optimization import constraint_terms, objective_terms, solver_factories
from retarget.pipeline import BatchJob, BatchManifest, BatchRunner, BatchRunRecord, Retargeter, RetargetingProblem
from retarget.results import EvaluationManifest, EvaluationRecord, RetargetingResult
from retarget.robots import RobotSpec, robots
from retarget.visualization import view_result, visualizers

console = Console()

app = typer.Typer(
    name="retarget",
    help="Typed, extensible motion retargeting workflows.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
assets_app = typer.Typer(help="List or import local assets.", no_args_is_help=True, rich_markup_mode="rich")
app.add_typer(assets_app, name="assets")


@app.command()
def doctor() -> None:
    """Check registered components and optional dependencies."""

    console.print(_registry_table())
    console.print(_dependency_table())


def _registry_table() -> Table:
    table = Table(title="registered extension points")
    table.add_column("Registry")
    table.add_column("Count", justify="right")
    table.add_column("Names")
    rows = (
        ("motion formats", motion_formats.names()),
        ("motion loaders", motion_loaders.names()),
        ("robots", robots.names()),
        ("kinematics backends", kinematics_backends.names()),
        ("objective terms", objective_terms.names()),
        ("constraint terms", constraint_terms.names()),
        ("solver factories", solver_factories.names()),
        ("metrics", metrics.names()),
        ("exporters", exporters.names()),
        ("visualizers", visualizers.names()),
    )
    for label, names in rows:
        table.add_row(label, str(len(names)), _format_names(names))
    return table


def _dependency_table() -> Table:
    table = Table(title="optional dependencies")
    table.add_column("Package")
    table.add_column("Status")
    for module in ("cvxpy", "mujoco", "viser", "yourdfpy", "torch", "smplx", "igl"):
        table.add_row(module, "installed" if _can_import(module) else "optional missing")
    return table


def _format_names(names: tuple[str, ...]) -> str:
    return ", ".join(names) if names else "<none>"


@app.command()
def run(
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, readable=True, help="TOML/YAML/JSON run spec."),
    ] = None,
    motion: Annotated[
        Path | None,
        typer.Option("--motion", exists=True, dir_okay=False, readable=True, help="Motion file to retarget."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, writable=True, help="Destination .npz result path."),
    ] = None,
    format_name: Annotated[str | None, typer.Option("--format", help="Registered motion format name.")] = None,
    robot_name: Annotated[str | None, typer.Option("--robot", help="Registered robot name.")] = None,
    task_kind: Annotated[
        TaskKind | None,
        typer.Option("--task-kind", help="Retargeting workflow kind."),
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Optional run/result name.")] = None,
) -> None:
    """Run a single retargeting job."""

    run_config = _run_config_from_inputs(
        config_path=config,
        motion=motion,
        output=output,
        format_name=format_name,
        robot_name=robot_name,
        task_kind=task_kind,
        name=name,
    )
    result = _run_from_config(run_config)
    console.print(f"Saved {result.name} to {run_config.output}")


@app.command()
def batch(
    input_dir: Annotated[
        Path,
        typer.Option("--input-dir", exists=True, file_okay=False, readable=True, help="Directory of motion files."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False, writable=True, help="Directory for retargeted results."),
    ],
    pattern: Annotated[str, typer.Option("--pattern", help="Glob pattern inside input directory.")] = "*.json",
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, readable=True, help="Shared TOML/YAML/JSON run spec."),
    ] = None,
    format_name: Annotated[str | None, typer.Option("--format", help="Registered motion format name.")] = None,
    robot_name: Annotated[str | None, typer.Option("--robot", help="Registered robot name.")] = None,
    max_workers: Annotated[int, typer.Option("--max-workers", min=1, help="Parallel worker count.")] = 1,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing results.")] = False,
) -> None:
    """Run a directory of motions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_batch_jobs(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        format_name=format_name,
        robot_name=robot_name,
        config_path=config,
    )
    manifest_path = output_dir / "batch_manifest.json"
    manifest = BatchRunner().run(
        jobs,
        _run_batch_job,
        manifest_path=manifest_path,
        max_workers=max_workers,
        force=force,
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
    )
    console.print(
        f"Processed {manifest.success_count} motions, skipped {manifest.skipped_count}, "
        f"failed {manifest.failed_count}; manifest saved to {manifest_path}"
    )
    if manifest.failed_count:
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    result: Annotated[
        Path | None,
        typer.Option("--result", exists=True, dir_okay=False, readable=True, help="Retargeting result .npz."),
    ] = None,
    batch_manifest: Annotated[
        Path | None,
        typer.Option(
            "--batch-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Batch manifest produced by `retarget batch`.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, writable=True, help="Optional report or summary JSON path."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", file_okay=False, writable=True, help="Directory for batch metric reports."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, readable=True, help="Optional run spec for metrics."),
    ] = None,
) -> None:
    """Evaluate one result or every output in a batch manifest."""

    if (result is None) == (batch_manifest is None):
        raise typer.BadParameter("use exactly one of --result or --batch-manifest")
    if result is not None:
        if output_dir is not None:
            raise typer.BadParameter("--output-dir is only valid with --batch-manifest")
        _evaluate_single_result(result=result, output=output, config=config)
        return
    if batch_manifest is None:
        raise typer.BadParameter("--batch-manifest is required")
    manifest = _evaluate_batch_manifest(
        manifest_path=batch_manifest,
        output_dir=output_dir,
        summary_path=output,
        config_path=config,
    )
    console.print(
        f"Evaluated {manifest.success_count} results, partial {manifest.partial_count}, "
        f"skipped {manifest.skipped_count}, failed {manifest.failed_count}"
    )
    if manifest.failed_count:
        raise typer.Exit(code=1)


@app.command()
def view(
    result: Annotated[
        Path,
        typer.Option("--result", exists=True, dir_okay=False, readable=True, help="Retargeting result .npz."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--live", help="Print a summary instead of launching an interactive visualizer."),
    ] = True,
    robot_spec: Annotated[
        Path | None,
        typer.Option(
            "--robot-spec",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional robot spec file used to render URDF-backed live playback.",
        ),
    ] = None,
    show_diagnostics: Annotated[
        bool,
        typer.Option("--show-diagnostics", help="Overlay source points, root paths, and link diagnostics."),
    ] = False,
) -> None:
    """View or summarize a retargeting result."""

    robot = RobotSpec.load(robot_spec) if robot_spec is not None else None
    view_result(
        RetargetingResult.load_npz(result),
        dry_run=dry_run,
        robot_spec=robot,
        show_diagnostics=show_diagnostics,
    )


@app.command()
def export(
    result: Annotated[
        Path,
        typer.Option("--result", exists=True, dir_okay=False, readable=True, help="Retargeting result .npz."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, writable=True, help="Destination tracking .npz path."),
    ],
    format_name: Annotated[str, typer.Option("--format", help="Registered export format.")] = "mujoco_npz",
    output_fps: Annotated[int | None, typer.Option("--output-fps", min=1, help="Optional exported frame rate.")] = None,
    robot_name: Annotated[
        str | None,
        typer.Option("--robot", help="Registered robot name for backend-aware qvel export."),
    ] = None,
    robot_spec: Annotated[
        Path | None,
        typer.Option(
            "--robot-spec",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Robot spec file for qvel export.",
        ),
    ] = None,
    kinematics_backend: Annotated[
        str | None,
        typer.Option("--kinematics-backend", help="Registered backend used to convert qpos to qvel."),
    ] = None,
) -> None:
    """Export a result to a downstream tracking format."""

    backend = _resolve_export_backend(robot_name, robot_spec, kinematics_backend)
    exported = export_tracking(
        RetargetingResult.load_npz(result),
        ExportSpec(
            format_name=format_name,
            output_path=output,
            output_fps=output_fps,
            kinematics_backend=backend,
        ),
    )
    console.print(
        f"Exported {exported.frame_count} frames at {exported.fps:g} fps "
        f"to {exported.path} ({exported.format_name})"
    )


@assets_app.command("list")
def assets_list(
    store: Annotated[Path, typer.Option("--store", file_okay=False, help="Asset store directory.")] = Path(
        ".retarget_assets"
    ),
) -> None:
    """List tracked assets."""

    records = AssetStore(store).list()
    table = Table(title=f"assets: {store}")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Path")
    table.add_column("License")
    for record in records:
        table.add_row(record.name, record.kind.value, str(record.path), record.license or "")
    console.print(table)


@assets_app.command("import")
def assets_import(
    path: Annotated[Path, typer.Argument(exists=True, readable=True, help="File or directory to import.")],
    name: Annotated[str, typer.Option("--name", help="Asset manifest name.")],
    kind: Annotated[AssetKind, typer.Option("--kind", help="Asset kind.")],
    store: Annotated[Path, typer.Option("--store", file_okay=False, help="Asset store directory.")] = Path(
        ".retarget_assets"
    ),
    copy: Annotated[bool, typer.Option("--copy", help="Copy into the store instead of referencing in place.")] = False,
    sha256: Annotated[str | None, typer.Option("--sha256", help="Expected SHA-256 for a file asset.")] = None,
    license_name: Annotated[
        str | None,
        typer.Option("--license", help="License identifier or short license note."),
    ] = None,
    notice: Annotated[str | None, typer.Option("--notice", help="Attribution or NOTICE text.")] = None,
) -> None:
    """Import or register a local asset path."""

    record = AssetStore(store).import_path(
        path,
        name=name,
        kind=kind,
        copy=copy,
        sha256=sha256,
        license=license_name,
        notice=notice,
    )
    console.print(f"Imported {record.name} ({record.kind.value}) -> {record.path}")


@assets_app.command("install")
def assets_install(
    manifest: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="TOML/YAML/JSON asset install manifest."),
    ],
    store: Annotated[Path, typer.Option("--store", file_okay=False, help="Asset store directory.")] = Path(
        ".retarget_assets"
    ),
    allow_downloads: Annotated[
        bool,
        typer.Option("--allow-downloads", help="Allow HTTP(S) downloads declared by the manifest."),
    ] = False,
) -> None:
    """Install or reference assets declared in a manifest."""

    records = AssetStore(store).install_manifest(manifest, allow_downloads=allow_downloads)
    table = Table(title=f"installed assets: {store}")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Path")
    for record in records:
        table.add_row(record.name, record.kind.value, str(record.path))
    console.print(table)


@assets_app.command("validate")
def assets_validate(
    manifest: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="TOML/YAML/JSON asset install manifest."),
    ],
) -> None:
    """Validate an asset install manifest without installing anything."""

    loaded = AssetInstallManifest.load(manifest)
    table = Table(title=f"asset manifest: {manifest}")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Source")
    table.add_column("Mode")
    for requirement in loaded.assets:
        table.add_row(
            requirement.name,
            requirement.kind.value,
            requirement.source,
            "download" if requirement.is_download else "local",
        )
    console.print(table)


def _run_one(
    motion_path: Path,
    *,
    format_name: str = "minimal",
    robot_name: str = "synthetic_humanoid",
    task_kind: TaskKind,
    output: Path,
    name: str | None = None,
) -> RetargetingResult:
    return _run_from_config(
        RetargetingRunConfig(
            name=name,
            motion=motion_path,
            output=output,
            format_name=format_name,
            robot=robot_name,
            task_kind=task_kind,
        )
    )


def _run_config_from_inputs(
    *,
    config_path: Path | None,
    motion: Path | None,
    output: Path | None,
    format_name: str | None,
    robot_name: str | None,
    task_kind: TaskKind | None,
    name: str | None,
) -> RetargetingRunConfig:
    if config_path is not None:
        return RetargetingRunConfig.load(config_path).with_overrides(
            motion=motion,
            output=output,
            format_name=format_name,
            robot=robot_name,
            task_kind=task_kind,
            name=name,
        )
    if motion is None:
        raise typer.BadParameter("--motion is required when --config is not provided")
    if output is None:
        raise typer.BadParameter("--output is required when --config is not provided")
    return RetargetingRunConfig(
        name=name,
        motion=motion,
        output=output,
        format_name=format_name or "minimal",
        robot=robot_name or "synthetic_humanoid",
        task_kind=task_kind or TaskKind.ROBOT_ONLY,
    )


def _run_from_config(config: RetargetingRunConfig) -> RetargetingResult:
    problem = _problem_from_config(config)
    result = Retargeter().run(problem)
    result.save_npz(config.output)
    return result


def _problem_from_config(config: RetargetingRunConfig) -> RetargetingProblem:
    try:
        return config.build_problem()
    except (ImportError, KeyError, ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(_exception_message(exc)) from exc


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _evaluate_single_result(
    *,
    result: Path,
    output: Path | None,
    config: Path | None,
) -> None:
    problem = _problem_from_config(RetargetingRunConfig.load(config)) if config is not None else None
    report = evaluate_result(RetargetingResult.load_npz(result), problem)
    if output:
        report.save_json(output)
        console.print(f"Saved evaluation report to {output}")
    else:
        console.print_json(report.model_dump_json())


def _evaluate_batch_manifest(
    *,
    manifest_path: Path,
    output_dir: Path | None,
    summary_path: Path | None,
    config_path: Path | None,
) -> EvaluationManifest:
    batch_manifest = BatchManifest.load(manifest_path)
    reports_dir = output_dir or manifest_path.parent / "metrics"
    records = [
        _evaluate_batch_record(record, batch_manifest=batch_manifest, reports_dir=reports_dir, config_path=config_path)
        for record in batch_manifest.records
    ]
    manifest = EvaluationManifest.from_records(records)
    summary = summary_path or reports_dir / "evaluation_manifest.json"
    manifest.save_json(summary)
    _print_evaluation_manifest(manifest, summary)
    return manifest


def _evaluate_batch_record(
    record: BatchRunRecord,
    *,
    batch_manifest: BatchManifest,
    reports_dir: Path,
    config_path: Path | None,
) -> EvaluationRecord:
    report_path = _batch_report_path(record.output, batch_manifest=batch_manifest, reports_dir=reports_dir)
    if record.status == RunStatus.FAILED:
        return EvaluationRecord(
            job_id=record.job_id,
            result_path=record.output,
            report_path=None,
            status=RunStatus.SKIPPED,
            message="batch job failed; no result was evaluated",
        )
    if not record.output.exists():
        return EvaluationRecord(
            job_id=record.job_id,
            result_path=record.output,
            report_path=None,
            status=RunStatus.SKIPPED,
            message="result file is missing",
        )
    try:
        problem = _batch_record_problem(config_path, record)
        report = evaluate_result(RetargetingResult.load_npz(record.output), problem)
        report.save_json(report_path)
        return EvaluationRecord(
            job_id=record.job_id,
            result_path=record.output,
            report_path=report_path,
            status=report.status,
            source_name=report.source_name,
            frame_count=report.frame_count,
            metrics=report.metrics,
            warnings=report.warnings,
        )
    except Exception as exc:  # pragma: no cover - exact metric/plugin failures are user-defined
        return EvaluationRecord(
            job_id=record.job_id,
            result_path=record.output,
            report_path=None,
            status=RunStatus.FAILED,
            message=f"{exc.__class__.__name__}: {exc}",
        )


def _batch_record_problem(config_path: Path | None, record: BatchRunRecord) -> RetargetingProblem | None:
    if config_path is None:
        return None
    config = (
        RetargetingRunConfig.load(config_path)
        .with_overrides(
            motion=record.motion,
            output=record.output,
            name=Path(record.job_id).with_suffix("").as_posix(),
        )
    )
    return _problem_from_config(config)


def _batch_report_path(result_path: Path, *, batch_manifest: BatchManifest, reports_dir: Path) -> Path:
    if batch_manifest.output_dir is not None:
        try:
            relative = result_path.relative_to(batch_manifest.output_dir)
        except ValueError:
            relative = Path(result_path.name)
    else:
        relative = Path(result_path.name)
    return reports_dir / relative.with_suffix(".metrics.json")


def _print_evaluation_manifest(manifest: EvaluationManifest, summary_path: Path) -> None:
    table = Table(title=f"evaluation manifest: {summary_path}")
    table.add_column("Job")
    table.add_column("Status")
    table.add_column("Report")
    table.add_column("Metrics")
    for record in manifest.records:
        table.add_row(
            record.job_id or record.result_path.name,
            record.status.value,
            str(record.report_path or ""),
            ", ".join(sorted(record.metrics)),
        )
    console.print(table)


def _build_batch_jobs(
    *,
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    format_name: str | None,
    robot_name: str | None,
    config_path: Path | None,
) -> tuple[BatchJob, ...]:
    config_value = str(config_path.resolve()) if config_path is not None else ""
    jobs: list[BatchJob] = []
    for motion_path in sorted(input_dir.glob(pattern)):
        job_id = _batch_job_id(input_dir, motion_path)
        jobs.append(
            BatchJob(
                id=job_id,
                motion=motion_path,
                output=_batch_output_path(input_dir, output_dir, motion_path),
                name=Path(job_id).with_suffix("").as_posix(),
                metadata={
                    "format_name": format_name or "",
                    "robot_name": robot_name or "",
                    "config_path": config_value,
                },
            )
        )
    return tuple(jobs)


def _batch_job_id(input_dir: Path, motion_path: Path) -> str:
    try:
        return motion_path.relative_to(input_dir).as_posix()
    except ValueError:
        return motion_path.name


def _batch_output_path(input_dir: Path, output_dir: Path, motion_path: Path) -> Path:
    try:
        relative = motion_path.relative_to(input_dir)
    except ValueError:
        relative = Path(motion_path.name)
    return output_dir / relative.with_suffix(".npz")


def _run_batch_job(job: BatchJob) -> dict[str, Any]:
    format_name = _metadata_string(job, "format_name")
    robot_name = _metadata_string(job, "robot_name")
    config_path = _metadata_string(job, "config_path")
    if config_path:
        run_config = RetargetingRunConfig.load(config_path).with_overrides(
            motion=job.motion,
            output=job.output,
            format_name=format_name or None,
            robot=robot_name or None,
            name=job.name,
        )
        result = _run_from_config(run_config)
    else:
        result = _run_one(
            job.motion,
            format_name=format_name or "minimal",
            robot_name=robot_name or "synthetic_humanoid",
            task_kind=TaskKind.ROBOT_ONLY,
            output=job.output,
            name=job.name,
        )
    return {"result_name": result.name, "frames": result.frame_count, "fps": result.fps}


def _metadata_string(job: BatchJob, key: str) -> str:
    value = job.metadata.get(key, "")
    return value if isinstance(value, str) else str(value)


def _resolve_export_backend(
    robot_name: str | None,
    robot_spec: Path | None,
    backend_name: str | None,
) -> KinematicsBackend | None:
    if robot_name is None and robot_spec is None and backend_name is None:
        return None
    if robot_name is not None and robot_spec is not None:
        raise typer.BadParameter("use either --robot or --robot-spec, not both")
    if robot_name is None and robot_spec is None:
        raise typer.BadParameter("--robot or --robot-spec is required when --kinematics-backend is provided")

    if robot_spec is not None:
        robot = RobotSpec.load(robot_spec)
    else:
        if robot_name is None:
            raise typer.BadParameter("--robot or --robot-spec is required when --kinematics-backend is provided")
        try:
            robots.require_all((robot_name,))
        except KeyError as exc:
            raise typer.BadParameter(_exception_message(exc)) from exc
        robot = robots.get(robot_name)
    resolved_backend_name = backend_name or "simple"
    try:
        kinematics_backends.require_all((resolved_backend_name,))
    except KeyError as exc:
        raise typer.BadParameter(_exception_message(exc)) from exc
    return kinematics_backends.get(resolved_backend_name)(robot)


def _can_import(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    if argv is None:
        app(prog_name="retarget")
    else:
        app(args=argv, prog_name="retarget", standalone_mode=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
