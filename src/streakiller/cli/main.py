"""
Command-line interface for the streakiller pipeline.

Usage
-----
    streakiller process --images-dir images/ --config config.json
    streakiller process images/img1.fits images/img2.fits
    streakiller validate-config --config config.json
    streakiller list-files --images-dir images/
    python -m streakiller process --images-dir images/
"""
from __future__ import annotations

import csv
import glob as _glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple, Optional

import click

from streakiller.config.schema import ConfigError, PipelineConfig

logger = logging.getLogger(__name__)


class _WorkerResult(NamedTuple):
    error: Optional[str]
    elapsed_s: float
    pid: int


@click.group()
def cli() -> None:
    """Satellite streak detection for FITS astronomical images."""


# ------------------------------------------------------------------ #
# process                                                              #
# ------------------------------------------------------------------ #

@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--config", "config_path", default="config.json", show_default=True,
              help="Path to config.json")
@click.option("--images-dir", default=None,
              help="Directory of FITS files (mutually exclusive with FILES)")
@click.option("--glob", "glob_pattern", default="*.fit*", show_default=True,
              help="Glob pattern inside --images-dir")
@click.option("--output-dir", default=None, help="Override output_dir from config")
@click.option("--workers", default=1, show_default=True,
              help="Number of parallel worker processes")
@click.option("--log-format", type=click.Choice(["text", "json"]), default="text",
              show_default=True)
@click.option("--dry-run", is_flag=True, help="Print matched files and exit")
@click.option("--fail-fast", is_flag=True,
              help="Exit immediately on first error (default: collect errors)")
@click.option("--timing-log", "timing_log_path", default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Write per-file timing data to this CSV file")
def process(
    files: tuple,
    config_path: str,
    images_dir: Optional[str],
    glob_pattern: str,
    output_dir: Optional[str],
    workers: int,
    log_format: str,
    dry_run: bool,
    fail_fast: bool,
    timing_log_path: Optional[str],
) -> None:
    """Process one or more FITS files for satellite streaks."""
    _setup_logging(log_format)

    # Load and validate config ----------------------------------------- #
    try:
        cfg = PipelineConfig.from_json(config_path)
    except FileNotFoundError:
        click.echo(f"Error: config file not found: {config_path}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Error reading config: {exc}", err=True)
        sys.exit(2)

    if output_dir:
        cfg.output_dir = output_dir

    try:
        cfg.validate()
    except ConfigError as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(2)

    # Resolve input files ---------------------------------------------- #
    if files and images_dir:
        click.echo("Error: specify either FILES or --images-dir, not both.", err=True)
        sys.exit(2)

    if files:
        paths = [Path(f) for f in files]
    elif images_dir:
        pattern = str(Path(images_dir) / glob_pattern)
        paths = sorted(Path(p) for p in _glob.glob(pattern))
    else:
        # Fall back to config images_dir
        pattern = str(Path(cfg.images_dir) / glob_pattern)
        paths = sorted(Path(p) for p in _glob.glob(pattern))

    if not paths:
        click.echo("Error: no FITS files matched.", err=True)
        sys.exit(3)

    if dry_run:
        click.echo(f"Matched {len(paths)} file(s):")
        for p in paths:
            click.echo(f"  {p}")
        sys.exit(0)

    # Process ----------------------------------------------------------- #
    click.echo(f"Processing {len(paths)} file(s) with {workers} worker(s)...")
    errors = _run_pipeline(
        paths, cfg, workers, fail_fast,
        timing_log_path=Path(timing_log_path) if timing_log_path else None,
    )

    if errors:
        click.echo(f"\n{len(errors)} file(s) failed:", err=True)
        for path, err in errors:
            click.echo(f"  {path}: {err}", err=True)
        sys.exit(1)

    click.echo(f"Done. {len(paths)} file(s) processed successfully.")


# ------------------------------------------------------------------ #
# validate-config                                                      #
# ------------------------------------------------------------------ #

@cli.command("validate-config")
@click.option("--config", "config_path", default="config.json", show_default=True)
def validate_config(config_path: str) -> None:
    """Validate a config.json file and print the parsed result."""
    try:
        cfg = PipelineConfig.from_json(config_path)
        cfg.validate()
        import dataclasses
        click.echo(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
        click.echo("\nConfig is valid.")
    except FileNotFoundError:
        click.echo(f"Error: file not found: {config_path}", err=True)
        sys.exit(2)
    except ConfigError as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)


# ------------------------------------------------------------------ #
# list-files                                                           #
# ------------------------------------------------------------------ #

@cli.command("list-files")
@click.option("--images-dir", required=True)
@click.option("--glob", "glob_pattern", default="*.fit*", show_default=True)
def list_files(images_dir: str, glob_pattern: str) -> None:
    """List FITS files that would be processed."""
    pattern = str(Path(images_dir) / glob_pattern)
    paths = sorted(_glob.glob(pattern))
    if not paths:
        click.echo("No files matched.")
        sys.exit(3)
    for p in paths:
        click.echo(p)


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _setup_logging(log_format: str) -> None:
    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def _run_pipeline(
    paths: list[Path],
    cfg: PipelineConfig,
    workers: int,
    fail_fast: bool,
    timing_log_path: Optional[Path] = None,
) -> list[tuple[Path, str]]:
    from streakiller.io.fits_loader import FitsLoader
    from streakiller.pipeline.streak_pipeline import StreakPipeline

    errors: list[tuple[Path, str]] = []
    timings: list[tuple[Path, _WorkerResult]] = []

    if workers == 1:
        loader = FitsLoader()
        pipeline = StreakPipeline.from_config(cfg)
        for path in paths:
            wr = _process_one_path(loader, pipeline, path)
            timings.append((path, wr))
            _log_file_timing(path, wr)
            if wr.error:
                errors.append((path, wr.error))
                if fail_fast:
                    _write_timing_log(timings, timing_log_path)
                    return errors
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(cfg,),
        ) as pool:
            future_to_path = {
                pool.submit(_process_path_worker, str(path)): path
                for path in paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    wr = future.result()
                    timings.append((path, wr))
                    _log_file_timing(path, wr)
                    if wr.error:
                        errors.append((path, wr.error))
                        if fail_fast:
                            _write_timing_log(timings, timing_log_path)
                            return errors
                except Exception as exc:
                    wr = _WorkerResult(error=str(exc), elapsed_s=0.0, pid=0)
                    timings.append((path, wr))
                    errors.append((path, str(exc)))
                    if fail_fast:
                        _write_timing_log(timings, timing_log_path)
                        return errors

    _log_timing_summary(timings)
    _write_timing_log(timings, timing_log_path)
    return errors


def _log_file_timing(path: Path, wr: _WorkerResult) -> None:
    status = "error" if wr.error else "ok"
    logger.info(
        "[pid %d] %s — %.2fs (%s)",
        wr.pid, path.name, wr.elapsed_s, status,
    )


def _log_timing_summary(timings: list[tuple[Path, _WorkerResult]]) -> None:
    if not timings:
        return
    elapsed = [wr.elapsed_s for _, wr in timings]
    total = sum(elapsed)
    avg = total / len(elapsed)
    min_idx = elapsed.index(min(elapsed))
    max_idx = elapsed.index(max(elapsed))
    logger.info(
        "Timing summary: %d file(s) | total=%.2fs | avg=%.2fs | "
        "min=%.2fs (%s) | max=%.2fs (%s)",
        len(timings),
        total, avg,
        elapsed[min_idx], timings[min_idx][0].name,
        elapsed[max_idx], timings[max_idx][0].name,
    )


def _write_timing_log(
    timings: list[tuple[Path, _WorkerResult]],
    timing_log_path: Optional[Path],
) -> None:
    if not timing_log_path or not timings:
        return
    with open(timing_log_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "worker_pid", "elapsed_s", "status"])
        for path, wr in timings:
            status = f"error: {wr.error}" if wr.error else "ok"
            writer.writerow([str(path), wr.pid, f"{wr.elapsed_s:.4f}", status])
    logger.info("Timing log written to %s", timing_log_path)


def _process_one_path(loader, pipeline, path: Path) -> _WorkerResult:
    pid = os.getpid()
    t0 = time.perf_counter()
    try:
        image = loader.load(path)
        result = pipeline.process(image)
        elapsed = time.perf_counter() - t0
        return _WorkerResult(error=result.error or None, elapsed_s=elapsed, pid=pid)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return _WorkerResult(error=str(exc), elapsed_s=elapsed, pid=pid)


# Module-level state populated once per worker process by _init_worker.
_worker_loader = None
_worker_pipeline = None


def _init_worker(cfg: PipelineConfig) -> None:
    """Called once per worker process at startup — build the pipeline here, not per-file."""
    global _worker_loader, _worker_pipeline
    from streakiller.io.fits_loader import FitsLoader
    from streakiller.pipeline.streak_pipeline import StreakPipeline

    _worker_loader = FitsLoader()
    _worker_pipeline = StreakPipeline.from_config(cfg)


def _process_path_worker(path_str: str) -> _WorkerResult:
    """Top-level function suitable for ProcessPoolExecutor (must be picklable)."""
    return _process_one_path(_worker_loader, _worker_pipeline, Path(path_str))


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        from datetime import datetime, timezone
        d = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source_file": getattr(record, "source_file", None),
            "stage": getattr(record, "stage", None),
        }
        return json.dumps({k: v for k, v in d.items() if v is not None})
