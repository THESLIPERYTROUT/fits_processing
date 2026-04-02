"""
OutputWriter protocol and LocalOutputWriter implementation.

The OutputWriter Protocol means any class with a write(result) method can be
used in the pipeline — a future S3OutputWriter just needs to satisfy this
interface without changing any pipeline code.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from streakiller.config.schema import OutputOptions
from streakiller.models.result import PipelineResult

logger = logging.getLogger(__name__)

# Okabe-Ito-inspired BGR palette chosen for stronger separation under
# red-green color vision deficiencies.
STAGE_OVERLAY_COLORS: list[tuple[int, int, int]] = [
    (233, 180, 86),   # sky blue
    (0, 159, 230),    # orange
    (115, 158, 0),    # bluish green
    (66, 228, 240),   # yellow
    (178, 114, 0),    # blue
    (0, 94, 213),     # vermillion
    (167, 121, 204),  # reddish purple
]


@runtime_checkable
class OutputWriter(Protocol):
    def write(self, result: PipelineResult) -> None: ...


class LocalOutputWriter:
    """
    Writes pipeline results to the local filesystem.

    For each processed image, outputs are written to::

        output_dir / <source_stem> /
            detected_streaks.png
            streaks.csv
            filter_stage_overlays.png
            processing_results.json

    If ``options.save_intermediate_images`` is True, also writes::

        binary.png
        normalized_display.png
    """

    def __init__(self, output_dir: Path, options: OutputOptions | None = None) -> None:
        self._dir = Path(output_dir)
        self._options = options or OutputOptions()
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: PipelineResult) -> None:
        stem = result.source_path.stem if result.source_path else "unknown"
        run_dir = self._dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_csv(result, run_dir)
        self._write_annotated_image(result, run_dir)
        self._write_overlay(result, run_dir)
        self._write_provenance(result, run_dir)

        if self._options.save_intermediate_images:
            self._write_intermediates(result, run_dir)

        if self._options.save_text_summary:
            self._write_text_summary(result, run_dir)

        logger.info("Results written to %s", run_dir)

    # ------------------------------------------------------------------ #
    # Private writers                                                      #
    # ------------------------------------------------------------------ #

    def _write_csv(self, result: PipelineResult, run_dir: Path) -> None:
        path = run_dir / "streaks.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label", "x1", "y1", "x2", "y2", "midpoint_x", "midpoint_y"])
            for i, line in enumerate(result.detected_lines):
                x1, y1, x2, y2 = line[0]
                writer.writerow([
                    str(i + 1), x1, y1, x2, y2,
                    (x1 + x2) / 2, (y1 + y2) / 2,
                ])
        logger.debug("Wrote %s", path)

    def _write_annotated_image(self, result: PipelineResult, run_dir: Path) -> None:
        if result.normalized_display is None:
            return
        display = result.normalized_display.copy()
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        for i, line in enumerate(result.detected_lines):
            x1, y1, x2, y2 = line[0]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                display, str(i + 1), (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )

        path = run_dir / "detected_streaks.png"
        cv2.imwrite(str(path), display)
        logger.debug("Wrote %s", path)

    def _write_overlay(self, result: PipelineResult, run_dir: Path) -> None:
        if not result.filter_snapshots or result.binary_image is None:
            return
        base = result.binary_image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        stages = [("initial_detected", result.initial_detected_lines)] + [
            (s.stage_name, s.lines) for s in result.filter_snapshots
        ] + [("final", result.detected_lines)]
        overlay = _draw_filter_stage_overlays(base, stages)

        path = run_dir / "filter_stage_overlays.png"
        cv2.imwrite(str(path), overlay)
        logger.debug("Wrote %s", path)

    def _write_provenance(self, result: PipelineResult, run_dir: Path) -> None:
        if result.provenance is None:
            return
        record: dict = {
            "source_file": str(result.source_path),
            "streak_count": result.streak_count,
            "error": result.error,
            **{k: v for k, v in vars(result.provenance).items()},
        }
        path = run_dir / "processing_results.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        logger.debug("Wrote %s", path)

    def _write_text_summary(self, result: PipelineResult, run_dir: Path) -> None:
        p = result.provenance
        source = result.source_path.name if result.source_path else "unknown"
        width = 60

        def divider(char="─"):
            return char * width

        def row(label: str, value: str) -> str:
            return f"  {label:<28}{value}"

        lines = [
            divider("═"),
            "  STREAK DETECTION SUMMARY".center(width),
            divider("═"),
            "",
        ]

        # Source file
        lines += [
            row("File:", source),
            row("Processed:", p.processing_start_utc[:19].replace("T", "  ") if p else "—"),
            row("Duration:", _format_duration(p) if p else "—"),
            row("Software version:", p.software_version if p else "—"),
            "",
            divider(),
            "  RESULT",
            divider(),
            "",
            row("Streaks detected:", str(result.streak_count)),
            row("Status:", "OK" if result.succeeded else f"ERROR — {result.error}"),
            "",
        ]

        # Per-streak table
        if result.streak_count > 0:
            lines += [
                divider(),
                "  DETECTED STREAKS",
                divider(),
                "",
                f"  {'#':<5} {'Start (x,y)':<18} {'End (x,y)':<18} {'Length (px)':<12}",
                f"  {'─'*4} {'─'*16:<18} {'─'*16:<18} {'─'*10:<12}",
            ]
            for i, line in enumerate(result.detected_lines):
                x1, y1, x2, y2 = line[0]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                lines.append(
                    f"  {i+1:<5} ({x1:>5}, {y1:>5})     ({x2:>5}, {y2:>5})     {length:>8.1f}"
                )
            lines.append("")

        # Pipeline settings
        if p:
            lines += [
                divider(),
                "  PIPELINE SETTINGS",
                divider(),
                "",
                row("Background method:", p.background_method_used),
                row("Min line length:", f"{p.min_line_length_used:.1f} px"),
                row("Hough threshold:", str(p.hough_threshold_used)),
                "",
            ]

            # Filter stage counts
            if p.stage_line_counts:
                lines += [
                    divider(),
                    "  FILTER STAGES",
                    divider(),
                    "",
                ]
                prev = None
                for stage, count in p.stage_line_counts.items():
                    if prev is not None:
                        removed = prev - count
                        tag = f"(−{removed})" if removed > 0 else "(no change)"
                    else:
                        tag = "(raw detection)"
                    lines.append(row(f"{stage}:", f"{count:>4}  {tag}"))
                    prev = count
                lines.append("")

        lines += [divider("═"), ""]

        path = run_dir / "processing_results.txt"
        path.write_text("\n".join(lines))
        logger.debug("Wrote %s", path)

    def _write_intermediates(self, result: PipelineResult, run_dir: Path) -> None:
        if result.binary_image is not None:
            cv2.imwrite(str(run_dir / "binary.png"), result.binary_image)
        if result.normalized_display is not None:
            cv2.imwrite(str(run_dir / "normalized_display.png"), result.normalized_display)


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _format_duration(provenance) -> str:
    from datetime import datetime, timezone
    try:
        start = datetime.fromisoformat(provenance.processing_start_utc)
        end = datetime.fromisoformat(provenance.processing_end_utc)
        seconds = (end - start).total_seconds()
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    except Exception:
        return "—"


# ------------------------------------------------------------------ #
# Visualisation helper                                                #
# ------------------------------------------------------------------ #

def _draw_filter_stage_overlays(
    base_image: np.ndarray,
    stages: list[tuple[str, np.ndarray]],
    thickness: int = 3,
) -> np.ndarray:
    out = base_image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    n = max(1, len(stages))

    for idx, (name, lines) in enumerate(stages):
        if lines is None or len(lines) == 0:
            continue
        color = STAGE_OVERLAY_COLORS[idx % len(STAGE_OVERLAY_COLORS)]
        overlay = out.copy()
        for line in lines:
            l = np.asarray(line).reshape(-1)
            if l.size < 4:
                continue
            cv2.line(overlay, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, thickness, cv2.LINE_AA)
        alpha = 0.35 + 0.55 * (idx / max(1, n - 1))
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Legend
    pad = 50
    lx, ly = pad, pad
    box_w, box_h = 350, 35 * len(stages) + 12
    legend_bg = out.copy()
    cv2.rectangle(legend_bg, (lx - 4, ly - 4), (lx + box_w, ly + box_h), (0, 0, 0), -1)
    cv2.addWeighted(legend_bg, 0.45, out, 0.55, 0, out)

    for idx, (name, lines) in enumerate(stages):
        color = STAGE_OVERLAY_COLORS[idx % len(STAGE_OVERLAY_COLORS)]
        ty = ly + 6 + idx * 35
        cv2.rectangle(out, (lx, ty), (lx + 20, ty + 20), color, -1)
        count = len(lines) if lines is not None and hasattr(lines, "__len__") else 0
        cv2.putText(out, f"{name}: {count}", (lx + 30, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return out
