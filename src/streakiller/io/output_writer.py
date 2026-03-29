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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, PngImagePlugin

from streakiller.config.schema import OutputOptions
from streakiller.models.result import PipelineResult

logger = logging.getLogger(__name__)


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

        stages = [("detected", result.detected_lines)] + [
            (s.stage_name, s.lines) for s in result.filter_snapshots
        ]
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

    def _write_intermediates(self, result: PipelineResult, run_dir: Path) -> None:
        if result.binary_image is not None:
            cv2.imwrite(str(run_dir / "binary.png"), result.binary_image)
        if result.normalized_display is not None:
            cv2.imwrite(str(run_dir / "normalized_display.png"), result.normalized_display)


# ------------------------------------------------------------------ #
# Visualisation helper                                                #
# ------------------------------------------------------------------ #

def _draw_filter_stage_overlays(
    base_image: np.ndarray,
    stages: list[tuple[str, np.ndarray]],
    cmap_name: str = "tab10",
    thickness: int = 2,
) -> np.ndarray:
    out = base_image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    h, w = out.shape[:2]
    cmap = plt.get_cmap(cmap_name)
    n = max(1, len(stages))

    for idx, (name, lines) in enumerate(stages):
        if lines is None or len(lines) == 0:
            continue
        c = cmap(idx / max(1, n - 1))[:3]
        color = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
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
        c = cmap(idx / max(1, n - 1))[:3]
        color = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        ty = ly + 6 + idx * 35
        cv2.rectangle(out, (lx, ty), (lx + 20, ty + 20), color, -1)
        count = len(lines) if lines is not None and hasattr(lines, "__len__") else 0
        cv2.putText(out, f"{name}: {count}", (lx + 30, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return out
