"""
Pipeline result and provenance types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from streakiller.models.streak import FilterStageSnapshot, StreakSNR


@dataclass(frozen=True)
class Provenance:
    """
    Immutable audit record for a single image run.

    Serialised to processing_results.json so any output directory can be
    retrospectively analysed to see exactly what config and parameters
    produced a given set of detections.
    """

    software_version: str
    config_snapshot: dict              # JSON-serialisable copy of PipelineConfig at run time
    processing_start_utc: str          # ISO 8601
    processing_end_utc: str            # ISO 8601
    background_method_used: str
    min_line_length_used: float
    hough_threshold_used: int
    stage_line_counts: dict            # {"initial_detected": 42, "midpoint_filter": 31, "final": 12, ...}


@dataclass
class PipelineResult:
    """
    The complete output of processing one FITS image.

    ``detected_lines`` has shape (N, 1, 4) int32 — the same format as
    cv2.HoughLinesP output.  N == 0 when no streaks were found (never None).

    ``error`` is set to a non-empty string if the pipeline encountered a
    non-fatal error (e.g. calibration frames missing) and fell back to a
    degraded mode.  It is None on clean runs.
    """

    source_path: Optional[Path]
    initial_detected_lines: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 4), dtype=np.int32))
    detected_lines: np.ndarray = field(default_factory=lambda: np.empty((0, 1, 4), dtype=np.int32))
    filter_snapshots: list[FilterStageSnapshot] = field(default_factory=list)
    snr_estimates: list[StreakSNR] = field(default_factory=list)
    normalized_display: Optional[np.ndarray] = None   # uint8 (H, W) for visualisation
    binary_image: Optional[np.ndarray] = None         # uint8 (H, W) Hough input
    provenance: Optional[Provenance] = None
    error: Optional[str] = None

    @property
    def streak_count(self) -> int:
        return len(self.detected_lines)

    @property
    def succeeded(self) -> bool:
        return self.error is None
