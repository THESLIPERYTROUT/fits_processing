"""
Streak-related data types.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StreakLine:
    """A single detected streak segment in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""

    @property
    def midpoint(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @classmethod
    def from_array(cls, arr: np.ndarray, label: str = "") -> "StreakLine":
        """Build from a (1, 4) or (4,) int array as returned by HoughLinesP."""
        flat = np.asarray(arr).reshape(-1)
        return cls(int(flat[0]), int(flat[1]), int(flat[2]), int(flat[3]), label=label)


@dataclass
class FilterStageSnapshot:
    """Captures the state of detected lines after one filter stage."""

    stage_name: str
    lines_before: int
    lines_after: int
    lines: np.ndarray  # shape (N, 1, 4), dtype int32


@dataclass(frozen=True)
class StreakSNR:
    """
    Signal-to-noise ratio estimate for a single detected streak.

    Computed via aperture photometry on the raw float32 image:
      SNR = (mean_on - median_off) / (MAD_off * 1.4826)

    ``snr``, ``signal``, and ``noise`` are NaN when there are fewer
    than ``min_off_pixels`` background pixels available (e.g. the
    streak runs very close to the image edge).
    """

    streak_index: int            # index into PipelineResult.detected_lines
    snr: float                   # signal / noise; NaN if estimation failed
    signal: float                # mean on-streak excess above background (ADU)
    noise: float                 # robust local noise (MAD-based sigma, ADU)
    n_on_pixels: int             # pixels sampled in the on-streak aperture
    n_off_pixels: int            # pixels sampled in the off-streak aperture

    @property
    def is_valid(self) -> bool:
        return not math.isnan(self.snr)
