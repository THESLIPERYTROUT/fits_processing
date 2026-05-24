"""
PeakHoughDetector — faithful port of devtools/plotter.py.

Algorithm (matches plotter.py collect_signal_peaks + detect_streaks_from_peaks):
1. Percentile-clip the image.
2. Gaussian-smooth along rows, then for each row fit a polynomial to binned
   medians to get a slowly-varying baseline.
3. Find peaks in (smoothed_row − baseline) above threshold_sigma × std(residual).
   One peak per local maximum — truly sparse, not everything above a global cut.
4. Build a sparse binary mask from those peak positions, dilate for connectivity,
   run HoughLinesP.  The filter chain downstream handles multi-detections per streak.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from streakiller.config.schema import PeakHoughParams
from streakiller.detection.detector import RawDetection
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers — direct ports of plotter.py functions                      #
# ------------------------------------------------------------------ #

def _row_median_points(
    image: np.ndarray,
    row_index: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    row = image[row_index, :]
    x_edges = np.linspace(0, len(row), bins + 1, dtype=int)
    xs: list[float] = []
    medians: list[float] = []
    for start, end in zip(x_edges[:-1], x_edges[1:]):
        chunk = row[start:end]
        if len(chunk) == 0:
            continue
        xs.append((start + end) / 2)
        medians.append(float(np.median(chunk)))
    return np.array(xs), np.array(medians)


def _collect_signal_peaks(
    image: np.ndarray,
    params: PeakHoughParams,
) -> tuple[list[int], list[int]]:
    smoothed = gaussian_filter1d(
        image.astype(float),
        sigma=params.background_smooth_sigma,
        axis=1,
    )
    bins   = max(2, min(params.median_bins, image.shape[1]))
    degree = max(0, min(params.polynomial_degree, bins - 1))

    peak_xs: list[int] = []
    peak_ys: list[int] = []

    for row_index in range(image.shape[0]):
        xs, ys = _row_median_points(smoothed, row_index=row_index, bins=bins)
        coeffs = np.polyfit(xs, ys, deg=degree)
        curve  = np.polyval(coeffs, np.arange(image.shape[1]))

        residual_row = smoothed[row_index, :] - curve
        threshold    = params.threshold_sigma * float(np.std(residual_row))
        peaks, _     = find_peaks(residual_row, height=threshold)

        peak_xs.extend(peaks.tolist())
        peak_ys.extend([row_index] * len(peaks))

    return peak_xs, peak_ys


def _binary_from_peaks(
    image_shape: tuple[int, int],
    peak_xs: list[int],
    peak_ys: list[int],
) -> np.ndarray:
    height, width = image_shape[:2]
    binary = np.zeros((height, width), dtype=bool)
    if not peak_xs:
        return binary
    xs = np.asarray(peak_xs, dtype=int)
    ys = np.asarray(peak_ys, dtype=int)
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    binary[ys[in_bounds], xs[in_bounds]] = True
    return binary


# ------------------------------------------------------------------ #
# Detector                                                            #
# ------------------------------------------------------------------ #

class PeakHoughDetector:
    """
    Detects streaks via per-row polynomial baseline subtraction + find_peaks
    → sparse binary mask → HoughLinesP.

    The *binary* argument accepted by detect() is ignored — the detector
    derives its own mask from *source_data*, matching the FftCorrelationDetector
    interface.  Multi-detections per streak are handled by the filter chain
    downstream, not inside this detector.
    """

    def __init__(self, params: PeakHoughParams) -> None:
        self._p = params

    def detect(
        self,
        binary: np.ndarray,
        source_data: np.ndarray,
        min_line_length: float,
    ) -> RawDetection:
        p = self._p

        # 1. Percentile clip (plotter.py: percentile_clip inside process_image1)
        vmin, vmax = np.percentile(source_data, (p.clip_percentile_low, p.clip_percentile_high))
        clipped = np.clip(source_data, vmin, vmax).astype(float)

        # 2. Collect per-row signal peaks (plotter.py: collect_signal_peaks)
        peak_xs, peak_ys = _collect_signal_peaks(clipped, p)
        logger.info("PeakHoughDetector: %d signal peaks collected", len(peak_xs))

        # 3. Sparse binary mask (plotter.py: binary_from_peaks)
        sparse = _binary_from_peaks(clipped.shape, peak_xs, peak_ys)

        if not np.any(sparse):
            logger.info("PeakHoughDetector: no peaks found")
            return RawDetection(
                lines=np.empty((0, 1, 4), dtype=np.int32),
                binary_image=sparse.astype(np.uint8),
                normalized_display=normalize_for_display(source_data),
            )

        # 4. Dilate for HoughLinesP connectivity (plotter.py: detect_streaks_from_peaks)
        k = max(1, int(p.dilation_kernel))
        hough_binary = cv2.dilate(
            sparse.astype(np.uint8) * 255,
            np.ones((k, k), dtype=np.uint8),
        )

        # 5. HoughLinesP (plotter.py: detect_streaks_from_peaks)
        theta = np.pi / 180.0 * p.theta_deg
        raw = cv2.HoughLinesP(
            hough_binary,
            rho=p.rho,
            theta=theta,
            threshold=p.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=p.max_line_gap,
        )
        if raw is None:
            logger.info("PeakHoughDetector: HoughLinesP found no lines")
            lines: np.ndarray = np.empty((0, 1, 4), dtype=np.int32)
        else:
            lines = raw.astype(np.int32, copy=False)
            logger.info("PeakHoughDetector: HoughLinesP found %d raw lines", len(lines))

        return RawDetection(
            lines=lines,
            binary_image=hough_binary,
            normalized_display=normalize_for_display(source_data),
        )
