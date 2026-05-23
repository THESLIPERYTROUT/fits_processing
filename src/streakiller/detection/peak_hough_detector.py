"""
PeakHoughDetector - row median-curve peak detection followed by HoughLinesP.

This detector was promoted from the development plotting workflow.  It builds a
sparse binary mask from row-wise signal peaks, then runs probabilistic Hough on
that peak mask.  The background-estimator binary passed by the pipeline is kept
for interface compatibility but is intentionally ignored.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter, uniform_filter
from scipy.signal import find_peaks

from streakiller.config.schema import PeakHoughParams
from streakiller.detection.detector import RawDetection
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


class PeakHoughDetector:
    """Detect streaks by finding per-row peaks and linking them with Hough."""

    def __init__(self, params: PeakHoughParams) -> None:
        self._p = params

    def detect(
        self,
        binary: np.ndarray,
        source_data: np.ndarray,
        min_line_length: float,
    ) -> RawDetection:
        """
        Detect streak segments in *source_data*.

        The *binary* argument is ignored.  A fresh peak mask is derived from the
        source image because the row-baseline peak detector needs intensity
        values, not a pre-thresholded foreground mask.
        """
        residual = self._residual_after_row_baseline(source_data)
        peak_mask = self._build_peak_mask(residual)

        logger.info(
            "PeakHoughDetector: peak_mode=%s peak mask pixels=%d",
            self._p.peak_mode,
            int(np.count_nonzero(peak_mask)),
        )
        if not np.any(peak_mask):
            return self._empty(source_data, peak_mask)

        hough_input = self._prepare_hough_input(peak_mask)
        theta = np.pi / 180.0 * self._p.theta_deg
        raw = cv2.HoughLinesP(
            hough_input,
            rho=self._p.rho,
            theta=theta,
            threshold=self._p.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=self._p.max_line_gap,
        )

        if raw is None:
            logger.info("PeakHoughDetector: HoughLinesP found no lines")
            lines = np.empty((0, 1, 4), dtype=np.int32)
        else:
            lines = raw.astype(np.int32, copy=False)
            logger.info(
                "PeakHoughDetector: HoughLinesP detected %d raw lines",
                len(lines),
            )

        return RawDetection(
            lines=lines,
            binary_image=hough_input,
            normalized_display=normalize_for_display(source_data),
        )

    def _residual_after_row_baseline(self, image: np.ndarray) -> np.ndarray:
        p = self._p
        smoothed = gaussian_filter1d(
            image.astype(np.float32, copy=False),
            sigma=p.smooth_sigma,
            axis=1,
        )
        residual = np.zeros_like(smoothed, dtype=np.float32)

        for row_index in range(image.shape[0]):
            curve = self._row_median_curve(
                smoothed,
                row_index=row_index,
                bins=p.median_bins,
                degree=p.polynomial_degree,
            )
            residual[row_index, :] = smoothed[row_index, :] - curve

        return residual

    def _build_peak_mask(self, residual: np.ndarray) -> np.ndarray:
        if self._p.peak_mode == "row_1d":
            return self._build_row_peak_mask(residual)
        return self._build_2d_peak_mask(residual)

    def _build_row_peak_mask(self, residual: np.ndarray) -> np.ndarray:
        p = self._p
        peak_mask = np.zeros(residual.shape, dtype=np.uint8)

        for row_index in range(residual.shape[0]):
            residual_row = residual[row_index, :]
            threshold = p.threshold_sigma * np.std(residual_row)
            peaks, _ = find_peaks(residual_row, height=threshold)
            peak_mask[row_index, peaks] = 255

        return peak_mask

    def _build_2d_peak_mask(self, residual: np.ndarray) -> np.ndarray:
        p = self._p
        local_window = self._odd_at_least(p.local_window, 3)
        local_max_size = self._odd_at_least(p.local_max_size, 1)

        residual_float = residual.astype(np.float32, copy=False)
        local_mean = uniform_filter(residual_float, size=local_window, mode="reflect")
        local_sq_mean = uniform_filter(
            residual_float * residual_float,
            size=local_window,
            mode="reflect",
        )
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

        global_floor = p.global_floor_sigma * float(np.std(residual_float))
        threshold = np.maximum(
            local_mean + p.threshold_sigma * local_std,
            global_floor,
        )

        local_max = maximum_filter(
            residual_float,
            size=local_max_size,
            mode="reflect",
        )
        peak_mask = (
            (residual_float > threshold)
            & (residual_float == local_max)
            & (residual_float > 0)
        )
        return peak_mask.astype(np.uint8) * 255

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(minimum, int(value))
        if value % 2 == 0:
            value += 1
        return value

    @staticmethod
    def _row_median_curve(
        image: np.ndarray,
        row_index: int,
        bins: int,
        degree: int,
    ) -> np.ndarray:
        width = image.shape[1]
        bins = max(2, min(bins, width))
        degree = max(0, min(degree, bins - 1))

        row = image[row_index, :]
        x_edges = np.linspace(0, width, bins + 1, dtype=int)
        xs: list[float] = []
        medians: list[float] = []

        for start, end in zip(x_edges[:-1], x_edges[1:]):
            chunk = row[start:end]
            if len(chunk) == 0:
                continue
            xs.append((start + end) / 2)
            medians.append(float(np.median(chunk)))

        x_full = np.arange(width)
        coeffs = np.polyfit(np.asarray(xs), np.asarray(medians), deg=degree)
        return np.polyval(coeffs, x_full)

    def _prepare_hough_input(self, peak_mask: np.ndarray) -> np.ndarray:
        k = self._p.dilation_kernel
        if k <= 1:
            return peak_mask
        kernel = np.ones((k, k), dtype=np.uint8)
        return cv2.dilate(peak_mask, kernel)

    @staticmethod
    def _empty(source_data: np.ndarray, peak_mask: np.ndarray) -> RawDetection:
        return RawDetection(
            lines=np.empty((0, 1, 4), dtype=np.int32),
            binary_image=peak_mask,
            normalized_display=normalize_for_display(source_data),
        )
