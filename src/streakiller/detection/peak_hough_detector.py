"""
PeakHoughDetector — Hough detection on a peak-thresholded residual,
with local-noise-aware endpoint walk-out.

The detector builds its own high-pass residual from the raw image (Gaussian
blur subtraction), thresholds it to a sparse binary mask, runs HoughLinesP,
then walks outward from each Hough endpoint through the continuous residual.
The walk-out threshold is derived from a perpendicular background strip at
each endpoint (MAD-based, same approach as SnrApertureEstimator), so the
walk stops at the actual signal-to-local-noise transition rather than against
a global image statistic.
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
from streakiller.snr.aperture import _collect_pixels

logger = logging.getLogger(__name__)

_MAD_FACTOR = 1.4826
_OFF_GAP    = 3   # pixels from streak centre to start of background band
_OFF_WIDTH  = 7   # width of each background band


class PeakHoughDetector:
    """
    Detects streaks using peak thresholding on a high-pass residual + HoughLinesP,
    with local-noise-aware endpoint extension.

    The *binary* argument accepted by detect() is ignored — the detector derives
    its own mask from *source_data*, matching the FftCorrelationDetector interface.
    """

    def __init__(self, params: PeakHoughParams) -> None:
        self._p = params

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        binary: np.ndarray,
        source_data: np.ndarray,
        min_line_length: float,
    ) -> RawDetection:
        p = self._p

        # 1. Gaussian high-pass → residual
        ksize = p.gaussian_kernel_size if p.gaussian_kernel_size % 2 == 1 \
                else p.gaussian_kernel_size + 1
        bg       = cv2.GaussianBlur(source_data, (ksize, ksize), 0)
        residual = (source_data - bg).astype(np.float32)

        # 2. Sparse binary mask from global peak threshold
        global_std = float(np.std(residual))
        floor_peak = p.threshold_sigma * global_std
        mask = (residual >= floor_peak).astype(np.uint8) * 255

        # 3. HoughLinesP on the sparse mask
        theta = np.pi / 180.0 * p.theta_deg
        raw = cv2.HoughLinesP(
            mask,
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
            lines = raw
            logger.info("PeakHoughDetector: HoughLinesP found %d raw lines", len(lines))

        # 4. Endpoint walk-out through continuous residual
        lines = self._refine_endpoints(lines, residual)

        return RawDetection(
            lines=lines,
            binary_image=mask,
            normalized_display=normalize_for_display(source_data),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _refine_endpoints(self, lines: np.ndarray, residual: np.ndarray) -> np.ndarray:
        p = self._p
        if p.endpoint_walk_sigma <= 0 or len(lines) == 0:
            return lines

        H, W = residual.shape
        out = lines.copy()
        for i, seg in enumerate(lines[:, 0, :]):
            x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 1.0:
                continue

            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux   # perpendicular unit vector

            floor1 = self._local_floor(residual, x1, y1, nx, ny, H, W)
            floor2 = self._local_floor(residual, x2, y2, nx, ny, H, W)

            nx1, ny1 = self._walk_endpoint(
                residual, x1, y1, -ux, -uy, floor1, p.endpoint_gap_tolerance, H, W
            )
            nx2, ny2 = self._walk_endpoint(
                residual, x2, y2,  ux,  uy, floor2, p.endpoint_gap_tolerance, H, W
            )
            out[i, 0, :] = [nx1, ny1, nx2, ny2]

        return out

    def _local_floor(
        self,
        residual: np.ndarray,
        x: int,
        y: int,
        nx: float,
        ny: float,
        H: int,
        W: int,
    ) -> float:
        """Return walk-out floor = endpoint_walk_sigma × local MAD noise."""
        inner = _OFF_GAP + 1
        outer = inner + _OFF_WIDTH - 1
        offsets = np.concatenate([
            np.arange(inner, outer + 1),
            np.arange(-outer, -inner + 1),
        ])
        cx = np.array([float(x)])
        cy = np.array([float(y)])
        off_px = _collect_pixels(residual, cx, cy, nx, ny, offsets, H, W)
        if len(off_px) >= 5:
            mad = float(np.median(np.abs(off_px - np.median(off_px))))
            sigma_local = _MAD_FACTOR * mad + 1e-6
        else:
            sigma_local = float(np.std(residual)) + 1e-6
        return self._p.endpoint_walk_sigma * sigma_local

    @staticmethod
    def _walk_endpoint(
        residual: np.ndarray,
        x: int,
        y: int,
        ux: float,
        uy: float,
        floor: float,
        gap_tolerance: int,
        H: int,
        W: int,
        max_steps: int = 200,
    ) -> tuple[int, int]:
        """
        Walk along (ux, uy) from (x, y), returning the furthest pixel
        where residual > floor.  Stops after gap_tolerance consecutive
        below-floor pixels or at the image boundary.
        """
        best_x, best_y = x, y
        gap = 0
        cx, cy = float(x), float(y)
        for _ in range(max_steps):
            cx += ux
            cy += uy
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                break
            if residual[iy, ix] > floor:
                best_x, best_y = ix, iy
                gap = 0
            else:
                gap += 1
                if gap > gap_tolerance:
                    break
        return best_x, best_y
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
