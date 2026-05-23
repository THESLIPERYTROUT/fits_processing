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
        if global_std < 1e-10:
            logger.info("PeakHoughDetector: image has no variance, returning empty")
            return RawDetection(
                lines=np.empty((0, 1, 4), dtype=np.int32),
                binary_image=np.zeros_like(source_data, dtype=np.uint8),
                normalized_display=normalize_for_display(source_data),
            )
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
