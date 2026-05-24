"""
PeakHoughDetector — per-row polynomial baseline + find_peaks → HoughLinesP,
with local-noise-aware endpoint walk-out.

Core algorithm (ported from devtools/plotter.py):
1. Percentile-clip the image.
2. For each row, fit a polynomial to binned medians of a Gaussian-smoothed copy
   (per_row_median_background) to capture the slowly-varying row baseline.
3. For each row find peaks in (smoothed_row − baseline) above
   threshold_sigma × std(residual_row) using scipy.signal.find_peaks.
   This produces a truly sparse set of locally-prominent pixels — one per
   detected local maximum — rather than everything above a global cut.
4. Assemble a sparse binary mask from those peak positions, dilate it slightly
   for HoughLinesP connectivity, then run HoughLinesP.
5. Walk outward from each Hough endpoint through the continuous per-row residual
   using a locally-estimated noise floor (perpendicular MAD strip, same technique
   as SnrApertureEstimator) to recover faint streak tails below the peak threshold.
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
from streakiller.snr.aperture import _collect_pixels

logger = logging.getLogger(__name__)

_MAD_FACTOR = 1.4826
_OFF_GAP    = 3   # pixels from streak centre to start of background band
_OFF_WIDTH  = 7   # width of each background band


# ------------------------------------------------------------------ #
# Module-level helpers (ported from devtools/plotter.py)              #
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


def _per_row_median_background(
    image: np.ndarray,
    bins: int,
    degree: int,
    smooth_sigma: float,
) -> np.ndarray:
    """Per-row polynomial baseline fitted to Gaussian-smoothed bin medians."""
    height, width = image.shape
    bins   = max(2, min(bins, width))
    degree = max(0, min(degree, bins - 1))

    smoothed = gaussian_filter1d(image.astype(float), sigma=smooth_sigma, axis=1)
    x_full   = np.arange(width)
    background = np.zeros_like(smoothed, dtype=float)

    for row_index in range(height):
        xs, ys = _row_median_points(smoothed, row_index=row_index, bins=bins)
        coeffs = np.polyfit(xs, ys, deg=degree)
        background[row_index, :] = np.polyval(coeffs, x_full)

    return background


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
        coeffs  = np.polyfit(xs, ys, deg=degree)
        curve   = np.polyval(coeffs, np.arange(image.shape[1]))

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
    → sparse binary mask → HoughLinesP, with local-noise-aware endpoint walk-out.

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

        # 1. Percentile clip
        vmin, vmax = np.percentile(source_data, (p.clip_percentile_low, p.clip_percentile_high))
        clipped = np.clip(source_data, vmin, vmax).astype(np.float32)

        # 2. Walk-out residual: per-row median subtraction.  For a diagonal streak
        # only one pixel per row is elevated, so the row median is unaffected and
        # the residual preserves the full per-pixel streak amplitude.
        row_medians = np.median(clipped, axis=1, keepdims=True)
        walk_residual = (clipped - row_medians).astype(np.float32)

        # 3. Collect per-row signal peaks and build sparse binary mask
        peak_xs, peak_ys = _collect_signal_peaks(clipped, p)
        logger.info("PeakHoughDetector: %d signal peaks collected", len(peak_xs))

        sparse = _binary_from_peaks(clipped.shape, peak_xs, peak_ys)

        # 4. Dilate slightly so adjacent peak pixels form connected blobs for Hough
        hough_binary = cv2.dilate(
            sparse.astype(np.uint8) * 255,
            np.ones((3, 3), dtype=np.uint8),
        )

        # 5. HoughLinesP on the sparse dilated mask
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
            lines = raw
            logger.info("PeakHoughDetector: HoughLinesP found %d raw lines", len(lines))

        # 6. Endpoint walk-out through continuous walk residual
        lines = self._refine_endpoints(lines, walk_residual)

        return RawDetection(
            lines=lines,
            binary_image=hough_binary,
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

            # Pre-compute floors at the Hough endpoints (before any snapping).
            floor1 = self._local_floor(residual, x1, y1, nx, ny, H, W)
            floor2 = self._local_floor(residual, x2, y2, nx, ny, H, W)

            # Snap Hough endpoints onto the streak centerline, but only when the
            # candidate pixel is clearly above the local noise floor.  Without this
            # guard the snap can move to a bright noise pixel that is even further
            # from the centerline, causing the walk to immediately lose the streak.
            sx1, sy1 = self._snap_to_peak(residual, x1, y1, nx, ny, H, W, min_val=floor1)
            sx2, sy2 = self._snap_to_peak(residual, x2, y2, nx, ny, H, W, min_val=floor2)

            # Recompute floor at the snapped position (better noise estimate on streak).
            if (sx1, sy1) != (x1, y1):
                floor1 = self._local_floor(residual, sx1, sy1, nx, ny, H, W)
            if (sx2, sy2) != (x2, y2):
                floor2 = self._local_floor(residual, sx2, sy2, nx, ny, H, W)

            nx1, ny1 = self._walk_endpoint(
                residual, sx1, sy1, -ux, -uy, floor1, p.endpoint_gap_tolerance, H, W
            )
            nx2, ny2 = self._walk_endpoint(
                residual, sx2, sy2,  ux,  uy, floor2, p.endpoint_gap_tolerance, H, W
            )
            out[i, 0, :] = [nx1, ny1, nx2, ny2]

        return out

    @staticmethod
    def _snap_to_peak(
        residual: np.ndarray,
        x: int,
        y: int,
        nx: float,
        ny: float,
        H: int,
        W: int,
        search_half: int = 6,
        min_val: float = 0.0,
    ) -> tuple[int, int]:
        """
        Return the highest-residual pixel within ±search_half steps perpendicular
        to the streak, provided it exceeds min_val.  Falls back to (x, y) if no
        candidate qualifies, preventing a snap to a random noise peak.
        """
        best_x, best_y = x, y
        best_val = residual[min(max(y, 0), H - 1), min(max(x, 0), W - 1)]
        for d in range(-search_half, search_half + 1):
            px = int(round(x + d * nx))
            py = int(round(y + d * ny))
            if 0 <= px < W and 0 <= py < H:
                v = residual[py, px]
                if v > best_val:
                    best_val = v
                    best_x, best_y = px, py
        if best_val < min_val:
            return x, y
        return best_x, best_y

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
        """Walk-out floor = endpoint_walk_sigma × local MAD noise from perpendicular strip."""
        inner = _OFF_GAP + 1
        outer = inner + _OFF_WIDTH - 1
        offsets = np.concatenate([
            np.arange(inner, outer + 1),
            np.arange(-outer, -inner + 1),
        ])
        off_px = _collect_pixels(
            residual,
            np.array([float(x)]),
            np.array([float(y)]),
            nx, ny, offsets, H, W,
        )
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
        Walk along (ux, uy) from (x, y), returning the furthest pixel where
        the local neighborhood is above floor.  Stops after gap_tolerance
        consecutive below-floor steps or at the image boundary.

        At each step the 4-connected neighbourhood (centre + N/S/E/W) is
        checked; using the neighbourhood maximum makes the walk robust to
        ±1 px off-centre tracking caused by integer rounding on diagonal paths.
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
            # 4-connected neighbourhood max to catch the streak when the walk
            # is 1 px off the centreline due to diagonal discretisation.
            nbr_max = residual[iy, ix]
            for dx2, dy2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                px2, py2 = ix + dx2, iy + dy2
                if 0 <= px2 < W and 0 <= py2 < H:
                    v = residual[py2, px2]
                    if v > nbr_max:
                        nbr_max = v
            if nbr_max > floor:
                best_x, best_y = ix, iy
                gap = 0
            else:
                gap += 1
                if gap > gap_tolerance:
                    break
        return best_x, best_y
