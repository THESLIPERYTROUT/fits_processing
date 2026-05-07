"""
FftCorrelationDetector — streak detection via PCA template extraction + FFT cross-correlation.

Algorithm:
1. Threshold the image to isolate bright objects (99th percentile by default).
2. Label connected components and measure their shapes via PCA covariance.
3. Select the most elongated, non-edge, thin component as the Master Template.
4. Cross-correlate the image with the template using FFT convolution.
5. Extract peaks from the correlation map as streak centres.
6. Compute streak endpoints from each centre using the template angle and length.

Ported from FourierStreakDetector (Streak_Detector.py) and adapted to the
streakiller RawDetection interface so it is a drop-in alternative to StreakDetector.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_closing, find_objects, label
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max

from streakiller.config.schema import FftDetectorParams
from streakiller.detection.detector import RawDetection
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


class FftCorrelationDetector:
    """
    Detects streaks using PCA template extraction + FFT cross-correlation.

    Unlike HoughLinesP, this method does not require a pre-computed binary mask:
    it derives its own mask from the raw image.  The *binary* argument accepted by
    detect() exists only for interface compatibility and is intentionally ignored.
    """

    def __init__(self, params: FftDetectorParams) -> None:
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
        """
        Detect streaks in *source_data* using FFT template matching.

        Parameters
        ----------
        binary : uint8 ndarray, shape (H, W)
            Pre-computed foreground mask — not used by this detector; the FFT
            method builds its own mask from *source_data*.
        source_data : float32 ndarray, shape (H, W)
            Normalised float image (zero-mean, unit-variance expected).
        min_line_length : float
            Minimum streak length in pixels.  A template shorter than this
            triggers an early return with zero detections.

        Returns
        -------
        RawDetection
            ``lines`` has shape (N, 1, 4) int32 (x1, y1, x2, y2), N == 0 when
            nothing is found.  ``binary_image`` holds the normalised correlation
            map (same visual role as the Hough binary mask).
        """
        p = self._p
        ih, iw = source_data.shape

        # ---- 1. Binary mask from high-percentile threshold ----
        # Sample every 4th pixel (16× speedup) for the percentile calculation
        thresh = np.percentile(source_data[::4, ::4], p.percentile_threshold)
        mask = source_data > thresh
        # binary_closing fills sub-pixel gaps so one streak isn't split into many components
        mask = binary_closing(mask, structure=np.ones((3, 3)))

        # ---- 2. PCA measurement of every connected component ----
        labeled, n_features = label(mask)
        if n_features == 0:
            logger.info("FftCorrelationDetector: no features found in binary mask")
            return self._empty(source_data)

        features = self._measure_features(labeled, n_features, source_data.shape)
        if not features:
            logger.info("FftCorrelationDetector: no features survived area filter")
            return self._empty(source_data)

        # ---- 3. Select the Master Template ----
        template_feature = self._select_template(features, ih, iw)
        if template_feature is None:
            logger.info("FftCorrelationDetector: no valid template candidate found")
            return self._empty(source_data)

        angle_deg, streak_length_px, template = self._extract_template(
            template_feature, source_data, ih, iw
        )

        if streak_length_px < min_line_length:
            logger.info(
                "FftCorrelationDetector: template length %.1f px < min_line_length %.1f px",
                streak_length_px,
                min_line_length,
            )
            return self._empty(source_data)

        logger.info(
            "FftCorrelationDetector: template angle=%.2f°  length=%.1f px  elongation=%.2f",
            angle_deg,
            streak_length_px,
            template_feature["elongation"],
        )

        # ---- 4. FFT cross-correlation ----
        # Flipping the template 180° converts convolution into cross-correlation
        corr = fftconvolve(source_data, template[::-1, ::-1], mode="same")

        # ---- 5. Extract correlation peaks ----
        med_c = np.median(corr)
        std_c = np.std(corr)
        threshold_abs = med_c + p.threshold_sigma * std_c

        peak_yx = peak_local_max(
            corr, min_distance=p.min_distance, threshold_abs=threshold_abs
        )
        if peak_yx.size == 0:
            logger.info("FftCorrelationDetector: no peaks above threshold")
            return self._empty(source_data, corr)

        logger.info("FftCorrelationDetector: %d raw peaks found", len(peak_yx))

        # ---- 6. Build and filter line endpoints ----
        lines_arr = self._peaks_to_lines(
            peak_yx, corr, angle_deg, streak_length_px, ih, iw
        )
        logger.info("FftCorrelationDetector: %d lines accepted", len(lines_arr))

        corr_u8 = normalize_for_display(corr)
        return RawDetection(
            lines=lines_arr,
            binary_image=corr_u8,
            normalized_display=normalize_for_display(source_data),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _measure_features(
        self, labeled: np.ndarray, n_features: int, shape: tuple[int, int]
    ) -> list[dict]:
        """Run PCA on every labeled component; return feature dicts."""
        p = self._p
        features = []
        slices = find_objects(labeled)

        for i, slc in enumerate(slices):
            if slc is None:
                continue
            label_id = i + 1
            local_coords = np.argwhere(labeled[slc] == label_id)
            if len(local_coords) < p.min_template_area:
                continue

            y0, x0 = slc[0].start, slc[1].start
            coords = local_coords + np.array([y0, x0])

            cov = np.cov(coords.T)
            if np.isnan(cov).any():
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals[0] <= 1e-6:
                continue

            elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            features.append({
                "coords": coords,
                "elongation": elongation,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "bbox": (y_min, y_max, x_min, x_max),
            })

        # Sort longest-first so the first valid candidate is the champion template
        features.sort(key=lambda f: f["eigvals"][1], reverse=True)
        return features

    def _select_template(
        self, features: list[dict], ih: int, iw: int
    ) -> dict | None:
        """Return the best template candidate, or None if none qualify."""
        p = self._p
        m = p.template_edge_margin
        for f in features:
            y_min, y_max, x_min, x_max = f["bbox"]
            if x_min < m or x_max > iw - m or y_min < m or y_max > ih - m:
                continue
            if f["elongation"] < p.min_elongation:
                continue
            if np.sqrt(f["eigvals"][0]) > p.max_width_std:
                continue
            return f
        return None

    def _extract_template(
        self,
        feature: dict,
        source_data: np.ndarray,
        ih: int,
        iw: int,
    ) -> tuple[float, float, np.ndarray]:
        """
        Return (angle_deg, streak_length_px, template_patch) for *feature*.

        The template patch is mean-normalised so the FFT correlation is unbiased.
        streak_length_px is derived from the PCA primary variance:
            length ≈ sqrt(12 * var_principal)
        This comes from the fact that for a uniform line of length L,
        the variance along the long axis is L² / 12.
        """
        p = self._p
        eigvals, eigvecs = feature["eigvals"], feature["eigvecs"]

        # Principal eigenvector → angle
        v = eigvecs[:, -1]
        angle_deg = np.degrees(np.arctan2(v[0], v[1]))

        # Streak length from primary variance
        streak_length_px = np.sqrt(12.0 * eigvals[1])

        # Cut out the template with padding
        y_min, y_max, x_min, x_max = feature["bbox"]
        pad = p.template_padding
        ys = max(0, y_min - pad)
        ye = min(ih, y_max + pad)
        xs = max(0, x_min - pad)
        xe = min(iw, x_max + pad)
        patch = source_data[ys:ye, xs:xe].copy()

        std_p = np.std(patch)
        if std_p > 1e-6:
            patch = (patch - np.mean(patch)) / std_p
        else:
            patch = patch - np.mean(patch)

        return angle_deg, streak_length_px, patch

    def _peaks_to_lines(
        self,
        peak_yx: np.ndarray,
        corr: np.ndarray,
        angle_deg: float,
        streak_length_px: float,
        ih: int,
        iw: int,
    ) -> np.ndarray:
        """Convert (y, x) correlation peaks to an (N, 1, 4) int32 line array."""
        p = self._p
        half_len = streak_length_px / 2.0
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        dx = half_len * cos_a
        dy = half_len * sin_a
        m = p.streak_edge_margin

        corr_vals = corr[peak_yx[:, 0], peak_yx[:, 1]]
        max_corr = float(corr_vals.max())

        lines = []
        for (y_c, x_c), corr_val in zip(peak_yx, corr_vals):
            # Reject faint detections below the prominence threshold
            if float(corr_val) < max_corr * p.prominence_fraction:
                continue

            x1, y1 = x_c - dx, y_c - dy
            x2, y2 = x_c + dx, y_c + dy

            # Edge filter
            if (
                min(x1, x2) < m
                or max(x1, x2) > iw - m
                or min(y1, y2) < m
                or max(y1, y2) > ih - m
            ):
                continue

            lines.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])

        if not lines:
            return np.empty((0, 1, 4), dtype=np.int32)
        return np.array(lines, dtype=np.int32).reshape(-1, 1, 4)

    def _empty(
        self, source_data: np.ndarray, corr: np.ndarray | None = None
    ) -> RawDetection:
        display = corr if corr is not None else source_data
        return RawDetection(
            lines=np.empty((0, 1, 4), dtype=np.int32),
            binary_image=normalize_for_display(display),
            normalized_display=normalize_for_display(source_data),
        )
