"""
Per-row median-curve background estimator.

Fits a smooth polynomial background curve across every image row from binned
median samples, subtracts that model, then thresholds the residual.  This is
useful for images whose background varies mostly left-to-right, while a small
vertical sampling window keeps single hot pixels and thin streaks from being
absorbed into the background model.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, label, median_filter, sum as ndi_sum

from streakiller.config.defaults import MAD_NORMALIZATION_FACTOR
from streakiller.config.schema import BackgroundParams

logger = logging.getLogger(__name__)


class PerRowMedianCurveEstimator:
    """
    Background estimation by fitting a median-derived curve for each row.

    Algorithm:
    1. Smooth horizontally to reduce pixel-scale noise before sampling.
    2. For each output row, split the image width into bins and take one median
       per bin from a small vertical window centered on that row.
    3. Fit a polynomial through those median samples to build a row background.
    4. Subtract the background, median-filter the residual, and threshold with
       a robust MAD-based sigma estimate.
    5. Remove tiny connected components that are likely isolated hot pixels.
    """

    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        H, W = data.shape
        if H == 0 or W == 0:
            return np.zeros_like(data, dtype=np.uint8)

        data_float = data.astype(np.float32, copy=False)
        background = self._build_background(data_float, params)
        residual = data_float - background

        filter_size = params.per_row_median_filter_size
        if filter_size > 1:
            residual_for_threshold = median_filter(residual, size=(1, filter_size))
        else:
            residual_for_threshold = residual

        global_med = float(np.median(residual_for_threshold))
        global_mad = (
            float(np.median(np.abs(residual_for_threshold - global_med))) + 1e-6
        )
        global_sigma = MAD_NORMALIZATION_FACTOR * global_mad

        row_meds = np.median(residual_for_threshold, axis=1, keepdims=True)
        row_mads = (
            np.median(np.abs(residual_for_threshold - row_meds), axis=1, keepdims=True)
            + 1e-6
        )
        row_sigmas = MAD_NORMALIZATION_FACTOR * row_mads
        sigma = np.maximum(row_sigmas, global_sigma)
        threshold = global_med + params.per_row_median_sigma_mult * sigma

        binary = (residual_for_threshold >= threshold).astype(np.uint8) * 255

        binary = self._remove_small_components(
            binary,
            params.per_row_median_min_component_pixels,
        )

        k = params.per_row_median_morph_kernel
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        binary = self._remove_small_components(
            binary,
            params.per_row_median_min_component_pixels,
        )

        logger.debug(
            "per_row_median_curve: bins=%d degree=%d row_window=%d "
            "sigma_mult=%.2f threshold=%.4f foreground_pixels=%d",
            params.per_row_median_bins,
            params.per_row_median_degree,
            params.per_row_median_row_window,
            params.per_row_median_sigma_mult,
            float(np.median(threshold)),
            int(np.count_nonzero(binary)),
        )
        return binary

    def _build_background(
        self,
        data: np.ndarray,
        params: BackgroundParams,
    ) -> np.ndarray:
        H, W = data.shape
        bins = max(2, min(params.per_row_median_bins, W))
        degree = max(0, min(params.per_row_median_degree, bins - 1))
        row_window = max(1, params.per_row_median_row_window)
        if row_window % 2 == 0:
            row_window += 1

        smoothed = gaussian_filter1d(
            data,
            sigma=params.per_row_median_smooth_sigma,
            axis=1,
        ).astype(np.float32)

        x_edges = np.linspace(0, W, bins + 1, dtype=int)
        x_centers = np.array(
            [(start + end - 1) / 2 for start, end in zip(x_edges[:-1], x_edges[1:])],
            dtype=np.float32,
        )
        x_full = np.arange(W, dtype=np.float32)
        background = np.zeros((H, W), dtype=np.float32)

        half_window = row_window // 2
        for y in range(H):
            y0 = max(0, y - half_window)
            y1 = min(H, y + half_window + 1)
            medians = []

            for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
                patch = smoothed[y0:y1, x0:x1]
                medians.append(float(np.median(patch)))

            coeffs = np.polyfit(x_centers, np.asarray(medians), deg=degree)
            background[y, :] = np.polyval(coeffs, x_full)

        return background

    @staticmethod
    def _remove_small_components(binary: np.ndarray, min_pixels: int) -> np.ndarray:
        if min_pixels <= 1:
            return binary

        labeled, n_features = label(binary > 0)
        if n_features == 0:
            return binary

        sizes = ndi_sum(
            binary > 0,
            labeled,
            index=np.arange(1, n_features + 1),
        )
        keep_labels = np.where(sizes >= min_pixels)[0] + 1
        return np.isin(labeled, keep_labels).astype(np.uint8) * 255
