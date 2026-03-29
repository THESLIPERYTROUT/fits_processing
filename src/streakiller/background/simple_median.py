"""
Simple median background estimator.

Threshold = median + sigma_mult * stddev.  Fast, assumes a uniform background.
Works well for clean images where background occupies >95% of pixels.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from streakiller.config.schema import BackgroundParams

logger = logging.getLogger(__name__)


class SimpleMedianEstimator:
    """Background estimation via global median + stddev threshold."""

    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        """
        Parameters
        ----------
        data : float32 ndarray, shape (H, W)
        params : BackgroundParams

        Returns
        -------
        binary : uint8 ndarray, shape (H, W), values in {0, 255}
        """
        median = float(np.median(data))
        stddev = float(np.std(data))
        threshold = median + params.simple_median_sigma_mult * stddev

        binary = (data >= threshold).astype(np.uint8) * 255

        kernel_size = params.simple_median_sigma_mult  # reuse morph kernel param
        # Use dedicated morph kernel size from defaults
        k = 5  # matches original hard-coded value; exposed in BackgroundParams if needed
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        foreground_px = int(np.count_nonzero(binary))
        logger.debug(
            "simple_median: threshold=%.2f  foreground_pixels=%d",
            threshold, foreground_px,
        )
        return binary
