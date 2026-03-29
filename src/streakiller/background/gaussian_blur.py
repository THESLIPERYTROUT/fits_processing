"""
Gaussian-blur high-pass background estimator.

Removes low-frequency background via Gaussian blur subtraction, then applies
adaptive MAD-based thresholding with a k-value ladder to handle faint streaks.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from streakiller.config.schema import BackgroundParams
from streakiller.config.defaults import GAUSSIAN_MIN_BINARY_PIXELS, MAD_NORMALIZATION_FACTOR

logger = logging.getLogger(__name__)


class GaussianBlurEstimator:
    """
    Background estimation using Gaussian high-pass filtering.

    1. Percentile-clip to suppress hot pixels / bright stars.
    2. Subtract a Gaussian-blurred background to isolate high-frequency features.
    3. Compute gradient magnitude to enhance edges.
    4. Apply MAD-based adaptive threshold, trying a k-value ladder until enough
       foreground pixels are found for the Hough transform.
    """

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
        # 1. Percentile clip to tame hot pixels and bright stars
        p1, p99 = np.percentile(data, (2, 99.8))
        clipped = np.clip(data, p1, p99)

        # 2. Estimate and subtract background (top-hat filter)
        ksize = params.gaussian_kernel_size
        if ksize % 2 == 0:
            ksize += 1  # OpenCV requires odd kernel size
        background = cv2.GaussianBlur(clipped, (ksize, ksize), 0)
        highpass = clipped - background

        # 3. Slight blur to improve SNR before thresholding
        hp_blur = cv2.GaussianBlur(highpass, (5, 5), 0)

        # 4. Robust sigma estimate via MAD
        med = float(np.median(hp_blur))
        mad = float(np.median(np.abs(hp_blur - med))) + 1e-6
        sigma = MAD_NORMALIZATION_FACTOR * mad

        # 5. Try k-value ladder — stop when enough pixels are foreground
        binary = None
        for k in params.gaussian_sigma_ladder:
            thr = med + k * sigma
            candidate = (hp_blur >= thr).astype(np.uint8) * 255
            candidate = cv2.morphologyEx(
                candidate,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
            foreground_px = int(np.count_nonzero(candidate))
            logger.debug("gaussian_blur: k=%.1f  threshold=%.4f  foreground=%d", k, thr, foreground_px)
            binary = candidate
            if foreground_px >= GAUSSIAN_MIN_BINARY_PIXELS:
                logger.debug("gaussian_blur: accepted k=%.1f", k)
                return binary

        logger.warning(
            "gaussian_blur: no k-value produced >= %d foreground pixels; using best-effort binary",
            GAUSSIAN_MIN_BINARY_PIXELS,
        )
        return binary  # type: ignore[return-value]
