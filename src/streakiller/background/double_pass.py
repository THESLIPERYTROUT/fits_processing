"""
Double-pass inpainting background estimator.

The most sophisticated method — handles images with variable backgrounds,
crowded star fields, and cosmic rays.

PASS 1: Mask hot pixels with inpainting to reconstruct clean background.
PASS 2: Subtract background from original image, re-threshold on residual.

Bug fixed vs original:
  Original: highpass = background - hot_mask  (subtracts boolean 0/1 mask — wrong)
  Fixed:    highpass = image - background      (residual signal after bg removal)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from streakiller.config.schema import BackgroundParams

logger = logging.getLogger(__name__)


class DoublePassEstimator:
    """
    Background estimation using two-pass thresholding with inpainting.
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
        sigma_mult = params.double_pass_sigma_mult

        # PASS 1: Build inpainted background estimate ------------------- #
        median = float(np.median(data))
        stddev = float(np.std(data))
        threshold1 = median + sigma_mult * stddev

        masked = data.copy()
        hot_mask = masked > threshold1
        masked[hot_mask] = np.nan

        nan_mask = np.isnan(masked).astype(np.uint8)
        masked_filled = np.nan_to_num(masked, nan=median)
        background = cv2.inpaint(
            masked_filled.astype(np.float32),
            nan_mask,
            params.double_pass_inpaint_radius,
            cv2.INPAINT_TELEA,
        )

        # PASS 2: Threshold residual (image – background) --------------- #
        # BUG FIX: was `background - hot_mask` in original; correct is image - background
        highpass = data - background

        median2 = float(np.nanmedian(highpass))
        stddev2 = float(np.nanstd(highpass))
        threshold2 = median2 + sigma_mult * stddev2

        binary = (highpass >= threshold2).astype(np.uint8) * 255
        k = 5  # morphology kernel; matches original
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        foreground_px = int(np.count_nonzero(binary))
        logger.debug(
            "double_pass: thr1=%.2f  thr2=%.2f  foreground=%d",
            threshold1, threshold2, foreground_px,
        )
        return binary
