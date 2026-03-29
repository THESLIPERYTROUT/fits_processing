"""
Image normalisation utilities for display and Hough preprocessing.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.defaults import NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH


def normalize_for_display(
    data: np.ndarray,
    p_low: float = NORM_PERCENTILE_LOW,
    p_high: float = NORM_PERCENTILE_HIGH,
) -> np.ndarray:
    """
    Percentile-clip and scale to uint8 [0, 255].

    Parameters
    ----------
    data : float ndarray, shape (H, W)
    p_low, p_high : percentile bounds for clipping

    Returns
    -------
    uint8 ndarray, shape (H, W)
    """
    lo, hi = np.percentile(data, (p_low, p_high))
    clipped = np.clip(data, lo, hi)
    if hi > lo:
        scaled = (clipped - lo) / (hi - lo) * 255.0
    else:
        scaled = np.zeros_like(clipped)
    return scaled.astype(np.uint8)
