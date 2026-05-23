"""
Length filter — removes segments that are clearly too short or suspiciously long.

Lower floor  : length_fraction * median  — drops short noise fragments.
Upper cap    : max_length_factor * median — drops merged/overlapping detections
               that HoughLinesP bridges into one abnormally long segment.

The two parameters are independent so each concern can be tuned separately.
Median is used as the reference instead of mode for stability with sparse counts.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def length_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines between ``length_fraction * median`` and
    ``max_length_factor * median``.

    If every line would be dropped, the original set is returned unchanged.

    Parameters
    ----------
    lines : ndarray, shape (N, 1, 4)
    params : FilterParams

    Returns
    -------
    filtered : ndarray, shape (M, 1, 4)  where M <= N
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)

    coords = lines[:, 0, :]
    dx = coords[:, 2] - coords[:, 0]
    dy = coords[:, 3] - coords[:, 1]
    lengths = np.hypot(dx, dy)

    median_len = float(np.median(lengths))
    min_allowed = params.length_fraction * median_len
    max_allowed = params.max_length_factor * median_len

    kept = lines[(lengths >= min_allowed) & (lengths <= max_allowed)]

    if len(kept) == 0:
        return lines.astype(np.int32, copy=False)
    return kept.astype(np.int32, copy=False)
