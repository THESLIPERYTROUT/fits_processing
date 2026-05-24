"""
Midpoint filter — removes near-duplicate detections by midpoint proximity.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def midpoint_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines whose midpoint is at least *params.midpoint_min_distance* pixels
    away from every already-accepted line's midpoint.

    Parameters
    ----------
    lines : ndarray, shape (N, 1, 4), dtype int-like
    params : FilterParams

    Returns
    -------
    filtered : ndarray, shape (M, 1, 4)  where M <= N
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)

    threshold = params.midpoint_min_distance
    accepted: list[np.ndarray] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        length = np.hypot(x2 - x1, y2 - y1)

        close_idx = None
        close_length = None

        for i, kept in enumerate(accepted):
            fx1, fy1, fx2, fy2 = kept[0]
            fmid_x = (fx1 + fx2) / 2.0
            fmid_y = (fy1 + fy2) / 2.0

            if np.hypot(mid_x - fmid_x, mid_y - fmid_y) < threshold:
                close_idx = i
                close_length = np.hypot(fx2 - fx1, fy2 - fy1)
                break

        if close_idx is None:
            accepted.append(line)
        elif length > close_length:
            accepted[close_idx] = line


    if not accepted:
        return np.empty((0, 1, 4), dtype=np.int32)
    return np.array(accepted, dtype=np.int32)
