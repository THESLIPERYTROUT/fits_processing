"""
Endpoint filter — removes detections whose endpoints are too close to an
already-accepted line's endpoints.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def endpoint_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines where all four endpoint-to-endpoint distances to every
    already-accepted line are >= *params.endpoint_min_distance*.

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

    threshold = params.endpoint_min_distance
    accepted: list[np.ndarray] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        too_close = False
        for kept in accepted:
            fx1, fy1, fx2, fy2 = kept[0]
            distances = [
                np.hypot(x1 - fx1, y1 - fy1),
                np.hypot(x1 - fx2, y1 - fy2),
                np.hypot(x2 - fx1, y2 - fy1),
                np.hypot(x2 - fx2, y2 - fy2),
            ]
            if any(d < threshold for d in distances):
                too_close = True
                break
        if not too_close:
            accepted.append(line)

    if not accepted:
        return np.empty((0, 1, 4), dtype=np.int32)
    return np.array(accepted, dtype=np.int32)
