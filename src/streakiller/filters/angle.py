"""
Angle filter — removes near-duplicate detections by line orientation.

BUG FIX vs original (streakprocessing.py:261):
  Original kept lines whose angle differed by MORE than the threshold — the
  opposite of deduplication.  The intent is to *discard* lines that are
  within min_angle_diff degrees of an already-accepted line (i.e. they're
  essentially parallel duplicates pointing the same direction).

  Fixed condition: `abs(angle - fangle) < min_angle_diff` marks a line as
  a duplicate to be dropped.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def angle_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines whose angle differs by at least *params.angle_min_diff_deg*
    degrees from every already-accepted line.

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

    threshold = params.angle_min_diff_deg
    accepted: list[np.ndarray] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        is_duplicate = False
        for kept in accepted:
            fx1, fy1, fx2, fy2 = kept[0]
            fangle = np.degrees(np.arctan2(fy2 - fy1, fx2 - fx1))
            # BUG FIX: was `> threshold` (kept all different-angle lines, never deduped)
            if abs(angle - fangle) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            accepted.append(line)

    if not accepted:
        return np.empty((0, 1, 4), dtype=np.int32)
    return np.array(accepted, dtype=np.int32)
