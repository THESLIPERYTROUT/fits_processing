
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


MIN_LINES_FOR_ANGLE_FILTER = 5


def _line_orientation_deg(line: np.ndarray) -> float:
    """Return line orientation in [0, 180), ignoring endpoint order."""
    x1, y1, x2, y2 = line[0]
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return angle % 180.0


def _orientation_diff_deg(a: float, b: float) -> float:
    """Return the smallest difference between two orientations in degrees."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def angle_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep the majority orientation cluster and discard angular outliers.

    Lines are treated as undirected segments, so 0 and 180 degrees are
    considered the same orientation. For sparse detections this filter is
    skipped to avoid throwing away plausible streaks based on too little data.

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

    if len(lines) < MIN_LINES_FOR_ANGLE_FILTER:
        return np.array(lines, dtype=np.int32, copy=True)

    orientation_tolerance_deg = params.angle_min_diff_deg
    orientations = np.array([_line_orientation_deg(line) for line in lines], dtype=np.float32)
    cluster_sizes: list[int] = []

    for candidate_orientation in orientations:
        cluster_size = sum(
            _orientation_diff_deg(candidate_orientation, orientation) <= orientation_tolerance_deg
            for orientation in orientations
        )
        cluster_sizes.append(cluster_size)

    dominant_orientation = float(orientations[int(np.argmax(cluster_sizes))])
    keep_mask = np.array(
        [
            _orientation_diff_deg(orientation, dominant_orientation) <= orientation_tolerance_deg
            for orientation in orientations
        ],
        dtype=bool,
    )
    return np.array(lines[keep_mask], dtype=np.int32, copy=True)
