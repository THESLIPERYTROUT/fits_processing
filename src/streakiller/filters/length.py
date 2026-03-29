"""
Length filter — keeps only lines that are close to the longest detected line.

Useful for removing short spurious segments after other filters have run.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def length_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines whose length >= *params.length_fraction* * max_length.

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

    lengths = np.array([
        np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1])
        for line in lines
    ])
    max_len = float(np.max(lengths))
    min_allowed = params.length_fraction * max_len

    kept = [line for line, ln in zip(lines, lengths) if ln >= min_allowed]

    if not kept:
        return np.empty((0, 1, 4), dtype=np.int32)
    return np.array(kept, dtype=np.int32)
