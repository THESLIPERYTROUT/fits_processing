"""
Length filter — keeps only lines that are close to the longest detected line.

Useful for removing short spurious segments after other filters have run.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def length_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines whose length >= *params.length_fraction* * modal_length.

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
    
    elif len(lines) <=2:
        # With 2 or fewer lines, we can't reliably estimate a modal length, so we keep them all.
        return lines.astype(np.int32, copy=False)

    coords = lines[:, 0, :]
    dx = coords[:, 2] - coords[:, 0]
    dy = coords[:, 3] - coords[:, 1]
    lengths = np.hypot(dx, dy)
    rounded_lengths = np.rint(lengths).astype(np.int32)
    modal_len = float(np.bincount(rounded_lengths).argmax())
    min_allowed = params.length_fraction * modal_len
    max_allowed = params.length_fraction * modal_len * 2

    kept = lines[(lengths >= min_allowed) & (lengths <= max_allowed)]

    if len(kept) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)
    return kept.astype(np.int32, copy=False)
