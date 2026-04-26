"""
Length filter — keeps only lines that are close to the modal streak length.

Useful for removing segments that are much shorter or longer than the dominant
streak family after other filters have run.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def length_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines within a symmetric band around the modal streak length.

    For example, with ``params.length_fraction == 0.90``, lines are kept when
    they fall within +/- 10% of the modal length.

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
    
    elif len(lines) <= 15:
        # With 5 or fewer lines, we can't reliably estimate a modal length, so baseline from max
        coords = lines[:, 0, :]
        dx = coords[:, 2] - coords[:, 0]
        dy = coords[:, 3] - coords[:, 1]
        lengths = np.hypot(dx, dy)
        max_len = lengths.max()
        kept = lines[lengths >= params.length_fraction * max_len]
        if len(kept) == 0:  
            return lines.astype(np.int32, copy=False)
        return kept.astype(np.int32, copy=False)

    coords = lines[:, 0, :]
    dx = coords[:, 2] - coords[:, 0]
    dy = coords[:, 3] - coords[:, 1]
    lengths = np.hypot(dx, dy)
    rounded_lengths = np.rint(lengths).astype(np.int32)
    modal_len = float(np.bincount(rounded_lengths).argmax())
    min_allowed = params.length_fraction * modal_len
    max_allowed = (2.0 - params.length_fraction) * modal_len

    kept = lines[(lengths >= min_allowed) & (lengths <= max_allowed)]

    if len(kept) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)
    return kept.astype(np.int32, copy=False)
