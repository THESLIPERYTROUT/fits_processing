"""
Length filter - removes segments that are clearly too short or suspiciously long.

The reference length is the binned mode of detected segment lengths: the center
of the densest length cluster. Using a binned mode is more useful than an exact
mode because Hough lengths are continuous enough that exact duplicates are not
guaranteed.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def length_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines between ``length_fraction * modal_length`` and
    ``max_length_factor * modal_length``.

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

    modal_len = _modal_length(lengths)
    min_allowed = params.length_fraction * modal_len
    max_allowed = params.max_length_factor * modal_len

    kept = lines[(lengths >= min_allowed) & (lengths <= max_allowed)]

    if len(kept) == 0:
        return lines.astype(np.int32, copy=False)
    return kept.astype(np.int32, copy=False)


def _modal_length(lengths: np.ndarray) -> float:
    """
    Estimate the dominant line length.

    Exact mode is only useful when lengths repeat after pixel rounding. When
    they do not, use a Freedman-Diaconis histogram and return the median length
    inside the densest bin.
    """
    if len(lengths) == 1:
        return float(lengths[0])

    rounded = np.rint(lengths).astype(int)
    unique_lengths, counts = np.unique(rounded, return_counts=True)
    if counts.max() > 1:
        return float(unique_lengths[np.argmax(counts)])

    q25, q75 = np.percentile(lengths, [25, 75])
    iqr = float(q75 - q25)
    bin_width = 2 * iqr / np.cbrt(len(lengths)) if iqr > 0 else 0.0
    if bin_width <= 0:
        bin_width = max(float(np.std(lengths)), 1.0)

    length_min = float(lengths.min())
    length_max = float(lengths.max())
    bins = max(1, int(np.ceil((length_max - length_min) / bin_width)))
    counts, edges = np.histogram(lengths, bins=bins)
    if counts.max() <= 1:
        return float(np.median(lengths))

    best_bin = int(np.argmax(counts))

    in_bin = (lengths >= edges[best_bin]) & (lengths <= edges[best_bin + 1])
    if np.any(in_bin):
        return float(np.median(lengths[in_bin]))
    return float((edges[best_bin] + edges[best_bin + 1]) / 2)
