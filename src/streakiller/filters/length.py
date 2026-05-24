"""
Length filter - removes segments that are clearly too short or suspiciously long.

The reference length is the dominant peak of the length histogram. In a
multimodal distribution (e.g. noise fragments at one scale and real streaks at
another), the longest significant peak is chosen so the filter anchors to the
actual streak population rather than the noise cluster.

Using a binned histogram rather than exact mode avoids an artifact where
axis-aligned streaks that share the same integer pixel span create a false
exact-tie "mode" that does not correspond to the densest cluster.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams

# A peak must reach at least this fraction of the tallest peak's count to be
# treated as a significant mode.  Prevents isolated far-tail detections from
# being selected as the "longest mode".
_PEAK_SIGNIFICANCE = 0.30


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
    Return the length at the dominant streak cluster.

    Builds a Freedman-Diaconis histogram, finds all local peaks, and returns
    the median length inside the longest *significant* peak bin.  In a
    multimodal distribution (e.g. short noise fragments mixed with real
    streaks) this selects the streak population rather than the noise.
    """
    if len(lengths) == 1:
        return float(lengths[0])

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

    peaks = _histogram_peaks(counts)
    if not peaks:
        peaks = [int(np.argmax(counts))]

    tallest = max(counts[i] for i in peaks)
    significant = [i for i in peaks if counts[i] >= tallest * _PEAK_SIGNIFICANCE]
    best_bin = significant[-1]  # rightmost = longest significant peak

    in_bin = (lengths >= edges[best_bin]) & (lengths <= edges[best_bin + 1])
    if np.any(in_bin):
        return float(np.median(lengths[in_bin]))
    return float((edges[best_bin] + edges[best_bin + 1]) / 2)


def _histogram_peaks(counts: np.ndarray) -> list[int]:
    """Return indices of local maxima in a histogram count array."""
    n = len(counts)
    peaks = []
    for i in range(n):
        left = int(counts[i - 1]) if i > 0 else -1
        right = int(counts[i + 1]) if i < n - 1 else -1
        if counts[i] > left and counts[i] > right:
            peaks.append(i)
    return peaks
