"""
Endpoint filter — removes detections whose endpoints are too close to an
already-accepted line's endpoints.

PERFORMANCE: the inner accepted-set scan is vectorised with NumPy so the
O(k) Python loop per line is replaced by a single batched distance call.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def endpoint_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Keep lines where no endpoint is within *params.endpoint_min_distance*
    of any endpoint of an already-accepted line.  When two lines conflict,
    the longer one wins.

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
    pts    = lines[:, 0, :].astype(float)   # (n, 4)
    starts = pts[:, :2]                      # (n, 2)
    ends   = pts[:, 2:]                      # (n, 2)
    lengths = np.hypot(ends[:, 0] - starts[:, 0], ends[:, 1] - starts[:, 1])

    n = len(lines)

    # Pre-allocate accepted-slot buffers (worst case: every line is accepted).
    acc_starts  = np.empty((n, 2))
    acc_ends    = np.empty((n, 2))
    acc_lengths = np.empty(n)
    acc_orig    = np.empty(n, dtype=int)   # original index in `lines`
    n_acc = 0

    for i in range(n):
        if n_acc == 0:
            acc_starts[0]  = starts[i]
            acc_ends[0]    = ends[i]
            acc_lengths[0] = lengths[i]
            acc_orig[0]    = i
            n_acc = 1
            continue

        s = starts[i]  # (2,)
        e = ends[i]    # (2,)
        as_ = acc_starts[:n_acc]   # (k, 2)
        ae_ = acc_ends[:n_acc]     # (k, 2)

        # Minimum endpoint distance from line i to each accepted line — vectorised.
        min_dists = np.minimum(
            np.minimum(
                np.hypot(s[0] - as_[:, 0], s[1] - as_[:, 1]),
                np.hypot(s[0] - ae_[:, 0], s[1] - ae_[:, 1]),
            ),
            np.minimum(
                np.hypot(e[0] - as_[:, 0], e[1] - as_[:, 1]),
                np.hypot(e[0] - ae_[:, 0], e[1] - ae_[:, 1]),
            ),
        )  # (k,)

        close = np.where(min_dists < threshold)[0]
        if close.size == 0:
            acc_starts[n_acc]  = s
            acc_ends[n_acc]    = e
            acc_lengths[n_acc] = lengths[i]
            acc_orig[n_acc]    = i
            n_acc += 1
        else:
            j = close[0]
            if lengths[i] > acc_lengths[j]:
                acc_starts[j]  = s
                acc_ends[j]    = e
                acc_lengths[j] = lengths[i]
                acc_orig[j]    = i

    if n_acc == 0:
        return np.empty((0, 1, 4), dtype=np.int32)
    return lines[acc_orig[:n_acc]]
