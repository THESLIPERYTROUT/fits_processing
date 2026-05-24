"""
On-streak duplicate filter — removes detections whose endpoints fall on the
infinite line spanned by an already-accepted (longer) detection.

HoughLinesP routinely fires multiple overlapping line segments along the same
physical streak.  The length filter struggles with this because small duplicate
detections swell the population and drag the modal-length estimate down, causing
the filter to discard good long-streak candidates.

This filter runs before the length filter to remove those short duplicates first:

  For each candidate line C (processed longest-first):
    For each already-accepted line A:
      Compute the perpendicular distance from each of C's endpoints to the
      infinite line through A's endpoints.
      If *either* endpoint is within ``on_streak_proximity_px``, C is a
      duplicate shadow of A → reject C.

  If accepting C would leave zero lines, the original set is returned unchanged
  (same fallback contract as the length filter).

Parameters
----------
on_streak_proximity_px : float
    Maximum perpendicular distance (pixels) from an endpoint to an accepted
    line's infinite span before the detection is treated as a duplicate.
    Default 3 px.  Raise to catch looser Hough responses; lower to avoid
    false-positive removal of genuinely separate nearby features.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def on_streak_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Remove duplicate HoughLinesP detections that lie on an already-accepted
    line's infinite span.

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
    if len(lines) == 1:
        return lines.astype(np.int32, copy=False)

    proximity = params.on_streak_proximity_px
    pts = lines[:, 0, :].astype(float)      # (n, 4)
    starts = pts[:, :2]                      # (n, 2)  — A endpoints
    ends   = pts[:, 2:]                      # (n, 2)  — B endpoints
    dx = ends[:, 0] - starts[:, 0]          # (n,)
    dy = ends[:, 1] - starts[:, 1]          # (n,)
    lengths = np.hypot(dx, dy)              # (n,)

    # Process longest lines first so they win conflicts.
    order = np.argsort(lengths)[::-1]

    # Accepted-slot buffers (worst case: every line accepted).
    n = len(lines)
    acc_ax  = np.empty(n)   # accepted start x
    acc_ay  = np.empty(n)   # accepted start y
    acc_dx  = np.empty(n)   # accepted direction x
    acc_dy  = np.empty(n)   # accepted direction y
    acc_len = np.empty(n)   # accepted lengths (for valid-line guard)
    acc_orig = np.empty(n, dtype=int)
    n_acc = 0

    for i_pos in range(len(order)):
        i = int(order[i_pos])

        if n_acc == 0:
            acc_ax[0]   = starts[i, 0]
            acc_ay[0]   = starts[i, 1]
            acc_dx[0]   = dx[i]
            acc_dy[0]   = dy[i]
            acc_len[0]  = lengths[i]
            acc_orig[0] = i
            n_acc = 1
            continue

        sx, sy = starts[i, 0], starts[i, 1]
        ex, ey = ends[i, 0],   ends[i, 1]

        # Slice accepted buffers.
        aax = acc_ax[:n_acc]
        aay = acc_ay[:n_acc]
        adx = acc_dx[:n_acc]
        ady = acc_dy[:n_acc]
        alen = acc_len[:n_acc]
        valid = alen > 1e-9

        safe_len = np.where(valid, alen, 1.0)

        # Perpendicular distance from candidate's start endpoint to each accepted line.
        # d = |adx*(sy - aay) - ady*(sx - aax)| / alen
        d_start = np.where(
            valid,
            np.abs(adx * (sy - aay) - ady * (sx - aax)) / safe_len,
            np.inf,
        )
        # Perpendicular distance from candidate's end endpoint to each accepted line.
        d_end = np.where(
            valid,
            np.abs(adx * (ey - aay) - ady * (ex - aax)) / safe_len,
            np.inf,
        )

        # If either endpoint lands on any accepted line, reject this candidate.
        if np.any(np.minimum(d_start, d_end) <= proximity):
            continue

        acc_ax[n_acc]   = starts[i, 0]
        acc_ay[n_acc]   = starts[i, 1]
        acc_dx[n_acc]   = dx[i]
        acc_dy[n_acc]   = dy[i]
        acc_len[n_acc]  = lengths[i]
        acc_orig[n_acc] = i
        n_acc += 1

    if n_acc == 0:
        return lines.astype(np.int32, copy=False)

    return lines[acc_orig[:n_acc]].astype(np.int32, copy=False)
