"""
Colinear merge filter — merges collinear line segments into single segments.

BUG FIX vs original (streakprocessing.py:244-246):
  Original called .append() and .pop() on the list while iterating over it
  via index, which caused index errors and skipped elements.

  Fixed: uses a union-find approach — build a graph of which segments are
  collinear, then merge each connected component into one segment.

PERFORMANCE: the collinearity test is fully vectorised with NumPy broadcasting
so the O(n²) work runs in C rather than a Python double-loop.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from streakiller.config.schema import FilterParams


def colinear_merge(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Merge collinear line segments into single longer segments.

    Two segments are considered collinear when:
      1. Their direction vectors are parallel (cross-product / lengths < tol)
      2. A start-point of one lies on the infinite line through the other
         (cross-product / length < tol)
      3. Their nearest endpoints are within *colinear_max_endpoint_distance* px

    Parameters
    ----------
    lines : ndarray, shape (N, 1, 4)
    params : FilterParams

    Returns
    -------
    merged : ndarray, shape (M, 1, 4)  where M <= N
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 1, 4), dtype=np.int32)
    if len(lines) == 1:
        return np.array(lines, dtype=np.int32)

    n = len(lines)
    tol = params.colinear_orientation_tol
    max_ep_dist = params.colinear_max_endpoint_distance

    pts = lines[:, 0, :].astype(float)   # (n, 4)
    starts = pts[:, :2]                   # (n, 2) — A endpoints
    ends   = pts[:, 2:]                   # (n, 2) — B endpoints
    AB     = ends - starts                # (n, 2) — direction vectors
    ab_len = np.hypot(AB[:, 0], AB[:, 1]) # (n,)
    valid  = ab_len > 1e-9                # non-degenerate segments

    AB_x = AB[:, 0]
    AB_y = AB[:, 1]

    # ---- direction cross products ----------------------------------------
    # |AB[i] × AB[j]| / (ab_len[i] * ab_len[j])  — shape (n, n)
    cross_dir = np.abs(
        AB_x[:, None] * AB_y[None, :] - AB_y[:, None] * AB_x[None, :]
    )
    denom = ab_len[:, None] * ab_len[None, :]
    direction_crosses = np.where(denom > 0, cross_dir / denom, np.inf)

    # ---- point cross products -------------------------------------------
    # |AB[i] × AC[i,j]| / ab_len[i],  AC[i,j] = starts[j] - starts[i]
    AC_x = starts[None, :, 0] - starts[:, None, 0]  # (n, n)
    AC_y = starts[None, :, 1] - starts[:, None, 1]  # (n, n)
    point_cross = np.abs(AB_x[:, None] * AC_y - AB_y[:, None] * AC_x)
    safe_len = np.where(ab_len > 0, ab_len, 1.0)
    point_crosses = point_cross / safe_len[:, None]

    # ---- minimum endpoint-to-endpoint distances -------------------------
    def _ep_dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.hypot(p[:, None, 0] - q[None, :, 0], p[:, None, 1] - q[None, :, 1])

    min_ep = np.minimum(
        np.minimum(_ep_dist(starts, starts), _ep_dist(starts, ends)),
        np.minimum(_ep_dist(ends,   starts), _ep_dist(ends,   ends)),
    )  # (n, n)

    # ---- adjacency: upper triangle only ---------------------------------
    should_merge = (
        (direction_crosses < tol) &
        (point_crosses     < tol) &
        (min_ep           <= max_ep_dist) &
        valid[:, None] & valid[None, :]
    )
    i_idx, j_idx = np.where(np.triu(should_merge, k=1))

    # ---- union-find -----------------------------------------------------
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        parent[find(i)] = find(j)

    # ---- merge each connected component into a bounding segment ---------
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged: list[np.ndarray] = []
    for indices in groups.values():
        all_pts = pts[indices]  # (k, 4)
        xs = np.concatenate([all_pts[:, 0], all_pts[:, 2]])
        ys = np.concatenate([all_pts[:, 1], all_pts[:, 3]])
        merged.append(
            np.array([[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]],
                     dtype=np.int32)
        )

    return np.array(merged, dtype=np.int32)
