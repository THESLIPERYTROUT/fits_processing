"""
Colinear merge filter — merges collinear line segments into single segments.

BUG FIX vs original (streakprocessing.py:244-246):
  Original called .append() and .pop() on the list while iterating over it
  via index, which caused index errors and skipped elements.

  Fixed: uses a union-find approach — build a graph of which segments are
  collinear, then merge each connected component into one segment.
"""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import FilterParams


def colinear_merge(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    Merge collinear line segments into single longer segments.

    Two segments are considered collinear when the cross-product magnitude of
    their direction vectors is below *params.colinear_orientation_tol*.

    Parameters
    ----------
    lines : ndarray, shape (N, 1, 4)
    params : FilterParams

    Returns
    -------
    merged : ndarray, shape (M, 1, 4)  where M <= N
    """
    if lines is None or len(lines) <= 1:
        if lines is None or len(lines) == 0:
            return np.empty((0, 1, 4), dtype=np.int32)
        return np.array(lines, dtype=np.int32)

    n = len(lines)
    tol = params.colinear_orientation_tol

    # Union-find -------------------------------------------------------- #
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for i in range(n):
        x1, y1, x2, y2 = lines[i][0]
        A = np.array([x1, y1], dtype=float)
        B = np.array([x2, y2], dtype=float)
        AB = B - A
        ab_len = np.linalg.norm(AB)
        if ab_len < 1e-9:
            continue
        for j in range(i + 1, n):
            x3, y3, x4, y4 = lines[j][0]
            C = np.array([x3, y3], dtype=float)
            D = np.array([x4, y4], dtype=float)
            CD = D - C
            cd_len = np.linalg.norm(CD)
            if cd_len < 1e-9:
                continue
            # Two segments are collinear iff:
            # 1. Their directions are parallel (cross product of unit vectors ≈ 0)
            # 2. A point on line j lies on the line through line i
            direction_cross = abs(float(AB[0] * CD[1] - AB[1] * CD[0]) / (ab_len * cd_len))
            AC = C - A
            point_cross = abs(float(AB[0] * AC[1] - AB[1] * AC[0])) / ab_len
            if direction_cross < tol and point_cross < tol:
                union(i, j)

    # Merge each connected component into a bounding segment ------------ #
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged: list[np.ndarray] = []
    for indices in groups.values():
        xs = []
        ys = []
        for idx in indices:
            x1, y1, x2, y2 = lines[idx][0]
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        merged.append(
            np.array([[min(xs), min(ys), max(xs), max(ys)]], dtype=np.int32)
        )

    return np.array(merged, dtype=np.int32)
