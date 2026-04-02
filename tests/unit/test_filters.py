"""Unit tests for the five filter pure functions."""
from __future__ import annotations

import numpy as np
import pytest

from streakiller.config.schema import FilterParams
from streakiller.filters.midpoint import midpoint_filter
from streakiller.filters.angle import angle_filter
from streakiller.filters.endpoint import endpoint_filter
from streakiller.filters.colinear import colinear_merge
from streakiller.filters.length import length_filter


def _line(x1, y1, x2, y2) -> np.ndarray:
    """Build a (1, 1, 4) int32 array matching HoughLinesP output."""
    return np.array([[[x1, y1, x2, y2]]], dtype=np.int32)


def _lines(*coords) -> np.ndarray:
    """Build (N, 1, 4) from a sequence of (x1,y1,x2,y2) tuples."""
    return np.array([[[x1, y1, x2, y2]] for x1, y1, x2, y2 in coords], dtype=np.int32)


# ------------------------------------------------------------------ #
# Shared edge-case tests applied to all five filters                  #
# ------------------------------------------------------------------ #

FILTER_FNS = [midpoint_filter, angle_filter, endpoint_filter, colinear_merge, length_filter]


@pytest.mark.parametrize("fn", FILTER_FNS)
def test_empty_input_returns_empty(fn):
    empty = np.empty((0, 1, 4), dtype=np.int32)
    result = fn(empty, FilterParams())
    assert result.shape == (0, 1, 4)
    assert result.dtype == np.int32


@pytest.mark.parametrize("fn", FILTER_FNS)
def test_none_input_returns_empty(fn):
    result = fn(None, FilterParams())
    assert result.shape == (0, 1, 4)


@pytest.mark.parametrize("fn", FILTER_FNS)
def test_single_line_passes_through(fn):
    single = _lines((0, 0, 100, 100))
    result = fn(single, FilterParams())
    assert len(result) == 1


# ------------------------------------------------------------------ #
# MidpointFilter                                                       #
# ------------------------------------------------------------------ #

class TestMidpointFilter:
    def test_removes_close_duplicate(self):
        # Two lines with midpoints ~2.8 px apart, threshold=10 → keep first only
        lines = _lines((0, 0, 20, 20), (2, 2, 22, 22))
        result = midpoint_filter(lines, FilterParams(midpoint_min_distance=10.0))
        assert len(result) == 1

    def test_keeps_far_lines(self):
        # Midpoints are 100 px apart → both kept
        lines = _lines((0, 0, 10, 0), (0, 100, 10, 100))
        result = midpoint_filter(lines, FilterParams(midpoint_min_distance=10.0))
        assert len(result) == 2

    def test_respects_custom_threshold(self):
        # Midpoints 15 px apart: kept at threshold=10, removed at threshold=20
        lines = _lines((0, 0, 10, 0), (0, 15, 10, 15))
        assert len(midpoint_filter(lines, FilterParams(midpoint_min_distance=10.0))) == 2
        assert len(midpoint_filter(lines, FilterParams(midpoint_min_distance=20.0))) == 1

    def test_output_shape(self):
        lines = _lines((0, 0, 50, 50), (200, 200, 250, 250))
        result = midpoint_filter(lines, FilterParams())
        assert result.ndim == 3
        assert result.shape[1] == 1
        assert result.shape[2] == 4


# ------------------------------------------------------------------ #
# AngleFilter                                                          #
# ------------------------------------------------------------------ #

class TestAngleFilter:
    def test_removes_parallel_duplicate(self):
        # 5 horizontal + 1 diagonal — dominant cluster is horizontal, diagonal removed
        lines = _lines(
            (0, 0, 100, 0), (0, 10, 100, 10), (0, 20, 100, 20),
            (0, 30, 100, 30), (0, 40, 100, 40),  # 5 horizontal (0°)
            (0, 0, 100, 100),                     # 45° outlier
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 5

    def test_keeps_perpendicular_lines(self):
        # Fewer than MIN_LINES_FOR_ANGLE_FILTER → all returned unchanged
        lines = _lines((0, 0, 100, 0), (50, 0, 50, 100))
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 2

    def test_bug_fix_not_inverted(self):
        # 5 horizontal + 1 diagonal: dominant cluster (horizontal) should be kept,
        # angular outlier removed. An inverted filter would keep only the outlier.
        lines = _lines(
            (0, 0, 100, 0), (0, 10, 100, 10), (0, 20, 100, 20),
            (0, 30, 100, 30), (0, 40, 100, 40),  # 5 horizontal
            (0, 0, 100, 100),                     # 45° outlier
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 5, (
            "angle_filter should keep the dominant cluster and remove outliers"
        )

    def test_three_different_angles_all_kept(self):
        # Fewer than MIN_LINES_FOR_ANGLE_FILTER → all returned unchanged
        lines = _lines(
            (0, 0, 100, 0),    # 0°  (horizontal)
            (0, 0, 0, 100),    # 90° (vertical)
            (0, 0, 100, 100),  # 45° (diagonal)
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 3


# ------------------------------------------------------------------ #
# EndpointFilter                                                       #
# ------------------------------------------------------------------ #

class TestEndpointFilter:
    def test_removes_close_endpoint(self):
        # Second line's endpoint overlaps first → removed
        lines = _lines((0, 0, 100, 100), (2, 2, 200, 200))
        result = endpoint_filter(lines, FilterParams(endpoint_min_distance=10.0))
        assert len(result) == 1

    def test_keeps_distant_lines(self):
        lines = _lines((0, 0, 10, 10), (100, 100, 200, 200))
        result = endpoint_filter(lines, FilterParams(endpoint_min_distance=10.0))
        assert len(result) == 2


# ------------------------------------------------------------------ #
# ColinearMerge                                                        #
# ------------------------------------------------------------------ #

class TestColinearMerge:
    def test_merges_two_collinear_horizontal_segments(self):
        # Two horizontal segments along y=0: (0,0)→(50,0) and (60,0)→(110,0)
        # They share the same y, so they're collinear. Should merge to one.
        lines = _lines((0, 0, 50, 0), (60, 0, 110, 0))
        result = colinear_merge(lines, FilterParams(colinear_orientation_tol=1.0))
        assert len(result) == 1

    def test_keeps_non_collinear(self):
        # Perpendicular lines should NOT be merged
        lines = _lines((0, 0, 100, 0), (0, 0, 0, 100))
        result = colinear_merge(lines, FilterParams(colinear_orientation_tol=1.0))
        assert len(result) == 2

    def test_three_collinear_segments_merge_to_one(self):
        lines = _lines((0, 0, 30, 0), (40, 0, 70, 0), (80, 0, 110, 0))
        result = colinear_merge(lines, FilterParams(colinear_orientation_tol=1.0))
        assert len(result) == 1

    def test_no_mutation_of_input(self):
        lines = _lines((0, 0, 50, 0), (60, 0, 110, 0))
        original_copy = lines.copy()
        colinear_merge(lines, FilterParams())
        np.testing.assert_array_equal(lines, original_copy)


# ------------------------------------------------------------------ #
# LengthFilter                                                         #
# ------------------------------------------------------------------ #

class TestLengthFilter:
    def test_removes_short_lines(self):
        # 5 long diagonal lines (~141px) and 1 short (10px)
        # Fewer than 6 lines triggers the guard; use 6 total so filtering applies.
        # Modal length ≈ 141; short (10) < 0.8*141 → removed
        lines = _lines(
            (0, 0, 100, 100), (10, 10, 110, 110), (20, 20, 120, 120),
            (30, 30, 130, 130), (40, 40, 140, 140),  # ~141px each
            (0, 0, 10, 0),                            # 10px
        )
        result = length_filter(lines, FilterParams(length_fraction=0.8))
        assert len(result) == 5

    def test_keeps_all_equal_length_lines(self):
        # Fewer than 6 lines → guard returns all unchanged
        lines = _lines((0, 0, 50, 0), (0, 100, 50, 100), (0, 200, 50, 200))
        result = length_filter(lines, FilterParams(length_fraction=0.8))
        assert len(result) == 3

    def test_custom_fraction(self):
        # 5 long (100px) and 1 short (50px)
        # Modal = 100. fraction=0.6 → min=60 → 50 removed
        #              fraction=0.4 → min=40 → 50 kept
        lines = _lines(
            (0, 0, 100, 0), (0, 10, 100, 10), (0, 20, 100, 20),
            (0, 30, 100, 30), (0, 40, 100, 40),  # 100px each
            (0, 0, 50, 0),                        # 50px
        )
        result_strict = length_filter(lines, FilterParams(length_fraction=0.6))
        assert len(result_strict) == 5  # 50 < 0.6*100=60, removed
        result_loose = length_filter(lines, FilterParams(length_fraction=0.4))
        assert len(result_loose) == 6   # 50 >= 0.4*100=40, kept
