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
    def test_skips_filter_for_small_inputs(self):
        lines = _lines((0, 0, 100, 0), (0, 5, 100, 5))
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 2

    def test_keeps_majority_parallel_cluster(self):
        lines = _lines(
            (0, 0, 100, 0),
            (0, 5, 100, 5),
            (0, 10, 100, 8),
            (0, 15, 100, 12),
            (50, 0, 50, 100),
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 4

    def test_drops_angular_outlier_from_majority(self):
        lines = _lines(
            (0, 0, 100, 0),
            (0, 4, 100, 4),
            (0, 8, 100, 10),
            (0, 12, 100, 13),
            (0, 0, 100, 100),
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 4

    def test_handles_reversed_lines_as_same_orientation(self):
        lines = _lines(
            (0, 0, 100, 0),
            (100, 5, 0, 5),
            (0, 10, 100, 10),
            (100, 15, 0, 15),
            (50, 0, 50, 100),
        )
        result = angle_filter(lines, FilterParams(angle_min_diff_deg=10.0))
        assert len(result) == 4


# ------------------------------------------------------------------ #
# EndpointFilter                                                       #
# ------------------------------------------------------------------ #

class TestEndpointFilter:
    def test_removes_close_endpoint(self):
        # Both endpoints of the second line are within threshold of the first → duplicate, removed
        lines = _lines((0, 0, 100, 100), (2, 2, 102, 102))
        result = endpoint_filter(lines, FilterParams(endpoint_min_distance=10.0))
        assert len(result) == 1

    def test_keeps_distant_lines(self):
        lines = _lines((0, 0, 10, 10), (100, 100, 200, 200))
        result = endpoint_filter(lines, FilterParams(endpoint_min_distance=10.0))
        assert len(result) == 2

    def test_keeps_line_sharing_only_one_endpoint(self):
        # Second line starts near the end of the first but diverges completely.
        # Only one endpoint pair is close → should NOT be treated as a duplicate.
        lines = _lines((0, 0, 100, 0), (98, 0, 200, 100))
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

    def test_does_not_merge_distant_collinear_segments(self):
        # Two collinear horizontal segments 500 px apart; nearest endpoints are 400 px away.
        # With max_endpoint_distance=100, they should stay separate.
        lines = _lines((0, 0, 50, 0), (450, 0, 500, 0))
        result = colinear_merge(
            lines,
            FilterParams(colinear_orientation_tol=1.0, colinear_max_endpoint_distance=100.0),
        )
        assert len(result) == 2

    def test_merges_nearby_collinear_segments(self):
        # Same geometry but within the distance threshold → should merge.
        lines = _lines((0, 0, 50, 0), (80, 0, 130, 0))
        result = colinear_merge(
            lines,
            FilterParams(colinear_orientation_tol=1.0, colinear_max_endpoint_distance=100.0),
        )
        assert len(result) == 1

    def test_merges_negative_slope_collinear_segments(self):
        # Streak going bottom-left → top-right: x increases, y decreases.
        # min(xs)/min(ys)/max(xs)/max(ys) would produce (10,60,90,200) — wrong slope.
        # Farthest-pair logic must produce (10,200,90,60) or (90,60,10,200).
        lines = _lines((10, 200, 50, 130), (50, 130, 90, 60))
        result = colinear_merge(lines, FilterParams(colinear_orientation_tol=1.0))
        assert len(result) == 1
        x1, y1, x2, y2 = result[0][0]
        # The merged segment should span the true extremes (10,200) and (90,60)
        # regardless of endpoint ordering
        pts = {(x1, y1), (x2, y2)}
        assert pts == {(10, 200), (90, 60)}, f"Got endpoints {pts}"


# ------------------------------------------------------------------ #
# LengthFilter                                                         #
# ------------------------------------------------------------------ #

class TestLengthFilter:
    def test_keeps_all_equal_length_lines(self):
        lines = _lines((0, 0, 50, 0), (0, 100, 50, 100), (0, 200, 50, 200))
        result = length_filter(lines, FilterParams(length_fraction=0.8))
        assert len(result) == 3

    def test_lower_floor_drops_short_fragments(self):
        # median = 100; min_allowed = 0.9 * 100 = 90; max_allowed = 3.0 * 100 = 300
        # 89px line is below the floor and should be dropped
        lines = _lines(
            (0, 0, 100, 0),   # 100 — kept
            (0, 10, 100, 10), # 100 — kept
            (0, 20, 100, 20), # 100 — kept
            (0, 30, 100, 30), # 100 — kept
            (0, 40, 100, 40), # 100 — kept
            (0, 50, 108, 50), # 108 — kept (above floor, below cap)
            (0, 60, 89, 60),  # 89  — dropped (below 0.9 * 100)
        )
        result = length_filter(lines, FilterParams(length_fraction=0.9))
        assert len(result) == 6

    def test_upper_cap_drops_merged_detections(self):
        # median = 100; max_allowed = 2.0 * 100 = 200
        # 250px line is above cap and should be dropped
        lines = _lines(
            (0, 0, 100, 0),   # 100 — kept
            (0, 10, 100, 10), # 100 — kept
            (0, 20, 100, 20), # 100 — kept
            (0, 30, 100, 30), # 100 — kept
            (0, 40, 100, 40), # 100 — kept
            (0, 50, 250, 50), # 250 — dropped (above 2.0 * 100)
        )
        result = length_filter(lines, FilterParams(length_fraction=0.9, max_length_factor=2.0))
        assert len(result) == 5

    def test_lower_fraction_loosens_floor(self):
        # With fraction=0.7, min_allowed = 0.7 * 100 = 70, so the 75px line is kept
        lines = _lines(
            (0, 0, 100, 0),   # 100
            (0, 10, 100, 10), # 100
            (0, 20, 100, 20), # 100
            (0, 30, 100, 30), # 100
            (0, 40, 100, 40), # 100
            (0, 50, 75, 50),  # 75 — dropped at 0.9, kept at 0.7
        )
        result_strict = length_filter(lines, FilterParams(length_fraction=0.9))
        assert len(result_strict) == 5
        result_loose = length_filter(lines, FilterParams(length_fraction=0.7))
        assert len(result_loose) == 6

    def test_fallback_returns_all_when_nothing_survives(self):
        # Two lines with wildly different lengths — median lands between them,
        # both end up outside the band; fallback should return all lines unchanged.
        lines = _lines((0, 0, 10, 0), (0, 0, 400, 0))
        result = length_filter(lines, FilterParams(length_fraction=0.99, max_length_factor=1.01))
        assert len(result) == 2
