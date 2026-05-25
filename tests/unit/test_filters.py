"""Unit tests for the filter pure functions."""
from __future__ import annotations

import numpy as np
import pytest

from streakiller.config.schema import EnabledFilters, FilterParams
from streakiller.filters.chain import FilterChain
from streakiller.filters.midpoint import midpoint_filter
from streakiller.filters.angle import angle_filter
from streakiller.filters.endpoint import endpoint_filter
from streakiller.filters.colinear import colinear_merge
from streakiller.filters.length import length_filter
from streakiller.filters.on_streak import on_streak_filter


def _line(x1, y1, x2, y2) -> np.ndarray:
    """Build a (1, 1, 4) int32 array matching HoughLinesP output."""
    return np.array([[[x1, y1, x2, y2]]], dtype=np.int32)


def _lines(*coords) -> np.ndarray:
    """Build (N, 1, 4) from a sequence of (x1,y1,x2,y2) tuples."""
    return np.array([[[x1, y1, x2, y2]] for x1, y1, x2, y2 in coords], dtype=np.int32)


# ------------------------------------------------------------------ #
# Shared edge-case tests applied to all filters                       #
# ------------------------------------------------------------------ #

FILTER_FNS = [midpoint_filter, angle_filter, endpoint_filter, colinear_merge, length_filter, on_streak_filter]


def test_filter_chain_full_order():
    chain = FilterChain.from_config(
        EnabledFilters(
            midpoint_filter=True,
            line_angle=True,
            colinear_filter=True,
            on_streak_filter=True,
            endpoint_filter=True,
            length_filter=True,
        )
    )
    assert chain.step_names == [
        "angle_filter",
        "colinear_merge",
        "midpoint_filter",
        "endpoint_filter",
        "on_streak_filter",
        "length_filter",
    ]


def test_filter_chain_order_without_on_streak():
    chain = FilterChain.from_config(
        EnabledFilters(
            midpoint_filter=True,
            line_angle=True,
            colinear_filter=True,
            on_streak_filter=False,
            endpoint_filter=True,
            length_filter=True,
        )
    )
    assert chain.step_names == [
        "angle_filter",
        "colinear_merge",
        "midpoint_filter",
        "endpoint_filter",
        "length_filter",
    ]


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
        # Start point of the second line is within threshold of the first → removed
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


# ------------------------------------------------------------------ #
# LengthFilter                                                         #
# ------------------------------------------------------------------ #

class TestLengthFilter:
    def test_sparse_set_keeps_only_one_line(self):
        lines = _lines((0, 0, 50, 0), (0, 100, 50, 100), (0, 200, 50, 200))
        result = length_filter(lines, FilterParams(length_fraction=0.8))
        assert len(result) == 1

    def test_sparse_set_prefers_longest_line(self):
        lines = _lines((0, 0, 10, 0), (0, 0, 400, 0))
        result = length_filter(lines, FilterParams(length_fraction=0.99, max_length_factor=1.01))
        assert len(result) == 1
        assert result[0, 0, 2] - result[0, 0, 0] == 400

    def test_lower_floor_drops_short_fragments(self):
        # modal length = 100; min_allowed = 0.9 * 100 = 90
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
        # modal length = 100; max_allowed = 2.0 * 100 = 200
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
        # The selected modal bin has an even-pair median between actual lengths,
        # so an exact band keeps nothing and the fallback returns all lines.
        lines = _lines(
            (0, 0, 100, 0),
            (0, 10, 101, 10),
            (0, 20, 102, 20),
            (0, 30, 500, 30),
            (0, 40, 520, 40),
            (0, 50, 540, 50),
        )
        result = length_filter(lines, FilterParams(length_fraction=1.0, max_length_factor=1.0))
        assert len(result) == 6

    def test_uses_modal_cluster_instead_of_median(self):
        lines = _lines(
            (0, 0, 40, 0),    # 40
            (0, 10, 41, 10),  # 41
            (0, 20, 42, 20),  # 42
            (0, 30, 100, 30), # 100
            (0, 40, 100, 40), # 100
            (0, 50, 100, 50), # 100
            (0, 60, 100, 60), # 100
            (0, 70, 100, 70), # 100
        )
        result = length_filter(lines, FilterParams(length_fraction=0.9))
        assert len(result) == 5

    def test_multimodal_picks_longest_mode(self):
        # Bimodal: 4 short fragments (~43 px, including axis-aligned exact ties)
        # and 6 long streaks (~103-107 px).  The filter must anchor to the long
        # cluster even though the short cluster contains exact-pixel-length ties
        # that the old exact-mode path would have incorrectly selected.
        lines = _lines(
            (0, 0, 0, 43),       # 43 — vertical exact tie
            (10, 0, 10, 43),     # 43 — vertical exact tie
            (20, 0, 20, 43),     # 43 — vertical exact tie
            (30, 5, 30, 50),     # 45
            (0, 100, 105, 100),  # 105
            (0, 110, 105, 110),  # 105
            (0, 120, 104, 120),  # 104
            (0, 130, 106, 130),  # 106
            (0, 140, 103, 140),  # 103
            (0, 150, 107, 150),  # 107
        )
        result = length_filter(lines, FilterParams(length_fraction=0.9))
        # modal ~105, min_allowed ~94.5 — only the 6 long streaks survive
        assert len(result) == 6


# ------------------------------------------------------------------ #
# OnStreakFilter                                                       #
# ------------------------------------------------------------------ #

class TestOnStreakFilter:
    def test_removes_short_duplicate_on_same_horizontal_line(self):
        # Long line: horizontal at y=50 from x=0 to x=200.
        # Short duplicate: same y, subset span — both endpoints within 0px of the long line.
        lines = _lines((0, 50, 200, 50), (50, 50, 100, 50))
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=2.0))
        assert len(result) == 1
        # The survivor must be the longer one.
        assert result[0, 0, 2] - result[0, 0, 0] == 200

    def test_removes_short_duplicate_offset_by_1px(self):
        # Long diagonal; short fragment whose endpoints are 1 px off the infinite line.
        # Line A: (0,0)→(100,100), infinite line is y=x.
        # Line B: (40,41)→(60,61) — both endpoints are 1/√2 ≈ 0.7 px from y=x.
        lines = _lines((0, 0, 100, 100), (40, 41, 60, 61))
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=2.0))
        assert len(result) == 1

    def test_keeps_truly_separate_parallel_lines(self):
        # Two parallel horizontal lines 20 px apart — not the same streak.
        lines = _lines((0, 0, 200, 0), (0, 20, 200, 20))
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=3.0))
        assert len(result) == 2

    def test_longer_line_wins_over_shorter(self):
        # Two overlapping horizontal lines; shorter submitted first.
        # Filter is longest-first so the short one should always be dropped.
        lines = _lines((20, 50, 80, 50), (0, 50, 200, 50))
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=2.0))
        assert len(result) == 1
        assert result[0, 0, 0] == 0   # starts at x=0 (the longer line)

    def test_many_duplicates_of_one_streak_collapse_to_one(self):
        # 1 long line + 4 short fragments all on the same horizontal infinite line.
        lines = _lines(
            (0, 100, 500, 100),  # long — kept
            (10, 100, 80, 100),  # short duplicate
            (100, 100, 180, 100),
            (200, 100, 280, 100),
            (300, 100, 380, 100),
        )
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=2.0))
        assert len(result) == 1

    def test_proximity_threshold_is_respected(self):
        # Short line whose endpoints are exactly 5 px from the accepted line.
        # At proximity=3, kept; at proximity=6, removed.
        # Long line: (0,0)→(200,0), infinite line is y=0.
        # Short line at y=5: endpoints are exactly 5 px away.
        lines = _lines((0, 0, 200, 0), (50, 5, 100, 5))
        tight = on_streak_filter(lines, FilterParams(on_streak_proximity_px=3.0))
        loose = on_streak_filter(lines, FilterParams(on_streak_proximity_px=6.0))
        assert len(tight) == 2   # 5 px > 3 → not a duplicate, both kept
        assert len(loose) == 1   # 5 px < 6 → duplicate removed

    def test_no_mutation_of_input(self):
        lines = _lines((0, 0, 200, 0), (50, 1, 100, 1))
        original = lines.copy()
        on_streak_filter(lines, FilterParams(on_streak_proximity_px=3.0))
        np.testing.assert_array_equal(lines, original)

    def test_output_dtype_is_int32(self):
        lines = _lines((0, 0, 100, 0), (10, 0, 50, 0))
        result = on_streak_filter(lines, FilterParams(on_streak_proximity_px=2.0))
        assert result.dtype == np.int32
