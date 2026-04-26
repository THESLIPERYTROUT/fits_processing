"""Unit tests for per-streak SNR estimation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from streakiller.config.schema import SnrParams
from streakiller.models.streak import StreakSNR
from streakiller.snr.aperture import sample_apertures
from streakiller.snr.estimator import StreakSNREstimator


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def flat_image() -> np.ndarray:
    """256x256 uniform noise image — no streak."""
    rng = np.random.default_rng(42)
    return rng.normal(1000.0, 30.0, (256, 256)).astype(np.float32)


@pytest.fixture
def streak_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """256x256 image with a bright horizontal streak at row 128."""
    rng = np.random.default_rng(7)
    data = rng.normal(1000.0, 30.0, (256, 256)).astype(np.float32)
    x1, y1, x2, y2 = 20, 128, 236, 128
    data[128, x1:x2 + 1] += 3000.0
    return data, (x1, y1, x2, y2)


@pytest.fixture
def default_params() -> SnrParams:
    return SnrParams()


# ------------------------------------------------------------------ #
# aperture.sample_apertures                                           #
# ------------------------------------------------------------------ #

class TestSampleApertures:
    def test_returns_two_arrays(self, flat_image, default_params):
        on_px, off_px = sample_apertures(
            flat_image, 20, 128, 236, 128,
            half_width=default_params.half_width_px,
            off_gap=default_params.off_gap_px,
            off_width=default_params.off_width_px,
        )
        assert isinstance(on_px, np.ndarray)
        assert isinstance(off_px, np.ndarray)

    def test_arrays_are_float64(self, flat_image, default_params):
        on_px, off_px = sample_apertures(
            flat_image, 20, 128, 236, 128,
            half_width=default_params.half_width_px,
            off_gap=default_params.off_gap_px,
            off_width=default_params.off_width_px,
        )
        assert on_px.dtype == np.float64
        assert off_px.dtype == np.float64

    def test_degenerate_line_returns_empty(self, flat_image):
        on_px, off_px = sample_apertures(
            flat_image, 100, 100, 100, 100,
            half_width=3, off_gap=3, off_width=10,
        )
        assert len(on_px) == 0
        assert len(off_px) == 0

    def test_on_pixels_nonempty_for_valid_line(self, flat_image, default_params):
        on_px, _ = sample_apertures(
            flat_image, 20, 128, 236, 128,
            half_width=default_params.half_width_px,
            off_gap=default_params.off_gap_px,
            off_width=default_params.off_width_px,
        )
        assert len(on_px) > 0

    def test_off_pixels_nonempty_for_centred_line(self, flat_image, default_params):
        _, off_px = sample_apertures(
            flat_image, 20, 128, 236, 128,
            half_width=default_params.half_width_px,
            off_gap=default_params.off_gap_px,
            off_width=default_params.off_width_px,
        )
        assert len(off_px) > 0

    def test_pixel_values_within_image_range(self, flat_image, default_params):
        on_px, off_px = sample_apertures(
            flat_image, 20, 128, 236, 128,
            half_width=default_params.half_width_px,
            off_gap=default_params.off_gap_px,
            off_width=default_params.off_width_px,
        )
        lo, hi = float(flat_image.min()), float(flat_image.max())
        assert on_px.min() >= lo and on_px.max() <= hi
        assert off_px.min() >= lo and off_px.max() <= hi


# ------------------------------------------------------------------ #
# StreakSNREstimator.estimate_all                                     #
# ------------------------------------------------------------------ #

class TestStreakSNREstimator:
    def test_empty_lines_returns_empty_list(self, flat_image, default_params):
        estimator = StreakSNREstimator()
        lines = np.empty((0, 1, 4), dtype=np.int32)
        result = estimator.estimate_all(flat_image, lines, default_params)
        assert result == []

    def test_returns_one_result_per_streak(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert len(result) == 1

    def test_result_is_streak_snr_instance(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert isinstance(result[0], StreakSNR)

    def test_streak_index_matches_position(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert result[0].streak_index == 0

    def test_bright_streak_has_positive_snr(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        snr = result[0]
        assert snr.is_valid
        assert snr.snr > 0

    def test_bright_streak_snr_is_high(self, streak_image, default_params):
        """A 3000 ADU signal on 30 ADU noise should give SNR >> 10."""
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert result[0].snr > 10.0

    def test_noise_is_positive(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert result[0].noise > 0.0

    def test_n_on_and_n_off_positive(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        snr = result[0]
        assert snr.n_on_pixels > 0
        assert snr.n_off_pixels > 0

    def test_insufficient_background_produces_nan_snr(self, flat_image):
        """When min_off_pixels cannot be met, SNR is NaN."""
        estimator = StreakSNREstimator()
        # A 5-pixel streak yields at most ~100 off-pixels; demand 999999 to force NaN.
        params = SnrParams(half_width_px=3, off_gap_px=3, off_width_px=10, min_off_pixels=999999)
        lines = np.array([[[100, 128, 105, 128]]], dtype=np.int32)
        result = estimator.estimate_all(flat_image, lines, params)
        snr = result[0]
        assert not snr.is_valid
        assert math.isnan(snr.snr)

    def test_multiple_streaks_all_indexed_correctly(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        estimator = StreakSNREstimator()
        lines = np.array([
            [[x1, y1, x2, y2]],
            [[x1, y1 + 50, x2, y2 + 50]],
        ], dtype=np.int32)
        result = estimator.estimate_all(data, lines, default_params)
        assert len(result) == 2
        assert result[0].streak_index == 0
        assert result[1].streak_index == 1

    def test_does_not_modify_input_image(self, streak_image, default_params):
        data, (x1, y1, x2, y2) = streak_image
        original = data.copy()
        estimator = StreakSNREstimator()
        lines = np.array([[[x1, y1, x2, y2]]], dtype=np.int32)
        estimator.estimate_all(data, lines, default_params)
        np.testing.assert_array_equal(data, original)
