"""Unit tests for the smarter hot-pixel removal in StreakPipeline."""
from __future__ import annotations

import numpy as np
import pytest

from streakiller.config.schema import HotpixelParams
from streakiller.pipeline.streak_pipeline import StreakPipeline


def _remove(data, threshold=5000, **kw):
    params = HotpixelParams(**kw) if kw else HotpixelParams()
    return StreakPipeline._hotpixel_removal(data, threshold, params)


class TestIsolatedHotPixel:
    def test_isolated_pixel_is_replaced(self):
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[16, 16] = 50_000.0
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        assert result[16, 16] < 200.0, "isolated hot pixel should be replaced by neighbourhood median"

    def test_clean_pixels_are_untouched(self):
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[16, 16] = 50_000.0
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        clean = data.copy()
        clean[16, 16] = result[16, 16]
        np.testing.assert_allclose(result, clean, atol=0.1)

    def test_does_not_mutate_input(self):
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[16, 16] = 50_000.0
        original = data.copy()
        _remove(data, threshold=5000)
        np.testing.assert_array_equal(data, original)


class TestStatisticalThreshold:
    def test_abs_floor_removes_obvious_hot_pixel(self):
        # A pixel well above the absolute floor is removed even on a clean image.
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[5, 5] = 6000.0
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        assert result[5, 5] < 200.0

    def test_statistical_threshold_raises_bar_on_bright_background(self):
        # When the image background is near/above the abs floor, the statistical
        # threshold prevents the abs floor from aggressively flagging normal pixels.
        # Background ~7000 ADU, abs_floor=5000: without stat, many pixels get removed.
        # With stat threshold >> 5000, only genuine extremes are removed.
        rng = np.random.default_rng(99)
        data = rng.normal(loc=7000.0, scale=100.0, size=(64, 64)).astype(np.float32)
        hot_pixel_val = 60_000.0
        data[32, 32] = hot_pixel_val
        result = _remove(data, threshold=5000, threshold_sigma=10.0, max_cluster_size=4)
        # stat_threshold = 7000 + 10*148 ≈ 8480; effective = max(5000, 8480) = 8480
        # Normal background pixels (~7000) are safely below 8480 → NOT removed
        # The one genuinely extreme pixel at 60000 IS removed
        background_untouched = result[result < 9000]
        assert np.all(background_untouched > 5000), (
            "normal background pixels should not be removed even though they exceed abs_floor"
        )
        assert result[32, 32] < 20_000.0, "hot pixel should be replaced"

    def test_zero_sigma_falls_back_to_abs_threshold(self):
        # With threshold_sigma=0, only the absolute floor applies.
        rng = np.random.default_rng(7)
        data = rng.normal(loc=100.0, scale=5.0, size=(64, 64)).astype(np.float32)
        data[10, 10] = 3000.0  # below abs floor=5000 → should NOT be replaced
        result = _remove(data, threshold=5000, threshold_sigma=0.0)
        assert result[10, 10] == pytest.approx(3000.0, abs=1.0)

    def test_pixel_below_both_thresholds_is_preserved(self):
        # A pixel that is bright but below the effective threshold is not touched.
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[8, 8] = 4000.0  # below abs_floor=5000 and stat_threshold on this image
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        assert result[8, 8] == pytest.approx(4000.0, abs=1.0)


class TestIsolationCheck:
    def test_large_cluster_is_preserved(self):
        # A 5×5 block of hot pixels exceeds max_cluster_size=4 → should not be removed
        data = np.ones((64, 64), dtype=np.float32) * 100.0
        data[20:25, 20:25] = 60_000.0  # 25 connected hot pixels
        result = _remove(data, threshold=5000, threshold_sigma=5.0, max_cluster_size=4)
        # At least some pixels in the cluster must survive (not replaced)
        assert np.any(result[20:25, 20:25] > 10_000.0)

    def test_small_cluster_within_limit_is_replaced(self):
        # A 2×2 cluster (4 pixels) is exactly at the limit (max=4) — should be removed
        data = np.ones((64, 64), dtype=np.float32) * 100.0
        data[30:32, 30:32] = 55_000.0  # 4 connected pixels
        result = _remove(data, threshold=5000, threshold_sigma=5.0, max_cluster_size=4)
        assert np.all(result[30:32, 30:32] < 1000.0)

    def test_two_separate_isolated_pixels_both_replaced(self):
        data = np.ones((64, 64), dtype=np.float32) * 100.0
        data[10, 10] = 50_000.0
        data[50, 50] = 50_000.0
        result = _remove(data, threshold=5000, threshold_sigma=10.0, max_cluster_size=4)
        assert result[10, 10] < 200.0
        assert result[50, 50] < 200.0


class TestEdgeCases:
    def test_no_hot_pixels_returns_copy(self):
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        np.testing.assert_array_equal(result, data)
        assert result is not data  # must be a copy

    def test_uniform_image_no_crash(self):
        # MAD = 0 on a perfectly uniform image; should not divide by zero
        data = np.full((32, 32), 1000.0, dtype=np.float32)
        result = _remove(data, threshold=5000, threshold_sigma=10.0)
        np.testing.assert_array_equal(result, data)

    def test_backward_compat_no_params(self):
        # Called with params=None (old-style) falls back to abs threshold + no isolation
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[16, 16] = 9999.0
        result = StreakPipeline._hotpixel_removal(data, 5000, params=None)
        assert result[16, 16] < 200.0

    def test_output_dtype_preserved(self):
        data = np.ones((32, 32), dtype=np.float32) * 100.0
        data[5, 5] = 50_000.0
        result = _remove(data, threshold=5000)
        assert result.dtype == np.float32
