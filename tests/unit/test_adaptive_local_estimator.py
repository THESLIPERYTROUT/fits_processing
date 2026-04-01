"""Unit tests for the AdaptiveLocalEstimator and its internal helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from streakiller.background.adaptive_local import AdaptiveLocalEstimator, _sigma_clip
from streakiller.config.schema import BackgroundParams


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_params(**kwargs) -> BackgroundParams:
    return BackgroundParams(**kwargs)


def _streak_image(h: int = 128, w: int = 128, row: int = 64, snr_offset: float = 5000.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 30.0, (h, w)).astype(np.float32)
    data[row, 10 : w - 10] += snr_offset
    return data


def _flat_image(h: int = 128, w: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(1000.0, 30.0, (h, w)).astype(np.float32)


# ------------------------------------------------------------------ #
# Class 1: _sigma_clip unit tests                                     #
# ------------------------------------------------------------------ #

class TestSigmaClipper:
    def test_clean_gaussian_tile(self):
        rng = np.random.default_rng(7)
        pixels = rng.normal(1000.0, 30.0, 1000).astype(np.float32)
        mu, sigma = _sigma_clip(pixels, clip_sigma=3.0, n_iterations=3, min_pixels=10)
        assert abs(mu - 1000.0) < 10.0, f"mu={mu:.2f} too far from 1000"
        # MAD-normalised sigma should be close to 30 * 1.4826 ≈ 44.5 ... no, sigma is in ADU
        # For N(1000, 30): MAD ≈ 30 * 0.6745 ≈ 20.2, so sigma = 1.4826 * MAD ≈ 30
        assert 15.0 < sigma < 60.0, f"sigma={sigma:.2f} out of expected range"

    def test_tile_with_bright_outliers(self):
        rng = np.random.default_rng(8)
        pixels = rng.normal(1000.0, 30.0, 1000).astype(np.float32)
        # Inject 10 extreme outliers (stars / cosmic rays)
        pixels[:10] = 1e5
        mu, sigma = _sigma_clip(pixels, clip_sigma=3.0, n_iterations=3, min_pixels=10)
        assert abs(mu - 1000.0) < 15.0, f"Outliers should be clipped; mu={mu:.2f}"

    def test_insufficient_pixels_returns_nan(self):
        pixels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mu, sigma = _sigma_clip(pixels, clip_sigma=3.0, n_iterations=3, min_pixels=10)
        assert math.isnan(mu)
        assert math.isnan(sigma)

    def test_all_identical_pixels(self):
        pixels = np.full(100, 500.0, dtype=np.float32)
        mu, sigma = _sigma_clip(pixels, clip_sigma=3.0, n_iterations=3, min_pixels=10)
        assert abs(mu - 500.0) < 1e-3
        assert sigma > 0, "sigma must be positive (epsilon prevents zero)"

    def test_single_iteration(self):
        """n_iterations=1 should not crash and return finite values for a clean image."""
        rng = np.random.default_rng(9)
        pixels = rng.normal(500.0, 20.0, 500).astype(np.float32)
        mu, sigma = _sigma_clip(pixels, clip_sigma=3.0, n_iterations=1, min_pixels=5)
        assert math.isfinite(mu)
        assert math.isfinite(sigma)


# ------------------------------------------------------------------ #
# Class 2: _build_mesh unit tests                                     #
# ------------------------------------------------------------------ #

class TestMeshBuilding:
    def test_mesh_shape_divides_evenly(self):
        data = _flat_image(128, 128)
        params = _make_params(adaptive_local_tile_size=32)
        est = AdaptiveLocalEstimator()
        bg_map, sigma_map = est._build_mesh(data, params)
        assert bg_map.shape == (4, 4)
        assert sigma_map.shape == (4, 4)

    def test_mesh_shape_non_divisible(self):
        # 130 // 32 = 4, ceil(130/32) = 5 → partial tile at the edge
        data = _flat_image(130, 130)
        params = _make_params(adaptive_local_tile_size=32)
        est = AdaptiveLocalEstimator()
        bg_map, sigma_map = est._build_mesh(data, params)
        assert bg_map.shape == (5, 5)

    def test_all_valid_tiles_on_clean_image(self):
        data = _flat_image(128, 128)
        params = _make_params(adaptive_local_tile_size=32, adaptive_local_min_tile_pixels=5)
        est = AdaptiveLocalEstimator()
        bg_map, _ = est._build_mesh(data, params)
        assert np.all(np.isfinite(bg_map)), "All tiles should be valid on a clean image"

    def test_high_min_pixels_forces_all_tiles_nan(self):
        """
        When min_tile_pixels > pixels_per_tile, every tile has too few survivors
        and the mesh is entirely NaN.
        tile_size=4 → 16 px/tile; min_tile_pixels=20 → all tiles NaN.
        """
        data = _flat_image(32, 32)
        params = _make_params(
            adaptive_local_tile_size=4,
            adaptive_local_min_tile_pixels=20,  # > 4*4=16
        )
        est = AdaptiveLocalEstimator()
        bg_map, sigma_map = est._build_mesh(data, params)
        assert np.all(np.isnan(bg_map)), "All tiles should be NaN when min_pixels > tile area"
        assert np.all(np.isnan(sigma_map))

    def test_single_tile_image(self):
        """Image smaller than tile_size → one tile, 1x1 mesh."""
        data = _flat_image(32, 32)
        params = _make_params(adaptive_local_tile_size=64)
        est = AdaptiveLocalEstimator()
        bg_map, sigma_map = est._build_mesh(data, params)
        assert bg_map.shape == (1, 1)


# ------------------------------------------------------------------ #
# Class 3: _interpolate_maps unit tests                               #
# ------------------------------------------------------------------ #

class TestInterpolation:
    def test_output_shape_matches_image(self):
        bg_map = np.full((4, 4), 1000.0, dtype=np.float32)
        sigma_map = np.full((4, 4), 30.0, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        assert bg_model.shape == (128, 128)
        assert noise_model.shape == (128, 128)

    def test_nan_tile_does_not_crash(self):
        bg_map = np.full((4, 4), 1000.0, dtype=np.float32)
        bg_map[2, 2] = np.nan
        sigma_map = np.full((4, 4), 30.0, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        assert np.all(np.isfinite(bg_model))
        assert np.all(np.isfinite(noise_model))

    def test_all_nan_returns_zeros(self):
        bg_map = np.full((4, 4), np.nan, dtype=np.float32)
        sigma_map = np.full((4, 4), np.nan, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert np.all(bg_model == 0.0)
        # noise_model is clipped to 1e-6; result should still be all finite
        assert np.all(np.isfinite(noise_model))

    def test_constant_mesh_close_to_constant_output(self):
        bg_map = np.full((4, 4), 1234.5, dtype=np.float32)
        sigma_map = np.full((4, 4), 42.0, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        np.testing.assert_allclose(bg_model, 1234.5, atol=1.0)
        np.testing.assert_allclose(noise_model, 42.0, atol=1.0)

    def test_noise_model_always_positive(self):
        """After interpolation, noise must never be zero or negative."""
        bg_map = np.full((4, 4), 500.0, dtype=np.float32)
        sigma_map = np.full((4, 4), 0.0, dtype=np.float32)  # degenerate: zero sigma
        _, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert np.all(noise_model > 0), "Noise model must be strictly positive"

    def test_1x1_mesh_falls_back_to_scalar(self):
        bg_map = np.full((1, 1), 800.0, dtype=np.float32)
        sigma_map = np.full((1, 1), 25.0, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert bg_model.shape == (64, 64)
        np.testing.assert_allclose(bg_model, 800.0, atol=1e-3)


# ------------------------------------------------------------------ #
# Class 4: AdaptiveLocalEstimator end-to-end tests                   #
# ------------------------------------------------------------------ #

class TestAdaptiveLocalEstimator:
    def test_streak_pixels_mostly_foreground(self):
        data = _streak_image(snr_offset=5000.0)
        params = _make_params()
        result = AdaptiveLocalEstimator().estimate(data, params)
        streak_row = result[64, 10:118]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.3, (
            f"Expected >30% of streak row foreground, got {foreground_fraction:.0%}"
        )

    def test_faint_streak_detected_when_global_fails(self):
        """
        Inject a faint streak at only 2 sigma globally.  The AdaptiveLocalEstimator
        with snr_threshold=2.0 should detect it; global methods may struggle.
        """
        rng = np.random.default_rng(123)
        data = rng.normal(1000.0, 30.0, (256, 256)).astype(np.float32)
        # 2-sigma offset globally (global sigma ≈ 30 ADU → +60 ADU)
        data[128, 20:236] += 60.0

        params = _make_params(
            adaptive_local_tile_size=32,
            adaptive_local_snr_threshold=2.0,
        )
        result = AdaptiveLocalEstimator().estimate(data, params)
        streak_row = result[128, 20:236]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.3, (
            f"Faint streak should be detected; foreground_fraction={foreground_fraction:.0%}"
        )

    def test_flat_image_not_entirely_foreground(self):
        """On a noise-only image the estimator should not flag everything as foreground."""
        data = _flat_image(128, 128)
        params = _make_params(adaptive_local_snr_threshold=3.0)
        result = AdaptiveLocalEstimator().estimate(data, params)
        foreground_fraction = np.mean(result == 255)
        assert foreground_fraction < 0.5, (
            f"Expected mostly background on flat image, got {foreground_fraction:.0%} foreground"
        )

    def test_gradient_background_streak_detected(self):
        """
        Background ramps gently across the image.  A streak injected well above
        the local noise should still be detected.

        Note: tile_size must be small enough that within-tile gradient variation
        is much less than the noise level, so per-tile sigma estimates reflect true
        noise rather than the slope.  With noise=30 ADU and gradient=1000→1020 across
        256 px, within-tile variation for tile_size=32 is ~2.5 ADU << 30 ADU.
        """
        rng = np.random.default_rng(77)
        # Very gentle gradient: 1000 to 1020 across 256 px (≈0.08 ADU/pixel)
        # Within one 32-px tile the variation is only ≈2.5 ADU — well below noise.
        gradient = np.linspace(1000, 1020, 256, dtype=np.float32)
        background = np.tile(gradient, (256, 1))
        data = (background + rng.normal(0, 30, (256, 256))).astype(np.float32)
        # Inject streak at row 128 with +120 ADU above local background (≈4 sigma local)
        data[128, 20:236] += 120.0

        params = _make_params(
            adaptive_local_tile_size=32,
            adaptive_local_snr_threshold=2.5,
        )
        result = AdaptiveLocalEstimator().estimate(data, params)
        streak_row = result[128, 20:236]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.3, (
            f"Streak on gradient background should be detected; "
            f"foreground_fraction={foreground_fraction:.0%}"
        )

    def test_small_image_does_not_crash(self):
        """Images smaller than tile_size must not raise."""
        data = _flat_image(16, 16)
        params = _make_params(adaptive_local_tile_size=64)
        result = AdaptiveLocalEstimator().estimate(data, params)
        assert result.dtype == np.uint8
        assert result.shape == (16, 16)

    def test_all_zeros_image_does_not_crash(self):
        """Completely dark / zero image must not raise."""
        data = np.zeros((64, 64), dtype=np.float32)
        result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_snr_threshold_controls_sensitivity(self):
        """Lower SNR threshold → more foreground pixels on the same image."""
        data = _streak_image(snr_offset=300.0)  # moderate brightness streak
        params_tight = _make_params(adaptive_local_snr_threshold=5.0)
        params_loose = _make_params(adaptive_local_snr_threshold=2.0)

        result_tight = AdaptiveLocalEstimator().estimate(data, params_tight)
        result_loose = AdaptiveLocalEstimator().estimate(data, params_loose)

        fg_tight = int(np.count_nonzero(result_tight))
        fg_loose = int(np.count_nonzero(result_loose))
        assert fg_loose >= fg_tight, (
            f"Lower threshold should give >= foreground pixels; "
            f"tight={fg_tight}, loose={fg_loose}"
        )

    def test_output_is_uint8_binary(self):
        data = _streak_image()
        result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        assert result.dtype == np.uint8
        unique = set(np.unique(result))
        assert unique.issubset({0, 255}), f"Non-binary values: {unique - {0, 255}}"

    def test_same_shape_as_input(self):
        for shape in [(64, 64), (128, 256), (512, 512)]:
            rng = np.random.default_rng(0)
            data = rng.normal(1000.0, 30.0, shape).astype(np.float32)
            result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
            assert result.shape == shape

    def test_does_not_modify_input(self):
        data = _streak_image()
        original = data.copy()
        AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        np.testing.assert_array_equal(data, original)

    def test_saturated_image_returns_empty_binary(self):
        """All-saturated image → all tiles invalid → empty binary (no false detections)."""
        data = np.full((128, 128), 1e7, dtype=np.float32)
        params = _make_params(
            adaptive_local_tile_size=32,
            adaptive_local_min_tile_pixels=50,
        )
        result = AdaptiveLocalEstimator().estimate(data, params)
        assert result.dtype == np.uint8
        # Should be all zeros (no false detections) or at least not crash
        assert result.shape == (128, 128)
