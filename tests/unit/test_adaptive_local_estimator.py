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
        assert 15.0 < sigma < 60.0, f"sigma={sigma:.2f} out of expected range"

    def test_tile_with_bright_outliers(self):
        rng = np.random.default_rng(8)
        pixels = rng.normal(1000.0, 30.0, 1000).astype(np.float32)
        pixels[:10] = 1e5  # extreme outliers (stars / cosmic rays)
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
        bg_map, sigma_map = AdaptiveLocalEstimator()._build_mesh(data, params)
        assert bg_map.shape == (4, 4)
        assert sigma_map.shape == (4, 4)

    def test_mesh_shape_non_divisible(self):
        data = _flat_image(130, 130)
        params = _make_params(adaptive_local_tile_size=32)
        bg_map, sigma_map = AdaptiveLocalEstimator()._build_mesh(data, params)
        assert bg_map.shape == (5, 5)

    def test_all_valid_tiles_on_clean_image(self):
        data = _flat_image(128, 128)
        params = _make_params(adaptive_local_tile_size=32, adaptive_local_min_tile_pixels=5)
        bg_map, _ = AdaptiveLocalEstimator()._build_mesh(data, params)
        assert np.all(np.isfinite(bg_map)), "All tiles should be valid on a clean image"

    def test_high_min_pixels_forces_all_tiles_nan(self):
        """tile_size=4 → 16 px/tile; min_tile_pixels=20 → all tiles NaN."""
        data = _flat_image(32, 32)
        params = _make_params(adaptive_local_tile_size=4, adaptive_local_min_tile_pixels=20)
        bg_map, sigma_map = AdaptiveLocalEstimator()._build_mesh(data, params)
        assert np.all(np.isnan(bg_map))
        assert np.all(np.isnan(sigma_map))

    def test_single_tile_image(self):
        data = _flat_image(32, 32)
        params = _make_params(adaptive_local_tile_size=64)
        bg_map, sigma_map = AdaptiveLocalEstimator()._build_mesh(data, params)
        assert bg_map.shape == (1, 1)

    def test_highpass_residual_has_zero_median_per_tile(self):
        """
        When _build_mesh is called on a Gaussian high-pass residual (as the
        hybrid estimate() does), bg_map values should be near zero since the
        high-pass already removed the background.
        """
        import cv2
        rng = np.random.default_rng(5)
        data = rng.normal(1000.0, 30.0, (128, 128)).astype(np.float32)
        blur = cv2.GaussianBlur(data, (51, 51), 0)
        highpass = data - blur

        params = _make_params(adaptive_local_tile_size=32)
        bg_map, sigma_map = AdaptiveLocalEstimator()._build_mesh(highpass, params)

        # Background of the high-pass should be near 0 everywhere
        assert np.all(np.abs(bg_map[np.isfinite(bg_map)]) < 5.0), (
            "bg_map on high-pass residual should be near zero"
        )
        # Noise should reflect the true pixel noise (~30 ADU)
        assert np.all(sigma_map[np.isfinite(sigma_map)] > 5.0)


# ------------------------------------------------------------------ #
# Class 3: _interpolate_maps unit tests                               #
# ------------------------------------------------------------------ #

class TestInterpolation:
    def test_output_shape_matches_image(self):
        bg_map    = np.full((4, 4), 1000.0, dtype=np.float32)
        sigma_map = np.full((4, 4), 30.0,   dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        assert bg_model.shape    == (128, 128)
        assert noise_model.shape == (128, 128)

    def test_nan_tile_does_not_crash(self):
        bg_map    = np.full((4, 4), 1000.0, dtype=np.float32)
        bg_map[2, 2] = np.nan
        sigma_map = np.full((4, 4), 30.0, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        assert np.all(np.isfinite(bg_model))
        assert np.all(np.isfinite(noise_model))

    def test_all_nan_returns_zeros(self):
        bg_map    = np.full((4, 4), np.nan, dtype=np.float32)
        sigma_map = np.full((4, 4), np.nan, dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert np.all(bg_model == 0.0)
        assert np.all(np.isfinite(noise_model))

    def test_constant_mesh_close_to_constant_output(self):
        bg_map    = np.full((4, 4), 1234.5, dtype=np.float32)
        sigma_map = np.full((4, 4), 42.0,   dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 128, 128)
        np.testing.assert_allclose(bg_model,    1234.5, atol=1.0)
        np.testing.assert_allclose(noise_model, 42.0,   atol=1.0)

    def test_noise_model_always_positive(self):
        bg_map    = np.full((4, 4), 500.0, dtype=np.float32)
        sigma_map = np.full((4, 4), 0.0,   dtype=np.float32)
        _, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert np.all(noise_model > 0)

    def test_1x1_mesh_falls_back_to_scalar(self):
        bg_map    = np.full((1, 1), 800.0, dtype=np.float32)
        sigma_map = np.full((1, 1), 25.0,  dtype=np.float32)
        bg_model, noise_model = AdaptiveLocalEstimator._interpolate_maps(bg_map, sigma_map, 64, 64)
        assert bg_model.shape == (64, 64)
        np.testing.assert_allclose(bg_model, 800.0, atol=1e-3)


# ------------------------------------------------------------------ #
# Class 4: AdaptiveLocalEstimator end-to-end tests                   #
# ------------------------------------------------------------------ #

class TestAdaptiveLocalEstimator:
    def test_streak_pixels_mostly_foreground(self):
        data = _streak_image(snr_offset=5000.0)
        result = AdaptiveLocalEstimator().estimate(data, _make_params())
        streak_row = result[64, 10:118]
        assert np.mean(streak_row == 255) > 0.3

    def test_faint_streak_detected(self):
        """
        Faint streak at ~2.5 sigma above local background.

        The Gaussian high-pass (kernel=51) attenuates a 1-px-wide horizontal
        streak by ~5% in the Y direction (centre weight of 1D Gaussian kernel),
        so +80 ADU streak → ~76 ADU after high-pass, SNR ≈ 2.5 at threshold=2.0.
        """
        rng = np.random.default_rng(123)
        data = rng.normal(1000.0, 30.0, (256, 256)).astype(np.float32)
        data[128, 20:236] += 80.0  # ~2.7 sigma global; ~2.5 sigma after high-pass

        params = _make_params(
            adaptive_local_tile_size=32,
            adaptive_local_snr_threshold=2.0,
        )
        result = AdaptiveLocalEstimator().estimate(data, params)
        foreground_fraction = np.mean(result[128, 20:236] == 255)
        assert foreground_fraction > 0.3, (
            f"Faint streak should be detected; foreground_fraction={foreground_fraction:.0%}"
        )

    def test_pure_background_false_positive_rate(self):
        """
        Regression test for the core bug: on a pure-noise background plate the
        hybrid estimator must not flag an excessive fraction of pixels as foreground.

        At snr_threshold=3.0 the theoretical false-positive rate for a Gaussian is
        ~0.13 % per pixel.  We allow up to 5 % to account for the morphological
        close and any remaining estimation bias, while being strict enough to catch
        a regression to the old tile-only behaviour (which could reach 10–30 %).
        """
        rng = np.random.default_rng(99)
        data = rng.normal(1000.0, 30.0, (256, 256)).astype(np.float32)

        params = _make_params(adaptive_local_snr_threshold=3.0)
        result = AdaptiveLocalEstimator().estimate(data, params)
        foreground_fraction = float(np.mean(result == 255))

        assert foreground_fraction < 0.05, (
            f"Too many false positives on pure background: {foreground_fraction:.1%} "
            f"(expected < 5 % at SNR threshold 3.0)"
        )

    def test_flat_image_not_entirely_foreground(self):
        data = _flat_image(128, 128)
        result = AdaptiveLocalEstimator().estimate(data, _make_params(adaptive_local_snr_threshold=3.0))
        assert np.mean(result == 255) < 0.5

    def test_gradient_background_streak_detected(self):
        """
        Steep background gradient — the Gaussian high-pass removes it before
        noise estimation, so the method handles this without needing tiny tiles.
        """
        rng = np.random.default_rng(77)
        # Gradient: 500 (left) to 1500 (right) — 1000 ADU range across 256 px
        gradient = np.linspace(500, 1500, 256, dtype=np.float32)
        background = np.tile(gradient, (256, 1))
        data = (background + rng.normal(0, 30, (256, 256))).astype(np.float32)
        data[128, 20:236] += 120.0  # streak at ~4 sigma above local noise

        params = _make_params(
            adaptive_local_tile_size=32,
            adaptive_local_snr_threshold=2.5,
        )
        result = AdaptiveLocalEstimator().estimate(data, params)
        foreground_fraction = np.mean(result[128, 20:236] == 255)
        assert foreground_fraction > 0.3, (
            f"Streak on gradient background should be detected; "
            f"foreground_fraction={foreground_fraction:.0%}"
        )

    def test_small_image_does_not_crash(self):
        data = _flat_image(16, 16)
        result = AdaptiveLocalEstimator().estimate(data, _make_params(adaptive_local_tile_size=64))
        assert result.dtype == np.uint8
        assert result.shape == (16, 16)

    def test_all_zeros_image_does_not_crash(self):
        data = np.zeros((64, 64), dtype=np.float32)
        result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_snr_threshold_controls_sensitivity(self):
        data = _streak_image(snr_offset=300.0)
        result_tight = AdaptiveLocalEstimator().estimate(data, _make_params(adaptive_local_snr_threshold=5.0))
        result_loose = AdaptiveLocalEstimator().estimate(data, _make_params(adaptive_local_snr_threshold=2.0))
        assert np.count_nonzero(result_loose) >= np.count_nonzero(result_tight)

    def test_output_is_uint8_binary(self):
        data = _streak_image()
        result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})

    def test_same_shape_as_input(self):
        for shape in [(64, 64), (128, 256), (512, 512)]:
            rng = np.random.default_rng(0)
            data = rng.normal(1000.0, 30.0, shape).astype(np.float32)
            assert AdaptiveLocalEstimator().estimate(data, BackgroundParams()).shape == shape

    def test_does_not_modify_input(self):
        data = _streak_image()
        original = data.copy()
        AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        np.testing.assert_array_equal(data, original)

    def test_saturated_image_does_not_crash(self):
        """Fully saturated image: high-pass ≈ 0, SNR ≈ 0, binary should be all zeros."""
        data = np.full((128, 128), 1e7, dtype=np.float32)
        result = AdaptiveLocalEstimator().estimate(data, BackgroundParams())
        assert result.dtype == np.uint8
        assert result.shape == (128, 128)
        # High-pass of a flat field is zero, so no pixel exceeds the SNR threshold
        assert np.count_nonzero(result) == 0, "Saturated image should produce no foreground"
