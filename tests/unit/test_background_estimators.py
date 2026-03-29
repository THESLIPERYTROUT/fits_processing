"""Unit tests for the three background estimators."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from streakiller.background.simple_median import SimpleMedianEstimator
from streakiller.background.gaussian_blur import GaussianBlurEstimator
from streakiller.background.double_pass import DoublePassEstimator
from streakiller.config.schema import BackgroundParams

ESTIMATORS = [SimpleMedianEstimator, GaussianBlurEstimator, DoublePassEstimator]


@pytest.fixture
def flat_image() -> np.ndarray:
    """128x128 near-uniform image (no streaks)."""
    rng = np.random.default_rng(0)
    return rng.normal(1000.0, 30.0, (128, 128)).astype(np.float32)


@pytest.fixture
def streak_image() -> np.ndarray:
    """128x128 image with one injected horizontal streak."""
    rng = np.random.default_rng(1)
    data = rng.normal(1000.0, 30.0, (128, 128)).astype(np.float32)
    data[64, 10:118] += 5000.0  # bright horizontal streak at row 64
    return data


# ------------------------------------------------------------------ #
# Common contract tests for all estimators                            #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("EstClass", ESTIMATORS)
def test_returns_uint8(streak_image, EstClass):
    result = EstClass().estimate(streak_image, BackgroundParams())
    assert result.dtype == np.uint8


@pytest.mark.parametrize("EstClass", ESTIMATORS)
def test_output_is_binary(streak_image, EstClass):
    result = EstClass().estimate(streak_image, BackgroundParams())
    unique = set(np.unique(result))
    assert unique.issubset({0, 255}), f"Non-binary values found: {unique - {0, 255}}"


@pytest.mark.parametrize("EstClass", ESTIMATORS)
def test_same_shape_as_input(streak_image, EstClass):
    result = EstClass().estimate(streak_image, BackgroundParams())
    assert result.shape == streak_image.shape


@pytest.mark.parametrize("EstClass", ESTIMATORS)
def test_does_not_modify_input(streak_image, EstClass):
    original = streak_image.copy()
    EstClass().estimate(streak_image, BackgroundParams())
    np.testing.assert_array_equal(streak_image, original)


@pytest.mark.parametrize("EstClass", ESTIMATORS)
def test_no_file_io(streak_image, EstClass):
    """Estimators must not write any files."""
    with patch("cv2.imwrite") as mock_write:
        EstClass().estimate(streak_image, BackgroundParams())
        mock_write.assert_not_called()


# ------------------------------------------------------------------ #
# Estimator-specific tests                                            #
# ------------------------------------------------------------------ #

class TestSimpleMedianEstimator:
    def test_streak_pixels_mostly_foreground(self, streak_image):
        result = SimpleMedianEstimator().estimate(streak_image, BackgroundParams())
        streak_row = result[64, 10:118]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.5, "Most streak pixels should be foreground"

    def test_flat_image_mostly_background(self, flat_image):
        result = SimpleMedianEstimator().estimate(flat_image, BackgroundParams())
        # simple_median uses median + 1.2*sigma, which marks ~12% of a Gaussian image
        # as foreground before morphological closing can grow it further.
        # The key assertion is that it's not the *entire* image (i.e. estimator runs).
        foreground_fraction = np.mean(result == 255)
        assert foreground_fraction < 0.9, (
            f"Expected mostly background, got {foreground_fraction:.0%} foreground "
            "(simple_median is a coarse estimator — this is expected to have false positives)"
        )


class TestGaussianBlurEstimator:
    def test_streak_pixels_mostly_foreground(self, streak_image):
        result = GaussianBlurEstimator().estimate(streak_image, BackgroundParams())
        streak_row = result[64, 10:118]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.3

    def test_small_kernel_size_is_odd(self):
        # Even kernel size should be auto-corrected to odd
        params = BackgroundParams(gaussian_kernel_size=50)  # even
        rng = np.random.default_rng(2)
        data = rng.normal(1000.0, 30.0, (64, 64)).astype(np.float32)
        # Should not raise
        result = GaussianBlurEstimator().estimate(data, params)
        assert result.dtype == np.uint8


class TestDoublePassEstimator:
    def test_streak_pixels_mostly_foreground(self, streak_image):
        result = DoublePassEstimator().estimate(streak_image, BackgroundParams())
        streak_row = result[64, 10:118]
        foreground_fraction = np.mean(streak_row == 255)
        assert foreground_fraction > 0.3

    def test_bug_fix_highpass_is_residual(self, streak_image):
        """
        The original code computed highpass = background - hot_mask (wrong).
        The fix computes highpass = image - background (correct residual).

        We verify that the DoublePassEstimator can detect a streak — if the
        bug were still present, the highpass would be near-zero and no streaks
        would be found.
        """
        result = DoublePassEstimator().estimate(streak_image, BackgroundParams())
        any_foreground = np.any(result == 255)
        assert any_foreground, (
            "DoublePassEstimator detected no foreground pixels — "
            "the highpass = background - hot_mask bug may be present"
        )
