"""Unit tests for PeakHoughDetector."""
from __future__ import annotations

import numpy as np
import pytest

from streakiller.config.schema import PeakHoughParams
from streakiller.detection.peak_hough_detector import PeakHoughDetector


class TestDetectInterface:

    def test_returns_empty_when_no_streak(self):
        """Pure noise image should yield zero lines."""
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
        params = PeakHoughParams(threshold_sigma=3.5)
        result = PeakHoughDetector(params).detect(binary=None, source_data=img, min_line_length=20)
        assert result.lines.shape == (0, 1, 4)

    def test_output_shapes(self):
        """binary_image and normalized_display have the same spatial shape as input."""
        rng = np.random.default_rng(1)
        img = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        params = PeakHoughParams(threshold_sigma=2.0)
        result = PeakHoughDetector(params).detect(binary=None, source_data=img, min_line_length=10)
        assert result.binary_image.shape       == (64, 64)
        assert result.normalized_display.shape == (64, 64)

    def test_ignores_binary_argument(self):
        """Passing a dummy binary array should not affect the result."""
        rng = np.random.default_rng(2)
        img   = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        dummy = np.zeros((64, 64), dtype=np.uint8)
        params = PeakHoughParams(threshold_sigma=2.0)
        det = PeakHoughDetector(params)
        r1 = det.detect(binary=None,  source_data=img, min_line_length=10)
        r2 = det.detect(binary=dummy, source_data=img, min_line_length=10)
        assert np.array_equal(r1.lines, r2.lines)
