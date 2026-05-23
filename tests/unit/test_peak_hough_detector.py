"""Unit tests for the row-peak Hough detector."""
from __future__ import annotations

import numpy as np

from streakiller.config.schema import PeakHoughParams
from streakiller.detection.peak_hough_detector import PeakHoughDetector


def test_peak_hough_detector_returns_raw_detection_shape():
    rng = np.random.default_rng(11)
    data = rng.normal(1000.0, 20.0, (128, 128)).astype(np.float32)

    x0, y0, x1, y1 = 15, 20, 115, 90
    length = int(np.hypot(x1 - x0, y1 - y0))
    xs = np.linspace(x0, x1, length).astype(int)
    ys = np.linspace(y0, y1, length).astype(int)
    data[ys, xs] += 3000.0

    detector = PeakHoughDetector(
        PeakHoughParams(
            median_bins=32,
            polynomial_degree=3,
            threshold_sigma=2.0,
            hough_threshold=5,
            max_line_gap=8,
        )
    )
    result = detector.detect(
        binary=np.zeros_like(data, dtype=np.uint8),
        source_data=data,
        min_line_length=30,
    )

    assert result.lines.ndim == 3
    assert result.lines.shape[1:] == (1, 4)
    assert result.lines.dtype == np.int32
    assert result.binary_image.shape == data.shape
    assert result.binary_image.dtype == np.uint8
    assert len(result.lines) >= 1


def test_peak_hough_detector_handles_empty_image():
    data = np.zeros((64, 64), dtype=np.float32)
    detector = PeakHoughDetector(PeakHoughParams())

    result = detector.detect(
        binary=np.zeros_like(data, dtype=np.uint8),
        source_data=data,
        min_line_length=20,
    )

    assert result.lines.shape == (0, 1, 4)
    assert result.binary_image.shape == data.shape


def test_peak_hough_detector_supports_legacy_row_peak_mode():
    rng = np.random.default_rng(12)
    data = rng.normal(1000.0, 20.0, (96, 96)).astype(np.float32)
    data[45, 10:85] += 2500.0

    detector = PeakHoughDetector(
        PeakHoughParams(
            peak_mode="row_1d",
            median_bins=24,
            polynomial_degree=3,
            threshold_sigma=2.0,
            hough_threshold=5,
        )
    )
    result = detector.detect(
        binary=np.zeros_like(data, dtype=np.uint8),
        source_data=data,
        min_line_length=20,
    )

    assert result.binary_image.shape == data.shape
    assert np.count_nonzero(result.binary_image) > 0
