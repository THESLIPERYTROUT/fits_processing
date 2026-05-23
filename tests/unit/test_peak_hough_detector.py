"""Unit tests for the row-peak Hough detector."""
from __future__ import annotations

import numpy as np
import pytest

from streakiller.config.schema import PeakHoughParams
from streakiller.detection.peak_hough_detector import PeakHoughDetector


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_fading_streak_image(
    shape: tuple[int, int],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    peak_adu: float,
    fade_px: int,
    noise_sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic float32 image with a horizontal streak whose amplitude linearly
    ramps down over *fade_px* pixels at each tip (simulating faint endpoints).

    The inner segment (x1+fade_px … x2-fade_px) has amplitude *peak_adu*.
    The ramp regions taper from peak_adu → 0.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise_sigma, shape).astype(np.float32)

    H, W = shape
    row = y1  # horizontal streak assumed
    for x in range(x1, x2 + 1):
        if x < 0 or x >= W:
            continue
        dist_left  = x - x1
        dist_right = x2 - x
        dist_edge  = min(dist_left, dist_right)
        if dist_edge >= fade_px:
            amp = peak_adu
        else:
            amp = peak_adu * (dist_edge / fade_px)
        if 0 <= row < H:
            img[row, x] += amp

    return img


def _detected_extent(detector: PeakHoughDetector, img: np.ndarray) -> tuple[int, int] | None:
    """Return (min_x, max_x) of the detected segment with largest horizontal span, or None."""
    result = detector.detect(binary=None, source_data=img, min_line_length=30)
    if len(result.lines) == 0:
        return None
    spans = []
    for seg in result.lines[:, 0, :]:
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        spans.append((min(x1, x2), max(x1, x2)))
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)
    return spans[0]


# ------------------------------------------------------------------ #
# Tests                                                               #
# ------------------------------------------------------------------ #

class TestEndpointRefinement:

    _SHAPE    = (256, 256)
    _X1, _Y1  = 30, 128
    _X2, _Y2  = 226, 128
    _PEAK_ADU = 10.0
    _FADE_PX  = 20

    @pytest.fixture
    def streak_image(self):
        return _make_fading_streak_image(
            self._SHAPE,
            self._X1, self._Y1,
            self._X2, self._Y2,
            self._PEAK_ADU,
            self._FADE_PX,
        )

    def test_refinement_extends_endpoints(self, streak_image):
        """With walk-out enabled, detected endpoints should reach close to the true streak extent."""
        params = PeakHoughParams(
            threshold_sigma=3.0,
            endpoint_walk_sigma=1.5,
            endpoint_gap_tolerance=3,
        )
        detector = PeakHoughDetector(params)
        extent = _detected_extent(detector, streak_image)
        assert extent is not None, "detector found no lines"

        detected_x1, detected_x2 = extent
        tolerance = 10
        assert detected_x1 <= self._X1 + tolerance, (
            f"left endpoint {detected_x1} is more than {tolerance} px inside true start {self._X1}"
        )
        assert detected_x2 >= self._X2 - tolerance, (
            f"right endpoint {detected_x2} is more than {tolerance} px inside true end {self._X2}"
        )

    def test_refinement_off_gives_shorter_segment(self, streak_image):
        """Disabling walk-out (walk_sigma=0) should yield a shorter segment than with it enabled."""
        params_on = PeakHoughParams(
            threshold_sigma=3.0,
            endpoint_walk_sigma=1.5,
            endpoint_gap_tolerance=3,
        )
        params_off = PeakHoughParams(
            threshold_sigma=3.0,
            endpoint_walk_sigma=0,
            endpoint_gap_tolerance=3,
        )

        extent_on  = _detected_extent(PeakHoughDetector(params_on),  streak_image)
        extent_off = _detected_extent(PeakHoughDetector(params_off), streak_image)

        assert extent_on  is not None, "detector (walk on)  found no lines"
        assert extent_off is not None, "detector (walk off) found no lines"

        span_on  = extent_on[1]  - extent_on[0]
        span_off = extent_off[1] - extent_off[0]
        assert span_on > span_off, (
            f"expected walk-out to extend segment: on={span_on} px, off={span_off} px"
        )


class TestDetectInterface:

    def test_returns_empty_when_no_streak(self):
        """Pure noise image should yield zero lines."""
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
        params = PeakHoughParams(threshold_sigma=3.0)
        result = PeakHoughDetector(params).detect(binary=None, source_data=img, min_line_length=20)
        assert result.lines.shape == (0, 1, 4)

    def test_output_shapes(self):
        """binary_image and normalized_display have the same spatial shape as input."""
        rng = np.random.default_rng(1)
        img = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        params = PeakHoughParams(threshold_sigma=3.0)
        result = PeakHoughDetector(params).detect(binary=None, source_data=img, min_line_length=10)
        assert result.binary_image.shape       == (64, 64)
        assert result.normalized_display.shape == (64, 64)

    def test_ignores_binary_argument(self):
        """Passing a dummy binary array should not affect the result."""
        rng = np.random.default_rng(2)
        img   = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        dummy = np.zeros((64, 64), dtype=np.uint8)
        params = PeakHoughParams(threshold_sigma=3.0)
        det = PeakHoughDetector(params)
        r1 = det.detect(binary=None,  source_data=img, min_line_length=10)
        r2 = det.detect(binary=dummy, source_data=img, min_line_length=10)
        assert np.array_equal(r1.lines, r2.lines)
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
