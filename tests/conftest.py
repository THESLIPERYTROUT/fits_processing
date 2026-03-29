"""Shared pytest fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from streakiller.config.schema import PipelineConfig
from streakiller.models.fits_image import FitsImage, ObservationMetadata


@pytest.fixture
def minimal_config(tmp_path: Path) -> PipelineConfig:
    """A valid PipelineConfig pointing at temporary directories."""
    (tmp_path / "images").mkdir()
    (tmp_path / "output").mkdir()
    return PipelineConfig(
        images_dir=str(tmp_path / "images"),
        output_dir=str(tmp_path / "output"),
        logging_level="WARNING",
    )


@pytest.fixture
def synthetic_fits_image() -> FitsImage:
    """
    512x512 float32 image with Gaussian noise and one injected diagonal streak.

    Streak runs from (50, 50) to (460, 400) — the coordinates are (x, y) i.e.
    (col, row), so in array indexing that is rows 50→400, cols 50→460.
    """
    rng = np.random.default_rng(seed=42)
    data = rng.normal(1000.0, 50.0, (512, 512)).astype(np.float32)

    # Draw a streak by linear interpolation between two points
    x0, y0, x1, y1 = 50, 50, 460, 400
    length = int(np.hypot(x1 - x0, y1 - y0))
    xs = np.linspace(x0, x1, length).astype(int)
    ys = np.linspace(y0, y1, length).astype(int)
    data[ys, xs] += 3000.0  # elevated streak signal

    meta = ObservationMetadata(
        exposure_time=5.0,
        date_obs="2025-01-01T00:00:00",
        telescope="TestScope",
        camera="QHY268M",
        focal_length_mm=500.0,
        lat=51.5,
        lon=-0.1,
        elevation_m=50.0,
        binning=1,
        pixel_size_um=3.76,
        pixel_scale_arcsec=1.554,
    )
    return FitsImage(
        source_path=Path("synthetic_test.fits"),
        data=data,
        raw_header={},
        metadata=meta,
    )


@pytest.fixture
def mock_tle_text() -> str:
    """A plausible TLE string for unit tests (no network required)."""
    return (
        "ISS (ZARYA)\n"
        "1 25544U 98067A   25001.50000000  .00001764  00000-0  40720-4 0  9993\n"
        "2 25544  51.6416  72.0563 0002429 131.4956 228.6407 15.49549852479186\n"
    )


@pytest.fixture
def sample_config_json(tmp_path: Path) -> Path:
    """Write a minimal valid config.json to tmp_path and return its path."""
    cfg = {
        "images_dir": "images",
        "output_dir": "output",
        "logging_level": "WARNING",
        "image_calibration": False,
        "calibration_dir": "calibration_frames",
        "estimated_streak_length_enabled": False,
        "default_minlinelength": 25,
        "enabled_line_filters": {
            "midpoint_filter": True,
            "colinear_filter": False,
            "line_angle": True,
            "endpoint_filter": True,
            "length_filter": True,
        },
        "background_detection_method": {
            "simple_median": False,
            "gaussian_blur": True,
            "double_pass": False,
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p
