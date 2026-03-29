"""Unit tests for CalibrationStep."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from streakiller.calibration.calibrator import CalibrationStep, CalibrationError
from streakiller.models.fits_image import FitsImage, ObservationMetadata


def _make_fits(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.writeto(str(path), overwrite=True)


def _make_image(data: np.ndarray) -> FitsImage:
    meta = ObservationMetadata(
        exposure_time=1.0, date_obs=None, telescope=None, camera=None,
        focal_length_mm=None, lat=None, lon=None, elevation_m=None,
        binning=1, pixel_size_um=None, pixel_scale_arcsec=None,
    )
    return FitsImage(source_path=Path("test.fits"), data=data, raw_header={}, metadata=meta)


class TestCalibrationStep:
    def test_applies_dark_subtraction(self, tmp_path):
        data = np.full((32, 32), 1000.0, dtype=np.float32)
        dark = np.full((32, 32), 100.0, dtype=np.float32)
        flat = np.ones((32, 32), dtype=np.float32)

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        step = CalibrationStep(tmp_path)
        step.load_frames()
        result = step.apply(_make_image(data))

        np.testing.assert_allclose(result.data, 900.0, atol=1e-3)

    def test_applies_flat_division(self, tmp_path):
        data = np.full((32, 32), 1000.0, dtype=np.float32)
        dark = np.zeros((32, 32), dtype=np.float32)
        flat = np.full((32, 32), 2.0, dtype=np.float32)

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        step = CalibrationStep(tmp_path)
        step.load_frames()
        result = step.apply(_make_image(data))

        np.testing.assert_allclose(result.data, 500.0, atol=1e-3)

    def test_flat_division_handles_zero_pixels(self, tmp_path):
        data = np.full((8, 8), 100.0, dtype=np.float32)
        dark = np.zeros((8, 8), dtype=np.float32)
        flat = np.zeros((8, 8), dtype=np.float32)  # all zeros — division by zero

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        step = CalibrationStep(tmp_path)
        step.load_frames()
        result = step.apply(_make_image(data))

        # inf and NaN should be replaced with 0
        assert np.all(np.isfinite(result.data))
        np.testing.assert_array_equal(result.data, 0.0)

    def test_raises_on_shape_mismatch(self, tmp_path):
        data = np.ones((64, 64), dtype=np.float32)
        dark = np.ones((32, 32), dtype=np.float32)  # wrong shape
        flat = np.ones((64, 64), dtype=np.float32)

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        step = CalibrationStep(tmp_path)
        step.load_frames()

        with pytest.raises(CalibrationError, match="shape"):
            step.apply(_make_image(data))

    def test_raises_if_frames_not_loaded(self, tmp_path):
        step = CalibrationStep(tmp_path)
        with pytest.raises(CalibrationError, match="load_frames"):
            step.apply(_make_image(np.ones((8, 8), dtype=np.float32)))

    def test_raises_if_dark_missing(self, tmp_path):
        flat = np.ones((8, 8), dtype=np.float32)
        _make_fits(tmp_path / "mflat.fits", flat)
        # mdark.fits is deliberately absent

        step = CalibrationStep(tmp_path)
        with pytest.raises(CalibrationError, match="not found"):
            step.load_frames()

    def test_returns_new_image_does_not_mutate_input(self, tmp_path):
        data = np.full((16, 16), 500.0, dtype=np.float32)
        dark = np.full((16, 16), 50.0, dtype=np.float32)
        flat = np.ones((16, 16), dtype=np.float32)

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        step = CalibrationStep(tmp_path)
        step.load_frames()
        image = _make_image(data.copy())
        result = step.apply(image)

        # Original image data must be unchanged
        np.testing.assert_array_equal(image.data, data)
        # Result must be different
        assert not np.array_equal(result.data, data)

    def test_no_files_written_during_apply(self, tmp_path, monkeypatch):
        data = np.ones((8, 8), dtype=np.float32)
        dark = np.zeros((8, 8), dtype=np.float32)
        flat = np.ones((8, 8), dtype=np.float32)

        _make_fits(tmp_path / "mdark.fits", dark)
        _make_fits(tmp_path / "mflat.fits", flat)

        files_written = []
        from astropy.io import fits as af

        original_writeto = af.HDUList.writeto

        def mock_writeto(self, *args, **kwargs):
            files_written.append(args[0] if args else kwargs.get("fileobj"))
            return original_writeto(self, *args, **kwargs)

        # We only care that calibrator doesn't write extra FITS files
        step = CalibrationStep(tmp_path)
        step.load_frames()

        before = list(tmp_path.iterdir())
        step.apply(_make_image(data))
        after = list(tmp_path.iterdir())

        # No new files should appear
        assert set(after) == set(before), f"Unexpected new files: {set(after) - set(before)}"
