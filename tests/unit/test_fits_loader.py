"""Unit tests for FitsLoader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from streakiller.io.fits_loader import FitsLoader, FitsLoadError


def _write_fits(path: Path, data: np.ndarray, header_updates: dict | None = None) -> None:
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    if header_updates:
        for k, v in header_updates.items():
            hdu.header[k] = v
    hdu.writeto(str(path), overwrite=True)


class TestFitsLoader:
    def test_loads_simple_fits(self, tmp_path):
        data = np.ones((64, 64), dtype=np.float32) * 1000
        path = tmp_path / "test.fits"
        _write_fits(path, data)

        image = FitsLoader().load(path)

        assert image.data.dtype == np.float32
        assert image.data.shape == (64, 64)
        np.testing.assert_allclose(image.data, 1000.0, atol=1e-3)

    def test_source_path_set(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)))
        image = FitsLoader().load(path)
        assert image.source_path == path

    def test_extracts_exptime(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)), {"EXPTIME": 5.0})
        image = FitsLoader().load(path)
        assert image.metadata.exposure_time == pytest.approx(5.0)

    def test_extracts_location(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)), {
            "SITELAT": 51.5, "SITELONG": -0.1, "SITEELEV": 50.0
        })
        image = FitsLoader().load(path)
        assert image.metadata.lat == pytest.approx(51.5)
        assert image.metadata.lon == pytest.approx(-0.1)
        assert image.metadata.has_location is True

    def test_missing_location_is_none(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)))
        image = FitsLoader().load(path)
        assert image.metadata.has_location is False

    def test_qhy268m_pixel_scale(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)), {"INSTRUME": "QHY268M", "FOCALLEN": 500.0})
        image = FitsLoader().load(path)
        # pixel_size = 3.76 µm * 1 binning; scale = 206.265 * 3.76 / 500 ≈ 1.551
        assert image.metadata.pixel_scale_arcsec == pytest.approx(1.551, rel=0.01)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FitsLoadError, FileNotFoundError)):
            FitsLoader().load(tmp_path / "nonexistent.fits")

    def test_raw_header_dict(self, tmp_path):
        path = tmp_path / "img.fits"
        _write_fits(path, np.zeros((8, 8)), {"MYKEY": "hello"})
        image = FitsLoader().load(path)
        assert "MYKEY" in image.raw_header
        assert image.raw_header["MYKEY"] == "hello"
