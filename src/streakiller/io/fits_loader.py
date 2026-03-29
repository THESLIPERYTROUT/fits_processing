"""
FITS file loading — converts a file path to a FitsImage domain object.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits

from streakiller.models.fits_image import FitsImage, ObservationMetadata

logger = logging.getLogger(__name__)


class FitsLoadError(OSError):
    """Raised when a FITS file cannot be loaded or is unreadable."""


class FitsLoader:
    """Loads a FITS file into a FitsImage domain object."""

    def load(self, file_path: Union[str, Path]) -> FitsImage:
        """
        Read *file_path* and return a FitsImage with float32 data.

        Tries HDU index 1 first (common FITS layout), then falls back to
        iterating all HDUs to find one with 2-D image data.

        Raises FitsLoadError if no suitable HDU is found.
        """
        path = Path(file_path)
        logger.info("Loading FITS file: %s", path)

        data, header = self._read_fits(path)
        header_dict = dict(header)
        metadata = ObservationMetadata.from_fits_header(header_dict)

        self._log_metadata(metadata, path)

        return FitsImage(
            source_path=path,
            data=data,
            raw_header=header_dict,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _read_fits(self, path: Path) -> tuple[np.ndarray, fits.Header]:
        try:
            return self._try_hdu_index(path, index=1)
        except Exception:
            pass

        # Fallback: scan all HDUs
        try:
            with fits.open(path) as hdul:
                for hdu in hdul:
                    if hdu.data is None:
                        continue
                    if hdu.data.ndim == 3:
                        return hdu.data[0].astype(np.float32), hdu.header
                    if hdu.data.ndim == 2:
                        return hdu.data.astype(np.float32), hdu.header
        except Exception as exc:
            raise FitsLoadError(f"Cannot open FITS file {path}: {exc}") from exc

        raise FitsLoadError(f"No 2-D image data found in {path}")

    def _try_hdu_index(self, path: Path, index: int) -> tuple[np.ndarray, fits.Header]:
        with fits.open(path) as hdul:
            hdu = hdul[index]
            if hdu.data is None:
                raise ValueError("HDU has no data")
            if hdu.data.ndim == 3:
                return hdu.data[0].astype(np.float32), hdu.header
            return hdu.data.astype(np.float32), hdu.header

    def _log_metadata(self, meta: ObservationMetadata, path: Path) -> None:
        logger.info(
            "Loaded %s — exposure=%.1fs  camera=%s  pixel_scale=%.3f arcsec/px",
            path.name,
            meta.exposure_time or 0,
            meta.camera or "unknown",
            meta.pixel_scale_arcsec or 0,
        )
        if not meta.has_location:
            logger.warning(
                "Missing observatory location in FITS headers "
                "(SITELAT, SITELONG, SITEELEV) — TLE streak estimation unavailable"
            )
