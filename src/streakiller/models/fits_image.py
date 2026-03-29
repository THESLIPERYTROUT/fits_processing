"""
Domain model for a loaded FITS image.

FitsImage is the primary data carrier through the pipeline.  It is immutable
after construction — processing stages return new instances rather than
mutating the original.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ObservationMetadata:
    """Extracted FITS header fields relevant to streak detection."""

    exposure_time: Optional[float]       # seconds (EXPTIME / EXPOSURE / ACT / KCT)
    date_obs: Optional[str]              # ISO 8601 (DATE-OBS / DATE / FRAME)
    telescope: Optional[str]             # TELESCOP header
    camera: Optional[str]                # INSTRUME header
    focal_length_mm: Optional[float]     # FOCALLEN header
    lat: Optional[float]                 # SITELAT header (degrees)
    lon: Optional[float]                 # SITELONG header (degrees)
    elevation_m: Optional[float]         # SITEELEV header (metres)
    binning: int                         # XBINNING header (default 1)
    pixel_size_um: Optional[float]       # physical pixel size in microns
    pixel_scale_arcsec: Optional[float]  # derived: 206.265 * pixel_size / focal_length

    # Known camera pixel sizes (microns) — avoids relying on XPIXSZ header for these models
    _CAMERA_PIXEL_SIZES: dict = None  # not a real field; see classmethod

    @classmethod
    def from_fits_header(cls, header: dict) -> "ObservationMetadata":
        """Build metadata from a FITS header dict."""
        exposure_time = (
            header.get("EXPTIME")
            or header.get("EXPOSURE")
            or header.get("ACT")
            or header.get("KCT")
        )
        date_obs = (
            header.get("DATE-OBS")
            or header.get("DATE")
            or header.get("FRAME")
        )
        telescope = header.get("TELESCOP")
        camera = header.get("INSTRUME")
        focal_length_mm = header.get("FOCALLEN")
        lat = header.get("SITELAT")
        lon = header.get("SITELONG")
        elevation_m = header.get("SITEELEV")
        binning = int(header.get("XBINNING") or 1)

        # Pixel size lookup — hardcoded models first, XPIXSZ header as fallback
        _known = {"QHY268M": 3.76, "QHY600": 3.76}
        if camera in _known:
            raw_pixel_um = _known[camera]
        else:
            raw_pixel_um = header.get("XPIXSZ")

        pixel_size_um: Optional[float] = None
        pixel_scale_arcsec: Optional[float] = None
        if raw_pixel_um is not None:
            pixel_size_um = float(raw_pixel_um) * binning
            if focal_length_mm:
                pixel_scale_arcsec = 206.265 * pixel_size_um / float(focal_length_mm)

        return cls(
            exposure_time=float(exposure_time) if exposure_time is not None else None,
            date_obs=str(date_obs) if date_obs is not None else None,
            telescope=str(telescope) if telescope is not None else None,
            camera=str(camera) if camera is not None else None,
            focal_length_mm=float(focal_length_mm) if focal_length_mm is not None else None,
            lat=float(lat) if lat is not None else None,
            lon=float(lon) if lon is not None else None,
            elevation_m=float(elevation_m) if elevation_m is not None else None,
            binning=binning,
            pixel_size_um=pixel_size_um,
            pixel_scale_arcsec=pixel_scale_arcsec,
        )

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None and self.elevation_m is not None


@dataclass
class FitsImage:
    """
    A loaded FITS image with its header metadata.

    ``data`` is always float32, 2-D.  Calibration stages return a new
    FitsImage with the updated array rather than mutating this one.
    """

    source_path: Optional[Path]   # None for synthetic/derived images
    data: np.ndarray               # float32, shape (H, W)
    raw_header: dict               # all FITS header key-value pairs
    metadata: ObservationMetadata

    def derive(self, new_data: np.ndarray) -> "FitsImage":
        """Return a new FitsImage with updated pixel data, keeping all metadata."""
        return FitsImage(
            source_path=self.source_path,
            data=new_data.astype(np.float32),
            raw_header=self.raw_header,
            metadata=self.metadata,
        )
