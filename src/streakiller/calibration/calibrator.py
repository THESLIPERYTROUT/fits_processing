"""
Image calibration: dark subtraction and flat-field division.

CalibrationStep loads master frames once via load_frames() and can then be
applied to many images via apply() without hitting disk again.  apply()
returns a new FitsImage — it never modifies the input or writes files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from streakiller.models.fits_image import FitsImage

logger = logging.getLogger(__name__)


class CalibrationError(ValueError):
    """Raised when calibration cannot be performed."""


class CalibrationStep:
    """
    Applies dark subtraction and flat-field division to a FitsImage.

    Usage::

        step = CalibrationStep(Path("calibration_frames"))
        step.load_frames()          # load mdark.fits and mflat.fits once
        calibrated = step.apply(image)   # returns new FitsImage
    """

    def __init__(self, calibration_dir: Path) -> None:
        self._dir = Path(calibration_dir)
        self._dark: Optional[np.ndarray] = None
        self._flat: Optional[np.ndarray] = None

    def load_frames(self) -> None:
        """
        Load master dark and master flat from the calibration directory.

        Expects:
          <calibration_dir>/mdark.fits
          <calibration_dir>/mflat.fits

        Raises CalibrationError if either file is missing or unreadable.
        """
        dark_path = self._dir / "mdark.fits"
        flat_path = self._dir / "mflat.fits"

        for label, path in (("master dark", dark_path), ("master flat", flat_path)):
            if not path.exists():
                raise CalibrationError(f"Calibration frame not found: {path}")

        try:
            self._dark = self._load_frame(dark_path, "dark")
            self._flat = self._load_frame(flat_path, "flat")
        except Exception as exc:
            raise CalibrationError(f"Failed to load calibration frames: {exc}") from exc

        logger.info("Loaded calibration frames from %s", self._dir)

    def apply(self, image: FitsImage) -> FitsImage:
        """
        Subtract dark and divide flat.  Returns a new FitsImage.

        Raises CalibrationError if frames have not been loaded or if the
        image shape does not match the calibration frames.
        """
        if self._dark is None or self._flat is None:
            raise CalibrationError("load_frames() must be called before apply()")

        data = image.data
        self._check_shapes(data, self._dark, "master dark")
        self._check_shapes(data, self._flat, "master flat")

        dark_sub = self._subtract_dark(data, self._dark)
        calibrated = self._divide_flat(dark_sub, self._flat)

        logger.info("Calibration applied to %s", image.source_path)
        return image.derive(calibrated)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _subtract_dark(self, data: np.ndarray, dark: np.ndarray) -> np.ndarray:
        return (data - dark).astype(np.float32)

    def _divide_flat(self, data: np.ndarray, flat: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.true_divide(data, flat)
            result[~np.isfinite(result)] = 0.0
        return result.astype(np.float32)

    def _check_shapes(
        self, data: np.ndarray, frame: np.ndarray, label: str
    ) -> None:
        if data.shape != frame.shape:
            raise CalibrationError(
                f"Image shape {data.shape} does not match {label} shape {frame.shape}"
            )

    def _load_frame(self, path: Path, label: str) -> np.ndarray:
        from astropy.io import fits as astropy_fits

        try:
            with astropy_fits.open(path) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        return hdu.data.astype(np.float32)
        except Exception as exc:
            raise CalibrationError(f"Cannot read {label} from {path}: {exc}") from exc

        raise CalibrationError(f"No image data in {label} file: {path}")
