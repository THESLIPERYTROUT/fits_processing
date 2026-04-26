"""
Per-streak SNR estimator using aperture photometry.

Algorithm
---------
For each detected streak (x1, y1, x2, y2):

1. Sample on-streak pixels: a strip of width (2*half_width_px + 1) centred on
   the streak axis, running its full length.
2. Sample off-streak pixels: two bands of width off_width_px on either side of
   the streak, separated from it by a gap of off_gap_px.
3. Compute:
     background = median(off_pixels)          — robust sky level
     noise      = MAD(off_pixels) * 1.4826    — robust per-pixel noise sigma
     signal     = mean(on_pixels) - background — excess above sky
     SNR        = signal / noise

SNR is NaN when fewer than min_off_pixels background samples are available
(e.g. the streak lies near an image edge).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from streakiller.config.schema import SnrParams
from streakiller.models.streak import StreakSNR
from streakiller.snr.aperture import sample_apertures

logger = logging.getLogger(__name__)

_MAD_FACTOR = 1.4826   # converts MAD to a consistent Gaussian sigma estimate


class StreakSNREstimator:
    """
    Computes per-streak SNR from the raw float32 image.

    Stateless — safe for concurrent use from multiple threads or processes.
    """

    def estimate_all(
        self,
        data: np.ndarray,
        lines: np.ndarray,
        params: SnrParams,
    ) -> list[StreakSNR]:
        """
        Estimate SNR for every detected streak.

        Parameters
        ----------
        data   : float32 (H, W) raw image (before binary thresholding)
        lines  : (N, 1, 4) int32 array — [x1, y1, x2, y2] per streak (HoughLinesP format)
        params : SnrParams controlling aperture geometry and minimum sample count

        Returns
        -------
        List of StreakSNR, one per streak, in the same order as *lines*.
        """
        results: list[StreakSNR] = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = (
                int(line[0, 0]), int(line[0, 1]),
                int(line[0, 2]), int(line[0, 3]),
            )
            results.append(self._estimate_one(data, i, x1, y1, x2, y2, params))
        return results

    # ------------------------------------------------------------------ #
    # Private                                                             #
    # ------------------------------------------------------------------ #

    def _estimate_one(
        self,
        data: np.ndarray,
        index: int,
        x1: int, y1: int,
        x2: int, y2: int,
        params: SnrParams,
    ) -> StreakSNR:
        on_px, off_px = sample_apertures(
            data, x1, y1, x2, y2,
            half_width=params.half_width_px,
            off_gap=params.off_gap_px,
            off_width=params.off_width_px,
        )

        n_on = len(on_px)
        n_off = len(off_px)

        if n_off < params.min_off_pixels:
            logger.debug(
                "streak %d: only %d background pixels (need %d) — SNR is NaN",
                index, n_off, params.min_off_pixels,
            )
            return StreakSNR(
                streak_index=index,
                snr=float("nan"),
                signal=float("nan"),
                noise=float("nan"),
                n_on_pixels=n_on,
                n_off_pixels=n_off,
            )

        background = float(np.median(off_px))
        mad = float(np.median(np.abs(off_px - background)))
        noise = _MAD_FACTOR * mad + 1e-6

        signal = float(np.mean(on_px)) - background if n_on > 0 else float("nan")
        snr = signal / noise if not math.isnan(signal) else float("nan")

        logger.debug(
            "streak %d: signal=%.1f  noise=%.2f  SNR=%.2f  n_on=%d  n_off=%d",
            index, signal, noise, snr, n_on, n_off,
        )
        return StreakSNR(
            streak_index=index,
            snr=snr,
            signal=signal,
            noise=noise,
            n_on_pixels=n_on,
            n_off_pixels=n_off,
        )
