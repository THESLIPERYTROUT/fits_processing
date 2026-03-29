"""
StreakDetector — runs OpenCV HoughLinesP on a binary image.

This class is responsible *only* for Hough detection.  Filtering is done
by FilterChain downstream.  Returns a RawDetection that always contains a
valid array (never None), fixing the original 2-tuple/4-tuple return bug.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from streakiller.config.schema import HoughParams
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    """Output of a single Hough detection pass."""

    lines: np.ndarray            # shape (N, 1, 4), dtype int32; N==0 when none found
    binary_image: np.ndarray     # uint8 (H, W) — the Hough input
    normalized_display: np.ndarray  # uint8 (H, W) — for visualisation


class StreakDetector:
    """
    Detects line segments in a binary image using HoughLinesP.

    Parameters are taken from HoughParams so they are configurable without
    touching the code.
    """

    def __init__(self, params: HoughParams) -> None:
        self._params = params

    def detect(self, binary: np.ndarray, source_data: np.ndarray, min_line_length: float) -> RawDetection:
        """
        Run Hough line detection on *binary*.

        Parameters
        ----------
        binary : uint8 ndarray, shape (H, W)
            Foreground mask from a BackgroundEstimator.
        source_data : float32 ndarray, shape (H, W)
            Original float image used to build the display image.
        min_line_length : float
            Minimum segment length in pixels.

        Returns
        -------
        RawDetection with ``lines`` of shape (N, 1, 4).  N == 0 if Hough
        found nothing — callers should not check for None.
        """
        p = self._params
        theta = np.pi / 180.0 * p.theta_deg

        raw = cv2.HoughLinesP(
            binary,
            rho=p.rho,
            theta=theta,
            threshold=p.threshold,
            minLineLength=min_line_length,
            maxLineGap=p.max_line_gap,
        )

        if raw is None:
            logger.info("HoughLinesP found no lines (minLineLength=%.1f)", min_line_length)
            lines = np.empty((0, 1, 4), dtype=np.int32)
        else:
            lines = raw
            logger.info("HoughLinesP detected %d raw lines", len(lines))

        return RawDetection(
            lines=lines,
            binary_image=binary,
            normalized_display=normalize_for_display(source_data),
        )
