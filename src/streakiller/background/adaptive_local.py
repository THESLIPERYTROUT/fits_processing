"""
Adaptive Local Background Estimator — hybrid Gaussian high-pass + local noise mesh.

Problem with pure tile-mesh thresholding
-----------------------------------------
A tile-only approach estimates both background level *and* noise from the same
tile sample.  The per-tile noise estimate has sampling variance (σ/√N per tile),
and bicubic interpolation of that noisy sigma mesh creates local dips in the noise
model.  Pixels that fall in those dips get artificially inflated SNR, causing pure
background noise spikes to be flagged as foreground — exactly what is observed on
real background-only plates compared to GaussianBlurEstimator.

Hybrid solution
---------------
Split the two jobs that the tile mesh was trying to do simultaneously:

  1. Background removal  →  Gaussian high-pass filter (deterministic, zero sampling
                             noise — the same step used by GaussianBlurEstimator).

  2. Noise estimation    →  Per-tile MAD on the high-pass residual (locally adaptive,
                             so quiet sky regions keep a low noise floor and faint
                             streaks there are still detectable).

The residual after step 1 is centered at zero everywhere; the tile mesh only sees
the noise distribution (plus any real signals it sigma-clips out).  Crucially the
tile sigma estimates are now measuring *true photon/read noise*, not a mixture of
noise and background gradient — so the interpolated noise model is smooth and
accurate everywhere.

Algorithm summary
-----------------
1.  Gaussian high-pass: subtract a large-kernel Gaussian blur to remove all slowly-
    varying background structure (gradients, vignetting, Moon glow).
2.  Build tile mesh on the high-pass residual: sigma-clip each tile to get a
    per-tile noise (MAD) estimate, robust to stars and cosmic rays in the tile.
3.  Interpolate the sigma mesh to full image resolution via bicubic resize.
4.  Local SNR: divide the high-pass residual by the interpolated noise model.
5.  Threshold pixels where local_snr ≥ snr_threshold.
6.  Morphological close to fill small gaps within streak pixels.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from streakiller.config.schema import BackgroundParams

logger = logging.getLogger(__name__)

_MAD_FACTOR = 1.4826  # MAD → Gaussian σ; same constant used throughout the project


class AdaptiveLocalEstimator:
    """
    Hybrid Gaussian high-pass + local noise mesh background estimator.

    Stateless — safe to call from multiple threads.
    """

    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        """
        Parameters
        ----------
        data   : float32 ndarray, shape (H, W)
        params : BackgroundParams

        Returns
        -------
        binary : uint8 ndarray, shape (H, W), values in {0, 255}
        """
        H, W = data.shape

        # ── Step 1: Gaussian high-pass ────────────────────────────────────────
        # A large Gaussian blur gives a deterministic, sampling-noise-free model
        # of the slowly-varying sky background.  Subtracting it removes gradients
        # and vignetting before we touch the tile mesh — so the tile mesh only
        # has to estimate noise, not noise *and* background simultaneously.
        ksize = params.adaptive_local_gaussian_kernel_size
        if ksize % 2 == 0:
            ksize += 1  # OpenCV requires odd kernel size
        blur_bg = cv2.GaussianBlur(data, (ksize, ksize), 0)
        highpass = data - blur_bg

        # ── Step 2: Per-tile noise estimation on the high-pass residual ───────
        # Tile pixel values are now centered near zero; stars / cosmic rays are
        # clipped by iterative sigma-clipping so they don't inflate the local σ.
        # We discard bg_map (its values are ~0 by construction) and keep only
        # sigma_map, which carries the spatially-varying noise floor.
        _, sigma_map = self._build_mesh(highpass, params)

        n_valid = int(np.sum(np.isfinite(sigma_map)))
        if n_valid == 0:
            logger.warning(
                "adaptive_local: all tiles were invalid; returning empty binary"
            )
            return np.zeros((H, W), dtype=np.uint8)

        # ── Step 3: Interpolate sigma mesh → full-resolution noise model ──────
        noise_model = self._interpolate_sigma(sigma_map, H, W)

        # ── Step 4: Local SNR ─────────────────────────────────────────────────
        local_snr = highpass / np.maximum(noise_model, 1e-6)

        # ── Step 5: Threshold ─────────────────────────────────────────────────
        binary_raw = (local_snr >= params.adaptive_local_snr_threshold).astype(np.uint8) * 255

        # ── Step 6: Morphological close ───────────────────────────────────────
        k = params.adaptive_local_morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel)

        logger.debug(
            "adaptive_local: gaussian_kernel=%d  snr_threshold=%.2f  "
            "tile_size=%d  valid_tiles=%d  foreground_pixels=%d",
            ksize,
            params.adaptive_local_snr_threshold,
            params.adaptive_local_tile_size,
            n_valid,
            int(np.count_nonzero(binary)),
        )
        return binary

    # ------------------------------------------------------------------ #
    # Tile mesh construction                                               #
    # ------------------------------------------------------------------ #

    def _build_mesh(
        self, data: np.ndarray, params: BackgroundParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide *data* into non-overlapping tiles and sigma-clip each one.

        When called on the Gaussian high-pass residual, bg_map values are ~0
        (the high-pass already removed the background); only sigma_map is used
        by the caller.

        Returns
        -------
        bg_map    : float32 (n_ty, n_tx) — per-tile median, NaN where invalid
        sigma_map : float32 (n_ty, n_tx) — per-tile MAD noise, NaN where invalid
        """
        H, W = data.shape
        tile_size = params.adaptive_local_tile_size

        if tile_size > min(H, W) // 2:
            logger.warning(
                "adaptive_local: tile_size=%d is large relative to image (%dx%d); "
                "consider reducing adaptive_local_tile_size",
                tile_size, H, W,
            )

        n_ty = math.ceil(H / tile_size)
        n_tx = math.ceil(W / tile_size)

        bg_map    = np.full((n_ty, n_tx), np.nan, dtype=np.float32)
        sigma_map = np.full((n_ty, n_tx), np.nan, dtype=np.float32)

        clip_sigma = params.adaptive_local_clip_sigma
        n_iter     = params.adaptive_local_n_iterations
        min_pixels = params.adaptive_local_min_tile_pixels

        for r in range(n_ty):
            y0, y1 = r * tile_size, min((r + 1) * tile_size, H)
            for c in range(n_tx):
                x0, x1 = c * tile_size, min((c + 1) * tile_size, W)
                mu, sigma = _sigma_clip(
                    data[y0:y1, x0:x1].ravel(), clip_sigma, n_iter, min_pixels
                )
                bg_map[r, c]    = mu
                sigma_map[r, c] = sigma

        logger.debug(
            "adaptive_local: mesh %dx%d  valid_tiles=%d/%d",
            n_ty, n_tx,
            int(np.sum(np.isfinite(bg_map))),
            n_ty * n_tx,
        )
        return bg_map, sigma_map

    # ------------------------------------------------------------------ #
    # Map interpolation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _interpolate_sigma(sigma_map: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Interpolate the tile-level sigma mesh to full image resolution and
        clamp to a strictly-positive floor.
        """
        _, noise_model = AdaptiveLocalEstimator._interpolate_maps(
            np.zeros_like(sigma_map), sigma_map, H, W
        )
        return noise_model

    @staticmethod
    def _interpolate_maps(
        bg_map: np.ndarray,
        sigma_map: np.ndarray,
        H: int,
        W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate two tile-level meshes to full image size via bicubic resize.

        NaN tiles are filled with the global nanmedian of valid tiles before
        resizing so that cv2.resize does not propagate NaN artefacts.
        Falls back to a scalar broadcast when the mesh is too small (< 2 tiles
        in either dimension) for 2-D interpolation.
        """
        def _resize_mesh(mesh: np.ndarray) -> np.ndarray:
            valid_mask = np.isfinite(mesh)
            if not valid_mask.any():
                return np.zeros((H, W), dtype=np.float32)

            fill_val = float(np.nanmedian(mesh))
            filled   = np.where(valid_mask, mesh, fill_val).astype(np.float32)

            n_ty, n_tx = filled.shape
            if n_ty < 2 or n_tx < 2:
                logger.warning(
                    "adaptive_local: mesh too small (%dx%d) for 2D interpolation; "
                    "using scalar fallback",
                    n_ty, n_tx,
                )
                return np.full((H, W), fill_val, dtype=np.float32)

            return cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        background_model = _resize_mesh(bg_map)
        noise_model      = _resize_mesh(sigma_map)
        noise_model      = np.maximum(noise_model, 1e-6)
        return background_model, noise_model


# ------------------------------------------------------------------ #
# Module-level sigma-clipping helper                                   #
# ------------------------------------------------------------------ #

def _sigma_clip(
    pixels: np.ndarray,
    clip_sigma: float,
    n_iterations: int,
    min_pixels: int,
) -> tuple[float, float]:
    """
    Iterative MAD-based sigma-clipping on a 1-D pixel array.

    Uses median and MAD (not mean/stddev) so that stars, cosmic rays, and
    streak pixels are rejected without biasing the background estimate.

    Returns (NaN, NaN) if fewer than *min_pixels* pixels survive.
    """
    surviving = pixels.astype(np.float64)

    for _ in range(n_iterations):
        if len(surviving) < min_pixels:
            return float("nan"), float("nan")
        mu  = float(np.median(surviving))
        mad = float(np.median(np.abs(surviving - mu)))
        sigma = _MAD_FACTOR * mad + 1e-6
        surviving = surviving[np.abs(surviving - mu) <= clip_sigma * sigma]

    if len(surviving) < min_pixels:
        return float("nan"), float("nan")

    mu  = float(np.median(surviving))
    mad = float(np.median(np.abs(surviving - mu)))
    return mu, _MAD_FACTOR * mad + 1e-6
