"""
Adaptive Local Background Estimator.

Divides the image into a tile mesh, estimates per-tile background and
noise via iterative sigma-clipping (robust to stars and cosmic rays),
interpolates both into smooth 2D maps using bicubic interpolation, then
thresholds the residual using local per-pixel SNR.

This allows detection of faint streaks (SNR >= snr_threshold locally)
that global-threshold methods miss because bright image regions inflate
the global sigma estimate.  The algorithm is structurally similar to
SExtractor's background mesh approach and photutils.Background2D.

Algorithm summary
-----------------
1.  Tile the image into an (n_ty x n_tx) mesh.
2.  For each tile: iterative sigma-clipping to estimate local background
    level (mu) and noise (sigma) via MAD.  Tiles with too few survivors
    are marked NaN and interpolated over.
3.  Interpolate the tile-level mu and sigma meshes back to full image
    resolution via bicubic resize (OpenCV INTER_CUBIC) — no scipy needed.
4.  Subtract the background model; divide residual by the noise model to
    obtain a local SNR image.
5.  Threshold: pixels with local_snr >= snr_threshold are foreground.
6.  Morphological close to fill small gaps within streak pixels.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from streakiller.config.schema import BackgroundParams

logger = logging.getLogger(__name__)

# MAD → Gaussian sigma conversion factor (same constant used throughout the project)
_MAD_FACTOR = 1.4826


class AdaptiveLocalEstimator:
    """
    Background estimation via adaptive local mesh + local SNR thresholding.

    All state is derived from the input image and params at call time; the
    estimator instance is stateless and safe to call from multiple threads.
    """

    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        """
        Parameters
        ----------
        data : float32 ndarray, shape (H, W)
        params : BackgroundParams

        Returns
        -------
        binary : uint8 ndarray, shape (H, W), values in {0, 255}
        """
        H, W = data.shape

        # Step 1 & 2: Build tile mesh via iterative sigma-clipping
        bg_map, sigma_map = self._build_mesh(data, params)

        # Degenerate case: every tile was invalid
        n_valid = int(np.sum(np.isfinite(bg_map)))
        if n_valid == 0:
            logger.warning(
                "adaptive_local: all tiles were invalid (image may be saturated); "
                "returning empty binary"
            )
            return np.zeros((H, W), dtype=np.uint8)

        # Step 3: Interpolate mesh maps to full image resolution
        background_model, noise_model = self._interpolate_maps(bg_map, sigma_map, H, W)

        # Step 4: Compute residual and local SNR
        residual = data - background_model
        local_snr = residual / np.maximum(noise_model, 1e-6)

        # Step 5: Threshold by SNR
        binary_raw = (local_snr >= params.adaptive_local_snr_threshold).astype(np.uint8) * 255

        # Step 6: Morphological close to fill gaps within streaks
        k = params.adaptive_local_morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary_raw, cv2.MORPH_CLOSE, kernel)

        foreground_px = int(np.count_nonzero(binary))
        logger.debug(
            "adaptive_local: snr_threshold=%.2f  tile_size=%d  valid_tiles=%d  foreground_pixels=%d",
            params.adaptive_local_snr_threshold,
            params.adaptive_local_tile_size,
            n_valid,
            foreground_px,
        )
        return binary

    # ------------------------------------------------------------------ #
    # Tile mesh construction                                               #
    # ------------------------------------------------------------------ #

    def _build_mesh(
        self, data: np.ndarray, params: BackgroundParams
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide the image into non-overlapping tiles and sigma-clip each one.

        Returns
        -------
        bg_map   : float32 (n_tiles_y, n_tiles_x) — NaN where tile was invalid
        sigma_map: float32 (n_tiles_y, n_tiles_x) — NaN where tile was invalid
        """
        H, W = data.shape
        tile_size = params.adaptive_local_tile_size
        clip_sigma = params.adaptive_local_clip_sigma
        n_iter = params.adaptive_local_n_iterations
        min_pixels = params.adaptive_local_min_tile_pixels

        n_ty = math.ceil(H / tile_size)
        n_tx = math.ceil(W / tile_size)

        if tile_size > min(H, W) // 2:
            logger.warning(
                "adaptive_local: tile_size=%d is large relative to image (%dx%d); "
                "consider reducing adaptive_local_tile_size for better spatial resolution",
                tile_size, H, W,
            )

        bg_map = np.full((n_ty, n_tx), np.nan, dtype=np.float32)
        sigma_map = np.full((n_ty, n_tx), np.nan, dtype=np.float32)

        for r in range(n_ty):
            y0 = r * tile_size
            y1 = min(y0 + tile_size, H)
            for c in range(n_tx):
                x0 = c * tile_size
                x1 = min(x0 + tile_size, W)
                tile_pixels = data[y0:y1, x0:x1].ravel()
                mu, sigma = _sigma_clip(tile_pixels, clip_sigma, n_iter, min_pixels)
                bg_map[r, c] = mu
                sigma_map[r, c] = sigma

        n_valid = int(np.sum(np.isfinite(bg_map)))
        logger.debug(
            "adaptive_local: mesh %dx%d  valid_tiles=%d/%d",
            n_ty, n_tx, n_valid, n_ty * n_tx,
        )
        return bg_map, sigma_map

    # ------------------------------------------------------------------ #
    # Map interpolation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _interpolate_maps(
        bg_map: np.ndarray,
        sigma_map: np.ndarray,
        H: int,
        W: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate tile-level background and sigma meshes to full image size.

        Invalid (NaN) tiles are replaced by the median of valid tiles before
        bicubic resize so that cv2.resize doesn't propagate NaN artefacts.
        Falls back to a scalar broadcast when the mesh has fewer than 2 tiles
        in either dimension (too small for meaningful 2D interpolation).
        """
        def _resize_mesh(mesh: np.ndarray) -> np.ndarray:
            valid_mask = np.isfinite(mesh)
            n_valid = int(valid_mask.sum())

            if n_valid == 0:
                return np.zeros((H, W), dtype=np.float32)

            fill_val = float(np.nanmedian(mesh))
            filled = np.where(valid_mask, mesh, fill_val).astype(np.float32)

            n_ty, n_tx = filled.shape
            if n_ty < 2 or n_tx < 2:
                # Only one tile row or column — scalar broadcast
                logger.warning(
                    "adaptive_local: mesh too small (%dx%d) for 2D interpolation, "
                    "using scalar background",
                    n_ty, n_tx,
                )
                return np.full((H, W), fill_val, dtype=np.float32)

            # Bicubic resize: (n_tx, n_ty) → (W, H) — note cv2 uses (width, height)
            resized = cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)
            return resized.astype(np.float32)

        background_model = _resize_mesh(bg_map)
        noise_model = _resize_mesh(sigma_map)
        noise_model = np.maximum(noise_model, 1e-6)  # guard against non-positive noise
        return background_model, noise_model


# ------------------------------------------------------------------ #
# Module-level helper (no self-state, easy to unit-test in isolation) #
# ------------------------------------------------------------------ #

def _sigma_clip(
    pixels: np.ndarray,
    clip_sigma: float,
    n_iterations: int,
    min_pixels: int,
) -> tuple[float, float]:
    """
    Iterative sigma-clipping on a 1-D array of pixel values.

    Uses median and MAD for robustness against stars, cosmic rays, and
    other bright outliers that would bias a mean/stddev estimate.

    Returns
    -------
    (mu, sigma) : both float
        mu    — median of surviving pixels
        sigma — MAD-normalised noise estimate (consistent with Gaussian sigma)
    Returns (NaN, NaN) if fewer than *min_pixels* pixels survive.
    """
    surviving = pixels.astype(np.float64)

    for _ in range(n_iterations):
        if len(surviving) < min_pixels:
            return float("nan"), float("nan")
        mu = float(np.median(surviving))
        mad = float(np.median(np.abs(surviving - mu)))
        sigma = _MAD_FACTOR * mad + 1e-6  # +epsilon prevents zero-sigma tile
        mask = np.abs(surviving - mu) <= clip_sigma * sigma
        surviving = surviving[mask]

    if len(surviving) < min_pixels:
        return float("nan"), float("nan")

    mu = float(np.median(surviving))
    mad = float(np.median(np.abs(surviving - mu)))
    sigma = _MAD_FACTOR * mad + 1e-6
    return mu, sigma
