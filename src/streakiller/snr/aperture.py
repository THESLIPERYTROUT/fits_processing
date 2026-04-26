"""
Aperture sampling for streak SNR estimation.

For a line segment (x1,y1)-(x2,y2), two rectangular apertures are defined
perpendicular to the streak axis:

  |← off_width →|← off_gap →|← 2*half_width+1 →|← off_gap →|← off_width →|
                                    (streak)

All sampling uses nearest-neighbour pixel lookup on the raw float32 image.
Pixels that fall outside the image boundary are silently excluded.
"""
from __future__ import annotations

import numpy as np


def sample_apertures(
    data: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    half_width: int,
    off_gap: int,
    off_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample on-streak and off-streak (background) pixel values.

    Parameters
    ----------
    data        : float32 (H, W) raw image
    x1, y1     : streak start endpoint (pixels)
    x2, y2     : streak end endpoint (pixels)
    half_width  : on-streak half-width — the aperture spans ±half_width from the centerline
    off_gap     : gap in pixels between the on-streak edge and the background band
    off_width   : width in pixels of each background band

    Returns
    -------
    on_pixels  : 1-D float64 array — pixel values inside the on-streak aperture
    off_pixels : 1-D float64 array — pixel values inside the two off-streak bands
    """
    H, W = data.shape
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1.0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    tx, ty = dx / length, dy / length   # unit tangent
    nx, ny = -ty, tx                    # unit normal (perpendicular, pointing left)

    n_steps = int(round(length)) + 1
    t_vals = np.linspace(0.0, length, n_steps)

    cx = x1 + t_vals * tx   # (n_steps,) along-streak centre x
    cy = y1 + t_vals * ty   # (n_steps,) along-streak centre y

    on_offsets = np.arange(-half_width, half_width + 1)

    off_inner = half_width + off_gap + 1
    off_outer = off_inner + off_width - 1
    off_offsets = np.concatenate([
        np.arange(off_inner, off_outer + 1),
        np.arange(-off_outer, -off_inner + 1),
    ])

    on_pixels = _collect_pixels(data, cx, cy, nx, ny, on_offsets, H, W)
    off_pixels = _collect_pixels(data, cx, cy, nx, ny, off_offsets, H, W)
    return on_pixels, off_pixels


def _collect_pixels(
    data: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    nx: float,
    ny: float,
    offsets: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Vectorised pixel sampling for a set of perpendicular offsets.

    Constructs a (n_offsets, n_steps) grid of (x, y) coordinates, clips to
    image bounds, and returns the surviving pixel values as a flat float64 array.
    """
    # Shape: (n_offsets, n_steps)
    xs = np.round(cx[np.newaxis, :] + offsets[:, np.newaxis] * nx).astype(np.intp)
    ys = np.round(cy[np.newaxis, :] + offsets[:, np.newaxis] * ny).astype(np.intp)
    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return data[ys[mask], xs[mask]].astype(np.float64)
