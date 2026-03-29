#!/usr/bin/env python3
"""
fits_surface.py

Load a FITS image and plot it as a smooth 3D surface.

Usage:
    python fits_surface.py path/to/image.fits \
        --downsample 4 \
        --sigma 2.0 \
        --output surface.png

If --output is not given, it will just show the plot (if a display is available).
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from astropy.io import fits
import scipy

# Try to import scipy for Gaussian smoothing (optional)
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_fits_data(path):
    """Load the primary image data from a FITS file as a 2D float array."""
    with fits.open(path) as hdul:
        data = hdul[1].data

    if data is None:
        raise ValueError("No image data found in primary HDU.")

    # Ensure 2D (some FITS may be 3D, e.g. (n, y, x))
    if data.ndim > 2:
        # Take the first slice along the extra axes
        while data.ndim > 2:
            data = data[0]

    data = np.array(data, dtype=float)

    # Replace NaNs with median for plotting
    if np.isnan(data).any():
        median_val = np.nanmedian(data)
        data = np.where(np.isnan(data), median_val, data)

    return data


def downsample(data, factor):
    """Downsample the image by an integer factor using simple slicing."""
    if factor <= 1:
        return data
    return data[::factor, ::factor]


def smooth(data, sigma):
    """Apply Gaussian smoothing if scipy is available; otherwise return original."""
    if sigma <= 0:
        return data

    if not HAS_SCIPY:
        print(
            "[WARNING] scipy is not installed, skipping smoothing. "
            "Install scipy or set --sigma 0.",
            file=sys.stderr,
        )
        return data

    return gaussian_filter(data, sigma=sigma)


def make_surface_plot(data, output_path=None, show=True, title=None):
    """
    Create a 3D surface plot from a 2D numpy array.

    If output_path is given, saves the figure there.
    If show is True, calls plt.show().
    """
    ny, nx = data.shape

    # Create coordinate grid
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        X, Y, data,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Intensity")
    if title is not None:
        ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Intensity")

    plt.tight_layout()

    
    print(f"[INFO] Saving figure to ")
    plt.savefig("smoothmap" , dpi=150)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a FITS image as a smooth 3D surface."
    )
    parser.add_argument(
        "fits_path",
        help="Path to the FITS image file.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Integer downsampling factor (default: 4). Use 1 for no downsampling.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in pixels (default: 2.0). Use 0 for no smoothing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. surface.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (useful on headless systems).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading FITS file: {args.fits_path}")
    data = load_fits_data(args.fits_path)

    print(f"[INFO] Original shape: {data.shape}")
    data = downsample(data, args.downsample)
    print(f"[INFO] After downsampling (factor={args.downsample}): {data.shape}")

    if args.sigma > 0:
        print(f"[INFO] Applying Gaussian smoothing (sigma={args.sigma})")
    data = smooth(data, args.sigma)

    title = f"3D Surface: {args.fits_path}"
    make_surface_plot(
        data,
        output_path=args.output,
        show=not args.no_show,
        title=title,
    )


if __name__ == "__main__":
    main()
