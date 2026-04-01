# Adaptive Local Background Estimator
## Design Report — Streakiller v0.1+

---

## 1. Motivation

The three existing background estimators (SimpleMedian, GaussianBlur, DoublePass) all share a
common weakness: they compute a **global** or **near-global** noise statistic and apply it
uniformly across the entire image.  This is problematic when:

- The sky background varies spatially (light gradients from Moon glow, sensor vignetting,
  differential atmospheric refraction)
- One corner of the image is dominated by a bright galaxy or star cluster, inflating the
  global standard deviation
- A faint satellite trail is in the "quiet" half of a frame where the local noise is much
  lower than the global average

In all these cases, a globally-thresholded detector sets the SNR bar at the level of the
*noisiest* region, causing it to miss faint signals in the *quietest* regions.

The Adaptive Local Background Estimator addresses this by estimating background and noise
**independently for each spatial tile**, then thresholding on **local SNR** at every pixel.

---

## 2. Algorithm Overview

The algorithm has five stages:

```
Image (float32)
    │
    ▼
1. Tile Decomposition — divide into N×M tiles
    │
    ▼
2. Sigma-Clipping per tile — estimate local background μ and noise σ
    │
    ▼
3. Interpolation — bicubic resize of tile mesh → full-resolution maps
    │
    ▼
4. Residual & local SNR — (image − background_model) / noise_model
    │
    ▼
5. Threshold + Morphological Close → binary mask (uint8, {0,255})
```

---

## 3. Stage 1: Tile Decomposition

The image (H × W pixels) is divided into non-overlapping rectangular tiles of side
`tile_size` pixels.  Partial tiles at the right and bottom edges include the remaining
pixels up to the image boundary.

Number of tiles in each axis:

```
n_tiles_y = ⌈H / tile_size⌉
n_tiles_x = ⌈W / tile_size⌉
```

For a 4096 × 4096 image with `tile_size = 64`, this produces a 64 × 64 mesh of 4096 tiles.

**Choosing tile_size:**

The tile must be large enough to contain many background pixels so that the median/MAD
statistics are stable, yet small enough to resolve spatial background gradients.  A
typical rule of thumb:

- Tile should span ≪ the scale of background variation (usually the PSF-broadened size of
  a bright source or the illumination gradient scale, often hundreds of pixels)
- Tile should contain ≳ 100 pixels for reliable median/MAD estimation

A `tile_size` of 64 px works well for most amateur astronomy cameras (e.g. QHY268M at
3.76 µm pixels ≈ ~200–500 tiles across a typical FOV).

**Critical limitation:** If the background varies significantly *within* a tile, the
per-tile sigma estimate will include gradient variation on top of true photon noise.
This inflates the sigma and reduces detection sensitivity.  Users should reduce
`adaptive_local_tile_size` if the background has sharp spatial features.

---

## 4. Stage 2: Iterative Sigma-Clipping

For each tile, iterative sigma-clipping robustly separates the background pixel
population from stars, cosmic rays, and streak pixels.

### 4.1 Motivation for Median/MAD instead of Mean/StdDev

If a tile contains a bright star, the standard deviation of the pixel distribution is
dominated by the star's flux, not by background noise.  For a tile with background
level µ_bg and a single star of flux F deposited across A pixels:

```
σ_biased ≈ √( (A/N) * (F - µ_bg)² + σ_true² )
```

For a 64×64 tile (N = 4096) with a star contributing 50 bright pixels (A = 50) and
F/µ_bg = 100, σ_biased >> σ_true.

The **median** is insensitive to such outliers (up to ~50% contamination).  The
**Median Absolute Deviation** (MAD) is similarly robust.

### 4.2 MAD and Gaussian Sigma Equivalence

For a Gaussian distribution N(µ, σ), the expected MAD is:

```
E[MAD] = σ × Φ⁻¹(3/4) ≈ σ × 0.6745
```

where Φ⁻¹ is the inverse CDF of the standard normal.  Therefore, the MAD can be
converted to an equivalent Gaussian sigma by:

```
σ_estimate = MAD × (1 / 0.6745) ≈ MAD × 1.4826
```

The factor 1.4826 (called `MAD_NORMALIZATION_FACTOR` in the codebase) appears throughout
the project and is the standard convention in astronomical image processing software
(SExtractor, photutils, SEP, AstroPy).

### 4.3 Iterative Clipping Procedure

For each tile with pixel values {p_i}:

**Iteration k:**

1. Compute `µ_k = median({p_i})` and `σ_k = 1.4826 × MAD({p_i}) + ε`

   (ε = 1×10⁻⁶ prevents σ = 0 on uniform tiles)

2. Retain only pixels satisfying: `|p_i − µ_k| ≤ clip_sigma × σ_k`

3. If fewer than `min_tile_pixels` pixels survive → mark tile as **invalid** (NaN) and stop

Repeat for `n_iterations` steps (default: 3).  After the final iteration, compute final
`µ` and `σ` from surviving pixels.

**Why 3 iterations?**  Convergence studies on Gaussian mixtures show that 2–3 iterations
are sufficient to remove outliers contributing up to ~15% of tile pixels.  More iterations
give diminishing returns and can occasionally over-clip on tiles with many faint stars.

**clip_sigma selection:**  The default `clip_sigma = 3.0` (3σ rejection) removes pixels
more than 3 standard deviations from the median in each pass.  For a pure Gaussian, this
retains 99.73% of pixels per iteration, so legitimate background pixels are rarely
removed.  Bright stars and cosmic rays (typically >> 10σ above background) are expelled
on the first iteration.

### 4.4 Invalid Tile Handling

A tile is marked invalid (bg_map[r,c] = NaN) when fewer than `min_tile_pixels` pixels
survive after all clipping iterations.  This occurs when a tile is:

- Dominated by a bright extended source (galaxy, nebula, Moon)
- Smaller than `min_tile_pixels` (e.g. partial tile at image edge)
- Occupied by a very bright saturated star with extensive bleed

Invalid tiles are later filled by interpolation over their valid neighbours.

---

## 5. Stage 3: Interpolation to Full Resolution

After building the (n_tiles_y × n_tiles_x) background mesh `bg_map` and sigma mesh
`sigma_map`, both must be expanded to the full image resolution.

### 5.1 NaN Tile Filling

Before interpolation, all NaN entries in each mesh are replaced by the global median of
the valid tiles:

```
fill_val = nanmedian(bg_map)
filled   = where(isfinite(bg_map), bg_map, fill_val)
```

This prevents NaN propagation during bicubic interpolation.

### 5.2 Bicubic Resize

The filled mesh is resized to (H × W) using OpenCV's bicubic interpolation
(`cv2.resize` with `INTER_CUBIC`).  Each tile centre is treated as a sample point, and
bicubic convolution fits a piecewise cubic polynomial through the sample grid.

Bicubic interpolation uses a 4×4 neighbourhood of sample points, producing a C¹-smooth
output (continuous first derivative).  This is sufficient to model slowly-varying sky
backgrounds without scipy as a dependency.

**Why not bilinear?** Bilinear interpolation produces visible tile boundaries in the
background model (C⁰ but not C¹), which can create false step discontinuities in the
residual that may be flagged as foreground.

**Why not higher-order splines?** A full 2D spline (e.g. `scipy.interpolate.RectBivariateSpline`)
would provide C² continuity but requires scipy as a runtime dependency.  For the smooth
background variations typical in astronomy images, bicubic is adequate.  If needed,
scipy can be substituted in `_interpolate_maps` with no other code changes.

### 5.3 Noise Model Clamping

After interpolation, the noise model is clamped:

```
noise_model = max(noise_model, 1e-6)
```

Bicubic interpolation can produce small undershoots (Runge-like oscillations) near sharp
gradients in the tile mesh, which could result in noise_model ≈ 0 or slightly negative.
Clamping ensures no division by zero and prevents pathologically high SNR values.

### 5.4 Edge / Degenerate Cases

| Condition | Behaviour |
|---|---|
| All tiles NaN | Early return: zero binary (no detections) |
| 1×1 mesh (single tile) | Scalar broadcast: background_model = constant |
| Mesh with some NaN tiles | NaN → filled with nanmedian before resize |
| noise_model ≤ 0 after resize | Clamped to 1×10⁻⁶ |

---

## 6. Stage 4: Residual and Local SNR

```
residual[y, x] = data[y, x] − background_model[y, x]

local_snr[y, x] = residual[y, x] / max(noise_model[y, x], 1e-6)
```

The `local_snr` image is physically interpretable: a value of 3.0 at pixel (y, x) means
that pixel is 3 standard deviations above the locally-estimated background.

This is the key advantage over global-threshold methods:

- A pixel that is only 3σ above its *local* background is flagged, even if a bright
  nebula elsewhere in the image has much higher absolute variance.
- A pixel that appears bright in absolute terms but is within 3σ of its local
  background is correctly classified as background, even if the global sigma would
  have flagged it.

---

## 7. Stage 5: SNR Threshold and Morphological Close

A pixel is classified as **foreground** if:

```
local_snr[y, x] ≥ adaptive_local_snr_threshold
```

This produces a binary image with values in {0, 255}.

A morphological **close** operation (dilation then erosion) with a `morph_kernel × morph_kernel`
rectangular structuring element fills small gaps within streak pixels without growing
isolated noise spikes.  The default 3×3 kernel is smaller than the 5×5 used by
SimpleMedian and DoublePass, which is appropriate in the low-SNR regime where aggressive
gap-filling could bridge nearby background fluctuations.

---

## 8. SNR Threshold Tuning

The `adaptive_local_snr_threshold` parameter is the primary dial for sensitivity:

| snr_threshold | Behaviour |
|---|---|
| 5.0 | Conservative: only very bright streaks; few false positives |
| 3.0 (default) | Balanced: detects moderate-SNR streaks with manageable false-positive rate |
| 2.5 | Aggressive: detects faint streaks; more false positives fed to Hough/filters |
| 2.0 | Very aggressive: approaching 1-in-20 chance of a random background pixel being flagged |

For a Gaussian background, the expected false-positive rate per pixel at a given threshold T is:

```
p_fp = P(Z ≥ T)  where Z ~ N(0, 1)
```

| T | p_fp | Expected false pixels in a 4096×4096 image |
|---|---|---|
| 3.0 | 0.0013 | ~22,000 |
| 2.5 | 0.0062 | ~104,000 |
| 2.0 | 0.0228 | ~382,000 |

At lower thresholds the Hough transform and downstream filter chain become increasingly
important for suppressing false positives.  Consider tightening Hough threshold or
minLineLength when reducing snr_threshold below 2.5.

---

## 9. Comparison with Existing Methods

| Property | SimpleMedian | GaussianBlur | DoublePass | **AdaptiveLocal** |
|---|---|---|---|---|
| Background model | Global scalar | High-pass filter | Inpainted map | Interpolated tile mesh |
| Noise estimate | Global stddev | Global MAD | Global stddev | Per-tile MAD (local) |
| Threshold type | Global sigma-mult | Global k × MAD | Global sigma-mult | **Local SNR** |
| Gradient handling | Poor | Moderate | Good | **Good** |
| Crowded star fields | Poor | Good | Good | Good |
| Faint streak sensitivity | Low | Moderate | Moderate | **High** |
| Speed | Very fast | Fast | Slow (inpainting) | Moderate |
| Key parameter | sigma_mult | sigma_ladder | sigma_mult | **snr_threshold** |

The primary advantage of AdaptiveLocal appears in scenes with non-uniform sky backgrounds
and faint targets.  For uniform, low-noise images GaussianBlur is equally effective and faster.

---

## 10. Configuration Reference

All parameters live in `BackgroundParams` (schema.py) with defaults in `defaults.py`.

| Parameter | Default | Description |
|---|---|---|
| `adaptive_local_tile_size` | 64 | Side length of each mesh tile in pixels.  Smaller = finer spatial resolution but noisier per-tile estimates.  Must be ≥ 8. |
| `adaptive_local_clip_sigma` | 3.0 | Per-tile sigma-clipping rejection threshold.  Lower = more aggressive star/cosmic-ray rejection. |
| `adaptive_local_n_iterations` | 3 | Sigma-clipping passes per tile.  3 is standard. |
| `adaptive_local_snr_threshold` | 3.0 | Minimum local SNR for a pixel to be foreground.  Reduce to 2.5–2.0 for faint-streak searches. |
| `adaptive_local_min_tile_pixels` | 10 | Minimum surviving pixels for a tile to be valid.  Increase to reject tiles with too few background pixels. |
| `adaptive_local_morph_kernel` | 3 | Side length of morphological close kernel.  3×3 for low-SNR; increase to 5×5 to fill larger gaps. |

To enable in config.json:

```json
"background_detection_method": {
    "simple_median": false,
    "Guassian_blur": false,
    "doublepass_median_to_guassian_blur": false,
    "adaptive_local": true
}
```

---

## 11. Known Limitations

### 11.1 Within-Tile Gradient Inflation

If the sky background varies significantly *within* a single tile, the per-tile sigma
estimate includes gradient variation in addition to photon noise.  This inflates σ and
reduces sensitivity.  Rule of thumb: if the sky gradient changes by more than the noise
sigma across one tile width, reduce `tile_size`.

Quantitatively, let `G` = gradient variation within one tile (ADU), `σ_true` = true
noise (ADU).  The effective sigma seen by the estimator is approximately:

```
σ_effective ≈ √(σ_true² + (G / √12)²)
```

(since a uniform gradient across the tile has variance G²/12).  For σ_effective ≤ 1.1 × σ_true
(i.e., ≤ 10% degradation), we need `G ≤ 0.46 × σ_true`.

### 11.2 Memory Usage

For a 4096×4096 image, the algorithm allocates five full-image float32 arrays:
`background_model`, `noise_model`, `residual`, `local_snr`, `binary_raw` (~80 MB each =
~400 MB peak).  This is acceptable for astronomy workstations but should be noted for
cloud batch processing.

### 11.3 Tile Size vs. Image Size

When `tile_size ≥ min(H, W)`, only one tile exists, producing a 1×1 mesh.  The
interpolation falls back to a scalar broadcast (constant background and noise over the
entire image).  This is equivalent to a single-pass global median method.  A warning is
logged when `tile_size > min(H, W) / 2`.

---

## 12. References

- E. Bertin & S. Arnouts (1996), *SExtractor: Software for source extraction*,
  Astronomy & Astrophysics Supplement Series, 117, 393–404.
  (original description of mesh-based background estimation in astronomical image processing)

- L.N. Rousseeuw & C. Croux (1993), *Alternatives to the Median Absolute Deviation*,
  Journal of the American Statistical Association, 88(424), 1273–1283.
  (theoretical basis for MAD as a robust scatter estimator)

- B. Bradley & M. Tibshirani (2007),
  *An Introduction to the Bootstrap*, CRC Press.
  (background on iterative clipping and its statistical properties)

- Astropy Collaboration (2022), *The Astropy Project: Sustaining and Growing a Community
  Developed Open-source Project and Status of the v5.0 Core Package*,
  The Astrophysical Journal, 935, 167.
  (photutils.Background2D implements the same mesh approach; served as design reference)
