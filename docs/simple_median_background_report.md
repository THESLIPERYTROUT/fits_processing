# Simple Median Background Estimator
## Design Report — Streakiller

---

## 1. Purpose

The SimpleMedianEstimator is the fastest and most transparent background detection
method in the pipeline.  It answers a single question: *is this pixel significantly
brighter than the typical pixel in the image?*

It is best suited for images with:
- Uniform, flat sky backgrounds (no gradient, no vignetting)
- Low star density (stars occupy << 5% of pixels)
- High-SNR targets that are clearly above background

---

## 2. Algorithm

The method completes in four steps:

```
data (float32)
    │
    ▼
1. global median  + global stddev
    │
    ▼
2. threshold = median + sigma_mult × stddev
    │
    ▼
3. binary mask: pixel ≥ threshold → 255, else 0
    │
    ▼
4. morphological close (5 × 5 kernel)
    │
    ▼
binary (uint8, {0, 255})
```

### Step 1 — Global statistics

```
μ = median(all pixels)
σ = stddev(all pixels)
```

Both are computed over every pixel in the image, including stars, cosmic rays,
and any streak signal.  The median is used as the background level estimator
rather than the mean because it is robust to the bright-tail contamination from
stars: for a distribution where > 50% of pixels are background, the median falls
in the background population.  The standard deviation is not robust, but for
clean, low-contamination frames it is a serviceable noise estimator.

### Step 2 — Threshold

```
T = μ + sigma_mult × σ
```

`sigma_mult` (default 1.2) controls how far above the background level a pixel
must be to count as foreground.

For a pure Gaussian background N(μ, σ), the fraction of background pixels that
exceed the threshold is:

```
p_fp = P(Z > sigma_mult)  where Z ~ N(0, 1)
```

| sigma_mult | p_fp per pixel | Expected FP in 4096 × 4096 image |
|---|---|---|
| 1.2 | 11.5% | ~1,930,000 |
| 2.0 | 2.3% | ~386,000 |
| 3.0 | 0.13% | ~22,000 |

At the default of 1.2, roughly 1 in 9 background pixels is flagged.  The intent
is to cast a wide net and rely on the Hough transform and downstream filter chain
to reject isolated noise pixels — only long, straight, connected lines survive.

### Step 3 — Binary thresholding

```
binary[y, x] = 255  if data[y, x] ≥ T
binary[y, x] = 0    otherwise
```

### Step 4 — Morphological close

A 5 × 5 rectangular structuring element closes small gaps within high-signal
regions.  The closing operation is dilation (grows foreground regions by the kernel
radius) followed by erosion (shrinks back), so isolated single pixels are also
removed if they are not part of a larger connected region.

---

## 3. Why Median Rather Than Mean

The mean is dominated by the bright tail of the pixel distribution.  For a frame
with a bright star contributing pixels at 10 × the background level:

```
mean  ≈ μ_bg + (N_star / N_total) × F_star
median ≈ μ_bg    (if N_star < N_total / 2)
```

A star covering only 0.1% of pixels (still large for an astronomy image) can
shift the mean by tens of ADU, raising the threshold and causing faint streaks
near that star to be missed.  The median is unaffected unless > 50% of pixels
are contaminated.

---

## 4. Limitations

### 4.1 Non-robust noise estimate

The standard deviation σ is not resistant to outliers.  A frame with many bright
stars will have an inflated σ, raising the threshold globally.  As a rough rule:
if more than ~5% of pixels are significantly above background, SimpleMedian's σ
estimate becomes unreliable.

For those cases, GaussianBlurEstimator (which uses the robust MAD) or
AdaptiveLocalEstimator is more appropriate.

### 4.2 No background model

SimpleMedian applies a single threshold T to the raw pixel values.  It makes no
attempt to model or remove the background before thresholding.  Any image with a
background gradient will therefore have one side of the frame heavily
over-thresholded and the other side under-thresholded.

**Example:** an image with background ramping from 800 ADU (left) to 1200 ADU
(right), stddev = 50 ADU, sigma_mult = 1.2:

```
T = 1000 + 1.2 × 50 = 1060 ADU
```

Left side (bg = 800): nearly all pixels are below T → almost nothing flagged.
Right side (bg = 1200): nearly all pixels are above T → nearly everything flagged.

### 4.3 High false-positive rate at the default sigma_mult

The default `sigma_mult = 1.2` was chosen in the original codebase to ensure
that even faint signal gets captured before Hough.  This is intentional — the
filter chain is the gating mechanism, not the background estimator.  However, it
means the binary image fed to Hough is quite noisy, which increases Hough
processing time and can produce spurious short line detections.

---

## 5. Configuration

| Parameter | Default | Description |
|---|---|---|
| `simple_median_sigma_mult` | 1.2 | Threshold = median + mult × stddev.  Increase to reduce false positives at the cost of missing faint streaks. |

The morphological close kernel is fixed at 5 × 5 (not currently exposed as a
config parameter).

---

## 6. When to Use

| Situation | Recommendation |
|---|---|
| Clean, flat background, bright satellite streak | ✅ Ideal — fast and reliable |
| Gradient background or vignetting | ❌ Use GaussianBlur or AdaptiveLocal |
| Crowded star field | ❌ Inflated stddev raises threshold; use GaussianBlur |
| Faint or low-SNR streak | ❌ High false-positive rate from sigma_mult=1.2 and non-robust σ |
| Real-time or batch processing (speed critical) | ✅ Fastest option |

---

## 7. Computational Cost

- `np.median` on an N-pixel image: O(N log N) — one sort
- `np.std`: O(N) — one pass
- Thresholding: O(N) — one pass
- Morphological close: O(N × k²) where k = 5

Total: dominated by the sort in `np.median` — O(N log N).

For a 4096 × 4096 image (16 M pixels) this runs in well under 1 second on modern
hardware.  It is roughly 10–20× faster than GaussianBlurEstimator and > 100×
faster than DoublePassEstimator.
