# Gaussian Blur Background Estimator
## Design Report — Streakiller

---

## 1. Purpose

The GaussianBlurEstimator is the default background detection method and provides
the best general-purpose balance between sensitivity, robustness, and speed.

It separates the image into a low-frequency *background* component (removed by
Gaussian blur) and a high-frequency *signal* component (retained), then applies
a robust adaptive threshold to the signal component.  This removes the primary
limitation of SimpleMedian — sensitivity to background gradients — while avoiding
the computational cost of inpainting (DoublePass) or tile-mesh construction
(AdaptiveLocal).

---

## 2. Algorithm

```
data (float32)
    │
    ▼
1. Percentile clip → clipped  (suppress hot pixels / bright stars)
    │
    ▼
2. Gaussian blur of clipped → background model
   highpass = clipped − background
    │
    ▼
3. Slight blur of highpass (5×5) → hp_blur  (improve SNR)
    │
    ▼
4. Global MAD-based σ on hp_blur
    │
    ▼
5. K-ladder: try k = 3.0, 2.5, 2.0, 1.5, 1.2 in order
   accept first k producing ≥ 50 foreground pixels
    │
    ▼
6. morphological close (3 × 3)
    │
    ▼
binary (uint8, {0, 255})
```

---

## 3. Step-by-Step Detail

### Step 1 — Percentile clipping

```python
p1,  p99 = np.percentile(data, (2, 99.8))
clipped   = np.clip(data, p1, p99)
```

Hot pixels (sensor defects producing extreme ADU values) and saturated stars are
clipped to the 99.8th percentile.  This prevents them from smearing across the
image during the subsequent Gaussian blur, which would create false halos in the
background model.

The lower clip at the 2nd percentile removes bias from cold pixels and
dark-current subtraction artefacts.

### Step 2 — Gaussian high-pass filter

```python
background = cv2.GaussianBlur(clipped, (ksize, ksize), 0)
highpass   = clipped − background
```

The Gaussian blur with `ksize = 51` (σ ≈ 8.3 pixels by OpenCV's formula
`σ = 0.3 × ((ksize − 1)/2 − 1) + 0.8`) acts as a low-pass filter that captures
all spatial structure at scales larger than roughly 2σ ≈ 17 pixels:

- Sky gradient from Moon glow → removed
- Sensor vignetting (slow fall-off to edges) → removed
- Extended nebulosity at large scales → mostly removed
- Stars (PSF FWHM typically 2–8 pixels) → mostly *retained* in highpass
- Satellite streaks (1–3 pixels wide) → *fully retained* in highpass

The kernel size `ksize` must be odd (OpenCV requirement).  If an even value is
configured, the estimator auto-corrects to `ksize + 1`.

**Why Gaussian rather than median filter?**  A Gaussian blur is separable
(O(N × ksize) vs O(N × ksize²) for a 2D box filter) and linear, so stars that
are smaller than the kernel bleed smoothly into the background estimate rather
than creating sharp discontinuities.  A median filter would be more outlier-robust
but significantly slower for large kernels.

### Step 3 — Slight blur of highpass

```python
hp_blur = cv2.GaussianBlur(highpass, (5, 5), 0)
```

A small 5 × 5 Gaussian blur smooths noise in the highpass image before
thresholding.  This has two effects:

1. Isolated hot pixels that survived the percentile clip no longer appear as
   single-pixel foreground spikes after thresholding.
2. Streak pixels that are slightly below the threshold due to noise fluctuation
   get a contribution from their bright neighbours, making the streak more
   continuously foreground.

### Step 4 — Robust σ via MAD

```python
med   = median(hp_blur)
mad   = median(|hp_blur − med|)  +  ε
sigma = 1.4826 × mad
```

The MAD (Median Absolute Deviation) is used instead of the standard deviation
because it is resistant to the bright tails introduced by stars and cosmic rays
in the high-pass residual.

For a pure Gaussian distribution N(0, σ):
```
E[MAD] = σ × Φ⁻¹(3/4) ≈ σ × 0.6745
```

Multiplying by 1.4826 = 1 / 0.6745 recovers an estimate that is numerically
consistent with the standard deviation of the background noise component.  This
normalization constant is shared across all estimators in the codebase.

The ε = 1 × 10⁻⁶ guard prevents σ = 0 on pathologically uniform images.

**Key difference from SimpleMedian:** SimpleMedian uses `np.std(data)`, which is
computed on the raw image and is dominated by stars.  GaussianBlurEstimator uses
`1.4826 × MAD(hp_blur)`, where `hp_blur` has had the large-scale background
removed and stars partially suppressed by the small blur — so σ reflects the true
noise floor rather than the star distribution.

### Step 5 — Adaptive k-ladder threshold

Rather than using a fixed multiple of σ, the estimator tries a descending sequence
of k values and accepts the first one that produces at least 50 foreground pixels:

```
for k in (3.0, 2.5, 2.0, 1.5, 1.2):
    T = med + k × sigma
    binary = (hp_blur >= T)
    if count_nonzero(binary) >= 50:
        return morphological_close(binary)
return best_effort_binary  # if none produced ≥ 50 pixels
```

**Why this matters:** On a pure background frame with no streak (e.g. between
exposures), even the most aggressive k = 1.2 might only flag a handful of noise
pixels.  On a crowded stellar field, k = 3.0 may already produce thousands.  The
ladder finds the most conservative threshold that still gives the Hough transform
enough foreground material to vote on.

The minimum of 50 pixels was chosen empirically as the approximate lower bound of
foreground coverage needed for `HoughLinesP` to successfully accumulate votes on a
real streak at the default `threshold = 60`.

### Step 6 — Morphological close

A 3 × 3 kernel (smaller than the 5 × 5 used by SimpleMedian and DoublePass)
closes small gaps.  The smaller kernel is appropriate because the highpass image
already has much less background noise contamination than the raw pixel values.

---

## 4. The High-Pass Filter as a Shape Pre-Filter

An important secondary effect of the Gaussian high-pass: it suppresses
slowly-varying extended structure relative to compact, high-contrast objects.

For a star with PSF FWHM = 3 pixels and a Gaussian blur kernel with σ = 8.3 px:
- The star's peak survives the high-pass with most of its amplitude intact
  (the blur kernel integrates far less of the star's flux than the star's peak)
- The star's wings are partially removed (they are at scales comparable to the blur)

For a satellite streak 1–2 pixels wide and hundreds of pixels long:
- In the cross-streak direction (1–2 px): streak survives nearly fully
- In the along-streak direction: no attenuation (streak extends past the kernel)

This means GaussianBlurEstimator is inherently more sensitive to *elongated*
high-spatial-frequency structures (streaks) relative to compact round ones (stars
and noise spikes) — a beneficial property for the downstream Hough detector.

---

## 5. Limitations

### 5.1 Global noise estimate

The MAD is still computed globally over the entire image.  If one region of the
frame has genuinely higher noise than another (e.g., near a bright galaxy that
contributes significant Poisson noise), the global σ will be raised and the
threshold in quieter regions will be set too conservatively — faint streaks in
quiet regions may be missed.

AdaptiveLocalEstimator addresses this by estimating noise per tile.

### 5.2 Kernel size must be tuned to background scale

If the background varies at scales smaller than `ksize / 2 ≈ 25 pixels` (rare but
possible near bright extended sources), the Gaussian blur will not fully remove it.
In practice, `ksize = 51` handles the overwhelming majority of sky gradient and
vignetting patterns seen in amateur astronomy cameras.

### 5.3 Kernel attenuation of wide streaks

A satellite streak that is many pixels wide (e.g., out-of-focus, or from a
tumbling object with a diffuse halo) will be partially attenuated by the Gaussian
blur, reducing its SNR in the highpass.  This is generally not a problem since
wide streaks are by definition high total flux, but it is worth noting for very
diffuse low-surface-brightness objects.

---

## 6. Configuration

| Parameter | Default | Description |
|---|---|---|
| `gaussian_kernel_size` | 51 | Gaussian blur kernel side length (must be odd; auto-corrected if even). Larger values remove more large-scale structure but attenuate broader streaks. |
| `gaussian_sigma_ladder` | (3.0, 2.5, 2.0, 1.5, 1.2) | Sequence of k-values tried in order. Add smaller values to detect fainter signals; remove them to run faster. |

`GAUSSIAN_MIN_BINARY_PIXELS = 50` and `MAD_NORMALIZATION_FACTOR = 1.4826` are
compile-time constants in `defaults.py`.

---

## 7. When to Use

| Situation | Recommendation |
|---|---|
| Typical sky background with moderate gradient | ✅ Ideal — default choice |
| Crowded star field | ✅ MAD is robust to star contamination |
| Uniform flat background, speed critical | SimpleMedian may be faster |
| Strongly non-uniform noise (detector edge effects) | AdaptiveLocal better |
| Very faint streak on complex background | AdaptiveLocal or DoublePass |

---

## 8. Computational Cost

- Percentile clip: O(N log N) — sort
- GaussianBlur (ksize=51): O(N × ksize) ≈ O(51N) — separable implementation
- MAD: O(N log N) — median sort
- K-ladder: up to 5 × O(N) threshold passes

For a 4096 × 4096 image: typically 0.5–2 seconds depending on hardware.
Approximately 10–20× slower than SimpleMedian but 10–50× faster than DoublePass.
