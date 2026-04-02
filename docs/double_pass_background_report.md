# Double-Pass Background Estimator
## Design Report — Streakiller

---

## 1. Purpose

The DoublePassEstimator is the most sophisticated background detection method in
the pipeline.  It uses image inpainting to construct a background model that
"fills in" the positions of bright sources as if they were not there, then
thresholds the residual between the original image and that reconstructed
background.

It is best suited for:
- Images with complex, spatially-varying backgrounds
- Crowded star fields where simple statistics are heavily biased
- Frames with large cosmetic defects (bleed trails, diffraction spikes) that
  must be excluded from background estimation

---

## 2. Algorithm

```
data (float32)
    │
    ┌──────── PASS 1 ──────────────────────────────────────┐
    │                                                       │
    ▼                                                       │
2a. global median + sigma_mult × stddev → threshold1       │
    │                                                       │
    ▼                                                       │
2b. Mask pixels > threshold1 as NaN → masked               │
    │                                                       │
    ▼                                                       │
2c. Telea inpainting of NaN regions → background           │
    └───────────────────────────────────────────────────────┘
    │
    ┌──────── PASS 2 ──────────────────────────────────────┐
    │                                                       │
    ▼                                                       │
3a. highpass = data − background                           │
    │                                                       │
    ▼                                                       │
3b. median2 + sigma_mult × stddev2 of highpass → threshold2│
    │                                                       │
    ▼                                                       │
3c. binary = (highpass ≥ threshold2)                       │
    └───────────────────────────────────────────────────────┘
    │
    ▼
4. morphological close (5 × 5)
    │
    ▼
binary (uint8, {0, 255})
```

---

## 3. Step-by-Step Detail

### Step 2a — First-pass threshold

```
μ₁    = median(data)
σ₁    = stddev(data)
T₁    = μ₁ + sigma_mult × σ₁
```

`sigma_mult = 2.0` by default.  This is a coarse cut — its job is to identify
pixels that are *probably* not pure background (stars, cosmic rays, potential
streak pixels) so they can be excluded from background reconstruction.  The
threshold does not need to be precise; false negatives (missing a faint star)
are acceptable since inpainting reconstructs locally.

### Step 2b — Masking

Pixels above T₁ are set to NaN.  This produces `masked`, an image where all
bright sources are holes.

```python
hot_mask        = data > T₁
masked[hot_mask] = NaN
```

### Step 2c — Telea inpainting

```python
background = cv2.inpaint(masked_float32, nan_mask, inpaint_radius, cv2.INPAINT_TELEA)
```

OpenCV's `INPAINT_TELEA` implements the algorithm of Telea (2004): *An Image
Inpainting Technique Based on the Fast Marching Method*.

The method propagates information from the boundary of each masked region inward,
filling each unknown pixel as a weighted average of known neighbours.  The weights
favour:
- Proximity (closer known pixels contribute more)
- Gradient continuity (neighbours whose gradient points toward the unknown pixel)

The `inpaint_radius` parameter (default 3 pixels) defines the neighbourhood from
which boundary information is drawn.  A larger radius smooths more aggressively
but risks smearing the background across large masked regions.

**Key property:** inpainting reconstructs the background as it would appear *if the
stars and streaks were not there*.  This is qualitatively superior to simply
replacing masked pixels with the global median (as SimpleMedian implicitly does),
because inpainting respects spatial continuity — a gradient that runs across a
masked star will be continued through the hole rather than abruptly replaced with
a flat level.

### Step 3a — Residual (high-pass)

```python
highpass = data − background
```

After subtracting the inpainted background, `highpass` contains only:
- True sources (stars, streak) that were above T₁ and thus inpainted over
- Noise fluctuations everywhere (since the background model is smooth, it
  cannot reproduce pixel-scale noise)
- Any sources that were below T₁ and therefore NOT masked (very faint stars
  and, crucially, faint streak sections that might be below T₁)

**Historical bug fixed here:** the original implementation computed
`highpass = background − hot_mask`.  This subtracted a boolean (0/1) mask from
the inpainted background, giving a near-constant image with tiny booleans
subtracted — which produced essentially no foreground pixels.  The correct
computation is `highpass = image − background`, which is the residual of the
original data relative to the estimated background.

### Step 3b — Second-pass threshold

```python
μ₂  = nanmedian(highpass)
σ₂  = nanstd(highpass)
T₂  = μ₂ + sigma_mult × σ₂
```

`nanmedian` and `nanstd` are used in case any NaN values propagated from the
inpainting step.  `sigma_mult` is the same value as Pass 1 (default 2.0), but
the distribution being thresholded is now the residual, not the raw image.

### Step 4 — Morphological close

5 × 5 rectangular kernel, same size as SimpleMedian.

---

## 4. Why Two Passes Are Needed

The single-pass approach (threshold on raw data, return binary) suffers from a
fundamental problem: the threshold T₁ is computed from a distribution that
includes stars, which inflates σ.

**Pass 1** solves this by building a clean background model that does *not* include
the star distribution.

**Pass 2** then thresholds on the *residual* image where the background has been
removed.  The noise in the residual reflects true photon and read noise rather
than the star distribution, giving a more accurate and scene-independent threshold.

---

## 5. The Role of Telea Inpainting

### 5.1 Fast Marching Method

Telea inpainting uses the Fast Marching Method (FMM) to propagate the inpainting
front from the boundary of the masked region inward.  At each step, the pixel
closest to the known boundary is filled using:

```
u(p) = Σ_q w(p, q) × [u(q) + ∇u(q) · (p − q)]
```

where:
- u(p) is the intensity at unknown pixel p
- u(q) is the intensity at known neighbour q
- ∇u(q) is the gradient at q (estimated from the known region)
- w(p, q) is a weight combining distance and gradient alignment

This first-order Taylor expansion means the inpainting smoothly extrapolates the
local gradient through the masked region.

### 5.2 Practical consequence

For a masked star sitting on a sloped sky background (e.g., part of a gradient
from Moon glow), the inpainted value at the star's position will follow the
gradient as if the star were transparent, rather than just inserting the global
median.  This gives a more accurate background estimate under the star, which in
turn gives a cleaner residual in Pass 2.

### 5.3 Limitation

Inpainting works best when masked regions are small relative to the inpaint
radius.  For very large masked areas (e.g., a bright star filling 10% of the
frame, or a long streak that crosses the entire image and is above T₁), the
reconstructed background inside the masked region becomes increasingly inaccurate.
Large satellite streaks that are bright enough to be above T₁ end up being
inpainted over, which is actually desirable — the inpainted background under the
streak gives Pass 2 a clean baseline to difference against.

---

## 6. Comparison with GaussianBlurEstimator

| Property | GaussianBlur | DoublePass |
|---|---|---|
| Background model | Gaussian-blurred image | Inpainted (source-free) image |
| Handles point sources | Good (MAD clips outliers) | Excellent (inpainting removes stars) |
| Handles gradients | Good (large kernel removes slow variation) | Good (inpainting respects gradients) |
| Speed | Fast | Slow (inpainting is O(N × radius²)) |
| Large masked regions | N/A | Degrades gracefully |
| Bug history | Clean | Bug fixed: `highpass = image − background` |

DoublePass is most useful when GaussianBlur gives poor results — typically on
images with very dense star fields where the global MAD is still contaminated, or
where there are large structured artefacts (bleed columns, diffraction spikes)
that are not well-handled by the Gaussian blur.

---

## 7. Configuration

| Parameter | Default | Description |
|---|---|---|
| `double_pass_sigma_mult` | 2.0 | Threshold multiplier for both passes.  Applied to raw stddev in Pass 1 and residual stddev in Pass 2. |
| `double_pass_inpaint_radius` | 3 | Telea inpainting neighbourhood radius in pixels.  Increase for larger masked regions; decrease for speed. |

The morphological close kernel is fixed at 5 × 5.

---

## 8. When to Use

| Situation | Recommendation |
|---|---|
| Dense star field with bright sources | ✅ Inpainting isolates background well |
| Complex spatially-varying background | ✅ Inpainting respects local gradients |
| Large bleed trails or diffraction spikes | ✅ Inpainted over cleanly |
| Clean frame, speed matters | ❌ GaussianBlur is 10–50× faster |
| Very faint streaks on quiet background | ❌ AdaptiveLocal has lower effective threshold |

---

## 9. Computational Cost

- Pass 1 statistics: O(N log N) — median sort
- Inpainting: O(N × inpaint_radius²) — FMM propagation
- Pass 2 statistics: O(N log N)
- Thresholding + morphology: O(N)

For a 4096 × 4096 image with many masked pixels, the inpainting step dominates
and can take 10–60 seconds depending on the fraction of masked area and hardware.
This is typically 10–100× slower than GaussianBlurEstimator.

---

## 10. Bug History

The original `streakprocessing.py` implementation computed:

```python
highpass = background - hot_mask  # WRONG
```

`hot_mask` is a boolean array with values in {0, 1}.  Subtracting it from the
inpainted background produces values within 1 ADU of the background — essentially
a flat image.  Pass 2 would then find very few pixels above its threshold,
making the method completely ineffective.

The corrected implementation:

```python
highpass = data - background  # CORRECT residual
```

This is tested by `TestDoublePassEstimator::test_bug_fix_highpass_is_residual` in
the unit test suite, which verifies that foreground pixels are actually produced.

---

## 11. Reference

A. Telea (2004), *An Image Inpainting Technique Based on the Fast Marching Method*,
Journal of Graphics Tools, 9(1), 23–34.
