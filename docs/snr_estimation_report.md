# Streak SNR Estimation
## Design Report — Streakiller

---

## 1. Purpose

After the Hough detector and filter chain identify a set of streak candidates, the
pipeline knows *where* each streak is but nothing about *how confidently* it was
detected.  Two detections at the same pixel coordinates can have very different
reliability depending on the local sky brightness and noise floor.

The SNR estimator measures each streak against its *own* local background, producing
two complementary metrics:

| Metric | Answers |
|---|---|
| **Peak SNR** | "Is the brightest pixel on this streak real?" |
| **Integrated SNR** | "How significant is the streak as a whole?" |

These feed into the CSV output, optional downstream filtering (`MinSNRFilter`), and
human review.

---

## 2. Aperture Model

The geometry is built entirely from the Hough line endpoints `(x1, y1, x2, y2)`.

### 2.1 Coordinate frame

Two orthogonal unit vectors are derived from the line direction:

```
dx = x2 - x1,   dy = y2 - y1
L  = sqrt(dx² + dy²)           # streak length in pixels

û = (dx/L,  dy/L)              # along-streak unit vector
v̂ = (-dy/L, dx/L)             # perpendicular unit vector (rotated 90° CCW)
```

Any image pixel `p = (x, y)` can be projected onto this frame:

```
s = (p - p1) · û              # along-streak coordinate  [0 … L]
t = (p - p1) · v̂              # perpendicular coordinate [−∞ … +∞]
```

### 2.2 Signal aperture

A pixel is inside the signal aperture when:

```
0 ≤ s ≤ L           (between the endpoints)
|t| ≤ half_width     (within the streak's cross-section)
```

`half_width = 3` pixels by default.  The aperture is walked as a discrete pixel
grid: for each integer step along the streak axis, sample all pixels within
`±half_width` in the perpendicular direction (nearest-integer rounding).  This
gives approximately `L × (2 × half_width + 1)` signal pixels.

### 2.3 Background aperture

Two parallel strips flank the signal aperture at a standoff distance:

```
Strip A:   bg_offset ≤  t ≤  bg_offset + bg_half_width
Strip B:  -bg_offset ≥  t ≥ -(bg_offset + bg_half_width)
```

`bg_offset = 10` pixels, `bg_half_width = 5` pixels by default.  The standoff
prevents signal-aperture flux (including PSF wings) from contaminating the
background estimate.

```
          ┌──────────────────────────────────┐
          │  bg strip B  (t ≈ -10 … -15)    │
          └──────────────────────────────────┘
          ══════════════════════════════════════  ← streak (t = 0 ± 3)
          ┌──────────────────────────────────┐
          │  bg strip A  (t ≈ +10 … +15)    │
          └──────────────────────────────────┘
```

All pixels whose projected coordinates fall outside `[0, W)` × `[0, H)` are
discarded.  If fewer than `min_bg_pixels = 20` background pixels survive after
clipping, the estimator falls back to a global background computed from the
full-image median and MAD.

---

## 3. Background Estimation

### 3.1 Robust noise floor

From the background aperture pixel set **B**:

```
bg_median = median(B)
mad       = median(|B − bg_median|)
bg_noise  = 1.4826 × mad  +  ε          (ε = 1 × 10⁻⁶)
```

The `1.4826` factor is the same constant used throughout the codebase
(`_MAD_FACTOR` in `adaptive_local.py`, `MAD_NORMALIZATION_FACTOR` in
`defaults.py`).  It converts the MAD to a Gaussian-equivalent standard deviation:

```
For X ~ N(μ, σ):    E[MAD] = σ / Φ⁻¹(¾) ≈ 0.6745 σ
                    ⟹  σ_equiv = MAD / 0.6745 = 1.4826 × MAD
```

The MAD is used — rather than the standard deviation — because background strips
near crowded star fields may contain star PSF wings.  The MAD ignores outliers;
the standard deviation is inflated by them.

### 3.2 Background-subtracted signal pixels

For each pixel `p_i` in the signal aperture with raw value `I(p_i)`:

```
s_i = I(p_i) − bg_median
```

`s_i` is the background-subtracted signal at that pixel.

---

## 4. SNR Metrics

### 4.1 Peak SNR

```
peak_signal = max(s_i)

peak_snr = peak_signal / bg_noise
```

This is the per-pixel SNR at the brightest point on the streak.  It directly
answers the question "is the strongest pixel on this streak above the noise floor?"
and matches the intuition of the per-pixel local SNR computed by
`AdaptiveLocalEstimator` during detection (where `local_snr = highpass / noise_model`).

**Limitation:** A single hot pixel or cosmic ray can inflate `peak_snr` without
the rest of the streak being real.  Use `integrated_snr` as a cross-check.

### 4.2 Integrated SNR

Assumes uncorrelated Gaussian noise with standard deviation `bg_noise` per pixel.
Summing N independent noisy measurements reduces noise by √N:

```
S_sum = Σ s_i          (sum over all N signal-aperture pixels)
N     = |signal aperture|

integrated_snr = S_sum / (bg_noise × √N)
```

**Derivation:**

Let each signal pixel be `s_i = μ_i + ε_i` where `ε_i ~ N(0, bg_noise²)`.
The sum is:

```
S_sum = Σ μ_i  +  Σ ε_i
```

The noise on the sum is:

```
Var(Σ ε_i) = N × bg_noise²
Std(S_sum)  = bg_noise × √N
```

Therefore:

```
SNR(S_sum) = E[S_sum] / Std(S_sum) = Σ μ_i / (bg_noise × √N)
```

**Scaling with streak length:**  For a uniform streak with mean surface
brightness `μ` above background on every pixel:

```
S_sum = N × μ
integrated_snr = N × μ / (bg_noise × √N) = μ × √N / bg_noise
```

Integrated SNR grows as `√N`, i.e. `√L`.  A streak twice as long is `√2 ≈ 1.41×`
more significant at fixed surface brightness.

### 4.3 When peak_snr > integrated_snr

This happens when the streak is very short (N is small) and the peak pixel is
well above average.  For a point source masquerading as a very short streak,
`peak_snr ≫ integrated_snr` is expected.

For a genuine satellite trail at uniform brightness:

```
peak_snr ≈ μ / bg_noise
integrated_snr = μ × √N / bg_noise = peak_snr × √N
```

So `integrated_snr > peak_snr` always holds when `N > 1`, which is almost always
the case.  A detection where `peak_snr > integrated_snr` is a red flag for a
non-streak artifact.

---

## 5. Full Per-Streak Computation Summary

```
Input: calibrated image I (float32, H×W),  line endpoints (x1,y1,x2,y2)

1.  Build û, v̂, L from endpoints

2.  Collect signal aperture pixels:
      S = {I(p) : 0 ≤ s(p) ≤ L,  |t(p)| ≤ half_width}

3.  Collect background aperture pixels:
      B = {I(p) : 0 ≤ s(p) ≤ L,
                  bg_offset ≤ |t(p)| ≤ bg_offset + bg_half_width}

4.  Compute background statistics:
      bg_median = median(B)
      bg_noise  = 1.4826 × median(|B − bg_median|)  +  ε
        (fall back to global MAD if |B| < min_bg_pixels)

5.  Background-subtract signal:
      s_i = S_i − bg_median  for each pixel i

6.  Compute metrics:
      peak_signal    = max(s_i)
      peak_snr       = peak_signal / bg_noise

      S_sum          = Σ s_i
      N              = |S|
      integrated_snr = S_sum / (bg_noise × √N)

Output: StreakMetrics(peak_snr, integrated_snr, peak_signal,
                      S_sum, bg_median, bg_noise, N)
```

---

## 6. Edge Cases

| Situation | Handling |
|---|---|
| Streak near image boundary | Aperture pixels outside `[0,W)×[0,H)` are silently dropped; SNR computed on surviving pixels. |
| Background aperture clipped below `min_bg_pixels` | Fall back to global `bg_median = median(I)`, `bg_noise = 1.4826 × MAD(I)`. |
| Perfectly uniform image (`mad = 0`) | `ε = 1 × 10⁻⁶` prevents division by zero; SNR will be unrealistically large — treat as unreliable. |
| Negative `integrated_snr` | Streak pixels darker than sky (mis-detection or reflection ghost). Reported as-is — useful diagnostic signal. |
| Very short streak (`N < 5`) | SNR computed but flagged; `bg_noise × √N` is very small, so `integrated_snr` will appear inflated. |

---

## 7. Relationship to Existing Noise Estimates

The background estimators already compute noise-related quantities during
detection:

| Estimator | Noise quantity computed |
|---|---|
| `SimpleMedianEstimator` | Global `stddev(data)` |
| `GaussianBlurEstimator` | Global `1.4826 × MAD(highpass)` |
| `AdaptiveLocalEstimator` | Per-pixel `noise_model` (tile MAD mesh) |

The SNR estimator computes its **own** local background independently per streak,
rather than reusing these estimates.  This is intentional:

1. **Locality** — the noise floor measured in the streak's own background strip
   reflects the actual noise at that sky position, not a global or tile-averaged
   value.
2. **Decoupling** — no changes to the `BackgroundEstimator` protocol are needed.
3. **Consistency** — the same formula works regardless of which background
   estimator was used upstream.

The `AdaptiveLocalEstimator` noise mesh would be more accurate for the cost of
tighter coupling.  This is noted as a future optimisation (see open questions in
the SNR plan).

---

## 8. Configuration

| Parameter | Default | Description |
|---|---|---|
| `half_width` | 3 | Signal aperture half-width in pixels (perpendicular to streak). Increase for wide/defocused trails. |
| `bg_offset` | 10 | Distance from streak axis to the near edge of the background strip. Should exceed the PSF radius. |
| `bg_half_width` | 5 | Half-width of each background strip. Larger → more pixels, smaller variance in `bg_noise`. |
| `min_bg_pixels` | 20 | Minimum background pixels before falling back to global estimate. |
