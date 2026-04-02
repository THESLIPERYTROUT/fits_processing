# Hough Line Detection
## Design Report — Streakiller

---

## 1. Purpose

The `StreakDetector` converts the binary foreground mask produced by a background
estimator into a set of line segment detections using the **Probabilistic Hough
Line Transform** (`HoughLinesP`).  It is responsible only for detection — all
filtering and deduplication happens downstream in the `FilterChain`.

---

## 2. Background: The Hough Transform

### 2.1 Classic Hough Transform (for reference)

The original Hough Transform (Hough, 1962) operates in *parameter space*.  A
straight line in image space can be parameterised as:

```
ρ = x cos θ + y sin θ
```

where:
- ρ is the perpendicular distance from the origin to the line (pixels)
- θ is the angle of the perpendicular (radians)

For each foreground pixel (x, y) in the binary image, every possible line through
that pixel traces a sinusoid in (ρ, θ) space.  Collinear pixels produce sinusoids
that intersect at a common (ρ*, θ*), creating an *accumulator* peak there.  The
height of the peak is the number of foreground pixels that voted for that line
(the *vote count* or *Hough score*).

Lines with vote count ≥ `threshold` are accepted as detections.

### 2.2 Probabilistic Hough Transform (`HoughLinesP`)

The standard transform processes every foreground pixel, which is O(N × A) where
A is the accumulator size.  For large images with many foreground pixels this is
slow.

The Probabilistic Hough Transform (Matas et al., 2000) samples a random subset of
foreground pixels and processes them one at a time, stopping once a sufficient
accumulator peak is found.  It additionally extracts **line segments** (with
specific start and end pixel coordinates) rather than infinite lines.  This is the
version used by OpenCV and by Streakiller.

For satellite streak detection, the segment endpoint coordinates are essential
because:
- They directly give the streak length (used by the length filter)
- They allow angle and collinearity computation (angle filter, colinear filter)
- They give the streak's on-sky position (for TLE correlation, if enabled)

---

## 3. OpenCV `HoughLinesP` Parameters

The detector calls:

```python
cv2.HoughLinesP(
    binary,
    rho        = p.rho,           # accumulator ρ resolution (pixels)
    theta      = p.theta_deg × π/180,  # accumulator θ resolution (radians)
    threshold  = p.threshold,     # minimum vote count to accept a line
    minLineLength = min_line_length,   # minimum segment length (pixels)
    maxLineGap    = p.max_line_gap,    # max collinear gap to bridge (pixels)
)
```

### 3.1 `rho` — distance resolution (default 1.0 pixel)

The accumulator is a 2D histogram in (ρ, θ).  `rho` sets the bin width in the ρ
dimension.  A value of 1.0 pixel is standard and matches the spatial resolution
of the binary image.

Increasing `rho` to 2.0 or 3.0:
- Reduces accumulator memory
- Makes the transform more tolerant of straight-but-wobbly streaks (pointing
  errors, atmospheric dispersion)
- Risks merging two nearby parallel streaks into one detection

Decreasing `rho` below 1.0 provides sub-pixel resolution but makes the accumulator
much larger and voting sparser.

### 3.2 `theta` — angle resolution (default 1.0°)

Sets the bin width in the θ dimension.  At 1.0°, the accumulator has 180 bins
covering 0–179°.

Fine θ (0.5°) distinguishes streaks at very similar angles but doubles accumulator
size and computation.  Coarse θ (2°) merges nearby angles, potentially causing
two nearly-parallel streaks to collide in the accumulator.

For satellite streaks, which typically subtend a well-defined angle (determined
by the satellite's pass geometry), 1.0° is a sensible default.

### 3.3 `threshold` — minimum vote count (default 60)

A candidate line is accepted only if ≥ `threshold` foreground pixels voted for it.
This is the primary gate against short noise artefacts: a random cluster of noise
pixels will rarely produce a straight run of 60+ collinear points.

**Relationship to `minLineLength`:** The two parameters interact.  A line segment
must be ≥ `minLineLength` pixels long AND accumulate ≥ `threshold` votes.  If the
binary image is sparse (low foreground density), a 200-pixel streak might only
have 60 foreground pixels along it (because the estimator did not flag every
pixel), meaning `threshold = 60` is the binding constraint.

Increasing `threshold`:
- Fewer false detections
- Longer minimum effective streak length

Decreasing `threshold`:
- More sensitivity to short faint streaks
- More false detections from noise

### 3.4 `minLineLength` — minimum segment length (default 30 px, TLE-adaptive)

The minimum length (in pixels) that a detected line segment must have.  This is
the most important parameter for rejecting false positives: a random noise cluster
rarely forms a line longer than 20–30 pixels.

**TLE-adaptive minimum line length:** If `estimated_streak_length_enabled = true`
and a valid NORAD ID is configured, the pipeline estimates the expected on-sky
angular velocity of the satellite and converts it to a pixel displacement for the
given exposure time and pixel scale.  This sets `minLineLength` dynamically,
making it longer for fast-moving targets (which would produce long streaks) and
shorter for slow or geostationary satellites.

When TLE estimation is disabled or unavailable, `default_minlinelength` from
config.json is used (default 30 px).

### 3.5 `maxLineGap` — maximum bridging gap (default 5 px)

Foreground pixels along a streak may not be continuous — there can be gaps due to:
- Background estimator thresholding leaving some streak pixels as background
- Diffraction rings or diffuse halo alternating above/below threshold
- Sensor dead columns

`maxLineGap` allows `HoughLinesP` to bridge gaps of up to this many pixels when
assembling a segment.  A value of 5 pixels bridges typical gaps without merging
distinct streaks that happen to be aligned.

---

## 4. Input: Binary Image

`HoughLinesP` takes a uint8 binary image (values in {0, 255}).  Only pixels = 255
contribute votes to the accumulator.

The binary image comes directly from the background estimator.  Its quality
strongly determines Hough detection performance:
- **Too many foreground pixels** (low threshold): accumulator becomes noisy;
  random lines accumulate high vote counts; many false detections.
- **Too few foreground pixels** (high threshold): legitimate streak pixels are
  absent; even long streaks fail to accumulate `threshold` votes.

This tension is why GaussianBlurEstimator uses the k-ladder: it finds the lowest
threshold that still gives enough foreground pixels for Hough to work (≥ 50
foreground pixels as a proxy for "enough material for at least one line").

---

## 5. Output Format

`HoughLinesP` returns a shape `(N, 1, 4)` int32 array, or `None` if no lines
were found.  The inner `(1, 4)` comes from OpenCV's representation of a single
line as a 1×4 matrix:

```
lines[i][0] = [x1, y1, x2, y2]
```

where (x1, y1) and (x2, y2) are the pixel coordinates of the two endpoints of the
detected segment.

The `StreakDetector.detect()` method normalises this: if `HoughLinesP` returns
`None`, a `(0, 1, 4)` empty array is returned instead.  This contract ensures
downstream code never has to check for `None` — an important invariant for
safe iteration.

---

## 6. Coordinate Convention

OpenCV images use the convention:
- x increases left → right (column index)
- y increases top → bottom (row index)
- Origin (0, 0) at top-left corner

So `(x1, y1) = (50, 30)` means column 50, row 30.

All filters operate in this convention.  When computing angles, the apparent
vertical-flip relative to a mathematical x-y plane must be accounted for:
a streak from top-left to bottom-right has a *positive* slope in image coordinates
(y increases as x increases) but a *negative* slope in standard math coordinates.
The filter implementations use `atan2(dy, dx)` in image coordinates and are
internally consistent.

---

## 7. Vote Count and Streak SNR

The Hough vote count for a streak is approximately proportional to the number of
foreground streak pixels that land in the same (ρ, θ) bin.  For a streak of
length L pixels where a fraction f are flagged as foreground:

```
votes ≈ f × L
```

For detection, we need `votes ≥ threshold`:

```
f × L ≥ threshold
L ≥ threshold / f
```

At `threshold = 60`:
- If f = 1.0 (every streak pixel is foreground): minimum detectable length = 60 px
- If f = 0.5 (half flagged): minimum detectable length = 120 px
- If f = 0.3 (faint streak, one-third flagged): minimum detectable length = 200 px

This shows the relationship between background estimator sensitivity (f) and the
effective minimum detectable streak length.  Lowering the SNR threshold in
AdaptiveLocalEstimator raises f for faint streaks, directly enabling detection
of shorter faint streaks at fixed Hough `threshold`.

---

## 8. Configuration

| Parameter | Default | Description |
|---|---|---|
| `hough_params.threshold` | 60 | Minimum votes for line acceptance. Increase to reduce false positives; decrease for shorter faint streaks. |
| `hough_params.rho` | 1.0 | Accumulator ρ bin width (pixels). Rarely needs changing. |
| `hough_params.theta_deg` | 1.0 | Accumulator θ bin width (degrees). |
| `hough_params.max_line_gap` | 5 | Maximum pixel gap bridged within one segment. |
| `default_minlinelength` | 30 | Minimum segment length when TLE estimation is off. |
| `estimated_streak_length_enabled` | false | If true, derives minLineLength from TLE + exposure time + pixel scale. |

---

## 9. Post-Detection: Filter Chain

`HoughLinesP` produces *raw* detections that include:
- Duplicates (the same streak detected multiple times at slightly different angles)
- Star trail segments (slow-moving stars produce short arcs over long exposures)
- Edge artefacts (sensor borders may appear as lines)
- Noise-sourced lines (rare, but possible with low `threshold`)

The `FilterChain` applies five sequential filters to remove these:

1. **Length filter** — keeps only lines within 10% of the mode length of all
   detected lines (eliminates very short artefacts when a long streak dominates)
2. **Midpoint filter** — removes duplicates by dropping lines whose midpoints are
   within 10 px of an already-accepted line
3. **Angle filter** — removes lines whose angle differs from the majority by > 10°
   (eliminates star trails and noise lines at random angles)
4. **Colinear filter** (optional) — merges end-to-end collinear segments into a
   single longer segment
5. **Endpoint filter** — removes lines whose endpoints are within 10 px of another
   line's endpoints (deduplication pass)

The filter chain is detailed in `ARCHITECTURE.md`.

---

## 10. Computational Cost

`HoughLinesP` is O(F × A) where F is the number of foreground pixels and A is the
number of accumulator cells (≈ `(2 × D_max / rho) × (180 / theta_deg)` where
D_max = √(H² + W²)).

For a 4096 × 4096 image (D_max ≈ 5793 px) with rho = 1.0, theta = 1°:
- Accumulator size: 11586 × 180 ≈ 2 M cells
- At f = 5% foreground: F ≈ 840,000 pixels

Typical runtime: 0.1–2 seconds, dominated by the accumulator voting loop.
Very high foreground density (e.g., from a coarse background threshold) is the
most common cause of slow Hough performance.

---

## 11. References

- P. Hough (1962), *Method and Means for Recognizing Complex Patterns*, US Patent
  3,069,654.

- J. Matas, C. Galambos & J. Kittler (2000), *Robust Detection of Lines Using the
  Progressive Probabilistic Hough Transform*, Computer Vision and Image
  Understanding, 78(1), 119–137.

- OpenCV documentation: `cv2.HoughLinesP`
  https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
