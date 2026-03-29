# Architecture & Developer Guide

> **Intended audience:** A new developer implementing a new streak detection algorithm,
> background estimator, or post-detection filter.
> This document covers: how data flows end-to-end, what every class owns, what
> contracts must be satisfied, and a worked example of adding a new detector.

---

## 1. The big picture — data flow

```
┌──────────────┐
│  config.json │
└──────┬───────┘
       │ PipelineConfig.from_json()
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                       StreakPipeline                             │
│                                                                  │
│  FitsLoader ──► FitsImage ──► CalibrationStep ──► FitsImage     │
│                                    │                             │
│                                    ▼                             │
│                          BackgroundEstimator ──► binary uint8    │
│                                    │                             │
│                                    ▼                             │
│                           StreakDetector ──► RawDetection        │
│                                    │         (lines, binary,     │
│                                    │          display)           │
│                                    ▼                             │
│                            FilterChain ──► (lines, snapshots)   │
│                                    │                             │
│                                    ▼                             │
│                           PipelineResult + Provenance            │
│                                    │                             │
│                                    ▼                             │
│                           OutputWriter ──► disk / cloud          │
└──────────────────────────────────────────────────────────────────┘
```

Every stage receives its inputs as plain Python objects or numpy arrays.
No stage reads from disk or talks to a database except `FitsLoader`,
`CalibrationStep.load_frames()`, `TleCache`, and `OutputWriter`.
Everything in between is pure computation.

---

## 2. Module dependency rules

```
config/          ← no internal imports. Pure data + validation.
    │
models/          ← imports config/ only. Pure data. No I/O, no OpenCV.
    │
io/              ← imports models/ and config/. All file I/O lives here.
    │
calibration/     ─┐
background/       ├─ import models/ and config/ only.
detection/        │  No I/O. No imports from each other.
filters/          │
satellite/       ─┘
    │
pipeline/        ← imports everything above. The only place stages are
    │              assembled into a complete run.
    │
cli/             ← imports pipeline/ and config/ only.
                   Entry point. Never imported by other modules.
```

**Why this matters for you:** When adding a new algorithm, you can import
`FitsImage`, `BackgroundParams`, `FilterParams` etc. freely. You cannot import
`OutputWriter` or `FitsLoader` inside a processing module — I/O belongs in
`io/` or `pipeline/` only.

---

## 3. The key data types

These are the objects you will receive and return. Understand them before
writing any algorithm code.

### `FitsImage` — `src/streakiller/models/fits_image.py`

The primary carrier. You will receive one of these in almost every stage.

```python
@dataclass
class FitsImage:
    source_path: Optional[Path]    # None for synthetic/derived images
    data: np.ndarray               # float32, shape (H, W) — the pixel data
    raw_header: dict               # all original FITS header key-value pairs
    metadata: ObservationMetadata  # structured, typed header fields
```

**Never mutate `data` in place.** Return a new image via `image.derive(new_data)`.

```python
# ✅ correct
def my_preprocessing(image: FitsImage) -> FitsImage:
    processed = do_something(image.data)
    return image.derive(processed)   # preserves metadata, source_path

# ❌ wrong
def my_preprocessing(image: FitsImage) -> FitsImage:
    image.data[...] = do_something(image.data)  # mutates in place
    return image
```

### `ObservationMetadata` — `src/streakiller/models/fits_image.py`

Everything you need about the observation. All fields are `Optional` — always
guard before using them.

```python
@dataclass(frozen=True)
class ObservationMetadata:
    exposure_time: Optional[float]       # seconds
    date_obs: Optional[str]              # ISO 8601 string
    telescope: Optional[str]
    camera: Optional[str]
    focal_length_mm: Optional[float]
    lat: Optional[float]                 # degrees
    lon: Optional[float]                 # degrees
    elevation_m: Optional[float]         # metres
    binning: int                         # default 1
    pixel_size_um: Optional[float]       # microns, after applying binning
    pixel_scale_arcsec: Optional[float]  # arcsec/pixel

    @property
    def has_location(self) -> bool: ...  # True if lat, lon, elevation all present
```

### Lines format — `np.ndarray` shape `(N, 1, 4)` `int32`

This is the universal line format throughout the pipeline, matching
`cv2.HoughLinesP` output exactly.

```python
lines[i]      # shape (1, 4)
lines[i][0]   # shape (4,)  — [x1, y1, x2, y2]

x1, y1, x2, y2 = lines[i][0]   # unpack one line
```

An empty result is always `np.empty((0, 1, 4), dtype=np.int32)` — never `None`.
If your detector finds nothing, return this empty array.

### `RawDetection` — `src/streakiller/detection/detector.py`

What `StreakDetector.detect()` returns. Bundles lines with the images used
to produce them (needed by `OutputWriter` for visualisation).

```python
@dataclass
class RawDetection:
    lines: np.ndarray            # (N, 1, 4) int32 — raw Hough output
    binary_image: np.ndarray     # uint8 (H, W) — the mask fed to Hough
    normalized_display: np.ndarray  # uint8 (H, W) — for visualisation
```

### `FilterStageSnapshot` — `src/streakiller/models/streak.py`

Automatically created by `FilterChain` for every filter stage. You do not
build these yourself unless writing a custom chain.

```python
@dataclass
class FilterStageSnapshot:
    stage_name: str
    lines_before: int
    lines_after: int
    lines: np.ndarray   # the lines after this stage
```

### `PipelineResult` — `src/streakiller/models/result.py`

The final output of one complete pipeline run.

```python
@dataclass
class PipelineResult:
    source_path: Optional[Path]
    detected_lines: np.ndarray              # (N, 1, 4) — final filtered lines
    filter_snapshots: list[FilterStageSnapshot]
    normalized_display: Optional[np.ndarray]
    binary_image: Optional[np.ndarray]
    provenance: Optional[Provenance]        # full audit record
    error: Optional[str]                    # set on non-fatal failure

    @property
    def streak_count(self) -> int: ...
    @property
    def succeeded(self) -> bool: ...
```

---

## 4. The three injectable protocols

These are the three places where you plug in a new algorithm. Each is a
`Protocol` (structural typing — no inheritance required, just match the
method signature).

### `BackgroundEstimator` — `src/streakiller/background/base.py`

Converts the raw float image into a binary foreground mask for the detector.

```python
class BackgroundEstimator(Protocol):
    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        """
        Input:  float32 (H, W) image
        Output: uint8   (H, W) binary mask — pixel is 0 or 255

        Rules:
        - Must NOT modify `data`
        - Must NOT write any files
        - Must return same spatial shape as input
        """
```

Existing implementations: `SimpleMedianEstimator`, `GaussianBlurEstimator`,
`DoublePassEstimator`.

### `StreakDetector` — `src/streakiller/detection/detector.py`

Takes the binary mask and finds line segments.

```python
class StreakDetector:
    def detect(
        self,
        binary: np.ndarray,        # uint8 (H, W) foreground mask
        source_data: np.ndarray,   # float32 (H, W) original image (for display only)
        min_line_length: float,    # minimum segment length in pixels
    ) -> RawDetection:
        """
        Output: RawDetection with lines (N, 1, 4) int32.
        If nothing found, lines must be np.empty((0,1,4), dtype=np.int32).
        Never return None. Never raise — return empty RawDetection on failure.
        """
```

Only one implementation exists today (`HoughLinesP`). **This is where you plug
in a new detection algorithm.**

### `OutputWriter` — `src/streakiller/io/output_writer.py`

Writes a completed `PipelineResult` somewhere.

```python
class OutputWriter(Protocol):
    def write(self, result: PipelineResult) -> None: ...
```

Existing implementation: `LocalOutputWriter`. A future `S3OutputWriter` would
satisfy this with no changes to the pipeline.

---

## 5. `PipelineConfig` — where parameters live

Every tunable number has a home in `src/streakiller/config/schema.py`.
The defaults mirror the original hard-coded values and are documented in
`src/streakiller/config/defaults.py`.

```
PipelineConfig
├── HoughParams          threshold, max_line_gap, rho, theta_deg
├── FilterParams         midpoint_min_distance, endpoint_min_distance,
│                        angle_min_diff_deg, length_fraction,
│                        colinear_orientation_tol
├── BackgroundParams     gaussian_kernel_size, gaussian_sigma_ladder,
│                        simple_median_sigma_mult, double_pass_sigma_mult,
│                        double_pass_inpaint_radius
├── EnabledFilters       midpoint_filter, line_angle, colinear_filter,
│                        endpoint_filter, length_filter
├── BackgroundMethod     simple_median, gaussian_blur, double_pass
└── OutputOptions        save_intermediate_images
```

When you add a new algorithm, add a new nested dataclass (e.g.
`MyDetectorParams`) to `PipelineConfig` and expose its fields in `config.json`.
Do not add loose constants to your module.

---

## 6. `FilterChain` — how filters compose

`FilterChain` runs a list of pure functions in order and records snapshots.

```python
# Each filter has this exact signature:
FilterFn = Callable[[np.ndarray, FilterParams], np.ndarray]
#                    (N,1,4) in               (M,1,4) out

# FilterChain assembles them:
chain = FilterChain.from_config(config.enabled_line_filters)
final_lines, snapshots = chain.run(raw_lines, config.filter_params)
```

`FilterChain.from_config()` wires them in fixed order:
```
midpoint → angle → colinear → endpoint → length
```

A filter receives all lines seen so far and returns a subset (or
merged set, in the case of `colinear_merge`). It must not modify
the input array.

---

## 7. `StreakPipeline` — how stages are assembled

```python
class StreakPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        calibration_step=None,        # optional CalibrationStep
        background_estimator=None,    # optional BackgroundEstimator
        streak_estimator=None,        # optional StreakLengthEstimator (TLE)
        output_writer=None,           # optional OutputWriter
    ): ...

    def process(self, image: FitsImage) -> PipelineResult: ...
```

`process()` is the only public method you interact with during normal use.
All dependencies are injected — pass `None` for `output_writer` to skip
writing files (useful in tests and notebooks).

The internal `_run()` calls stages in this order:

```
_prepare_image()          → calibration or hot-pixel removal
_resolve_min_line_length() → TLE estimate or config default
background.estimate()     → binary mask
detector.detect()         → RawDetection (raw lines + images)
filter_chain.run()        → filtered lines + snapshots
build Provenance
output_writer.write()     → disk / cloud
```

---

## 8. Worked example — implementing a new detector

Say you want to replace Hough with a **neural-network-based line detector**
or a **RANSAC-based approach**. Here is exactly what to create and where
to hook it in.

### Step 1 — Add parameters to config

In `src/streakiller/config/schema.py`, add a new nested dataclass:

```python
@dataclass
class MyDetectorParams:
    confidence_threshold: float = 0.7
    max_iterations: int = 1000
    min_inliers: int = 20
```

Add it to `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    ...
    my_detector: MyDetectorParams = field(default_factory=MyDetectorParams)
```

Add the corresponding defaults to `src/streakiller/config/defaults.py`:

```python
# --- MyDetector (src/streakiller/detection/my_detector.py) ---
MY_DETECTOR_CONFIDENCE = 0.7
MY_DETECTOR_MAX_ITER   = 1000
MY_DETECTOR_MIN_INLIERS = 20
```

Expose it in `config.json` (optional — defaults will apply if omitted):

```json
{
    "my_detector": {
        "confidence_threshold": 0.8,
        "max_iterations": 500
    }
}
```

---

### Step 2 — Create the detector class

Create `src/streakiller/detection/my_detector.py`:

```python
import logging
import numpy as np
from streakiller.config.schema import MyDetectorParams   # your new params class
from streakiller.detection.detector import RawDetection  # the return type
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


class MyDetector:
    """
    Example: RANSAC-based streak detector.

    Satisfies the same interface as StreakDetector so it is a drop-in
    replacement inside StreakPipeline.
    """

    def __init__(self, params: MyDetectorParams) -> None:
        self._params = params

    def detect(
        self,
        binary: np.ndarray,       # uint8 (H, W) foreground mask from BackgroundEstimator
        source_data: np.ndarray,  # float32 (H, W) original image — for display only
        min_line_length: float,   # minimum segment length in pixels
    ) -> RawDetection:
        """
        Run your algorithm on `binary` and return a RawDetection.

        The only contract:
        - lines must be shape (N, 1, 4) int32  — [x1, y1, x2, y2] per line
        - If nothing found, return np.empty((0, 1, 4), dtype=np.int32)
        - Never return None
        - Never raise — return empty RawDetection on failure
        """
        try:
            lines = self._run_ransac(binary, min_line_length)
        except Exception as exc:
            logger.error("MyDetector failed: %s", exc)
            lines = np.empty((0, 1, 4), dtype=np.int32)

        logger.info("MyDetector found %d lines", len(lines))

        return RawDetection(
            lines=lines,
            binary_image=binary,
            normalized_display=normalize_for_display(source_data),
        )

    def _run_ransac(self, binary: np.ndarray, min_length: float) -> np.ndarray:
        # your implementation here
        # must return shape (N, 1, 4) int32
        ...
```

**What you have access to inside `detect()`:**

| Variable | Type | What it is |
|----------|------|------------|
| `binary` | `uint8 (H, W)` | Foreground mask — white pixels are candidate streak locations |
| `source_data` | `float32 (H, W)` | Raw calibrated image — for display and any intensity-based logic |
| `min_line_length` | `float` | Minimum valid segment length in pixels |
| `self._params` | `MyDetectorParams` | Your config values |

**What you do NOT have access to (and should not need):**

- The `FitsImage` or its headers — those belong to the pipeline orchestrator
- `PipelineConfig` directly — your params were extracted into `MyDetectorParams`
- Any previous filter state — you operate on the binary mask only

---

### Step 3 — Wire it into the pipeline

In `src/streakiller/pipeline/streak_pipeline.py`, `StreakPipeline.__init__`
currently builds a `StreakDetector`. Add a branch for your new detector:

```python
# Before (existing):
self._detector = StreakDetector(config.hough_params)

# After:
if config.use_my_detector:          # add this bool to PipelineConfig
    from streakiller.detection.my_detector import MyDetector
    self._detector = MyDetector(config.my_detector)
else:
    self._detector = StreakDetector(config.hough_params)
```

Or, for testing only, inject it directly without touching config:

```python
from streakiller.detection.my_detector import MyDetector
from streakiller.config.schema import MyDetectorParams

my_det = MyDetector(MyDetectorParams(confidence_threshold=0.9))
pipeline = StreakPipeline(config=cfg, output_writer=None)
pipeline._detector = my_det   # swap in for a single test run
```

---

### Step 4 — Write tests

Create `tests/unit/test_my_detector.py`:

```python
import numpy as np
import pytest
from streakiller.detection.my_detector import MyDetector
from streakiller.config.schema import MyDetectorParams


@pytest.fixture
def empty_binary():
    return np.zeros((128, 128), dtype=np.uint8)


@pytest.fixture
def streak_binary():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[64, 10:118] = 255   # horizontal streak
    return img


class TestMyDetector:
    def test_empty_binary_returns_empty_lines(self, empty_binary):
        det = MyDetector(MyDetectorParams())
        result = det.detect(empty_binary, empty_binary.astype(np.float32), min_line_length=25)
        assert result.lines.shape == (0, 1, 4)
        assert result.lines.dtype == np.int32

    def test_detects_clear_streak(self, streak_binary):
        det = MyDetector(MyDetectorParams())
        result = det.detect(streak_binary, streak_binary.astype(np.float32), min_line_length=25)
        assert len(result.lines) >= 1

    def test_never_returns_none(self, empty_binary):
        det = MyDetector(MyDetectorParams())
        result = det.detect(empty_binary, empty_binary.astype(np.float32), min_line_length=999)
        assert result is not None
        assert result.lines is not None

    def test_output_shape_is_correct(self, streak_binary):
        det = MyDetector(MyDetectorParams())
        result = det.detect(streak_binary, streak_binary.astype(np.float32), min_line_length=10)
        if len(result.lines) > 0:
            assert result.lines.ndim == 3
            assert result.lines.shape[1] == 1
            assert result.lines.shape[2] == 4
            assert result.lines.dtype == np.int32
```

Run them:

```bash
pytest tests/unit/test_my_detector.py -v
```

---

## 9. Adding a new background estimator (summary)

If you want to change what happens *before* the detector (e.g. a
wavelet-based background model), follow the same pattern:

1. Create `src/streakiller/background/my_estimator.py` with a class that has:
   ```python
   def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
       # float32 (H,W) in → uint8 (H,W) binary out
       # no file I/O, do not modify data
   ```
2. Add a toggle to `BackgroundMethod` in `config/schema.py`
3. Wire it into `StreakPipeline.__init__()` alongside the three existing estimators
4. Add it to the `ESTIMATORS` list in `tests/unit/test_background_estimators.py` —
   all the shared contract tests run automatically

---

## 10. Adding a new post-detection filter (summary)

If you want to add a filter step after detection:

1. Create `src/streakiller/filters/my_filter.py`:
   ```python
   def my_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
       # (N,1,4) in → (M,1,4) out
       # pure function — no side effects, do not modify input
   ```
2. Add a bool to `EnabledFilters` in `config/schema.py`
3. Register it in `FilterChain.from_config()` in `filters/chain.py` at the
   correct position in the order
4. Add it to `FILTER_FNS` in `tests/unit/test_filters.py` — the edge-case
   tests run automatically

---

## 11. Quick reference — what lives where

| You want to... | Look here |
|----------------|-----------|
| Read the image from disk | `src/streakiller/io/fits_loader.py` |
| Access image pixels / headers | `src/streakiller/models/fits_image.py` |
| Access observation metadata | `FitsImage.metadata` (`ObservationMetadata`) |
| Tune a number without hardcoding | `src/streakiller/config/schema.py` + `defaults.py` |
| Understand the line array format | `src/streakiller/models/streak.py` |
| Add a new background method | `src/streakiller/background/` |
| Replace or extend the detector | `src/streakiller/detection/` |
| Add or modify a post-detection filter | `src/streakiller/filters/` |
| Change what gets written to disk | `src/streakiller/io/output_writer.py` |
| Write to S3 / cloud storage | Implement `OutputWriter` Protocol, inject into `StreakPipeline` |
| Understand the full run sequence | `src/streakiller/pipeline/streak_pipeline.py` |
| Run from the command line | `src/streakiller/cli/main.py` |
| See original (pre-refactor) code | `legacy/streakprocessing.py` |
