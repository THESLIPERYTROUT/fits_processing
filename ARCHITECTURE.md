# Developer Guide

> **Audience:** A developer adding a new background estimator, detector, filter,
> output writer, or CLI command. This guide covers: the source tree, how data
> moves end-to-end, all key data types, and a concrete recipe for every
> extension point.

---

## Source tree

```
src/streakiller/
├── __init__.py               # package version
├── __main__.py               # python -m streakiller entry point
│
├── config/                   # all tuneable parameters live here
│   ├── schema.py             # frozen dataclasses; PipelineConfig.from_json()
│   ├── defaults.py           # every magic number, documented with its origin
│   └── __init__.py           # public re-exports
│
├── models/                   # shared domain objects — pure data, no I/O
│   ├── fits_image.py         # FitsImage + ObservationMetadata
│   ├── streak.py             # StreakLine, FilterStageSnapshot, StreakSNR
│   └── result.py             # PipelineResult + Provenance
│
├── io/                       # all file I/O is isolated here
│   ├── fits_loader.py        # FitsLoader — reads .fits → FitsImage
│   ├── output_writer.py      # OutputWriter Protocol + LocalOutputWriter
│   └── tle_cache.py          # TleCache — downloads & caches TLE data from CelesTrak
│
├── calibration/              # dark subtraction + flat-field division
│   └── calibrator.py         # CalibrationStep — apply() returns new FitsImage
│
├── background/               # background subtraction → binary foreground mask
│   ├── base.py               # BackgroundEstimator Protocol (the interface)
│   ├── gaussian_blur.py      # GaussianBlurEstimator  (default)
│   ├── simple_median.py      # SimpleMedianEstimator
│   ├── double_pass.py        # DoublePassEstimator
│   ├── adaptive_local.py     # AdaptiveLocalEstimator
│   └── per_row_median_curve.py  # PerRowMedianCurveEstimator
│
├── detection/                # line segment detection on binary mask
│   ├── detector.py           # StreakDetector (HoughLinesP) + RawDetection dataclass
│   ├── fft_detector.py       # FftCorrelationDetector — FFT cross-correlation approach
│   ├── peak_hough_detector.py # PeakHoughDetector — sparse peak mask + Hough
│   └── normalizer.py         # normalize_for_display() utility
│
├── filters/                  # post-detection line deduplication and filtering
│   ├── chain.py              # FilterChain — runs filters in order, records snapshots
│   ├── angle.py              # angle_filter — remove angular duplicates
│   ├── colinear.py           # colinear_merge — merge collinear segments
│   ├── length.py             # length_filter — drop outlier-short/long lines
│   ├── midpoint.py           # midpoint_filter — remove midpoint-proximate duplicates
│   ├── endpoint.py           # endpoint_filter — remove endpoint-proximate duplicates
│   └── on_streak.py          # on_streak_filter — remove lines that lie on another line
│
├── snr/                      # per-streak signal-to-noise estimation
│   ├── aperture.py           # sample_apertures() — on/off streak pixel sampling
│   └── estimator.py          # StreakSNREstimator — aperture photometry per line
│
├── satellite/                # TLE-based streak length prediction
│   └── streak_estimator.py   # StreakLengthEstimator — angular velocity → pixel length
│
├── pipeline/                 # assembles everything into a single run
│   └── streak_pipeline.py    # StreakPipeline — the central orchestrator
│
└── cli/                      # command-line interface
    └── main.py               # Click commands: process, validate-config, list-files
```

---

## Module dependency rules

The layering is strict. If you break it, the build or tests will catch circular imports.

```
config/          ← no internal imports; pure data + validation
    │
models/          ← imports config/ only; pure data; no I/O; no OpenCV
    │
io/              ← imports models/ + config/; all file I/O lives here
    │
calibration/ ──┐
background/    ├─ import models/ + config/ only
detection/     │  no I/O; no cross-imports between these packages
filters/       │
satellite/    ─┘
    │
pipeline/        ← imports everything above; assembles the complete run
    │
cli/             ← imports pipeline/ + config/ only; never imported by others
```

**Rule of thumb:** processing code never reads from disk or writes files. That is `io/`'s job. `pipeline/` and `cli/` are the only places that import `io/`.

---

## Pipeline data flow

```
config.json
    │ PipelineConfig.from_json()
    ▼
┌──────────────────────────────────────────────────────────────┐
│  StreakPipeline                                               │
│                                                              │
│  FitsLoader ──► FitsImage                                    │
│                    │                                         │
│                    ▼                                         │
│            CalibrationStep (or hot-pixel removal)            │
│                    │ FitsImage                               │
│                    ▼                                         │
│          StreakLengthEstimator (optional, TLE)               │
│                    │ min_line_length: float                  │
│                    ▼                                         │
│           BackgroundEstimator ──► binary uint8 (H, W)        │
│                    │                                         │
│                    ▼                                         │
│            StreakDetector ──► RawDetection                   │
│                    │         (lines, binary_image, display)  │
│                    ▼                                         │
│            FilterChain ──► (filtered lines, snapshots)       │
│                    │                                         │
│                    ▼                                         │
│          StreakSNREstimator ──► [StreakSNR, ...]              │
│                    │                                         │
│                    ▼                                         │
│           PipelineResult + Provenance                        │
│                    │                                         │
│                    ▼                                         │
│           OutputWriter ──► disk / cloud                      │
└──────────────────────────────────────────────────────────────┘
    │
output/<stem>/
    ├── detected_streaks.png
    ├── filter_stage_overlays.png
    ├── streaks.csv
    └── processing_results.json
```

Every stage receives plain Python objects or NumPy arrays. No stage reads from disk or writes files except `FitsLoader`, `CalibrationStep.load_frames()`, `TleCache`, and `OutputWriter`. Everything else is pure computation.

---

## Key data types

These are the objects you will receive and return. Understand them before writing any algorithm code.

### `FitsImage` — `src/streakiller/models/fits_image.py`

The primary image carrier passed through every stage.

```python
@dataclass
class FitsImage:
    source_path: Optional[Path]    # None for synthetic/derived images
    data: np.ndarray               # float32, shape (H, W)
    raw_header: dict               # all original FITS header key-value pairs
    metadata: ObservationMetadata  # typed, structured header fields
```

**Never mutate `data` in place.** Return a new image via `image.derive(new_data)`:

```python
# ✅ correct
def preprocess(image: FitsImage) -> FitsImage:
    return image.derive(do_something(image.data))   # preserves metadata

# ❌ wrong
def preprocess(image: FitsImage) -> FitsImage:
    image.data[...] = do_something(image.data)      # mutates in place
    return image
```

### `ObservationMetadata` — `src/streakiller/models/fits_image.py`

Typed FITS header fields. Every field is `Optional` — always guard before use.

```python
@dataclass(frozen=True)
class ObservationMetadata:
    exposure_time: Optional[float]      # seconds
    date_obs: Optional[str]             # ISO 8601 string
    telescope: Optional[str]
    camera: Optional[str]
    focal_length_mm: Optional[float]
    lat: Optional[float]                # degrees
    lon: Optional[float]                # degrees
    elevation_m: Optional[float]        # metres
    binning: int                        # default 1
    pixel_size_um: Optional[float]      # microns, after binning
    pixel_scale_arcsec: Optional[float] # arcsec/pixel

    @property
    def has_location(self) -> bool: ... # True if lat + lon + elevation all present
```

### Lines format — `np.ndarray` shape `(N, 1, 4)` `int32`

The universal line format throughout the pipeline, matching `cv2.HoughLinesP` output exactly.

```python
lines[i]      # shape (1, 4)
lines[i][0]   # shape (4,)  — [x1, y1, x2, y2]

x1, y1, x2, y2 = lines[i][0]   # unpack one line
```

An empty result is always `np.empty((0, 1, 4), dtype=np.int32)` — never `None`. Return this when your algorithm finds nothing.

### `RawDetection` — `src/streakiller/detection/detector.py`

Returned by every detector. Bundles lines with the images used to produce them (needed by `OutputWriter` for visualisation).

```python
@dataclass
class RawDetection:
    lines: np.ndarray             # (N, 1, 4) int32
    binary_image: np.ndarray      # uint8 (H, W) — the mask fed to detection
    normalized_display: np.ndarray # uint8 (H, W) — for visualisation
```

### `PipelineResult` — `src/streakiller/models/result.py`

The complete output of one pipeline run.

```python
@dataclass
class PipelineResult:
    source_path: Optional[Path]
    detected_lines: np.ndarray             # (N, 1, 4) — final filtered lines
    filter_snapshots: list[FilterStageSnapshot]
    normalized_display: Optional[np.ndarray]
    binary_image: Optional[np.ndarray]
    provenance: Optional[Provenance]       # full audit record
    error: Optional[str]                   # set on non-fatal failure

    @property
    def streak_count(self) -> int: ...
    @property
    def succeeded(self) -> bool: ...
```

### `FilterStageSnapshot` — `src/streakiller/models/streak.py`

Created automatically by `FilterChain` for each enabled filter. You do not create these directly.

```python
@dataclass
class FilterStageSnapshot:
    stage_name: str
    lines_before: int
    lines_after: int
    lines: np.ndarray   # the surviving lines after this stage
```

---

## Where parameters live

Every tunable number lives in `src/streakiller/config/schema.py`. The defaults (with source annotations) are in `src/streakiller/config/defaults.py`. Never hard-code a constant in your algorithm module.

```
PipelineConfig
├── HoughParams          threshold, max_line_gap, rho, theta_deg
├── FilterParams         midpoint_min_distance, endpoint_min_distance,
│                        angle_min_diff_deg, length_fraction, max_length_factor,
│                        colinear_orientation_tol, on_streak_proximity_px
├── BackgroundParams     gaussian_kernel_size, gaussian_sigma_ladder,
│                        simple_median_sigma_mult, double_pass_sigma_mult,
│                        double_pass_inpaint_radius, (adaptive_local_*, per_row_*)
├── EnabledFilters       midpoint_filter, line_angle, colinear_filter,
│                        endpoint_filter, length_filter, on_streak_filter
├── BackgroundMethod     simple_median, gaussian_blur, double_pass,
│                        adaptive_local, per_row_median_curve
└── OutputOptions        save_intermediate_images
```

---

## How to implement X

---

### Add a new background estimator

Background estimators convert the float32 image into a binary foreground mask.

**1. Create the class** in `src/streakiller/background/my_estimator.py`:

```python
import numpy as np
from streakiller.config.schema import BackgroundParams


class MyEstimator:
    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        """
        data  — float32 (H, W), do not modify
        return — uint8  (H, W), values in {0, 255}
        """
        # your algorithm here
        binary = (data > some_threshold(data, params)).astype(np.uint8) * 255
        return binary
```

Contract:
- Must not modify `data`
- Must not write files or log anything at INFO level during the hot path
- Output must be `uint8` with the same shape as `data`
- Values must be `0` or `255` — not a float mask

**2. Add a toggle** to `BackgroundMethod` in `src/streakiller/config/schema.py`:

```python
@dataclass
class BackgroundMethod:
    gaussian_blur: bool = True
    simple_median: bool = False
    double_pass: bool = False
    my_estimator: bool = False   # ← add this
```

Add its parameters to `BackgroundParams` if needed, and document the defaults in `src/streakiller/config/defaults.py`.

**3. Wire it** in `StreakPipeline.__init__()` in `src/streakiller/pipeline/streak_pipeline.py`:

```python
elif config.background_detection_method.my_estimator:
    from streakiller.background.my_estimator import MyEstimator
    self._background_estimator = MyEstimator()
```

**4. Add to the shared contract tests** in `tests/unit/test_background_estimators.py`:

```python
ESTIMATORS = [
    GaussianBlurEstimator(),
    SimpleMedianEstimator(),
    DoublePassEstimator(),
    MyEstimator(),   # ← add this
]
```

All shared contract tests (output shape, output dtype, no-mutation) run automatically.

---

### Add a new detector

Detectors receive the binary foreground mask and return line segments.

**1. Create the class** in `src/streakiller/detection/my_detector.py`:

```python
import logging
import numpy as np
from streakiller.detection.detector import RawDetection
from streakiller.detection.normalizer import normalize_for_display

logger = logging.getLogger(__name__)


class MyDetector:
    def __init__(self, params) -> None:
        self._params = params

    def detect(
        self,
        binary: np.ndarray,       # uint8 (H, W) foreground mask
        source_data: np.ndarray,  # float32 (H, W) original image — display only
        min_line_length: float,
    ) -> RawDetection:
        try:
            lines = self._find_lines(binary, min_line_length)
        except Exception as exc:
            logger.error("MyDetector failed: %s", exc)
            lines = np.empty((0, 1, 4), dtype=np.int32)

        return RawDetection(
            lines=lines,
            binary_image=binary,
            normalized_display=normalize_for_display(source_data),
        )

    def _find_lines(self, binary: np.ndarray, min_length: float) -> np.ndarray:
        # must return shape (N, 1, 4) int32
        ...
```

Contract:
- Never return `None`
- Never raise — catch all exceptions and return an empty `RawDetection`
- `lines` must be `(N, 1, 4)` `int32`; empty case is `np.empty((0, 1, 4), dtype=np.int32)`
- `source_data` is for display only — do not use it to make detection decisions

**2. Add a params class and config toggle** in `src/streakiller/config/schema.py`, and add defaults to `defaults.py`.

**3. Wire it** in `StreakPipeline.__init__()`:

```python
if config.use_my_detector:
    from streakiller.detection.my_detector import MyDetector
    self._detector = MyDetector(config.my_detector_params)
else:
    self._detector = StreakDetector(config.hough_params)
```

For quick local testing without touching config, inject directly:

```python
pipeline = StreakPipeline(config=cfg, output_writer=None)
pipeline._detector = MyDetector(MyDetectorParams())
```

**4. Write tests** in `tests/unit/test_my_detector.py`:

```python
import numpy as np
import pytest
from streakiller.detection.my_detector import MyDetector


@pytest.fixture
def empty_binary():
    return np.zeros((128, 128), dtype=np.uint8)

@pytest.fixture
def streak_binary():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[64, 10:118] = 255
    return img


class TestMyDetector:
    def test_empty_input_returns_empty_lines(self, empty_binary):
        result = MyDetector(MyDetectorParams()).detect(
            empty_binary, empty_binary.astype(np.float32), min_line_length=25
        )
        assert result.lines.shape == (0, 1, 4)
        assert result.lines.dtype == np.int32

    def test_detects_clear_streak(self, streak_binary):
        result = MyDetector(MyDetectorParams()).detect(
            streak_binary, streak_binary.astype(np.float32), min_line_length=25
        )
        assert len(result.lines) >= 1

    def test_output_shape_when_found(self, streak_binary):
        result = MyDetector(MyDetectorParams()).detect(
            streak_binary, streak_binary.astype(np.float32), min_line_length=10
        )
        if len(result.lines) > 0:
            assert result.lines.ndim == 3
            assert result.lines.shape[1:] == (1, 4)
            assert result.lines.dtype == np.int32
```

---

### Add a new post-detection filter

Filters are pure functions that receive all lines and return a subset (or a merged set).

**1. Create the function** in `src/streakiller/filters/my_filter.py`:

```python
import numpy as np
from streakiller.config.schema import FilterParams


def my_filter(lines: np.ndarray, params: FilterParams) -> np.ndarray:
    """
    lines  — (N, 1, 4) int32
    return — (M, 1, 4) int32, M <= N (or merged, M could be < N for different reasons)

    Pure function — no side effects, do not modify the input array.
    Return np.empty((0, 1, 4), dtype=np.int32) if no lines survive.
    """
    if len(lines) == 0:
        return lines

    # unpack all lines at once
    x1 = lines[:, 0, 0]; y1 = lines[:, 0, 1]
    x2 = lines[:, 0, 2]; y2 = lines[:, 0, 3]

    # ... your logic ...
    keep = np.ones(len(lines), dtype=bool)
    return lines[keep]
```

**2. Add a bool to `EnabledFilters`** in `src/streakiller/config/schema.py`:

```python
@dataclass
class EnabledFilters:
    midpoint_filter: bool = True
    line_angle: bool = True
    colinear_filter: bool = False
    endpoint_filter: bool = True
    length_filter: bool = True
    my_filter: bool = False   # ← add this
```

**3. Register it in `FilterChain.from_config()`** in `src/streakiller/filters/chain.py` at the correct position in the execution order:

```python
# current order: angle → colinear → length → midpoint → endpoint → on_streak
if enabled.my_filter:
    filters.append(("my_filter", my_filter))
```

**4. Add to the shared edge-case tests** in `tests/unit/test_filters.py`:

```python
FILTER_FNS = [
    angle_filter, colinear_merge, length_filter,
    midpoint_filter, endpoint_filter, my_filter,  # ← add
]
```

Shared tests cover: empty input, single line, all-identical lines, non-mutation of input.

---

### Add a new output writer

Output writers consume a completed `PipelineResult` and write it somewhere.

**1. Implement the protocol** in a new file or alongside your integration:

```python
from streakiller.models.result import PipelineResult


class S3OutputWriter:
    def __init__(self, bucket: str, prefix: str) -> None:
        self._bucket = bucket
        self._prefix = prefix

    def write(self, result: PipelineResult) -> None:
        # upload result.detected_lines, result.provenance, etc.
        ...
```

No base class or inheritance is required — Python structural typing means any object with a `write(result: PipelineResult) -> None` method satisfies the `OutputWriter` Protocol.

**2. Inject it** when constructing the pipeline:

```python
writer = S3OutputWriter(bucket="my-bucket", prefix="streaks/")
pipeline = StreakPipeline.from_config(config, output_writer=writer)
```

Or pass `output_writer=None` to skip all output (useful in tests and notebooks).

---

### Add a new CLI command

CLI commands are Click functions registered in `src/streakiller/cli/main.py`.

**1. Add the command** to the `cli` group:

```python
@cli.command()
@click.option("--images-dir", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default="config.json", type=click.Path())
def my_command(images_dir: str, config_path: str) -> None:
    """One-line description shown in --help."""
    try:
        config = PipelineConfig.from_json(config_path)
    except ConfigError as exc:
        click.echo(f"Config error: {exc}", err=True)
        raise SystemExit(2)

    # your logic
    click.echo("done")
```

**2. Exit codes** should follow the convention established by `process`:
- `0` — success
- `1` — partial failure (file-level errors, not config)
- `2` — config error
- `3` — no files matched

---

## Testing patterns

### Fixtures

All shared fixtures are in `tests/conftest.py`:

| Fixture | What it provides |
|---------|-----------------|
| `minimal_config` | Minimal `PipelineConfig` with all required fields and safe defaults |
| `synthetic_fits_image` | 512×512 `FitsImage` with a diagonal streak injected at a known position |
| `mock_tle_text` | Valid TLE text string for satellite estimator tests |
| `sample_config_json` | Path to a temp `config.json` file for config-loading tests |

### Writing a unit test

```python
import numpy as np
from streakiller.filters.my_filter import my_filter


def make_lines(*segments):
    """Helper: make_lines((0,0,10,0), (5,5,15,5)) → (2,1,4) int32"""
    return np.array([[[*s]] for s in segments], dtype=np.int32)


class TestMyFilter:
    def test_empty_input(self, minimal_config):
        lines = np.empty((0, 1, 4), dtype=np.int32)
        result = my_filter(lines, minimal_config.filter_params)
        assert result.shape == (0, 1, 4)

    def test_does_not_mutate_input(self, minimal_config):
        lines = make_lines((0, 0, 100, 0), (0, 2, 100, 2))
        original = lines.copy()
        my_filter(lines, minimal_config.filter_params)
        np.testing.assert_array_equal(lines, original)
```

### Testing the full pipeline

Use `StreakPipeline` with `output_writer=None` and the `synthetic_fits_image` fixture:

```python
from streakiller.pipeline.streak_pipeline import StreakPipeline


def test_pipeline_detects_injected_streak(synthetic_fits_image, minimal_config):
    pipeline = StreakPipeline.from_config(minimal_config, output_writer=None)
    result = pipeline.process(synthetic_fits_image)
    assert result.succeeded
    assert result.streak_count >= 1
```

---

## Quick reference

| I want to... | Go here |
|---|---|
| Add a background estimator | `src/streakiller/background/` + wire in `pipeline/streak_pipeline.py` |
| Add a detector | `src/streakiller/detection/` + wire in `pipeline/streak_pipeline.py` |
| Add a post-detection filter | `src/streakiller/filters/` + register in `filters/chain.py` |
| Add an output writer | Implement `write(result)`, inject into `StreakPipeline` |
| Add a CLI command | `src/streakiller/cli/main.py` |
| Change a tunable number | `src/streakiller/config/defaults.py` → expose via `schema.py` |
| Read the image from disk | `src/streakiller/io/fits_loader.py` |
| Access image pixels / headers | `src/streakiller/models/fits_image.py` |
| Understand the lines array | `src/streakiller/models/streak.py` |
| Understand the full run sequence | `src/streakiller/pipeline/streak_pipeline.py` |
| Write to S3 / cloud storage | Implement `OutputWriter` Protocol — see *Add a new output writer* |
