# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run unit tests only (no FITS files required)
pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_config.py -v

# Run with coverage
pytest tests/ --cov=src/streakiller --cov-report=term-missing

# Run the pipeline (use python -m on Windows if the script isn't on PATH)
python -m streakiller process --images-dir images/
python -m streakiller validate-config --config config.json
python -m streakiller list-files --images-dir images/
```

## Architecture

`StreakPipeline` (`src/streakiller/pipeline/streak_pipeline.py`) is the central orchestrator. Its `process(FitsImage) -> PipelineResult` method is stateless and runs these stages in order:

1. **Calibration or hot-pixel removal** — `src/streakiller/calibration/calibrator.py`
2. **TLE-based streak length estimation** (optional) — `src/streakiller/satellite/streak_estimator.py`
3. **Background estimation → binary mask** — `src/streakiller/background/`
4. **Detection** — `src/streakiller/detection/` (Hough or FFT cross-correlation)
5. **Filter chain** — `src/streakiller/filters/chain.py`
6. **SNR estimation** — `src/streakiller/snr/`
7. **Output writing** — `src/streakiller/io/output_writer.py`

All dependencies are constructor-injected into `StreakPipeline`, making the test surface clean. `StreakPipeline.from_config(config)` wires the real dependencies; tests can pass lightweight fakes directly.

### Key protocols / extension points

- `BackgroundEstimator` (`src/streakiller/background/base.py`) — Protocol with `estimate(data, params) -> uint8 mask`. Implementations: `GaussianBlurEstimator`, `SimpleMedianEstimator`, `DoublePassEstimator`, `AdaptiveLocalEstimator`, `PerRowMedianCurveEstimator`.
- `OutputWriter` (`src/streakiller/io/output_writer.py`) — Protocol with `write(result)`. `LocalOutputWriter` is the only current implementation; a future cloud writer just needs to satisfy the same interface.
- `FilterChain` (`src/streakiller/filters/chain.py`) — Ordered pipeline of pure filter functions, each `(ndarray, FilterParams) -> ndarray`. Filter order is fixed: `length → midpoint → angle → colinear → endpoint`.

### Config system

`PipelineConfig` (`src/streakiller/config/schema.py`) is a frozen dataclass loaded via `PipelineConfig.from_json(path)`. It:
- Remaps legacy misspelled keys (e.g. `Guassian_blur` → `gaussian_blur`) with `DeprecationWarning`
- Resolves relative paths against the config file's parent directory
- Applies `STREAKILLER_*` environment variable overrides on top of the file
- `validate()` raises `ConfigError` for invalid combinations (e.g. multiple background methods enabled)

Numeric defaults live in `src/streakiller/config/defaults.py` and are imported into `schema.py`.

### Data model

- `FitsImage` (`src/streakiller/models/fits_image.py`) — immutable FITS image with `data` (float32 ndarray) and `ObservationMetadata`
- `PipelineResult` (`src/streakiller/models/result.py`) — complete output: `detected_lines` (shape `(N, 1, 4)` int32, same as `cv2.HoughLinesP`), `filter_snapshots`, `snr_estimates`, `provenance`, and optional intermediate images
- `Provenance` — frozen audit record serialized into `processing_results.json`

### Detection methods

Two detectors, exactly one enabled per run (config `detection_method`):
- `StreakDetector` (`src/streakiller/detection/detector.py`) — probabilistic Hough transform via `cv2.HoughLinesP`
- `FftCorrelationDetector` (`src/streakiller/detection/fft_detector.py`) — FFT cross-correlation approach for cases where Hough struggles

### CLI

Built with Click (`src/streakiller/cli/main.py`), entry point `streakiller`. Three subcommands: `process`, `validate-config`, `list-files`. Multi-worker batch processing uses `ProcessPoolExecutor`; the worker function must be a module-level callable for pickling.

### Tests

- `tests/unit/` — fast, no real FITS files; uses `synthetic_fits_image` fixture (512×512 with injected streak) from `tests/conftest.py`
- `tests/integration/test_full_pipeline.py` — end-to-end pipeline test
- `tests/conftest.py` — shared fixtures: `minimal_config`, `synthetic_fits_image`, `mock_tle_text`, `sample_config_json`
