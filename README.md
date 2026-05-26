# streakiller

Satellite streak detection pipeline for FITS astronomy images.

Detects and measures satellite streak artifacts in astronomical frames using configurable background subtraction, probabilistic Hough line detection, and per-streak SNR aperture photometry. Each processed image produces annotated overlays, CSV coordinates, and a complete JSON audit record.

---

## Installation

```bash
pip install -e ".[dev]"
```

> **Windows:** The `streakiller` entry-point script may not be on PATH after install. Use `python -m streakiller` as a drop-in replacement for all commands, or add the Python Scripts directory to your PATH permanently:
> ```powershell
> $s = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"
> [Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$s", "User")
> ```
> Restart your terminal afterwards.

---

## Quick start

```bash
# Validate config before processing
python -m streakiller validate-config --config config.json

# Preview which files would be processed
python -m streakiller process --images-dir images/ --dry-run

# Process all FITS files in a directory
python -m streakiller process --images-dir images/

# Process specific files
python -m streakiller process images/img1.fits images/img2.fits
```

---

## Commands

### `process` — detect streaks

```
python -m streakiller process [OPTIONS] [FILES...]
```

Processes one or more FITS files and writes results to the output directory.

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | `config.json` | Path to configuration file |
| `--images-dir PATH` | — | Directory of FITS files (mutually exclusive with `FILES`) |
| `--glob TEXT` | `*.fit*` | Glob pattern when using `--images-dir` |
| `--output-dir PATH` | *(from config)* | Override the output directory |
| `--workers INTEGER` | `1` | Number of parallel worker processes |
| `--log-format [text\|json]` | `text` | Human-readable or structured JSON logs |
| `--dry-run` | — | Print matched files and exit without processing |
| `--fail-fast` | — | Stop immediately on the first error |

```bash
# Only files matching a glob
python -m streakiller process --images-dir images/ --glob "Intelsat*.fits"

# Write results to a custom directory
python -m streakiller process --images-dir images/ --output-dir /tmp/results

# Four parallel workers for a large batch
python -m streakiller process --images-dir images/ --workers 4

# Structured JSON logs (useful for log aggregators / CI)
python -m streakiller process --images-dir images/ --log-format json
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | All files processed successfully |
| `1` | One or more files failed (details on stderr) |
| `2` | Config validation error — nothing was processed |
| `3` | No FITS files matched |

**Output layout** (one subdirectory per image):

```
output/
└── Intelsat-40_G200_05s/
    ├── detected_streaks.png       # annotated image with bounding boxes and labels
    ├── streaks.csv                # detected line coordinates
    ├── filter_stage_overlays.png  # colour-coded overlay per filter stage
    └── processing_results.json   # full audit record
```

With `"save_intermediate_images": true`:

```
    ├── binary.png                 # foreground mask fed to the detector
    └── normalized_display.png     # percentile-clipped display image
```

---

### `validate-config` — check a config file

```
python -m streakiller validate-config [OPTIONS]
```

Parses and validates `config.json`, prints the fully-resolved configuration with all defaults filled in, and exits. Run this before a long batch to catch typos.

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | `config.json` | Path to configuration file |

Legacy misspelled keys (`cailbration_dir`, `Guassian_blur`, `endpoint_filer`, `doublepass_median_to_guassian_blur`) are accepted with a deprecation warning that shows the correct key name.

---

### `list-files` — preview matched files

```
python -m streakiller list-files [OPTIONS]
```

Lists all FITS files that would be matched, then exits without processing anything.

| Option | Default | Description |
|--------|---------|-------------|
| `--images-dir PATH` | *(required)* | Directory to search |
| `--glob TEXT` | `*.fit*` | Glob pattern |

---

## Configuration reference

`config.json` controls every aspect of the pipeline. All values shown below are the defaults.

```json
{
    "images_dir": "images",
    "output_dir": "output",
    "logging_level": "INFO",

    "image_calibration": false,
    "calibration_dir": "calibration_frames",

    "estimated_streak_length_enabled": false,
    "norad_id": null,
    "tle_cache_ttl_hours": 24,

    "default_minlinelength": 25,
    "save_intermediate_images": false,

    "background_detection_method": {
        "simple_median": false,
        "gaussian_blur": true,
        "double_pass": false
    },

    "enabled_line_filters": {
        "midpoint_filter": true,
        "line_angle": true,
        "colinear_filter": false,
        "endpoint_filter": true,
        "length_filter": true
    }
}
```

### Background detection method

Exactly one method must be `true`.

| Method | When to use |
|--------|-------------|
| `gaussian_blur` | Default. Most images — handles smooth background gradients efficiently. |
| `simple_median` | Clean images with a uniform background and no gradients. |
| `double_pass` | Complex scenes: variable backgrounds, crowded star fields, cosmic rays. Slowest. |
| `adaptive_local` | Hybrid: Gaussian high-pass + per-tile MAD noise mesh. Good for patchy backgrounds. |
| `per_row_median_curve` | Images with left-to-right brightness gradients along rows. |

Full method details: [`docs/gaussian_blur_background_report.md`](docs/gaussian_blur_background_report.md), [`docs/simple_median_background_report.md`](docs/simple_median_background_report.md), [`docs/double_pass_background_report.md`](docs/double_pass_background_report.md), [`docs/adaptive_local_background_report.md`](docs/adaptive_local_background_report.md).

### Line filters

Filters run in a fixed order: `angle → colinear → length → midpoint → endpoint`.

| Filter | What it removes |
|--------|----------------|
| `line_angle` | Near-duplicate lines within 10° of an already-accepted line |
| `colinear_filter` | Merges collinear segments separated by a gap into a single longer segment |
| `length_filter` | Short lines below 88% of the median detected length; also drops outlier-long lines above 1.4× median |
| `midpoint_filter` | Lines whose midpoint is within 10 px of an accepted line's midpoint |
| `endpoint_filter` | Lines whose endpoints are within 10 px of an accepted line's endpoints (shorter line loses) |

### TLE-based streak length estimation

When enabled, uses satellite TLE data to estimate expected streak length from angular velocity, replacing `default_minlinelength`.

```json
{
    "estimated_streak_length_enabled": true,
    "norad_id": 56174,
    "tle_cache_ttl_hours": 24
}
```

Requires `SITELAT`, `SITELONG`, `SITEELEV`, and `DATE-OBS` in the FITS headers. TLE data is cached to disk for `tle_cache_ttl_hours` so repeated runs on the same satellite do not re-download.

### Calibration

Applies dark subtraction and flat-field division before detection.

```json
{
    "image_calibration": true,
    "calibration_dir": "calibration_frames"
}
```

Expects `calibration_frames/mdark.fits` and `calibration_frames/mflat.fits`. When disabled, a hot-pixel removal pass is applied instead.

### Intermediate images

```json
{ "save_intermediate_images": true }
```

Saves `binary.png` (the foreground mask) and `normalized_display.png` alongside the main outputs. Use these to diagnose why streaks are or are not being detected.

---

### Environment variable overrides

Any config value can be overridden at runtime without editing the file:

| Variable | Config key |
|----------|------------|
| `STREAKILLER_IMAGES_DIR` | `images_dir` |
| `STREAKILLER_OUTPUT_DIR` | `output_dir` |
| `STREAKILLER_LOGGING_LEVEL` | `logging_level` |
| `STREAKILLER_NORAD_ID` | `norad_id` |
| `STREAKILLER_TLE_CACHE_TTL_HOURS` | `tle_cache_ttl_hours` |

```bash
STREAKILLER_OUTPUT_DIR=/tmp/results python -m streakiller process --images-dir images/
```

---

## Understanding the output

### `streaks.csv`

One row per detected streak segment. Coordinates are in pixels from the top-left corner.

```
label,x1,y1,x2,y2,midpoint_x,midpoint_y
1,112,88,934,701,523.0,394.5
2,115,91,937,704,526.0,397.5
```

### `processing_results.json`

Complete audit record for the run. The `stage_line_counts` field shows exactly how many lines survived each filter, making it easy to diagnose whether a filter is being too aggressive.

```json
{
  "source_file": "images/Intelsat-40_G200_05s.fits",
  "streak_count": 2,
  "error": null,
  "software_version": "0.1.0",
  "processing_start_utc": "2026-03-29T14:23:01.123456+00:00",
  "processing_end_utc": "2026-03-29T14:23:04.891234+00:00",
  "background_method_used": "gaussian_blur",
  "min_line_length_used": 25.0,
  "hough_threshold_used": 60,
  "stage_line_counts": {
    "detected": 18,
    "angle_filter": 14,
    "length_filter": 8,
    "midpoint_filter": 4,
    "endpoint_filter": 2
  },
  "config_snapshot": { "..." : "..." }
}
```

---

## Troubleshooting

**No streaks detected**
- Enable `"save_intermediate_images": true` and inspect `binary.png`. If it is mostly black, the background estimator is suppressing the streak.
- Try a different background method (`double_pass` for non-uniform backgrounds).
- Lower `default_minlinelength` or reduce the Hough `threshold`.
- Set `"logging_level": "DEBUG"` to see the k-value ladder and threshold values being tried.

**Too many false detections**
- Enable `length_filter` and `endpoint_filter` if they are off.
- Raise `default_minlinelength` to discard short segments.
- Enable `colinear_filter` to merge fragmented detections into fewer lines.

**Config file not found**
- Run from the directory containing `config.json`, or pass `--config /full/path/config.json`.

**Calibration frames not found**
- Check that `calibration_dir` points to a folder with `mdark.fits` and `mflat.fits`.
- Run `validate-config` to see the resolved absolute path.

**TLE download fails**
- Verify the NORAD ID at [celestrak.org](https://celestrak.org).
- The pipeline retries 5 times with exponential backoff; check your connection.
- Delete the cache file at `%TEMP%/streakiller_tle_cache/<norad_id>.json` to force a fresh download.

---

## Running tests

```bash
# Full suite
pytest tests/ -v

# Unit tests only (fast, no FITS files needed)
pytest tests/unit/ -v

# With coverage report
pytest tests/ --cov=src/streakiller --cov-report=term-missing
```
