# streakiller

Satellite streak detection pipeline for FITS astronomy images.

---

## Installation

```bash
pip install -e ".[dev]"
```

> **Windows note:** The `streakiller` script may not be on your PATH after install.
> Use `python -m streakiller` as a drop-in replacement for all commands below,
> or add the Python Scripts directory to your PATH permanently:
> ```powershell
> $scriptsPath = "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"
> [Environment]::SetEnvironmentVariable("PATH", "$env:PATH;$scriptsPath", "User")
> ```
> Restart your terminal after running that command.

---

## Quick start

```bash
# Validate your config before processing
python -m streakiller validate-config --config config.json

# See which files would be processed
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
| `--images-dir PATH` | — | Directory of FITS files. Mutually exclusive with `FILES` |
| `--glob TEXT` | `*.fit*` | Glob pattern inside `--images-dir` |
| `--output-dir PATH` | *(from config)* | Override the output directory |
| `--workers INTEGER` | `1` | Number of parallel worker processes |
| `--log-format [text\|json]` | `text` | Human-readable or structured JSON logs |
| `--dry-run` | — | Print matched files and exit without processing |
| `--fail-fast` | — | Stop immediately on the first error |

**Examples:**

```bash
# Process everything in the images/ folder
python -m streakiller process --images-dir images/

# Only process files matching a pattern
python -m streakiller process --images-dir images/ --glob "Intelsat*.fits"

# Process two specific files
python -m streakiller process images/img1.fits images/img2.fits

# Write results to a custom directory
python -m streakiller process --images-dir images/ --output-dir /tmp/results

# Use 4 parallel workers for a large batch
python -m streakiller process --images-dir images/ --workers 4

# Emit structured JSON logs (useful for log aggregators or CI)
python -m streakiller process --images-dir images/ --log-format json

# Stop on the first failure instead of collecting all errors
python -m streakiller process --images-dir images/ --fail-fast
```

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | All files processed successfully |
| `1` | One or more files failed (details printed to stderr) |
| `2` | Config validation error — nothing was processed |
| `3` | No FITS files matched |

**Output files** are written to `<output_dir>/<image_stem>/` for each image:

```
output/
└── Intelsat-40_G200_05s/
    ├── detected_streaks.png       ← annotated image with bounding boxes and labels
    ├── streaks.csv                ← detected line coordinates
    ├── filter_stage_overlays.png  ← colour-coded overlay showing each filter stage
    └── processing_results.json    ← full audit record (see below)
```

If `save_intermediate_images` is enabled in config, two extra files appear:

```
    ├── binary.png                 ← foreground mask fed to Hough transform
    └── normalized_display.png     ← percentile-clipped display image
```

---

### `validate-config` — check a config file

```
python -m streakiller validate-config [OPTIONS]
```

Parses and validates `config.json`, prints the fully-resolved configuration
(with all defaults filled in), and exits. Useful for catching typos before
running a long batch.

| Option | Default | Description |
|--------|---------|-------------|
| `--config PATH` | `config.json` | Path to configuration file |

```bash
python -m streakiller validate-config
python -m streakiller validate-config --config /path/to/other_config.json
```

**Example output:**

```json
{
  "images_dir": "/absolute/path/to/images",
  "output_dir": "/absolute/path/to/output",
  "logging_level": "INFO",
  "default_minlinelength": 25,
  "hough_params": {
    "threshold": 60,
    "max_line_gap": 5,
    "rho": 1.0,
    "theta_deg": 1.0
  },
  ...
}

Config is valid.
```

If your config contains any legacy misspelled keys (`cailbration_dir`,
`endpoint_filer`, `Guassian_blur`, `doublepass_median_to_guassian_blur`),
they will be accepted with a deprecation warning showing the correct key name.

---

### `list-files` — preview matched files

```
python -m streakiller list-files [OPTIONS]
```

Lists all FITS files that would be matched, then exits. Does not process anything.

| Option | Default | Description |
|--------|---------|-------------|
| `--images-dir PATH` | *(required)* | Directory to search |
| `--glob TEXT` | `*.fit*` | Glob pattern |

```bash
python -m streakiller list-files --images-dir images/
python -m streakiller list-files --images-dir images/ --glob "*.fits"
```

---

## Configuration reference

`config.json` controls every aspect of the pipeline. All values shown are the defaults.

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

### Key options explained

#### Background detection method
Exactly one method must be set to `true`.

| Method | Best for |
|--------|----------|
| `gaussian_blur` | Most images. Smooth background gradients, computationally fast. |
| `simple_median` | Simple, clean images with a uniform background. |
| `double_pass` | Complex scenes: variable backgrounds, crowded star fields, cosmic rays. Slowest. |

#### Line filters
Filters run in a fixed order: midpoint → angle → colinear → endpoint → length.

| Filter | What it removes |
|--------|----------------|
| `midpoint_filter` | Duplicate detections whose midpoints are within 10 px of each other |
| `line_angle` | Near-parallel duplicate lines (within 10° of an already-accepted line) |
| `colinear_filter` | Merges collinear segments into a single longer segment |
| `endpoint_filter` | Lines whose endpoints are within 10 px of another line's endpoints |
| `length_filter` | Short lines below 80% of the longest detected line |

#### Satellite TLE mode
When enabled, downloads the satellite's TLE data and uses its angular velocity
at the observation time to estimate the expected streak length, replacing
`default_minlinelength`.

```json
{
    "estimated_streak_length_enabled": true,
    "norad_id": 56174,
    "tle_cache_ttl_hours": 24
}
```

The TLE is cached to disk for `tle_cache_ttl_hours` hours so repeated runs on the
same satellite don't re-download it. Requires `SITELAT`, `SITELONG`, `SITEELEV`,
and `DATE-OBS` to be present in the FITS headers.

#### Calibration
When enabled, applies dark subtraction and flat-field division before detection.

```json
{
    "image_calibration": true,
    "calibration_dir": "calibration_frames"
}
```

Expects `calibration_frames/mdark.fits` and `calibration_frames/mflat.fits`.
When disabled, a simple hot-pixel removal pass is applied instead.

#### Intermediate images
```json
{
    "save_intermediate_images": true
}
```
Saves `binary.png` (the foreground mask) and `normalized_display.png` alongside
the main outputs. Useful for diagnosing why streaks are or aren't being detected.

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

One row per detected streak segment.

```
label,x1,y1,x2,y2,midpoint_x,midpoint_y
1,112,88,934,701,523.0,394.5
2,115,91,937,704,526.0,397.5
```

Coordinates are in pixels from the top-left of the image.

### `processing_results.json`

A complete audit record for the run. Useful for comparing detection results across
different configs or software versions.

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
    "midpoint_filter": 12,
    "angle_filter": 8,
    "endpoint_filter": 4,
    "length_filter": 2
  },
  "config_snapshot": { ... }
}
```

`stage_line_counts` shows exactly how many lines survived each filter stage,
making it easy to diagnose whether a filter is being too aggressive.

---

## Troubleshooting

**No streaks detected**
- Set `"save_intermediate_images": true` and inspect `binary.png` — if it's mostly black, try switching the background method or lowering `default_minlinelength`.
- Set `"logging_level": "DEBUG"` to see the k-value ladder and threshold values being tried.
- Try `double_pass` for images with non-uniform backgrounds.

**Too many false detections**
- Enable `length_filter` and `endpoint_filter` if they're off.
- Raise `default_minlinelength` to ignore short segments.
- Enable `colinear_filter` to merge fragmented streak segments.

**Config file not found**
- Run from the directory containing `config.json`, or use `--config /full/path/config.json`.

**Calibration frames not found**
- Check that `calibration_dir` points to a folder containing `mdark.fits` and `mflat.fits`.
- Run `validate-config` to see the resolved absolute path being used.

**TLE download fails**
- Verify the NORAD ID at [celestrak.org](https://celestrak.org).
- Check your internet connection — the pipeline retries 5 times with backoff.
- Delete the cache file at `%TEMP%/streakiller_tle_cache/<norad_id>.json` to force a fresh download.

---

## Running tests

```bash
# Full suite
pytest tests/ -v

# Unit tests only (fast, no FITS files needed)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src/streakiller --cov-report=term-missing
```
