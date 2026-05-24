"""
Pipeline configuration schema.

All parameters have sensible defaults matching the original hard-coded values.
Use PipelineConfig.from_json() to load from a JSON file with environment variable
overrides. Use PipelineConfig.validate() to catch invalid combinations early.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from streakiller.config.defaults import (
    HOUGH_THRESHOLD,
    HOUGH_MAX_LINE_GAP,
    HOUGH_RHO,
    HOUGH_THETA_DEG,
    MIDPOINT_MIN_DISTANCE,
    ENDPOINT_MIN_DISTANCE,
    ANGLE_MIN_DIFF_DEG,
    ANGLE_FILTER_MIN_LINES,
    LENGTH_FRACTION,
    MAX_LENGTH_FACTOR,
    COLINEAR_ORIENTATION_TOL,
    COLINEAR_MAX_ENDPOINT_DISTANCE,
    ON_STREAK_PROXIMITY_PX,
    MAX_RAW_LINES,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA_LADDER,
    SIMPLE_MEDIAN_SIGMA_MULT,
    DOUBLE_PASS_SIGMA_MULT,
    DOUBLE_PASS_INPAINT_RADIUS,
    HOTPIXEL_THRESHOLD,
    HOTPIXEL_SIGMA,
    HOTPIXEL_MAX_CLUSTER_SIZE,
    HOTPIXEL_NEIGHBORHOOD,
    ADAPTIVE_LOCAL_TILE_SIZE,
    ADAPTIVE_LOCAL_CLIP_SIGMA,
    ADAPTIVE_LOCAL_N_ITERATIONS,
    ADAPTIVE_LOCAL_SNR_THRESHOLD,
    ADAPTIVE_LOCAL_MIN_TILE_PIXELS,
    ADAPTIVE_LOCAL_MORPH_KERNEL,
    ADAPTIVE_LOCAL_GAUSSIAN_KERNEL_SIZE,
    PER_ROW_MEDIAN_BINS,
    PER_ROW_MEDIAN_DEGREE,
    PER_ROW_MEDIAN_SMOOTH_SIGMA,
    PER_ROW_MEDIAN_ROW_WINDOW,
    PER_ROW_MEDIAN_SIGMA_MULT,
    PER_ROW_MEDIAN_FILTER_SIZE,
    PER_ROW_MEDIAN_MIN_COMPONENT_PIXELS,
    PER_ROW_MEDIAN_MORPH_KERNEL,
    SNR_HALF_WIDTH_PX,
    SNR_OFF_GAP_PX,
    SNR_OFF_WIDTH_PX,
    SNR_MIN_OFF_PIXELS,
    FFT_THRESHOLD_SIGMA,
    FFT_MIN_DISTANCE,
    FFT_MIN_TEMPLATE_AREA,
    FFT_TEMPLATE_PADDING,
    FFT_MAX_WIDTH_STD,
    FFT_MIN_ELONGATION,
    FFT_PERCENTILE_THRESHOLD,
    FFT_TEMPLATE_EDGE_MARGIN,
    FFT_STREAK_EDGE_MARGIN,
    FFT_PROMINENCE_FRACTION,
    PEAK_HOUGH_CLIP_PERCENTILE_LOW,
    PEAK_HOUGH_CLIP_PERCENTILE_HIGH,
    PEAK_HOUGH_MEDIAN_BINS,
    PEAK_HOUGH_POLYNOMIAL_DEGREE,
    PEAK_HOUGH_BACKGROUND_SMOOTH_SIGMA,
    PEAK_HOUGH_THRESHOLD_SIGMA,
    PEAK_HOUGH_HOUGH_THRESHOLD,
    PEAK_HOUGH_MAX_LINE_GAP,
    PEAK_HOUGH_DILATION_KERNEL,
)

# Keys in old config.json that were misspelled.  Maps old_key -> canonical_key.
_COMPAT_KEY_MAP: dict[str, str] = {
    "cailbration_dir": "calibration_dir",
    "endpoint_filer": "endpoint_filter",        # filter name inside enabled_line_filters
    "Guassian_blur": "gaussian_blur",            # background method key
    "doublepass_median_to_guassian_blur": "double_pass",
}


class ConfigError(ValueError):
    """Raised when the pipeline configuration is invalid."""


@dataclass
class HoughParams:
    threshold: int = HOUGH_THRESHOLD
    max_line_gap: int = HOUGH_MAX_LINE_GAP
    rho: float = HOUGH_RHO
    theta_deg: float = HOUGH_THETA_DEG


@dataclass
class PeakHoughParams:
    # per-row polynomial background subtraction
    clip_percentile_low: float     = PEAK_HOUGH_CLIP_PERCENTILE_LOW
    clip_percentile_high: float    = PEAK_HOUGH_CLIP_PERCENTILE_HIGH
    median_bins: int               = PEAK_HOUGH_MEDIAN_BINS
    polynomial_degree: int         = PEAK_HOUGH_POLYNOMIAL_DEGREE
    background_smooth_sigma: float = PEAK_HOUGH_BACKGROUND_SMOOTH_SIGMA
    threshold_sigma: float         = PEAK_HOUGH_THRESHOLD_SIGMA
    # Hough parameters (lower threshold than dense-mask StreakDetector — sparse mask has fewer pixels)
    hough_threshold: int           = PEAK_HOUGH_HOUGH_THRESHOLD
    max_line_gap: int              = PEAK_HOUGH_MAX_LINE_GAP
    rho: float                     = HOUGH_RHO
    theta_deg: float               = HOUGH_THETA_DEG
    dilation_kernel: int           = PEAK_HOUGH_DILATION_KERNEL

    @classmethod
    def from_dict(cls, raw: dict) -> "PeakHoughParams":
        remapped = _remap_keys(raw)
        values = {
            field_name: remapped[field_name]
            for field_name in cls.__dataclass_fields__
            if field_name in remapped
        }
        return cls(**values)


@dataclass
class FilterParams:
    midpoint_min_distance: float = MIDPOINT_MIN_DISTANCE
    endpoint_min_distance: float = ENDPOINT_MIN_DISTANCE
    angle_min_diff_deg: float = ANGLE_MIN_DIFF_DEG
    angle_filter_min_lines: int = ANGLE_FILTER_MIN_LINES
    length_fraction: float = LENGTH_FRACTION
    max_length_factor: float = MAX_LENGTH_FACTOR
    colinear_orientation_tol: float = COLINEAR_ORIENTATION_TOL
    colinear_max_endpoint_distance: float = COLINEAR_MAX_ENDPOINT_DISTANCE
    on_streak_proximity_px: float = ON_STREAK_PROXIMITY_PX
    max_raw_lines: int = MAX_RAW_LINES


@dataclass
class BackgroundParams:
    gaussian_kernel_size: int = GAUSSIAN_KERNEL_SIZE
    gaussian_sigma_ladder: tuple = field(default_factory=lambda: GAUSSIAN_SIGMA_LADDER)
    simple_median_sigma_mult: float = SIMPLE_MEDIAN_SIGMA_MULT
    double_pass_sigma_mult: float = DOUBLE_PASS_SIGMA_MULT
    double_pass_inpaint_radius: int = DOUBLE_PASS_INPAINT_RADIUS
    adaptive_local_tile_size: int = ADAPTIVE_LOCAL_TILE_SIZE
    adaptive_local_clip_sigma: float = ADAPTIVE_LOCAL_CLIP_SIGMA
    adaptive_local_n_iterations: int = ADAPTIVE_LOCAL_N_ITERATIONS
    adaptive_local_snr_threshold: float = ADAPTIVE_LOCAL_SNR_THRESHOLD
    adaptive_local_min_tile_pixels: int = ADAPTIVE_LOCAL_MIN_TILE_PIXELS
    adaptive_local_morph_kernel: int = ADAPTIVE_LOCAL_MORPH_KERNEL
    adaptive_local_gaussian_kernel_size: int = ADAPTIVE_LOCAL_GAUSSIAN_KERNEL_SIZE
    per_row_median_bins: int = PER_ROW_MEDIAN_BINS
    per_row_median_degree: int = PER_ROW_MEDIAN_DEGREE
    per_row_median_smooth_sigma: float = PER_ROW_MEDIAN_SMOOTH_SIGMA
    per_row_median_row_window: int = PER_ROW_MEDIAN_ROW_WINDOW
    per_row_median_sigma_mult: float = PER_ROW_MEDIAN_SIGMA_MULT
    per_row_median_filter_size: int = PER_ROW_MEDIAN_FILTER_SIZE
    per_row_median_min_component_pixels: int = PER_ROW_MEDIAN_MIN_COMPONENT_PIXELS
    per_row_median_morph_kernel: int = PER_ROW_MEDIAN_MORPH_KERNEL

    @classmethod
    def from_dict(cls, raw: dict) -> "BackgroundParams":
        remapped = _remap_keys(raw)
        values = {
            field_name: remapped[field_name]
            for field_name in cls.__dataclass_fields__
            if field_name in remapped
        }
        if "gaussian_sigma_ladder" in values:
            values["gaussian_sigma_ladder"] = tuple(values["gaussian_sigma_ladder"])
        return cls(**values)


@dataclass
class EnabledFilters:
    midpoint_filter: bool = True
    line_angle: bool = True
    colinear_filter: bool = False
    on_streak_filter: bool = False
    endpoint_filter: bool = True
    length_filter: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "EnabledFilters":
        # Remap any legacy misspelled keys before building the dataclass.
        remapped = _remap_keys(raw)
        return cls(
            midpoint_filter=remapped.get("midpoint_filter", True),
            line_angle=remapped.get("line_angle", True),
            colinear_filter=remapped.get("colinear_filter", False),
            on_streak_filter=remapped.get("on_streak_filter", False),
            endpoint_filter=remapped.get("endpoint_filter", True),
            length_filter=remapped.get("length_filter", True),
        )


@dataclass
class BackgroundMethod:
    simple_median: bool = False
    gaussian_blur: bool = True
    double_pass: bool = False
    adaptive_local: bool = False
    per_row_median_curve: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "BackgroundMethod":
        remapped = _remap_keys(raw)
        return cls(
            simple_median=remapped.get("simple_median", False),
            gaussian_blur=remapped.get("gaussian_blur", True),
            double_pass=remapped.get("double_pass", False),
            adaptive_local=remapped.get("adaptive_local", False),
            per_row_median_curve=remapped.get("per_row_median_curve", False),
        )

    def active_name(self) -> str:
        if self.simple_median:
            return "simple_median"
        if self.gaussian_blur:
            return "gaussian_blur"
        if self.double_pass:
            return "double_pass"
        if self.adaptive_local:
            return "adaptive_local"
        if self.per_row_median_curve:
            return "per_row_median_curve"
        return "gaussian_blur"  # fallback, validate() will catch multiple-enabled


@dataclass
class SnrParams:
    """
    Aperture-photometry parameters for per-streak SNR estimation.

    For each detected streak the estimator samples two rectangular apertures
    perpendicular to the streak axis:

      |← off_width →|← off_gap →|← 2*half_width+1 →|← off_gap →|← off_width →|
                                        (streak)

    SNR = (mean_on − median_off) / (MAD_off × 1.4826)
    """

    half_width_px: int = SNR_HALF_WIDTH_PX
    off_gap_px: int = SNR_OFF_GAP_PX
    off_width_px: int = SNR_OFF_WIDTH_PX
    min_off_pixels: int = SNR_MIN_OFF_PIXELS


@dataclass
class FftDetectorParams:
    """
    Parameters for the FFT cross-correlation streak detector.

    See defaults.py for the rationale behind each value.
    """

    threshold_sigma: float = FFT_THRESHOLD_SIGMA
    min_distance: int = FFT_MIN_DISTANCE
    min_template_area: int = FFT_MIN_TEMPLATE_AREA
    template_padding: int = FFT_TEMPLATE_PADDING
    max_width_std: float = FFT_MAX_WIDTH_STD
    min_elongation: float = FFT_MIN_ELONGATION
    percentile_threshold: float = FFT_PERCENTILE_THRESHOLD
    template_edge_margin: int = FFT_TEMPLATE_EDGE_MARGIN
    streak_edge_margin: int = FFT_STREAK_EDGE_MARGIN
    prominence_fraction: float = FFT_PROMINENCE_FRACTION


@dataclass
class DetectionMethod:
    """Selects which streak detector is active.  Exactly one must be True."""

    hough: bool = True
    fft_correlation: bool = False
    peak_hough: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "DetectionMethod":
        hough_default = not raw.get("fft_correlation", False) and not raw.get(
            "peak_hough", False
        )
        return cls(
            hough=raw.get("hough", hough_default),
            fft_correlation=raw.get("fft_correlation", False),
            peak_hough=raw.get("peak_hough", False),
        )

    def active_name(self) -> str:
        if self.peak_hough:
            return "peak_hough"
        if self.fft_correlation:
            return "fft_correlation"
        return "hough"


@dataclass
class HotpixelParams:
    """Parameters for the statistical hot-pixel removal step."""
    # Statistical threshold: flag pixels more than N·σ_MAD above the image median.
    # Set to 0 to rely solely on hotpixel_threshold (absolute ADU floor).
    threshold_sigma: float = HOTPIXEL_SIGMA
    # Isolated-feature guard: connected regions larger than this many pixels are
    # preserved unchanged (cosmic-ray tracks, saturated stars, etc.).
    max_cluster_size: int = HOTPIXEL_MAX_CLUSTER_SIZE
    # Side length (pixels, must be odd) of the median filter used for replacement.
    neighborhood_size: int = HOTPIXEL_NEIGHBORHOOD


@dataclass
class OutputOptions:
    save_intermediate_images: bool = False
    save_text_summary: bool = True


@dataclass
class PipelineConfig:
    images_dir: str
    output_dir: str
    logging_level: str = "INFO"
    image_calibration: bool = False
    calibration_dir: str = "calibration_frames"
    estimated_streak_length_enabled: bool = False
    norad_id: Optional[int] = None
    default_minlinelength: int = 35
    hotpixel_threshold: int = HOTPIXEL_THRESHOLD
    hotpixel_params: HotpixelParams = field(default_factory=HotpixelParams)
    enabled_line_filters: EnabledFilters = field(default_factory=EnabledFilters)
    background_detection_method: BackgroundMethod = field(default_factory=BackgroundMethod)
    background_params: BackgroundParams = field(default_factory=BackgroundParams)
    filter_params: FilterParams = field(default_factory=FilterParams)
    hough_params: HoughParams = field(default_factory=HoughParams)
    detection_method: DetectionMethod = field(default_factory=DetectionMethod)
    fft_detector_params: FftDetectorParams = field(default_factory=FftDetectorParams)
    peak_hough_params: PeakHoughParams = field(default_factory=PeakHoughParams)
    snr_params: SnrParams = field(default_factory=SnrParams)
    output_options: OutputOptions = field(default_factory=OutputOptions)
    tle_cache_ttl_hours: int = 24

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise ConfigError if any field combination is invalid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging_level.upper() not in valid_levels:
            raise ConfigError(
                f"logging_level must be one of {valid_levels}, got {self.logging_level!r}"
            )

        bg = self.background_detection_method
        enabled_count = sum([
            bg.simple_median,
            bg.gaussian_blur,
            bg.double_pass,
            bg.adaptive_local,
            bg.per_row_median_curve,
        ])
        if enabled_count > 1:
            raise ConfigError(
                "background_detection_method: exactly one method must be enabled, "
                f"but got {enabled_count} enabled"
            )

        dm = self.detection_method
        det_count = sum([dm.hough, dm.fft_correlation, dm.peak_hough])
        if det_count == 0:
            raise ConfigError("detection_method: at least one method must be enabled")
        if det_count > 1:
            raise ConfigError(
                "detection_method: exactly one method must be enabled, "
                f"but got {det_count} enabled"
            )

        if self.estimated_streak_length_enabled and self.norad_id is None:
            raise ConfigError(
                "estimated_streak_length_enabled=true requires norad_id to be set"
            )

        if self.hough_params.threshold < 1:
            raise ConfigError(
                f"hough_params.threshold must be >= 1, got {self.hough_params.threshold}"
            )
        php = self.peak_hough_params
        if php.median_bins < 2:
            raise ConfigError(
                f"peak_hough_params.median_bins must be >= 2, got {php.median_bins}"
            )
        if php.polynomial_degree < 0:
            raise ConfigError(
                "peak_hough_params.polynomial_degree must be >= 0, "
                f"got {php.polynomial_degree}"
            )
        if php.threshold_sigma <= 0:
            raise ConfigError(
                f"peak_hough_params.threshold_sigma must be > 0, got {php.threshold_sigma}"
            )
        if php.hough_threshold < 1:
            raise ConfigError(
                f"peak_hough_params.hough_threshold must be >= 1, got {php.hough_threshold}"
            )
        if php.dilation_kernel < 1:
            raise ConfigError(
                f"peak_hough_params.dilation_kernel must be >= 1, got {php.dilation_kernel}"
            )
            raise ConfigError(
                f"peak_hough_params.dilation_kernel must be >= 1, got {php.dilation_kernel}"
            )

        fp = self.filter_params
        if not (0.0 < fp.length_fraction <= 1.0):
            raise ConfigError(
                f"filter_params.length_fraction must be in (0, 1], got {fp.length_fraction}"
            )

        bp = self.background_params
        if bp.adaptive_local_tile_size < 8:
            raise ConfigError(
                f"background_params.adaptive_local_tile_size must be >= 8, got {bp.adaptive_local_tile_size}"
            )
        if bp.adaptive_local_snr_threshold <= 0:
            raise ConfigError(
                f"background_params.adaptive_local_snr_threshold must be > 0, got {bp.adaptive_local_snr_threshold}"
            )
        if bp.per_row_median_bins < 2:
            raise ConfigError(
                f"background_params.per_row_median_bins must be >= 2, got {bp.per_row_median_bins}"
            )
        if bp.per_row_median_degree < 0:
            raise ConfigError(
                f"background_params.per_row_median_degree must be >= 0, got {bp.per_row_median_degree}"
            )
        if bp.per_row_median_row_window < 1:
            raise ConfigError(
                f"background_params.per_row_median_row_window must be >= 1, got {bp.per_row_median_row_window}"
            )
        if bp.per_row_median_sigma_mult <= 0:
            raise ConfigError(
                f"background_params.per_row_median_sigma_mult must be > 0, got {bp.per_row_median_sigma_mult}"
            )
        if bp.per_row_median_filter_size < 1:
            raise ConfigError(
                f"background_params.per_row_median_filter_size must be >= 1, got {bp.per_row_median_filter_size}"
            )
        if bp.per_row_median_min_component_pixels < 1:
            raise ConfigError(
                "background_params.per_row_median_min_component_pixels must be >= 1, "
                f"got {bp.per_row_median_min_component_pixels}"
            )
        if bp.per_row_median_morph_kernel < 1:
            raise ConfigError(
                "background_params.per_row_median_morph_kernel must be >= 1, "
                f"got {bp.per_row_median_morph_kernel}"
            )

        sp = self.snr_params
        if sp.half_width_px < 1:
            raise ConfigError(
                f"snr_params.half_width_px must be >= 1, got {sp.half_width_px}"
            )
        if sp.off_gap_px < 0:
            raise ConfigError(
                f"snr_params.off_gap_px must be >= 0, got {sp.off_gap_px}"
            )
        if sp.off_width_px < 1:
            raise ConfigError(
                f"snr_params.off_width_px must be >= 1, got {sp.off_width_px}"
            )
        if sp.min_off_pixels < 1:
            raise ConfigError(
                f"snr_params.min_off_pixels must be >= 1, got {sp.min_off_pixels}"
            )

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """
        Load config from a JSON file and apply environment variable overrides.

        Backward-compatible: silently remaps legacy misspelled keys and emits
        a DeprecationWarning for each one found so callers can update their files.
        """
        path = Path(path)
        with open(path) as fh:
            raw: dict = json.load(fh)

        base_dir = path.parent
        raw = _remap_keys(raw)

        cfg = cls(
            images_dir=_resolve_path(raw.get("images_dir", "images"), base_dir),
            output_dir=_resolve_path(raw.get("output_dir", "output"), base_dir),
            logging_level=raw.get("logging_level", "INFO"),
            image_calibration=raw.get("image_calibration", False),
            calibration_dir=_resolve_path(
                raw.get("calibration_dir", "calibration_frames"), base_dir
            ),
            estimated_streak_length_enabled=raw.get(
                "estimated_streak_length_enabled", False
            ),
            norad_id=raw.get("norad_id"),
            default_minlinelength=raw.get("default_minlinelength", 25),
            hotpixel_threshold=raw.get("hotpixel_threshold", HOTPIXEL_THRESHOLD),
            hotpixel_params=HotpixelParams(
                **{k: v for k, v in raw.get("hotpixel_params", {}).items()
                   if k in ("threshold_sigma", "max_cluster_size", "neighborhood_size")}
            ),
            enabled_line_filters=EnabledFilters.from_dict(
                raw.get("enabled_line_filters", {})
            ),
            background_detection_method=BackgroundMethod.from_dict(
                raw.get("background_detection_method", {})
            ),
            background_params=BackgroundParams.from_dict(
                raw.get("background_params", {})
            ),
            filter_params=FilterParams(
                **{k: v for k, v in raw.get("filter_params", {}).items()
                   if k in FilterParams.__dataclass_fields__}
            ),
            hough_params=HoughParams(),
            detection_method=DetectionMethod.from_dict(
                raw.get("detection_method", {})
            ),
            fft_detector_params=FftDetectorParams(),
            peak_hough_params=PeakHoughParams.from_dict(
                raw.get("peak_hough_params", {})
            ),
            output_options=OutputOptions(
                save_intermediate_images=raw.get("save_intermediate_images", False),
                save_text_summary=raw.get("save_text_summary", True),
            ),
            tle_cache_ttl_hours=raw.get("tle_cache_ttl_hours", 24),
        )

        cfg = _apply_env_overrides(cfg)
        return cfg


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _remap_keys(d: dict) -> dict:
    """Return a copy of *d* with legacy misspelled keys renamed."""
    result = {}
    for k, v in d.items():
        if k in _COMPAT_KEY_MAP:
            new_key = _COMPAT_KEY_MAP[k]
            warnings.warn(
                f"Config key {k!r} is deprecated; use {new_key!r} instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            result[new_key] = v
        else:
            result[k] = v
    return result


def _resolve_path(raw: str, base: Path) -> str:
    """Resolve a possibly-relative path against the config file's parent dir."""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((base / p).resolve())


def _apply_env_overrides(cfg: PipelineConfig) -> PipelineConfig:
    """Apply STREAKILLER_* environment variables on top of the loaded config."""
    overrides: dict[str, object] = {}
    env_map = {
        "STREAKILLER_IMAGES_DIR": "images_dir",
        "STREAKILLER_OUTPUT_DIR": "output_dir",
        "STREAKILLER_LOGGING_LEVEL": "logging_level",
        "STREAKILLER_NORAD_ID": "norad_id",
        "STREAKILLER_TLE_CACHE_TTL_HOURS": "tle_cache_ttl_hours",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            overrides[attr] = val

    if not overrides:
        return cfg

    # Convert numeric types
    if "norad_id" in overrides:
        overrides["norad_id"] = int(overrides["norad_id"])  # type: ignore[arg-type]
    if "tle_cache_ttl_hours" in overrides:
        overrides["tle_cache_ttl_hours"] = int(overrides["tle_cache_ttl_hours"])  # type: ignore[arg-type]

    for attr, val in overrides.items():
        object.__setattr__(cfg, attr, val)

    return cfg
