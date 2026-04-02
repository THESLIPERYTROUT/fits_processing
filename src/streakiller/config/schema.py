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
    LENGTH_FRACTION,
    COLINEAR_ORIENTATION_TOL,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA_LADDER,
    SIMPLE_MEDIAN_SIGMA_MULT,
    DOUBLE_PASS_SIGMA_MULT,
    DOUBLE_PASS_INPAINT_RADIUS,
    HOTPIXEL_THRESHOLD,
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
class FilterParams:
    midpoint_min_distance: float = MIDPOINT_MIN_DISTANCE
    endpoint_min_distance: float = ENDPOINT_MIN_DISTANCE
    angle_min_diff_deg: float = ANGLE_MIN_DIFF_DEG
    length_fraction: float = LENGTH_FRACTION
    colinear_orientation_tol: float = COLINEAR_ORIENTATION_TOL


@dataclass
class BackgroundParams:
    gaussian_kernel_size: int = GAUSSIAN_KERNEL_SIZE
    gaussian_sigma_ladder: tuple = field(default_factory=lambda: GAUSSIAN_SIGMA_LADDER)
    simple_median_sigma_mult: float = SIMPLE_MEDIAN_SIGMA_MULT
    double_pass_sigma_mult: float = DOUBLE_PASS_SIGMA_MULT
    double_pass_inpaint_radius: int = DOUBLE_PASS_INPAINT_RADIUS


@dataclass
class EnabledFilters:
    midpoint_filter: bool = True
    line_angle: bool = True
    colinear_filter: bool = False
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
            endpoint_filter=remapped.get("endpoint_filter", True),
            length_filter=remapped.get("length_filter", True),
        )


@dataclass
class BackgroundMethod:
    simple_median: bool = False
    gaussian_blur: bool = True
    double_pass: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "BackgroundMethod":
        remapped = _remap_keys(raw)
        return cls(
            simple_median=remapped.get("simple_median", False),
            gaussian_blur=remapped.get("gaussian_blur", True),
            double_pass=remapped.get("double_pass", False),
        )

    def active_name(self) -> str:
        if self.simple_median:
            return "simple_median"
        if self.gaussian_blur:
            return "gaussian_blur"
        if self.double_pass:
            return "double_pass"
        return "gaussian_blur"  # fallback, validate() will catch multiple-enabled


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
    default_minlinelength: int = 25
    hotpixel_threshold: int = HOTPIXEL_THRESHOLD
    enabled_line_filters: EnabledFilters = field(default_factory=EnabledFilters)
    background_detection_method: BackgroundMethod = field(default_factory=BackgroundMethod)
    background_params: BackgroundParams = field(default_factory=BackgroundParams)
    filter_params: FilterParams = field(default_factory=FilterParams)
    hough_params: HoughParams = field(default_factory=HoughParams)
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
        enabled_count = sum([bg.simple_median, bg.gaussian_blur, bg.double_pass])
        if enabled_count > 1:
            raise ConfigError(
                "background_detection_method: exactly one method must be enabled, "
                f"but got {enabled_count} enabled"
            )

        if self.estimated_streak_length_enabled and self.norad_id is None:
            raise ConfigError(
                "estimated_streak_length_enabled=true requires norad_id to be set"
            )

        if self.hough_params.threshold < 1:
            raise ConfigError(
                f"hough_params.threshold must be >= 1, got {self.hough_params.threshold}"
            )

        fp = self.filter_params
        if not (0.0 < fp.length_fraction <= 1.0):
            raise ConfigError(
                f"filter_params.length_fraction must be in (0, 1], got {fp.length_fraction}"
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
            enabled_line_filters=EnabledFilters.from_dict(
                raw.get("enabled_line_filters", {})
            ),
            background_detection_method=BackgroundMethod.from_dict(
                raw.get("background_detection_method", {})
            ),
            background_params=BackgroundParams(),
            filter_params=FilterParams(),
            hough_params=HoughParams(),
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
