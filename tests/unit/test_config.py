"""Unit tests for PipelineConfig."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from streakiller.config.schema import (
    BackgroundMethod,
    ConfigError,
    EnabledFilters,
    FilterParams,
    HoughParams,
    PipelineConfig,
)


# ------------------------------------------------------------------ #
# Validation tests                                                     #
# ------------------------------------------------------------------ #

class TestValidation:
    def test_rejects_invalid_logging_level(self):
        cfg = PipelineConfig(images_dir=".", output_dir=".", logging_level="VERBOSE")
        with pytest.raises(ConfigError, match="logging_level"):
            cfg.validate()

    def test_accepts_all_valid_logging_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = PipelineConfig(images_dir=".", output_dir=".", logging_level=level)
            cfg.validate()  # must not raise

    def test_rejects_multiple_background_methods(self):
        cfg = PipelineConfig(
            images_dir=".", output_dir=".",
            background_detection_method=BackgroundMethod(
                simple_median=True, gaussian_blur=True, double_pass=False
            ),
        )
        with pytest.raises(ConfigError, match="background_detection_method"):
            cfg.validate()

    def test_rejects_satellite_mode_without_norad_id(self):
        cfg = PipelineConfig(
            images_dir=".", output_dir=".",
            estimated_streak_length_enabled=True,
            norad_id=None,
        )
        with pytest.raises(ConfigError, match="norad_id"):
            cfg.validate()

    def test_accepts_satellite_mode_with_norad_id(self):
        cfg = PipelineConfig(
            images_dir=".", output_dir=".",
            estimated_streak_length_enabled=True,
            norad_id=25544,
        )
        cfg.validate()  # must not raise

    def test_rejects_hough_threshold_below_one(self):
        cfg = PipelineConfig(
            images_dir=".", output_dir=".",
            hough_params=HoughParams(threshold=0),
        )
        with pytest.raises(ConfigError, match="threshold"):
            cfg.validate()

    def test_rejects_length_fraction_out_of_range(self):
        for bad in (0.0, -0.1, 1.1):
            cfg = PipelineConfig(
                images_dir=".", output_dir=".",
                filter_params=FilterParams(length_fraction=bad),
            )
            with pytest.raises(ConfigError, match="length_fraction"):
                cfg.validate()

    def test_accepts_length_fraction_boundary(self):
        cfg = PipelineConfig(
            images_dir=".", output_dir=".",
            filter_params=FilterParams(length_fraction=1.0),
        )
        cfg.validate()  # must not raise


# ------------------------------------------------------------------ #
# from_json tests                                                      #
# ------------------------------------------------------------------ #

class TestFromJson:
    def test_loads_valid_config(self, sample_config_json: Path):
        cfg = PipelineConfig.from_json(sample_config_json)
        assert cfg.logging_level == "WARNING"
        assert cfg.default_minlinelength == 25
        assert cfg.background_detection_method.gaussian_blur is True
        assert cfg.enabled_line_filters.midpoint_filter is True

    def test_resolves_relative_paths(self, sample_config_json: Path):
        cfg = PipelineConfig.from_json(sample_config_json)
        # Paths should be absolute after loading
        assert Path(cfg.images_dir).is_absolute()
        assert Path(cfg.output_dir).is_absolute()

    def test_backward_compat_cailbration_dir_typo(self, tmp_path: Path):
        cfg_data = {
            "images_dir": "images",
            "output_dir": "output",
            "cailbration_dir": "my_frames",  # legacy misspelling
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg_data))

        with pytest.warns(DeprecationWarning, match="cailbration_dir"):
            cfg = PipelineConfig.from_json(p)

        assert "my_frames" in cfg.calibration_dir

    def test_backward_compat_endpoint_filer_typo(self, tmp_path: Path):
        cfg_data = {
            "images_dir": ".",
            "output_dir": ".",
            "enabled_line_filters": {
                "endpoint_filer": False,  # legacy misspelling
            },
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg_data))

        with pytest.warns(DeprecationWarning, match="endpoint_filer"):
            cfg = PipelineConfig.from_json(p)

        assert cfg.enabled_line_filters.endpoint_filter is False

    def test_backward_compat_guassian_blur_typo(self, tmp_path: Path):
        cfg_data = {
            "images_dir": ".",
            "output_dir": ".",
            "background_detection_method": {
                "Guassian_blur": True,   # legacy misspelling
                "simple_median": False,
                "doublepass_median_to_guassian_blur": False,
            },
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg_data))

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = PipelineConfig.from_json(p)

        assert cfg.background_detection_method.gaussian_blur is True

    def test_env_override_output_dir(self, sample_config_json: Path, monkeypatch, tmp_path):
        custom = str(tmp_path / "custom_out")
        monkeypatch.setenv("STREAKILLER_OUTPUT_DIR", custom)
        cfg = PipelineConfig.from_json(sample_config_json)
        assert cfg.output_dir == custom

    def test_env_override_norad_id(self, sample_config_json: Path, monkeypatch):
        monkeypatch.setenv("STREAKILLER_NORAD_ID", "99999")
        cfg = PipelineConfig.from_json(sample_config_json)
        assert cfg.norad_id == 99999

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_json(tmp_path / "nonexistent.json")


# ------------------------------------------------------------------ #
# EnabledFilters / BackgroundMethod helper tests                      #
# ------------------------------------------------------------------ #

class TestEnabledFilters:
    def test_from_dict_defaults(self):
        f = EnabledFilters.from_dict({})
        assert f.midpoint_filter is True
        assert f.endpoint_filter is True
        assert f.colinear_filter is False

    def test_from_dict_override(self):
        f = EnabledFilters.from_dict({"midpoint_filter": False, "length_filter": False})
        assert f.midpoint_filter is False
        assert f.length_filter is False
        assert f.line_angle is True  # default unchanged


class TestBackgroundMethod:
    def test_active_name_gaussian(self):
        bg = BackgroundMethod(simple_median=False, gaussian_blur=True, double_pass=False)
        assert bg.active_name() == "gaussian_blur"

    def test_active_name_simple_median(self):
        bg = BackgroundMethod(simple_median=True, gaussian_blur=False, double_pass=False)
        assert bg.active_name() == "simple_median"

    def test_active_name_double_pass(self):
        bg = BackgroundMethod(simple_median=False, gaussian_blur=False, double_pass=True)
        assert bg.active_name() == "double_pass"
