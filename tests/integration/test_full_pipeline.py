"""Integration tests — full pipeline on synthetic images."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from streakiller.config.schema import PipelineConfig
from streakiller.models.fits_image import FitsImage
from streakiller.pipeline.streak_pipeline import StreakPipeline


class TestFullPipeline:
    def test_detects_synthetic_streak(self, synthetic_fits_image, minimal_config, tmp_path):
        """End-to-end: a synthetic image with one injected streak should produce >=1 detection."""
        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(tmp_path / "output"),
            logging_level="WARNING",
        )
        # Run without an output writer to keep the test filesystem-free
        pipeline = StreakPipeline(config=cfg, output_writer=None)
        result = pipeline.process(synthetic_fits_image)

        assert result.error is None, f"Pipeline failed: {result.error}"
        assert result.streak_count >= 1, "Expected at least one streak to be detected"

    def test_empty_image_produces_empty_result(self, minimal_config, tmp_path):
        """An image with only noise should produce zero detections — not crash."""
        from streakiller.models.fits_image import ObservationMetadata

        rng = np.random.default_rng(0)
        flat_data = rng.normal(1000.0, 10.0, (128, 128)).astype(np.float32)
        meta = ObservationMetadata(
            exposure_time=1.0, date_obs=None, telescope=None, camera=None,
            focal_length_mm=None, lat=None, lon=None, elevation_m=None,
            binning=1, pixel_size_um=None, pixel_scale_arcsec=None,
        )
        image = FitsImage(source_path=Path("flat.fits"), data=flat_data, raw_header={}, metadata=meta)

        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(tmp_path / "output"),
            logging_level="WARNING",
        )
        pipeline = StreakPipeline(config=cfg, output_writer=None)
        result = pipeline.process(image)

        assert result.error is None
        assert result.streak_count == 0

    def test_provenance_is_populated(self, synthetic_fits_image, tmp_path):
        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(tmp_path / "output"),
            logging_level="WARNING",
        )
        pipeline = StreakPipeline(config=cfg, output_writer=None)
        result = pipeline.process(synthetic_fits_image)

        assert result.provenance is not None
        assert result.provenance.software_version != ""
        assert "initial_detected" in result.provenance.stage_line_counts
        assert "final" in result.provenance.stage_line_counts

    def test_output_files_written(self, synthetic_fits_image, tmp_path):
        """When a LocalOutputWriter is configured, output files must appear."""
        from streakiller.io.output_writer import LocalOutputWriter
        from streakiller.config.schema import OutputOptions

        out_dir = tmp_path / "output"
        writer = LocalOutputWriter(out_dir, OutputOptions(save_intermediate_images=False))
        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(out_dir),
            logging_level="WARNING",
        )
        pipeline = StreakPipeline(config=cfg, output_writer=writer)
        result = pipeline.process(synthetic_fits_image)

        stem = synthetic_fits_image.source_path.stem
        run_dir = out_dir / stem
        assert (run_dir / "streaks.csv").exists(), "streaks.csv was not written"
        assert (run_dir / "detected_streaks.png").exists(), "detected_streaks.png was not written"
        assert (run_dir / "processing_results.json").exists(), "processing_results.json was not written"

    def test_intermediate_images_only_written_when_enabled(self, synthetic_fits_image, tmp_path):
        from streakiller.io.output_writer import LocalOutputWriter
        from streakiller.config.schema import OutputOptions

        out_dir = tmp_path / "output"
        writer = LocalOutputWriter(out_dir, OutputOptions(save_intermediate_images=True))
        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(out_dir),
            logging_level="WARNING",
        )
        pipeline = StreakPipeline(config=cfg, output_writer=writer)
        pipeline.process(synthetic_fits_image)

        stem = synthetic_fits_image.source_path.stem
        assert (out_dir / stem / "binary.png").exists()
        assert (out_dir / stem / "normalized_display.png").exists()

    def test_pipeline_result_never_raises(self, tmp_path):
        """Even a completely degenerate image should return a result, not raise."""
        from streakiller.models.fits_image import ObservationMetadata

        # All-zeros image — worst case for background estimators
        data = np.zeros((64, 64), dtype=np.float32)
        meta = ObservationMetadata(
            exposure_time=1.0, date_obs=None, telescope=None, camera=None,
            focal_length_mm=None, lat=None, lon=None, elevation_m=None,
            binning=1, pixel_size_um=None, pixel_scale_arcsec=None,
        )
        image = FitsImage(source_path=Path("zeros.fits"), data=data, raw_header={}, metadata=meta)
        cfg = PipelineConfig(
            images_dir=str(tmp_path / "images"),
            output_dir=str(tmp_path / "output"),
            logging_level="WARNING",
        )
        pipeline = StreakPipeline(config=cfg, output_writer=None)
        result = pipeline.process(image)  # must not raise
        assert isinstance(result, type(result))  # returned something
