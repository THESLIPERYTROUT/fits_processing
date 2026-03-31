"""
StreakPipeline — the main orchestrator.

Assembles all stages and runs them in order for a single FitsImage.
The process() method is stateless: calling it twice with the same input
produces the same PipelineResult.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from streakiller import __version__
from streakiller.config.schema import PipelineConfig
from streakiller.models.fits_image import FitsImage
from streakiller.models.result import PipelineResult, Provenance

logger = logging.getLogger(__name__)


class StreakPipeline:
    """
    Full streak-detection pipeline for one FITS image.

    All dependencies are injected via the constructor so that tests can
    substitute lightweight fakes without touching the filesystem.

    Typical usage::

        pipeline = StreakPipeline.from_config(config)
        result = pipeline.process(image)
    """

    def __init__(
        self,
        config: PipelineConfig,
        calibration_step=None,
        background_estimator=None,
        streak_estimator=None,
        output_writer=None,
    ) -> None:
        from streakiller.background import GaussianBlurEstimator, SimpleMedianEstimator, DoublePassEstimator
        from streakiller.detection.detector import StreakDetector
        from streakiller.filters.chain import FilterChain

        self._config = config
        self._calibration = calibration_step
        self._streak_estimator = streak_estimator

        if background_estimator is not None:
            self._background = background_estimator
        else:
            method = config.background_detection_method
            if method.simple_median:
                self._background = SimpleMedianEstimator()
            elif method.double_pass:
                self._background = DoublePassEstimator()
            else:
                self._background = GaussianBlurEstimator()

        self._detector = StreakDetector(config.hough_params)
        self._filter_chain = FilterChain.from_config(config.enabled_line_filters)
        self._writer = output_writer

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "StreakPipeline":
        """Convenience constructor — wires all real dependencies from config."""
        from streakiller.io.output_writer import LocalOutputWriter

        writer = LocalOutputWriter(Path(config.output_dir), config.output_options)
        return cls(config=config, output_writer=writer)

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def process(self, image: FitsImage) -> PipelineResult:
        """
        Run the full pipeline on one image.

        Stages
        ------
        1. Calibration (optional)
        2. Hot-pixel removal (if calibration disabled)
        3. Streak length estimation via TLE (optional)
        4. Background estimation → binary mask
        5. Hough line detection
        6. Filter chain
        7. Write outputs (if writer is set)

        Returns a PipelineResult. Never raises — errors are captured into
        result.error so batch processing can continue.
        """
        start_utc = datetime.now(tz=timezone.utc).isoformat()
        img_log = logging.LoggerAdapter(logger, {"source_file": str(image.source_path)})
        img_log.info("Pipeline started")

        try:
            return self._run(image, start_utc, img_log)
        except Exception as exc:
            img_log.error("Pipeline failed: %s", exc, exc_info=True)
            return PipelineResult(
                source_path=image.source_path,
                initial_detected_lines=np.empty((0, 1, 4), dtype=np.int32),
                detected_lines=np.empty((0, 1, 4), dtype=np.int32),
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _run(self, image: FitsImage, start_utc: str, img_log) -> PipelineResult:
        cfg = self._config

        # 1 & 2: Calibration or hot-pixel removal
        image = self._prepare_image(image, img_log)

        # 3: Determine minimum line length
        min_line_length = self._resolve_min_line_length(image, img_log)

        # 4: Background estimation
        img_log.info("Background estimation: %s", cfg.background_detection_method.active_name())
        binary = self._background.estimate(image.data, cfg.background_params)

        # 5: Hough detection
        detection = self._detector.detect(binary, image.data, min_line_length)
        img_log.info("Detected %d raw lines", len(detection.lines))

        # 6: Filter chain
        final_lines, snapshots = self._filter_chain.run(detection.lines, cfg.filter_params)
        img_log.info("Final streak count: %d", len(final_lines))

        end_utc = datetime.now(tz=timezone.utc).isoformat()

        stage_counts: dict[str, int] = {"initial_detected": len(detection.lines)}
        for snap in snapshots:
            stage_counts[snap.stage_name] = snap.lines_after
        stage_counts["final"] = len(final_lines)

        provenance = Provenance(
            software_version=__version__,
            config_snapshot=self._config_snapshot(),
            processing_start_utc=start_utc,
            processing_end_utc=end_utc,
            background_method_used=cfg.background_detection_method.active_name(),
            min_line_length_used=min_line_length,
            hough_threshold_used=cfg.hough_params.threshold,
            stage_line_counts=stage_counts,
        )

        result = PipelineResult(
            source_path=image.source_path,
            initial_detected_lines=detection.lines.copy(),
            detected_lines=final_lines,
            filter_snapshots=snapshots,
            normalized_display=detection.normalized_display,
            binary_image=detection.binary_image,
            provenance=provenance,
        )

        # 7: Write outputs
        if self._writer is not None:
            self._writer.write(result)

        return result

    def _prepare_image(self, image: FitsImage, img_log) -> FitsImage:
        cfg = self._config
        if cfg.image_calibration and self._calibration is not None:
            img_log.info("Applying calibration")
            return self._calibration.apply(image)
        else:
            img_log.info("Applying hot-pixel removal (threshold=%d)", cfg.hotpixel_threshold)
            return image.derive(self._hotpixel_removal(image.data, cfg.hotpixel_threshold))

    def _resolve_min_line_length(self, image: FitsImage, img_log) -> float:
        cfg = self._config
        if (
            cfg.estimated_streak_length_enabled
            and cfg.norad_id is not None
            and self._streak_estimator is not None
            and image.metadata.pixel_scale_arcsec is not None
            and image.metadata.exposure_time is not None
            and image.metadata.has_location
            and image.metadata.date_obs is not None
        ):
            try:
                length = self._streak_estimator.estimate(
                    norad_id=cfg.norad_id,
                    exposure_time=image.metadata.exposure_time,
                    pixel_scale_arcsec=image.metadata.pixel_scale_arcsec,
                    lat=image.metadata.lat,
                    lon=image.metadata.lon,
                    elevation_m=image.metadata.elevation_m,
                    date_obs=image.metadata.date_obs,
                )
                img_log.info("TLE-estimated min line length: %.1f px", length)
                return length
            except Exception as exc:
                img_log.warning(
                    "TLE estimation failed (%s), falling back to default minlinelength", exc
                )

        length = float(cfg.default_minlinelength)
        img_log.info("Using default min line length: %.1f px", length)
        return length

    @staticmethod
    def _hotpixel_removal(data: np.ndarray, threshold: float) -> np.ndarray:
        cleaned = data.copy()
        hot_y, hot_x = np.where(cleaned > threshold)
        for y, x in zip(hot_y, hot_x):
            y_min, y_max = max(y - 1, 0), min(y + 2, cleaned.shape[0])
            x_min, x_max = max(x - 1, 0), min(x + 2, cleaned.shape[1])
            cleaned[y, x] = np.median(cleaned[y_min:y_max, x_min:x_max])
        if len(hot_y):
            logger.debug("Hot-pixel removal: %d pixels replaced", len(hot_y))
        return cleaned

    def _config_snapshot(self) -> dict:
        import dataclasses
        cfg = self._config
        try:
            return dataclasses.asdict(cfg)
        except Exception:
            return {"error": "could not serialise config"}
