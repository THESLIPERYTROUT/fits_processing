"""
StreakLengthEstimator — uses TLE data to estimate the expected minimum streak
length for a given satellite, observer location, and exposure time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StreakLengthEstimator:
    """
    Estimates the minimum Hough line length for a satellite streak.

    Downloads (or retrieves from cache) the TLE for a given NORAD ID, builds
    an EarthSatellite object, computes the angular velocity at the observation
    time, and converts it to a pixel length using the image's pixel scale and
    exposure time.
    """

    def __init__(self, tle_cache=None) -> None:
        """
        Parameters
        ----------
        tle_cache : TleCache, optional
            If None a cache in the system temp directory is used.
        """
        if tle_cache is None:
            import tempfile
            from streakiller.io.tle_cache import TleCache
            tle_cache = TleCache(Path(tempfile.gettempdir()) / "streakiller_tle_cache")
        self._cache = tle_cache

    def estimate(
        self,
        norad_id: int,
        exposure_time: float,
        pixel_scale_arcsec: float,
        lat: float,
        lon: float,
        elevation_m: float,
        date_obs: str,
        safety_factor: float = 0.7,
    ) -> float:
        """
        Estimate minimum streak length in pixels.

        Returns safety_factor * (angular_velocity * exposure_time / pixel_scale).

        Raises ValueError if TLE data cannot be retrieved or location is missing.
        """
        from skyfield.api import load, EarthSatellite, wgs84
        from skyfield.framelib import ICRS
        from datetime import datetime, timezone

        tle_text = self._cache.fetch_or_refresh(norad_id)
        lines = tle_text.strip().split("\n")
        if len(lines) < 3:
            raise ValueError(f"Incomplete TLE data for NORAD {norad_id}")

        name, line1, line2 = lines[0], lines[1], lines[2]
        ts = load.timescale()
        satellite = EarthSatellite(line1, line2, name)
        planets = load("de421.bsp")
        earth = planets["earth"]
        observer = wgs84.latlon(lat, lon, elevation_m)

        dt = datetime.fromisoformat(date_obs).replace(tzinfo=timezone.utc)
        t = ts.from_datetime(dt)

        observation = (earth + observer).at(t).observe(earth + satellite).apparent()
        dec, ra, _dist, dec_rate, ra_rate, _dist_rate = observation.frame_latlon_and_rates(ICRS)

        angular_velocity = float(
            np.hypot(
                dec_rate.arcseconds.per_second,
                ra_rate.arcseconds.per_second * np.cos(np.radians(dec.degrees)),
            )
        )
        max_streak_px = (angular_velocity * exposure_time) / pixel_scale_arcsec
        min_line_length = safety_factor * max_streak_px

        logger.info(
            "NORAD %d: angular_velocity=%.2f arcsec/s  streak=%.1f px  "
            "minLineLength=%.1f px",
            norad_id, angular_velocity, max_streak_px, min_line_length,
        )
        return float(min_line_length)
