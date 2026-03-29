"""
TLE cache — downloads TLE data from CelesTrak and caches it to disk.

This avoids re-downloading the same TLE on every pipeline run.  The cache
is keyed by NORAD ID and invalidated when older than *ttl_hours*.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}"
_MAX_RETRIES = 5
_RETRY_BASE_SECONDS = 5
_REQUEST_TIMEOUT = 15


class TleFetchError(OSError):
    """Raised when TLE data cannot be retrieved."""


class TleCache:
    """
    Disk-backed TLE cache with configurable TTL.

    Cache files are stored as ``<cache_dir>/<norad_id>.json`` containing the
    raw TLE text and a timestamp.
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 24) -> None:
        self._dir = Path(cache_dir)
        self._ttl = timedelta(hours=ttl_hours)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_or_refresh(self, norad_id: int) -> str:
        """
        Return TLE text for *norad_id*, downloading if the cache is stale.

        Raises TleFetchError on network failure or invalid response.
        """
        cached = self.get(norad_id)
        if cached is not None:
            logger.debug("TLE cache hit for NORAD %d", norad_id)
            return cached

        logger.info("TLE cache miss for NORAD %d — downloading", norad_id)
        tle_text = self._download(norad_id)
        self.put(norad_id, tle_text)
        return tle_text

    def get(self, norad_id: int) -> Optional[str]:
        """Return cached TLE text if it exists and is still fresh, else None."""
        path = self._cache_path(norad_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(record["cached_at"])
            if datetime.now(tz=timezone.utc) - cached_at > self._ttl:
                logger.debug("TLE cache expired for NORAD %d", norad_id)
                return None
            return record["tle_text"]
        except Exception as exc:
            logger.warning("Could not read TLE cache for NORAD %d: %s", norad_id, exc)
            return None

    def put(self, norad_id: int, tle_text: str) -> None:
        """Write TLE text to cache with a current timestamp."""
        record = {
            "norad_id": norad_id,
            "tle_text": tle_text,
            "cached_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._cache_path(norad_id).write_text(json.dumps(record))
        logger.debug("TLE cached for NORAD %d", norad_id)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _cache_path(self, norad_id: int) -> Path:
        return self._dir / f"{norad_id}.json"

    def _download(self, norad_id: int) -> str:
        url = _CELESTRAK_URL.format(norad_id=norad_id)
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = requests.get(url, timeout=_REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("TLE download attempt %d failed: %s", attempt, exc)
                time.sleep(_RETRY_BASE_SECONDS * attempt)
                continue

            if response.status_code != 200:
                logger.warning(
                    "TLE download attempt %d returned HTTP %d", attempt, response.status_code
                )
                time.sleep(_RETRY_BASE_SECONDS * attempt)
                continue

            text = response.text
            if "No GP data found" in text:
                raise TleFetchError(
                    f"NORAD {norad_id} not found in CelesTrak catalog"
                )
            if text.count("\n") != 3:
                raise TleFetchError(
                    f"Unexpected TLE format for NORAD {norad_id} "
                    f"(expected 3 lines, got {text.count(chr(10))})"
                )

            return text

        raise TleFetchError(
            f"Failed to download TLE for NORAD {norad_id} after {_MAX_RETRIES} attempts"
        ) from last_exc
