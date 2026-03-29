import requests
import logging
import time
from datetime import datetime, timezone
from skyfield.api import load, EarthSatellite
from skyfield.framelib import ICRS
from skyfield.api import wgs84
logging.basicConfig(level=logging.INFO)

TS = load.timescale()
PLANETS = load('de421.bsp')
EARTH = PLANETS['earth']



def download_tle(NORAD_id: int) -> str | None:
    tle_url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={NORAD_id}"

    for connection_attempt in range(5):
        response = requests.get(tle_url, timeout=15)
        if response.status_code != 200:
            logging.error(f"Failed to download TLE data: try {connection_attempt}")
            time.sleep(5 * connection_attempt)
            continue
        if "No GP data found" in response.text:
            logging.error("Failed to download TLE data: satellite not in catalog. Ensure the name is correct.")
            return None
        if response.text.count('\n') != 3:
            logging.error("Failed to download TLE data: multiple satellites match the specified name.")
            return None
        logging.info("TLE data downloaded successfully:")
        logging.info(response.text)
        return response.text

    logging.error("Failed to download TLE data: too many connection attempts.")
    return None



def build_satellite(NORAD_id, lat, lon, elevation):
    tle_data = download_tle(NORAD_id)
    if tle_data is None:
        raise ValueError("Could not retrieve TLE data.")
    lines = tle_data.strip().split('\n')
    if len(lines) < 3:
        raise ValueError("Incomplete TLE data.")
    name, line1, line2 = lines
    satellite = EarthSatellite(line1, line2, name)
    OBSERVER = wgs84.latlon(lat, lon, elevation)
    return satellite, OBSERVER



def get_ra_dec_rates(sat: EarthSatellite, OBSERVER, date_obs):
    dt = datetime.fromisoformat(date_obs).replace(tzinfo=timezone.utc)
    if date_obs is not None:
        t = TS.from_datetime(dt)
    else:
        t = TS.now()
        logging.warning("DATE-OBS not found in FITS header, using current time which may lead to inaccurate results.")

    observe_sat = (EARTH + OBSERVER).at(t).observe(EARTH + sat).apparent()
    dec, ra, dist, dec_rate, ra_rate, dist_rate = observe_sat.frame_latlon_and_rates(ICRS)
    
    return ra_rate.arcseconds.per_second, dec_rate.arcseconds.per_second, dec.degrees


if __name__ == "__main__":
    download_tle(25544)  # Example NORAD ID for the ISS
    sat, OBSERVER = build_satellite(25544)
    ra_rate, dec_rate = get_ra_dec_rates(sat, OBSERVER)
    print(f"RA rate: {ra_rate} arcsec/sec, Dec rate: {dec_rate} arcsec/sec")