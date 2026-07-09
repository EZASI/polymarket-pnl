"""
ZOMBIE HUNTER — Settlement Arbitrage Engine
============================================
Finds Polymarket weather markets where:
  1. The event day has ENDED (sun has set, high temp is locked)
  2. The market is still ACTIVE (oracle hasn't resolved yet)
  3. Shares trade at 95-99¢ discount (impatient traders selling early)

Strategy: Buy at 98¢, wait for resolution, collect $1.00 → 2% risk-free yield.

Ground truth: NWS observation history (weather.gov) + METAR final readings.
"""

import json
import time
import math
import logging
import requests
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

log = logging.getLogger("zombie_hunter")

# ── Multi-source weather cache ──
_MULTI_WX_CACHE = {}
_MULTI_WX_TTL = 900  # 15 min

# ── Polymarket Gamma API ──
GAMMA_HOST = "https://gamma-api.polymarket.com"

# ── City configs with timezone offsets and sunset approximation ──
ZOMBIE_CITIES = {
    "new-york-city": {
        "name": "New York City",
        "slug": "new-york-city",
        "tz_offset": -5,
        "lat": 40.71, "lon": -74.01,
        "metar": "KNYC",
        "nws_station": "KNYC",
    },
    "chicago": {
        "name": "Chicago",
        "slug": "chicago",
        "tz_offset": -6,
        "lat": 41.88, "lon": -87.63,
        "metar": "KORD",
        "nws_station": "KORD",
    },
    "miami": {
        "name": "Miami",
        "slug": "miami",
        "tz_offset": -5,
        "lat": 25.76, "lon": -80.19,
        "metar": "KMIA",
        "nws_station": "KMIA",
    },
    "seattle": {
        "name": "Seattle",
        "slug": "seattle",
        "tz_offset": -8,
        "lat": 47.61, "lon": -122.33,
        "metar": "KSEA",
        "nws_station": "KSEA",
    },
    "dallas": {
        "name": "Dallas",
        "slug": "dallas",
        "tz_offset": -6,
        "lat": 32.78, "lon": -96.80,
        "metar": "KDFW",
        "nws_station": "KDFW",
    },
    "atlanta": {
        "name": "Atlanta",
        "slug": "atlanta",
        "tz_offset": -5,
        "lat": 33.75, "lon": -84.39,
        "metar": "KATL",
        "nws_station": "KATL",
    },
    "london": {
        "name": "London",
        "slug": "london",
        "tz_offset": 0,
        "lat": 51.51, "lon": -0.13,
        "metar": "EGLL",
        "metar_alt": ["EGLC", "EGLL"],  # EGLC=London City (closer to center), EGLL=Heathrow
        "nws_station": None,
    },
    "toronto": {
        "name": "Toronto",
        "slug": "toronto",
        "tz_offset": -5,
        "lat": 43.65, "lon": -79.38,
        "metar": "CYYZ",
        "nws_station": None,
    },
    "buenos-aires": {
        "name": "Buenos Aires",
        "slug": "buenos-aires",
        "tz_offset": -3,
        "lat": -34.60, "lon": -58.38,
        "metar": "SAEZ",
        "nws_station": None,
    },
    "seoul": {
        "name": "Seoul",
        "slug": "seoul",
        "tz_offset": 9,
        "lat": 37.57, "lon": 126.98,
        "metar": "RKSI",
        "nws_station": None,
    },
    "ankara": {
        "name": "Ankara",
        "slug": "ankara",
        "tz_offset": 3,
        "lat": 39.93, "lon": 32.86,
        "metar": "LTAC",
        "nws_station": None,
    },
    "wellington": {
        "name": "Wellington",
        "slug": "wellington",
        "tz_offset": 13,
        "lat": -41.29, "lon": 174.78,
        "metar": "NZWN",
        "nws_station": None,
    },
}

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


def get_sun_status(lat, lon, tz_offset):
    """Determine whether the sun has set for weather max-temp purposes.

    IMPORTANT: For weather markets, "sun has set" means we're past the
    afternoon sunset, so the daily max temperature is locked in.
    This is NOT the same as "sun is below horizon" which is also true
    at 3 AM before sunrise (pre-dawn ≠ post-sunset).

    Logic:
      - Compute sunset local hour from astronomical formula
      - sun_set = True only if local_hour > sunset_local (afternoon passed)
      - Handle midnight wrap: if local_hour < sunrise (early AM), sun set
        yesterday afternoon so sun_set = True and hours_since_sunset wraps
    """
    now = datetime.now(timezone.utc)
    n = now.timetuple().tm_yday
    hour_utc = now.hour + now.minute / 60.0

    # Solar declination
    decl = 23.45 * math.sin(math.radians(360 / 365 * (n - 81)))

    # Solar elevation (for display only)
    lat_r = math.radians(lat)
    decl_r = math.radians(decl)
    solar_hour = hour_utc + lon / 15.0
    ha = (solar_hour - 12) * 15
    ha_r = math.radians(ha)
    elev = math.degrees(math.asin(
        math.sin(lat_r) * math.sin(decl_r) +
        math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    ))

    # Compute sunset & sunrise local hours from astronomical formula
    cos_ha_sunset = -math.tan(lat_r) * math.tan(decl_r)
    cos_ha_sunset = max(-1, min(1, cos_ha_sunset))
    ha_sunset_deg = math.degrees(math.acos(cos_ha_sunset))

    # Sunset/sunrise in UTC hours
    sunset_utc = 12 + ha_sunset_deg / 15 - lon / 15
    sunrise_utc = 12 - ha_sunset_deg / 15 - lon / 15

    # Convert to local hours
    local_hour = hour_utc + tz_offset
    if local_hour < 0:
        local_hour += 24
    elif local_hour >= 24:
        local_hour -= 24

    sunset_local = sunset_utc + tz_offset
    if sunset_local < 0:
        sunset_local += 24
    elif sunset_local >= 24:
        sunset_local -= 24

    sunrise_local = sunrise_utc + tz_offset
    if sunrise_local < 0:
        sunrise_local += 24
    elif sunrise_local >= 24:
        sunrise_local -= 24

    # Determine if sun has set for TODAY's max temp purpose:
    # sun_set = True if we're past sunset (afternoon) OR it's past midnight
    # but before sunrise (the max temp from yesterday is already locked)
    #
    # Timeline: [sunrise ~7am] ... [peak ~2pm] ... [sunset ~5pm] ... [midnight] ... [sunrise]
    #   Before sunrise: sun_set=True  (yesterday's max is locked)
    #   sunrise to sunset: sun_set=False (today's max still forming)
    #   After sunset: sun_set=True (today's max is locked)
    if local_hour >= sunset_local:
        # Past sunset in the afternoon/evening
        sun_set = True
        hours_since = round(local_hour - sunset_local, 1)
    elif local_hour < sunrise_local:
        # Past midnight, before sunrise — yesterday's sunset already happened
        sun_set = True
        hours_since = round((24 - sunset_local) + local_hour, 1)
    else:
        # Daytime — sun is up, max temp still forming
        sun_set = False
        hours_since = 0

    return {
        "solar_elevation": round(elev, 1),
        "sun_set": sun_set,
        "local_hour": round(local_hour, 1),
        "sunset_local": round(sunset_local, 1),
        "sunrise_local": round(sunrise_local, 1),
        "hours_since_sunset": hours_since,
    }


def fetch_observed_high(lat, lon, target_date):
    """Fetch actual observed high temperature from Open-Meteo historical API."""
    date_str = target_date.strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).date()

    # For today or yesterday, use the forecast API with past_days
    if target_date >= today - timedelta(days=1):
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "timezone": "auto",
                    "past_days": 2,
                    "forecast_days": 1,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                dates = data.get("daily", {}).get("time", [])
                temps = data.get("daily", {}).get("temperature_2m_max", [])
                for i, d in enumerate(dates):
                    if d == date_str and i < len(temps) and temps[i] is not None:
                        return {"temp_f": temps[i], "source": "Open-Meteo observation", "date": date_str}
        except Exception as e:
            log.debug("Open-Meteo obs error: %s", e)

    # For older dates, use archive API
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": date_str,
                "end_date": date_str,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            if temps and temps[0] is not None:
                return {"temp_f": temps[0], "source": "Open-Meteo archive", "date": date_str}
    except Exception as e:
        log.debug("Open-Meteo archive error: %s", e)

    return None


def fetch_metar_max(station, hours_back=24):
    """Get the maximum temperature from recent METAR observations."""
    try:
        resp = requests.get(
            "https://aviationweather.gov/api/data/metar",
            params={
                "ids": station,
                "format": "json",
                "hours": hours_back,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            max_temp_c = None
            for obs in data:
                temp_c = obs.get("temp")
                if temp_c is not None:
                    if max_temp_c is None or temp_c > max_temp_c:
                        max_temp_c = temp_c
            if max_temp_c is not None:
                max_temp_f = max_temp_c * 9 / 5 + 32
                return {"temp_f": round(max_temp_f, 1), "temp_c": round(max_temp_c, 1), "source": "METAR", "n_obs": len(data)}
    except Exception as e:
        log.debug("METAR error for %s: %s", station, e)
    return None


def fetch_metar_multi(city_info, hours_back=24):
    """Fetch METAR from multiple stations for a city.

    Uses primary + alternate stations (e.g. EGLC + EGLL for London)
    and returns the best reading (highest max temp from closest station).
    """
    stations = [city_info["metar"]]
    if "metar_alt" in city_info:
        stations = list(city_info["metar_alt"])

    results = []
    for station in stations:
        r = fetch_metar_max(station, hours_back)
        if r:
            r["station"] = station
            results.append(r)

    if not results:
        return None

    # Return highest observed max (most conservative for trading)
    results.sort(key=lambda x: x["temp_f"], reverse=True)
    best = results[0]
    if len(results) > 1:
        best["source"] = "METAR (%s, max of %d stations)" % (
            best["station"], len(results))
        best["all_stations"] = results
    return best


def fetch_visual_crossing_forecast(lat, lon, target_date):
    """Fetch forecast from Visual Crossing free API (1000 records/day free).

    Visual Crossing provides high-accuracy forecasts and historical data
    using ~30k weather stations worldwide.
    """
    global _MULTI_WX_CACHE
    cache_key = "vc_%s_%s_%s" % (round(lat, 2), round(lon, 2), target_date)
    if cache_key in _MULTI_WX_CACHE:
        cached = _MULTI_WX_CACHE[cache_key]
        if time.time() - cached["ts"] < _MULTI_WX_TTL:
            return cached["data"]

    vc_key = "HFNP8CDTHWQJP6VNRNTKTLPWZ"  # Free tier key
    date_str = target_date.strftime("%Y-%m-%d") if hasattr(target_date, "strftime") else str(target_date)
    try:
        resp = requests.get(
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{lat},{lon}/{date_str}",
            params={
                "unitGroup": "us",
                "key": vc_key,
                "include": "days",
                "elements": "tempmax,tempmin,temp,humidity,precip,windspeed,conditions",
            },
            timeout=12,
        )
        if resp.status_code == 200:
            data = resp.json()
            days = data.get("days", [])
            if days:
                day = days[0]
                result = {
                    "temp_max_f": day.get("tempmax"),
                    "temp_min_f": day.get("tempmin"),
                    "conditions": day.get("conditions", ""),
                    "humidity": day.get("humidity"),
                    "wind_mph": day.get("windspeed"),
                    "source": "Visual Crossing",
                }
                _MULTI_WX_CACHE[cache_key] = {"ts": time.time(), "data": result}
                return result
    except Exception as e:
        log.debug("Visual Crossing error: %s", e)
    return None


def fetch_openmeteo_ensemble(lat, lon, days_ahead=3):
    """Fetch multi-model ensemble forecast from Open-Meteo.

    Uses GFS, ECMWF IFS, DWD ICON, GEM, JMA for maximum accuracy.
    Returns per-day synthesized forecasts with confidence intervals.
    """
    global _MULTI_WX_CACHE
    cache_key = "ensemble_%s_%s" % (round(lat, 2), round(lon, 2))
    if cache_key in _MULTI_WX_CACHE:
        cached = _MULTI_WX_CACHE[cache_key]
        if time.time() - cached["ts"] < _MULTI_WX_TTL:
            return cached["data"]

    model_configs = {
        "gfs": {"url": "https://api.open-meteo.com/v1/gfs", "name": "GFS (NCEP)", "weight": 0.20, "spread": 4.0},
        "ecmwf": {"url": "https://api.open-meteo.com/v1/ecmwf", "name": "ECMWF (IFS)", "weight": 0.30, "spread": 3.5},
        "icon": {"url": "https://api.open-meteo.com/v1/dwd-icon", "name": "ICON (DWD)", "weight": 0.20, "spread": 4.0},
        "gem": {"url": "https://api.open-meteo.com/v1/gem", "name": "GEM (Canada)", "weight": 0.15, "spread": 4.5},
        "jma": {"url": "https://api.open-meteo.com/v1/jma", "name": "JMA (Japan)", "weight": 0.15, "spread": 5.0},
    }

    models = {}
    for model_key, config in model_configs.items():
        try:
            resp = requests.get(config["url"], params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "forecast_days": max(days_ahead, 3),
            }, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                models[model_key] = {
                    "name": config["name"],
                    "weight": config["weight"],
                    "spread": config["spread"],
                    "temp_max_f": daily.get("temperature_2m_max", []),
                    "temp_min_f": daily.get("temperature_2m_min", []),
                    "dates": daily.get("time", []),
                }
        except Exception:
            continue

    if not models:
        return None

    # Synthesize per-day forecasts using Bayesian precision weighting
    result = {"models": models, "per_day": []}
    n_days = min(days_ahead, max(len(m["temp_max_f"]) for m in models.values()))

    for day_idx in range(n_days):
        forecasts = []
        for mk, md in models.items():
            temps = md.get("temp_max_f", [])
            if len(temps) > day_idx and temps[day_idx] is not None:
                forecasts.append({
                    "model": mk,
                    "temp_f": temps[day_idx],
                    "weight": md["weight"],
                    "spread": md["spread"],
                })

        if not forecasts:
            result["per_day"].append(None)
            continue

        # Bayesian precision combination
        total_precision = 0
        weighted_mean = 0
        for fc in forecasts:
            precision = fc["weight"] / (fc["spread"] ** 2)
            total_precision += precision
            weighted_mean += precision * fc["temp_f"]

        synth_mean = weighted_mean / total_precision if total_precision else 0
        synth_std = max(math.sqrt(1.0 / total_precision) if total_precision else 5.0, 3.0)

        temps_list = [fc["temp_f"] for fc in forecasts]
        model_spread = max(temps_list) - min(temps_list) if temps_list else 0
        agreement = "HIGH" if model_spread < 2 else "MODERATE" if model_spread < 4 else "LOW"

        date_str = ""
        for md in models.values():
            dates = md.get("dates", [])
            if len(dates) > day_idx:
                date_str = dates[day_idx]
                break

        result["per_day"].append({
            "date": date_str,
            "day_index": day_idx,
            "synth_mean_f": round(synth_mean, 1),
            "synth_std_f": round(synth_std, 2),
            "model_spread_f": round(model_spread, 1),
            "agreement": agreement,
            "n_models": len(forecasts),
            "model_forecasts": forecasts,
        })

    _MULTI_WX_CACHE[cache_key] = {"ts": time.time(), "data": result}
    return result


def bracket_probability(synth_mean_f, synth_std_f, low, high, unit):
    """Calculate probability that temperature falls in bracket using Gaussian CDF.

    Handles °C brackets by converting thresholds to °F for comparison.
    Returns probability between 0 and 1.
    """
    if unit == "C":
        # Convert bracket bounds from °C to °F for comparison with synth_mean_f (in °F)
        if low is not None:
            low_f = low * 9 / 5 + 32
        else:
            low_f = -100
        if high is not None:
            high_f = high * 9 / 5 + 32
        else:
            high_f = 200
    else:
        low_f = low if low is not None else -100
        high_f = high if high is not None else 200

    def phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    z_lo = (low_f - synth_mean_f) / synth_std_f if synth_std_f > 0 else -10
    z_hi = (high_f - synth_mean_f) / synth_std_f if synth_std_f > 0 else 10

    prob = phi(z_hi) - phi(z_lo)
    return max(0.001, min(0.999, prob))


def parse_market_bracket(question):
    """Parse the temperature bracket and unit from market question."""
    import re
    q = question.lower()

    # "between 18-19°F"
    m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).upper()
    # "15°F or below" / "15°f or lower"
    m = re.search(r'(-?\d+)\s*°?\s*([fFcC])\s+or\s+(?:below|lower)', q)
    if m:
        return None, int(m.group(1)), m.group(2).upper()
    # "26°F or higher" / "26°f or above"
    m = re.search(r'(-?\d+)\s*°?\s*([fFcC])\s+or\s+(?:higher|above)', q)
    if m:
        return int(m.group(1)), None, m.group(2).upper()
    # "exactly 33°C"
    m = re.search(r'exactly\s+(-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(1)), m.group(2).upper()
    return None, None, None


def check_outcome(observed_temp_f, low, high, unit):
    """Check if observed temp falls in bracket.
    Returns 'YES_WINS', 'NO_WINS', 'MARGINAL', or 'UNCERTAIN'.

    Polymarket weather oracles typically use Wunderground rounding:
    - Fahrenheit: rounded to nearest integer
    - Celsius: rounded to nearest integer
    We add a ±1 degree safety margin for MARGINAL classification.
    """
    if unit == "C":
        observed = round((observed_temp_f - 32) * 5 / 9)  # Round to match oracle
    else:
        observed = round(observed_temp_f)  # Round to match oracle

    # "X or below" → low is None, high is threshold
    if low is None and high is not None:
        if observed <= high - 1:  # Clearly inside with margin
            return "YES_WINS"
        elif observed <= high:  # Right at boundary
            return "MARGINAL"
        elif observed > high + 1:  # Clearly outside
            return "NO_WINS"
        else:
            return "UNCERTAIN"

    # "X or higher" → low is threshold, high is None
    if high is None and low is not None:
        if observed >= low + 1:  # Clearly above with margin
            return "YES_WINS"
        elif observed >= low:  # Right at boundary
            return "MARGINAL"
        elif observed < low - 1:
            return "NO_WINS"
        else:
            return "UNCERTAIN"

    # "between X-Y"
    if low is not None and high is not None:
        if low + 1 <= observed <= high - 1:  # Clearly inside
            return "YES_WINS"
        elif low <= observed <= high:  # Inside but at boundary
            return "MARGINAL"
        elif observed < low - 1 or observed > high + 1:  # Clearly outside
            return "NO_WINS"
        else:
            return "UNCERTAIN"

    return "UNCERTAIN"


def discover_zombie_markets():
    """Find weather markets for dates where the event is likely over."""
    now = datetime.now(timezone.utc)
    zombies = []
    graveyard = []

    # Check today and yesterday's markets
    for day_offset in [0, -1]:
        target = now + timedelta(days=day_offset)
        target_date = target.date()
        month = MONTH_NAMES[target.month]
        day = target.day
        year = target.year

        for city_key, info in ZOMBIE_CITIES.items():
            slug = "highest-temperature-in-%s-on-%s-%d-%d" % (info["slug"], month, day, year)

            # Check sun status for this city
            sun = get_sun_status(info["lat"], info["lon"], info["tz_offset"])

            # Skip if today's max temp hasn't been locked yet
            if day_offset == 0:
                # For TODAY's market: only trade after the sun sets in the afternoon.
                # Pre-dawn (before sunrise) does NOT count — the heating cycle hasn't
                # started yet so today's max temp is still unknown.
                local_h = sun["local_hour"]
                sunset_h = sun["sunset_local"]
                if local_h < sunset_h:
                    # Before sunset: day is still in progress, skip
                    continue

            # Fetch the event from Polymarket
            try:
                resp = requests.get(
                    "%s/events" % GAMMA_HOST,
                    params={"slug": slug},
                    timeout=10,
                )
                if not resp or resp.status_code != 200:
                    continue
                data = resp.json()
                event = data[0] if isinstance(data, list) and data else data
                if not event or not event.get("markets"):
                    continue
            except Exception:
                continue

            # Get observed high temperature (multi-source)
            observed = fetch_observed_high(info["lat"], info["lon"], target_date)
            metar = fetch_metar_multi(info, hours_back=24 if day_offset == 0 else 36)
            vc = fetch_visual_crossing_forecast(info["lat"], info["lon"], target_date)

            # Use best available observation (priority: METAR > VC > Open-Meteo)
            obs_temp_f = None
            obs_source = "none"
            sources_checked = []
            if observed:
                obs_temp_f = observed["temp_f"]
                obs_source = observed["source"]
                sources_checked.append(("Open-Meteo", observed["temp_f"]))
            if vc and vc.get("temp_max_f") is not None:
                sources_checked.append(("Visual Crossing", vc["temp_max_f"]))
                if obs_temp_f is None or abs(vc["temp_max_f"] - (obs_temp_f or 0)) < 5:
                    obs_temp_f = vc["temp_max_f"]
                    obs_source = "Visual Crossing"
            if metar:
                # METAR is ground truth, prefer it if available
                sources_checked.append(("METAR " + metar.get("station", info["metar"]), metar["temp_f"]))
                if obs_temp_f is None or abs(metar["temp_f"] - (obs_temp_f or 0)) < 5:
                    obs_temp_f = metar["temp_f"]
                    obs_source = metar.get("source", "METAR (%s)" % info["metar"])

            if obs_temp_f is None:
                continue

            # Analyze each market in this event
            for m in event.get("markets", []):
                question = m.get("question", "")
                cid = m.get("conditionId", "")
                if not cid:
                    continue

                # Check if market is still active
                closed = m.get("closed", False)
                resolved = m.get("resolved", False)

                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except Exception:
                        continue
                if not prices or len(prices) < 2:
                    continue

                yes_price = float(prices[0])
                no_price = float(prices[1])

                vol = float(m.get("volume", 0) or 0)
                liq = float(m.get("liquidity", 0) or 0)

                # Parse bracket
                low, high, unit = parse_market_bracket(question)
                if unit is None:
                    continue

                # Determine outcome
                outcome = check_outcome(obs_temp_f, low, high, unit)

                # Convert observed to display unit
                if unit == "C":
                    obs_display = round((obs_temp_f - 32) * 5 / 9, 1)
                    obs_unit = "°C"
                else:
                    obs_display = round(obs_temp_f, 1)
                    obs_unit = "°F"

                # Calculate zombie opportunity
                zombie_entry = {
                    "city": city_key,
                    "city_name": info["name"],
                    "question": question[:100],
                    "slug": slug,
                    "conditionId": cid,
                    "market_date": target_date.strftime("%Y-%m-%d"),
                    "day_offset": day_offset,
                    "yes_price": round(yes_price, 4),
                    "no_price": round(no_price, 4),
                    "volume": vol,
                    "liquidity": liq,
                    "observed_temp": obs_display,
                    "observed_unit": obs_unit,
                    "observed_temp_f": round(obs_temp_f, 1),
                    "obs_source": obs_source,
                    "sources_checked": sources_checked,
                    "outcome": outcome,
                    "sun_set": sun["sun_set"],
                    "local_hour": sun["local_hour"],
                    "hours_since_sunset": sun["hours_since_sunset"],
                    "resolved": resolved,
                    "closed": closed,
                }

                if resolved or closed:
                    # Already resolved — graveyard
                    zombie_entry["status"] = "RESOLVED"
                    zombie_entry["status_icon"] = "⚰️"
                    graveyard.append(zombie_entry)
                    continue

                if outcome == "YES_WINS":
                    buy_price = yes_price
                    zombie_entry["action"] = "BUY YES"
                    zombie_entry["buy_price"] = round(buy_price, 4)
                    zombie_entry["yield_pct"] = round((1.0 - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0
                    zombie_entry["profit_per_1k"] = round((1.0 - buy_price) * 1000, 2)
                elif outcome == "NO_WINS":
                    buy_price = no_price
                    zombie_entry["action"] = "BUY NO"
                    zombie_entry["buy_price"] = round(buy_price, 4)
                    zombie_entry["yield_pct"] = round((1.0 - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0
                    zombie_entry["profit_per_1k"] = round((1.0 - buy_price) * 1000, 2)
                elif outcome == "MARGINAL":
                    # Outcome likely correct but temp is at boundary — risky
                    # Still show but flag as marginal
                    # Use whichever side the simple round suggests
                    obs_rounded = round((obs_temp_f - 32) * 5 / 9) if unit == "C" else round(obs_temp_f)
                    if low is None and high is not None:
                        likely_yes = obs_rounded <= high
                    elif high is None and low is not None:
                        likely_yes = obs_rounded >= low
                    else:
                        likely_yes = low is not None and high is not None and low <= obs_rounded <= high
                    if likely_yes:
                        buy_price = yes_price
                        zombie_entry["action"] = "BUY YES (MARGINAL)"
                    else:
                        buy_price = no_price
                        zombie_entry["action"] = "BUY NO (MARGINAL)"
                    zombie_entry["buy_price"] = round(buy_price, 4)
                    zombie_entry["yield_pct"] = round((1.0 - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0
                    zombie_entry["profit_per_1k"] = round((1.0 - buy_price) * 1000, 2)
                else:
                    zombie_entry["action"] = "AVOID"
                    zombie_entry["buy_price"] = None
                    zombie_entry["yield_pct"] = 0
                    zombie_entry["profit_per_1k"] = 0

                # Status assignment
                if outcome in ("UNCERTAIN", "MARGINAL"):
                    zombie_entry["status"] = "MARGINAL" if outcome == "MARGINAL" else "DISPUTED"
                    zombie_entry["status_icon"] = "⚠️"
                elif sun["hours_since_sunset"] > 3:
                    zombie_entry["status"] = "DEAD_CONFIRMED"
                    zombie_entry["status_icon"] = "🧟"
                elif sun["sun_set"]:
                    zombie_entry["status"] = "SUN_SET"
                    zombie_entry["status_icon"] = "🌑"
                else:
                    zombie_entry["status"] = "WATCHING"
                    zombie_entry["status_icon"] = "👀"

                # Estimate hours until resolution (~3 AM next day local)
                local_h = sun["local_hour"]
                if local_h < 3:
                    eta_hours = 3 - local_h
                else:
                    eta_hours = 27 - local_h  # Next day 3 AM
                zombie_entry["eta_hours"] = round(eta_hours, 1)
                eta_h = int(eta_hours)
                eta_m = int((eta_hours - eta_h) * 60)
                zombie_entry["eta_display"] = "%dh %02dm" % (eta_h, eta_m)

                # Only include if there's an actual opportunity (discount exists)
                if zombie_entry.get("buy_price") and zombie_entry["buy_price"] < 0.995:
                    zombies.append(zombie_entry)

    # Sort zombies by yield (highest first)
    zombies.sort(key=lambda z: z.get("yield_pct", 0), reverse=True)

    # Separate into tiers
    ready = [z for z in zombies if z["status"] in ("DEAD_CONFIRMED", "SUN_SET")
             and z["action"] not in ("AVOID",) and "MARGINAL" not in z.get("action", "")
             and z["yield_pct"] > 0.5]
    marginal = [z for z in zombies if "MARGINAL" in z.get("action", "")]
    watching = [z for z in zombies if z["status"] == "WATCHING" or z["yield_pct"] <= 0.5]
    disputed = [z for z in zombies if z["status"] in ("DISPUTED", "MARGINAL")]

    # ════════════════════════════════════════════════════════════════
    # FORWARD OPPORTUNITIES — Tomorrow (+1d) and Day After (+2d)
    # Uses multi-model ensemble forecasts to find mispriced markets
    # ════════════════════════════════════════════════════════════════
    forecast_opportunities = []

    for day_offset in [1, 2]:
        target = now + timedelta(days=day_offset)
        target_date = target.date()
        month = MONTH_NAMES[target.month]
        day = target.day
        year = target.year

        for city_key, info in ZOMBIE_CITIES.items():
            slug = "highest-temperature-in-%s-on-%s-%d-%d" % (info["slug"], month, day, year)

            # Fetch the event from Polymarket
            try:
                resp = requests.get(
                    "%s/events" % GAMMA_HOST,
                    params={"slug": slug},
                    timeout=10,
                )
                if not resp or resp.status_code != 200:
                    continue
                data = resp.json()
                event = data[0] if isinstance(data, list) and data else data
                if not event or not event.get("markets"):
                    continue
            except Exception:
                continue

            # Get ensemble forecast for this city
            ensemble = fetch_openmeteo_ensemble(info["lat"], info["lon"], days_ahead=3)
            if not ensemble or not ensemble.get("per_day"):
                continue

            day_forecast = None
            for pday in ensemble["per_day"]:
                if pday and pday.get("day_index") == day_offset:
                    day_forecast = pday
                    break

            if not day_forecast:
                continue

            synth_mean_f = day_forecast["synth_mean_f"]
            synth_std_f = day_forecast["synth_std_f"]

            # Also fetch Visual Crossing for comparison
            vc = fetch_visual_crossing_forecast(info["lat"], info["lon"], target_date)
            vc_temp_f = vc.get("temp_max_f") if vc else None

            # Cross-validate: average if both sources available
            if vc_temp_f is not None:
                # Weighted average: 70% ensemble, 30% VC
                blended_f = synth_mean_f * 0.7 + vc_temp_f * 0.3
            else:
                blended_f = synth_mean_f

            # Determine display unit
            is_celsius = info.get("slug") in ("london", "toronto", "buenos-aires", "seoul", "ankara", "wellington")

            for m in event.get("markets", []):
                question = m.get("question", "")
                cid = m.get("conditionId", "")
                if not cid:
                    continue

                closed = m.get("closed", False)
                resolved = m.get("resolved", False)
                if closed or resolved:
                    continue

                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except Exception:
                        continue
                if not prices or len(prices) < 2:
                    continue

                yes_price = float(prices[0])
                no_price = float(prices[1])
                vol = float(m.get("volume", 0) or 0)
                liq = float(m.get("liquidity", 0) or 0)

                low, high, unit = parse_market_bracket(question)
                if unit is None:
                    continue

                # Calculate probability of landing in this bracket
                our_prob = bracket_probability(blended_f, synth_std_f, low, high, unit)

                # Edge = our_prob - market_price
                edge_yes = our_prob - yes_price
                edge_no = (1 - our_prob) - no_price

                # Pick the side with positive edge
                if edge_yes > 0.02:  # >2% edge threshold
                    action = "BUY YES"
                    buy_price = yes_price
                    edge = edge_yes
                elif edge_no > 0.02:
                    action = "BUY NO"
                    buy_price = no_price
                    edge = edge_no
                else:
                    continue  # No meaningful edge

                # EV calculation
                if action == "BUY YES":
                    ev_pct = (our_prob * (1.0 - buy_price) - (1 - our_prob) * buy_price) / buy_price * 100 if buy_price > 0 else 0
                else:
                    prob_no = 1 - our_prob
                    ev_pct = (prob_no * (1.0 - buy_price) - our_prob * buy_price) / buy_price * 100 if buy_price > 0 else 0

                # Half Kelly sizing
                if buy_price > 0 and buy_price < 1:
                    b = (1.0 - buy_price) / buy_price
                    p = our_prob if action == "BUY YES" else (1 - our_prob)
                    kelly_f = max(0, (b * p - (1 - p)) / b * 0.5)
                else:
                    kelly_f = 0

                bet_size = min(kelly_f * 5000, 2000)  # Cap at $2k per forecast market

                # Display temp in local unit
                if is_celsius or unit == "C":
                    display_temp = "%d°C" % round((blended_f - 32) * 5 / 9)
                    display_range = "%d-%d°C" % (
                        round((blended_f - synth_std_f - 32) * 5 / 9),
                        round((blended_f + synth_std_f - 32) * 5 / 9),
                    )
                else:
                    display_temp = "%d°F" % round(blended_f)
                    display_range = "%d-%d°F" % (
                        round(blended_f - synth_std_f),
                        round(blended_f + synth_std_f),
                    )

                forecast_opportunities.append({
                    "type": "FORECAST",
                    "city": city_key,
                    "city_name": info["name"],
                    "question": question[:100],
                    "slug": slug,
                    "conditionId": cid,
                    "market_date": target_date.strftime("%Y-%m-%d"),
                    "day_offset": day_offset,
                    "yes_price": round(yes_price, 4),
                    "no_price": round(no_price, 4),
                    "volume": vol,
                    "liquidity": liq,
                    "action": action,
                    "buy_price": round(buy_price, 4),
                    "our_prob": round(our_prob, 4),
                    "edge": round(edge, 4),
                    "ev_pct": round(ev_pct, 1),
                    "kelly_fraction": round(kelly_f, 4),
                    "bet_size": round(bet_size, 2),
                    "forecast_temp_f": round(blended_f, 1),
                    "display_temp": display_temp,
                    "display_range": display_range,
                    "synth_mean_f": synth_mean_f,
                    "synth_std_f": synth_std_f,
                    "model_agreement": day_forecast["agreement"],
                    "n_models": day_forecast["n_models"],
                    "model_spread_f": day_forecast["model_spread_f"],
                    "vc_temp_f": vc_temp_f,
                    "status": "FORECAST",
                    "status_icon": "🔮",
                    "confidence": day_forecast["agreement"],
                })

    # Sort forecast opportunities by EV% (highest first)
    forecast_opportunities.sort(key=lambda x: x.get("ev_pct", 0), reverse=True)

    # Capital allocation (from $15k reserve)
    total_capital = 15000
    for z in ready:
        max_per_market = min(5000, total_capital * 0.4)
        available_liq = z.get("liquidity", 0)
        z["deploy_amount"] = round(min(max_per_market, available_liq * 0.8, total_capital), 2)
        z["expected_profit"] = round(z["deploy_amount"] * (1.0 - z["buy_price"]), 2)
        total_capital -= z["deploy_amount"]
        if total_capital <= 0:
            break

    # Stats
    total_deployed = sum(z.get("deploy_amount", 0) for z in ready)
    total_profit = sum(z.get("expected_profit", 0) for z in ready)
    avg_yield = sum(z["yield_pct"] for z in ready) / len(ready) if ready else 0
    forecast_ev = sum(o["ev_pct"] * o["bet_size"] / 100 for o in forecast_opportunities if o.get("ev_pct", 0) > 0)

    return {
        "zombies_ready": ready,
        "zombies_watching": watching,
        "zombies_marginal": marginal,
        "zombies_disputed": disputed,
        "graveyard": graveyard[:10],
        "forecast_opportunities": forecast_opportunities[:30],  # Top 30 forward-looking opportunities
        "stats": {
            "total_ready": len(ready),
            "total_watching": len(watching),
            "total_marginal": len(marginal),
            "total_disputed": len(disputed),
            "total_forecast": len(forecast_opportunities),
            "capital_reserved": 15000,
            "capital_deployed": round(total_deployed, 2),
            "expected_profit": round(total_profit, 2),
            "forecast_ev": round(forecast_ev, 2),
            "avg_yield_pct": round(avg_yield, 2),
            "status": "HUNTING" if ready else "SCANNING",
        },
        "data_sources": ["Open-Meteo Ensemble (GFS/ECMWF/ICON/GEM/JMA)", "Visual Crossing", "METAR/ASOS (multi-station)", "Open-Meteo Historical"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = discover_zombie_markets()
    print("\n" + "=" * 70)
    print("  ZOMBIE HUNTER — Settlement Arbitrage Report")
    print("=" * 70)
    stats = data["stats"]
    print("  Ready: %d | Watching: %d | Disputed: %d" % (
        stats["total_ready"], stats["total_watching"], stats["total_disputed"]))
    print("  Capital: $%s deployed | Expected Profit: $%s | Avg Yield: %.2f%%" % (
        int(stats["capital_deployed"]), int(stats["expected_profit"]), stats["avg_yield_pct"]))

    for z in data["zombies_ready"]:
        print("\n  %s %s | %s" % (z["status_icon"], z["city_name"], z["question"]))
        print("    Observed: %s%s | %s → %s at %s¢" % (
            z["observed_temp"], z["observed_unit"], z["outcome"],
            z["action"], int(z["buy_price"] * 100)))
        print("    Yield: +%.2f%% | Deploy: $%s | Profit: $%s | ETA: %s" % (
            z["yield_pct"], int(z.get("deploy_amount", 0)),
            round(z.get("expected_profit", 0), 2), z["eta_display"]))
    print()
