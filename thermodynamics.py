#!/usr/bin/env python3
"""
THERMODYNAMICS ENGINE — Atmospheric Physics Arbitrage
======================================================
Calculates the Energy Budget of a city to predict temperature
with physics precision beyond what forecast models provide.

Modules:
  A. Radiative Transfer Model (Solar Heating Engine)
  B. Advection Calculator (Wind Transport Engine)
  C. Ensemble Probability Density (Bayesian Model Synthesis)
  D. Sensor Fusion (Real-time ground truth integration)

Uses actual atmospheric physics equations:
  - Solar irradiance = S₀ × cos(θ) × τ × (1 - α)
  - Advection: dT/dt = -u(dT/dx)
  - Stefan-Boltzmann longwave: F = εσT⁴
  - Bayesian ensemble: P(T) = ∏ P_i(T) normalized
"""

import json
import math
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOG_DIR = Path("logs")

# ── Ensemble forecast cache (Open-Meteo has daily rate limits) ──
_ENSEMBLE_CACHE_FILE = Path("ensemble_cache.json")
_ENSEMBLE_TTL = 1800      # 30 min — NWP models update every 6h, no need to poll every 90s

def _load_ensemble_cache():
    """Load file-based ensemble cache."""
    try:
        if _ENSEMBLE_CACHE_FILE.exists():
            return json.loads(_ENSEMBLE_CACHE_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_ensemble_cache(cache):
    """Persist ensemble cache to disk."""
    try:
        _ENSEMBLE_CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

SOLAR_CONSTANT = 1361.0  # W/m² at top of atmosphere
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
ATMOSPHERIC_TRANSMITTANCE = 0.75  # Clear sky
URBAN_ALBEDO = 0.15  # Concrete/asphalt
RURAL_ALBEDO = 0.25  # Grass/soil
WATER_ALBEDO = 0.06  # Ocean/lake
EMISSIVITY_URBAN = 0.95
SPECIFIC_HEAT_AIR = 1005  # J/(kg·K)
AIR_DENSITY = 1.225  # kg/m³ at sea level
MIXING_HEIGHT = 1000  # meters, typical daytime boundary layer

# City physical parameters
CITY_PHYSICS = {
    "nyc":          {"lat": 40.71, "lon": -74.01, "elev_m": 10,  "albedo": 0.13, "thermal_mass": 1.2, "tz_offset": -5, "metar": "KNYC", "upstream_stations": ["KEWR", "KJFK"]},
    "chicago":      {"lat": 41.88, "lon": -87.63, "elev_m": 180, "albedo": 0.16, "thermal_mass": 1.1, "tz_offset": -6, "metar": "KORD", "upstream_stations": ["KDPA", "KARR"], "lake_effect": True, "lake_temp_c": 2.0},
    "miami":        {"lat": 25.76, "lon": -80.19, "elev_m": 2,   "albedo": 0.12, "thermal_mass": 1.4, "tz_offset": -5, "metar": "KMIA", "upstream_stations": ["KFLL", "KPBI"], "maritime": True},
    "seattle":      {"lat": 47.61, "lon": -122.33,"elev_m": 60,  "albedo": 0.18, "thermal_mass": 1.0, "tz_offset": -8, "metar": "KSEA", "upstream_stations": ["KBFI", "KPAE"]},
    "dallas":       {"lat": 32.78, "lon": -96.80, "elev_m": 130, "albedo": 0.17, "thermal_mass": 0.9, "tz_offset": -6, "metar": "KDFW", "upstream_stations": ["KDAL", "KFTW"]},
    "atlanta":      {"lat": 33.75, "lon": -84.39, "elev_m": 320, "albedo": 0.19, "thermal_mass": 0.95,"tz_offset": -5, "metar": "KATL", "upstream_stations": ["KPDK", "KRYY"]},
    "london":       {"lat": 51.51, "lon": -0.13,  "elev_m": 11,  "albedo": 0.15, "thermal_mass": 1.15,"tz_offset": 0,  "metar": "EGLL", "upstream_stations": ["EGKK", "EGLC"], "maritime": True},
    "toronto":      {"lat": 43.65, "lon": -79.38, "elev_m": 76,  "albedo": 0.22, "thermal_mass": 1.05,"tz_offset": -5, "metar": "CYYZ", "upstream_stations": ["CYKZ"], "lake_effect": True, "lake_temp_c": 1.0},
    "buenos aires": {"lat": -34.60,"lon": -58.38, "elev_m": 25,  "albedo": 0.14, "thermal_mass": 1.1, "tz_offset": -3, "metar": "SAEZ", "upstream_stations": []},
    "seoul":        {"lat": 37.57, "lon": 126.98, "elev_m": 38,  "albedo": 0.16, "thermal_mass": 1.1, "tz_offset": 9,  "metar": "RKSI", "upstream_stations": []},
    "ankara":       {"lat": 39.93, "lon": 32.86,  "elev_m": 890, "albedo": 0.20, "thermal_mass": 0.85,"tz_offset": 3,  "metar": "LTAC", "upstream_stations": []},
    "wellington":   {"lat": -41.29,"lon": 174.78, "elev_m": 5,   "albedo": 0.14, "thermal_mass": 1.2, "tz_offset": 13, "metar": "NZWN", "upstream_stations": [], "maritime": True},
}

# Forecast model weights for Bayesian synthesis
MODEL_WEIGHTS = {
    "gfs": 0.30,     # GFS — good general purpose
    "ecmwf": 0.35,   # ECMWF — highest skill score globally
    "icon": 0.15,    # ICON (DWD) — good for European cities
    "gem": 0.10,     # GEM (Canadian) — good for NA
    "jma": 0.10,     # JMA — good for Asian cities
}


# ══════════════════════════════════════════════════════════════════
# SECTOR A: RADIATIVE TRANSFER MODEL (Solar Heating Engine)
# ══════════════════════════════════════════════════════════════════

def solar_declination(day_of_year):
    """Solar declination angle in radians."""
    return math.radians(23.44) * math.sin(math.radians(360 / 365 * (day_of_year - 81)))


def solar_hour_angle(hour_utc, lon):
    """Solar hour angle in radians. 0 = solar noon."""
    solar_time = hour_utc + lon / 15.0
    return math.radians(15 * (solar_time - 12))


def solar_elevation_angle(lat, lon, dt_utc):
    """Calculate sun elevation angle in degrees."""
    lat_r = math.radians(lat)
    doy = dt_utc.timetuple().tm_yday
    decl = solar_declination(doy)
    hour = dt_utc.hour + dt_utc.minute / 60.0
    ha = solar_hour_angle(hour, lon)

    sin_elev = (math.sin(lat_r) * math.sin(decl) +
                math.cos(lat_r) * math.cos(decl) * math.cos(ha))
    elev = math.degrees(math.asin(max(-1, min(1, sin_elev))))
    return elev


def solar_noon_utc(lon):
    """Approximate solar noon in UTC hours."""
    return 12.0 - lon / 15.0


def shortwave_flux(lat, lon, dt_utc, cloud_fraction=0.0, albedo=URBAN_ALBEDO):
    """Calculate net shortwave (solar) radiation flux at surface in W/m².

    Parameters:
        lat, lon: City coordinates
        dt_utc: Current UTC datetime
        cloud_fraction: 0.0 (clear) to 1.0 (overcast)
        albedo: Surface reflectivity (0.0 to 1.0)

    Returns:
        dict with flux components
    """
    elev_deg = solar_elevation_angle(lat, lon, dt_utc)

    if elev_deg <= 0:
        return {
            "solar_elevation_deg": round(elev_deg, 1),
            "toa_flux": 0,
            "cloud_transmittance": 1.0 - cloud_fraction * 0.7,
            "surface_flux": 0,
            "net_absorbed": 0,
            "status": "NIGHT",
        }

    elev_rad = math.radians(elev_deg)

    # Top-of-atmosphere flux
    toa_flux = SOLAR_CONSTANT * math.sin(elev_rad)

    # Atmospheric transmittance depends on air mass
    air_mass = 1.0 / max(math.sin(elev_rad), 0.01)
    atm_transmittance = ATMOSPHERIC_TRANSMITTANCE ** min(air_mass, 38)

    # Cloud effect: clouds reduce solar by ~70% of their coverage
    cloud_transmittance = 1.0 - cloud_fraction * 0.70

    # Surface flux
    surface_flux = toa_flux * atm_transmittance * cloud_transmittance

    # Net absorbed (after albedo reflection)
    net_absorbed = surface_flux * (1.0 - albedo)

    return {
        "solar_elevation_deg": round(elev_deg, 1),
        "toa_flux": round(toa_flux, 1),
        "atm_transmittance": round(atm_transmittance, 3),
        "cloud_fraction": round(cloud_fraction, 2),
        "cloud_transmittance": round(cloud_transmittance, 3),
        "surface_flux": round(surface_flux, 1),
        "albedo": albedo,
        "net_absorbed": round(net_absorbed, 1),
        "status": "DAY" if elev_deg > 5 else "TWILIGHT",
    }


def longwave_emission(temp_f, emissivity=EMISSIVITY_URBAN):
    """Calculate outgoing longwave radiation (Stefan-Boltzmann).

    Returns W/m² radiated away.
    """
    temp_k = (temp_f - 32) * 5 / 9 + 273.15
    flux = emissivity * STEFAN_BOLTZMANN * temp_k ** 4
    return round(flux, 1)


def net_radiation(shortwave_net, temp_f, emissivity=EMISSIVITY_URBAN):
    """Net radiation = incoming solar - outgoing longwave.

    Positive = surface is heating.
    Negative = surface is cooling.
    """
    lw_out = longwave_emission(temp_f, emissivity)
    # Approximate downwelling longwave from atmosphere (~80% of surface emission)
    lw_down = lw_out * 0.80
    net = shortwave_net - (lw_out - lw_down)
    return {
        "shortwave_in": shortwave_net,
        "longwave_out": lw_out,
        "longwave_down": round(lw_down, 1),
        "net_radiation": round(net, 1),
        "status": "HEATING" if net > 0 else "COOLING",
    }


def heating_rate(net_rad_wm2, mixing_height=MIXING_HEIGHT):
    """Convert net radiation to temperature change rate.

    dT/dt = Q / (ρ × Cp × h)

    Where:
        Q = net radiation (W/m²)
        ρ = air density (kg/m³)
        Cp = specific heat of air (J/kg/K)
        h = mixing height (m)

    Returns °F per hour.
    """
    dt_per_sec = net_rad_wm2 / (AIR_DENSITY * SPECIFIC_HEAT_AIR * mixing_height)
    dt_per_hour_c = dt_per_sec * 3600
    dt_per_hour_f = dt_per_hour_c * 9 / 5
    return round(dt_per_hour_f, 2)


# ══════════════════════════════════════════════════════════════════
# SECTOR B: ADVECTION CALCULATOR (Wind Transport Engine)
# ══════════════════════════════════════════════════════════════════

def advection_cooling(wind_speed_kt, upstream_temp_f, local_temp_f, distance_km=100):
    """Calculate temperature change from wind advection.

    dT/dt = -u × (dT/dx)

    Parameters:
        wind_speed_kt: Wind speed in knots
        upstream_temp_f: Temperature upwind
        local_temp_f: Current local temperature
        distance_km: Distance to upstream air mass

    Returns dict with advection components.
    """
    # Convert knots to m/s
    wind_ms = wind_speed_kt * 0.514444

    # Temperature gradient (°C per meter)
    dt_c = (upstream_temp_f - local_temp_f) * 5 / 9  # delta in Celsius
    dx_m = distance_km * 1000

    if dx_m == 0 or wind_ms == 0:
        return {
            "advection_rate_f_hr": 0,
            "wind_ms": round(wind_ms, 1),
            "temp_gradient_c_km": 0,
            "effect": "NONE",
        }

    # Advection rate: dT/dt = -u × (dT/dx)
    dt_dx = dt_c / dx_m  # °C per meter
    advection_c_per_s = -wind_ms * dt_dx
    advection_c_per_hr = advection_c_per_s * 3600
    advection_f_per_hr = advection_c_per_hr * 9 / 5

    effect = "WARMING" if advection_f_per_hr > 0.2 else "COOLING" if advection_f_per_hr < -0.2 else "NEUTRAL"

    return {
        "advection_rate_f_hr": round(advection_f_per_hr, 2),
        "wind_speed_kt": wind_speed_kt,
        "wind_ms": round(wind_ms, 1),
        "upstream_temp_f": upstream_temp_f,
        "local_temp_f": local_temp_f,
        "temp_gradient_c_km": round(dt_c / distance_km, 3),
        "effect": effect,
    }


def lake_effect_modifier(wind_dir, wind_speed_kt, lake_temp_c, air_temp_c, city_key):
    """Calculate lake effect temperature modification for lakeside cities.

    The lake acts as a thermal buffer:
    - In winter: lake warms air (lake > air → warming)
    - In summer: lake cools air (lake < air → cooling)
    """
    if city_key == "chicago":
        # Lake Michigan is to the east (45°-135°)
        lake_sector = (45, 135)
    elif city_key == "toronto":
        # Lake Ontario is to the south (135°-225°)
        lake_sector = (135, 225)
    else:
        return {"lake_effect_f_hr": 0, "active": False}

    # Check if wind is from the lake
    try:
        wind_dir = float(wind_dir)
    except (TypeError, ValueError):
        return {"lake_effect_f_hr": 0, "active": False, "reason": "Invalid wind dir"}
    from_lake = lake_sector[0] <= wind_dir <= lake_sector[1]

    if not from_lake or wind_speed_kt < 5:
        return {"lake_effect_f_hr": 0, "active": False, "reason": "Wind not from lake"}

    # Temperature difference drives the effect
    delta_c = lake_temp_c - air_temp_c
    # Lake effect is typically 1-3°F total modification, spread over hours
    # Intensity: bounded adjustment based on delta and wind fetch
    wind_ms = wind_speed_kt * 0.514444
    # Max lake effect is ~0.5°F/hr even in strong conditions
    raw_effect_c_hr = wind_ms * delta_c / 500.0  # °C/hr (much gentler)
    effect_c_hr = max(-0.3, min(0.3, raw_effect_c_hr))  # Cap at ±0.3°C/hr
    effect_f_hr = effect_c_hr * 9 / 5

    return {
        "lake_effect_f_hr": round(effect_f_hr, 2),
        "active": True,
        "lake_temp_c": lake_temp_c,
        "delta_c": round(delta_c, 1),
        "from_lake": from_lake,
    }


# ══════════════════════════════════════════════════════════════════
# SECTOR C: ENSEMBLE PROBABILITY DENSITY (Bayesian Synthesis)
# ══════════════════════════════════════════════════════════════════

def fetch_ensemble_forecasts(lat, lon):
    """Fetch multi-model ensemble forecasts from Open-Meteo.

    Uses GFS, ECMWF, ICON, GEM, JMA models.
    Returns temperature distributions per model.
    Cached for 30 min to avoid hitting Open-Meteo daily rate limit.
    """
    import time
    cache_key = f"{round(lat, 2)},{round(lon, 2)}"
    cache = _load_ensemble_cache()
    cached = cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _ENSEMBLE_TTL:
        return cached["data"]

    models = {}

    model_configs = {
        "gfs": {"url": "https://api.open-meteo.com/v1/gfs", "name": "GFS (NCEP)"},
        "ecmwf": {"url": "https://api.open-meteo.com/v1/ecmwf", "name": "ECMWF (IFS)"},
        "icon": {"url": "https://api.open-meteo.com/v1/dwd-icon", "name": "ICON (DWD)"},
        "gem": {"url": "https://api.open-meteo.com/v1/gem", "name": "GEM (Canada)"},
        "jma": {"url": "https://api.open-meteo.com/v1/jma", "name": "JMA (Japan)"},
    }

    for model_key, config in model_configs.items():
        try:
            resp = requests.get(config["url"], params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "hourly": "temperature_2m,cloud_cover,wind_speed_10m,wind_direction_10m,relative_humidity_2m,surface_pressure",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "kn",
                "timezone": "auto",
                "forecast_days": 3,
            }, timeout=12)

            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                hourly = data.get("hourly", {})
                models[model_key] = {
                    "name": config["name"],
                    "temp_max_f": daily.get("temperature_2m_max", []),
                    "temp_min_f": daily.get("temperature_2m_min", []),
                    "dates": daily.get("time", []),
                    "hourly_temp_f": hourly.get("temperature_2m", []),
                    "hourly_cloud": hourly.get("cloud_cover", []),
                    "hourly_wind_kt": hourly.get("wind_speed_10m", []),
                    "hourly_wind_dir": hourly.get("wind_direction_10m", []),
                    "hourly_humidity": hourly.get("relative_humidity_2m", []),
                    "hourly_pressure": hourly.get("surface_pressure", []),
                    "hourly_times": hourly.get("time", []),
                }
        except Exception:
            continue

    if models:
        cache[cache_key] = {"ts": time.time(), "data": models}
        _save_ensemble_cache(cache)

    return models


def bayesian_ensemble_synthesis(models, day_index=0):
    """Synthesize multi-model ensemble into Bayesian probability distribution.

    For each model:
      - Extract max temp forecast
      - Assign weight based on historical skill score
      - Build Gaussian PDF
      - Multiply PDFs (Bayesian update) and normalize

    Returns synthesized mean, std, and per-model breakdown.
    """
    forecasts = []
    for model_key, data in models.items():
        temps = data.get("temp_max_f", [])
        if len(temps) > day_index and temps[day_index] is not None:
            weight = MODEL_WEIGHTS.get(model_key, 0.1)
            forecasts.append({
                "model": model_key,
                "name": data.get("name", model_key),
                "temp_max_f": temps[day_index],
                "weight": weight,
                # Model-specific spread estimates
                "spread_f": _model_spread(model_key),
            })

    if not forecasts:
        return None

    # Bayesian synthesis: weighted precision combination
    # Combined precision = sum of (weight / variance)
    total_precision = 0
    weighted_mean = 0

    for fc in forecasts:
        variance = fc["spread_f"] ** 2
        precision = fc["weight"] / variance
        total_precision += precision
        weighted_mean += precision * fc["temp_max_f"]

    if total_precision == 0:
        return None

    synth_mean = weighted_mean / total_precision
    synth_var = 1.0 / total_precision
    synth_std = math.sqrt(synth_var)

    # FLOOR: Models are correlated (share initial conditions / physics),
    # so combined σ should never go below realistic forecast error.
    # NWS verification: 24h high temp forecast error ~3-4°F minimum.
    # Wunderground resolution can differ from model grid by ±1-2°F.
    MIN_SYNTH_STD_F = 3.0  # Absolute floor: 3°F (~1.7°C)
    if synth_std < MIN_SYNTH_STD_F:
        synth_std = MIN_SYNTH_STD_F

    # Model spread / agreement
    temps_list = [fc["temp_max_f"] for fc in forecasts]
    model_spread = max(temps_list) - min(temps_list) if temps_list else 0
    agreement = "HIGH" if model_spread < 2 else "MODERATE" if model_spread < 4 else "LOW"

    return {
        "synth_mean_f": round(synth_mean, 1),
        "synth_std_f": round(synth_std, 2),
        "model_spread_f": round(model_spread, 1),
        "agreement": agreement,
        "n_models": len(forecasts),
        "models": forecasts,
    }


def _model_spread(model_key):
    """Estimated forecast spread (standard deviation) per model in °F.

    IMPORTANT: These are real-world forecast errors, not theoretical.
    NWS verification data shows 24h high temp MAE of ~3-5°F for GFS,
    and models share initial conditions so they are NOT independent.
    Previous values (2.2-3.5) were too narrow and inflated probabilities.
    Wunderground station readings can also differ from model grid points.
    """
    spreads = {
        "gfs": 4.0,      # Was 3.0 — GFS 24h high MAE is ~3.5-4.5°F
        "ecmwf": 3.5,    # Was 2.2 — ECMWF is best but still ~3-4°F error
        "icon": 4.0,     # Was 2.8 — ICON similar to GFS for non-European
        "gem": 4.5,      # Was 3.2 — GEM has larger spread
        "jma": 5.0,      # Was 3.5 — JMA less skilled outside Asia
    }
    return spreads.get(model_key, 4.0)


def ensemble_probability(synth_mean, synth_std, threshold_f, direction="over"):
    """Calculate probability of temperature exceeding or falling below threshold.

    Uses Gaussian CDF on the synthesized distribution.
    """
    if synth_std == 0:
        synth_std = 0.1
    z = (threshold_f - synth_mean) / synth_std
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    if direction == "over":
        prob = 1.0 - cdf
    else:
        prob = cdf

    return round(prob, 4)


def sigma_confidence(our_prob, market_price):
    """Express edge confidence in sigma notation.

    sigma = |our_prob - market_prob| / std_deviation_of_estimate
    """
    if market_price <= 0 or market_price >= 1:
        return 0
    # Use market-implied probability as the null hypothesis
    # Standard error of a proportion
    se = math.sqrt(market_price * (1 - market_price))
    if se == 0:
        return 0
    sigma = abs(our_prob - market_price) / se
    return round(sigma, 1)


# ══════════════════════════════════════════════════════════════════
# SECTOR D: SENSOR FUSION (Real-time ground truth)
# ══════════════════════════════════════════════════════════════════

def fetch_metar_enhanced():
    """Fetch METAR with enhanced parsing for sensor fusion.

    Returns temperature, dewpoint, wet-bulb, pressure, wind vector, trend.
    """
    stations = [c["metar"] for c in CITY_PHYSICS.values()]
    # Add upstream stations
    for c in CITY_PHYSICS.values():
        stations.extend(c.get("upstream_stations", []))
    stations = list(set(stations))
    station_str = ",".join(stations)

    metar_data = {}
    try:
        url = f"https://aviationweather.gov/api/data/metar?ids={station_str}&format=json&hours=3"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            for obs in resp.json():
                station = obs.get("icaoId", "")
                temp_c = obs.get("temp")
                if temp_c is None:
                    continue
                dewp_c = obs.get("dewp", temp_c)
                temp_f = temp_c * 9 / 5 + 32
                dewp_f = dewp_c * 9 / 5 + 32

                # Wet bulb temperature approximation (Stull 2011)
                rh = _rh_from_dewpoint(temp_c, dewp_c)
                wetbulb_c = temp_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + \
                            math.atan(temp_c + rh) - math.atan(rh - 1.676331) + \
                            0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035
                wetbulb_f = wetbulb_c * 9 / 5 + 32

                # Pressure in inHg
                altim = obs.get("altim")
                pressure_inhg = altim if altim else None
                pressure_hpa = altim / 0.02953 if altim else obs.get("slp")

                metar_data[station] = {
                    "station": station,
                    "temp_c": round(temp_c, 1),
                    "temp_f": round(temp_f, 1),
                    "dewpoint_c": round(dewp_c, 1),
                    "dewpoint_f": round(dewp_f, 1),
                    "wetbulb_f": round(wetbulb_f, 1),
                    "wetbulb_c": round(wetbulb_c, 1),
                    "rh": round(rh, 1),
                    "wind_speed_kt": obs.get("wspd", 0) or 0,
                    "wind_dir": obs.get("wdir", 0) or 0,
                    "wind_gust_kt": obs.get("wgst"),
                    "pressure_inhg": round(pressure_inhg, 2) if pressure_inhg else None,
                    "pressure_hpa": round(pressure_hpa, 1) if pressure_hpa else None,
                    "visibility_mi": obs.get("visib"),
                    "cloud_cover": _parse_cloud_cover(obs.get("clouds", [])),
                    "flight_cat": obs.get("fltcat", ""),
                    "raw": obs.get("rawOb", ""),
                    "time": obs.get("reportTime", ""),
                }
    except Exception:
        pass

    # Fallback XML endpoint
    if len(metar_data) < 4:
        try:
            url2 = (
                "https://aviationweather.gov/cgi-bin/data/dataserver.php"
                f"?dataSource=metars&requestType=retrieve&format=xml"
                f"&hoursBeforeNow=3&mostRecent=true&stationString={station_str}"
            )
            resp = requests.get(url2, timeout=15)
            if resp.status_code == 200:
                tree = ET.fromstring(resp.content)
                for m in tree.findall(".//METAR"):
                    sid = m.findtext("station_id", "")
                    tc = m.findtext("temp_c")
                    if sid and tc and sid not in metar_data:
                        tc = float(tc)
                        dc = float(m.findtext("dewpoint_c", str(tc)) or tc)
                        tf = tc * 9 / 5 + 32
                        rh = _rh_from_dewpoint(tc, dc)
                        metar_data[sid] = {
                            "station": sid,
                            "temp_c": round(tc, 1),
                            "temp_f": round(tf, 1),
                            "dewpoint_c": round(dc, 1),
                            "dewpoint_f": round(dc * 9 / 5 + 32, 1),
                            "rh": round(rh, 1),
                            "wind_speed_kt": float(m.findtext("wind_speed_kt", "0") or 0),
                            "wind_dir": float(m.findtext("wind_dir_degrees", "0") or 0),
                            "raw": m.findtext("raw_text", ""),
                            "time": m.findtext("observation_time", ""),
                        }
        except Exception:
            pass

    return metar_data


def _rh_from_dewpoint(temp_c, dewpoint_c):
    """Calculate relative humidity from temperature and dewpoint."""
    try:
        a, b = 17.27, 237.7
        gamma_t = (a * temp_c) / (b + temp_c)
        gamma_d = (a * dewpoint_c) / (b + dewpoint_c)
        rh = 100 * math.exp(gamma_d - gamma_t)
        return min(100, max(0, rh))
    except Exception:
        return 50


def _parse_cloud_cover(clouds):
    """Parse METAR cloud layers into a single cloud fraction 0-1."""
    if not clouds:
        return 0.0
    max_cover = 0
    for layer in clouds:
        cover_str = layer.get("cover", "") if isinstance(layer, dict) else str(layer)
        if "OVC" in str(cover_str):
            max_cover = max(max_cover, 1.0)
        elif "BKN" in str(cover_str):
            max_cover = max(max_cover, 0.75)
        elif "SCT" in str(cover_str):
            max_cover = max(max_cover, 0.4)
        elif "FEW" in str(cover_str):
            max_cover = max(max_cover, 0.15)
    return max_cover


def temperature_trend(metar_data, station, hours=2):
    """Calculate temperature trend from multiple observations.

    Note: With only most-recent METAR, we estimate trend from
    current temp vs time of day and sun angle.
    """
    obs = metar_data.get(station, {})
    if not obs:
        return {"trend": "UNKNOWN", "rate_f_hr": 0}

    temp_f = obs.get("temp_f", 0)
    obs_time = obs.get("time", "")

    # Parse observation time
    try:
        if obs_time:
            dt = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
        else:
            dt = datetime.now(timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)

    # Find city for this station
    city_key = None
    for ck, info in CITY_PHYSICS.items():
        if info["metar"] == station:
            city_key = ck
            break

    if city_key:
        info = CITY_PHYSICS[city_key]
        sw = shortwave_flux(info["lat"], info["lon"], dt, obs.get("cloud_cover", 0), info["albedo"])
        net_rad = sw.get("net_absorbed", 0) - longwave_emission(temp_f) * 0.20  # Simplified net
        rate = heating_rate(net_rad)
    else:
        rate = 0

    trend = "RISING" if rate > 0.3 else "FALLING" if rate < -0.3 else "STEADY"

    return {
        "trend": trend,
        "rate_f_hr": rate,
        "observation_time": obs_time,
    }


# ══════════════════════════════════════════════════════════════════
# SECTOR E: IEM ASOS STATION DATA (Wunderground-Matching Layer)
# ══════════════════════════════════════════════════════════════════
#
# Iowa Environmental Mesonet ingests the same ASOS/METAR data that
# Weather Underground uses for market resolution. This gives us the
# ACTUAL station daily high/low — not model forecasts — which is
# what Polymarket resolves against.
#
# Key field: max_dayairtemp[F] = rolling calendar-day maximum
# This matches Wunderground's "Daily High" for resolution.

# IEM network codes per station (US stations use STATE_ASOS,
# international use CC__ASOS where CC is 2-letter country code)
IEM_STATIONS = {
    "nyc":          {"station": "NYC",  "network": "NY_ASOS"},
    "chicago":      {"station": "ORD",  "network": "IL_ASOS"},
    "miami":        {"station": "MIA",  "network": "FL_ASOS"},
    "seattle":      {"station": "SEA",  "network": "WA_ASOS"},
    "dallas":       {"station": "DFW",  "network": "TX_ASOS"},
    "atlanta":      {"station": "ATL",  "network": "GA_ASOS"},
    "london":       {"station": "EGLL", "network": "GB__ASOS"},
    "toronto":      {"station": "CYYZ", "network": "CA_ON_ASOS"},
    "buenos aires": {"station": "SAEZ", "network": "AR__ASOS"},
    "seoul":        {"station": "RKSI", "network": "KR__ASOS"},
    "ankara":       {"station": "LTAC", "network": "TR__ASOS"},
    # Wellington not available on IEM — uses aviationweather.gov fallback
}


def fetch_iem_station_data():
    """Fetch real-time ASOS station data from Iowa Environmental Mesonet.

    This data comes from the same ASOS/METAR network that Weather Underground
    uses for market resolution. The key fields are:
      - airtemp[F]: Current temperature
      - max_dayairtemp[F]: Calendar-day maximum (what WU reports as daily high)
      - min_dayairtemp[F]: Calendar-day minimum

    Returns dict keyed by city_key with station observation data.
    """
    iem_data = {}

    for city_key, info in IEM_STATIONS.items():
        try:
            url = (
                f"https://mesonet.agron.iastate.edu/json/current.py"
                f"?station={info['station']}&network={info['network']}"
            )
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                ob = data.get("last_ob", {})
                if not ob:
                    continue

                temp_f = ob.get("airtemp[F]")
                if temp_f is None:
                    continue

                # Calendar-day max — THIS is what Wunderground reports
                day_max_f = ob.get("max_dayairtemp[F]")
                day_min_f = ob.get("min_dayairtemp[F]")

                # Convert to Celsius for international markets
                temp_c = round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None
                day_max_c = round((day_max_f - 32) * 5 / 9, 1) if day_max_f is not None else None
                day_min_c = round((day_min_f - 32) * 5 / 9, 1) if day_min_f is not None else None

                iem_data[city_key] = {
                    "source": "IEM_ASOS",
                    "station": info["station"],
                    "network": info["network"],
                    "temp_f": round(temp_f, 1),
                    "temp_c": temp_c,
                    "day_max_f": round(day_max_f, 1) if day_max_f is not None else None,
                    "day_max_c": day_max_c,
                    "day_min_f": round(day_min_f, 1) if day_min_f is not None else None,
                    "day_min_c": day_min_c,
                    "dewpoint_f": ob.get("dewpointtemp[F]"),
                    "wind_speed_kt": ob.get("windspeed[kt]"),
                    "wind_dir": ob.get("winddirection[deg]"),
                    "altimeter_inhg": ob.get("altimeter[in]"),
                    "mslp_mb": ob.get("mslp[mb]"),
                    "visibility_mi": ob.get("visibility[mile]"),
                    "sky_cover": ob.get("skycover[code]", []),
                    "raw_metar": ob.get("raw", ""),
                    "utc_valid": ob.get("utc_valid", ""),
                    "local_valid": ob.get("local_valid", ""),
                }
        except Exception:
            continue

    return iem_data


def compare_forecast_vs_station(ensemble_max_f, iem_day_max_f, city_key):
    """Compare our ensemble forecast against actual station daily max.

    IMPORTANT: The daily max from IEM is a RUNNING maximum that accumulates
    throughout the day. Before peak heating (~2-3 PM local), the station max
    will be lower than the actual daily high. Only AFTER peak heating does
    the comparison become meaningful.

    Returns analysis dict with:
      - delta: how far off our forecast is from station reality
      - accuracy_grade: A/B/C/D/F based on error magnitude
      - wunderground_likely: estimated WU resolution value
      - day_phase: PRE_PEAK / POST_PEAK / NIGHT
    """
    if ensemble_max_f is None or iem_day_max_f is None:
        return {"status": "NO_DATA"}

    # Determine local time to know if daily max is complete
    now = datetime.now(timezone.utc)
    tz_offset = CITY_PHYSICS.get(city_key, {}).get("tz_offset", 0)
    local_hour = now.hour + now.minute / 60 + tz_offset
    if local_hour < 0:
        local_hour += 24
    elif local_hour >= 24:
        local_hour -= 24

    # Day phase matters for interpretation
    if local_hour < 7 or local_hour >= 20:
        day_phase = "NIGHT"
        phase_note = "Day max incomplete (nighttime) — station will rise"
    elif local_hour < 15:
        day_phase = "PRE_PEAK"
        phase_note = "Day max still building (before 3 PM local)"
    else:
        day_phase = "POST_PEAK"
        phase_note = "Day max likely reached — comparison valid"

    delta_f = round(ensemble_max_f - iem_day_max_f, 1)
    abs_delta = abs(delta_f)

    # Only grade accuracy when we have complete data (POST_PEAK)
    if day_phase == "POST_PEAK":
        if abs_delta <= 1.5:
            grade = "A"
        elif abs_delta <= 3.0:
            grade = "B"
        elif abs_delta <= 5.0:
            grade = "C"
        elif abs_delta <= 8.0:
            grade = "D"
        else:
            grade = "F"
    elif day_phase == "PRE_PEAK":
        # Station max is still building — if our forecast is above station, OK
        # Only flag as bad if station already EXCEEDS our forecast
        if iem_day_max_f > ensemble_max_f + 2:
            grade = "F"  # Station already exceeded our forecast!
        elif abs_delta <= 3:
            grade = "A"  # Station is close and still building — good
        else:
            grade = "B"  # Still early — can't grade yet
    else:
        # Night — station max from yesterday or very early. Can't grade.
        grade = "--"

    # Wunderground typically reports the ASOS daily max
    # rounded to nearest integer °F (US) or °C (intl)
    is_celsius = city_key in ("london", "toronto", "buenos aires", "seoul", "ankara", "wellington")

    if is_celsius:
        wu_likely_c = round((iem_day_max_f - 32) * 5 / 9)
        wu_likely_f = round(wu_likely_c * 9 / 5 + 32, 1)
    else:
        wu_likely_f = round(iem_day_max_f)
        wu_likely_c = round((wu_likely_f - 32) * 5 / 9, 1)

    return {
        "status": "OK",
        "ensemble_max_f": ensemble_max_f,
        "station_day_max_f": iem_day_max_f,
        "delta_f": delta_f,
        "abs_delta_f": abs_delta,
        "accuracy_grade": grade,
        "wunderground_likely_f": wu_likely_f,
        "wunderground_likely_c": wu_likely_c,
        "direction": "OVER" if delta_f > 0 else "UNDER" if delta_f < 0 else "MATCH",
        "day_phase": day_phase,
        "local_hour": round(local_hour, 1),
        "phase_note": phase_note,
    }


# ══════════════════════════════════════════════════════════════════
# MASTER FUNCTION: Build Thermodynamic Alpha
# ══════════════════════════════════════════════════════════════════

def get_thermodynamic_data():
    """Build the complete Thermodynamic Alpha Terminal data payload.

    Combines all four sectors:
      A: Radiative Transfer
      B: Advection
      C: Ensemble Synthesis
      D: Sensor Fusion
    """
    now = datetime.now(timezone.utc)

    # ── Fetch real-time METAR observations ──
    metar_data = fetch_metar_enhanced()

    # ── Fetch IEM ASOS station data (Wunderground-matching layer) ──
    iem_data = fetch_iem_station_data()

    # ── Load weather predictions from multi_predictor ──
    weather_preds = []
    pred_path = LOG_DIR / "all_predictions.json"
    if pred_path.exists():
        try:
            data = json.load(open(pred_path))
            weather_preds = [p for p in data.get("predictions", [])
                            if p.get("category") == "weather"]
        except Exception:
            pass

    # ── Load agent_23 data ──
    agent_data = {}
    agent_path = LOG_DIR / "signals_agent_23.json"
    if agent_path.exists():
        try:
            agent_data = json.load(open(agent_path))
        except Exception:
            pass

    # ── Build per-city thermodynamic analysis ──
    city_analyses = {}

    for city_key, phys in CITY_PHYSICS.items():
        obs = metar_data.get(phys["metar"], {})
        current_temp_f = obs.get("temp_f")
        cloud_frac = obs.get("cloud_cover", 0.2)
        wind_kt = obs.get("wind_speed_kt", 0)
        wind_dir = obs.get("wind_dir", 0)

        # ── SECTOR A: Radiative Transfer ──
        sw = shortwave_flux(phys["lat"], phys["lon"], now, cloud_frac, phys["albedo"])
        net_abs = sw.get("net_absorbed", 0)

        if current_temp_f is not None:
            net_rad = net_radiation(net_abs, current_temp_f)
            rate = heating_rate(net_rad["net_radiation"], MIXING_HEIGHT * phys.get("thermal_mass", 1.0))
            lw = longwave_emission(current_temp_f)
        else:
            net_rad = {"net_radiation": 0, "status": "NO DATA"}
            rate = 0
            lw = 0

        # Solar noon for this city
        s_noon = solar_noon_utc(phys["lon"])
        s_noon_str = f"{int(s_noon)}:{int((s_noon % 1) * 60):02d} UTC"

        # ── SECTOR B: Advection ──
        advection = {"advection_rate_f_hr": 0, "effect": "NO DATA"}
        upstream_temp_f = None

        # Check upstream stations
        for us_station in phys.get("upstream_stations", []):
            us_obs = metar_data.get(us_station, {})
            if us_obs.get("temp_f") is not None:
                upstream_temp_f = us_obs["temp_f"]
                break

        if upstream_temp_f is not None and current_temp_f is not None and wind_kt > 2:
            advection = advection_cooling(wind_kt, upstream_temp_f, current_temp_f)

        # Lake effect
        lake_eff = {"active": False, "lake_effect_f_hr": 0}
        if phys.get("lake_effect") and current_temp_f is not None:
            lake_temp_c = phys.get("lake_temp_c", 2.0)
            air_temp_c = (current_temp_f - 32) * 5 / 9
            lake_eff = lake_effect_modifier(wind_dir, wind_kt, lake_temp_c, air_temp_c, city_key)

        # ── NET TEMPERATURE CHANGE ──
        net_rate = rate + advection.get("advection_rate_f_hr", 0) + lake_eff.get("lake_effect_f_hr", 0)

        # Project max temperature
        # NOTE: The radiative model only works during daytime when solar heating
        # is active. At nighttime, net_rate is negative (cooling), so extrapolating
        # it forward gives absurdly low projected highs. We flag this and will
        # correct it after ensemble data is available (see below).
        is_nighttime = sw.get("solar_elevation_deg", sw.get("solar_elevation", 0)) < 0
        if current_temp_f is not None:
            local_hour = now.hour + now.minute / 60 + phys["tz_offset"]
            if local_hour < 0:
                local_hour += 24
            elif local_hour >= 24:
                local_hour -= 24
            hours_to_peak = max(0, 14.5 - local_hour)  # Peak usually ~2:30 PM local

            if is_nighttime and net_rate < 0:
                # At night with cooling rate: the projected max will come from
                # tomorrow's solar heating, not from current cooling.
                # Use current temp as placeholder — ensemble will override below.
                projected_max_f = current_temp_f
                physics_is_nighttime = True
            else:
                projected_max_f = current_temp_f + net_rate * min(hours_to_peak, 6)
                physics_is_nighttime = False
        else:
            projected_max_f = None
            physics_is_nighttime = is_nighttime

        # ── City predictions from market ──
        city_preds = [p for p in weather_preds
                      if p.get("details", {}).get("city") == city_key]

        # Arbitrage status
        n_positive_ev = sum(1 for p in city_preds if p.get("edge", 0) > 0.02)
        best_edge = max([abs(p.get("edge", 0)) for p in city_preds], default=0)

        if n_positive_ev >= 3:
            arb_status = "BUYING"
            arb_icon = "buy"
        elif n_positive_ev >= 1:
            arb_status = "WATCHING"
            arb_icon = "watch"
        elif best_edge < 0.02:
            arb_status = "EFFICIENT"
            arb_icon = "efficient"
        else:
            arb_status = "HOLD"
            arb_icon = "hold"

        city_analyses[city_key] = {
            "city_name": CITY_PHYSICS[city_key].get("name", city_key),
            "metar_station": phys["metar"],

            # Sector A: Radiative
            "solar": sw,
            "longwave_out": lw,
            "net_radiation": net_rad,
            "heating_rate_f_hr": rate,
            "solar_noon_utc": s_noon_str,

            # Sector B: Advection
            "advection": advection,
            "lake_effect": lake_eff,
            "upstream_temp_f": upstream_temp_f,

            # Net physics
            "net_rate_f_hr": round(net_rate, 2),
            "projected_max_f": round(projected_max_f, 1) if projected_max_f else None,
            "physics_is_nighttime": physics_is_nighttime,
            "projected_max_source": "PHYSICS",  # May be overridden by ensemble

            # Sensor fusion
            "current_temp_f": current_temp_f,
            "current_temp_c": obs.get("temp_c"),
            "dewpoint_f": obs.get("dewpoint_f"),
            "wetbulb_f": obs.get("wetbulb_f"),
            "pressure_inhg": obs.get("pressure_inhg"),
            "pressure_hpa": obs.get("pressure_hpa"),
            "wind_speed_kt": wind_kt,
            "wind_dir": wind_dir,
            "wind_gust_kt": obs.get("wind_gust_kt"),
            "cloud_cover": cloud_frac,
            "rh": obs.get("rh"),
            "metar_raw": obs.get("raw", ""),
            "metar_time": obs.get("time", ""),

            # Market data
            "n_markets": len(city_preds),
            "n_positive_ev": n_positive_ev,
            "best_edge": round(best_edge, 3),
            "arb_status": arb_status,
            "arb_icon": arb_icon,

            # Trend
            "trend": "RISING" if net_rate > 0.3 else "FALLING" if net_rate < -0.3 else "STEADY",
        }

    # ── SECTOR C: Ensemble Synthesis (batch all cities) ──
    ensemble_data = {}
    tomorrow_forecasts = {}
    tomorrow_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    for city_key, phys in CITY_PHYSICS.items():
        try:
            models = fetch_ensemble_forecasts(phys["lat"], phys["lon"])
            if models:
                synth = bayesian_ensemble_synthesis(models)
                if synth:
                    # Get current hourly data for cloud/wind (from best model)
                    best_model = models.get("ecmwf") or models.get("gfs") or next(iter(models.values()), {})
                    current_hour_idx = now.hour
                    hourly_cloud = best_model.get("hourly_cloud", [])
                    current_cloud = hourly_cloud[current_hour_idx] if len(hourly_cloud) > current_hour_idx else None

                    synth["current_cloud_pct"] = current_cloud
                    ensemble_data[city_key] = synth

                    # Update city analysis with ensemble data
                    if city_key in city_analyses:
                        ca = city_analyses[city_key]
                        ca["ensemble"] = synth

                        # Physics vs Ensemble comparison & correction
                        if synth.get("synth_mean_f"):
                            ensemble_max = synth["synth_mean_f"]
                            physics_max = ca.get("projected_max_f")
                            physics_nighttime = ca.get("physics_is_nighttime", False)

                            # CRITICAL FIX: If physics model was running at night,
                            # the projected max is wrong (based on cooling rate).
                            # Use the ensemble (Open-Meteo NWP) forecast instead.
                            if physics_nighttime and ensemble_max is not None:
                                ca["projected_max_f"] = round(ensemble_max, 1)
                                ca["projected_max_source"] = "ENSEMBLE (nighttime override)"
                            # Also override if physics max is significantly below ensemble
                            # (more than 5°F below — likely nighttime cooling artifact)
                            elif physics_max is not None and ensemble_max - physics_max > 5:
                                ca["projected_max_f"] = round(ensemble_max, 1)
                                ca["projected_max_source"] = "ENSEMBLE (physics diverged)"
                            else:
                                ca["projected_max_source"] = "PHYSICS"

                            ca["physics_vs_ensemble"] = {
                                "physics_max_f": physics_max,
                                "ensemble_max_f": round(ensemble_max, 1),
                                "delta_f": round((physics_max or 0) - ensemble_max, 1),
                                "agreement": "AGREE" if physics_max and abs(physics_max - ensemble_max) < 4 else "DIVERGE",
                                "source_used": ca.get("projected_max_source", "PHYSICS"),
                            }

                # ── TOMORROW's forecast from same models (day_index=1, no extra API calls) ──
                synth_t = bayesian_ensemble_synthesis(models, day_index=1)
                if synth_t:
                    # Tomorrow has wider uncertainty floor
                    if synth_t["synth_std_f"] < 4.0:
                        synth_t["synth_std_f"] = 4.0

                    model_temps = []
                    for mk, md in models.items():
                        temps = md.get("temp_max_f", [])
                        if len(temps) > 1 and temps[1] is not None:
                            model_temps.append({
                                "model": mk,
                                "name": md.get("name", mk),
                                "temp_max_f": round(temps[1], 1),
                            })

                    is_celsius = phys.get("unit") == "C"
                    mean_f = synth_t["synth_mean_f"]
                    mean_c = round((mean_f - 32) * 5 / 9, 1)
                    std_f = synth_t["synth_std_f"]

                    tomorrow_forecasts[city_key] = {
                        "city_name": phys.get("name", city_key.title()),
                        "date": tomorrow_date,
                        "synth_mean_f": round(mean_f, 1),
                        "synth_mean_c": mean_c,
                        "synth_std_f": round(std_f, 1),
                        "model_spread_f": synth_t.get("model_spread_f", 0),
                        "agreement": synth_t.get("agreement", "UNKNOWN"),
                        "models": model_temps,
                        "unit": "C" if is_celsius else "F",
                        "display_temp": f"{mean_c}\u00b0C" if is_celsius else f"{round(mean_f)}\u00b0F",
                        "range_low_f": round(mean_f - std_f, 1),
                        "range_high_f": round(mean_f + std_f, 1),
                        "range_low_c": round((mean_f - std_f - 32) * 5 / 9, 1),
                        "range_high_c": round((mean_f + std_f - 32) * 5 / 9, 1),
                    }

        except Exception:
            continue

    # ── SECTOR E: Inject IEM station data (Wunderground-matching) ──
    for city_key in city_analyses:
        ca = city_analyses[city_key]
        iem = iem_data.get(city_key, {})

        if iem:
            ca["iem_station"] = iem
            ca["iem_day_max_f"] = iem.get("day_max_f")
            ca["iem_day_max_c"] = iem.get("day_max_c")

            # Compare our ensemble forecast vs actual station reading
            ens_max = ca.get("ensemble", {}).get("synth_mean_f")
            station_max = iem.get("day_max_f")
            ca["forecast_vs_station"] = compare_forecast_vs_station(
                ens_max, station_max, city_key
            )

            # If station is already reporting a day max higher than our
            # ensemble forecast, that's a CRITICAL signal — the market
            # will likely resolve higher than we predicted.
            if ens_max and station_max and station_max > ens_max + 2:
                ca["station_warning"] = (
                    f"STATION ALREADY {round(station_max,1)}°F vs forecast "
                    f"{round(ens_max,1)}°F — market may resolve HIGHER"
                )
        else:
            ca["iem_station"] = None
            ca["iem_day_max_f"] = None
            ca["iem_day_max_c"] = None
            ca["forecast_vs_station"] = {"status": "NO_IEM_DATA"}

    # ── Build market arbitrage table with sigma confidence ──
    arb_table = []
    for pred in weather_preds:
        if not pred.get("edge"):
            continue
        details = pred.get("details", {})
        city = details.get("city", "")
        ca = city_analyses.get(city, {})
        ens = ca.get("ensemble", {})

        market_price = pred["market_price"]
        our_prob = pred["our_prob"]
        edge = pred["edge"]

        # Calculate sigma confidence
        sig = sigma_confidence(our_prob, market_price)

        # Minimum edge filter: skip picks with < 3% edge
        # (after recalculation with wider sigmas, these are noise)
        if abs(edge) < 0.03:
            continue

        # EV and Kelly
        ev = our_prob * (1.0 - market_price) - (1.0 - our_prob) * market_price
        ev_pct = ev / market_price * 100 if market_price > 0 else 0

        # Skip negative EV picks entirely
        if ev_pct <= 0:
            continue

        if 0 < market_price < 1:
            b = (1.0 - market_price) / market_price
            kelly_f = (b * our_prob - (1 - our_prob)) / b
            half_kelly = max(0, kelly_f * 0.5)
        else:
            half_kelly = 0

        bet_size = min(half_kelly * 5000, 2000)

        # Position size caps for high-risk stations
        # Buenos Aires SAEZ has known anomaly risk (station spikes +2-5°C)
        if city == "buenos aires" and market_price > 0.40:
            bet_size = min(bet_size, 500)  # Cap at $500 for expensive BA bets
        # Expensive bets (>80c) have large downside — cap position
        if market_price > 0.80:
            bet_size = min(bet_size, 800)  # Cap at $800 for very expensive shares
        potential_pnl = bet_size * (1.0 / market_price - 1) if market_price > 0 else 0

        arb_table.append({
            "city": city,
            "city_name": CITY_PHYSICS.get(city, {}).get("name", city) if city in CITY_PHYSICS else city,
            "title": pred.get("title", ""),
            "bracket": details.get("bracket", ""),
            "market_price": round(market_price, 3),
            "our_prob": round(our_prob, 3),
            "edge": round(edge, 3),
            "sigma": sig,
            "ev_pct": round(ev_pct, 1),
            "half_kelly": round(half_kelly, 4),
            "bet_size": round(bet_size, 2),
            "potential_pnl": round(potential_pnl, 2),
            "forecast_temp": details.get("forecast_temp"),
            "physics_max": ca.get("projected_max_f"),
            "ensemble_mean": ens.get("synth_mean_f"),
            "iem_day_max_f": ca.get("iem_day_max_f"),
            "forecast_vs_station": ca.get("forecast_vs_station", {}).get("accuracy_grade"),
            "wu_likely_f": ca.get("forecast_vs_station", {}).get("wunderground_likely_f"),
            "station_warning": ca.get("station_warning"),
            "pick": pred.get("pick", ""),
        })

    arb_table.sort(key=lambda x: abs(x.get("sigma", 0)), reverse=True)

    # ── TWAP execution plan ──
    positive_ev_arbs = [a for a in arb_table if a["ev_pct"] > 0]
    total_bet = sum(a["bet_size"] for a in positive_ev_arbs)
    total_expected_pnl = sum(a["bet_size"] * abs(a["edge"]) for a in positive_ev_arbs)

    # TWAP: split into $200 chunks every 45s
    twap_chunks = max(1, int(total_bet / 200))
    twap_duration_min = twap_chunks * 0.75  # 45 seconds each

    execution_plan = {
        "total_bet": round(total_bet, 2),
        "total_expected_pnl": round(total_expected_pnl, 2),
        "n_positions": len(positive_ev_arbs),
        "twap_chunk_size": 200,
        "twap_interval_sec": 45,
        "twap_total_chunks": twap_chunks,
        "twap_duration_min": round(twap_duration_min, 1),
        "strategy": "ICEBERG_TWAP",
    }

    # ── Agent 23 stats ──
    agent_summary = agent_data.get("summary", {})

    return {
        "timestamp": now.isoformat(),
        "capital": 50000,
        "latency_ms": 12,  # Simulated
        "active_nodes": sum(1 for ca in city_analyses.values() if ca.get("current_temp_f") is not None),
        "cities": city_analyses,
        "ensemble": ensemble_data,
        "arb_table": arb_table[:60],
        "execution_plan": execution_plan,
        "agent_23": {
            "pnl": agent_summary.get("pnl", 0),
            "capital": agent_summary.get("capital", 0),
            "open_positions": agent_summary.get("open_positions", 0),
            "win_rate": agent_summary.get("win_rate", 0),
            "sharpe": agent_summary.get("sharpe", 0),
            "generation": agent_summary.get("generation", 1),
        },
        "metar_raw": {k: v.get("raw", "") for k, v in metar_data.items()},
        "iem_stations": iem_data,
        "iem_coverage": f"{len(iem_data)}/{len(IEM_STATIONS)} stations",
        "tomorrow": {
            "date": tomorrow_date,
            "forecasts": tomorrow_forecasts,
            "n_cities": len(tomorrow_forecasts),
        },
    }
