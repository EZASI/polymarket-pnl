#!/usr/bin/env python3
"""
POLYMARKET WEATHER STATION — "The Isobar Terminal"
====================================================
Specialized dashboard for weather market arbitrage.

Components:
  1. METAR Ground Truth Fetcher (aviation weather servers)
  2. Temperature Ladder Strategy Engine
  3. Arbitrage Calculator
  4. Bell Curve Visualizer (NOAA vs Market)
  5. Live Radar Map (cities with arb opportunities)

Capital: $50,000 allocation across 12 cities.
"""

import json
import math
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOG_DIR = Path("logs")

# ── City Configurations ──
CITIES = {
    "nyc":           {"name": "New York City", "metar": "KNYC", "lat": 40.71, "lon": -74.01, "unit": "F"},
    "chicago":       {"name": "Chicago",       "metar": "KORD", "lat": 41.88, "lon": -87.63, "unit": "F"},
    "miami":         {"name": "Miami",         "metar": "KMIA", "lat": 25.76, "lon": -80.19, "unit": "F"},
    "seattle":       {"name": "Seattle",       "metar": "KSEA", "lat": 47.61, "lon": -122.33, "unit": "F"},
    "dallas":        {"name": "Dallas",        "metar": "KDFW", "lat": 32.78, "lon": -96.80, "unit": "F"},
    "atlanta":       {"name": "Atlanta",       "metar": "KATL", "lat": 33.75, "lon": -84.39, "unit": "F"},
    "london":        {"name": "London",        "metar": "EGLL", "lat": 51.51, "lon": -0.13, "unit": "C"},
    "toronto":       {"name": "Toronto",       "metar": "CYYZ", "lat": 43.65, "lon": -79.38, "unit": "C"},
    "buenos aires":  {"name": "Buenos Aires",  "metar": "SAEZ", "lat": -34.60, "lon": -58.38, "unit": "C"},
    "seoul":         {"name": "Seoul",         "metar": "RKSI", "lat": 37.57, "lon": 126.98, "unit": "C"},
    "ankara":        {"name": "Ankara",        "metar": "LTAC", "lat": 39.93, "lon": 32.86, "unit": "C"},
    "wellington":    {"name": "Wellington",    "metar": "NZWN", "lat": -41.29, "lon": 174.78, "unit": "C"},
}

CAPITAL = 50_000
MAX_PER_CITY = 5_000
MAX_PER_BRACKET = 2_000
DAILY_TARGET_LOW = 500
DAILY_TARGET_HIGH = 1_250


def fetch_metar():
    """Fetch real-time METAR observations from aviation weather servers."""
    stations = [c["metar"] for c in CITIES.values()]
    station_str = ",".join(stations)
    metar_data = {}

    try:
        url = (
            "https://aviationweather.gov/api/data/metar"
            f"?ids={station_str}&format=json&hours=2"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for obs in data:
                station = obs.get("icaoId", "")
                temp_c = obs.get("temp")
                if temp_c is not None:
                    temp_f = round(temp_c * 9 / 5 + 32, 1)
                    metar_data[station] = {
                        "station": station,
                        "temp_c": round(temp_c, 1),
                        "temp_f": temp_f,
                        "dewpoint_c": obs.get("dewp"),
                        "wind_kt": obs.get("wspd"),
                        "wind_dir": obs.get("wdir"),
                        "visibility_mi": obs.get("visib"),
                        "altimeter": obs.get("altim"),
                        "flight_cat": obs.get("fltcat", ""),
                        "raw": obs.get("rawOb", ""),
                        "time": obs.get("reportTime", ""),
                    }
    except Exception:
        pass

    # Fallback: try XML endpoint
    if len(metar_data) < 3:
        try:
            url2 = (
                "https://aviationweather.gov/cgi-bin/data/dataserver.php"
                f"?dataSource=metars&requestType=retrieve&format=xml"
                f"&hoursBeforeNow=2&mostRecent=true&stationString={station_str}"
            )
            resp = requests.get(url2, timeout=15)
            if resp.status_code == 200:
                tree = ET.fromstring(resp.content)
                for metar_el in tree.findall(".//METAR"):
                    station = metar_el.findtext("station_id", "")
                    temp_c_str = metar_el.findtext("temp_c")
                    if station and temp_c_str:
                        temp_c = float(temp_c_str)
                        if station not in metar_data:
                            metar_data[station] = {
                                "station": station,
                                "temp_c": round(temp_c, 1),
                                "temp_f": round(temp_c * 9 / 5 + 32, 1),
                                "dewpoint_c": float(metar_el.findtext("dewpoint_c", "0") or 0),
                                "wind_kt": float(metar_el.findtext("wind_speed_kt", "0") or 0),
                                "wind_dir": float(metar_el.findtext("wind_dir_degrees", "0") or 0),
                                "raw": metar_el.findtext("raw_text", ""),
                                "time": metar_el.findtext("observation_time", ""),
                            }
        except Exception:
            pass

    return metar_data


def fetch_forecasts():
    """Fetch multi-day forecasts from Open-Meteo for all cities."""
    forecasts = {}
    for city_key, info in CITIES.items():
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": info["lat"],
                    "longitude": info["lon"],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "hourly": "temperature_2m",
                    "temperature_unit": "fahrenheit",
                    "timezone": "auto",
                    "forecast_days": 3,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                hourly = data.get("hourly", {})

                # Get today's max
                temps_max = daily.get("temperature_2m_max", [])
                temps_min = daily.get("temperature_2m_min", [])

                forecasts[city_key] = {
                    "temp_max_f": temps_max,
                    "temp_min_f": temps_min,
                    "precip_mm": daily.get("precipitation_sum", []),
                    "wind_max": daily.get("wind_speed_10m_max", []),
                    "dates": daily.get("time", []),
                    "hourly_temps": hourly.get("temperature_2m", []),
                    "hourly_times": hourly.get("time", []),
                }
        except Exception:
            continue
    return forecasts


def _gauss_cdf(x):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calc_bracket_prob(forecast_temp, low, high, std_dev=3.0):
    """Calculate probability of temperature falling in bracket [low, high]."""
    if low == high:
        z_lo = (low - 0.5 - forecast_temp) / std_dev
        z_hi = (high + 0.5 - forecast_temp) / std_dev
    else:
        z_lo = (low - forecast_temp) / std_dev
        z_hi = (high - forecast_temp) / std_dev
    return max(_gauss_cdf(z_hi) - _gauss_cdf(z_lo), 0.001)


def build_arb_calculator(weather_preds, forecasts, metar_data):
    """Build arbitrage calculations for each market.

    Returns list of arb opportunities with:
      - EV (expected value)
      - Optimal bet size (fractional Kelly)
      - Ladder cost and payout
    """
    arbs = []
    for pred in weather_preds:
        if not pred.get("edge"):
            continue

        details = pred.get("details", {})
        city = details.get("city", "")
        edge = pred["edge"]
        our_prob = pred["our_prob"]
        market_price = pred["market_price"]
        bracket = details.get("bracket", "")

        # Expected Value calculation
        # EV = (prob_win * payout) - (prob_lose * cost)
        # For binary: payout = 1 - cost, cost = market_price
        ev = our_prob * (1.0 - market_price) - (1.0 - our_prob) * market_price
        ev_pct = ev / market_price * 100 if market_price > 0 else 0

        # Half Kelly sizing
        # Kelly fraction = (bp - q) / b
        # b = odds = (1 - market_price) / market_price
        # p = our_prob, q = 1 - our_prob
        if market_price > 0 and market_price < 1:
            b = (1.0 - market_price) / market_price
            kelly_f = (b * our_prob - (1 - our_prob)) / b
            half_kelly = max(0, kelly_f * 0.5)
        else:
            half_kelly = 0

        bet_size = min(half_kelly * MAX_PER_CITY, MAX_PER_BRACKET)
        potential_profit = bet_size * (1.0 / market_price - 1) if market_price > 0 else 0

        # METAR ground truth comparison
        ground_temp = None
        city_info = CITIES.get(city, {})
        metar_station = city_info.get("metar", "")
        if metar_station and metar_station in metar_data:
            ground_temp = metar_data[metar_station].get("temp_f")

        arbs.append({
            "city": city,
            "city_name": city_info.get("name", city),
            "title": pred.get("title", ""),
            "bracket": bracket,
            "market_price": round(market_price, 3),
            "our_prob": round(our_prob, 3),
            "edge": round(edge, 3),
            "ev": round(ev, 4),
            "ev_pct": round(ev_pct, 1),
            "half_kelly": round(half_kelly, 4),
            "bet_size": round(bet_size, 2),
            "potential_profit": round(potential_profit, 2),
            "forecast_temp": details.get("forecast_temp"),
            "ground_temp": ground_temp,
            "volume": details.get("volume", 0),
            "liquidity": details.get("liquidity", 0),
            "pick": pred.get("pick", ""),
            "conditionId": pred.get("conditionId", ""),
        })

    arbs.sort(key=lambda x: abs(x["ev"]), reverse=True)
    return arbs


def build_bell_curves(forecasts, weather_preds):
    """Build bell curve data for each city/date showing NOAA prob vs Market implied."""
    curves = {}

    for city_key, fc in forecasts.items():
        if not fc.get("temp_max_f"):
            continue
        city_info = CITIES.get(city_key, {})
        unit = city_info.get("unit", "F")
        forecast_f = fc["temp_max_f"][0] if fc["temp_max_f"] else None
        if forecast_f is None:
            continue

        if unit == "C":
            forecast = round((forecast_f - 32) * 5 / 9, 1)
            std_dev = 1.7
            step = 1
            spread = 12
        else:
            forecast = round(forecast_f, 1)
            std_dev = 3.0
            step = 1
            spread = 15

        # Build NOAA probability distribution
        noaa_curve = []
        for t in range(int(forecast - spread), int(forecast + spread + 1), step):
            prob = calc_bracket_prob(forecast, t, t + step - 1, std_dev)
            noaa_curve.append({"temp": t, "prob": round(prob, 4)})

        # Build Market implied distribution from predictions
        city_preds = [p for p in weather_preds
                      if p.get("details", {}).get("city") == city_key and p.get("market_price", 0) > 0.01]
        market_curve = []
        for pred in city_preds:
            bracket = pred.get("details", {}).get("bracket", "")
            mp = pred.get("market_price", 0)
            m = re.match(r'(-?\d+)-(-?\d+)', bracket)
            if m:
                mid = (int(m.group(1)) + int(m.group(2))) / 2
                market_curve.append({"temp": mid, "prob": round(mp, 4)})

        market_curve.sort(key=lambda x: x["temp"])

        curves[city_key] = {
            "city_name": city_info.get("name", city_key),
            "forecast": forecast,
            "unit": unit,
            "std_dev": std_dev,
            "noaa": noaa_curve,
            "market": market_curve,
        }

    return curves


def get_weather_station_data():
    """Build the complete data payload for the Weather Station dashboard."""
    now = datetime.now(timezone.utc)

    # Load weather predictions
    weather_preds = []
    pred_path = LOG_DIR / "all_predictions.json"
    if pred_path.exists():
        try:
            data = json.load(open(pred_path))
            weather_preds = [p for p in data.get("predictions", [])
                            if p.get("category") == "weather"]
        except Exception:
            pass

    # Load agent_23 data
    agent_data = {}
    agent_path = LOG_DIR / "signals_agent_23.json"
    if agent_path.exists():
        try:
            agent_data = json.load(open(agent_path))
        except Exception:
            pass

    # Fetch real-time data
    metar_data = fetch_metar()
    forecasts = fetch_forecasts()

    # Build arb calculator
    arbs = build_arb_calculator(weather_preds, forecasts, metar_data)

    # Build bell curves
    curves = build_bell_curves(forecasts, weather_preds)

    # City status for radar map
    city_status = []
    for city_key, info in CITIES.items():
        metar_station = info.get("metar", "")
        obs = metar_data.get(metar_station, {})
        fc = forecasts.get(city_key, {})
        city_arbs = [a for a in arbs if a["city"] == city_key]
        best_edge = max([abs(a["edge"]) for a in city_arbs], default=0)
        total_ev = sum(a["ev"] for a in city_arbs if a["ev"] > 0)

        city_status.append({
            "key": city_key,
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "unit": info.get("unit", "F"),
            "current_temp_f": obs.get("temp_f"),
            "current_temp_c": obs.get("temp_c"),
            "forecast_high_f": fc.get("temp_max_f", [None])[0] if fc.get("temp_max_f") else None,
            "metar_raw": obs.get("raw", ""),
            "metar_time": obs.get("time", ""),
            "n_arbs": len(city_arbs),
            "best_edge": round(best_edge, 3),
            "total_ev": round(total_ev, 4),
            "has_arb": best_edge > 0.02,
        })

    # Portfolio stats
    total_exposure = sum(a["bet_size"] for a in arbs if a["ev"] > 0)
    total_expected_profit = sum(a["bet_size"] * a["ev_pct"] / 100 for a in arbs if a["ev"] > 0)
    n_opportunities = sum(1 for a in arbs if a["ev_pct"] > 5)

    # Agent 23 trading stats
    agent_summary = agent_data.get("summary", {})

    return {
        "timestamp": now.isoformat(),
        "capital": CAPITAL,
        "daily_target_low": DAILY_TARGET_LOW,
        "daily_target_high": DAILY_TARGET_HIGH,
        "portfolio": {
            "total_exposure": round(total_exposure, 2),
            "expected_profit": round(total_expected_profit, 2),
            "n_opportunities": n_opportunities,
            "n_cities_active": sum(1 for c in city_status if c["has_arb"]),
            "n_markets": len(weather_preds),
            "n_with_edge": sum(1 for p in weather_preds if abs(p.get("edge", 0)) > 0.03),
        },
        "agent_23": {
            "pnl": agent_summary.get("pnl", 0),
            "capital": agent_summary.get("capital", 0),
            "open_positions": agent_summary.get("open_positions", 0),
            "win_rate": agent_summary.get("win_rate", 0),
            "sharpe": agent_summary.get("sharpe", 0),
            "generation": agent_summary.get("generation", 1),
        },
        "cities": city_status,
        "arbs": arbs[:50],  # top 50 opportunities
        "curves": curves,
        "metar": {k: v for k, v in metar_data.items()},
        "forecasts_summary": {
            city: {
                "high": fc.get("temp_max_f", [None])[0],
                "low": fc.get("temp_min_f", [None])[0],
                "dates": fc.get("dates", []),
            }
            for city, fc in forecasts.items()
        },
    }


# ════════════════════════════════════════════════════════════════════
# THERMODYNAMIC ALPHA TERMINAL — "The Isobar APA-9000"
# ════════════════════════════════════════════════════════════════════

WEATHER_STATION_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISOBAR: Thermodynamic Alpha Terminal</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SF Mono',Monaco,'Cascadia Code','Fira Code',monospace;background:#04060c;color:#c8d6e5;min-height:100vh}
a{color:inherit;text-decoration:none}

/* ── Header ── */
.top-bar{background:linear-gradient(180deg,#080c16 0%,#04060c 100%);border-bottom:1px solid #0f1a2e;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100;backdrop-filter:blur(16px)}
.brand{display:flex;align-items:center;gap:10px}
.brand-logo{font-size:14px;font-weight:800;letter-spacing:5px;color:#38bdf8;font-family:inherit}
.brand-tag{font-size:8px;color:#0f1a2e;background:#38bdf8;padding:1px 5px;border-radius:3px;font-weight:700;letter-spacing:1px;margin-left:6px}
.brand-sep{width:1px;height:18px;background:#0f1a2e}
.brand-sub{font-size:9px;color:#2d4a6a;letter-spacing:3px}
.nav{display:flex;gap:3px;flex-wrap:wrap}
.nav a{font-size:9px;padding:5px 12px;border-radius:5px;color:#2d4a6a;transition:.2s;font-weight:500;letter-spacing:.5px}
.nav a:hover{color:#c8d6e5;background:rgba(255,255,255,.03)}
.nav a.on{background:rgba(56,189,248,.1);color:#38bdf8;font-weight:600}

/* ── Layout ── */
.wrap{max-width:1500px;margin:0 auto;padding:12px 16px 60px}

/* ── System Banner ── */
.sys-banner{background:#080c16;border:1px solid #0f1a2e;border-radius:6px;padding:6px 12px;margin-bottom:8px;font-size:8px;color:#2d4a6a;display:flex;justify-content:space-between;align-items:center;letter-spacing:1px}
.sys-banner .sys-val{color:#38bdf8;font-weight:700}
.sys-banner .sys-sep{color:#0f1a2e;margin:0 8px}

/* ── Stat Cards ── */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:6px;margin-bottom:10px}
.stat{background:#080c16;border:1px solid #0f1a2e;border-radius:8px;padding:8px 8px;text-align:center}
.stat-label{font-size:7px;color:#2d4a6a;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:3px}
.stat-val{font-size:18px;font-weight:700}
.stat-sub{font-size:7px;color:#1a2538;margin-top:1px;letter-spacing:1px}

/* ── Sections ── */
.section{background:#080c16;border:1px solid #0f1a2e;border-radius:8px;margin-bottom:8px;overflow:hidden}
.sec-hdr{padding:8px 12px;border-bottom:1px solid #0f1a2e;font-size:10px;font-weight:700;display:flex;justify-content:space-between;align-items:center;letter-spacing:1px}
.sec-hdr.sector-a{color:#ffd43b;background:linear-gradient(90deg,rgba(255,212,59,.04),transparent)}
.sec-hdr.sector-b{color:#38bdf8;background:linear-gradient(90deg,rgba(56,189,248,.04),transparent)}
.sec-hdr.sector-c{color:#a78bfa;background:linear-gradient(90deg,rgba(167,139,250,.04),transparent)}
.sec-hdr.sector-d{color:#00d4aa;background:linear-gradient(90deg,rgba(0,212,170,.04),transparent)}
.sec-hdr .meta{font-size:8px;color:#2d4a6a;font-weight:400;letter-spacing:1px}
.sec-body{padding:8px 12px}

/* ── Grid ── */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.grid-2-1{display:grid;grid-template-columns:2fr 1fr;gap:12px}
@media(max-width:900px){.grid-2,.grid-3,.grid-2-1{grid-template-columns:1fr}}

/* ── Physics Readout ── */
.phys-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px}
.phys-card{background:#04060c;border:1px solid #0f1a2e;border-radius:6px;padding:8px 10px;position:relative}
.phys-card.active{border-color:#1a3a5c}
.phys-city{font-size:10px;font-weight:700;color:#38bdf8;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.phys-row{display:flex;justify-content:space-between;padding:2px 0;font-size:9px}
.phys-row .label{color:#2d4a6a}
.phys-row .val{font-weight:600}
.phys-divider{border-top:1px solid #0a0f1c;margin:4px 0}
.phys-action{margin-top:6px;font-size:9px;font-weight:700;padding:4px 8px;border-radius:4px;text-align:center}
.phys-action.buy{background:rgba(0,212,170,.1);color:#00d4aa;border:1px solid rgba(0,212,170,.2)}
.phys-action.sell{background:rgba(255,64,85,.08);color:#ff4055;border:1px solid rgba(255,64,85,.2)}
.phys-action.hold{background:rgba(45,74,106,.1);color:#2d4a6a;border:1px solid #0f1a2e}
.phys-action.watch{background:rgba(255,212,59,.08);color:#ffd43b;border:1px solid rgba(255,212,59,.2)}

/* ── Sensor Table ── */
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:5px 8px;font-size:8px;color:#2d4a6a;letter-spacing:1.5px;text-transform:uppercase;border-bottom:1px solid #0f1a2e}
td{padding:5px 8px;font-size:10px;border-bottom:1px solid #080c16}
tr:hover{background:#0a0f1c}
.ev-pos{color:#00d4aa;font-weight:700}
.ev-neg{color:#ff4055;font-weight:700}

/* ── Bell Curve ── */
.curve-container{height:220px;position:relative}
.curve-select{display:flex;gap:4px;margin-bottom:8px;flex-wrap:wrap}
.curve-btn{font-size:8px;padding:3px 8px;border-radius:3px;border:1px solid #0f1a2e;background:transparent;color:#2d4a6a;cursor:pointer;font-family:inherit;transition:.2s;letter-spacing:.5px}
.curve-btn:hover,.curve-btn.active{border-color:#38bdf8;color:#38bdf8;background:rgba(56,189,248,.06)}

/* ── Ensemble ── */
.ens-bar{display:flex;align-items:center;gap:8px;padding:3px 0}
.ens-name{font-size:9px;color:#2d4a6a;width:80px;text-align:right}
.ens-track{flex:1;height:16px;background:#04060c;border-radius:3px;border:1px solid #0f1a2e;position:relative;overflow:hidden}
.ens-fill{height:100%;border-radius:2px;position:absolute;top:0}
.ens-val{font-size:9px;font-weight:700;width:50px}

/* ── METAR Feed ── */
.metar-feed{font-size:9px;line-height:1.7;max-height:180px;overflow-y:auto}
.metar-line{padding:2px 0;border-bottom:1px solid #080c16}
.metar-station{color:#38bdf8;font-weight:700}
.metar-temp{color:#ffd43b;font-weight:700}

/* ── Ladder ── */
.ladder{display:flex;gap:2px;align-items:flex-end;height:100px;margin:8px 0}

/* ── Colors ── */
.g{color:#00d4aa}.r{color:#ff4055}.y{color:#ffd43b}.b{color:#38bdf8}.d{color:#2d4a6a}.p{color:#a78bfa}.o{color:#ff8c42}
.tag{font-size:7px;padding:2px 5px;border-radius:3px;font-weight:700;letter-spacing:.5px}
.tag-buy{background:rgba(0,212,170,.12);color:#00d4aa}
.tag-sell{background:rgba(255,64,85,.08);color:#ff4055}
.tag-hold{background:rgba(45,74,106,.08);color:#2d4a6a}
.tag-sigma{background:rgba(167,139,250,.1);color:#a78bfa;border:1px solid rgba(167,139,250,.2)}

/* ── TWAP ── */
.twap-bar{height:6px;background:#0f1a2e;border-radius:3px;overflow:hidden;margin:4px 0}
.twap-fill{height:100%;background:linear-gradient(90deg,#38bdf8,#00d4aa);border-radius:3px;transition:width .5s}

/* ── Live Pulse ── */
.live-dot{width:5px;height:5px;border-radius:50%;background:#00d4aa;display:inline-block;margin-right:4px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 4px rgba(0,212,170,.5)}50%{opacity:.2;box-shadow:none}}

/* ── Footer ── */
.foot{text-align:center;padding:16px;font-size:8px;color:#0f1a2e;letter-spacing:2px;margin-top:16px}

/* ── Collapsible Sections ── */
.sec-hdr.collapsible{cursor:pointer;user-select:none}
.sec-hdr.collapsible:hover{filter:brightness(1.15)}
.sec-hdr .chevron{transition:transform .25s ease;display:inline-block;font-size:10px;margin-right:6px;color:#2d4a6a}
.sec-hdr.collapsed .chevron{transform:rotate(-90deg)}
.sec-body-wrap{overflow:hidden;transition:max-height .35s ease,opacity .25s ease;max-height:5000px;opacity:1}
.sec-body-wrap.collapsed{max-height:0;opacity:0}

/* ── Expandable Physics Cards ── */
.phys-extra{overflow:hidden;max-height:0;opacity:0;transition:max-height .3s ease,opacity .2s ease}
.phys-extra.open{max-height:600px;opacity:1}
.phys-toggle{font-size:7px;color:#2d4a6a;cursor:pointer;text-align:center;padding:3px 0;letter-spacing:1px;transition:color .2s}
.phys-toggle:hover{color:#38bdf8}

/* ── Show More Button ── */
.show-more-btn{display:block;width:100%;padding:10px;margin-top:6px;border:1px dashed #1a2538;border-radius:6px;background:rgba(56,189,248,.03);color:#38bdf8;font-size:10px;font-weight:700;text-align:center;cursor:pointer;font-family:inherit;letter-spacing:1px;transition:all .2s}
.show-more-btn:hover{background:rgba(56,189,248,.08);border-color:#38bdf8}
</style></head><body>

<div class="top-bar">
<div class="brand">
  <div class="brand-logo">ISOBAR</div>
  <span class="brand-tag">APA-9000</span>
  <div class="brand-sep"></div>
  <div class="brand-sub">THERMODYNAMIC ALPHA TERMINAL</div>
</div>
<div class="nav">
  <a href="/">Dashboard</a>
  <a href="/command-center">Command</a>
  <a href="/swarm">Swarm</a>
  <a href="/weather" class="on">Weather</a>
  <a href="/picks">Picks</a>
</div>
</div>

<div class="wrap">

<!-- ═══ SYSTEM BANNER ═══ -->
<div class="sys-banner">
  <span><span class="sys-val" id="sb-cap">50,000 USD</span> ALLOCATION</span>
  <span class="sys-sep">│</span>
  <span><span class="sys-val" id="sb-nodes">--</span> ACTIVE NODES</span>
  <span class="sys-sep">│</span>
  <span>LATENCY: <span class="sys-val">12ms</span></span>
  <span class="sys-sep">│</span>
  <span>MODELS: <span class="sys-val" id="sb-models">5</span></span>
  <span class="sys-sep">│</span>
  <span><span class="live-dot"></span><span class="sys-val">LIVE</span></span>
</div>

<!-- ═══ STATS ROW ═══ -->
<div class="stats">
  <div class="stat"><div class="stat-label">Capital</div><div class="stat-val b" id="s-cap">$50,000</div><div class="stat-sub">ALLOCATED</div></div>
  <div class="stat"><div class="stat-label">Exposure</div><div class="stat-val y" id="s-exp">--</div><div class="stat-sub">DEPLOYED</div></div>
  <div class="stat"><div class="stat-label">Expected PnL</div><div class="stat-val g" id="s-pnl">--</div><div class="stat-sub">DAILY EV</div></div>
  <div class="stat"><div class="stat-label">Positions</div><div class="stat-val p" id="s-pos">--</div><div class="stat-sub">POSITIVE EV</div></div>
  <div class="stat"><div class="stat-label">TWAP Queue</div><div class="stat-val b" id="s-twap">--</div><div class="stat-sub" id="s-twap-dur">--</div></div>
  <div class="stat"><div class="stat-label">Agent 23</div><div class="stat-val" id="s-agent">--</div><div class="stat-sub" id="s-aginfo">GEN 1</div></div>
  <div class="stat"><div class="stat-label">IEM ASOS</div><div class="stat-val" id="s-iem" style="color:#e879f9">--</div><div class="stat-sub" id="s-iem-sub">STATIONS</div></div>
  <div class="stat"><div class="stat-label">Clock</div><div class="stat-val d" id="s-time" style="font-size:13px">--</div><div class="stat-sub">UTC</div></div>
</div>

<!-- ═══ TOP PICKS — WHAT TO BET ═══ -->
<style>
.pick-card{background:linear-gradient(135deg,rgba(0,212,170,.04),rgba(56,189,248,.02));border:1px solid rgba(0,212,170,.12);border-radius:8px;padding:8px 12px;margin-bottom:4px;display:grid;grid-template-columns:30px 1fr auto;gap:8px;align-items:center;transition:all .2s}
.pick-card:hover{border-color:rgba(0,212,170,.35);box-shadow:0 0 20px rgba(0,212,170,.06)}
.pick-card.top3{border-color:rgba(0,212,170,.25);background:linear-gradient(135deg,rgba(0,212,170,.06),rgba(56,189,248,.03))}
.pick-rank{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:12px;font-family:'Courier New',monospace}
.pick-info{display:flex;flex-direction:column;gap:2px}
.pick-city{font-size:11px;font-weight:700;color:#c8d6e5;letter-spacing:.5px}
.pick-market{font-size:9px;color:#4a6a8a;line-height:1.4}
.pick-temps{display:flex;gap:12px;align-items:center;margin-top:3px}
.pick-fc{font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:.5px}
.pick-fc.hot{background:rgba(255,100,50,.12);color:#ff8c42;border:1px solid rgba(255,100,50,.2)}
.pick-fc.warm{background:rgba(255,200,50,.1);color:#ffd43b;border:1px solid rgba(255,200,50,.2)}
.pick-fc.cool{background:rgba(56,189,248,.1);color:#38bdf8;border:1px solid rgba(56,189,248,.2)}
.pick-fc.cold{background:rgba(100,150,255,.1);color:#818cf8;border:1px solid rgba(100,150,255,.2)}
.pick-arrow{font-size:12px;color:#2d4a6a}
.pick-threshold{font-size:10px;font-weight:700;color:#c8d6e5;padding:2px 6px;border-radius:3px;background:rgba(200,214,229,.06);border:1px solid rgba(200,214,229,.1)}
.pick-right{display:flex;flex-direction:column;align-items:flex-end;gap:4px;min-width:140px}
.pick-action{padding:5px 14px;border-radius:5px;font-weight:900;font-size:11px;letter-spacing:1px;text-align:center;min-width:100px}
.pick-action.yes{background:linear-gradient(135deg,rgba(0,212,170,.25),rgba(0,212,170,.1));color:#00d4aa;border:1px solid rgba(0,212,170,.4);text-shadow:0 0 10px rgba(0,212,170,.3)}
.pick-action.no{background:linear-gradient(135deg,rgba(255,64,85,.2),rgba(255,64,85,.08));color:#ff4055;border:1px solid rgba(255,64,85,.3);text-shadow:0 0 10px rgba(255,64,85,.2)}
.pick-stats{display:flex;gap:10px;font-size:9px;color:#4a6a8a}
.pick-stats b{color:#c8d6e5}
.pick-prob{font-size:18px;font-weight:900;letter-spacing:-0.5px}
.pick-prob.high{color:#00d4aa;text-shadow:0 0 15px rgba(0,212,170,.4)}
.pick-prob.mid{color:#ffd43b;text-shadow:0 0 15px rgba(255,212,59,.3)}
.pick-prob.low{color:#38bdf8;text-shadow:0 0 15px rgba(56,189,248,.3)}
.pick-date{font-size:8px;color:#2d4a6a;font-weight:600;letter-spacing:1px}
</style>
<div class="section" style="border-color:rgba(0,212,170,.2)">
  <div class="sec-hdr" style="color:#00d4aa;background:linear-gradient(90deg,rgba(0,212,170,.1),rgba(56,189,248,.03),transparent);font-size:13px;border-bottom:1px solid rgba(0,212,170,.15)">
    <span style="display:flex;align-items:center;gap:8px"><span style="font-size:16px">&#127919;</span> TOP PICKS — WHAT TO BET NOW</span>
    <span class="meta" id="picks-count" style="font-size:10px;color:#00d4aa">--</span>
  </div>
  <div class="sec-body">
    <div style="margin-bottom:8px;padding:5px 10px;border-radius:4px;background:rgba(255,180,0,.04);border:1px solid rgba(255,180,0,.1);font-size:8px;color:#8a7530;line-height:1.4">
      &#9888; Model estimates from Open-Meteo. ~3-5&#176; error margin. Not financial advice.
    </div>
    <div id="picks-cards" style="display:flex;flex-direction:column;gap:4px">
      <div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">Loading picks...</div>
    </div>
  </div>
</div>

<!-- ═══ TOMORROW'S FORECAST ═══ -->
<style>
.tmrw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}
.tmrw-card{background:#04060c;border:1px solid rgba(255,140,66,.12);border-radius:8px;padding:10px 12px;position:relative;transition:all .2s}
.tmrw-card:hover{border-color:rgba(255,140,66,.3);box-shadow:0 0 15px rgba(255,140,66,.06)}
.tmrw-city{font-size:10px;font-weight:700;color:#ff8c42;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center}
.tmrw-temp{font-size:22px;font-weight:900;text-align:center;padding:6px 0;letter-spacing:-0.5px}
.tmrw-range{font-size:8px;color:#4a6a8a;text-align:center;margin-bottom:6px}
.tmrw-models{font-size:8px;color:#2d4a6a;line-height:1.6}
.tmrw-bar{height:4px;background:#0f1a2e;border-radius:2px;margin:6px 0;overflow:hidden;position:relative}
.tmrw-bar-fill{height:100%;border-radius:2px;position:absolute;top:0}
.tmrw-agree{font-size:7px;padding:2px 6px;border-radius:3px;font-weight:700;letter-spacing:.5px}
.tmrw-agree.high{background:rgba(0,212,170,.1);color:#00d4aa;border:1px solid rgba(0,212,170,.2)}
.tmrw-agree.moderate{background:rgba(255,212,59,.08);color:#ffd43b;border:1px solid rgba(255,212,59,.15)}
.tmrw-agree.low{background:rgba(255,64,85,.08);color:#ff4055;border:1px solid rgba(255,64,85,.15)}
</style>
<div class="section" style="border-color:rgba(255,140,66,.2)">
  <div class="sec-hdr" style="color:#ff8c42;background:linear-gradient(90deg,rgba(255,140,66,.08),rgba(255,200,50,.03),transparent);font-size:12px;border-bottom:1px solid rgba(255,140,66,.12)">
    <span style="display:flex;align-items:center;gap:8px"><span style="font-size:14px">&#9728;&#65039;</span> TOMORROW'S FORECAST</span>
    <span class="meta" id="tmrw-date" style="font-size:10px;color:#ff8c42">--</span>
  </div>
  <div class="sec-body">
    <div style="margin-bottom:8px;padding:5px 10px;border-radius:4px;background:rgba(255,140,66,.03);border:1px solid rgba(255,140,66,.08);font-size:8px;color:#8a6530;line-height:1.4">
      5-model ensemble (GFS, ECMWF, ICON, GEM, JMA). Tomorrow has wider uncertainty (&#177;4&#176;F floor). Use for pre-positioning.
    </div>
    <div class="tmrw-grid" id="tmrw-grid">
      <div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">Loading tomorrow's forecast...</div>
    </div>
  </div>
</div>

<!-- ═══ SECTOR A: RADIATIVE TRANSFER MODEL (Collapsed by default) ═══ -->
<div class="section" id="sec-sector-a">
  <div class="sec-hdr sector-a collapsible collapsed" onclick="toggleSection('sector-a')"><span class="chevron">&#9660;</span>[ SECTOR A ] RADIATIVE TRANSFER MODEL — Solar Heating Engine<span class="meta">dT/dt = Q/(ρCₚh)</span></div>
  <div class="sec-body-wrap collapsed" id="body-sector-a">
  <div class="sec-body">
    <div class="phys-grid" id="sector-a-grid">Loading radiative data...</div>
  </div>
  </div>
</div>

<!-- ═══ TWO COLUMN: SECTOR B + SECTOR C (Collapsed by default) ═══ -->
<div class="grid-2" id="sec-sector-bc">
  <div class="section">
    <div class="sec-hdr sector-b collapsible collapsed" onclick="toggleSection('sector-b')"><span class="chevron">&#9660;</span>[ SECTOR B ] ADVECTION — Wind Transport<span class="meta">dT/dt = -u(dT/dx)</span></div>
    <div class="sec-body-wrap collapsed" id="body-sector-b">
    <div class="sec-body" id="sector-b-body">Loading advection data...</div>
    </div>
  </div>
  <div class="section">
    <div class="sec-hdr sector-c collapsible collapsed" onclick="toggleSection('sector-c')"><span class="chevron">&#9660;</span>[ SECTOR C ] ENSEMBLE SYNTHESIS — Bayesian<span class="meta">5-MODEL SPREAD</span></div>
    <div class="sec-body-wrap collapsed" id="body-sector-c">
    <div class="sec-body">
      <div class="curve-select" id="ens-select"></div>
      <div id="ens-body">Loading ensemble data...</div>
      <div class="curve-container" style="margin-top:10px"><canvas id="bellCanvas"></canvas></div>
    </div>
    </div>
  </div>
</div>

<!-- ═══ SECTOR D: REMOVED (duplicate of Sector A) ═══ -->

<!-- ═══ TWO COLUMN: LADDER + METAR FEED (Collapsed by default) ═══ -->
<div class="grid-2-1" id="sec-ladder">
  <div class="section">
    <div class="sec-hdr sector-a collapsible collapsed" onclick="toggleSection('ladder')"><span class="chevron">&#9660;</span>TEMPERATURE LADDER — Bracket Pricing<span class="meta" id="ladder-city">SELECT CITY</span></div>
    <div class="sec-body-wrap collapsed" id="body-ladder">
    <div class="sec-body">
      <div class="curve-select" id="ladder-select"></div>
      <div class="ladder" id="ladder-viz"></div>
      <div style="margin-top:16px;font-size:9px;color:#2d4a6a" id="ladder-info"></div>
    </div>
    </div>
  </div>
  <div class="section">
    <div class="sec-hdr sector-d collapsible collapsed" onclick="toggleSection('metar')"><span class="chevron">&#9660;</span>RAW METAR FEED<span class="meta">AVIATION WX</span></div>
    <div class="sec-body-wrap collapsed" id="body-metar">
    <div class="sec-body">
      <div class="metar-feed" id="metar-feed">Awaiting...</div>
    </div>
    </div>
  </div>
</div>

<!-- ═══ OLD TOP PICKS + ARB TABLE REMOVED (moved up + consolidated) ═══ -->

</div>

<!-- ═══ SECTOR E: ZOMBIE HUNTER ═══ -->
<style>
/* ── Zombie Cards ── */
.zc-grid{display:flex;flex-direction:column;gap:6px}
.zc{background:linear-gradient(135deg,rgba(192,132,252,.04),rgba(139,92,246,.02));border:1px solid rgba(192,132,252,.12);border-radius:8px;padding:10px 12px;transition:all .2s;position:relative;overflow:hidden}
.zc:hover{border-color:rgba(192,132,252,.35);box-shadow:0 0 25px rgba(192,132,252,.08)}
.zc.confirmed{border-color:rgba(192,132,252,.25);background:linear-gradient(135deg,rgba(192,132,252,.07),rgba(139,92,246,.03))}
.zc-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
.zc-city{display:flex;align-items:center;gap:8px}
.zc-icon{font-size:18px}
.zc-name{font-size:13px;font-weight:800;color:#e2d6f8;letter-spacing:.3px}
.zc-date{font-size:8px;color:#7c5cad;font-weight:600;letter-spacing:1px;margin-top:1px}
.zc-status{font-size:8px;padding:3px 10px;border-radius:10px;font-weight:700;letter-spacing:.8px}
.zc-status.dead{background:rgba(192,132,252,.15);color:#c084fc;border:1px solid rgba(192,132,252,.3)}
.zc-status.sunset{background:rgba(129,140,248,.12);color:#818cf8;border:1px solid rgba(129,140,248,.25)}
.zc-status.marginal{background:rgba(251,191,36,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.2)}

/* Bracket name — mimicking Polymarket */
.zc-bracket{font-size:12px;font-weight:700;color:#c8d6e5;padding:8px 14px;border-radius:6px;background:rgba(200,214,229,.04);border:1px solid rgba(200,214,229,.08);margin-bottom:10px;display:flex;align-items:center;gap:10px;letter-spacing:.3px}
.zc-bracket .unit-badge{font-size:9px;font-weight:800;padding:2px 6px;border-radius:3px;letter-spacing:.5px}
.zc-bracket .unit-badge.celsius{background:rgba(56,189,248,.12);color:#38bdf8;border:1px solid rgba(56,189,248,.2)}
.zc-bracket .unit-badge.fahrenheit{background:rgba(255,140,66,.12);color:#ff8c42;border:1px solid rgba(255,140,66,.2)}

/* Observed temp + result */
.zc-observed{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:10px;color:#7a8ba0}
.zc-obs-temp{font-size:14px;font-weight:800;color:#38bdf8}
.zc-result{font-size:9px;font-weight:800;padding:3px 8px;border-radius:4px;letter-spacing:.5px}
.zc-result.yes-wins{background:rgba(0,212,170,.12);color:#00d4aa;border:1px solid rgba(0,212,170,.25)}
.zc-result.no-wins{background:rgba(255,64,85,.1);color:#ff4055;border:1px solid rgba(255,64,85,.2)}
.zc-result.uncertain{background:rgba(74,106,138,.1);color:#7a8ba0;border:1px solid rgba(74,106,138,.2)}

/* Polymarket-style price buttons */
.zc-buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.zc-btn{padding:10px 12px;border-radius:8px;text-align:center;font-weight:800;position:relative;transition:all .2s}
.zc-btn .btn-label{font-size:9px;letter-spacing:1px;margin-bottom:3px;opacity:.7}
.zc-btn .btn-price{font-size:22px;font-weight:900;letter-spacing:-0.5px}
.zc-btn .btn-sub{font-size:8px;margin-top:2px;opacity:.6}

/* YES button states */
.zc-btn.yes-dim{background:rgba(0,212,170,.04);border:1px solid rgba(0,212,170,.1);color:#2d5a4a}
.zc-btn.yes-active{background:linear-gradient(135deg,rgba(0,212,170,.2),rgba(0,212,170,.08));border:2px solid rgba(0,212,170,.5);color:#00d4aa;box-shadow:0 0 20px rgba(0,212,170,.15),inset 0 0 20px rgba(0,212,170,.05);animation:zbPulse 3s infinite}
.zc-btn.yes-active .btn-label{color:#00d4aa;opacity:1}
.zc-btn.yes-active::before{content:"\\25B2 ";font-size:10px}
.zc-btn.yes-active::after{content:"";position:absolute;top:-1px;left:-1px;right:-1px;bottom:-1px;border-radius:8px;border:1px solid rgba(0,212,170,.3);pointer-events:none}

/* NO button states */
.zc-btn.no-dim{background:rgba(255,64,85,.03);border:1px solid rgba(255,64,85,.08);color:#5a2d35}
.zc-btn.no-active{background:linear-gradient(135deg,rgba(255,64,85,.18),rgba(255,64,85,.06));border:2px solid rgba(255,64,85,.45);color:#ff4055;box-shadow:0 0 20px rgba(255,64,85,.12),inset 0 0 20px rgba(255,64,85,.04);animation:zbPulse 3s infinite}
.zc-btn.no-active .btn-label{color:#ff4055;opacity:1}
.zc-btn.no-active::before{content:"\\25BC ";font-size:10px}
.zc-btn.no-active::after{content:"";position:absolute;top:-1px;left:-1px;right:-1px;bottom:-1px;border-radius:8px;border:1px solid rgba(255,64,85,.25);pointer-events:none}

@keyframes zbPulse{0%,100%{box-shadow:0 0 20px rgba(192,132,252,.1)}50%{box-shadow:0 0 30px rgba(192,132,252,.2)}}

/* Bottom stats row */
.zc-stats{display:flex;gap:16px;font-size:9px;color:#5c4d7a;flex-wrap:wrap;align-items:center}
.zc-stats b{font-weight:700}
.zc-stats .profit{color:#00d4aa;font-weight:800}
.zc-stats .yield-val{color:#ffd43b;font-weight:800}
.zc-stats .eta-val{color:#818cf8;font-weight:700}
.zc-stats .deploy-val{color:#c084fc;font-weight:700}

/* Instruction badge */
.zc-instruction{margin-top:8px;padding:6px 12px;border-radius:5px;font-size:10px;font-weight:700;letter-spacing:.5px;display:flex;align-items:center;gap:6px}
.zc-instruction.buy-yes{background:rgba(0,212,170,.08);color:#00d4aa;border:1px solid rgba(0,212,170,.2)}
.zc-instruction.buy-no{background:rgba(255,64,85,.06);color:#ff4055;border:1px solid rgba(255,64,85,.15)}

/* Graveyard mini cards */
.zg{display:flex;align-items:center;gap:12px;padding:8px 14px;border-radius:6px;background:rgba(74,58,106,.06);border:1px solid rgba(74,58,106,.1);margin-bottom:4px;font-size:10px;opacity:.65;transition:opacity .2s}
.zg:hover{opacity:1}
</style>

<div class="section" style="border-color:#6b21a8">
  <div class="sec-hdr collapsible" style="color:#c084fc;background:linear-gradient(90deg,rgba(192,132,252,.08),rgba(139,92,246,.03),transparent);font-size:12px;border-bottom:1px solid rgba(192,132,252,.12)" onclick="toggleSection('zombie')">
    <span style="display:flex;align-items:center;gap:8px"><span class="chevron">&#9660;</span><span style="font-size:14px">&#x1F9DF;</span> ZOMBIE HUNTER — Settlement Arbitrage</span>
    <span style="display:flex;align-items:center;gap:10px">
      <span style="font-size:9px;color:#c084fc" id="z-summary">--</span>
      <span class="meta" id="zombie-status" style="font-size:10px;color:#c084fc">SCANNING</span>
    </span>
  </div>
  <div class="sec-body-wrap" id="body-zombie">
  <div class="sec-body">
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">
      <div class="stat" style="border-color:rgba(192,132,252,.2);padding:8px">
        <div class="stat-label">Capital</div>
        <div class="stat-val" style="color:#c084fc;font-size:16px" id="z-capital">$15,000</div>
        <div class="stat-sub">RESERVED</div>
      </div>
      <div class="stat" style="border-color:rgba(192,132,252,.2);padding:8px">
        <div class="stat-label">Deployed</div>
        <div class="stat-val g" style="font-size:16px" id="z-deployed">--</div>
        <div class="stat-sub">ACTIVE</div>
      </div>
      <div class="stat" style="border-color:rgba(192,132,252,.2);padding:8px">
        <div class="stat-label">Profit</div>
        <div class="stat-val g" style="font-size:16px" id="z-profit">--</div>
        <div class="stat-sub">EXPECTED</div>
      </div>
      <div class="stat" style="border-color:rgba(192,132,252,.2);padding:8px">
        <div class="stat-label">Yield</div>
        <div class="stat-val" style="color:#ffd43b;font-size:16px" id="z-yield">--</div>
        <div class="stat-sub">AVG</div>
      </div>
      <div class="stat" style="border-color:rgba(192,132,252,.2);padding:8px">
        <div class="stat-label">Ready</div>
        <div class="stat-val" style="color:#c084fc;font-size:16px" id="z-ready">--</div>
        <div class="stat-sub">ZOMBIES</div>
      </div>
    </div>

    <div id="zombie-cards" class="zc-grid">
      <div style="text-align:center;color:#4a3a6a;font-size:10px;padding:20px">Scanning for zombies...</div>
    </div>

    <!-- Graveyard -->
    <div style="margin-top:14px;margin-bottom:6px;font-size:9px;color:#4a3a6a;letter-spacing:1px;font-weight:700">
      &#x26B0;&#xFE0F; GRAVEYARD
    </div>
    <div id="graveyard-cards">
      <div style="text-align:center;color:#3a2a5a;font-size:10px;padding:12px">No recent harvests</div>
    </div>

    <!-- FORECAST OPPORTUNITIES — Tomorrow & Day After -->
    <div style="margin-top:18px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
      <div style="font-size:10px;color:#38bdf8;letter-spacing:1.5px;font-weight:700">
        &#x1F52E; FORECAST OPPORTUNITIES — Tomorrow & +2d
      </div>
      <div style="font-size:9px;color:#2d4a6a" id="forecast-summary">--</div>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <div class="stat" style="border-color:rgba(56,189,248,.15);padding:6px 10px;flex:1;min-width:100px">
        <div class="stat-label" style="font-size:8px">Opportunities</div>
        <div class="stat-val" style="color:#38bdf8;font-size:14px" id="fc-count">--</div>
      </div>
      <div class="stat" style="border-color:rgba(0,212,170,.15);padding:6px 10px;flex:1;min-width:100px">
        <div class="stat-label" style="font-size:8px">Expected EV</div>
        <div class="stat-val" style="color:#00d4aa;font-size:14px" id="fc-ev">--</div>
      </div>
      <div class="stat" style="border-color:rgba(255,212,59,.15);padding:6px 10px;flex:1;min-width:100px">
        <div class="stat-label" style="font-size:8px">Best Edge</div>
        <div class="stat-val" style="color:#ffd43b;font-size:14px" id="fc-edge">--</div>
      </div>
      <div class="stat" style="border-color:rgba(167,139,250,.15);padding:6px 10px;flex:1;min-width:100px">
        <div class="stat-label" style="font-size:8px">Data Sources</div>
        <div class="stat-val" style="color:#a78bfa;font-size:12px" id="fc-sources">--</div>
      </div>
    </div>
    <div id="forecast-cards" class="zc-grid">
      <div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">Scanning forward markets...</div>
    </div>
  </div>
  </div>
</div>

<div class="foot">ISOBAR APA-9000 — THERMODYNAMIC ALPHA TERMINAL — Atmospheric Physics Arbitrage — 60s refresh</div>

<script>
let bellChart = null;
let TD = null; // Thermodynamic data
let WS = null; // Weather station data
let ZH = null; // Zombie hunter data

async function safeFetch(url) {
  const r = await fetch(url);
  // If redirected to login (auth expired), reload page so user sees login
  if (r.redirected && r.url.includes('/login')) { window.location.href = '/login'; return null; }
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('application/json')) { window.location.href = '/login'; return null; }
  return r.json();
}

async function load() {
  try {
    const [td, ws, zh] = await Promise.all([
      safeFetch('/api/thermodynamic'),
      safeFetch('/api/weather-station'),
      safeFetch('/api/zombie-hunter')
    ]);
    if (!td && !ws) return; // redirected to login
    TD = td; WS = ws; ZH = zh;
    // If cache is still loading on first boot, retry after 3s
    if (TD && TD._loading) { setTimeout(load, 3000); return; }
    renderAll();
  } catch(e) {
    console.error('Terminal load error:', e);
    try {
      const ws = await safeFetch('/api/weather-station');
      if (!ws) return;
      WS = ws; TD = null;
      if (WS._loading) { setTimeout(load, 3000); return; }
      renderAll();
    } catch(e2) { console.error(e2); }
  }
}

function toggleSection(id) {
  const hdr = document.querySelector('#body-' + id)?.previousElementSibling || document.querySelector('[onclick*="' + id + '"]');
  const body = document.getElementById('body-' + id);
  if (!body) return;
  const isCollapsed = body.classList.contains('collapsed');
  body.classList.toggle('collapsed');
  if (hdr) hdr.classList.toggle('collapsed');
}

function renderAll() {
  renderStats();
  renderTopPicks();
  renderTomorrow();
  renderSectorA();
  renderSectorB();
  renderSectorC();
  renderLadder();
  renderMetarFeed();
  renderZombies();
}

function renderStats() {
  const ep = TD ? TD.execution_plan || {} : {};
  const ag = TD ? TD.agent_23 || {} : {};
  const wp = WS ? WS.portfolio || {} : {};

  document.getElementById('sb-nodes').textContent = TD ? TD.active_nodes || 0 : '--';
  document.getElementById('s-exp').textContent = '$' + (ep.total_bet || wp.total_exposure || 0).toLocaleString(undefined,{maximumFractionDigits:0});
  const pnl = ep.total_expected_pnl || wp.expected_profit || 0;
  const pnlEl = document.getElementById('s-pnl');
  pnlEl.textContent = '$' + pnl.toLocaleString(undefined,{maximumFractionDigits:0});
  pnlEl.className = 'stat-val ' + (pnl >= 0 ? 'g' : 'r');
  document.getElementById('s-pos').textContent = ep.n_positions || wp.n_opportunities || 0;
  document.getElementById('s-twap').textContent = (ep.twap_total_chunks || 0) + ' chunks';
  document.getElementById('s-twap-dur').textContent = (ep.twap_duration_min || 0) + ' MIN';

  const agPnl = ag.pnl || 0;
  const agEl = document.getElementById('s-agent');
  agEl.textContent = '$' + (agPnl >= 0 ? '+' : '') + agPnl.toFixed(0);
  agEl.className = 'stat-val ' + (agPnl >= 0 ? 'g' : 'r');
  document.getElementById('s-aginfo').textContent = 'GEN ' + (ag.generation||1) + ' | ' + (ag.open_positions||0) + ' POS';

  // IEM ASOS station coverage
  if (TD && TD.iem_coverage) {
    document.getElementById('s-iem').textContent = TD.iem_coverage;
    document.getElementById('s-iem-sub').textContent = 'WU-MATCH';
  }

  const now = new Date();
  document.getElementById('s-time').textContent = now.toUTCString().split(' ')[4];
}

function renderTomorrow() {
  const el = document.getElementById('tmrw-grid');
  const dateEl = document.getElementById('tmrw-date');
  if (!TD || !TD.tomorrow || !TD.tomorrow.forecasts) {
    el.innerHTML = '<div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">No tomorrow forecast data</div>';
    return;
  }
  const tmrw = TD.tomorrow;
  const fc = tmrw.forecasts;
  const cities = Object.entries(fc);
  dateEl.textContent = tmrw.date + ' | ' + tmrw.n_cities + ' CITIES';

  if (!cities.length) {
    el.innerHTML = '<div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">No forecast data available</div>';
    return;
  }

  el.innerHTML = cities.map(function(entry) {
    var key = entry[0], c = entry[1];
    var isC = c.unit === 'C';
    var mainTemp = c.display_temp;
    var meanF = c.synth_mean_f;
    var tempColor = meanF >= 75 ? '#ff8c42' : meanF >= 55 ? '#ffd43b' : meanF >= 32 ? '#38bdf8' : '#818cf8';
    var agreeClass = c.agreement === 'HIGH' ? 'high' : c.agreement === 'MODERATE' ? 'moderate' : 'low';

    // Range display
    var rangeStr = isC
      ? c.range_low_c + '\u00b0C \u2014 ' + c.range_high_c + '\u00b0C'
      : Math.round(c.range_low_f) + '\u00b0F \u2014 ' + Math.round(c.range_high_f) + '\u00b0F';

    // Model bars — show individual model temps relative to range
    var modelsHtml = '';
    if (c.models && c.models.length > 0) {
      var allTemps = c.models.map(function(m) { return m.temp_max_f; });
      var lo = Math.min.apply(null, allTemps) - 2;
      var hi = Math.max.apply(null, allTemps) + 2;
      var span = hi - lo || 10;
      var colors = {ecmwf:'#a78bfa',gfs:'#38bdf8',icon:'#ffd43b',gem:'#00d4aa',jma:'#ff8c42'};
      modelsHtml = c.models.map(function(m) {
        var pct = ((m.temp_max_f - lo) / span) * 100;
        var col = colors[m.model] || '#4a6a8a';
        var tStr = isC ? Math.round((m.temp_max_f - 32) * 5/9) + '\u00b0C' : Math.round(m.temp_max_f) + '\u00b0F';
        return '<div style="display:flex;align-items:center;gap:4px">' +
          '<span style="width:35px;text-align:right;color:' + col + '">' + m.model.toUpperCase().slice(0,4) + '</span>' +
          '<div class="tmrw-bar" style="flex:1"><div class="tmrw-bar-fill" style="left:' + Math.max(pct-1,0) + '%;width:3px;background:' + col + '"></div></div>' +
          '<span style="width:30px;color:' + col + ';font-weight:700">' + tStr + '</span>' +
        '</div>';
      }).join('');
    }

    return '<div class="tmrw-card">' +
      '<div class="tmrw-city">' + c.city_name + ' <span class="tmrw-agree ' + agreeClass + '">' + c.agreement + '</span></div>' +
      '<div class="tmrw-temp" style="color:' + tempColor + '">' + mainTemp + '</div>' +
      '<div class="tmrw-range">\u00b1' + c.synth_std_f + '\u00b0F | ' + rangeStr + '</div>' +
      '<div class="tmrw-models">' + modelsHtml + '</div>' +
    '</div>';
  }).join('');
}

function renderSectorA() {
  const el = document.getElementById('sector-a-grid');
  if (!TD || !TD.cities) { el.innerHTML = '<div style="color:#2d4a6a">No thermodynamic data</div>'; return; }

  const cities = Object.entries(TD.cities);
  el.innerHTML = cities.map(([key, c]) => {
    const sw = c.solar || {};
    const nr = c.net_radiation || {};
    const tempStr = c.current_temp_f != null ? Math.round(c.current_temp_f) + '\u00b0F' : '--';
    const projStr = c.projected_max_f != null ? Math.round(c.projected_max_f) + '\u00b0F' : '--';
    const statusColor = nr.status === 'HEATING' ? '#ffd43b' : nr.status === 'COOLING' ? '#38bdf8' : '#2d4a6a';
    const actionClass = c.arb_icon === 'buy' ? 'buy' : c.arb_icon === 'watch' ? 'watch' : c.arb_icon === 'sell' ? 'sell' : 'hold';
    const cid = 'phys-' + key.replace(/\s+/g,'-');

    // WU estimate line
    var wuLine = '';
    var fvs = c.forecast_vs_station;
    if (fvs && fvs.status === 'OK') {
      var gc = {'A':'#00d4aa','B':'#38bdf8','C':'#ffd43b','D':'#ff8c00','F':'#ff4055','--':'#4a6a8a'};
      var ph = fvs.day_phase==='POST_PEAK' ? '\u2705' : fvs.day_phase==='PRE_PEAK' ? '\u23f3' : '\ud83c\udf19';
      wuLine = '<div class="phys-row"><span class="label">WU Est ' + ph + '</span><span class="val" style="color:' + (gc[fvs.accuracy_grade]||'#4a6a8a') + '">' + Math.round(fvs.wunderground_likely_f) + '\u00b0F | ' + fvs.accuracy_grade + ' (' + (fvs.delta_f>0?'+':'') + fvs.delta_f + '\u00b0F)</span></div>';
    }

    return '<div class="phys-card ' + (c.current_temp_f != null ? 'active' : '') + '">' +
      '<div class="phys-city">' + c.city_name + ' (' + c.metar_station + ') <span style="font-size:8px;color:#2d4a6a">' + (sw.status || '--') + '</span></div>' +
      '<div class="phys-row"><span class="label">Current</span><span class="val y">' + tempStr + '</span></div>' +
      '<div class="phys-row"><span class="label">Projected Max</span><span class="val g">' + projStr + '</span></div>' +
      (c.iem_day_max_f != null ? '<div class="phys-row"><span class="label">ASOS Day Max</span><span class="val" style="color:#e879f9;font-weight:700">' + Math.round(c.iem_day_max_f) + '\u00b0F' + (c.iem_day_max_c != null ? ' (' + Math.round(c.iem_day_max_c) + '\u00b0C)' : '') + '</span></div>' : '') +
      wuLine +
      (c.station_warning ? '<div style="font-size:7px;color:#ff4055;padding:2px 5px;margin-top:3px;background:rgba(255,64,85,.08);border:1px solid rgba(255,64,85,.2);border-radius:3px">\u26a0 ' + c.station_warning + '</div>' : '') +
      '<div class="phys-extra" id="' + cid + '">' +
        '<div class="phys-divider"></div>' +
        '<div class="phys-row"><span class="label">SW Flux</span><span class="val y">' + (sw.surface_flux || 0) + ' W/m\u00b2</span></div>' +
        '<div class="phys-row"><span class="label">Cloud Opacity</span><span class="val">' + ((sw.cloud_fraction||0)*100).toFixed(0) + '%</span></div>' +
        '<div class="phys-row"><span class="label">Albedo</span><span class="val">' + (sw.albedo || '--') + '</span></div>' +
        '<div class="phys-row"><span class="label">Net Absorbed</span><span class="val" style="color:' + statusColor + '">' + (sw.net_absorbed > 0 ? '+' : '') + (sw.net_absorbed || 0) + ' W/m\u00b2</span></div>' +
        '<div class="phys-divider"></div>' +
        '<div class="phys-row"><span class="label">Sun Angle</span><span class="val b">' + (sw.solar_elevation_deg || 0) + '\u00b0</span></div>' +
        '<div class="phys-row"><span class="label">dT/dt</span><span class="val" style="color:' + statusColor + '">' + (c.heating_rate_f_hr > 0 ? '+' : '') + (c.heating_rate_f_hr || 0) + '\u00b0F/hr</span></div>' +
        '<div class="phys-row"><span class="label">Net Rate</span><span class="val" style="color:' + statusColor + '">' + (c.net_rate_f_hr > 0 ? '+' : '') + c.net_rate_f_hr + '\u00b0F/hr</span></div>' +
        '<div class="phys-divider"></div>' +
        '<div class="phys-row"><span class="label">Edge Markets</span><span class="val">' + (c.n_positive_ev || 0) + ' / ' + (c.n_markets || 0) + '</span></div>' +
      '</div>' +
      '<div class="phys-toggle" onclick="var e=document.getElementById(\'' + cid + '\');e.classList.toggle(\'open\');this.textContent=e.classList.contains(\'open\')?\'\\u25B2 COLLAPSE\':\'\\u25BC EXPAND\'">\u25bc EXPAND</div>' +
      '<div class="phys-action ' + actionClass + '">' + (c.arb_status || 'HOLD') + (c.best_edge > 0 ? ' | EDGE ' + (c.best_edge*100).toFixed(1) + '%' : '') + '</div>' +
    '</div>';
  }).join('');
}

function renderSectorB() {
  const el = document.getElementById('sector-b-body');
  if (!TD || !TD.cities) { el.innerHTML = '<div style="color:#2d4a6a">No advection data</div>'; return; }

  let html = '';
  for (const [key, c] of Object.entries(TD.cities)) {
    const adv = c.advection || {};
    const lake = c.lake_effect || {};
    if (adv.effect === 'NO DATA' && !lake.active) continue;

    const advRate = adv.advection_rate_f_hr || 0;
    const advColor = advRate > 0.2 ? '#ff4055' : advRate < -0.2 ? '#38bdf8' : '#2d4a6a';
    const windDir = adv.wind_speed_kt ? adv.wind_speed_kt + 'kt @ ' + (c.wind_dir||0) + '°' : 'Calm';

    html += `<div style="background:#04060c;border:1px solid #0f1a2e;border-radius:5px;padding:8px 10px;margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <span class="b" style="font-size:10px;font-weight:700">${c.city_name} (${c.metar_station})</span>
        <span style="font-size:9px;color:#2d4a6a">Wind: ${windDir}</span>
      </div>
      <div class="phys-row"><span class="label">Upstream Air Mass</span><span class="val">${c.upstream_temp_f != null ? Math.round(c.upstream_temp_f) + '°F' : '--'}</span></div>
      <div class="phys-row"><span class="label">Local Air Mass</span><span class="val">${c.current_temp_f != null ? Math.round(c.current_temp_f) + '°F' : '--'}</span></div>
      <div class="phys-row"><span class="label">Advection dT/dt</span><span class="val" style="color:${advColor}">${advRate > 0 ? '+' : ''}${advRate.toFixed(2)}°F/hr ${adv.effect || ''}</span></div>`;

    if (lake.active) {
      html += `<div class="phys-row"><span class="label">Lake Effect</span><span class="val b">${lake.lake_effect_f_hr > 0 ? '+' : ''}${lake.lake_effect_f_hr}°F/hr (water ${lake.lake_temp_c}°C)</span></div>`;
    }

    const solar = c.heating_rate_f_hr || 0;
    const net = c.net_rate_f_hr || 0;
    const netColor = net > 0.3 ? '#ffd43b' : net < -0.3 ? '#38bdf8' : '#2d4a6a';
    html += `<div style="margin-top:4px;font-size:9px;color:${netColor};font-weight:700">
      Solar ${solar > 0 ? '+' : ''}${solar}°F/hr + Advection ${advRate > 0 ? '+' : ''}${advRate.toFixed(1)}°F/hr = NET ${net > 0 ? '+' : ''}${net}°F/hr
    </div></div>`;
  }
  el.innerHTML = html || '<div style="color:#2d4a6a;font-size:10px;padding:10px">No advection signals — winds too light or no upstream data</div>';
}

function renderSectorC() {
  const selEl = document.getElementById('ens-select');
  const bodyEl = document.getElementById('ens-body');
  if (!TD || !TD.ensemble) { bodyEl.innerHTML = '<div style="color:#2d4a6a">No ensemble data</div>'; return; }

  const keys = Object.keys(TD.ensemble);
  selEl.innerHTML = keys.map((k, i) =>
    `<button class="curve-btn ${i===0?'active':''}" onclick="showEnsemble('${k}',this)">${TD.cities[k]?.city_name || k}</button>`
  ).join('');
  if (keys.length > 0) showEnsemble(keys[0]);
}

function showEnsemble(cityKey, btn) {
  if (btn) {
    document.querySelectorAll('#ens-select .curve-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  if (!TD || !TD.ensemble || !TD.ensemble[cityKey]) return;
  const ens = TD.ensemble[cityKey];
  const bodyEl = document.getElementById('ens-body');

  // Find temperature range for bar visualization
  const temps = ens.models.map(m => m.temp_max_f);
  const minT = Math.min(...temps) - 3;
  const maxT = Math.max(...temps) + 3;
  const range = maxT - minT || 10;

  let html = `<div style="font-size:9px;color:#a78bfa;margin-bottom:6px;font-weight:700">
    BAYESIAN SYNTHESIS: <span class="y">${ens.synth_mean_f}°F</span> ±${ens.synth_std_f}°F |
    Model Spread: <span class="${ens.agreement==='HIGH'?'g':ens.agreement==='MODERATE'?'y':'r'}">${ens.model_spread_f}°F (${ens.agreement})</span>
  </div>`;

  ens.models.forEach(m => {
    const pct = ((m.temp_max_f - minT) / range) * 100;
    const w = Math.max(m.weight * 200, 30);
    const color = m.model === 'ecmwf' ? '#a78bfa' : m.model === 'gfs' ? '#38bdf8' : m.model === 'icon' ? '#ffd43b' : m.model === 'gem' ? '#00d4aa' : '#ff8c42';
    html += `<div class="ens-bar">
      <span class="ens-name">${m.name.split(' ')[0]}</span>
      <div class="ens-track">
        <div class="ens-fill" style="left:${Math.max(pct-2,0)}%;width:${Math.min(w/5,20)}%;background:${color};opacity:.6"></div>
        <div style="position:absolute;left:${pct}%;top:0;width:2px;height:100%;background:${color}"></div>
      </div>
      <span class="ens-val" style="color:${color}">${m.temp_max_f}°F</span>
    </div>`;
  });

  // Synthesis marker
  const synthPct = ((ens.synth_mean_f - minT) / range) * 100;
  html += `<div class="ens-bar">
    <span class="ens-name" style="color:#ffd43b;font-weight:700">SYNTH</span>
    <div class="ens-track" style="border-color:#1a2038">
      <div style="position:absolute;left:${synthPct}%;top:0;width:3px;height:100%;background:#ffd43b;border-radius:1px"></div>
    </div>
    <span class="ens-val y">${ens.synth_mean_f}°F</span>
  </div>`;

  bodyEl.innerHTML = html;

  // Bell curve chart
  renderBellCurve(cityKey, ens);
}

function renderBellCurve(cityKey, ens) {
  if (!WS || !WS.curves || !WS.curves[cityKey]) return;
  const curve = WS.curves[cityKey];
  const ctx = document.getElementById('bellCanvas').getContext('2d');
  if (bellChart) bellChart.destroy();

  const labels = curve.noaa.map(p => p.temp + '°');
  const noaaData = curve.noaa.map(p => p.prob);
  const marketData = [];
  const noaaTemps = curve.noaa.map(p => p.temp);
  noaaTemps.forEach(t => {
    const mk = curve.market.find(m => Math.abs(m.temp - t) < 1.5);
    marketData.push(mk ? mk.prob : null);
  });

  bellChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Physics Model', data: noaaData, borderColor: '#38bdf8', borderWidth: 2, borderDash: [5,3], fill: false, tension: 0.4, pointRadius: 0 },
        { label: 'Market Implied', data: marketData, borderColor: '#00d4aa', borderWidth: 2, fill: {target:0,above:'rgba(0,212,170,0.06)',below:'rgba(255,64,85,0.06)'}, tension: 0.4, pointRadius: 2, pointBackgroundColor: '#00d4aa', spanGaps: true },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: {labels: {color:'#2d4a6a',font:{size:9,family:'SF Mono,Monaco,monospace'}}},
        tooltip: { callbacks: { label: c => c.dataset.label+': '+(c.parsed.y*100).toFixed(1)+'%' }}},
      scales: {
        x: {ticks:{color:'#1a2538',font:{size:7},maxRotation:45},grid:{color:'#080c16'}},
        y: {ticks:{color:'#1a2538',font:{size:7},callback:v=>(v*100).toFixed(0)+'%'},grid:{color:'#080c16'},beginAtZero:true}
      }
    }
  });
}

/* renderSensorTable removed — Sector D eliminated */

function renderLadder() {
  const selEl = document.getElementById('ladder-select');
  if (!WS || !WS.curves) return;
  const keys = Object.keys(WS.curves);
  selEl.innerHTML = keys.map((k,i) =>
    `<button class="curve-btn ${i===0?'active':''}" onclick="showLadder('${k}',this)">${WS.curves[k].city_name}</button>`
  ).join('');
  if (keys.length > 0) showLadder(keys[0]);
}

function showLadder(cityKey, btn) {
  if (btn) { document.querySelectorAll('#ladder-select .curve-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); }
  if (!WS || !WS.arbs) return;

  const cityArbs = WS.arbs.filter(a => a.city === cityKey);
  const vizEl = document.getElementById('ladder-viz');
  const infoEl = document.getElementById('ladder-info');
  document.getElementById('ladder-city').textContent = WS.curves[cityKey]?.city_name || cityKey;

  if (!cityArbs.length) { vizEl.innerHTML = '<div style="color:#2d4a6a;font-size:10px;padding:16px">No brackets</div>'; infoEl.innerHTML=''; return; }

  cityArbs.sort((a,b) => (parseFloat(a.bracket)||0) - (parseFloat(b.bracket)||0));
  const maxP = Math.max(...cityArbs.map(a => Math.max(a.our_prob, a.market_price)), 0.3);

  vizEl.innerHTML = cityArbs.map(a => {
    const nH = Math.max((a.our_prob/maxP)*100, 3);
    const mH = Math.max((a.market_price/maxP)*100, 3);
    const nC = a.our_prob > a.market_price ? '#38bdf8' : '#0f1a2e';
    const mC = a.market_price > a.our_prob ? '#ff4055' : '#00d4aa';
    return `<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:24px">
      <div style="font-size:6px;color:#2d4a6a;margin-bottom:1px">${(a.our_prob*100).toFixed(0)}%</div>
      <div style="display:flex;gap:1px;align-items:flex-end;height:80px">
        <div style="width:10px;height:${nH}%;background:${nC};border-radius:2px 2px 0 0"></div>
        <div style="width:10px;height:${mH}%;background:${mC};border-radius:2px 2px 0 0;opacity:.6"></div>
      </div>
      <div style="font-size:6px;color:#2d4a6a;margin-top:1px">${a.bracket}</div>
    </div>`;
  }).join('');

  const buyable = cityArbs.filter(a => a.ev_pct > 0);
  const cost = buyable.reduce((s,a) => s + a.market_price, 0);
  infoEl.innerHTML = `<b class="b">LADDER:</b> Buy ${buyable.length} brackets | Cost: <b class="y">${(cost*100).toFixed(0)}¢</b> | Payout: <b class="g">$1.00</b> | ROI: <b class="g">${cost>0?((1/cost-1)*100).toFixed(0):0}%</b>`;
}

function renderMetarFeed() {
  const el = document.getElementById('metar-feed');
  const raw = TD ? TD.metar_raw || {} : (WS ? WS.metar || {} : {});
  if (!Object.keys(raw).length) { el.innerHTML = '<div style="color:#2d4a6a">No METAR</div>'; return; }

  if (typeof Object.values(raw)[0] === 'string') {
    // Raw strings from thermodynamic API
    el.innerHTML = Object.entries(raw).map(([station, rawStr]) =>
      `<div class="metar-line"><span class="metar-station">${station}</span> ${rawStr || '--'}</div>`
    ).join('');
  } else {
    // Full objects from weather-station API
    el.innerHTML = Object.values(raw).map(m =>
      `<div class="metar-line"><span class="metar-station">${m.station}</span> <span class="metar-temp">${m.temp_f!=null?Math.round(m.temp_f)+'°F':'--'}</span> ${m.raw||''}</div>`
    ).join('');
  }
}

/* toggleAvoid + hideAvoid removed — Arb Table eliminated */

function parsePickInfo(a) {
  // Extract unit, threshold, and date from title/bracket
  const t = a.title || '';
  const b = a.bracket || '';
  const isC = b.includes('C') || t.includes('\u00b0C');
  const unit = isC ? '\u00b0C' : '\u00b0F';
  // Extract date
  const dm = t.match(/February (\d+)/);
  const date = dm ? 'FEB ' + dm[1] : '';
  // Extract threshold description
  let thresh = '';
  const mBelow = t.match(/(-?\d+)\s*\u00b0?\s*[FC]\s+or\s+below/i);
  const mHigher = t.match(/(-?\d+)\s*\u00b0?\s*[FC]\s+or\s+higher/i);
  const mBetween = t.match(/between\s+(-?\d+)[-\u2013](-?\d+)\s*\u00b0?\s*[FC]/i);
  if (mBelow) thresh = '\u2264 ' + mBelow[1] + unit;
  else if (mHigher) thresh = '\u2265 ' + mHigher[1] + unit;
  else if (mBetween) thresh = mBetween[1] + '-' + mBetween[2] + unit;
  // Forecast with unit
  const fc = a.forecast_temp != null ? a.forecast_temp.toFixed(1) + unit : '--';
  // Temp class for color coding
  const tempF = isC ? a.forecast_temp * 9/5 + 32 : a.forecast_temp;
  const tempClass = tempF >= 75 ? 'hot' : tempF >= 55 ? 'warm' : tempF >= 32 ? 'cool' : 'cold';
  return { unit, date, thresh, fc, tempClass, isC };
}

function renderTopPicks() {
  const el = document.getElementById('picks-cards');
  const countEl = document.getElementById('picks-count');
  const arbs = TD ? TD.arb_table : (WS ? WS.arbs : []);
  if (!arbs || !arbs.length) { el.innerHTML='<div style="text-align:center;color:#2d4a6a;padding:20px;font-size:10px">No picks available</div>'; countEl.textContent='0'; return; }

  const picks = arbs.filter(a => a.ev_pct > 0 && a.bet_size > 0);
  picks.forEach(a => { a._score = Math.abs(a.edge) * (a.sigma || 0.1) * (a.half_kelly || 0.01) * 1000; });
  picks.sort((a, b) => b._score - a._score);

  const totalBet = picks.reduce((s,a) => s + (a.bet_size||0), 0);
  const totalPnl = picks.reduce((s,a) => s + (a.bet_size||0) * Math.abs(a.edge||0), 0);
  countEl.textContent = picks.length + ' BETS | $' + totalBet.toLocaleString(undefined,{maximumFractionDigits:0}) + ' | EV $' + totalPnl.toLocaleString(undefined,{maximumFractionDigits:0});

  if (!picks.length) { el.innerHTML='<div style="text-align:center;color:#2d4a6a;padding:20px;font-size:10px">No positive EV opportunities</div>'; return; }

  el.innerHTML = picks.map((a, i) => {
    const p = parsePickInfo(a);
    const isTop3 = i < 3;
    const rankColor = i === 0 ? '#ffd43b' : i < 3 ? '#00d4aa' : i < 8 ? '#38bdf8' : '#4a6a8a';
    const prob = (a.our_prob * 100);
    const probClass = prob >= 60 ? 'high' : prob >= 30 ? 'mid' : 'low';
    const costPerShare = a.pick === 'Yes' ? a.market_price : (1 - a.market_price);
    const edge = Math.abs(a.edge * 100);
    const sig = a.sigma || 0;

    return '<div class="pick-card' + (isTop3 ? ' top3' : '') + '">' +
      '<div class="pick-rank" style="background:' + rankColor + ';color:#04060c">' + (i+1) + '</div>' +
      '<div class="pick-info">' +
        '<div style="display:flex;align-items:center;gap:8px">' +
          '<span class="pick-city">' + (a.city_name||a.city).toUpperCase() + '</span>' +
          '<span class="pick-date">' + p.date + '</span>' +
        '</div>' +
        '<div class="pick-temps">' +
          '<span class="pick-fc ' + p.tempClass + '">&#9729; ' + p.fc + '</span>' +
          '<span class="pick-arrow">&#10230;</span>' +
          '<span class="pick-threshold">' + p.thresh + '</span>' +
          (a.iem_day_max_f != null ? '<span style="font-size:8px;color:#e879f9;margin-left:6px;padding:1px 5px;border:1px solid rgba(232,121,249,.25);border-radius:3px">ASOS ' + Math.round(a.iem_day_max_f) + '\u00b0F' + (a.wu_likely_f != null ? ' \u2192 WU~' + Math.round(a.wu_likely_f) + '\u00b0F' : '') + '</span>' : '') +
        '</div>' +
        (a.station_warning ? '<div style="font-size:7.5px;color:#ff4055;padding:2px 0">\u26a0 ' + a.station_warning + '</div>' : '') +
        '<div class="pick-stats">' +
          'Edge <b>' + edge.toFixed(1) + '%</b> &middot; ' +
          sig.toFixed(1) + '\u03c3 &middot; ' +
          'Cost <b>' + (costPerShare*100).toFixed(0) + '\u00a2</b> &middot; ' +
          'Bet <b>$' + (a.bet_size||0).toLocaleString(undefined,{maximumFractionDigits:0}) + '</b>' +
          (a.forecast_vs_station ? ' &middot; <span style="color:' + (a.forecast_vs_station === 'A' ? '#00d4aa' : a.forecast_vs_station === 'B' ? '#38bdf8' : a.forecast_vs_station === 'C' ? '#ffd43b' : '#ff4055') + '">' + a.forecast_vs_station + '</span>' : '') +
        '</div>' +
      '</div>' +
      '<div class="pick-right">' +
        '<div class="pick-action ' + (a.pick === 'Yes' ? 'yes' : 'no') + '">' +
          'BUY ' + (a.pick === 'Yes' ? 'YES \u25b2' : 'NO \u25bc') +
        '</div>' +
        '<div class="pick-prob ' + probClass + '">' + prob.toFixed(0) + '%</div>' +
        '<div style="font-size:8px;color:#2d4a6a">WIN PROB</div>' +
      '</div>' +
    '</div>';
  }).join('');
}

/* renderArbTable removed — consolidated into TOP PICKS */

function formatBracketName(q) {
  // Clean up the bracket question into a short Polymarket-style label
  // "What will be the highest temperature in Toronto on February 8, 2026 if the highest temperature is between 18-19°F" → "between 18-19°F"
  var s = q || '';
  var m = s.match(/(between\s+-?\d+\s*[-\u2013]\s*-?\d+\s*\u00b0?\s*[FCfc])/i);
  if (m) return m[1];
  m = s.match(/(-?\d+\s*\u00b0?\s*[FCfc]\s+or\s+(?:below|lower|higher|above))/i);
  if (m) return m[1];
  m = s.match(/(exactly\s+-?\d+\s*\u00b0?\s*[FCfc])/i);
  if (m) return m[1];
  // Fallback: last meaningful part
  var parts = s.split('if ');
  if (parts.length > 1) {
    var tail = parts[parts.length - 1].replace(/^the highest temperature (is )?/i, '');
    return tail.length > 60 ? tail.substring(0, 57) + '...' : tail;
  }
  return s.length > 60 ? s.substring(0, 57) + '...' : s;
}

let zombieShowAll = false;
let graveyardShowAll = false;

function toggleZombieShowAll() {
  zombieShowAll = !zombieShowAll;
  renderZombies();
}
function toggleGraveyardShowAll() {
  graveyardShowAll = !graveyardShowAll;
  renderZombies();
}

function renderZombies() {
  if (!ZH) return;
  const s = ZH.stats || {};
  const ready = ZH.zombies_ready || [];
  const watching = ZH.zombies_watching || [];
  const graveyard = ZH.graveyard || [];
  const all = [...ready, ...watching];

  // Stats
  document.getElementById('zombie-status').textContent = s.status || 'OFFLINE';
  document.getElementById('z-capital').textContent = '$' + (s.capital_reserved || 15000).toLocaleString();
  document.getElementById('z-deployed').textContent = '$' + (s.capital_deployed || 0).toLocaleString(undefined,{maximumFractionDigits:0});
  document.getElementById('z-profit').textContent = '$' + (s.expected_profit || 0).toLocaleString(undefined,{maximumFractionDigits:2});
  document.getElementById('z-yield').textContent = (s.avg_yield_pct || 0).toFixed(2) + '%';
  document.getElementById('z-ready').textContent = s.total_ready || 0;
  var sumEl = document.getElementById('z-summary');
  if (sumEl) sumEl.textContent = (s.total_ready||0) + ' zombies + ' + (s.total_forecast||0) + ' forecasts | $' + ((s.expected_profit||0) + (s.forecast_ev||0)).toFixed(0) + ' EV';

  const el = document.getElementById('zombie-cards');
  const ZOMBIE_LIMIT = 5;
  const GRAVEYARD_LIMIT = 3;

  if (!all.length) {
    el.innerHTML = '<div style="text-align:center;color:#4a3a6a;font-size:10px;padding:20px">No zombies found. Markets may have resolved, or the sun has not set yet in target cities.</div>';
  } else {
    const visible = zombieShowAll ? all : all.slice(0, ZOMBIE_LIMIT);
    const remaining = all.length - ZOMBIE_LIMIT;
    let cardsHtml = visible.map(z => {
      const isConfirmed = z.status === 'DEAD_CONFIRMED';
      const isSunset = z.status === 'SUN_SET';
      const isMarginal = z.status === 'MARGINAL' || z.status === 'DISPUTED';
      const statusCls = isConfirmed ? 'dead' : isSunset ? 'sunset' : 'marginal';
      const statusLabel = isConfirmed ? 'CONFIRMED' : isSunset ? 'SUN SET' : z.status.replace('_',' ');

      // Bracket display
      const bracket = formatBracketName(z.question);
      const isCelsius = z.observed_unit === '\u00b0C';
      const unitCls = isCelsius ? 'celsius' : 'fahrenheit';
      const unitLabel = isCelsius ? '\u00b0C' : '\u00b0F';

      // Result tag
      const resultCls = z.outcome === 'YES_WINS' ? 'yes-wins' : z.outcome === 'NO_WINS' ? 'no-wins' : 'uncertain';
      const resultLabel = z.outcome === 'YES_WINS' ? 'YES WINS' : z.outcome === 'NO_WINS' ? 'NO WINS' : 'UNCERTAIN';

      // Button logic: which side is active (the one to buy)
      const isYesAction = z.action && z.action.indexOf('YES') >= 0;
      const isNoAction = z.action && z.action.indexOf('NO') >= 0;
      const yesCls = isYesAction ? 'yes-active' : 'yes-dim';
      const noCls = isNoAction ? 'no-active' : 'no-dim';
      const yesPriceCents = z.yes_price ? (z.yes_price * 100).toFixed(0) : '--';
      const noPriceCents = z.no_price ? (z.no_price * 100).toFixed(0) : '--';
      const isMarginalAction = z.action && z.action.indexOf('MARGINAL') >= 0;

      // Instruction text
      var instructionHtml = '';
      if (isYesAction) {
        instructionHtml = '<div class="zc-instruction buy-yes">' +
          '\u2714 Go to Polymarket \u2192 find <b>"' + bracket + '"</b> \u2192 click <b>Buy Yes ' + yesPriceCents + '\u00a2</b>' +
          (isMarginalAction ? ' <span style="color:#fbbf24;margin-left:4px">(MARGINAL \u2014 boundary risk)</span>' : '') +
          '</div>';
      } else if (isNoAction) {
        instructionHtml = '<div class="zc-instruction buy-no">' +
          '\u2714 Go to Polymarket \u2192 find <b>"' + bracket + '"</b> \u2192 click <b>Buy No ' + noPriceCents + '\u00a2</b>' +
          (isMarginalAction ? ' <span style="color:#fbbf24;margin-left:4px">(MARGINAL \u2014 boundary risk)</span>' : '') +
          '</div>';
      }

      return '<div class="zc' + (isConfirmed ? ' confirmed' : '') + '">' +
        '<div class="zc-top">' +
          '<div class="zc-city">' +
            '<span class="zc-icon">' + (z.status_icon || '\u{1F9DF}') + '</span>' +
            '<div><div class="zc-name">' + z.city_name + '</div>' +
            '<div class="zc-date">' + (z.market_date || '') + '</div></div>' +
          '</div>' +
          '<span class="zc-status ' + statusCls + '">' + statusLabel + '</span>' +
        '</div>' +
        '<div class="zc-bracket">' +
          '<span class="unit-badge ' + unitCls + '">' + unitLabel + '</span>' +
          '<span>' + bracket + '</span>' +
        '</div>' +
        '<div class="zc-observed">' +
          'Observed High: <span class="zc-obs-temp">' + z.observed_temp + z.observed_unit + '</span>' +
          '<span>\u2192</span>' +
          '<span class="zc-result ' + resultCls + '">' + resultLabel + '</span>' +
        '</div>' +
        '<div class="zc-buttons">' +
          '<div class="zc-btn ' + yesCls + '">' +
            '<div class="btn-label">Buy Yes</div>' +
            '<div class="btn-price">' + yesPriceCents + '\u00a2</div>' +
            '<div class="btn-sub">$1.00 if YES</div>' +
          '</div>' +
          '<div class="zc-btn ' + noCls + '">' +
            '<div class="btn-label">Buy No</div>' +
            '<div class="btn-price">' + noPriceCents + '\u00a2</div>' +
            '<div class="btn-sub">$1.00 if NO</div>' +
          '</div>' +
        '</div>' +
        '<div class="zc-stats">' +
          '<span>Yield: <b class="yield-val">+' + (z.yield_pct || 0).toFixed(2) + '%</b></span>' +
          '<span>Deploy: <b class="deploy-val">' + (z.deploy_amount ? '$' + z.deploy_amount.toLocaleString(undefined,{maximumFractionDigits:0}) : '--') + '</b></span>' +
          '<span>Profit: <b class="profit">' + (z.expected_profit ? '$' + z.expected_profit.toFixed(2) : '--') + '</b></span>' +
          '<span>ETA: <b class="eta-val">' + (z.eta_display || '--') + '</b></span>' +
        '</div>' +
        instructionHtml +
      '</div>';
    }).join('');
    if (!zombieShowAll && remaining > 0) {
      cardsHtml += '<button class="show-more-btn" onclick="toggleZombieShowAll()" style="border-color:rgba(192,132,252,.2);color:#c084fc;background:rgba(192,132,252,.03)">SHOW ' + remaining + ' MORE ZOMBIES</button>';
    } else if (zombieShowAll && all.length > ZOMBIE_LIMIT) {
      cardsHtml += '<button class="show-more-btn" onclick="toggleZombieShowAll()" style="border-color:rgba(192,132,252,.2);color:#c084fc;background:rgba(192,132,252,.03)">COLLAPSE</button>';
    }
    el.innerHTML = cardsHtml;
  }

  // Graveyard — mini cards with limit
  const gEl = document.getElementById('graveyard-cards');
  if (!graveyard.length) {
    gEl.innerHTML = '<div style="text-align:center;color:#3a2a5a;font-size:10px;padding:12px">No recent harvests yet</div>';
  } else {
    const gVisible = graveyardShowAll ? graveyard : graveyard.slice(0, GRAVEYARD_LIMIT);
    const gRemaining = graveyard.length - GRAVEYARD_LIMIT;
    let gHtml = gVisible.map(g => {
      return '<div class="zg">' +
        '<span>' + (g.status_icon || '\u26B0\uFE0F') + '</span>' +
        '<span style="color:#7a8ba0;flex:1">' + g.city_name + ': <b style="color:#c8d6e5">' + formatBracketName(g.question) + '</b></span>' +
        '<span style="color:#38bdf8;font-weight:700">' + g.observed_temp + g.observed_unit + '</span>' +
        '<span style="color:#c084fc">' + (g.yes_price ? (g.yes_price * 100).toFixed(0) + '\u00a2' : '--') + '</span>' +
        '<span style="color:#00d4aa;font-weight:700">' + (g.yield_pct > 0 ? '+' + g.yield_pct.toFixed(2) + '%' : '--') + '</span>' +
        '<span style="color:#4a3a6a;font-size:9px">' + (g.status || 'RESOLVED') + '</span>' +
      '</div>';
    }).join('');
    if (!graveyardShowAll && gRemaining > 0) {
      gHtml += '<button class="show-more-btn" onclick="toggleGraveyardShowAll()" style="border-color:rgba(74,58,106,.2);color:#4a3a6a;background:rgba(74,58,106,.03);font-size:9px;padding:6px">SHOW ' + gRemaining + ' MORE</button>';
    } else if (graveyardShowAll && graveyard.length > GRAVEYARD_LIMIT) {
      gHtml += '<button class="show-more-btn" onclick="toggleGraveyardShowAll()" style="border-color:rgba(74,58,106,.2);color:#4a3a6a;background:rgba(74,58,106,.03);font-size:9px;padding:6px">COLLAPSE</button>';
    }
    gEl.innerHTML = gHtml;
  }

  // ── FORECAST OPPORTUNITIES ──
  const forecasts = ZH.forecast_opportunities || [];
  document.getElementById('fc-count').textContent = forecasts.length;
  document.getElementById('fc-ev').textContent = '$' + (s.forecast_ev || 0).toFixed(2);
  const bestEdge = forecasts.length ? Math.max(...forecasts.map(f => f.edge || 0)) : 0;
  document.getElementById('fc-edge').textContent = (bestEdge * 100).toFixed(1) + '%';
  const dSources = ZH.data_sources || [];
  document.getElementById('fc-sources').textContent = dSources.length + ' APIs';
  document.getElementById('forecast-summary').textContent = forecasts.length + ' forward opps | ' + dSources.slice(0,2).join(', ');

  const fcEl = document.getElementById('forecast-cards');
  if (!forecasts.length) {
    fcEl.innerHTML = '<div style="text-align:center;color:#2d4a6a;font-size:10px;padding:20px">No forward opportunities found. Markets may not exist yet for tomorrow/+2d.</div>';
  } else {
    let fcHtml = forecasts.slice(0, 15).map(f => {
      const bracket = formatBracketName(f.question);
      const dayLabel = f.day_offset === 1 ? 'TOMORROW' : '+2 DAYS';
      const dayColor = f.day_offset === 1 ? '#38bdf8' : '#818cf8';
      const agreeCls = f.model_agreement === 'HIGH' ? '#00d4aa' : f.model_agreement === 'MODERATE' ? '#ffd43b' : '#ff4055';
      const isYes = f.action && f.action.indexOf('YES') >= 0;
      const buyPrice = f.buy_price ? (f.buy_price * 100).toFixed(0) : '--';

      return '<div class="zc" style="border-color:rgba(56,189,248,.12);background:linear-gradient(135deg,rgba(56,189,248,.03),rgba(0,212,170,.02))">' +
        '<div class="zc-top">' +
          '<div class="zc-city">' +
            '<span class="zc-icon">' + (f.status_icon || '\u{1F52E}') + '</span>' +
            '<div><div class="zc-name">' + f.city_name + '</div>' +
            '<div class="zc-date" style="color:' + dayColor + ';font-weight:700">' + dayLabel + ' \u2014 ' + (f.market_date || '') + '</div></div>' +
          '</div>' +
          '<span style="font-size:9px;padding:3px 8px;border-radius:4px;font-weight:700;letter-spacing:1px;background:rgba(56,189,248,.12);color:#38bdf8;border:1px solid rgba(56,189,248,.25)">FORECAST</span>' +
        '</div>' +
        '<div class="zc-bracket">' +
          '<span>' + bracket + '</span>' +
          '<span style="margin-left:8px;font-size:10px;color:#5c6478">\u2192 Forecast: <b style="color:#38bdf8">' + f.display_temp + '</b> \u00b1 ' + f.display_range + '</span>' +
        '</div>' +
        '<div style="display:flex;gap:16px;margin:6px 0;font-size:10px;flex-wrap:wrap">' +
          '<span>Our Prob: <b style="color:#00d4aa">' + (f.our_prob * 100).toFixed(1) + '%</b></span>' +
          '<span>Market: <b style="color:#ffd43b">' + (isYes ? (f.yes_price*100).toFixed(0) : (f.no_price*100).toFixed(0)) + '\u00a2</b></span>' +
          '<span>Edge: <b style="color:#38bdf8">' + (f.edge * 100).toFixed(1) + '%</b></span>' +
          '<span>EV%: <b style="color:#00d4aa">+' + (f.ev_pct || 0).toFixed(1) + '%</b></span>' +
          '<span>Models: <b style="color:' + agreeCls + '">' + f.n_models + ' (' + f.model_agreement + ')</b></span>' +
          (f.vc_temp_f ? '<span>VC: <b style="color:#a78bfa">' + Math.round(f.vc_temp_f) + '\u00b0F</b></span>' : '') +
        '</div>' +
        '<div class="zc-buttons">' +
          '<div class="zc-btn ' + (isYes ? 'yes-active' : 'yes-dim') + '">' +
            '<div class="btn-label">Buy Yes</div>' +
            '<div class="btn-price">' + (f.yes_price * 100).toFixed(0) + '\u00a2</div>' +
          '</div>' +
          '<div class="zc-btn ' + (!isYes ? 'no-active' : 'no-dim') + '">' +
            '<div class="btn-label">Buy No</div>' +
            '<div class="btn-price">' + (f.no_price * 100).toFixed(0) + '\u00a2</div>' +
          '</div>' +
        '</div>' +
        '<div class="zc-instruction ' + (isYes ? 'buy-yes' : 'buy-no') + '">' +
          '\u2714 ' + f.action + ' at ' + buyPrice + '\u00a2 \u2014 Kelly size: $' + (f.bet_size || 0).toLocaleString(undefined,{maximumFractionDigits:0}) +
        '</div>' +
      '</div>';
    }).join('');
    if (forecasts.length > 15) {
      fcHtml += '<div style="text-align:center;padding:8px;font-size:10px;color:#2d4a6a">' + (forecasts.length - 15) + ' more opportunities not shown</div>';
    }
    fcEl.innerHTML = fcHtml;
  }
}

load();
setInterval(load, 60000);
</script></body></html>"""
