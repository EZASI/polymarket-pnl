#!/usr/bin/env python3
"""
AGENT 23 — Weather Model Latency Arbitrage
=============================================
Exploits the speed gap between raw meteorological model updates (NOAA GFS,
NWS) and human traders on Polymarket weather markets.

Strategy: "Temperature Laddering"
  1. Fetch latest NWS forecast for key cities
  2. Discover Polymarket weather markets via event slugs + page scraping
  3. Compare model forecast vs market prices
  4. Buy underpriced brackets in a bell curve cluster around the model prediction
  5. Profit from the 2-4 hour latency before humans reprice

Data Sources:
  - NWS API (api.weather.gov) — free, no key, real-time forecasts
  - NOAA GFS model data via NWS gridpoint forecasts
  - Polymarket Gamma API events endpoint for weather market data
  - Polymarket climate-science page for market discovery
"""
import json
import math
import re
from datetime import datetime, timezone, timedelta
from swarm_base import (
    SwarmAgent, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_23", "weather")

MAX_POSITIONS = 8
TAKE_PROFIT = 0.12
STOP_LOSS = 0.06
BASE_BET = 1200.0

# NWS API gridpoints for cities with Polymarket temperature markets
NWS_GRIDPOINTS = {
    "nyc":         ("OKX/33,37", "new york city"),
    "chicago":     ("LOT/75,72", "chicago"),
    "miami":       ("MFL/110,65", "miami"),
    "seattle":     ("SEW/124,67", "seattle"),
    "dallas":      ("FWD/84,108", "dallas"),
    "atlanta":     ("FFC/50,86", "atlanta"),
    "london":      (None, "london"),       # No NWS (use Open-Meteo)
    "toronto":     (None, "toronto"),       # No NWS (use Open-Meteo)
    "buenos aires":(None, "buenos aires"),  # No NWS (use Open-Meteo)
    "seoul":       (None, "seoul"),         # No NWS (use Open-Meteo)
    "ankara":      (None, "ankara"),        # No NWS (use Open-Meteo)
    "wellington":  (None, "wellington"),    # No NWS (use Open-Meteo)
}

# Open-Meteo coordinates for international cities
OPEN_METEO_COORDS = {
    "london":       (51.51, -0.13),
    "toronto":      (43.65, -79.38),
    "buenos aires": (-34.60, -58.38),
    "seoul":        (37.57, 126.98),
    "ankara":       (39.93, 32.86),
    "wellington":   (-41.29, 174.78),
}

# Polymarket slug city names (how they appear in market slugs)
SLUG_CITY_NAMES = {
    "nyc": "nyc",
    "chicago": "chicago",
    "miami": "miami",
    "seattle": "seattle",
    "dallas": "dallas",
    "atlanta": "atlanta",
    "london": "london",
    "toronto": "toronto",
    "buenos aires": "buenos-aires",
    "seoul": "seoul",
    "ankara": "ankara",
    "wellington": "wellington",
}


def fetch_nws_forecast(gridpoint):
    """Fetch temperature forecast from NWS API for a US gridpoint."""
    import requests as _req
    try:
        url = f"https://api.weather.gov/gridpoints/{gridpoint}/forecast"
        headers = {"User-Agent": "PolymarketSwarm/1.0 (research@example.com)",
                   "Accept": "application/geo+json"}
        resp = _req.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        periods = data.get("properties", {}).get("periods", [])
        if not periods:
            return None
        forecasts = []
        for p in periods[:6]:
            forecasts.append({
                "name": p.get("name", ""),
                "temp_f": p.get("temperature", 0),
                "unit": p.get("temperatureUnit", "F"),
                "wind_speed": p.get("windSpeed", ""),
                "short_forecast": p.get("shortForecast", ""),
                "is_daytime": p.get("isDaytime", True),
            })
        return forecasts
    except Exception:
        return None


def fetch_open_meteo_forecast(lat, lon):
    """Fetch temperature forecast from Open-Meteo for international cities."""
    import requests as _req
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "forecast_days": 3,
        }
        resp = _req.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])
        if not dates or not temps:
            return None
        forecasts = []
        for d, t in zip(dates, temps):
            if t is not None:
                forecasts.append({"date": d, "temp_f": round(t, 1)})
        return forecasts
    except Exception:
        return None


def discover_weather_events():
    """Discover weather market events via Gamma API event slug construction.

    Polymarket weather markets use predictable slug patterns:
      highest-temperature-in-{city}-on-{month}-{day}-{year}
    We construct slugs for today and the next 2 days, then fetch events.
    """
    import requests as _req

    now = datetime.now(timezone.utc)
    events = []

    # Month name map
    month_names = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december",
    }

    # Try today, tomorrow, and day after
    for day_offset in range(3):
        target_date = now + timedelta(days=day_offset)
        month_name = month_names[target_date.month]
        day = target_date.day
        year = target_date.year

        for city_key, slug_city in SLUG_CITY_NAMES.items():
            slug = f"highest-temperature-in-{slug_city}-on-{month_name}-{day}-{year}"
            try:
                resp = _req.get(
                    f"{GAMMA_HOST}/events",
                    params={"slug": slug},
                    timeout=10,
                )
                if resp and resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        event = data[0]
                    elif isinstance(data, dict) and data.get("id"):
                        event = data
                    else:
                        continue

                    event_markets = event.get("markets", [])
                    if event_markets:
                        events.append({
                            "event_id": event.get("id"),
                            "title": event.get("title", ""),
                            "slug": slug,
                            "city": city_key,
                            "date": target_date.strftime("%Y-%m-%d"),
                            "markets": event_markets,
                        })
            except Exception:
                continue

    # Also try other market types
    other_slugs = []
    month_name = month_names[now.month]
    year = now.year

    # Monthly temperature increase
    other_slugs.append(f"{month_name}-{year}-temperature-increase-c")
    # Precipitation
    for slug_city in ["seattle", "nyc"]:
        other_slugs.append(f"precipitation-in-{slug_city}-in-{month_name}")
    # Earthquakes
    for day_offset in range(3):
        d = now + timedelta(days=day_offset)
        week_start = d - timedelta(days=d.weekday())
        week_end = week_start + timedelta(days=6)
        other_slugs.append(
            f"how-many-6pt5-or-above-earthquakes-{month_names[week_start.month]}-"
            f"{week_start.day}-{month_names[week_end.month]}-{week_end.day}"
        )
    # Tornadoes
    other_slugs.append(f"how-many-tornadoes-in-the-us-in-{month_name}")

    for slug in other_slugs:
        try:
            resp = _req.get(
                f"{GAMMA_HOST}/events",
                params={"slug": slug},
                timeout=10,
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    event = data[0]
                elif isinstance(data, dict) and data.get("id"):
                    event = data
                else:
                    continue
                event_markets = event.get("markets", [])
                if event_markets:
                    events.append({
                        "event_id": event.get("id"),
                        "title": event.get("title", ""),
                        "slug": slug,
                        "city": None,
                        "date": now.strftime("%Y-%m-%d"),
                        "markets": event_markets,
                    })
        except Exception:
            continue

    return events


def scrape_weather_page():
    """Fallback: scrape the Polymarket climate-science/weather page for event slugs."""
    import requests as _req
    try:
        resp = _req.get(
            "https://polymarket.com/climate-science/weather",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        text = resp.text
        # Extract event slugs from href patterns
        slugs = re.findall(r'/event/([a-z0-9\-]+(?:temperature|precipitation|earthquake|tornado|weather)[a-z0-9\-]*)', text, re.IGNORECASE)
        if not slugs:
            # Broader pattern: any /event/ link on the page
            slugs = re.findall(r'/event/([a-z0-9\-]+)', text)
            # Filter for weather-related slugs
            weather_terms = ["temperature", "precipitation", "earthquake", "tornado",
                           "weather", "hurricane", "climate", "rain", "snow"]
            slugs = [s for s in slugs if any(t in s for t in weather_terms)]
        return list(set(slugs))[:30]
    except Exception:
        return []


def parse_market_bracket(question):
    """Parse a temperature bracket market question.
    Returns (temp_low, temp_high, unit) or None.
    """
    q = question.lower()
    # "between 18-19°F" or "between 18-19°f"
    m = re.search(r'between\s+(\-?\d+)\s*[-–]\s*(\-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).upper()

    # "15°F or below" / "15°f or below"
    m = re.search(r'(\-?\d+)\s*°?\s*([fFcC])\s+or\s+below', q)
    if m:
        t = int(m.group(1))
        return t - 10, t, m.group(2).upper()

    # "26°F or higher" / "26°f or higher"
    m = re.search(r'(\-?\d+)\s*°?\s*([fFcC])\s+or\s+higher', q)
    if m:
        t = int(m.group(1))
        return t, t + 10, m.group(2).upper()

    # "exactly 33°C"
    m = re.search(r'exactly\s+(\-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        t = int(m.group(1))
        return t, t, m.group(2).upper()

    # Fallback: just find a temperature number
    m = re.search(r'(\-?\d+)\s*°\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(1)), m.group(2).upper()

    return None


def calc_bracket_probability(forecast_temp, low, high, std_dev=3.0):
    """Probability that actual temp falls in [low, high] given forecast.
    Uses Gaussian distribution centered on forecast_temp."""
    if low == high:
        # Point estimate: use ±0.5 bracket
        low -= 0.5
        high += 0.5
    z_low = (low - forecast_temp) / std_dev
    z_high = (high - forecast_temp) / std_dev
    prob_low = 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
    prob_high = 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
    return max(prob_high - prob_low, 0.005)


def f_to_c(f):
    return (f - 32) * 5 / 9


def c_to_f(c):
    return c * 9 / 5 + 32


def manage_positions():
    """Exit positions that hit TP or SL."""
    closed = 0
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current = fetch_market_price(pos["condition_id"])
        if current is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        if pos.get("outcome") == "NO":
            pnl_pct = -pnl_pct
        if pnl_pct >= TAKE_PROFIT or pnl_pct <= -STOP_LOSS:
            agent.close_position(pos["condition_id"], current)
            closed += 1
    return closed


def run():
    print(f"\n{'='*60}")
    print(f"  AGENT 23 — Weather Model Latency Arbitrage")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 23 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    # Phase 1: Fetch forecasts for all cities
    city_forecasts = {}  # city -> {"temp_f": float, "source": str}

    # NWS for US cities
    seen_gridpoints = set()
    for city, (gridpoint, _label) in NWS_GRIDPOINTS.items():
        if gridpoint is None or gridpoint in seen_gridpoints:
            continue
        seen_gridpoints.add(gridpoint)
        fc = fetch_nws_forecast(gridpoint)
        if fc:
            daytime = [p for p in fc if p["is_daytime"]]
            if daytime:
                temp_f = daytime[0]["temp_f"]
                city_forecasts[city] = {"temp_f": temp_f, "source": "NWS"}
                print(f"  NWS {city.upper()}: {temp_f}°F ({daytime[0]['short_forecast'][:30]})")

    # Open-Meteo for international cities
    for city, (lat, lon) in OPEN_METEO_COORDS.items():
        fc = fetch_open_meteo_forecast(lat, lon)
        if fc:
            # Use today's max temp
            city_forecasts[city] = {"temp_f": fc[0]["temp_f"], "source": "Open-Meteo"}
            print(f"  METEO {city.upper()}: {fc[0]['temp_f']}°F (max today)")

    if not city_forecasts:
        print("  WARNING: No forecast data available")

    # Phase 2: Discover weather markets on Polymarket
    print("  Discovering weather markets...")
    events = discover_weather_events()
    print(f"  Found {len(events)} weather events via slug construction")

    # If slug construction found nothing, try page scraping
    if len(events) < 3:
        print("  Trying page scrape fallback...")
        scraped_slugs = scrape_weather_page()
        if scraped_slugs:
            print(f"  Scraped {len(scraped_slugs)} weather slugs from page")
            import requests as _req
            for slug in scraped_slugs:
                # Skip slugs we already have
                if any(e["slug"] == slug for e in events):
                    continue
                try:
                    resp = _req.get(
                        f"{GAMMA_HOST}/events",
                        params={"slug": slug},
                        timeout=10,
                    )
                    if resp and resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list) and data:
                            event = data[0]
                        elif isinstance(data, dict) and data.get("id"):
                            event = data
                        else:
                            continue
                        event_markets = event.get("markets", [])
                        if event_markets:
                            # Try to match city
                            title = event.get("title", "").lower()
                            matched_city = None
                            for ck in SLUG_CITY_NAMES:
                                if ck in title or SLUG_CITY_NAMES[ck].replace("-", " ") in title:
                                    matched_city = ck
                                    break
                            events.append({
                                "event_id": event.get("id"),
                                "title": event.get("title", ""),
                                "slug": slug,
                                "city": matched_city,
                                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                "markets": event_markets,
                            })
                except Exception:
                    continue

    # Flatten all markets from all events
    weather_markets = []
    for evt in events:
        city = evt.get("city")
        for m in evt.get("markets", []):
            question = m.get("question", "")
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except Exception:
                    continue
            if not prices:
                continue
            price = float(prices[0])
            cid = m.get("conditionId", m.get("condition_id", ""))
            if not cid:
                continue
            weather_markets.append({
                "question": question,
                "condition_id": cid,
                "price": price,
                "volume": float(m.get("volume", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "city": city,
                "event_title": evt.get("title", ""),
                "event_slug": evt.get("slug", ""),
            })

    print(f"  Total weather markets: {len(weather_markets)}")

    # Phase 3: Compare model forecasts vs market prices
    signals = []
    n_open = len([p for p in agent.positions if p.get("status") == "open"])

    for wm in weather_markets:
        q = wm["question"]
        city = wm.get("city")
        price = wm["price"]
        cid = wm["condition_id"]

        bracket = parse_market_bracket(q)
        if not bracket:
            continue

        low, high, unit = bracket

        # Find matching forecast
        forecast = city_forecasts.get(city) if city else None
        if not forecast:
            continue

        forecast_f = forecast["temp_f"]

        # Convert if market is in Celsius
        if unit == "C":
            forecast_temp = f_to_c(forecast_f)
            std_dev = 1.7  # ~3°F in Celsius
        else:
            forecast_temp = forecast_f
            std_dev = 3.0

        model_prob = calc_bracket_probability(forecast_temp, low, high, std_dev)

        # Edge = model probability - market price
        edge = model_prob - price
        if abs(edge) < 0.08:
            continue

        direction = "YES" if edge > 0 else "NO"
        eff_price = price if direction == "YES" else (1 - price)
        confidence = min(abs(edge) / 0.15, 1.0)

        sig = {
            "market": q[:80], "condition_id": cid,
            "price": round(price, 3),
            "direction": direction,
            "forecast_temp": round(forecast_temp, 1),
            "bracket": f"{low}-{high}°{unit}",
            "model_prob": round(model_prob, 3),
            "edge": round(edge, 3),
            "confidence": round(confidence, 2),
            "city": city or "unknown",
            "strategy": "temp_ladder",
        }
        signals.append(sig)

        bet = min(BASE_BET * confidence, agent.capital * 0.10)
        if bet >= 10 and n_open < MAX_POSITIONS:
            reason = (f"TempLadder: fcst={forecast_temp:.0f}° "
                      f"bracket={low}-{high}°{unit} edge={edge:+.1%}")
            if agent.open_position(q, cid, direction, bet, eff_price, reason):
                n_open += 1
                print(f"  TRADE: {direction} @ {eff_price:.3f} | "
                      f"fcst={forecast_temp:.0f}° bracket={low}-{high}°{unit} "
                      f"edge={edge:+.1%} | {q[:40]}")

    # Write signals log
    log_data = {
        "agent": "agent_23", "strategy": "weather_latency_arb",
        "time": datetime.now(timezone.utc).isoformat(),
        "cities_fetched": list(city_forecasts.keys()),
        "weather_events_found": len(events),
        "weather_markets_found": len(weather_markets),
        "signals": signals[:20],
        "forecasts": {city: fc for city, fc in city_forecasts.items()},
        "events_discovered": [{"title": e["title"], "slug": e["slug"],
                               "n_markets": len(e["markets"])} for e in events],
        "summary": agent.get_summary(),
    }
    with open(LOG_DIR / "signals_agent_23.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Events: {len(events)} | Markets: {len(weather_markets)} | Signals: {len(signals)}")
    print(f"  Cities: {', '.join(city_forecasts.keys())}")
    print(f"  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
