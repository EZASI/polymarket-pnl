#!/usr/bin/env python3
"""
MTY MULTI-CATEGORY PREDICTOR v2 — Enhanced Edge Detection
===========================================================

Multi-source prediction engine for ALL Polymarket categories.
Each category uses specialized data pipelines for edge detection:

  SPORTS:   The Odds API (Vegas vs Polymarket mispricing) + copy signals
  CRYPTO:   Binance TA (RSI/MACD/Bollinger) for BTC/ETH price targets
  WEATHER:  Open-Meteo multi-model ensemble
  ALL:      Copy signal convergence + market microstructure analysis

Edge Sources (ranked by reliability):
  1. Sportsbook-vs-Polymarket mispricing (3-15% edges found)
  2. Copy signal convergence (2+ top traders agree)
  3. Crypto TA divergence from market price
  4. Weather model ensemble vs market
  5. Market microstructure (volume spikes, price momentum)

Results saved to logs/all_predictions.json for the dashboard.

Usage:
    python3 multi_predictor.py              # full scan
    python3 multi_predictor.py --category sports
    python3 multi_predictor.py --category crypto
"""

import argparse
import json
import logging
import os
import re
import time
import warnings
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multi_predictor")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

GAMMA_HOST = "https://gamma-api.polymarket.com"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
CLOB_HOST = "https://clob.polymarket.com"


# ============================================================
# 1. FETCH ALL POLYMARKET MARKETS
# ============================================================
def fetch_all_markets(limit=2000):
    """Fetch all active Polymarket markets."""
    all_markets = []
    offset = 0
    while offset < limit:
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/markets",
                params={"limit": 100, "active": "true", "closed": "false", "offset": offset},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            all_markets.extend(data)
            offset += 100
            if len(data) < 100:
                break
        except Exception as e:
            log.error(f"Market fetch error at offset {offset}: {e}")
            break

    # Also fetch markets referenced by copy signals (daily games etc.)
    copy_signals = load_copy_signals()
    existing_cids = {m.get("conditionId", "") for m in all_markets}
    for sig in copy_signals:
        cid = sig.get("conditionId", "")
        if cid and cid not in existing_cids:
            try:
                resp = requests.get(
                    f"{GAMMA_HOST}/markets",
                    params={"condition_id": cid},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        all_markets.extend(data)
                        existing_cids.add(cid)
            except:
                pass

    log.info(f"Fetched {len(all_markets)} active markets")
    return all_markets


# ============================================================
# 2. CATEGORIZE MARKETS
# ============================================================
CATEGORIES = {
    "weather": {
        "keywords": ["temperature", "weather forecast", "degrees fahrenheit",
                     "degrees celsius", "rainfall", "snowfall", "heat wave",
                     "coldest", "hottest", "warmest", "record high", "record low",
                     "average temperature", "hurricane", "tornado", "blizzard"],
        "icon": "cloud",
        "color": "#4da6ff",
    },
    "sports": {
        "keywords": ["nba", "nfl", "mlb", "nhl", "soccer", "super bowl", "superbowl",
                     "spread", "o/u ", "moneyline", "championship", "stanley cup",
                     "world series", "finals", "playoff", "lakers", "celtics",
                     "warriors", "rockets", "thunder", "cavaliers", "bucks", "nuggets",
                     "suns", "knicks", "nets", "grizzlies", "pacers", "hawks",
                     "magic", "mavericks", "clippers", "pelicans", "kings", "spurs",
                     "blazers", "raptors", "jazz", "wizards", "hornets", "pistons",
                     "timberwolves", "bulls", "76ers", "chiefs", "eagles", "cowboys",
                     "patriots", "seahawks", "ravens", "bills", "49ers", "steelers",
                     "packers", "dolphins", "bengals", "lions", "texans", "chargers",
                     "broncos", "vikings", "jaguars", "colts", "commanders",
                     "falcons", "cardinals", "bears", "giants", "jets", "panthers",
                     "saints", "rams", "buccaneers", "titans",
                     "premier league", "la liga", "serie a", "bundesliga",
                     "champions league", "uefa", "fifa", "mls", "ufc ", "boxing",
                     "tennis", "wimbledon", "us open", "french open", "australian open",
                     "pga", "golf", "masters", "formula 1", "f1 ", "grand prix",
                     "ncaa", "march madness", "college", "heisman",
                     "mvp", "rookie of the year", "scoring title",
                     "win the nba", "win the nfl", "win the nhl", "win the mlb",
                     "win the super bowl", "win the world series",
                     "regular season wins", "game 7",
                     "atletico", "barcelona", "real madrid", "manchester united",
                     "manchester city", "liverpool", "chelsea", "arsenal",
                     "tottenham", "bayern munich", "ac milan", "inter milan",
                     "win super bowl", "win stanley cup"],
        "icon": "trophy",
        "color": "#00d4aa",
    },
    "politics": {
        "keywords": ["trump", "biden", "election", "president", "congress", "senate",
                     "governor", "democrat", "republican", "poll", "vote", "cabinet",
                     "tariff", "executive order", "impeach", "legislation", "political",
                     "deport", "immigration", "doge ", "government shutdown",
                     "indictment", "supreme court", "scotus", "attorney general",
                     "secretary of", "ambassador", "pardon", "veto", "filibuster",
                     "primary", "nominee", "campaign", "midterm", "electoral",
                     "speaker of", "majority leader", "minority leader",
                     "colombian", "colombia", "seats in", "brazil", "mexico election",
                     "uk election", "parliament", "prime minister",
                     "macron", "trudeau", "modi", "erdogan", "netanyahu",
                     "party", "lib dem", "labour", "conservative",
                     "policy", "regulation", "sanctions", "ban ",
                     "approve", "confirm", "nomination", "appointed"],
        "icon": "landmark",
        "color": "#a78bfa",
    },
    "crypto": {
        "keywords": ["bitcoin", " btc ", "ethereum", " eth ", "crypto", "solana",
                     " sol ", "dogecoin", "token", "defi", "blockchain", "altcoin",
                     "xrp", "ripple", "cardano", "ada ", "polygon", "matic",
                     "avalanche", "avax", "chainlink", "link ", "uniswap",
                     "stablecoin", "usdc", "usdt", "tether", "binance",
                     "coinbase", "memecoin", "nft", "web3",
                     "crypto market cap", "halving", "mining",
                     "btc price", "eth price"],
        "icon": "coins",
        "color": "#ffd43b",
    },
    "entertainment": {
        "keywords": ["oscar", "academy award", "grammy", "emmy", "movie", "film",
                     "actor", "actress", "netflix", "disney", "album", "song",
                     "music", "box office", "streaming", "best picture",
                     "best director", "best supporting", "best cinematography",
                     "best international", "golden globe", "bafta", "tony award",
                     "pulitzer", "youtube", "tiktok", "viral", "podcast",
                     "tv show", "series finale", "renewal", "cancel",
                     "beyonce", "taylor swift", "drake", "kanye",
                     "marvel", "star wars", "game of thrones"],
        "icon": "star",
        "color": "#ff8c42",
    },
    "economics": {
        "keywords": ["fed ", "federal reserve", "interest rate", "inflation", "gdp",
                     "unemployment", "recession", "stock market", "s&p 500",
                     "nasdaq", "dow jones", "treasury", "economy", "cpi",
                     "jobs report", "federal spending", "debt ceiling",
                     "rate cut", "rate hike", "quantitative", "yield",
                     "bond", "market cap", "ipo", "earnings",
                     "revenue", "profit", "quarterly", "annual",
                     "trade deficit", "surplus", "tariff rate",
                     "oil price", "gas price", "commodity",
                     "housing market", "mortgage rate"],
        "icon": "chart-line",
        "color": "#66d9e8",
    },
    "science_tech": {
        "keywords": ["spacex", "nasa", "mars", "moon landing", "quantum",
                     "apple ", "iphone", "openai", "chatgpt", "google ",
                     "microsoft", "meta ", "tesla ", "self-driving", "robot",
                     "ai ", "artificial intelligence", "gpt", "llm",
                     "machine learning", "deepmind", "anthropic", "claude",
                     "launch", "satellite", "starship", "rocket",
                     "fda approval", "drug trial", "vaccine", "clinical",
                     "gene therapy", "crispr", "biotech",
                     "breakthrough", "discovery", "patent"],
        "icon": "flask",
        "color": "#ff6b6b",
    },
    "olympics": {
        "keywords": ["olympic", "winter games", "gold medal", "medal count",
                     "winter olympic", "2026 olympics", "ice hockey gold",
                     "biathlon", "alpine skiing", "summer games",
                     "world championship", "world cup"],
        "icon": "medal",
        "color": "#ffd43b",
    },
    "world": {
        "keywords": ["war", "ukraine", "russia", "ceasefire", "china",
                     "nato", "military", "peace deal", "sanctions",
                     "united nations", "middle east", "israel",
                     "taiwan", "north korea", "iran", "syria",
                     "refugee", "humanitarian", "coup", "invasion",
                     "territorial", "border", "conflict", "treaty",
                     "diplomacy", "geopolitical", "alliance"],
        "icon": "globe",
        "color": "#4dabf7",
    },
    "esports": {
        "keywords": ["valorant", "league of legends", "cs2", "dota",
                     "esport", "vct ", "worlds ", "major ",
                     "overwatch", "fortnite", "apex legends",
                     "twitch", "streamer"],
        "icon": "gamepad",
        "color": "#b197fc",
    },
    "pop_culture": {
        "keywords": ["twitter", "x.com", "elon musk tweet", "viral",
                     "celebrity", "scandal", "controversy", "podcast",
                     "influencer", "social media", "trending",
                     "bet ", "dare ", "challenge"],
        "icon": "fire",
        "color": "#ff6b9d",
    },
}


def categorize_market(market):
    """Assign a category to a market using keyword scoring."""
    q = market.get("question", "").lower()
    slug = market.get("slug", "").lower()
    desc = (market.get("description", "") or "").lower()
    full = f"{q} {slug} {desc}"

    # Score each category by number of keyword matches (most matches wins)
    best_cat = "other"
    best_score = 0

    for cat_name, cat_info in CATEGORIES.items():
        score = sum(1 for kw in cat_info["keywords"] if kw in full)
        # Bonus for title-level matches (more relevant than description)
        title_score = sum(1 for kw in cat_info["keywords"] if kw in q)
        total = score + title_score * 2  # Title matches count 3x

        if total > best_score:
            best_score = total
            best_cat = cat_name

    return best_cat


def categorize_all(markets):
    """Categorize all markets."""
    result = {}
    for m in markets:
        cat = categorize_market(m)
        if cat not in result:
            result[cat] = []
        result[cat].append(m)
    return result


# ============================================================
# 3a. SPORTSBOOK ODDS COMPARISON (The Odds API)
# ============================================================
def fetch_sportsbook_odds():
    """Fetch real sportsbook odds from The Odds API for edge detection."""
    if not ODDS_API_KEY:
        log.info("No ODDS_API_KEY set — skipping sportsbook odds")
        return {}

    odds_data = {}
    # Sports we can compare against Polymarket
    sports = [
        ("basketball_nba", "NBA"),
        ("americanfootball_nfl", "NFL"),
        ("soccer_epl", "EPL"),
        ("soccer_uefa_champs_league", "UCL"),
        ("mma_mixed_martial_arts", "UFC"),
        ("baseball_mlb", "MLB"),
        ("icehockey_nhl", "NHL"),
    ]

    for sport_key, label in sports:
        try:
            resp = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                try:
                    err = resp.json()
                    if "OUT_OF_USAGE_CREDITS" in str(err):
                        log.warning("Odds API quota exhausted")
                        return odds_data  # Return what we have
                except:
                    pass
                log.debug(f"Odds API {resp.status_code} for {label}")
                continue
            events = resp.json()
            for event in events:
                home = event.get("home_team", "")
                away = event.get("away_team", "")
                commence = event.get("commence_time", "")

                # Average across all bookmakers for consensus
                home_odds_list = []
                away_odds_list = []
                for bm in event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") == "h2h":
                            for oc in mkt.get("outcomes", []):
                                if oc["name"] == home:
                                    home_odds_list.append(float(oc["price"]))
                                elif oc["name"] == away:
                                    away_odds_list.append(float(oc["price"]))

                if home_odds_list and away_odds_list:
                    avg_home = np.mean(home_odds_list)
                    avg_away = np.mean(away_odds_list)
                    # Convert decimal odds to implied probability (remove vig)
                    raw_home = 1 / avg_home
                    raw_away = 1 / avg_away
                    total = raw_home + raw_away
                    fair_home = raw_home / total  # vig-free probability
                    fair_away = raw_away / total

                    # Also get sharpest line (lowest vig book = Pinnacle/Betfair)
                    sharp_home = min(home_odds_list)  # highest prob = lowest odds
                    sharp_away = min(away_odds_list)

                    key = f"{away} vs {home}".lower()
                    odds_data[key] = {
                        "sport": label,
                        "home": home,
                        "away": away,
                        "home_prob": round(fair_home, 4),
                        "away_prob": round(fair_away, 4),
                        "home_odds_avg": round(avg_home, 3),
                        "away_odds_avg": round(avg_away, 3),
                        "n_books": len(home_odds_list),
                        "spread": round(max(home_odds_list) - min(home_odds_list), 3),
                        "commence": commence,
                    }
                    # Also index by individual team names
                    odds_data[home.lower()] = odds_data[key]
                    odds_data[away.lower()] = odds_data[key]
            time.sleep(0.2)
        except Exception as e:
            log.debug(f"Odds API error for {label}: {e}")

    log.info(f"Fetched sportsbook odds for {len(odds_data)} matchups")
    return odds_data


def find_odds_for_market(market_title, odds_data):
    """Find sportsbook odds matching a Polymarket market title."""
    title_lower = market_title.lower()

    # Direct team name search
    for key, odds in odds_data.items():
        if not isinstance(odds, dict) or "home" not in odds:
            continue
        home_l = odds["home"].lower()
        away_l = odds["away"].lower()
        # Check if both teams appear in the title
        if home_l in title_lower and away_l in title_lower:
            return odds
        # Check partial matches (team last names)
        home_parts = home_l.split()
        away_parts = away_l.split()
        if len(home_parts) > 1 and len(away_parts) > 1:
            if home_parts[-1] in title_lower and away_parts[-1] in title_lower:
                return odds
    return None


# ============================================================
# 3b. CRYPTO TECHNICAL ANALYSIS (Binance)
# ============================================================
def get_crypto_ta(symbol="BTCUSDT", interval="1h", limit=100):
    """Get technical analysis indicators from Binance klines."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        klines = resp.json()
        closes = np.array([float(k[4]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])

        if len(closes) < 26:
            return None

        current = closes[-1]

        # RSI (14-period)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / max(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9) — compute EMA series for signal line
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_val = ema12 - ema26

        # Compute MACD values over recent window for signal line
        macd_series = []
        for i in range(min(9, len(closes) - 26)):
            if i == 0:
                macd_series.insert(0, macd_val)
            else:
                e12 = _ema(closes[:-i], 12)
                e26 = _ema(closes[:-i], 26)
                macd_series.insert(0, e12 - e26)

        signal_line = np.mean(macd_series) if macd_series else macd_val
        macd_hist = macd_val - signal_line

        # Bollinger Bands (20, 2)
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_pct = (current - bb_lower) / max(bb_upper - bb_lower, 1e-10)

        # Volume trend
        vol_avg = np.mean(volumes[-20:])
        vol_current = np.mean(volumes[-3:])
        vol_ratio = vol_current / max(vol_avg, 1)

        # Price momentum (7-day and 1-day)
        momentum_7d = (current - closes[-min(168, len(closes))]) / closes[-min(168, len(closes))] * 100 if len(closes) >= 24 else 0
        momentum_1d = (current - closes[-min(24, len(closes))]) / closes[-min(24, len(closes))] * 100 if len(closes) >= 2 else 0

        # Support/resistance (recent highs/lows)
        recent_high = float(np.max(highs[-48:]))
        recent_low = float(np.min(lows[-48:]))

        return {
            "price": round(current, 2),
            "rsi": round(float(rsi), 1),
            "macd": round(float(macd_val), 2),
            "macd_signal": round(float(signal_line), 2),
            "macd_hist": round(float(macd_hist), 2),
            "bb_upper": round(float(bb_upper), 2),
            "bb_lower": round(float(bb_lower), 2),
            "bb_pct": round(float(bb_pct), 3),
            "vol_ratio": round(float(vol_ratio), 2),
            "momentum_7d": round(float(momentum_7d), 2),
            "momentum_1d": round(float(momentum_1d), 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "sma20": round(float(sma20), 2),
        }
    except Exception as e:
        log.debug(f"Crypto TA error for {symbol}: {e}")
        return None


def _ema(data, period):
    """Calculate EMA for the last value."""
    if len(data) < period:
        return data[-1]
    multiplier = 2 / (period + 1)
    ema = data[-period]
    for price in data[-period + 1:]:
        ema = (price - ema) * multiplier + ema
    return ema


def _ema_from_array(data, period):
    """Calculate final EMA value from array."""
    if len(data) == 0:
        return 0
    return _ema(data, min(period, len(data)))


def estimate_crypto_prob(market_question, ta_data):
    """
    Estimate probability for crypto price target markets using TA.
    Handles: "Will BTC hit $150k?", "Bitcoin above $100,000", "ETH reach $5000"
    """
    if not ta_data:
        return None, None

    q = market_question.lower()
    current = ta_data["price"]

    # Extract price target — handle $150k, $1m, $100,000, $5000
    target_match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(k|m|b|K|M|B)?', q)
    if not target_match:
        return None, None

    target_str = target_match.group(1).replace(",", "")
    target = float(target_str)
    suffix = (target_match.group(2) or "").lower()

    if suffix == "k":
        target *= 1000
    elif suffix == "m":
        target *= 1_000_000
    elif suffix == "b":
        target *= 1_000_000_000
    elif target < 500:
        # Likely shorthand: "$150" meaning $150k for BTC
        if current > 10000:
            target *= 1000

    # Determine direction (above or below)
    above = any(w in q for w in ["above", "exceed", "over", "higher than", "reach", "hit", "break", "cross", "surpass"])
    below = any(w in q for w in ["below", "under", "drop", "fall", "lower than", "crash", "dip"])
    if not above and not below:
        above = True  # default assumption for "hit" style markets

    # Calculate distance to target
    distance_pct = (target - current) / current * 100

    # Base probability from distance + momentum
    if above:
        # Probability of going UP to target
        if distance_pct <= 0:
            # Already above target
            base_prob = 0.85 + min(abs(distance_pct) * 0.005, 0.10)
        elif distance_pct < 5:
            base_prob = 0.60 + ta_data["momentum_1d"] * 0.02
        elif distance_pct < 15:
            base_prob = 0.35 + ta_data["momentum_7d"] * 0.01
        else:
            base_prob = max(0.05, 0.20 - distance_pct * 0.005)
    else:
        # Probability of going DOWN to target
        if distance_pct >= 0:
            # Already below target
            base_prob = 0.85 + min(abs(distance_pct) * 0.005, 0.10)
        elif distance_pct > -5:
            base_prob = 0.60 - ta_data["momentum_1d"] * 0.02
        elif distance_pct > -15:
            base_prob = 0.35 - ta_data["momentum_7d"] * 0.01
        else:
            base_prob = max(0.05, 0.20 + distance_pct * 0.005)

    # RSI adjustment
    if ta_data["rsi"] > 70:
        # Overbought — less likely to go higher
        if above:
            base_prob -= 0.08
        else:
            base_prob += 0.05
    elif ta_data["rsi"] < 30:
        # Oversold — less likely to go lower
        if above:
            base_prob += 0.05
        else:
            base_prob -= 0.08

    # MACD adjustment
    if ta_data["macd_hist"] > 0:
        # Bullish momentum
        if above:
            base_prob += 0.03
        else:
            base_prob -= 0.03
    else:
        if above:
            base_prob -= 0.03
        else:
            base_prob += 0.03

    # Bollinger Band adjustment
    if ta_data["bb_pct"] > 0.95:
        # Near upper band — likely reversal
        if above:
            base_prob -= 0.05
    elif ta_data["bb_pct"] < 0.05:
        # Near lower band — likely bounce
        if below:
            base_prob -= 0.05

    # Volume confirmation
    if ta_data["vol_ratio"] > 1.5:
        # High volume confirms trend
        if ta_data["momentum_1d"] > 0 and above:
            base_prob += 0.04
        elif ta_data["momentum_1d"] < 0 and below:
            base_prob += 0.04

    base_prob = max(0.03, min(0.97, base_prob))
    confidence = min(abs(base_prob - 0.5) * 2.5, 0.90)

    details = {
        "target": target,
        "current": current,
        "distance_pct": round(distance_pct, 2),
        "direction": "above" if above else "below",
        "rsi": ta_data["rsi"],
        "macd_hist": ta_data["macd_hist"],
        "bb_pct": ta_data["bb_pct"],
        "vol_ratio": ta_data["vol_ratio"],
        "momentum_1d": ta_data["momentum_1d"],
        "momentum_7d": ta_data["momentum_7d"],
    }
    return round(base_prob, 3), details


# ============================================================
# 3c. MARKET MICROSTRUCTURE ANALYSIS
# ============================================================
def analyze_market_microstructure(market):
    """Analyze Polymarket market for volume/price signals."""
    vol = float(market.get("volume", 0) or 0)
    liq = float(market.get("liquidity", 0) or 0)

    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            prices = []

    market_price = float(prices[0]) if prices else 0.5

    signals = {}

    # Volume-to-liquidity ratio (high = lots of activity vs available liquidity)
    if liq > 0:
        vol_liq = vol / liq
        signals["vol_liq_ratio"] = round(vol_liq, 2)
        if vol_liq > 50:
            signals["volume_signal"] = "HIGH_ACTIVITY"
        elif vol_liq > 20:
            signals["volume_signal"] = "ACTIVE"
        else:
            signals["volume_signal"] = "NORMAL"
    else:
        signals["vol_liq_ratio"] = 0
        signals["volume_signal"] = "LOW_LIQ"

    # Price extremity (close to 0 or 1 = market has strong conviction)
    signals["price_extremity"] = round(abs(market_price - 0.5) * 2, 3)

    # Liquidity assessment
    if liq > 100000:
        signals["liquidity_grade"] = "A"
    elif liq > 10000:
        signals["liquidity_grade"] = "B"
    elif liq > 1000:
        signals["liquidity_grade"] = "C"
    else:
        signals["liquidity_grade"] = "D"

    signals["volume"] = vol
    signals["liquidity"] = liq

    return signals


# ============================================================
# 3d. ENHANCED EDGE SCORING
# ============================================================
def calculate_enhanced_edge(our_prob, market_price, sources):
    """
    Calculate edge with confidence weighting from multiple sources.
    sources: list of {"name": str, "prob": float, "weight": float}
    """
    if not sources:
        return our_prob - market_price, 0

    # Weighted average probability
    total_weight = sum(s["weight"] for s in sources)
    if total_weight == 0:
        return our_prob - market_price, 0

    weighted_prob = sum(s["prob"] * s["weight"] for s in sources) / total_weight

    # Edge
    edge = weighted_prob - market_price

    # Confidence: how much sources agree
    if len(sources) > 1:
        probs = [s["prob"] for s in sources]
        agreement = 1 - np.std(probs) * 2  # low std = high agreement
        confidence = max(0, min(agreement, 0.95)) * min(abs(edge) * 5, 1)
    else:
        confidence = min(abs(edge) * 3, 0.95)

    return round(edge, 4), round(confidence, 3)


def copy_signal_to_prob(market_price, avg_wr, n_traders, strength="WEAK"):
    """Convert copy signal data into a probability estimate.

    Key insight: A trader's overall win rate is NOT the probability of a
    specific market outcome.  A trader with 100% WR on 3 trades doesn't
    mean the next market has 95% probability.

    Instead, we treat the copy signal as a *bounded adjustment* to the
    market price (crowd consensus). The adjustment scales with:
      - Number of traders agreeing (more = stronger signal)
      - Their average win rate (higher = more credible)
      - Signal strength classification

    Max adjustment caps per scenario:
      1 trader, any WR  -> up to ±8%
      2 traders          -> up to ±12%
      3+ traders         -> up to ±18%
      STRONG + 3+ traders -> up to ±22%
    """
    # Base adjustment: how much the copy signal moves the price
    # WR contribution: a 70% WR trader is mildly bullish, 90%+ is stronger
    wr_factor = max(0, (avg_wr - 55)) / 45.0  # 0 at 55% WR, 1.0 at 100% WR

    # Trader count multiplier
    trader_mult = {1: 0.4, 2: 0.65, 3: 0.85, 4: 1.0}.get(
        min(n_traders, 4), 1.0
    )

    # Strength bonus
    str_mult = {"STRONG": 1.2, "MODERATE": 1.0, "WEAK": 0.7}.get(strength, 0.7)

    # Maximum adjustment: 22% for best case, ~5% for weakest
    max_adj = 0.22 * trader_mult * str_mult
    adjustment = wr_factor * max_adj

    # Apply adjustment to market price (nudge toward "Yes")
    copy_prob = market_price + adjustment

    # Bound: never let copy signal push probability below 0.03 or above 0.97
    copy_prob = max(0.03, min(0.97, copy_prob))

    return round(copy_prob, 4)


# ============================================================
# 4. WEATHER PREDICTIONS (Open-Meteo Multi-Model)
# ============================================================

# Cities tracked on Polymarket weather markets
WEATHER_SLUG_CITIES = {
    "nyc": {"slug": "nyc", "coords": (40.7128, -74.0060)},
    "chicago": {"slug": "chicago", "coords": (41.8781, -87.6298)},
    "miami": {"slug": "miami", "coords": (25.7617, -80.1918)},
    "seattle": {"slug": "seattle", "coords": (47.6062, -122.3321)},
    "dallas": {"slug": "dallas", "coords": (32.7767, -96.7970)},
    "atlanta": {"slug": "atlanta", "coords": (33.7490, -84.3880)},
    "london": {"slug": "london", "coords": (51.51, -0.13)},
    "toronto": {"slug": "toronto", "coords": (43.65, -79.38)},
    "buenos aires": {"slug": "buenos-aires", "coords": (-34.60, -58.38)},
    "seoul": {"slug": "seoul", "coords": (37.57, 126.98)},
    "ankara": {"slug": "ankara", "coords": (39.93, 32.86)},
    "wellington": {"slug": "wellington", "coords": (-41.29, 174.78)},
}

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


def discover_weather_markets():
    """Discover Polymarket weather markets via slug construction.

    The Gamma API standard search doesn't surface weather markets.
    Instead, we construct predictable event slugs and fetch directly.
    """
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    events = []

    # Temperature markets: highest-temperature-in-{city}-on-{month}-{day}-{year}
    for day_offset in range(3):  # today, tomorrow, day after
        target = now + timedelta(days=day_offset)
        month = MONTH_NAMES[target.month]
        day = target.day
        year = target.year

        for city_key, info in WEATHER_SLUG_CITIES.items():
            slug = f"highest-temperature-in-{info['slug']}-on-{month}-{day}-{year}"
            try:
                resp = requests.get(
                    f"{GAMMA_HOST}/events",
                    params={"slug": slug},
                    timeout=10,
                )
                if resp and resp.status_code == 200:
                    data = resp.json()
                    event = data[0] if isinstance(data, list) and data else data
                    if not event or not event.get("markets"):
                        continue
                    events.append({
                        "event_id": event.get("id"),
                        "title": event.get("title", ""),
                        "slug": slug,
                        "city": city_key,
                        "date": target.strftime("%Y-%m-%d"),
                        "markets": event.get("markets", []),
                    })
            except Exception:
                continue

    # Other weather market types
    month = MONTH_NAMES[now.month]
    year = now.year
    other_slugs = [
        (f"{month}-{year}-temperature-increase-c", None),
        (f"precipitation-in-seattle-in-{month}", "seattle"),
        (f"precipitation-in-nyc-in-{month}", "nyc"),
        (f"how-many-tornadoes-in-the-us-in-{month}", None),
    ]
    for slug, city in other_slugs:
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/events", params={"slug": slug}, timeout=10,
            )
            if resp and resp.status_code == 200:
                data = resp.json()
                event = data[0] if isinstance(data, list) and data else data
                if event and event.get("markets"):
                    events.append({
                        "event_id": event.get("id"),
                        "title": event.get("title", ""),
                        "slug": slug,
                        "city": city,
                        "date": now.strftime("%Y-%m-%d"),
                        "markets": event.get("markets", []),
                    })
        except Exception:
            continue

    return events


def predict_weather(markets):
    """Predict weather markets using slug discovery + Open-Meteo forecasts."""
    predictions = []

    # First: analyze any weather markets found in the standard Gamma fetch
    for m in markets:
        pred = analyze_weather_market(m)
        if pred:
            predictions.append(pred)

    # Second: discover real weather markets via slug construction
    log.info("  Discovering weather markets via slug construction...")
    events = discover_weather_markets()
    log.info(f"  Found {len(events)} weather events ({sum(len(e['markets']) for e in events)} markets)")

    # Fetch forecasts for cities we need — store ALL days, keyed by date
    forecast_cache = {}  # city -> {date_str: {"temp_f": float}}
    for evt in events:
        city = evt.get("city")
        if not city or city in forecast_cache:
            continue
        info = WEATHER_SLUG_CITIES.get(city)
        if info:
            lat, lon = info["coords"]
            fc = get_weather_forecast(lat, lon, days=5)
            if fc and "daily" in fc:
                daily = fc["daily"]
                temps = daily.get("temperature_2m_max", [])
                dates = daily.get("time", [])
                city_forecasts = {}
                for i, d in enumerate(dates):
                    if i < len(temps) and temps[i] is not None:
                        city_forecasts[d] = {"temp_f": temps[i]}
                if city_forecasts:
                    forecast_cache[city] = city_forecasts
                    fc_str = ', '.join('%s=%.1fF' % (d, v["temp_f"]) for d, v in city_forecasts.items())
                    log.info("    %s forecasts: %s" % (city, fc_str))

    # Analyze each market from discovered events
    for evt in events:
        city = evt.get("city")
        for m in evt.get("markets", []):
            question = m.get("question", "")
            slug = m.get("slug", "")
            cid = m.get("conditionId", "")
            if not cid:
                continue

            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except Exception:
                    continue
            if not prices:
                continue
            market_price = float(prices[0])
            if market_price >= 0.98 or market_price <= 0.02:
                continue

            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)

            # Parse temperature bracket from question
            bracket = _parse_temp_bracket(question)
            city_forecasts = forecast_cache.get(city) if city else None

            # Match event date to correct forecast day
            evt_date = evt.get("date", "")  # e.g. "2026-02-09"
            forecast = None
            if city_forecasts and evt_date:
                forecast = city_forecasts.get(evt_date)
                if not forecast:
                    # Fallback: try the nearest available date
                    for d in sorted(city_forecasts.keys()):
                        if d >= evt_date:
                            forecast = city_forecasts[d]
                            break
                    if not forecast:
                        # Use latest available
                        forecast = list(city_forecasts.values())[-1]

            if bracket and forecast:
                low, high, unit = bracket
                temp_f = forecast["temp_f"]

                # Convert forecast to match market unit
                # Std dev reflects typical NWS/model forecast error:
                #   ~3.5°F (2°C) for day 0, ~4.5°F (2.5°C) for day 1-2
                import math
                from datetime import datetime, timezone
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                days_ahead = 0
                if evt_date > today_str:
                    try:
                        d0 = date.fromisoformat(today_str)
                        d1 = date.fromisoformat(evt_date)
                        days_ahead = (d1 - d0).days
                    except Exception:
                        pass
                if unit == "C":
                    forecast_temp = (temp_f - 32) * 5 / 9
                    std_dev = 2.0 + 0.5 * days_ahead  # 2.0°C today, 2.5°C tomorrow, 3.0°C day after
                else:
                    forecast_temp = temp_f
                    std_dev = 3.5 + 1.0 * days_ahead  # 3.5°F today, 4.5°F tomorrow, 5.5°F day after

                # Gaussian probability
                import math
                # Handle open-ended brackets (None = infinity)
                if low is None:
                    # "X or below" → P(temp <= high)
                    z_hi = (high + 0.5 - forecast_temp) / std_dev
                    our_prob = 0.5 * (1 + math.erf(z_hi / math.sqrt(2)))
                elif high is None:
                    # "X or higher" → P(temp >= low) = 1 - P(temp < low)
                    z_lo = (low - 0.5 - forecast_temp) / std_dev
                    our_prob = 1.0 - 0.5 * (1 + math.erf(z_lo / math.sqrt(2)))
                elif low == high:
                    # "exactly X" → narrow range
                    z_lo = (low - 0.5 - forecast_temp) / std_dev
                    z_hi = (high + 0.5 - forecast_temp) / std_dev
                    prob_lo = 0.5 * (1 + math.erf(z_lo / math.sqrt(2)))
                    prob_hi = 0.5 * (1 + math.erf(z_hi / math.sqrt(2)))
                    our_prob = prob_hi - prob_lo
                else:
                    # "between X-Y"
                    z_lo = (low - 0.5 - forecast_temp) / std_dev
                    z_hi = (high + 0.5 - forecast_temp) / std_dev
                    prob_lo = 0.5 * (1 + math.erf(z_lo / math.sqrt(2)))
                    prob_hi = 0.5 * (1 + math.erf(z_hi / math.sqrt(2)))
                    our_prob = prob_hi - prob_lo
                our_prob = max(our_prob, 0.005)

                edge = our_prob - market_price
                confidence = min(abs(edge) / 0.15, 0.90)

                predictions.append({
                    "category": "weather",
                    "title": question[:100],
                    "slug": slug,
                    "conditionId": cid,
                    "our_prob": round(our_prob, 3),
                    "market_price": round(market_price, 3),
                    "edge": round(edge, 3),
                    "confidence": round(confidence, 3),
                    "pick": "Yes" if edge > 0 else "No",
                    "details": {
                        "city": city or "global",
                        "bracket": "%s-%s°%s" % (low if low is not None else "≤", high if high is not None else "≥", unit),
                        "forecast_temp": round(forecast_temp, 1),
                        "forecast_high_f": round(temp_f, 1),
                        "market_date": evt_date,
                        "volume": vol,
                        "liquidity": liq,
                        "event": evt.get("title", ""),
                    },
                    "source": "NWS/Open-Meteo forecast",
                })
            else:
                # No forecast match — include for browsing
                predictions.append({
                    "category": "weather",
                    "title": question[:100],
                    "slug": slug,
                    "conditionId": cid,
                    "our_prob": round(market_price, 3),
                    "market_price": round(market_price, 3),
                    "edge": 0,
                    "confidence": 0,
                    "pick": "Yes" if market_price > 0.5 else "No",
                    "details": {
                        "city": city or "unknown",
                        "volume": vol,
                        "liquidity": liq,
                        "event": evt.get("title", ""),
                    },
                    "source": "Market consensus",
                })

    return predictions


def _parse_temp_bracket(question):
    """Parse temperature bracket from market question.
    Returns (low, high, unit) or None."""
    q = question.lower()
    # "between 18-19°F"
    m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).upper()
    # "15°F or below" → (-inf, 15) — use None for low
    m = re.search(r'(-?\d+)\s*°?\s*([fFcC])\s+or\s+below', q)
    if m:
        return None, int(m.group(1)), m.group(2).upper()
    # "26°F or higher" → (26, +inf) — use None for high
    m = re.search(r'(-?\d+)\s*°?\s*([fFcC])\s+or\s+higher', q)
    if m:
        return int(m.group(1)), None, m.group(2).upper()
    # "exactly 33°C"
    m = re.search(r'exactly\s+(-?\d+)\s*°?\s*([fFcC])', q)
    if m:
        return int(m.group(1)), int(m.group(1)), m.group(2).upper()
    return None


def get_weather_forecast(lat, lon, days=16):
    """Get multi-model weather forecast from Open-Meteo."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",  # Use local timezone for correct daily max
                "forecast_days": days,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        log.debug(f"Weather API error: {e}")
    return None


def get_ensemble_forecast(lat, lon, days=16):
    """Get ensemble model forecasts for probability estimation."""
    models = {}
    for model in ["gfs_seamless", "ecmwf_ifs04", "icon_seamless"]:
        try:
            resp = requests.get(
                f"https://api.open-meteo.com/v1/{model.split('_')[0] if '_' in model else 'forecast'}",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York",
                    "forecast_days": days,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                models[model] = resp.json()
        except:
            pass
    return models


# City coordinates for weather predictions
CITY_COORDS = {
    "new york": (40.7128, -74.0060),
    "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "la": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "miami": (25.7617, -80.1918),
    "dallas": (32.7767, -96.7970),
    "denver": (39.7392, -104.9903),
    "seattle": (47.6062, -122.3321),
    "boston": (42.3601, -71.0589),
    "san francisco": (37.7749, -122.4194),
    "washington": (38.9072, -77.0369),
    "dc": (38.9072, -77.0369),
    "atlanta": (33.7490, -84.3880),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "st louis": (38.6270, -90.1994),
}


def analyze_weather_market(market):
    """Analyze a weather market and generate prediction."""
    q = market.get("question", "")
    slug = market.get("slug", "")
    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            prices = []

    # Find city
    q_lower = q.lower()
    city = None
    lat, lon = None, None
    for city_name, coords in CITY_COORDS.items():
        if city_name in q_lower:
            city = city_name
            lat, lon = coords
            break

    if not lat:
        return None

    # Find temperature threshold
    temp_match = re.search(r'(\d+)\s*(?:degrees?\s*)?(?:fahrenheit|°?f)', q_lower)
    if not temp_match:
        temp_match = re.search(r'exceed\s+(\d+)', q_lower)
    if not temp_match:
        return None

    threshold = int(temp_match.group(1))

    # Get forecast
    forecast = get_weather_forecast(lat, lon)
    if not forecast or "daily" not in forecast:
        return None

    daily = forecast["daily"]
    max_temps = daily.get("temperature_2m_max", [])
    dates = daily.get("time", [])

    if not max_temps:
        return None

    # Calculate probability of exceeding threshold
    # Use all forecast days within the relevant window
    exceed_count = sum(1 for t in max_temps if t and t > threshold)
    prob_exceed = exceed_count / len(max_temps) if max_temps else 0.5

    # Get market price for comparison
    market_price = float(prices[0]) if prices else 0.5

    # Confidence based on how clear the signal is
    confidence = abs(prob_exceed - 0.5) * 2

    return {
        "category": "weather",
        "title": q[:100],
        "slug": slug,
        "conditionId": market.get("conditionId", ""),
        "our_prob": round(prob_exceed, 3),
        "market_price": round(market_price, 3),
        "edge": round(prob_exceed - market_price, 3),
        "confidence": round(confidence, 3),
        "pick": "Yes" if prob_exceed > 0.5 else "No",
        "details": {
            "city": city,
            "threshold": threshold,
            "forecast_max": round(max(max_temps) if max_temps else 0, 1),
            "forecast_min": round(min(max_temps) if max_temps else 0, 1),
            "forecast_avg": round(np.mean(max_temps) if max_temps else 0, 1),
            "days_exceeding": exceed_count,
            "total_days": len(max_temps),
        },
        "source": "Open-Meteo (GFS/ECMWF ensemble)",
    }


# ============================================================
# 5. SPORTS PREDICTIONS (Sportsbook Odds + Copy Signals)
# ============================================================
def predict_sports(markets, odds_data=None):
    """Predict sports markets using sportsbook odds comparison + copy signals."""
    predictions = []
    copy_signals = load_copy_signals()
    if odds_data is None:
        odds_data = {}

    for m in markets:
        q = m.get("question", "")
        slug = m.get("slug", "")
        prices = m.get("outcomePrices", "[]")
        outcomes_raw = m.get("outcomes", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if isinstance(outcomes_raw, str):
            try: outcomes_raw = json.loads(outcomes_raw)
            except: outcomes_raw = []

        market_price = float(prices[0]) if prices else 0.5

        # Skip resolved markets
        if market_price >= 0.98 or market_price <= 0.02:
            continue

        # Edge sources for this market
        edge_sources = []
        details = {}
        best_source = "Market consensus"

        # Source 1: Sportsbook odds comparison
        odds = find_odds_for_market(q, odds_data) if odds_data else None
        if odds:
            # Determine which team the "Yes" outcome refers to
            yes_team = None
            for outcome in outcomes_raw:
                if isinstance(outcome, str):
                    for team_key in [odds["home"].lower(), odds["away"].lower()]:
                        if team_key in outcome.lower() or outcome.lower() in team_key:
                            yes_team = team_key
                            break

            if yes_team:
                if yes_team == odds["home"].lower():
                    sportsbook_prob = odds["home_prob"]
                else:
                    sportsbook_prob = odds["away_prob"]

                edge_sources.append({
                    "name": "sportsbook",
                    "prob": sportsbook_prob,
                    "weight": 3.0,  # highest weight — sharpest signal
                })
                details["sportsbook_prob"] = sportsbook_prob
                details["sportsbook_edge"] = round(sportsbook_prob - market_price, 4)
                details["n_books"] = odds["n_books"]
                details["odds_spread"] = odds["spread"]

        # Source 2: Copy signals
        signal = find_matching_signal(q, slug, copy_signals, condition_id=m.get("conditionId", ""))
        if signal:
            wr = signal.get("avg_wr", 50)
            n_traders = signal.get("n_traders", 0)
            strength = signal.get("strength", "WEAK")
            copy_prob = copy_signal_to_prob(market_price, wr, n_traders, strength)

            edge_sources.append({
                "name": "copy_signals",
                "prob": copy_prob,
                "weight": 1.0 + min(n_traders, 4) * 0.5,
            })
            details["n_traders"] = n_traders
            details["avg_wr"] = wr
            details["strength"] = strength
            details["total_invested"] = signal.get("total_invested", 0)

        # Source 3: Market microstructure
        micro = analyze_market_microstructure(m)
        details.update(micro)

        # Calculate combined edge
        if edge_sources:
            weighted_prob = sum(s["prob"] * s["weight"] for s in edge_sources) / sum(s["weight"] for s in edge_sources)
            edge, confidence = calculate_enhanced_edge(weighted_prob, market_price, edge_sources)

            source_names = [s["name"] for s in edge_sources]
            if "sportsbook" in source_names and "copy_signals" in source_names:
                best_source = f"Vegas + Copy ({details.get('n_traders', 0)} traders)"
            elif "sportsbook" in source_names:
                best_source = f"Vegas ({details.get('n_books', 0)} books)"
            elif "copy_signals" in source_names:
                best_source = f"Copy Signals ({details.get('n_traders', 0)} traders, {details.get('avg_wr', 0):.0f}% WR)"

            pick = signal.get("consensus", "Yes") if signal else (outcomes_raw[0] if outcomes_raw else "Yes")

            predictions.append({
                "category": "sports",
                "title": q[:100],
                "slug": slug,
                "conditionId": m.get("conditionId", ""),
                "our_prob": round(weighted_prob, 3),
                "market_price": round(market_price, 3),
                "edge": round(edge, 3),
                "confidence": round(confidence, 3),
                "pick": pick,
                "details": details,
                "source": best_source,
            })
        else:
            # No edge sources — include for browsing
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            if 0.02 < market_price < 0.98:
                predictions.append({
                    "category": "sports",
                    "title": q[:100],
                    "slug": slug,
                    "conditionId": m.get("conditionId", ""),
                    "our_prob": round(market_price, 3),
                    "market_price": round(market_price, 3),
                    "edge": 0,
                    "confidence": 0,
                    "pick": outcomes_raw[0] if outcomes_raw else "Yes",
                    "details": {"volume": vol, "liquidity": liq},
                    "source": "Market consensus",
                })

    return predictions


# ============================================================
# 6. POLITICS / ECONOMICS / OTHER PREDICTIONS
# ============================================================
def predict_general(markets, category):
    """Predict general category markets using copy signals + market microstructure."""
    predictions = []
    copy_signals = load_copy_signals()

    for m in markets:
        q = m.get("question", "")
        slug = m.get("slug", "")
        prices = m.get("outcomePrices", "[]")
        outcomes = m.get("outcomes", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: outcomes = []

        market_price = float(prices[0]) if prices else 0.5

        # Skip resolved
        if market_price >= 0.98 or market_price <= 0.02:
            continue

        edge_sources = []
        details = {}

        # Source 1: Copy signals
        signal = find_matching_signal(q, slug, copy_signals, condition_id=m.get("conditionId", ""))
        if signal:
            wr = signal.get("avg_wr", 50)
            n_traders = signal.get("n_traders", 0)
            strength = signal.get("strength", "WEAK")
            copy_prob = copy_signal_to_prob(market_price, wr, n_traders, strength)

            edge_sources.append({
                "name": "copy_signals",
                "prob": copy_prob,
                "weight": 1.0 + min(n_traders, 4) * 0.5,
            })
            details["n_traders"] = n_traders
            details["avg_wr"] = wr
            details["strength"] = strength
            details["total_invested"] = signal.get("total_invested", 0)

        # Market microstructure
        micro = analyze_market_microstructure(m)
        details.update(micro)

        if edge_sources:
            weighted_prob = sum(s["prob"] * s["weight"] for s in edge_sources) / sum(s["weight"] for s in edge_sources)
            edge, confidence = calculate_enhanced_edge(weighted_prob, market_price, edge_sources)

            predictions.append({
                "category": category,
                "title": q[:100],
                "slug": slug,
                "conditionId": m.get("conditionId", ""),
                "our_prob": round(weighted_prob, 3),
                "market_price": round(market_price, 3),
                "edge": round(edge, 3),
                "confidence": round(confidence, 3),
                "pick": signal.get("consensus", outcomes[0] if outcomes else "Yes"),
                "details": details,
                "source": f"Copy Signals ({details.get('n_traders', 0)} traders)",
            })
        else:
            # Include for browsing
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            if 0.02 < market_price < 0.98:
                predictions.append({
                    "category": category,
                    "title": q[:100],
                    "slug": slug,
                    "conditionId": m.get("conditionId", ""),
                    "our_prob": round(market_price, 3),
                    "market_price": round(market_price, 3),
                    "edge": 0,
                    "confidence": 0,
                    "pick": outcomes[0] if outcomes else "Yes",
                    "details": {"volume": vol, "liquidity": liq},
                    "source": "Market consensus",
                })

    return predictions


# ============================================================
# 7. CRYPTO PREDICTIONS (TA + Copy Signals)
# ============================================================
def predict_crypto(markets):
    """Predict crypto markets using technical analysis + copy signals."""
    predictions = []
    copy_signals = load_copy_signals()

    # Get TA data for major assets
    ta_cache = {}
    for sym, label in [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL")]:
        ta = get_crypto_ta(sym, "1h", 100)
        if ta:
            ta_cache[label] = ta
            log.info(f"  {label}: ${ta['price']:,.0f} | RSI {ta['rsi']} | MACD {ta['macd_hist']:+.0f} | BB% {ta['bb_pct']:.2f}")
        time.sleep(0.2)

    for m in markets:
        q = m.get("question", "")
        slug = m.get("slug", "")
        prices_raw = m.get("outcomePrices", "[]")
        outcomes = m.get("outcomes", "[]")
        if isinstance(prices_raw, str):
            try: prices_raw = json.loads(prices_raw)
            except: prices_raw = []
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: outcomes = []

        market_price = float(prices_raw[0]) if prices_raw else 0.5
        if market_price >= 0.98 or market_price <= 0.02:
            continue

        edge_sources = []
        details = {}
        best_source = "Market consensus"
        q_lower = q.lower()

        # Identify which asset this market is about
        asset_ta = None
        if any(w in q_lower for w in ["bitcoin", "btc"]):
            asset_ta = ta_cache.get("BTC")
            details["asset"] = "BTC"
        elif any(w in q_lower for w in ["ethereum", "eth"]):
            asset_ta = ta_cache.get("ETH")
            details["asset"] = "ETH"
        elif any(w in q_lower for w in ["solana", "sol"]):
            asset_ta = ta_cache.get("SOL")
            details["asset"] = "SOL"

        # Source 1: Technical analysis for price target markets
        if asset_ta:
            ta_prob, ta_details = estimate_crypto_prob(q, asset_ta)
            if ta_prob is not None:
                edge_sources.append({
                    "name": "crypto_ta",
                    "prob": ta_prob,
                    "weight": 2.5,
                })
                details.update(ta_details)
                details["ta_prob"] = ta_prob

            details["price"] = asset_ta["price"]
            details["rsi"] = asset_ta["rsi"]
            details["macd_hist"] = asset_ta["macd_hist"]
            details["momentum_1d"] = asset_ta["momentum_1d"]

        # Source 2: Copy signals
        signal = find_matching_signal(q, slug, copy_signals, condition_id=m.get("conditionId", ""))
        if signal:
            wr = signal.get("avg_wr", 50)
            n_traders = signal.get("n_traders", 0)
            strength = signal.get("strength", "WEAK")
            copy_prob = copy_signal_to_prob(market_price, wr, n_traders, strength)

            edge_sources.append({
                "name": "copy_signals",
                "prob": copy_prob,
                "weight": 1.0 + min(n_traders, 4) * 0.5,
            })
            details["n_traders"] = n_traders
            details["avg_wr"] = wr

        # Market microstructure
        micro = analyze_market_microstructure(m)
        details.update(micro)

        # Calculate combined edge
        if edge_sources:
            weighted_prob = sum(s["prob"] * s["weight"] for s in edge_sources) / sum(s["weight"] for s in edge_sources)
            edge, confidence = calculate_enhanced_edge(weighted_prob, market_price, edge_sources)

            source_names = [s["name"] for s in edge_sources]
            if "crypto_ta" in source_names and "copy_signals" in source_names:
                best_source = f"TA + Copy Signals"
            elif "crypto_ta" in source_names:
                best_source = f"Crypto TA (RSI:{details.get('rsi', '?')} MACD:{details.get('macd_hist', '?'):+.0f})"
            elif "copy_signals" in source_names:
                best_source = f"Copy Signals ({details.get('n_traders', 0)} traders)"

            pick = signal.get("consensus", "Yes") if signal else (outcomes[0] if outcomes else "Yes")

            predictions.append({
                "category": "crypto",
                "title": q[:100],
                "slug": slug,
                "conditionId": m.get("conditionId", ""),
                "our_prob": round(weighted_prob, 3),
                "market_price": round(market_price, 3),
                "edge": round(edge, 3),
                "confidence": round(confidence, 3),
                "pick": pick,
                "details": details,
                "source": best_source,
            })
        else:
            # No edge sources — include for browsing
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            predictions.append({
                "category": "crypto",
                "title": q[:100],
                "slug": slug,
                "conditionId": m.get("conditionId", ""),
                "our_prob": round(market_price, 3),
                "market_price": round(market_price, 3),
                "edge": 0,
                "confidence": 0,
                "pick": outcomes[0] if outcomes else "Yes",
                "details": {**details, "volume": vol, "liquidity": liq},
                "source": "Market consensus",
            })

    return predictions


# ============================================================
# HELPERS
# ============================================================
def load_copy_signals():
    """Load copy signals from cron output."""
    path = LOG_DIR / "copy_signals.json"
    if not path.exists():
        return []
    try:
        data = json.load(open(path))
        return data.get("signals", [])
    except:
        return []


def find_matching_signal(question, slug, signals, condition_id=None):
    """Find a matching copy signal for a market."""
    q_lower = question.lower()
    for s in signals:
        s_cid = s.get("conditionId", "")
        s_title = s.get("title", "").lower()
        s_slug = s.get("slug", "")

        # Condition ID match (most reliable)
        if condition_id and s_cid and s_cid == condition_id:
            return s
        # Direct slug match
        if s_slug and s_slug == slug:
            return s
        # Title containment
        if s_title and (s_title in q_lower or q_lower in s_title):
            return s
        # Key word overlap (at least 3 words, >50% overlap)
        s_words = set(w for w in s_title.split() if len(w) > 2)
        q_words = set(w for w in q_lower.split() if len(w) > 2)
        overlap = len(s_words & q_words)
        if overlap >= 3 and overlap / max(len(s_words), 1) > 0.5:
            return s
    return None


# ============================================================
# MAIN
# ============================================================
def run_predictions(category_filter=None):
    """Run all predictions with multi-source edge detection."""
    print(f"\n{'='*72}")
    print(f"  MTY MULTI-CATEGORY PREDICTOR v2")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*72}")

    # Fetch all markets
    markets = fetch_all_markets()
    categorized = categorize_all(markets)

    # Show category breakdown
    print(f"\n  Market Categories:")
    for cat, mkt_list in sorted(categorized.items(), key=lambda x: -len(x[1])):
        info = CATEGORIES.get(cat, {"icon": "?", "color": "#999"})
        print(f"    {cat:15s}: {len(mkt_list):4d} markets")

    # Pre-fetch edge data sources
    print(f"\n  Fetching edge data sources...")

    # Sportsbook odds (for sports category)
    odds_data = {}
    if not category_filter or category_filter == "sports":
        odds_data = fetch_sportsbook_odds()
        if odds_data:
            n_matchups = len(set(id(v) for v in odds_data.values()))
            print(f"  Sportsbook odds: {n_matchups} matchups from The Odds API")

    all_predictions = []

    # Run predictions per category
    for cat, mkt_list in categorized.items():
        if category_filter and cat != category_filter:
            continue

        if cat == "weather":
            preds = predict_weather(mkt_list)
        elif cat == "sports":
            preds = predict_sports(mkt_list, odds_data=odds_data)
        elif cat == "crypto":
            print(f"\n  Crypto TA Analysis:")
            preds = predict_crypto(mkt_list)
        elif cat in ("politics", "economics", "entertainment", "olympics",
                     "science_tech", "world", "esports", "pop_culture"):
            preds = predict_general(mkt_list, cat)
        else:
            preds = predict_general(mkt_list, cat)

        all_predictions.extend(preds)

        edge_preds = [p for p in preds if abs(p.get("edge", 0)) > 0.03]
        if edge_preds:
            print(f"\n  {cat.upper()}: {len(edge_preds)} predictions with >3% edge")
            for p in sorted(edge_preds, key=lambda x: -abs(x.get("edge", 0)))[:5]:
                edge_str = f"{p['edge']:+.1%}" if p['edge'] else "--"
                conf_str = f"{p['confidence']:.0%}" if p.get('confidence') else "--"
                print(f"    {p['pick']:12s} | {p['title'][:45]:45s} | edge {edge_str} | conf {conf_str} | {p['source'][:30]}")
        else:
            print(f"  {cat.upper()}: {len(preds)} markets tracked (no edge signals)")

    # Sort: edge predictions first, then by volume
    all_predictions.sort(key=lambda x: (
        -1 if abs(x.get("edge", 0)) > 0.03 else 0,
        -abs(x.get("edge", 0)),
        -float(x.get("details", {}).get("volume", 0) or 0),
    ))

    # Show top picks across all categories
    top = [p for p in all_predictions if abs(p.get("edge", 0)) > 0.03]
    with_data = [p for p in all_predictions if p.get("source", "") != "Market consensus"]
    if top:
        print(f"\n{'='*72}")
        print(f"  TOP PICKS ACROSS ALL CATEGORIES ({len(top)} with >3% edge)")
        print(f"{'='*72}")
        for i, p in enumerate(top[:15]):
            cat_emoji = {"weather": "W", "sports": "S", "politics": "P",
                        "crypto": "C", "entertainment": "E", "economics": "$",
                        "olympics": "O", "world": "G", "esports": "E",
                        "science_tech": "T", "pop_culture": "X"}.get(p["category"], "?")
            edge_str = f"{p['edge']:+.1%}"
            conf_str = f"{p['confidence']:.0%}"
            print(f"  [{cat_emoji}] #{i+1}  {p['pick']:12s} | {p['title'][:45]:45s} | edge {edge_str} | conf {conf_str}")

    # Edge source breakdown
    vegas_edges = [p for p in all_predictions if "sportsbook" in str(p.get("source", "")).lower() or "vegas" in str(p.get("source", "")).lower()]
    ta_edges = [p for p in all_predictions if "ta" in str(p.get("source", "")).lower() or "rsi" in str(p.get("source", "")).lower()]
    copy_edges = [p for p in all_predictions if "copy" in str(p.get("source", "")).lower()]

    print(f"\n  Summary: {len(all_predictions)} markets | {len(with_data)} with signals | {len(top)} with >3% edge")
    if vegas_edges or ta_edges or copy_edges:
        print(f"  Sources: Vegas={len(vegas_edges)} | Crypto TA={len(ta_edges)} | Copy Signals={len(copy_edges)}")

    # Save results
    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(markets),
        "categories": {cat: len(mkt_list) for cat, mkt_list in categorized.items()},
        "total_predictions": len(all_predictions),
        "with_signals": len(with_data),
        "top_picks": len(top) if top else 0,
        "edge_sources": {
            "vegas": len(vegas_edges),
            "crypto_ta": len(ta_edges),
            "copy_signals": len(copy_edges),
        },
        "predictions": all_predictions,
    }

    output_path = LOG_DIR / "all_predictions.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved {len(all_predictions)} predictions to {output_path}")

    # Also save daily snapshot
    daily_path = LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(daily_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


def main():
    parser = argparse.ArgumentParser(description="Multi-Category Predictor")
    parser.add_argument("--category", type=str, help="Filter to specific category")
    args = parser.parse_args()

    run_predictions(category_filter=args.category)


if __name__ == "__main__":
    main()
