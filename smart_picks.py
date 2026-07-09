#!/usr/bin/env python3
"""
SMART PICKS — Integrated Prediction + Copy Signal Recommendation Engine
========================================================================

Combines our ML prediction engine (80.6% accuracy at 65%+ confidence) with
live copy signals from top Polymarket traders to produce actionable picks.

ARCHITECTURE:
  1. Load copy signals (from copy_signal_bot cron)
  2. Fetch active Polymarket sports/political markets
  3. For NBA games: run prediction engine (ELO + LightGBM + Form + Advanced)
  4. Combine ML confidence with copy signal strength
  5. Output ranked picks with Kelly sizing

SCORING:
  - Copy signal score (0-40 pts): n_traders * 5 + avg_wr * 0.2 + strength_bonus
  - ML model score (0-40 pts): confidence * 40 (NBA games only)
  - Market edge score (0-20 pts): edge vs market price * 100

  Total: 0-100 pts → STRONG (70+), MODERATE (50+), WEAK (<50)

Usage:
    python3 smart_picks.py                    # show tomorrow's picks
    python3 smart_picks.py --output json      # JSON output for copy_executor
    python3 smart_picks.py --detailed         # verbose analysis
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smart_picks")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")

GAMMA_HOST = "https://gamma-api.polymarket.com"

# NBA team name variants for matching Polymarket titles
TEAM_NAMES = {
    "ATL": ["hawks", "atlanta"],
    "BOS": ["celtics", "boston"],
    "BKN": ["nets", "brooklyn"],
    "CHA": ["hornets", "charlotte"],
    "CHI": ["bulls", "chicago"],
    "CLE": ["cavaliers", "cavs", "cleveland"],
    "DAL": ["mavericks", "mavs", "dallas"],
    "DEN": ["nuggets", "denver"],
    "DET": ["pistons", "detroit"],
    "GSW": ["warriors", "golden state"],
    "HOU": ["rockets", "houston"],
    "IND": ["pacers", "indiana"],
    "LAC": ["clippers", "la clippers"],
    "LAL": ["lakers", "la lakers", "los angeles lakers"],
    "MEM": ["grizzlies", "memphis"],
    "MIA": ["heat", "miami"],
    "MIL": ["bucks", "milwaukee"],
    "MIN": ["timberwolves", "wolves", "minnesota"],
    "NOP": ["pelicans", "new orleans"],
    "NYK": ["knicks", "new york"],
    "OKC": ["thunder", "oklahoma city", "okc"],
    "ORL": ["magic", "orlando"],
    "PHI": ["76ers", "sixers", "philadelphia"],
    "PHX": ["suns", "phoenix"],
    "POR": ["blazers", "trail blazers", "portland"],
    "SAC": ["kings", "sacramento"],
    "SAS": ["spurs", "san antonio"],
    "TOR": ["raptors", "toronto"],
    "UTA": ["jazz", "utah"],
    "WAS": ["wizards", "washington"],
}

# Reverse mapping: name variant -> abbreviation
NAME_TO_ABBR = {}
for abbr, names in TEAM_NAMES.items():
    for n in names:
        NAME_TO_ABBR[n.lower()] = abbr
    NAME_TO_ABBR[abbr.lower()] = abbr


def find_team_abbr(text):
    """Extract team abbreviation from text."""
    text_lower = text.lower()
    for name, abbr in sorted(NAME_TO_ABBR.items(), key=lambda x: -len(x[0])):
        if name in text_lower:
            return abbr
    return None


def extract_teams_from_title(title):
    """Extract home and away teams from a Polymarket title like 'Rockets vs. Thunder'."""
    title_lower = title.lower()
    # Common patterns: "X vs Y", "X vs. Y", "X at Y"
    patterns = [
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+at\s+(.+)",
        r"(.+?)\s+@\s+(.+)",
    ]
    for pat in patterns:
        m = re.match(pat, title_lower.strip())
        if m:
            away_text, home_text = m.group(1).strip(), m.group(2).strip()
            # Remove any trailing markers like ": O/U 213.5"
            away_text = re.split(r"[:\-]", away_text)[0].strip()
            home_text = re.split(r"[:\-]", home_text)[0].strip()
            away = find_team_abbr(away_text)
            home = find_team_abbr(home_text)
            if away and home:
                return away, home
    return None, None


# ============================================================
# DATA LOADING
# ============================================================
def load_copy_signals():
    """Load latest copy signals from cron job output."""
    path = LOG_DIR / "copy_signals.json"
    if not path.exists():
        log.warning("No copy_signals.json found")
        return []
    try:
        data = json.load(open(path))
        signals = data.get("signals", [])
        scan_time = data.get("scan_time", data.get("time", ""))
        log.info(f"Loaded {len(signals)} copy signals (scan: {scan_time})")
        return signals
    except Exception as e:
        log.error(f"Failed to load copy signals: {e}")
        return []


def load_tracker_data():
    """Load trader tracker data for additional context."""
    path = LOG_DIR / "trader_tracker.json"
    if not path.exists():
        return {}
    try:
        data = json.load(open(path))
        traders = {}
        for t in data.get("traders", []):
            traders[t.get("name", "")] = {
                "win_rate": t.get("win_rate", 0),
                "total_pnl": t.get("total_pnl", 0),
                "active_positions": len(t.get("active", [])),
                "positions": t.get("active", []),
            }
        return traders
    except:
        return {}


def fetch_polymarket_sports_markets():
    """Fetch all active sports markets from Polymarket."""
    all_markets = []
    try:
        # Fetch NBA markets
        for tag in ["nba", "basketball", "sports"]:
            resp = requests.get(
                f"{GAMMA_HOST}/markets",
                params={"limit": 100, "active": "true", "closed": "false", "tag": tag},
                timeout=15,
            )
            if resp.status_code == 200:
                markets = resp.json()
                all_markets.extend(markets)

        # Also fetch without tag filter but search for sports keywords
        resp = requests.get(
            f"{GAMMA_HOST}/markets",
            params={"limit": 200, "active": "true", "closed": "false"},
            timeout=15,
        )
        if resp.status_code == 200:
            markets = resp.json()
            for m in markets:
                q = (m.get("question", "") + " " + m.get("slug", "")).lower()
                if any(kw in q for kw in [
                    "nba", "nfl", "mlb", "nhl", "soccer", "football",
                    "lakers", "celtics", "warriors", "rockets", "thunder",
                    "cavaliers", "knicks", "bucks", "nuggets", "suns",
                    "grizzlies", "heat", "pacers", "hawks", "magic",
                    "mavericks", "clippers", "pelicans", "kings", "spurs",
                    "blazers", "raptors", "jazz", "wizards", "hornets",
                    "pistons", "nets", "timberwolves", "bulls", "76ers",
                    "election", "president", "trump", "biden", "politics",
                    "superbowl", "super bowl", "spread", "over/under", "o/u",
                ]):
                    all_markets.append(m)

        # Deduplicate by conditionId
        seen = set()
        unique = []
        for m in all_markets:
            cid = m.get("conditionId", m.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(m)
            elif not cid:
                unique.append(m)
        all_markets = unique

    except Exception as e:
        log.error(f"Polymarket API error: {e}")

    log.info(f"Fetched {len(all_markets)} active Polymarket sports/political markets")
    return all_markets


def load_ml_model():
    """Load the trained prediction engine model and features."""
    csv_path = DATA_DIR / "nba_games_5seasons.csv"
    if not csv_path.exists():
        log.warning("No historical data found for ML model")
        return None, None

    try:
        import pandas as pd
        from prediction_engine import (
            build_comprehensive_features,
            create_matchup_features,
            EnsemblePredictor,
        )

        raw = pd.read_csv(csv_path)
        # Also load current season if available
        curr_path = DATA_DIR / "nba_current_season.csv"
        if curr_path.exists():
            curr = pd.read_csv(curr_path)
            raw = pd.concat([raw, curr], ignore_index=True).drop_duplicates(
                subset=["GAME_ID", "TEAM_ABBREVIATION"]
            )

        df = build_comprehensive_features(raw)
        features = create_matchup_features(df)

        # Train on all data
        predictor = EnsemblePredictor()
        predictor.train(features)

        return predictor, features

    except Exception as e:
        log.error(f"ML model load failed: {e}")
        return None, None


# ============================================================
# SCORING ENGINE
# ============================================================
def score_pick(signal, ml_prob=None, market_price=None):
    """
    Score a pick from 0-100 combining:
    - Copy signal (0-40 pts)
    - ML model (0-40 pts)
    - Market edge (0-20 pts)
    """
    score = 0.0
    breakdown = {}

    # ── Copy Signal Score (0-40) ──
    n_traders = signal.get("n_traders", 0)
    avg_wr = signal.get("avg_wr", 50)
    strength = signal.get("strength", "WEAK")
    total_invested = signal.get("total_invested", 0)

    # Trader count: 2 traders = 10pts, 3+ = 15pts
    trader_pts = min(n_traders * 5, 15)

    # Win rate: 60% = 4pts, 80% = 12pts, 90%+ = 16pts
    wr_pts = max(0, (avg_wr - 50) * 0.4)  # 0-20 range

    # Strength bonus
    str_bonus = {"STRONG": 5, "MODERATE": 2, "WEAK": 0}.get(strength, 0)

    copy_score = min(trader_pts + wr_pts + str_bonus, 40)
    breakdown["copy_signal"] = round(copy_score, 1)
    score += copy_score

    # ── ML Model Score (0-40) ──
    if ml_prob is not None:
        # ml_prob here is already oriented: probability that the consensus pick wins
        if ml_prob > 0.5:
            # ML agrees with copy signal — boost score
            confidence = (ml_prob - 0.5) * 2  # 0-1 range
            ml_score = confidence * 40
            breakdown["ml_agrees"] = True
        else:
            # ML disagrees — still add some points for having ML data, but less
            # A strong disagreement (ml_prob < 0.3) means caution
            confidence = (0.5 - ml_prob) * 2
            ml_score = max(0, 10 - confidence * 15)  # 0-10 pts when disagreeing
            breakdown["ml_agrees"] = False
            breakdown["ml_caution"] = f"ML favors opponent ({1-ml_prob:.0%})"
        breakdown["ml_model"] = round(ml_score, 1)
        score += ml_score
    else:
        breakdown["ml_model"] = 0

    # ── Market Edge Score (0-20) ──
    if market_price is not None and 0.02 < market_price < 0.98:
        # Edge = how much the market price differs from fair value
        # If top traders buy at 0.60 and their WR is 85%, that's value
        implied_prob = market_price
        trader_edge = (avg_wr / 100) - implied_prob
        if trader_edge > 0:
            edge_score = min(trader_edge * 100, 20)  # 10% edge = 10pts, 20%+ = 20pts
        else:
            edge_score = 0
        breakdown["market_edge"] = round(edge_score, 1)
        score += edge_score
    else:
        breakdown["market_edge"] = 0

    # Investment size bonus (0-5, included in copy but capped at 40)
    if total_invested > 100000:
        score = min(score + 3, 100)
        breakdown["high_conviction_bonus"] = 3

    return round(score, 1), breakdown


def get_pick_grade(score):
    """Convert score to grade."""
    if score >= 80:
        return "A+"
    elif score >= 70:
        return "A"
    elif score >= 60:
        return "B+"
    elif score >= 50:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


# ============================================================
# MAIN PICK GENERATION
# ============================================================
def generate_picks(detailed=False):
    """Generate tomorrow's smart picks."""
    print(f"\n{'='*72}")
    print(f"  MTY SMART PICKS — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Integrated ML + Copy Signal Recommendation Engine")
    print(f"{'='*72}")

    # 1. Load copy signals
    signals = load_copy_signals()
    if not signals:
        print("\n  No copy signals available. Run copy_signal_bot.py first.")
        return []

    # 2. Load tracker data for context
    tracker = load_tracker_data()

    # 3. Fetch live Polymarket markets
    markets = fetch_polymarket_sports_markets()

    # 4. Try to load ML model for NBA game predictions
    print("\n  Loading ML prediction model...")
    predictor, features = load_ml_model()
    if predictor:
        print("  ML model loaded (trained on 5 seasons, 5,995 games)")
    else:
        print("  ML model not available — using copy signals only")

    # 5. Build market lookup for prices
    market_lookup = {}
    for m in markets:
        slug = m.get("slug", "")
        question = m.get("question", "")
        cid = m.get("conditionId", "")
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except:
                prices = []
        market_lookup[slug] = {
            "question": question,
            "prices": prices,
            "outcomes": m.get("outcomes", "[]"),
            "conditionId": cid,
            "volume": float(m.get("volume", 0) or 0),
            "liquidity": float(m.get("liquidity", 0) or 0),
        }
        if cid:
            market_lookup[cid] = market_lookup[slug]

    # 6. Score each signal
    picks = []
    for signal in signals:
        title = signal.get("title", "")
        slug = signal.get("slug", "")
        consensus = signal.get("consensus", "")
        strength = signal.get("strength", "WEAK")

        # Get market price — try multiple lookup methods
        market_info = market_lookup.get(slug, market_lookup.get(signal.get("conditionId", ""), {}))

        # If not found in bulk fetch, try direct API lookup
        if not market_info.get("prices") and slug:
            try:
                resp = requests.get(f"{GAMMA_HOST}/markets", params={"slug": slug, "limit": 1}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        m = data[0] if isinstance(data, list) else data
                        prices = m.get("outcomePrices", "[]")
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        market_info = {
                            "prices": prices,
                            "outcomes": m.get("outcomes", "[]"),
                            "conditionId": m.get("conditionId", ""),
                            "volume": float(m.get("volume", 0) or 0),
                        }
            except:
                pass

        market_price = None
        if market_info.get("prices"):
            outcomes = market_info.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except:
                    outcomes = []
            for i, o in enumerate(outcomes):
                if isinstance(o, str) and (o.lower() == consensus.lower() or consensus.lower() in o.lower()):
                    if i < len(market_info["prices"]):
                        market_price = float(market_info["prices"][i])
                    break
            # If no match by name, and it's a Yes/No market, try matching
            if market_price is None and outcomes:
                if len(outcomes) == 2 and any(o.lower() in ["yes", "no"] for o in outcomes):
                    # For Yes/No markets, consensus usually maps to "Yes"
                    for i, o in enumerate(outcomes):
                        if o.lower() == "yes" and i < len(market_info["prices"]):
                            market_price = float(market_info["prices"][i])
                            break

        # Try ML prediction for NBA moneyline games
        ml_prob = None
        if predictor and features is not None:
            away_abbr, home_abbr = extract_teams_from_title(title)
            if away_abbr and home_abbr:
                # Check if this is a moneyline game (not spread/total)
                is_moneyline = "spread" not in title.lower() and "o/u" not in title.lower() and "total" not in title.lower()
                if is_moneyline:
                    import pandas as pd
                    from prediction_engine import FEATURE_COLS

                    # Get latest features for each team
                    home_rows = features[features["home_team"] == home_abbr]
                    away_rows = features[features["away_team"] == away_abbr]

                    if len(home_rows) > 0 and len(away_rows) > 0:
                        # Use the most recent game as feature template
                        test_row = home_rows.iloc[[-1]].copy()
                        try:
                            result = predictor.predict(test_row)
                            # predict() returns a DataFrame with ensemble_prob column
                            if hasattr(result, 'ensemble_prob'):
                                raw_prob = float(result['ensemble_prob'].iloc[0])
                            elif isinstance(result, (float, int, np.floating)):
                                raw_prob = float(result)
                            else:
                                raw_prob = float(result.iloc[0])
                            # raw_prob > 0.5 means model favors home team
                            consensus_is_home = find_team_abbr(consensus) == home_abbr
                            if consensus_is_home:
                                ml_prob = raw_prob
                            else:
                                ml_prob = 1 - raw_prob
                        except Exception as e:
                            log.debug(f"ML prediction failed for {title}: {e}")
                            ml_prob = None

        # Skip resolved markets (price at 0 or 1)
        if market_price is not None and (market_price >= 0.99 or market_price <= 0.01):
            log.debug(f"Skipping resolved market: {title} (price={market_price})")
            continue

        # Score
        total_score, breakdown = score_pick(signal, ml_prob, market_price)
        grade = get_pick_grade(total_score)

        pick = {
            "title": title,
            "slug": slug,
            "consensus": consensus,
            "strength": strength,
            "n_traders": signal.get("n_traders", 0),
            "avg_wr": signal.get("avg_wr", 0),
            "total_invested": signal.get("total_invested", 0),
            "market_price": market_price,
            "ml_prob": ml_prob,
            "score": total_score,
            "grade": grade,
            "breakdown": breakdown,
            "conditionId": signal.get("conditionId", ""),
            "traders": signal.get("traders", []),
        }

        # Get trader details
        trader_names = [t.get("name", "") for t in signal.get("traders", [])]
        pick["trader_names"] = trader_names
        pick["trader_details"] = []
        for tn in trader_names:
            if tn in tracker:
                pick["trader_details"].append({
                    "name": tn,
                    "win_rate": tracker[tn].get("win_rate", 0),
                    "pnl": tracker[tn].get("total_pnl", 0),
                })

        picks.append(pick)

    # Sort by score descending
    picks.sort(key=lambda x: x["score"], reverse=True)

    # 7. Display results
    print(f"\n{'='*72}")
    print(f"  TOMORROW'S PICKS — Ranked by Smart Score")
    print(f"{'='*72}")

    if not picks:
        print("\n  No actionable picks found.")
        return picks

    # Summary stats
    strong_picks = [p for p in picks if p["score"] >= 70]
    moderate_picks = [p for p in picks if 50 <= p["score"] < 70]
    print(f"\n  {len(strong_picks)} STRONG picks (70+) | {len(moderate_picks)} MODERATE picks (50-70) | {len(picks)} total\n")

    for i, pick in enumerate(picks):
        if pick["score"] < 30:
            continue  # Skip very weak signals

        emoji_grade = {
            "A+": "[A+]", "A": "[A ]", "B+": "[B+]",
            "B": "[B ]", "C": "[C ]", "D": "[D ]",
        }.get(pick["grade"], "[? ]")

        print(f"  {emoji_grade} #{i+1}  {pick['title']}")
        print(f"       Pick: {pick['consensus']} | Score: {pick['score']}/100 | {pick['strength']}")
        print(f"       Traders: {pick['n_traders']} (avg {pick['avg_wr']:.0f}% WR) | Invested: ${pick['total_invested']:,.0f}")

        if pick["market_price"]:
            implied = pick["market_price"] * 100
            print(f"       Market: {implied:.0f}c (implied {implied:.0f}%)")

        if pick["ml_prob"] is not None:
            print(f"       ML Model: {pick['ml_prob']:.1%} win probability", end="")
            if pick["breakdown"].get("ml_agrees") is False:
                print(f"  *** CAUTION: {pick['breakdown'].get('ml_caution', 'ML disagrees')}")
            else:
                print(f"  (agrees with pick)")

        # Breakdown
        bd = pick["breakdown"]
        parts = []
        if bd.get("copy_signal", 0) > 0:
            parts.append(f"Copy:{bd['copy_signal']:.0f}")
        if bd.get("ml_model", 0) > 0:
            parts.append(f"ML:{bd['ml_model']:.0f}")
        if bd.get("market_edge", 0) > 0:
            parts.append(f"Edge:{bd['market_edge']:.0f}")
        if parts:
            print(f"       Score: {' + '.join(parts)} = {pick['score']}")

        if detailed and pick["trader_names"]:
            for tn in pick["trader_names"]:
                td = next((t for t in pick["trader_details"] if t["name"] == tn), None)
                if td:
                    print(f"       -> {tn}: {td['win_rate']:.0f}% WR, ${td['pnl']:+,.0f} PnL")
                else:
                    print(f"       -> {tn}")

        # Kelly sizing recommendation
        if pick["score"] >= 50:
            wr = pick["avg_wr"] / 100
            if pick["ml_prob"] is not None:
                wr = (wr + pick["ml_prob"]) / 2  # blend copy WR with ML prob
            odds = 1 / max(pick["market_price"], 0.01) if pick["market_price"] else 1.8
            b = odds - 1
            if b > 0:
                kelly = max(0, (wr * b - (1 - wr)) / b)
                quarter_kelly = kelly * 0.25
                size = 50000 * min(quarter_kelly, 0.10)
                if size >= 50:
                    print(f"       Kelly: {quarter_kelly:.1%} of bankroll -> ${size:,.0f}")

        print()

    # 8. Save picks
    output_path = LOG_DIR / f"smart_picks_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, "w") as f:
        # Clean up non-serializable fields
        clean_picks = []
        for p in picks:
            cp = {k: v for k, v in p.items()}
            cp["ml_prob"] = float(cp["ml_prob"]) if cp["ml_prob"] is not None else None
            clean_picks.append(cp)
        json.dump({
            "generated": datetime.now(timezone.utc).isoformat(),
            "n_picks": len(picks),
            "strong_picks": len(strong_picks),
            "moderate_picks": len(moderate_picks),
            "picks": clean_picks,
        }, f, indent=2, default=str)

    print(f"  Saved to {output_path}")

    # 9. Also save as enhanced signals for copy_executor
    enhanced_path = LOG_DIR / "smart_signals.json"
    enhanced = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "signals": [
            {
                "title": p["title"],
                "slug": p["slug"],
                "consensus": p["consensus"],
                "strength": "STRONG" if p["score"] >= 70 else "MODERATE" if p["score"] >= 50 else "WEAK",
                "n_traders": p["n_traders"],
                "avg_wr": p["avg_wr"],
                "total_invested": p["total_invested"],
                "conditionId": p["conditionId"],
                "smart_score": p["score"],
                "grade": p["grade"],
                "ml_prob": float(p["ml_prob"]) if p["ml_prob"] is not None else None,
                "market_price": p["market_price"],
                "traders": p.get("traders", []),
            }
            for p in picks
            if p["score"] >= 40
        ],
    }
    with open(enhanced_path, "w") as f:
        json.dump(enhanced, f, indent=2, default=str)
    print(f"  Enhanced signals saved to {enhanced_path}")

    # 10. Top picks summary
    top = [p for p in picks if p["score"] >= 50]
    if top:
        print(f"\n{'='*72}")
        print(f"  TOP PICKS SUMMARY")
        print(f"{'='*72}")
        for p in top:
            arrow = ">>>" if p["score"] >= 70 else " > "
            print(f"  {arrow} {p['consensus']} in '{p['title']}' — Score {p['score']}, Grade {p['grade']}")
        print()

    return picks


def main():
    parser = argparse.ArgumentParser(description="Smart Picks — ML + Copy Signal Picks")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("--detailed", action="store_true", help="Show trader details")
    parser.add_argument("--min-score", type=float, default=30, help="Minimum score to show")
    args = parser.parse_args()

    picks = generate_picks(detailed=args.detailed)

    if args.output == "json":
        output = [p for p in picks if p["score"] >= args.min_score]
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
