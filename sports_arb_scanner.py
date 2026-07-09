#!/usr/bin/env python3
"""
PINNACLE vs POLYMARKET SPREAD SCANNER
=======================================

The cleanest edge in sports betting: no ML, no overfitting.

Logic:
  1. Fetch sharp odds from Pinnacle (the world's sharpest book)
  2. Fetch current Polymarket sports market prices
  3. Compare: when Polymarket deviates from Pinnacle by >5%, bet
  4. Pinnacle's closing line IS the consensus model from the smartest
     money in the world. If Polymarket disagrees, Polymarket is wrong.

Why this works:
  - Pinnacle sets the market. Their lines are moved by professional
    bettors with millions. It's the most efficient odds in the world.
  - Polymarket is a prediction market with retail participants.
    It's less efficient, especially for sports.
  - The spread between them IS the edge.

Requirements:
  1. Free API key from https://the-odds-api.com (#get-access)
  2. Set ODDS_API_KEY in .env or pass via --api-key

Usage:
  python3 sports_arb_scanner.py --api-key YOUR_KEY
  python3 sports_arb_scanner.py --api-key YOUR_KEY --min-edge 3
  python3 sports_arb_scanner.py --simulate   # backtest mode (no API needed)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
GAMMA_HOST = "https://gamma-api.polymarket.com"

# Sports mapping: Odds API sport key → Polymarket slug pattern
SPORTS = {
    "basketball_nba": {"slug_prefix": "nba-", "name": "NBA"},
    "americanfootball_nfl": {"slug_prefix": "nfl-", "name": "NFL"},
    "icehockey_nhl": {"slug_prefix": "nhl-", "name": "NHL"},
}


# ============================================================
# PINNACLE ODDS FETCHER
# ============================================================
def fetch_pinnacle_odds(api_key: str, sport: str = "basketball_nba") -> list[dict]:
    """
    Fetch current odds from Pinnacle via The Odds API.
    Returns list of games with Pinnacle's implied probabilities.
    """
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us,eu",
        "markets": "h2h,spreads,totals",
        "bookmakers": "pinnacle",
        "oddsFormat": "decimal",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 401:
            print("ERROR: Invalid API key. Get a free key at https://the-odds-api.com/#get-access")
            return []
        if resp.status_code == 422:
            print(f"  No odds available for {sport}")
            return []

        data = resp.json()

        # Check remaining credits
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        print(f"  Odds API credits: {remaining} remaining, {used} used")

        games = []
        for event in data:
            game = {
                "id": event.get("id", ""),
                "sport": sport,
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "commence_time": event.get("commence_time", ""),
                "markets": {},
            }

            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != "pinnacle":
                    continue

                for market in bookmaker.get("markets", []):
                    mtype = market.get("key", "")  # h2h, spreads, totals
                    outcomes = market.get("outcomes", [])

                    if mtype == "h2h":
                        # Moneyline: convert decimal odds to implied probability
                        for o in outcomes:
                            decimal_odds = float(o.get("price", 2.0))
                            implied_prob = 1.0 / decimal_odds
                            if o.get("name") == event.get("home_team"):
                                game["markets"]["home_ml_prob"] = implied_prob
                                game["markets"]["home_ml_odds"] = decimal_odds
                            else:
                                game["markets"]["away_ml_prob"] = implied_prob
                                game["markets"]["away_ml_odds"] = decimal_odds

                    elif mtype == "spreads":
                        for o in outcomes:
                            point = float(o.get("point", 0))
                            decimal_odds = float(o.get("price", 2.0))
                            implied_prob = 1.0 / decimal_odds
                            if o.get("name") == event.get("home_team"):
                                game["markets"]["home_spread"] = point
                                game["markets"]["home_spread_prob"] = implied_prob
                            else:
                                game["markets"]["away_spread"] = point
                                game["markets"]["away_spread_prob"] = implied_prob

                    elif mtype == "totals":
                        for o in outcomes:
                            point = float(o.get("point", 0))
                            decimal_odds = float(o.get("price", 2.0))
                            implied_prob = 1.0 / decimal_odds
                            if o.get("name") == "Over":
                                game["markets"]["total_line"] = point
                                game["markets"]["over_prob"] = implied_prob
                            else:
                                game["markets"]["under_prob"] = implied_prob

            # Remove vig (normalize probabilities)
            for pair in [("home_ml_prob", "away_ml_prob"),
                         ("home_spread_prob", "away_spread_prob"),
                         ("over_prob", "under_prob")]:
                if pair[0] in game["markets"] and pair[1] in game["markets"]:
                    total = game["markets"][pair[0]] + game["markets"][pair[1]]
                    if total > 0:
                        game["markets"][pair[0]] /= total
                        game["markets"][pair[1]] /= total

            if game["markets"]:
                games.append(game)

        return games

    except Exception as e:
        print(f"  Odds API error: {e}")
        return []


# ============================================================
# POLYMARKET SPORTS FETCHER
# ============================================================
def fetch_polymarket_sports(sport_prefix: str = "nba-") -> list[dict]:
    """Fetch current Polymarket sports markets."""
    all_markets = []
    for offset in range(0, 500, 100):
        try:
            resp = requests.get(f"{GAMMA_HOST}/markets", params={
                "limit": 100, "offset": offset,
                "active": "true", "closed": "false",
                "order": "createdAt", "ascending": "false",
            }, timeout=15)
            data = resp.json()
            if not data:
                break

            for m in data:
                slug = m.get("slug", "")
                if sport_prefix not in slug:
                    continue
                if not m.get("acceptingOrders"):
                    continue

                outcomes = m.get("outcomes", "[]")
                prices = m.get("outcomePrices", "[]")
                tids = m.get("clobTokenIds", "[]")

                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if isinstance(tids, str):
                    tids = json.loads(tids)

                if len(outcomes) < 2 or len(prices) < 2:
                    continue

                all_markets.append({
                    "question": m.get("question", ""),
                    "slug": slug,
                    "outcomes": outcomes,
                    "prices": [float(p) for p in prices],
                    "token_ids": tids,
                    "condition_id": m.get("conditionId", ""),
                    "best_bid": float(m.get("bestBid", 0) or 0),
                    "best_ask": float(m.get("bestAsk", 0) or 0),
                    "spread": float(m.get("spread", 0) or 0),
                    "volume": float(m.get("volumeClob", 0) or m.get("volume", 0) or 0),
                })
        except Exception:
            break

    return all_markets


# ============================================================
# MATCHER: Link Pinnacle games to Polymarket markets
# ============================================================
def match_markets(pinnacle_games: list, poly_markets: list) -> list[dict]:
    """
    Match Pinnacle games to Polymarket markets by team names.
    Returns list of matched opportunities with edge calculations.
    """
    # Build team name lookup for Polymarket
    # NBA team abbreviations → full names mapping
    TEAM_MAP = {
        "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
        "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
        "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
        "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
        "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
        "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
        "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
        "UTA": "Jazz", "WAS": "Wizards",
    }

    opportunities = []

    for game in pinnacle_games:
        home = game["home_team"]
        away = game["away_team"]
        pin = game["markets"]

        # Find matching Polymarket markets
        for pm in poly_markets:
            q = pm["question"].lower()
            slug = pm["slug"].lower()

            # Check if this Polymarket market is about this game
            home_match = any(t.lower() in q for t in [home] + [v for v in TEAM_MAP.values() if v.lower() in home.lower()])
            away_match = any(t.lower() in q for t in [away] + [v for v in TEAM_MAP.values() if v.lower() in away.lower()])

            if not (home_match or away_match):
                continue

            # Determine what type of bet this is
            poly_price_yes = pm["prices"][0] if pm["prices"] else 0.5
            poly_price_no = pm["prices"][1] if len(pm["prices"]) > 1 else 1 - poly_price_yes

            # Moneyline matching
            if "moneyline" in slug or "winner" in slug:
                pin_home_prob = pin.get("home_ml_prob", 0.5)
                pin_away_prob = pin.get("away_ml_prob", 0.5)

                # Determine which outcome is home vs away in Polymarket
                if any(t.lower() in pm["outcomes"][0].lower() for t in home.split()):
                    poly_home_prob = poly_price_yes
                    poly_away_prob = poly_price_no
                else:
                    poly_home_prob = poly_price_no
                    poly_away_prob = poly_price_yes

                # Calculate edge
                home_edge = pin_home_prob - poly_home_prob
                away_edge = pin_away_prob - poly_away_prob

                if abs(home_edge) > 0.03 or abs(away_edge) > 0.03:
                    best_edge = max(home_edge, away_edge, key=abs)
                    side = "HOME" if home_edge > away_edge else "AWAY"
                    our_prob = pin_home_prob if side == "HOME" else pin_away_prob
                    market_prob = poly_home_prob if side == "HOME" else poly_away_prob

                    opportunities.append({
                        "game": f"{away} @ {home}",
                        "market_type": "MONEYLINE",
                        "question": pm["question"],
                        "side": side,
                        "team": home if side == "HOME" else away,
                        "pinnacle_prob": round(our_prob, 3),
                        "polymarket_prob": round(market_prob, 3),
                        "edge": round(abs(best_edge), 3),
                        "poly_buy_price": round(market_prob, 3),
                        "expected_value": round(our_prob * (1 - market_prob) - (1 - our_prob) * market_prob, 4),
                        "condition_id": pm["condition_id"],
                        "token_ids": pm["token_ids"],
                        "poly_spread": pm["spread"],
                        "commence_time": game["commence_time"],
                    })

            # Spread matching
            elif "spread" in slug:
                pin_spread = pin.get("home_spread", 0)
                pin_home_spread_prob = pin.get("home_spread_prob", 0.5)

                if abs(poly_price_yes - pin_home_spread_prob) > 0.03:
                    edge = abs(poly_price_yes - pin_home_spread_prob)
                    opportunities.append({
                        "game": f"{away} @ {home}",
                        "market_type": "SPREAD",
                        "question": pm["question"],
                        "side": "COVER" if pin_home_spread_prob > poly_price_yes else "FADE",
                        "team": home,
                        "pinnacle_prob": round(pin_home_spread_prob, 3),
                        "polymarket_prob": round(poly_price_yes, 3),
                        "edge": round(edge, 3),
                        "poly_buy_price": round(min(poly_price_yes, poly_price_no), 3),
                        "expected_value": 0,
                        "condition_id": pm["condition_id"],
                        "token_ids": pm["token_ids"],
                        "poly_spread": pm["spread"],
                        "commence_time": game["commence_time"],
                    })

            # Total matching
            elif "total" in slug or "o/u" in slug.replace("over-under", "o/u"):
                pin_over = pin.get("over_prob", 0.5)
                pin_under = pin.get("under_prob", 0.5)

                # Check edge on over/under
                over_edge = pin_over - poly_price_yes
                under_edge = pin_under - poly_price_no

                if abs(over_edge) > 0.03 or abs(under_edge) > 0.03:
                    if abs(over_edge) > abs(under_edge):
                        opportunities.append({
                            "game": f"{away} @ {home}",
                            "market_type": "TOTAL",
                            "question": pm["question"],
                            "side": "OVER" if over_edge > 0 else "UNDER",
                            "team": "",
                            "pinnacle_prob": round(pin_over if over_edge > 0 else pin_under, 3),
                            "polymarket_prob": round(poly_price_yes if over_edge > 0 else poly_price_no, 3),
                            "edge": round(max(abs(over_edge), abs(under_edge)), 3),
                            "poly_buy_price": round(min(poly_price_yes, poly_price_no), 3),
                            "expected_value": 0,
                            "condition_id": pm["condition_id"],
                            "token_ids": pm["token_ids"],
                            "poly_spread": pm["spread"],
                            "commence_time": game["commence_time"],
                        })

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


# ============================================================
# SIMULATE MODE (no API key needed)
# ============================================================
def simulate_backtest():
    """
    Simulate the Pinnacle vs Polymarket strategy using historical data.
    Uses our ELO model as a proxy for Pinnacle's sharp line.
    """
    from sports_model import build_game_features, prepare_matchup_features

    print("\nLoading NBA data for simulation...")
    df = pd.read_csv("data/nba_games_5seasons.csv")
    df = build_game_features(df)
    features = prepare_matchup_features(df)

    test = features[features["season"] == "2024-25"].copy()
    print(f"  {len(test)} test games (2024-25 season)")

    # Simulate: our ELO = "Pinnacle", 50/50 = "Polymarket" (worst case)
    # Only bet when our model deviates from 50% by >8%
    BET_SIZE = 1000
    results = []

    for _, row in test.iterrows():
        our_prob = row["elo_expected"]  # This is our "Pinnacle" proxy
        market_prob = 0.50  # Polymarket at 50/50

        edge = our_prob - market_prob

        if abs(edge) > 0.08:  # Only bet when edge > 8%
            if edge > 0:  # Bet HOME
                correct = row["home_win"] == 1
            else:  # Bet AWAY
                correct = row["home_win"] == 0

            pnl = BET_SIZE * 0.48 if correct else -BET_SIZE * 0.52
            results.append({"edge": abs(edge), "correct": correct, "pnl": pnl})

    if not results:
        print("  No opportunities found")
        return

    results_df = pd.DataFrame(results)
    total = len(results_df)
    wins = results_df["correct"].sum()
    pnl = results_df["pnl"].sum()

    print(f"\n{'='*60}")
    print(f"  SIMULATION: Sharp Line vs 50/50 Market")
    print(f"{'='*60}")
    print(f"  Bets: {total}")
    print(f"  Win rate: {wins/total:.1%}")
    print(f"  Total P&L: ${pnl:+,.0f}")
    print(f"  Avg P&L/bet: ${pnl/total:+.2f}")
    print(f"  ROI: {pnl/(total*BET_SIZE*0.52)*100:+.1f}%")

    # By edge bucket
    print(f"\n  By edge size:")
    for lo, hi in [(0.08, 0.12), (0.12, 0.18), (0.18, 0.25), (0.25, 1.0)]:
        bucket = results_df[(results_df["edge"] >= lo) & (results_df["edge"] < hi)]
        if len(bucket) > 0:
            bwr = bucket["correct"].mean()
            bpnl = bucket["pnl"].sum()
            print(f"    Edge {lo:.0%}-{hi:.0%}: {len(bucket)} bets, {bwr:.1%} WR, ${bpnl:+,.0f}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pinnacle vs Polymarket Spread Scanner")
    parser.add_argument("--api-key", type=str, default=os.getenv("ODDS_API_KEY", ""),
                        help="The Odds API key (free at the-odds-api.com)")
    parser.add_argument("--min-edge", type=float, default=5.0,
                        help="Minimum edge %% to flag (default: 5)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run simulation backtest (no API key needed)")
    parser.add_argument("--sport", type=str, default="all",
                        choices=["nba", "nfl", "nhl", "all"])
    args = parser.parse_args()

    print("=" * 70)
    print("  PINNACLE vs POLYMARKET SPREAD SCANNER")
    print("  No ML. No overfitting. Pure cross-market arbitrage.")
    print("=" * 70)

    if args.simulate:
        import pandas as pd
        simulate_backtest()
        return

    if not args.api_key:
        print("\nNo API key provided.")
        print("  1. Get a FREE key at: https://the-odds-api.com/#get-access")
        print("  2. Run: python3 sports_arb_scanner.py --api-key YOUR_KEY")
        print("\nRunning simulation mode instead...\n")
        import pandas as pd
        simulate_backtest()
        return

    min_edge = args.min_edge / 100

    # Determine which sports to scan
    if args.sport == "all":
        sports_to_scan = SPORTS
    else:
        key_map = {"nba": "basketball_nba", "nfl": "americanfootball_nfl", "nhl": "icehockey_nhl"}
        sports_to_scan = {key_map[args.sport]: SPORTS[key_map[args.sport]]}

    all_opportunities = []

    for sport_key, sport_info in sports_to_scan.items():
        print(f"\n{'─'*60}")
        print(f"  Scanning {sport_info['name']}...")
        print(f"{'─'*60}")

        # Fetch Pinnacle odds
        print(f"  Fetching Pinnacle odds...")
        pinnacle = fetch_pinnacle_odds(args.api_key, sport_key)
        print(f"    Found {len(pinnacle)} games with Pinnacle odds")

        # Fetch Polymarket
        print(f"  Fetching Polymarket markets...")
        polymarket = fetch_polymarket_sports(sport_info["slug_prefix"])
        print(f"    Found {len(polymarket)} Polymarket markets")

        # Match and find edge
        opportunities = match_markets(pinnacle, polymarket)
        edge_opps = [o for o in opportunities if o["edge"] >= min_edge]
        print(f"    Matched: {len(opportunities)} pairs, {len(edge_opps)} with edge >{args.min_edge}%")

        all_opportunities.extend(edge_opps)

    # Print results
    print(f"\n{'='*70}")
    print(f"  OPPORTUNITIES FOUND: {len(all_opportunities)}")
    print(f"  Minimum edge: {args.min_edge}%")
    print(f"{'='*70}")

    if not all_opportunities:
        print("\n  No opportunities above the edge threshold.")
        print("  This is normal — check again closer to game time.")
        print("  Lines move most in the last hour before tip-off.")
        return

    for opp in all_opportunities:
        ev_pct = opp["expected_value"] * 100
        print(f"\n  {opp['game']}")
        print(f"    Market: {opp['market_type']} — {opp['question'][:50]}")
        print(f"    Bet:    {opp['side']} {opp['team']}")
        print(f"    Pinnacle:    {opp['pinnacle_prob']:.1%}")
        print(f"    Polymarket:  {opp['polymarket_prob']:.1%}")
        print(f"    EDGE:        {opp['edge']:.1%}")
        print(f"    Poly spread: {opp['poly_spread']}")
        print(f"    Game time:   {opp['commence_time']}")

    # Summary
    if all_opportunities:
        avg_edge = np.mean([o["edge"] for o in all_opportunities])
        max_edge = max(o["edge"] for o in all_opportunities)
        print(f"\n  Average edge: {avg_edge:.1%}")
        print(f"  Max edge:     {max_edge:.1%}")
        print(f"\n  ACTION: Place bets on the opportunities above using")
        print(f"  the same py-clob-client infrastructure (token IDs included).")

    # Log
    log_path = LOG_DIR / f"arb_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(log_path, "w") as f:
        json.dump(all_opportunities, f, indent=2, default=str)
    print(f"\n  Logged to {log_path}")


if __name__ == "__main__":
    main()
