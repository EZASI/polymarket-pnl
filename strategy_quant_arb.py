#!/usr/bin/env python3
"""
QUANT AGENT 1 — Sportsbook vs Polymarket Arbitrage
====================================================
Finds markets where Polymarket price deviates >5% from sportsbook consensus.
Bets the sportsbook-implied direction for mean reversion profit.
"""
import json, os, time, requests
from datetime import datetime, timezone
from swarm_base import SwarmAgent, rate_limited_get, fetch_active_markets, LOG_DIR

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
GAMMA_HOST = "https://gamma-api.polymarket.com"

agent = SwarmAgent("quant_arb", "quant")

def fetch_odds():
    if not ODDS_API_KEY:
        return []
    opportunities = []
    for sport in ["basketball_nba", "icehockey_nhl", "americanfootball_nfl"]:
        resp = rate_limited_get(
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h", "oddsFormat": "decimal"}
        )
        if not resp or resp.status_code != 200:
            continue
        for event in resp.json():
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            probs = {}
            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for oc in mkt["outcomes"]:
                            name = oc["name"]
                            if name not in probs:
                                probs[name] = []
                            probs[name].append(1.0 / float(oc["price"]))
            for team, p_list in probs.items():
                if p_list:
                    avg_prob = sum(p_list) / len(p_list)
                    opportunities.append({"team": team, "opponent": away if team == home else home,
                                          "prob": avg_prob, "sport": sport.split("_")[1].upper()})
    return opportunities

def find_polymarket_match(team_name, markets):
    team_lower = team_name.lower()
    for m in markets:
        q = (m.get("question", "") + " " + m.get("slug", "")).lower()
        if team_lower in q or team_lower.split()[-1] in q:
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try: prices = json.loads(prices)
                except: prices = []
            if prices:
                return m, float(prices[0])
    return None, None

def run():
    print(f"[{agent.agent_id}] Starting arbitrage scan...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED by risk manager. Stopping.")
        return

    markets = fetch_active_markets(500)
    odds = fetch_odds()
    signals = []

    for opp in odds:
        market, poly_price = find_polymarket_match(opp["team"], markets)
        if market and poly_price:
            book_prob = opp["prob"]
            edge = book_prob - poly_price
            if abs(edge) > 0.05:  # 5%+ edge
                bet_size = min(abs(edge) * agent.capital * 0.5, agent.capital * 0.10)
                signals.append({
                    "market": market.get("question", "")[:80],
                    "condition_id": market.get("conditionId", ""),
                    "team": opp["team"],
                    "book_prob": round(book_prob, 3),
                    "poly_price": round(poly_price, 3),
                    "edge": round(edge, 3),
                    "direction": "BUY" if edge > 0 else "SELL",
                    "bet_size": round(bet_size, 2),
                })

                if edge > 0 and bet_size >= 10:
                    agent.open_position(
                        market.get("question", ""), market.get("conditionId", ""),
                        "Yes", bet_size, poly_price,
                        reason=f"Arb: book={book_prob:.0%} poly={poly_price:.0%} edge={edge:+.1%}"
                    )

    # Save signals
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] Found {len(signals)} arbitrage opportunities. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
