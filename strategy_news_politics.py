#!/usr/bin/env python3
"""
NEWS HAWK AGENT 4 — Political Sentiment Scanner
=================================================
Scans political markets for sentiment shifts based on market price movements.
Uses volume spikes and price momentum as proxy for news events.
"""
import json, time
from datetime import datetime, timezone
from swarm_base import SwarmAgent, rate_limited_get, fetch_active_markets, LOG_DIR

GAMMA_HOST = "https://gamma-api.polymarket.com"
agent = SwarmAgent("news_politics", "news")

POLITICAL_KEYWORDS = ["trump", "biden", "election", "president", "congress", "senate",
    "governor", "republican", "democrat", "tariff", "executive order", "supreme court",
    "nomination", "cabinet", "impeach", "indictment", "policy"]

def run():
    print(f"[{agent.agent_id}] Scanning political markets for sentiment shifts...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    markets = fetch_active_markets(500)
    political = []
    for m in markets:
        q = (m.get("question","") + " " + m.get("slug","")).lower()
        if any(kw in q for kw in POLITICAL_KEYWORDS):
            political.append(m)

    # Look for high-volume markets with prices at extremes (strong conviction)
    signals = []
    for m in political:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.95 or poly_price <= 0.05: continue

        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if liq <= 0: continue

        # Volume-to-liquidity ratio indicates recent activity
        vol_liq = vol / liq
        # High activity + price near extremes = strong conviction trade
        extremity = abs(poly_price - 0.5) * 2

        if vol_liq > 30 and extremity > 0.4:
            # Market has strong conviction with high activity — follow the trend
            direction = "Yes" if poly_price > 0.5 else "No"
            confidence = min(extremity * 0.8, 0.85)
            edge = confidence - poly_price if direction == "Yes" else confidence - (1 - poly_price)

            if abs(edge) > 0.03:
                bet_size = min(abs(edge) * agent.capital * 0.3, agent.capital * 0.08)
                signals.append({"market": m.get("question","")[:80], "condition_id": m.get("conditionId",""),
                    "direction": direction, "poly_price": round(poly_price, 3), "edge": round(edge, 3),
                    "vol_liq": round(vol_liq, 1), "volume": vol, "confidence": round(confidence, 3)})

                if bet_size >= 10:
                    agent.open_position(m.get("question",""), m.get("conditionId",""),
                        direction, bet_size, poly_price, reason=f"Political momentum: vol/liq={vol_liq:.0f}")

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} political signals. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
