#!/usr/bin/env python3
"""
NEWS HAWK AGENT 6 — World Events & Geopolitics
================================================
Monitors global events markets (war, treaties, diplomacy).
Trades on high-volume conviction signals.
"""
import json
from datetime import datetime, timezone
from swarm_base import SwarmAgent, fetch_active_markets, LOG_DIR

agent = SwarmAgent("news_world", "news")

WORLD_KEYWORDS = ["ukraine", "russia", "china", "taiwan", "war", "ceasefire", "nato",
    "sanctions", "peace", "military", "israel", "iran", "north korea", "treaty",
    "united nations", "middle east", "invasion", "territorial", "climate", "who"]

def run():
    print(f"[{agent.agent_id}] Scanning world events markets...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    markets = fetch_active_markets(500)
    world = [m for m in markets if any(kw in (m.get("question","")+" "+m.get("slug","")).lower() for kw in WORLD_KEYWORDS)]

    signals = []
    for m in world:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.95 or poly_price <= 0.05: continue

        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 50000 or liq < 5000: continue

        # High volume world events with strong conviction
        extremity = abs(poly_price - 0.5) * 2
        if extremity > 0.6:
            direction = "Yes" if poly_price > 0.7 else "No"
            edge = 0.05 if extremity > 0.8 else 0.03
            bet_size = min(edge * agent.capital * 0.3, agent.capital * 0.06)

            signals.append({"market": m.get("question","")[:80], "condition_id": m.get("conditionId",""),
                "direction": direction, "poly_price": round(poly_price, 3), "volume": vol,
                "extremity": round(extremity, 2)})

            if bet_size >= 10:
                agent.open_position(m.get("question",""), m.get("conditionId",""),
                    direction, bet_size, poly_price, reason=f"World event conviction: {extremity:.0%}")

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} world event signals. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
