#!/usr/bin/env python3
"""
SCALPER AGENT 7 — Momentum Scalper
====================================
Detects short-term momentum bursts via rapid price changes.
Quick in/out trades on markets with sudden volume + price moves.
"""
import json
from datetime import datetime, timezone
from swarm_base import SwarmAgent, fetch_active_markets, fetch_market_price, LOG_DIR

agent = SwarmAgent("scalp_momentum", "scalp")

def run():
    print(f"[{agent.agent_id}] Scanning for momentum opportunities...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    markets = fetch_active_markets(500)
    signals = []

    # First pass: check existing positions for exit
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current_price = fetch_market_price(pos["condition_id"])
        if current_price is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current_price - entry) / max(entry, 0.01)

        # Scalp exit: take profit at 8%+ or stop loss at -5%
        if pnl_pct >= 0.08 or pnl_pct <= -0.05:
            realized = agent.close_position(pos["condition_id"], current_price)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    # Second pass: find new momentum entries
    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.92 or poly_price <= 0.08: continue

        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)

        # Skip low-activity markets
        if vol < 20000 or liq < 3000: continue

        # Momentum signal: high recent volume relative to total
        if vol > 0 and vol24 > 0:
            vol_ratio = vol24 / max(vol, 1)
            # Momentum = recent volume is disproportionately high
            if vol_ratio > 0.05:  # >5% of total volume in last 24h
                # Price displacement from 50/50 = directional conviction
                displacement = abs(poly_price - 0.5)
                momentum_score = vol_ratio * displacement * 4

                if momentum_score > 0.15:
                    direction = "Yes" if poly_price > 0.55 else "No"
                    bet_size = min(momentum_score * agent.capital * 0.15, agent.capital * 0.05)

                    signals.append({
                        "market": m.get("question", "")[:80],
                        "condition_id": m.get("conditionId", ""),
                        "direction": direction,
                        "poly_price": round(poly_price, 3),
                        "vol_ratio": round(vol_ratio, 4),
                        "momentum_score": round(momentum_score, 3),
                        "vol24": vol24,
                    })

                    if bet_size >= 10 and len(agent.positions) < 8:
                        agent.open_position(
                            m.get("question", ""), m.get("conditionId", ""),
                            direction, bet_size, poly_price,
                            reason=f"Momentum: score={momentum_score:.2f} vol_ratio={vol_ratio:.1%}"
                        )

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals,
                    "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} momentum signals, {len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
