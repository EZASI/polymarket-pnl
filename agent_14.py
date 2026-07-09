#!/usr/bin/env python3
"""
AGENT 14 — Thin Market Sniper
================================
Targets very low liquidity markets (<$5k liquidity) with moderate volume
(>$10k). These markets are easy to move and often mispriced. Takes small
positions betting on convergence to fair value (toward 0.5 or toward
the volume-implied direction).
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_14", "orderbook")

MAX_LIQUIDITY = 5000     # Only thin markets
MIN_VOLUME = 10000       # Must have some real trading interest
MAX_POSITIONS = 10       # More positions, smaller each
TAKE_PROFIT = 0.12       # Thin markets can move fast
STOP_LOSS = -0.06
MAX_BET_SIZE_PCT = 0.03  # Smaller positions on thin markets


def run():
    print(f"[{agent.agent_id}] Thin Market Sniper — hunting low-liq opportunities...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping.")
        return

    # ── Position management ──
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current = fetch_market_price(pos["condition_id"])
        if current is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        if pnl_pct >= TAKE_PROFIT or pnl_pct <= STOP_LOSS:
            realized = agent.close_position(pos["condition_id"], current)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    # ── Scan markets ──
    markets = fetch_active_markets(500)
    candidates = []

    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except:
                prices = []
        if not prices:
            continue

        poly_price = float(prices[0])
        if poly_price >= 0.95 or poly_price <= 0.05:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if liq > MAX_LIQUIDITY or liq <= 0:
            continue
        if vol < MIN_VOLUME:
            continue

        # Volume-to-liquidity ratio: higher = more traded per unit of depth
        vol_liq_ratio = vol / max(liq, 1)

        # Mispricing score: how far from 0.5 (more extreme = more likely mispriced)
        mispricing = abs(poly_price - 0.5)

        # Fair value convergence: extreme prices on thin markets tend to revert
        # But also check if volume confirms the direction
        vol24_ratio = vol24 / max(vol / 30, 1)  # Recent activity vs historical

        candidates.append({
            "market": m,
            "poly_price": poly_price,
            "liq": liq,
            "vol": vol,
            "vol24": vol24,
            "vol_liq_ratio": vol_liq_ratio,
            "mispricing": mispricing,
            "vol24_ratio": vol24_ratio,
        })

    # Sort by mispricing score (most mispriced thin markets first)
    candidates.sort(key=lambda x: x["mispricing"] * x["vol_liq_ratio"], reverse=True)

    signals = []
    for c in candidates[:30]:
        m = c["market"]
        poly_price = c["poly_price"]
        misp = c["mispricing"]

        # Decision: trade toward fair value if no strong recent volume,
        # otherwise follow the volume direction
        if c["vol24_ratio"] > 2.0 and misp > 0.15:
            # Strong recent volume confirming the direction — follow it
            direction = "Yes" if poly_price > 0.5 else "No"
            reason_tag = "follow_vol"
        elif misp > 0.20:
            # Extreme mispricing on thin market — bet on reversion
            direction = "No" if poly_price > 0.6 else "Yes"
            reason_tag = "reversion"
        else:
            continue

        entry_price = poly_price if direction == "Yes" else (1.0 - poly_price)
        signal_strength = min(misp * c["vol_liq_ratio"] * 0.05, 2.0)
        bet_size = min(signal_strength * agent.capital * 0.015, agent.capital * MAX_BET_SIZE_PCT)

        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction,
            "poly_price": round(poly_price, 3),
            "liquidity": round(c["liq"], 2),
            "vol_liq_ratio": round(c["vol_liq_ratio"], 1),
            "mispricing": round(misp, 3),
            "reason": reason_tag,
            "signal_strength": round(signal_strength, 3),
        })

        if bet_size >= 8 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                m.get("question", ""), m.get("conditionId", ""),
                direction, bet_size, entry_price,
                reason=f"ThinSnipe({reason_tag}): liq=${c['liq']:.0f}, misp={misp:.2f}",
            )

    # ── Save signals ──
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    print(
        f"[{agent.agent_id}] {len(signals)} thin market signals, "
        f"{len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}"
    )


if __name__ == "__main__":
    run()
