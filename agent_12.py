#!/usr/bin/env python3
"""
AGENT 12 — Depth Imbalance
============================
Compares Yes vs No pricing to detect orderbook imbalance.
When outcomePrices[0] + outcomePrices[1] != 1.0 there is a pricing gap.
Also looks for markets where one side has much more volume commitment
than the other, signaling depth imbalance.
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_12", "orderbook")

PRICING_GAP_THRESHOLD = 0.015  # Sum of prices deviates >1.5% from 1.0
MIN_VOLUME = 10000
MAX_POSITIONS = 8
TAKE_PROFIT = 0.06
STOP_LOSS = -0.04


def run():
    print(f"[{agent.agent_id}] Depth Imbalance — scanning for pricing gaps...")
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
        if len(prices) < 2:
            continue

        yes_price = float(prices[0])
        no_price = float(prices[1])

        if yes_price >= 0.93 or yes_price <= 0.07:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < MIN_VOLUME:
            continue

        # Pricing gap: how far from perfectly complementary
        price_sum = yes_price + no_price
        pricing_gap = price_sum - 1.0  # Positive = overpriced sum, negative = underpriced

        # Volume imbalance proxy: price displacement from 0.5
        # Markets with more "committed capital" on one side push price away from 0.5
        volume_imbalance = abs(yes_price - no_price)

        # Combined score
        gap_magnitude = abs(pricing_gap)
        if gap_magnitude < PRICING_GAP_THRESHOLD and volume_imbalance < 0.20:
            continue

        candidates.append({
            "market": m,
            "yes_price": yes_price,
            "no_price": no_price,
            "pricing_gap": pricing_gap,
            "gap_magnitude": gap_magnitude,
            "volume_imbalance": volume_imbalance,
            "vol": vol,
            "vol24": vol24,
            "liq": liq,
        })

    # Sort by gap magnitude descending
    candidates.sort(key=lambda x: x["gap_magnitude"] + x["volume_imbalance"], reverse=True)

    signals = []
    for c in candidates[:25]:
        m = c["market"]
        yes_p = c["yes_price"]
        no_p = c["no_price"]
        gap = c["pricing_gap"]

        # Strategy: if sum > 1.0, both sides are overpriced — sell the more expensive
        # If sum < 1.0, both sides are underpriced — buy the cheaper side
        # Also consider volume imbalance direction
        if gap > PRICING_GAP_THRESHOLD:
            # Overpriced sum: the cheaper side is relatively better value
            direction = "No" if yes_p > no_p else "Yes"
            entry_price = no_p if direction == "No" else yes_p
        elif gap < -PRICING_GAP_THRESHOLD:
            # Underpriced sum: buy whichever side has stronger momentum
            direction = "Yes" if yes_p > 0.5 else "No"
            entry_price = yes_p if direction == "Yes" else no_p
        else:
            # Pure volume imbalance: follow the heavy side
            direction = "Yes" if yes_p > no_p else "No"
            entry_price = yes_p if direction == "Yes" else no_p

        signal_strength = min((c["gap_magnitude"] * 10 + c["volume_imbalance"]) * 0.5, 2.0)
        bet_size = min(signal_strength * agent.capital * 0.02, agent.capital * 0.05)

        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction,
            "yes_price": round(yes_p, 4),
            "no_price": round(no_p, 4),
            "pricing_gap": round(gap, 4),
            "volume_imbalance": round(c["volume_imbalance"], 3),
            "signal_strength": round(signal_strength, 3),
        })

        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                m.get("question", ""), m.get("conditionId", ""),
                direction, bet_size, entry_price,
                reason=f"DepthImbal: gap={gap:+.3f}, imbal={c['volume_imbalance']:.2f}",
            )

    # ── Save signals ──
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    print(
        f"[{agent.agent_id}] {len(signals)} depth imbalance signals, "
        f"{len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}"
    )


if __name__ == "__main__":
    run()
