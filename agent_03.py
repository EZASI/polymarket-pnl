#!/usr/bin/env python3
"""
STAT ARB AGENT 03 — Volatility Arbitrage
==========================================
Identifies markets where the price is stable (low implied volatility) but
fundamentals suggest movement is imminent: high volume + narrow price band +
approaching event deadline.

Buys cheap "options-like" positions near 0.5 before expected volatility expansion.
When vol is low, both Yes and No sides near 0.50 are cheap — a big move in either
direction yields profit. Trades the side with stronger volume conviction.
"""
import json
import math
from datetime import datetime, timezone, timedelta
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_03", "stat_arb")

MAX_POSITIONS = 10
PROFIT_EXIT = 0.10       # Larger target — waiting for vol expansion
LOSS_EXIT = -0.04        # Tight stop — if vol doesn't come, cut quickly
VOL_SQUEEZE_BAND = 0.08  # Price within 0.42-0.58 = "low vol" zone


def calc_implied_vol(price):
    """Estimate implied volatility from binary option price.
    Price near 0.50 = maximum uncertainty = high implied vol.
    Price near 0 or 1 = low uncertainty = low implied vol.
    We want markets where price says 'low vol' but activity says otherwise."""
    # Binary entropy as vol proxy
    if price <= 0.01 or price >= 0.99:
        return 0.0
    return -(price * math.log2(price) + (1 - price) * math.log2(1 - price))


def calc_vol_squeeze_score(price, volume, vol24, liquidity):
    """Score how 'squeezed' a market is — high activity but narrow price range.
    Higher score = more likely to see a breakout."""
    # Price stability component: how close to 0.50
    price_stability = 1.0 - abs(price - 0.5) * 4  # Max at 0.50, zero at 0.25/0.75
    price_stability = max(price_stability, 0)

    # Volume pressure: recent volume relative to liquidity
    if liquidity < 1:
        return 0.0
    vol_pressure = vol24 / liquidity  # High ratio = lots of trading vs available depth

    # Overall activity intensity
    activity = math.log1p(vol24) / max(math.log1p(volume), 1)

    # Squeeze score: stable price + high activity = coiled spring
    squeeze = price_stability * vol_pressure * (1 + activity)
    return squeeze


def estimate_direction_from_volume(price, vol24, volume):
    """Guess which direction the breakout will go based on volume patterns.
    Rising volume with price slightly above 0.5 suggests Yes momentum.
    Rising volume with price slightly below 0.5 suggests No momentum."""
    if vol24 <= 0 or volume <= 0:
        return "Yes", 0.5
    vol_ratio = vol24 / max(volume, 1)
    # If price is drifting above 0.5 with volume, lean Yes
    if price >= 0.50:
        confidence = 0.5 + min(vol_ratio * 2, 0.3) + (price - 0.5) * 0.5
        return "Yes", min(confidence, 0.85)
    else:
        confidence = 0.5 + min(vol_ratio * 2, 0.3) + (0.5 - price) * 0.5
        return "No", min(confidence, 0.85)


def run():
    print(f"[{agent.agent_id}] Volatility arbitrage scan starting...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED by risk manager. Stopping.")
        return

    # --- Position management ---
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current = fetch_market_price(pos["condition_id"])
        if current is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        if pnl_pct >= PROFIT_EXIT or pnl_pct <= LOSS_EXIT:
            realized = agent.close_position(pos["condition_id"], current)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    # --- Find vol-squeezed markets ---
    markets = fetch_active_markets(500)
    squeeze_candidates = []

    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except:
                continue
        if not prices:
            continue
        price = float(prices[0])

        # Only interested in the "low vol" zone near 0.50
        if abs(price - 0.5) > VOL_SQUEEZE_BAND:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < 20000 or vol24 < 1000 or liq < 2000:
            continue

        squeeze_score = calc_vol_squeeze_score(price, vol, vol24, liq)
        implied_vol = calc_implied_vol(price)

        if squeeze_score < 0.5:
            continue

        direction, confidence = estimate_direction_from_volume(price, vol24, vol)

        squeeze_candidates.append({
            "question": m.get("question", ""),
            "condition_id": m.get("conditionId", ""),
            "price": price,
            "volume": vol,
            "vol24": vol24,
            "liquidity": liq,
            "squeeze_score": squeeze_score,
            "implied_vol": implied_vol,
            "direction": direction,
            "confidence": confidence,
        })

    # Sort by squeeze score — highest pressure first
    squeeze_candidates.sort(key=lambda x: x["squeeze_score"], reverse=True)
    signals = []

    for cand in squeeze_candidates[:20]:
        edge = (cand["confidence"] - cand["price"]) if cand["direction"] == "Yes" \
            else (cand["confidence"] - (1 - cand["price"]))
        edge = max(edge, cand["squeeze_score"] * 0.05)

        sig = {
            "market": cand["question"][:80],
            "condition_id": cand["condition_id"],
            "price": round(cand["price"], 3),
            "squeeze_score": round(cand["squeeze_score"], 3),
            "implied_vol": round(cand["implied_vol"], 3),
            "direction": cand["direction"],
            "confidence": round(cand["confidence"], 3),
            "vol24": cand["vol24"],
            "edge": round(edge, 4),
        }
        signals.append(sig)

        bet_size = min(edge * agent.capital * 0.3, agent.capital * 0.06)
        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                cand["question"], cand["condition_id"],
                cand["direction"], bet_size, cand["price"],
                reason=f"VolSqueeze: sq={cand['squeeze_score']:.2f} conf={cand['confidence']:.0%}"
            )

    # --- Write signals ---
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy": "volatility_arbitrage",
            "candidates_scanned": len(squeeze_candidates),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    s = agent.get_summary()
    print(f"[{agent.agent_id}] {len(squeeze_candidates)} squeeze candidates, {len(signals)} signals, "
          f"{s['open_positions']} open. PnL: ${s['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
