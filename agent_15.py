#!/usr/bin/env python3
"""
AGENT 15 — Spread Compression
================================
Monitors markets where the implied spread (abs(price - 0.5) * 2) is very
wide. When volume increases on a wide-spread market, the spread should
compress. Takes positions in the direction of the compression (toward 0.5).
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_15", "orderbook")

MIN_SPREAD = 0.40          # Implied spread must be at least 40% (price < 0.30 or > 0.70)
VOLUME_UPTICK_THRESHOLD = 1.5  # 24h vol must be 1.5x avg daily
MIN_VOLUME = 15000
MIN_LIQUIDITY = 1000
MAX_POSITIONS = 8
TAKE_PROFIT = 0.09
STOP_LOSS = -0.05


def run():
    print(f"[{agent.agent_id}] Spread Compression — scanning wide-spread markets...")
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
        if poly_price >= 0.97 or poly_price <= 0.03:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < MIN_VOLUME or liq < MIN_LIQUIDITY:
            continue

        # Implied spread: how far price is from 0.5, scaled to 0-1
        implied_spread = abs(poly_price - 0.5) * 2.0

        if implied_spread < MIN_SPREAD:
            continue  # Spread is already tight

        # Volume uptick: recent volume vs historical average
        avg_daily_vol = vol / 30  # Rough estimate
        vol_uptick = vol24 / max(avg_daily_vol, 1)

        if vol_uptick < VOLUME_UPTICK_THRESHOLD:
            continue  # No volume catalyst for compression

        # Compression direction: spread should tighten toward 0.5
        # If price is high (>0.5), compression means price drops — bet No
        # If price is low (<0.5), compression means price rises — bet Yes
        compression_direction = "Yes" if poly_price < 0.5 else "No"

        candidates.append({
            "market": m,
            "poly_price": poly_price,
            "implied_spread": implied_spread,
            "vol_uptick": vol_uptick,
            "vol24": vol24,
            "avg_daily_vol": avg_daily_vol,
            "liq": liq,
            "compression_direction": compression_direction,
        })

    # Sort by widest spread with highest volume uptick
    candidates.sort(key=lambda x: x["implied_spread"] * x["vol_uptick"], reverse=True)

    signals = []
    for c in candidates[:25]:
        m = c["market"]
        poly_price = c["poly_price"]
        direction = c["compression_direction"]

        # Entry price: buy the cheap side (compression = buying the underpriced side)
        entry_price = poly_price if direction == "Yes" else (1.0 - poly_price)

        # Wider spread + stronger volume uptick = bigger signal
        signal_strength = min(c["implied_spread"] * c["vol_uptick"] * 0.4, 2.0)
        bet_size = min(signal_strength * agent.capital * 0.02, agent.capital * 0.05)

        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction,
            "poly_price": round(poly_price, 3),
            "implied_spread": round(c["implied_spread"], 3),
            "vol_uptick": round(c["vol_uptick"], 2),
            "vol24": round(c["vol24"], 2),
            "liquidity": round(c["liq"], 2),
            "signal_strength": round(signal_strength, 3),
        })

        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                m.get("question", ""), m.get("conditionId", ""),
                direction, bet_size, entry_price,
                reason=f"SpreadCompr: spread={c['implied_spread']:.2f}, vol_up={c['vol_uptick']:.1f}x",
            )

    # ── Save signals ──
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    print(
        f"[{agent.agent_id}] {len(signals)} spread compression signals, "
        f"{len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}"
    )


if __name__ == "__main__":
    run()
