#!/usr/bin/env python3
"""
AGENT 11 — Liquidity Vacuum
============================
Finds markets where liquidity dropped significantly relative to volume
(liq/vol ratio < 0.05). Low liquidity = wide spreads = opportunity to
enter at favorable prices. Trades the direction of recent price movement.
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_11", "orderbook")

LIQ_VOL_THRESHOLD = 0.05   # Liquidity/volume ratio that signals a vacuum
MIN_VOLUME = 20000          # Minimum total volume to consider
MAX_POSITIONS = 8
TAKE_PROFIT = 0.08
STOP_LOSS = -0.05


def run():
    print(f"[{agent.agent_id}] Liquidity Vacuum — scanning for drained markets...")
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
        if poly_price >= 0.92 or poly_price <= 0.08:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < MIN_VOLUME or liq <= 0:
            continue

        liq_vol_ratio = liq / vol
        if liq_vol_ratio >= LIQ_VOL_THRESHOLD:
            continue  # Not a vacuum

        # Price direction proxy: if price > 0.5 momentum is toward Yes
        price_momentum = poly_price - 0.5
        # Stronger signal when vol24 is high relative to remaining liquidity
        vol24_liq_ratio = vol24 / max(liq, 1)

        candidates.append({
            "market": m,
            "poly_price": poly_price,
            "liq_vol_ratio": liq_vol_ratio,
            "vol24_liq_ratio": vol24_liq_ratio,
            "price_momentum": price_momentum,
            "liq": liq,
            "vol": vol,
            "vol24": vol24,
        })

    # Sort by lowest liq/vol ratio first (deepest vacuums)
    candidates.sort(key=lambda x: x["liq_vol_ratio"])

    signals = []
    for c in candidates[:25]:
        m = c["market"]
        poly_price = c["poly_price"]
        momentum = c["price_momentum"]

        # Trade in the direction of momentum
        direction = "Yes" if momentum > 0.03 else ("No" if momentum < -0.03 else None)
        if direction is None:
            continue

        signal_strength = min(c["vol24_liq_ratio"] * 0.3, 2.0)
        bet_size = min(signal_strength * agent.capital * 0.02, agent.capital * 0.05)

        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction,
            "poly_price": round(poly_price, 3),
            "liq_vol_ratio": round(c["liq_vol_ratio"], 4),
            "vol24_liq_ratio": round(c["vol24_liq_ratio"], 2),
            "liquidity": round(c["liq"], 2),
            "signal_strength": round(signal_strength, 3),
        })

        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                m.get("question", ""), m.get("conditionId", ""),
                direction, bet_size, poly_price,
                reason=f"LiqVacuum: liq/vol={c['liq_vol_ratio']:.3f}, v24/liq={c['vol24_liq_ratio']:.1f}x",
            )

    # ── Save signals ──
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    print(
        f"[{agent.agent_id}] {len(signals)} liquidity vacuum signals, "
        f"{len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}"
    )


if __name__ == "__main__":
    run()
