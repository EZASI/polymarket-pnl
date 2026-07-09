#!/usr/bin/env python3
"""
AGENT 13 — Whale Detector
============================
Identifies markets with unusually large single-trade volume
(vol24h > 3x average daily). Large trades = whale activity = informed flow.
Follows the whale's implied direction based on price movement.
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_13", "orderbook")

WHALE_SPIKE_THRESHOLD = 3.0  # 24h volume must be 3x average daily
MARKET_AGE_DAYS = 7          # Minimum age to estimate avg daily volume
MIN_TOTAL_VOLUME = 30000
MAX_POSITIONS = 8
TAKE_PROFIT = 0.10
STOP_LOSS = -0.05


def run():
    print(f"[{agent.agent_id}] Whale Detector — scanning for large-flow markets...")
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
    now = datetime.now(timezone.utc)
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
        if poly_price >= 0.93 or poly_price <= 0.07:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < MIN_TOTAL_VOLUME or vol24 <= 0:
            continue

        # Estimate market age from creation date
        created = m.get("createdAt") or m.get("startDate", "")
        if created:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                age_days = max((now - created_dt).days, 1)
            except:
                age_days = 30
        else:
            age_days = 30

        if age_days < MARKET_AGE_DAYS:
            continue  # Too new to establish baseline

        avg_daily_vol = vol / age_days
        whale_ratio = vol24 / max(avg_daily_vol, 1)

        if whale_ratio < WHALE_SPIKE_THRESHOLD:
            continue

        # Whale direction: inferred from price displacement
        # If whale pushed price toward Yes (>0.5), follow
        displacement = poly_price - 0.5

        candidates.append({
            "market": m,
            "poly_price": poly_price,
            "whale_ratio": whale_ratio,
            "vol24": vol24,
            "avg_daily_vol": avg_daily_vol,
            "age_days": age_days,
            "displacement": displacement,
            "liq": liq,
        })

    # Sort by whale ratio descending
    candidates.sort(key=lambda x: x["whale_ratio"], reverse=True)

    signals = []
    for c in candidates[:20]:
        m = c["market"]
        poly_price = c["poly_price"]
        disp = c["displacement"]

        # Only trade if whale moved price meaningfully
        if abs(disp) < 0.05:
            continue

        direction = "Yes" if disp > 0 else "No"
        entry_price = poly_price if direction == "Yes" else (1.0 - poly_price)

        # Bigger whale ratio + larger displacement = stronger signal
        signal_strength = min(c["whale_ratio"] * abs(disp) * 0.8, 2.5)
        bet_size = min(signal_strength * agent.capital * 0.015, agent.capital * 0.05)

        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction,
            "poly_price": round(poly_price, 3),
            "whale_ratio": round(c["whale_ratio"], 2),
            "vol24": round(c["vol24"], 2),
            "avg_daily_vol": round(c["avg_daily_vol"], 2),
            "displacement": round(disp, 3),
            "signal_strength": round(signal_strength, 3),
        })

        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                m.get("question", ""), m.get("conditionId", ""),
                direction, bet_size, entry_price,
                reason=f"Whale: {c['whale_ratio']:.1f}x avg, disp={disp:+.2f}",
            )

    # ── Save signals ──
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    print(
        f"[{agent.agent_id}] {len(signals)} whale signals, "
        f"{len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}"
    )


if __name__ == "__main__":
    run()
