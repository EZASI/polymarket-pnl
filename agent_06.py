#!/usr/bin/env python3
"""
AGENT 06 — Volume Shock Detector: Detects sudden volume spikes (24h volume > 8%
of total volume) as proxy for breaking news. Trades in the direction the price
is moving during the shock. Fast entries, tight exits.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, get_trader_data,
    LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_06", "sentiment")

VOL_SHOCK_THRESH = 0.08  # 24h vol > 8% of total volume
BASE_SIZE, MAX_OPEN = 1500.0, 5
PROFIT_TARGET, STOP_LOSS = 0.06, 0.04


def detect_volume_shocks(markets):
    """Find markets with abnormally high recent volume."""
    shocks = []
    for m in markets:
        try:
            vol_24h = float(m.get("volume24hr", 0) or 0)
            vol_total = float(m.get("volume", 0) or 0)
            if vol_total < 10000:
                continue
            vol_ratio = vol_24h / vol_total if vol_total > 0 else 0
            if vol_ratio < VOL_SHOCK_THRESH:
                continue
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            price = float(prices[0]) if prices else None
            if price is None or price < 0.10 or price > 0.90:
                continue
            direction = "YES" if price > 0.50 else "NO"
            eff_price = price if direction == "YES" else (1.0 - price)
            shocks.append({
                "title": m.get("question", m.get("title", ""))[:100],
                "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "price": price, "direction": direction, "eff_price": eff_price,
                "vol_ratio": vol_ratio, "vol_24h": vol_24h,
            })
        except (ValueError, TypeError, KeyError):
            continue
    shocks.sort(key=lambda x: x["vol_ratio"], reverse=True)
    return shocks[:10]


def manage_positions():
    closed = 0
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        cid, entry = pos.get("condition_id"), pos["entry_price"]
        current = fetch_market_price(cid)
        if current is None or entry <= 0:
            continue
        change = (current - entry) / entry * (-1 if pos.get("outcome") == "NO" else 1)
        if change >= PROFIT_TARGET or change <= -STOP_LOSS:
            agent.close_position(cid, current); closed += 1
    return closed


def run():
    print(f"\n{'='*60}")
    print(f"  AGENT 06 — Volume Shock Detector")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 06 has been terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s) on target/stop")

    markets = fetch_active_markets(300)
    if not markets:
        print("  No market data available"); agent.save_state(); return

    shocks = detect_volume_shocks(markets)
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    signals, new_trades = [], 0

    for shock in shocks:
        signals.append({"market": shock["title"], "direction": shock["direction"],
            "price": shock["price"], "vol_ratio": round(shock["vol_ratio"] * 100, 1)})
        if n_open >= MAX_OPEN:
            continue
        size = min(BASE_SIZE * min(shock["vol_ratio"] / VOL_SHOCK_THRESH, 2.5), agent.capital * 0.15)
        reason = f"vol_shock {shock['vol_ratio']*100:.0f}% ratio, 24h=${shock['vol_24h']:,.0f}"
        if agent.open_position(shock["title"], shock["condition_id"],
                               shock["direction"], size, shock["eff_price"], reason):
            new_trades += 1; n_open += 1
            print(f"  TRADE: {shock['direction']} @ {shock['eff_price']:.3f} "
                  f"| vol={shock['vol_ratio']*100:.1f}% | {shock['title'][:50]}")

    log_data = {"agent": "agent_06", "strategy": "volume_shock",
        "time": datetime.now(timezone.utc).isoformat(),
        "signals_found": len(signals), "new_trades": new_trades, "signals": signals[:10]}
    with open(LOG_DIR / "signals_agent_06.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
