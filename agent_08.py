#!/usr/bin/env python3
"""
AGENT 08 — Contrarian Sentiment: Looks for markets where price is extreme
(>85% or <15%) but volume is declining -- crowd is complacent. Takes the
opposite side, betting on mean reversion from extreme consensus.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, get_trader_data,
    LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_08", "sentiment")

EXTREME_HIGH, EXTREME_LOW = 0.85, 0.15
VOL_DECAY_RATIO = 0.04  # 24h vol < 4% of total = volume drying up
BASE_SIZE, MAX_OPEN = 1000.0, 5
PROFIT_TARGET, STOP_LOSS = 0.12, 0.07


def find_contrarian_setups(markets):
    """Find extreme-priced markets with declining volume."""
    setups = []
    for m in markets:
        try:
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            price = float(prices[0]) if prices else None
            if price is None:
                continue
            vol_24h = float(m.get("volume24hr", 0) or 0)
            vol_total = float(m.get("volume", 0) or 0)
            if vol_total < 20000:
                continue
            vol_ratio = vol_24h / vol_total if vol_total > 0 else 1.0
            is_high, is_low = price > EXTREME_HIGH, price < EXTREME_LOW
            if not (is_high or is_low):
                continue
            if vol_ratio > VOL_DECAY_RATIO:
                continue  # Too much volume = real conviction, don't fade
            if is_high:
                direction, eff_price, extremeness = "NO", 1.0 - price, price
            else:
                direction, eff_price, extremeness = "YES", price, 1.0 - price
            score = extremeness * (1.0 - vol_ratio * 10)
            setups.append({
                "title": m.get("question", m.get("title", ""))[:100],
                "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "price": price, "direction": direction, "eff_price": eff_price,
                "vol_ratio": vol_ratio, "score": round(score, 3),
            })
        except (ValueError, TypeError, KeyError):
            continue
    setups.sort(key=lambda x: x["score"], reverse=True)
    return setups[:8]


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
    print(f"  AGENT 08 — Contrarian Sentiment")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 08 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(400)
    if not markets:
        print("  No market data available"); agent.save_state(); return

    setups = find_contrarian_setups(markets)
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    signals_log, new_trades = [], 0

    for setup in setups:
        signals_log.append({"market": setup["title"], "direction": setup["direction"],
            "price": setup["price"], "score": setup["score"],
            "vol_ratio": round(setup["vol_ratio"] * 100, 2)})
        if n_open >= MAX_OPEN:
            continue
        size = min(BASE_SIZE * min(setup["score"] / 0.7, 2.0), agent.capital * 0.12)
        reason = f"contrarian p={setup['price']:.2f} vol_decay={setup['vol_ratio']*100:.1f}%"
        if agent.open_position(setup["title"], setup["condition_id"],
                               setup["direction"], size, setup["eff_price"], reason):
            new_trades += 1; n_open += 1
            print(f"  TRADE: {setup['direction']} @ {setup['eff_price']:.3f} "
                  f"| crowd={setup['price']:.2f} | {setup['title'][:45]}")

    log_data = {"agent": "agent_08", "strategy": "contrarian_sentiment",
        "time": datetime.now(timezone.utc).isoformat(),
        "setups_found": len(setups), "new_trades": new_trades, "signals": signals_log}
    with open(LOG_DIR / "signals_agent_08.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
