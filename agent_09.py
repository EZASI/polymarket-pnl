#!/usr/bin/env python3
"""
AGENT 09 — Political Momentum: Focuses on political/world markets.
Identifies momentum shifts (>5% move + high volume). Rides the wave.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, get_trader_data,
    LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_09", "sentiment")

POLITICAL_KW = [
    "president", "election", "trump", "biden", "congress", "senate",
    "governor", "poll", "vote", "democrat", "republican", "party",
    "minister", "nato", "ukraine", "russia", "china", "war",
    "sanctions", "tariff", "impeach", "scotus", "supreme court",
    "federal reserve", "government", "policy", "executive order",
    "primary", "nominee", "cabinet",
]
MIN_MOMENTUM = 0.05
MIN_VOL_24H = 15000
BASE_SIZE, MAX_OPEN = 1400.0, 5
PROFIT_TARGET, STOP_LOSS = 0.09, 0.05


def find_momentum_plays(markets):
    """Find political markets with strong recent momentum."""
    plays = []
    for m in markets:
        try:
            title = m.get("question", m.get("title", ""))
            if not any(kw in title.lower() for kw in POLITICAL_KW):
                continue
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            price = float(prices[0]) if prices else None
            if price is None or price < 0.08 or price > 0.92:
                continue
            vol_24h = float(m.get("volume24hr", 0) or 0)
            if vol_24h < MIN_VOL_24H:
                continue
            vol_total = float(m.get("volume", 0) or 0)
            vol_ratio = vol_24h / vol_total if vol_total > 0 else 0
            displacement = abs(price - 0.50)
            if displacement < MIN_MOMENTUM:
                continue
            if price > 0.55:
                direction, eff_price, momentum = "YES", price, price - 0.50
            elif price < 0.45:
                direction, eff_price, momentum = "NO", 1.0 - price, 0.50 - price
            else:
                continue
            score = momentum * (1.0 + vol_ratio * 5)
            plays.append({
                "title": title[:100], "condition_id": m.get("conditionId", m.get("condition_id", "")),
                "price": price, "direction": direction, "eff_price": eff_price,
                "momentum": round(momentum, 3), "vol_24h": vol_24h,
                "vol_ratio": round(vol_ratio, 4), "score": round(score, 3),
            })
        except (ValueError, TypeError, KeyError):
            continue
    plays.sort(key=lambda x: x["score"], reverse=True)
    return plays[:8]


def manage_positions():
    closed = 0
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        cid = pos.get("condition_id")
        current = fetch_market_price(cid)
        if current is None:
            continue
        entry = pos["entry_price"]
        if entry <= 0:
            continue
        change = (current - entry) / entry
        if pos.get("outcome") == "NO":
            change = -change
        if change >= PROFIT_TARGET or change <= -STOP_LOSS:
            agent.close_position(cid, current)
            closed += 1
    return closed


def run():
    print(f"\n{'='*60}")
    print(f"  AGENT 09 — Political Momentum")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 09 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(500)
    if not markets:
        print("  No market data available"); agent.save_state(); return

    plays = find_momentum_plays(markets)
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    signals_log, new_trades = [], 0

    for play in plays:
        signals_log.append({"market": play["title"], "direction": play["direction"],
            "price": play["price"], "momentum": play["momentum"], "score": play["score"]})
        if n_open >= MAX_OPEN:
            continue
        size = min(BASE_SIZE * min(play["score"] / 0.10, 2.0), agent.capital * 0.15)
        reason = f"political_momentum {play['momentum']*100:.0f}% vol24h=${play['vol_24h']:,.0f}"
        if agent.open_position(play["title"], play["condition_id"],
                               play["direction"], size, play["eff_price"], reason):
            new_trades += 1; n_open += 1
            print(f"  TRADE: {play['direction']} @ {play['eff_price']:.3f}"
                  f" | mom={play['momentum']*100:.0f}% | {play['title'][:45]}")

    log_data = {"agent": "agent_09", "strategy": "political_momentum",
        "time": datetime.now(timezone.utc).isoformat(),
        "plays_found": len(plays), "new_trades": new_trades, "signals": signals_log}
    with open(LOG_DIR / "signals_agent_09.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
