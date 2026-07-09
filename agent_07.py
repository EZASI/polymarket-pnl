#!/usr/bin/env python3
"""
AGENT 07 — Smart Money Follower: Uses copy_signals.json data. When 3+ tracked
traders with >55% WR all bet the same direction on a market, follows with
conviction. Position size weighted by trader P&L history.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, get_trader_data,
    LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_07", "sentiment")

MIN_TRADERS_AGREE = 3
MIN_WIN_RATE = 55.0
BASE_SIZE, MAX_OPEN = 1200.0, 6
PROFIT_TARGET, STOP_LOSS = 0.08, 0.05


def find_convergence_signals():
    """Find markets where multiple smart traders agree."""
    signals_data = get_copy_signals()
    trader_data = get_trader_data()
    # Build trader quality map
    trader_quality = {}
    for t in trader_data.get("traders", []):
        addr = t.get("address", t.get("id", ""))
        wr = float(t.get("win_rate", t.get("winRate", 0)) or 0)
        pnl = float(t.get("pnl", t.get("profit", 0)) or 0)
        if wr >= MIN_WIN_RATE:
            trader_quality[addr] = {"win_rate": wr, "pnl": pnl}

    # Group signals by market + direction
    market_votes = {}
    for sig in signals_data.get("signals", []):
        cid = sig.get("condition_id", sig.get("conditionId", ""))
        trader = sig.get("trader", sig.get("address", ""))
        direction = sig.get("direction", sig.get("outcome", "")).upper()
        title = sig.get("market", sig.get("title", ""))
        if not cid or direction not in ("YES", "NO") or trader not in trader_quality:
            continue
        if cid not in market_votes:
            market_votes[cid] = {"YES": [], "NO": [], "title": title}
        market_votes[cid][direction].append(trader_quality[trader])

    convergences = []
    for cid, votes in market_votes.items():
        for direction in ("YES", "NO"):
            traders = votes[direction]
            if len(traders) < MIN_TRADERS_AGREE:
                continue
            avg_wr = sum(t["win_rate"] for t in traders) / len(traders)
            total_pnl = sum(t["pnl"] for t in traders)
            conviction = len(traders) * (avg_wr / 50.0) * max(1.0, 1.0 + total_pnl / 50000)
            convergences.append({
                "condition_id": cid, "title": votes["title"][:100],
                "direction": direction, "n_traders": len(traders),
                "avg_win_rate": round(avg_wr, 1), "total_pnl": round(total_pnl, 2),
                "conviction": round(conviction, 2),
            })
    convergences.sort(key=lambda x: x["conviction"], reverse=True)
    return convergences[:8]


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
    print(f"  AGENT 07 — Smart Money Follower")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 07 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    convergences = find_convergence_signals()
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    signals_log, new_trades = [], 0

    for conv in convergences:
        signals_log.append(conv)
        if n_open >= MAX_OPEN:
            continue
        price = fetch_market_price(conv["condition_id"])
        if price is None or price < 0.10 or price > 0.90:
            continue
        eff_price = price if conv["direction"] == "YES" else (1.0 - price)
        size = min(BASE_SIZE * min(conv["conviction"] / 3.0, 2.5), agent.capital * 0.15)
        reason = f"smart_money {conv['n_traders']}traders avgWR={conv['avg_win_rate']}%"
        if agent.open_position(conv["title"], conv["condition_id"],
                               conv["direction"], size, eff_price, reason):
            new_trades += 1; n_open += 1
            print(f"  TRADE: {conv['direction']} @ {eff_price:.3f} "
                  f"| {conv['n_traders']} traders agree | {conv['title'][:45]}")

    log_data = {"agent": "agent_07", "strategy": "smart_money_follow",
        "time": datetime.now(timezone.utc).isoformat(),
        "convergences_found": len(convergences), "new_trades": new_trades,
        "signals": signals_log}
    with open(LOG_DIR / "signals_agent_07.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
