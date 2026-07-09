#!/usr/bin/env python3
"""
AGENT 10 — Event Catalyst Hunter: Finds markets near event dates (titles with
"2025", "February", "March", etc.). Prices should converge to 0/100 as events
approach. Bets on undecided markets using copy signals for direction.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, get_trader_data,
    LOG_DIR, GAMMA_HOST,
)

agent = SwarmAgent("agent_10", "sentiment")

EVENT_KW = ["2025", "2026", "february", "march", "april", "q1", "q2", "by march",
    "by february", "this month", "this week", "end of", "deadline", "inauguration",
    "hearing", "summit", "vote on", "decision", "ruling", "announcement"]
IMMINENT_KW = ["february", "this week", "this month", "by february",
               "february 2026", "march 2026"]
UNDECIDED_LOW, UNDECIDED_HIGH, MIN_VOLUME = 0.25, 0.75, 10000
BASE_SIZE, MAX_OPEN = 1300.0, 5
PROFIT_TARGET, STOP_LOSS = 0.15, 0.06


def find_catalyst_markets(markets):
    """Find event-approaching markets still in the undecided zone."""
    catalysts = []
    copy_data = get_copy_signals()
    copy_map = {}
    for sig in copy_data.get("signals", []):
        cid = sig.get("condition_id", sig.get("conditionId", ""))
        d = sig.get("direction", sig.get("outcome", "")).upper()
        if cid and d in ("YES", "NO"):
            copy_map.setdefault(cid, {"YES": 0, "NO": 0})[d] += 1

    for m in markets:
        try:
            title = m.get("question", m.get("title", ""))
            tl = title.lower()
            if not any(kw in tl for kw in EVENT_KW):
                continue
            is_imminent = any(kw in tl for kw in IMMINENT_KW)
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            price = float(prices[0]) if prices else None
            if price is None or price < UNDECIDED_LOW or price > UNDECIDED_HIGH:
                continue
            vol_total = float(m.get("volume", 0) or 0)
            if vol_total < MIN_VOLUME:
                continue
            cid = m.get("conditionId", m.get("condition_id", ""))
            cy = copy_map.get(cid, {}).get("YES", 0)
            cn = copy_map.get(cid, {}).get("NO", 0)
            if cy > cn:
                direction, sig_str = "YES", cy - cn
            elif cn > cy:
                direction, sig_str = "NO", cn - cy
            else:
                direction, sig_str = ("YES" if price > 0.50 else "NO"), 0
            eff_price = price if direction == "YES" else (1.0 - price)
            urgency = 2.0 if is_imminent else 1.0
            indecision = 1.0 - abs(price - 0.50) * 2
            score = urgency * indecision * (1.0 + sig_str * 0.3)
            catalysts.append({
                "title": title[:100], "condition_id": cid, "price": price,
                "direction": direction, "eff_price": eff_price,
                "imminent": is_imminent, "signal_strength": sig_str,
                "score": round(score, 3),
            })
        except (ValueError, TypeError, KeyError):
            continue
    catalysts.sort(key=lambda x: x["score"], reverse=True)
    return catalysts[:8]


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
    print(f"  AGENT 10 — Event Catalyst Hunter")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 10 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(500)
    if not markets:
        print("  No market data available"); agent.save_state(); return

    catalysts = find_catalyst_markets(markets)
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    signals_log, new_trades = [], 0

    for cat in catalysts:
        signals_log.append({"market": cat["title"], "direction": cat["direction"],
            "price": cat["price"], "imminent": cat["imminent"], "score": cat["score"]})
        if n_open >= MAX_OPEN:
            continue
        size = BASE_SIZE * min(cat["score"] / 1.0, 2.2)
        if cat["imminent"]:
            size *= 1.3
        size = min(size, agent.capital * 0.15)
        reason = f"catalyst {'IMMINENT' if cat['imminent'] else 'upcoming'} sig={cat['signal_strength']}"
        if agent.open_position(cat["title"], cat["condition_id"],
                               cat["direction"], size, cat["eff_price"], reason):
            new_trades += 1; n_open += 1
            tag = "IMMINENT" if cat["imminent"] else "upcoming"
            print(f"  TRADE: {cat['direction']} @ {cat['eff_price']:.3f} | {tag} | {cat['title'][:45]}")

    log_data = {"agent": "agent_10", "strategy": "event_catalyst",
        "time": datetime.now(timezone.utc).isoformat(),
        "catalysts_found": len(catalysts), "new_trades": new_trades, "signals": signals_log}
    with open(LOG_DIR / "signals_agent_10.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
