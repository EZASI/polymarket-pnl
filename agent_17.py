#!/usr/bin/env python3
"""
AGENT 17 — Half Kelly Conservative
====================================
Estimates true probability by blending copy-signal average win rate with
the current market price. Uses half-Kelly (f/2) for safer sizing and lower
variance. Takes more positions (up to 12) with smaller sizes. Targets
moderate edge in the 5-15% range.

Kelly math: f = (p*b - q) / b, then bet = (f/2) * capital.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_17", "kelly")

MAX_POSITIONS = 12
MIN_EDGE = 0.05
MAX_EDGE = 0.15        # Avoid extreme edges (likely data error)
MIN_VOLUME = 15000
TAKE_PROFIT = 0.07
STOP_LOSS = 0.045


def get_signal_win_rate():
    """Extract average win rate from copy signals."""
    signals = get_copy_signals()
    rates = []
    for sig in signals.get("signals", []):
        wr = sig.get("win_rate", sig.get("winRate", 0))
        if wr and wr > 0:
            rates.append(float(wr) / 100.0 if wr > 1 else float(wr))
    return sum(rates) / len(rates) if rates else 0.55


def estimate_prob(market, signal_wr):
    """Blend copy-signal win rate with market price for probability estimate."""
    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, ValueError):
            return None, None
    if not prices:
        return None, None

    price = float(prices[0])
    if price < 0.08 or price > 0.92:
        return None, None

    vol = float(market.get("volume", 0) or 0)
    if vol < MIN_VOLUME:
        return None, None

    # Blend: 60% market price + 40% signal win rate adjusted for direction
    # If price > 0.5, signal_wr reinforces YES; if < 0.5, reinforces NO
    if price > 0.5:
        est_prob = 0.60 * price + 0.40 * signal_wr
    else:
        est_prob = 0.60 * price + 0.40 * (1.0 - signal_wr)

    est_prob = max(0.05, min(0.95, est_prob))
    return price, est_prob


def kelly_fraction(p, price):
    """Full Kelly: f = (p*b - q) / b where b = (1/price) - 1."""
    if price <= 0.01 or price >= 0.99:
        return 0.0
    b = (1.0 / price) - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(f, 0.0)


def manage_positions():
    closed = 0
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current = fetch_market_price(pos["condition_id"])
        if current is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        if pos.get("outcome") == "NO":
            pnl_pct = -pnl_pct
        if pnl_pct >= TAKE_PROFIT or pnl_pct <= -STOP_LOSS:
            agent.close_position(pos["condition_id"], current)
            closed += 1
    return closed


def run():
    print(f"\n{'='*60}")
    print(f"  AGENT 17 -- Half Kelly Conservative")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 17 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(400)
    if not markets:
        print("  No market data available")
        agent.save_state()
        return

    signal_wr = get_signal_win_rate()
    print(f"  Copy signal avg win rate: {signal_wr:.1%}")

    signals = []
    new_trades = 0
    n_open = len([p for p in agent.positions if p.get("status") == "open"])

    for m in markets:
        price, est_prob = estimate_prob(m, signal_wr)
        if price is None:
            continue

        direction = "YES" if est_prob > price else "NO"
        eff_price = price if direction == "YES" else (1.0 - price)
        eff_prob = est_prob if direction == "YES" else (1.0 - est_prob)
        edge = eff_prob - eff_price

        if edge < MIN_EDGE or edge > MAX_EDGE:
            continue

        f = kelly_fraction(eff_prob, eff_price)
        if f <= 0:
            continue

        half_f = f / 2.0  # Half Kelly for lower variance
        bet_size = half_f * agent.capital
        bet_size = min(bet_size, agent.capital * 0.08)  # Conservative cap

        cid = m.get("conditionId", m.get("condition_id", ""))
        title = m.get("question", m.get("title", ""))[:100]

        signals.append({
            "market": title, "direction": direction,
            "price": round(eff_price, 4), "est_prob": round(eff_prob, 4),
            "edge": round(edge, 4), "full_kelly": round(f, 4),
            "half_kelly": round(half_f, 4), "bet_size": round(bet_size, 2),
        })

        if n_open < MAX_POSITIONS and bet_size >= 8:
            reason = f"HalfKelly f/2={half_f:.3f} edge={edge:.1%}"
            opened = agent.open_position(title, cid, direction, bet_size, eff_price, reason)
            if opened:
                new_trades += 1
                n_open += 1
                print(f"  TRADE: {direction} @ {eff_price:.3f} | f/2={half_f:.3f}"
                      f" | edge={edge:.1%} | {title[:45]}")

    log_data = {
        "agent": "agent_17", "strategy": "half_kelly",
        "time": datetime.now(timezone.utc).isoformat(),
        "signal_win_rate": round(signal_wr, 4),
        "signals_found": len(signals), "new_trades": new_trades,
        "signals": signals[:20],
    }
    with open(LOG_DIR / "signals_agent_17.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
