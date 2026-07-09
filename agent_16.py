#!/usr/bin/env python3
"""
AGENT 16 — Full Kelly Aggressor
=================================
Estimates true probability from volume-weighted price momentum. Calculates
the full Kelly fraction: f = (p*b - q) / b. No fractional Kelly -- uses the
full amount for maximum bankroll growth. Aggressive sizing on high-edge
markets (>10% edge). Max 5 positions.

Kelly math: p = estimated prob, q = 1-p, b = (1/price) - 1 (decimal odds).
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_16", "kelly")

MAX_POSITIONS = 5
MIN_EDGE = 0.10        # Only trade when estimated edge > 10%
MIN_VOLUME = 25000
TAKE_PROFIT = 0.10
STOP_LOSS = 0.06


def estimate_true_prob(market):
    """Estimate true probability from volume-weighted price momentum."""
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

    vol_total = float(market.get("volume", 0) or 0)
    vol_24h = float(market.get("volume24hr", 0) or 0)
    if vol_total < MIN_VOLUME:
        return None, None

    # Volume ratio indicates momentum strength
    vol_ratio = vol_24h / max(vol_total, 1)
    # Momentum adjustment: high recent volume pushes estimated prob toward price
    # direction, amplifying the market signal
    momentum_shift = vol_ratio * 0.15 * (1 if price > 0.5 else -1)
    estimated_prob = max(0.05, min(0.95, price + momentum_shift))

    return price, estimated_prob


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
    """Check open positions for profit target or stop loss."""
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
    print(f"  AGENT 16 -- Full Kelly Aggressor")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 16 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(400)
    if not markets:
        print("  No market data available")
        agent.save_state()
        return

    signals = []
    new_trades = 0
    n_open = len([p for p in agent.positions if p.get("status") == "open"])

    for m in markets:
        price, est_prob = estimate_true_prob(m)
        if price is None or est_prob is None:
            continue

        direction = "YES" if est_prob > price else "NO"
        eff_price = price if direction == "YES" else (1.0 - price)
        eff_prob = est_prob if direction == "YES" else (1.0 - est_prob)
        edge = eff_prob - eff_price

        if edge < MIN_EDGE:
            continue

        f = kelly_fraction(eff_prob, eff_price)
        if f <= 0:
            continue

        bet_size = f * agent.capital
        bet_size = min(bet_size, agent.capital * 0.15)

        cid = m.get("conditionId", m.get("condition_id", ""))
        title = m.get("question", m.get("title", ""))[:100]

        signals.append({
            "market": title, "direction": direction,
            "price": round(eff_price, 4), "est_prob": round(eff_prob, 4),
            "edge": round(edge, 4), "kelly_f": round(f, 4),
            "bet_size": round(bet_size, 2),
        })

        if n_open < MAX_POSITIONS and bet_size >= 10:
            reason = f"FullKelly f={f:.3f} edge={edge:.1%} p={eff_prob:.3f}"
            opened = agent.open_position(title, cid, direction, bet_size, eff_price, reason)
            if opened:
                new_trades += 1
                n_open += 1
                print(f"  TRADE: {direction} @ {eff_price:.3f} | kelly={f:.3f} "
                      f"| edge={edge:.1%} | {title[:50]}")

    log_data = {
        "agent": "agent_16", "strategy": "full_kelly",
        "time": datetime.now(timezone.utc).isoformat(),
        "signals_found": len(signals), "new_trades": new_trades,
        "signals": signals[:15],
    }
    with open(LOG_DIR / "signals_agent_16.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
