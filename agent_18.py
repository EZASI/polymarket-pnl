#!/usr/bin/env python3
"""
AGENT 18 — Dynamic Kelly Scaler
=================================
Adjusts the Kelly fraction based on a rolling window of the last 10 trades.
Winning streak -> scales up toward full Kelly. Losing streak -> scales down
to quarter Kelly. The adaptive fraction responds to regime changes.

Kelly math: f = (p*b - q) / b, then actual_bet = f * scale_factor * capital
where scale_factor ranges from 0.25 (cold) to 1.0 (hot).
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_18", "kelly")

MAX_POSITIONS = 8
MIN_EDGE = 0.06
MIN_VOLUME = 20000
ROLLING_WINDOW = 10   # Last N trades for streak detection
TAKE_PROFIT = 0.08
STOP_LOSS = 0.05


def get_kelly_scale():
    """Determine Kelly scale factor from rolling trade history."""
    recent = agent.trades[-ROLLING_WINDOW:] if agent.trades else []
    if len(recent) < 3:
        return 0.50  # Default to half-Kelly when insufficient data

    wins = sum(1 for t in recent if t.get("trade_pnl", 0) > 0)
    win_rate = wins / len(recent)

    # Map win rate to Kelly scale: 0% -> 0.25x, 50% -> 0.5x, 80%+ -> 1.0x
    if win_rate >= 0.70:
        scale = 1.0       # Full Kelly: hot streak
    elif win_rate >= 0.55:
        scale = 0.75      # Three-quarter Kelly: winning
    elif win_rate >= 0.40:
        scale = 0.50      # Half Kelly: neutral
    else:
        scale = 0.25      # Quarter Kelly: cold streak

    return scale


def estimate_prob(market):
    """Estimate true probability from volume momentum and price."""
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
    vol_24h = float(market.get("volume24hr", 0) or 0)
    if vol < MIN_VOLUME:
        return None, None

    # Volume acceleration as probability signal
    vol_ratio = vol_24h / max(vol, 1)
    accel = min(vol_ratio * 0.12, 0.10)
    direction_sign = 1 if price > 0.5 else -1
    est_prob = max(0.05, min(0.95, price + accel * direction_sign))

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
    print(f"  AGENT 18 -- Dynamic Kelly Scaler")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 18 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    scale = get_kelly_scale()
    recent_n = min(len(agent.trades), ROLLING_WINDOW)
    recent_wins = sum(1 for t in agent.trades[-ROLLING_WINDOW:] if t.get("trade_pnl", 0) > 0)
    print(f"  Kelly scale: {scale:.2f}x (last {recent_n}: {recent_wins}W)")

    markets = fetch_active_markets(400)
    if not markets:
        print("  No market data available")
        agent.save_state()
        return

    signals = []
    new_trades = 0
    n_open = len([p for p in agent.positions if p.get("status") == "open"])

    for m in markets:
        price, est_prob = estimate_prob(m)
        if price is None:
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

        scaled_f = f * scale
        bet_size = scaled_f * agent.capital
        bet_size = min(bet_size, agent.capital * 0.12)

        cid = m.get("conditionId", m.get("condition_id", ""))
        title = m.get("question", m.get("title", ""))[:100]

        signals.append({
            "market": title, "direction": direction,
            "price": round(eff_price, 4), "est_prob": round(eff_prob, 4),
            "edge": round(edge, 4), "raw_kelly": round(f, 4),
            "scale": scale, "scaled_kelly": round(scaled_f, 4),
            "bet_size": round(bet_size, 2),
        })

        if n_open < MAX_POSITIONS and bet_size >= 8:
            reason = f"DynKelly f={f:.3f}*{scale:.2f}={scaled_f:.3f} edge={edge:.1%}"
            opened = agent.open_position(title, cid, direction, bet_size, eff_price, reason)
            if opened:
                new_trades += 1
                n_open += 1
                print(f"  TRADE: {direction} @ {eff_price:.3f} | f*s={scaled_f:.3f}"
                      f" | edge={edge:.1%} | {title[:45]}")

    log_data = {
        "agent": "agent_18", "strategy": "dynamic_kelly",
        "time": datetime.now(timezone.utc).isoformat(),
        "kelly_scale": scale, "rolling_window": ROLLING_WINDOW,
        "signals_found": len(signals), "new_trades": new_trades,
        "signals": signals[:15],
    }
    with open(LOG_DIR / "signals_agent_18.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
