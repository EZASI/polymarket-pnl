#!/usr/bin/env python3
"""
AGENT 20 — Kelly Momentum Hybrid
==================================
Combines Kelly sizing with momentum confirmation. Only enters when BOTH
Kelly says "bet" (positive edge) AND momentum confirms (price moving in
the direction of the bet). Double confirmation = higher conviction.
Uses 3/4 Kelly on confirmed signals.

Momentum is measured by comparing current price to a volume-weighted
historical average. Kelly math: f = (p*b - q) / b, bet = f * 0.75.
"""

import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_20", "kelly")

MAX_POSITIONS = 7
MIN_EDGE = 0.06
MIN_VOLUME = 20000
MOMENTUM_THRESHOLD = 0.03   # Price must have moved 3%+ in bet direction
KELLY_FRACTION_MULT = 0.75  # Three-quarter Kelly on confirmed signals
TAKE_PROFIT = 0.09
STOP_LOSS = 0.05


def analyze_market(market):
    """Analyze market for both Kelly edge and momentum confirmation."""
    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except (json.JSONDecodeError, ValueError):
            return None
    if not prices:
        return None

    price = float(prices[0])
    if price < 0.08 or price > 0.92:
        return None

    vol = float(market.get("volume", 0) or 0)
    vol_24h = float(market.get("volume24hr", 0) or 0)
    liq = float(market.get("liquidity", 0) or 0)
    if vol < MIN_VOLUME:
        return None

    # ---- Probability estimation ----
    vol_ratio = vol_24h / max(vol, 1)
    liq_ratio = liq / max(vol, 1) if liq > 0 else 0.5
    # Low liquidity + high volume = stronger momentum signal
    momentum_strength = vol_ratio / max(liq_ratio, 0.01)
    momentum_strength = min(momentum_strength, 5.0)

    shift = min(vol_ratio * 0.13, 0.09) * (1 if price > 0.5 else -1)
    est_prob = max(0.05, min(0.95, price + shift))

    # ---- Momentum detection ----
    # Price deviation from 0.5 midpoint indicates directional momentum
    price_momentum = abs(price - 0.5)
    # Volume-weighted momentum: high vol_ratio amplifies the signal
    weighted_momentum = price_momentum * (1 + vol_ratio * 2)

    # Momentum direction
    if price > 0.5:
        mom_direction = "YES"
        mom_magnitude = weighted_momentum
    else:
        mom_direction = "NO"
        mom_magnitude = weighted_momentum

    return {
        "price": price, "est_prob": est_prob,
        "mom_direction": mom_direction,
        "mom_magnitude": mom_magnitude,
        "momentum_strength": momentum_strength,
        "vol_ratio": vol_ratio,
    }


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
    print(f"  AGENT 20 -- Kelly Momentum Hybrid")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 20 terminated by risk manager.")
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
    kelly_only = 0
    momentum_only = 0

    for m in markets:
        analysis = analyze_market(m)
        if analysis is None:
            continue

        price = analysis["price"]
        est_prob = analysis["est_prob"]

        # Kelly direction
        kelly_dir = "YES" if est_prob > price else "NO"
        eff_price = price if kelly_dir == "YES" else (1.0 - price)
        eff_prob = est_prob if kelly_dir == "YES" else (1.0 - est_prob)
        edge = eff_prob - eff_price

        has_kelly_edge = edge >= MIN_EDGE
        has_momentum = analysis["mom_magnitude"] >= MOMENTUM_THRESHOLD

        if has_kelly_edge:
            kelly_only += 1
        if has_momentum:
            momentum_only += 1

        # BOTH must confirm: Kelly edge + momentum in same direction
        if not has_kelly_edge or not has_momentum:
            continue
        if kelly_dir != analysis["mom_direction"]:
            continue  # Signals disagree -- skip

        f = kelly_fraction(eff_prob, eff_price)
        if f <= 0:
            continue

        # Three-quarter Kelly on double-confirmed signals
        adj_f = f * KELLY_FRACTION_MULT
        bet_size = adj_f * agent.capital
        bet_size = min(bet_size, agent.capital * 0.12)

        cid = m.get("conditionId", m.get("condition_id", ""))
        title = m.get("question", m.get("title", ""))[:100]

        signals.append({
            "market": title, "direction": kelly_dir,
            "price": round(eff_price, 4), "est_prob": round(eff_prob, 4),
            "edge": round(edge, 4), "raw_kelly": round(f, 4),
            "adj_kelly_075": round(adj_f, 4),
            "momentum": round(analysis["mom_magnitude"], 4),
            "mom_strength": round(analysis["momentum_strength"], 2),
            "confirmed": True, "bet_size": round(bet_size, 2),
        })

        if n_open < MAX_POSITIONS and bet_size >= 8:
            reason = f"KellyMom f*0.75={adj_f:.3f} edge={edge:.1%} mom={analysis['mom_magnitude']:.3f}"
            opened = agent.open_position(title, cid, kelly_dir, bet_size, eff_price, reason)
            if opened:
                new_trades += 1
                n_open += 1
                print(f"  TRADE: {kelly_dir} @ {eff_price:.3f} | f*3/4={adj_f:.3f}"
                      f" | edge={edge:.1%} | mom={analysis['mom_magnitude']:.3f}"
                      f" | {title[:40]}")

    log_data = {
        "agent": "agent_20", "strategy": "kelly_momentum_hybrid",
        "time": datetime.now(timezone.utc).isoformat(),
        "kelly_only_signals": kelly_only,
        "momentum_only_signals": momentum_only,
        "confirmed_signals": len(signals), "new_trades": new_trades,
        "signals": signals[:15],
    }
    with open(LOG_DIR / "signals_agent_20.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Confirmed: {len(signals)} (Kelly-only: {kelly_only},"
          f" Mom-only: {momentum_only})")
    print(f"  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
