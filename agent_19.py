#!/usr/bin/env python3
"""
AGENT 19 — Multi-Leg Kelly
============================
Applies Kelly across a portfolio of correlated bets. When multiple markets
in the same category have edge, distributes capital using a simplified
portfolio Kelly that accounts for overlap. Limits total category exposure
to prevent concentration risk.

Kelly math per market: f = (p*b - q) / b where b = (1/price) - 1.
Portfolio adjustment: each market's Kelly is scaled by 1/sqrt(n_category)
to account for correlation within the same category.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)
import math

agent = SwarmAgent("agent_19", "kelly")

MAX_POSITIONS = 10
MAX_PER_CATEGORY = 3       # Max bets in same category
MAX_CATEGORY_EXPOSURE = 0.25  # Max 25% capital per category
MIN_EDGE = 0.06
MIN_VOLUME = 15000
TAKE_PROFIT = 0.08
STOP_LOSS = 0.05


def extract_category(market):
    """Extract market category for correlation grouping."""
    tags = market.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, ValueError):
            tags = []
    if tags:
        return tags[0].get("label", tags[0]) if isinstance(tags[0], dict) else str(tags[0])
    title = (market.get("question", "") or market.get("title", "")).lower()
    for cat in ["crypto", "bitcoin", "eth", "election", "president", "nba", "nfl",
                "fed", "rate", "trump", "war", "ai", "tech", "weather"]:
        if cat in title:
            return cat
    return "general"


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


def estimate_prob(market):
    """Estimate probability from price + volume signal."""
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

    vol_ratio = vol_24h / max(vol, 1)
    shift = min(vol_ratio * 0.12, 0.08) * (1 if price > 0.5 else -1)
    est_prob = max(0.05, min(0.95, price + shift))
    return price, est_prob


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
    print(f"  AGENT 19 -- Multi-Leg Kelly")
    print(f"  Capital: ${agent.capital:,.2f} | P&L: ${agent.pnl:,.2f}")
    print(f"{'='*60}")

    if agent.is_killed():
        print("[KILLED] Agent 19 terminated by risk manager.")
        return

    closed = manage_positions()
    if closed:
        print(f"  Closed {closed} position(s)")

    markets = fetch_active_markets(400)
    if not markets:
        print("  No market data available")
        agent.save_state()
        return

    # Group candidates by category
    candidates_by_cat = defaultdict(list)
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

        cat = extract_category(m)
        candidates_by_cat[cat].append({
            "market": m, "direction": direction,
            "eff_price": eff_price, "eff_prob": eff_prob,
            "edge": edge, "kelly_f": f, "category": cat,
        })

    # Apply portfolio Kelly with correlation adjustment
    signals = []
    new_trades = 0
    n_open = len([p for p in agent.positions if p.get("status") == "open"])
    category_exposure = defaultdict(float)

    for cat, cands in candidates_by_cat.items():
        cands.sort(key=lambda x: x["edge"], reverse=True)
        n_in_cat = len(cands)
        # Correlation adjustment: scale down by 1/sqrt(n) for correlated bets
        corr_scale = 1.0 / math.sqrt(max(n_in_cat, 1))

        for c in cands[:MAX_PER_CATEGORY]:
            adjusted_f = c["kelly_f"] * corr_scale
            bet_size = adjusted_f * agent.capital
            bet_size = min(bet_size, agent.capital * 0.10)

            # Check category exposure limit
            if category_exposure[cat] + bet_size > agent.capital * MAX_CATEGORY_EXPOSURE:
                continue

            m = c["market"]
            cid = m.get("conditionId", m.get("condition_id", ""))
            title = m.get("question", m.get("title", ""))[:100]

            signals.append({
                "market": title, "category": cat, "direction": c["direction"],
                "price": round(c["eff_price"], 4), "est_prob": round(c["eff_prob"], 4),
                "edge": round(c["edge"], 4), "raw_kelly": round(c["kelly_f"], 4),
                "corr_scale": round(corr_scale, 3),
                "adj_kelly": round(adjusted_f, 4),
                "bet_size": round(bet_size, 2),
            })

            if n_open < MAX_POSITIONS and bet_size >= 8:
                reason = f"MultiKelly cat={cat} f={adjusted_f:.3f} edge={c['edge']:.1%}"
                opened = agent.open_position(
                    title, cid, c["direction"], bet_size, c["eff_price"], reason)
                if opened:
                    new_trades += 1
                    n_open += 1
                    category_exposure[cat] += bet_size
                    print(f"  TRADE: [{cat}] {c['direction']} @ {c['eff_price']:.3f}"
                          f" | adj_f={adjusted_f:.3f} | {title[:40]}")

    log_data = {
        "agent": "agent_19", "strategy": "multi_leg_kelly",
        "time": datetime.now(timezone.utc).isoformat(),
        "categories": len(candidates_by_cat),
        "signals_found": len(signals), "new_trades": new_trades,
        "signals": signals[:20],
    }
    with open(LOG_DIR / "signals_agent_19.json", "w") as f:
        json.dump(log_data, f, indent=2)

    agent.save_state()
    s = agent.get_summary()
    print(f"\n  Open: {s['open_positions']} | Trades: {s['total_trades']}"
          f" | WR: {s['win_rate']}% | Sharpe: {s['sharpe']}")
    print(f"  P&L: ${s['pnl']:,.2f} ({s['pnl_pct']:+.1f}%) | DD: {s['drawdown_pct']:.1f}%")


if __name__ == "__main__":
    run()
