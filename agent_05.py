#!/usr/bin/env python3
"""
STAT ARB AGENT 05 — Relative Value
=====================================
Ranks all markets by a composite "value score" combining:
  - Volume/liquidity ratio (activity vs depth)
  - Price displacement from 0.50 (conviction strength)
  - Freshness (recent volume as % of total)
  - Category peer comparison (is this market cheap vs siblings?)

Takes positions in the most undervalued markets relative to their peer group
within the same category. Sells the most overvalued.
"""
import json
import math
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_05", "stat_arb")

MAX_POSITIONS = 10
PROFIT_EXIT = 0.08
LOSS_EXIT = -0.05

# Category keywords for peer grouping
PEER_KEYWORDS = {
    "politics": ["president", "election", "democrat", "republican", "congress", "senate",
                 "governor", "vote", "nominee", "trump", "biden"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "xrp"],
    "sports": ["nba", "nfl", "mlb", "nhl", "champion", "win series", "mvp", "super bowl"],
    "entertainment": ["oscar", "grammy", "emmy", "movie", "album", "box office", "netflix"],
    "economics": ["fed", "inflation", "gdp", "recession", "rate cut", "unemployment",
                   "interest rate", "cpi", "jobs"],
    "tech": ["ai", "openai", "google", "apple", "nvidia", "tesla", "ipo", "market cap"],
    "world": ["war", "ceasefire", "nato", "china", "russia", "ukraine", "israel", "iran"],
}


def assign_peer_group(question):
    """Assign market to a peer group based on keyword matching."""
    q = question.lower()
    best_group = "other"
    best_score = 0
    for group, keywords in PEER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score = score
            best_group = group
    return best_group


def calc_value_score(price, volume, vol24, liquidity):
    """Compute composite value score for a single market.
    Higher score = more "undervalued" / better opportunity."""

    # Component 1: Volume/Liquidity ratio — high activity relative to depth
    if liquidity > 0:
        vol_liq = volume / liquidity
    else:
        vol_liq = 0
    vol_liq_score = math.log1p(vol_liq) / 3.0  # Normalize to ~0-1

    # Component 2: Price displacement from 0.50 — stronger conviction
    displacement = abs(price - 0.5)
    disp_score = displacement * 2  # 0-1 range

    # Component 3: Freshness — recent activity as fraction of total
    if volume > 0 and vol24 > 0:
        freshness = vol24 / volume
    else:
        freshness = 0
    fresh_score = min(freshness * 10, 1.0)  # Normalize

    # Component 4: Liquidity depth score — prefer some depth (not too thin)
    depth_score = min(math.log1p(liquidity) / 12.0, 1.0)

    # Weighted composite
    value = (vol_liq_score * 0.30
             + disp_score * 0.25
             + fresh_score * 0.30
             + depth_score * 0.15)
    return value


def run():
    print(f"[{agent.agent_id}] Relative value scan starting...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED by risk manager. Stopping.")
        return

    # --- Position management ---
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        current = fetch_market_price(pos["condition_id"])
        if current is None:
            continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        if pnl_pct >= PROFIT_EXIT or pnl_pct <= LOSS_EXIT:
            realized = agent.close_position(pos["condition_id"], current)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    # --- Score and group all markets ---
    markets = fetch_active_markets(500)
    scored = []

    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except:
                continue
        if not prices:
            continue
        price = float(prices[0])
        if price >= 0.95 or price <= 0.05:
            continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 5000:
            continue

        question = m.get("question", "")
        peer_group = assign_peer_group(question)
        value_score = calc_value_score(price, vol, vol24, liq)

        scored.append({
            "question": question,
            "condition_id": m.get("conditionId", ""),
            "price": price,
            "volume": vol,
            "vol24": vol24,
            "liquidity": liq,
            "peer_group": peer_group,
            "value_score": value_score,
        })

    # --- Compute peer-relative value (z-score within group) ---
    peer_groups = {}
    for s in scored:
        pg = s["peer_group"]
        if pg not in peer_groups:
            peer_groups[pg] = []
        peer_groups[pg].append(s)

    for pg, members in peer_groups.items():
        if len(members) < 2:
            for m in members:
                m["relative_value"] = 0
            continue
        scores = [m["value_score"] for m in members]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = math.sqrt(variance) if variance > 0 else 0.01
        for m in members:
            m["relative_value"] = (m["value_score"] - mean_score) / std_score

    # --- Sort by relative value: most undervalued first ---
    scored.sort(key=lambda x: x["relative_value"], reverse=True)
    signals = []

    for mkt in scored[:25]:
        rv = mkt["relative_value"]
        if abs(rv) < 0.5:
            continue

        # Undervalued (high relative value): the market deserves a higher price
        # Trade in the direction of the current price trend
        if rv > 0:
            direction = "Yes" if mkt["price"] > 0.45 else "No"
            edge = min(rv * 0.03, 0.12)
        else:
            # Overvalued: fade the current price
            direction = "No" if mkt["price"] > 0.55 else "Yes"
            edge = min(abs(rv) * 0.03, 0.12)

        sig = {
            "market": mkt["question"][:80],
            "condition_id": mkt["condition_id"],
            "price": round(mkt["price"], 3),
            "peer_group": mkt["peer_group"],
            "value_score": round(mkt["value_score"], 4),
            "relative_value": round(rv, 3),
            "direction": direction,
            "edge": round(edge, 4),
            "vol24": mkt["vol24"],
        }
        signals.append(sig)

        bet_size = min(edge * agent.capital * 0.35, agent.capital * 0.06)
        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                mkt["question"], mkt["condition_id"],
                direction, bet_size, mkt["price"],
                reason=f"RelVal: rv={rv:+.2f} peer={mkt['peer_group']} vs={mkt['value_score']:.2f}"
            )

    # --- Write signals ---
    peer_summary = {pg: len(members) for pg, members in peer_groups.items()}
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy": "relative_value",
            "markets_scored": len(scored),
            "peer_groups": peer_summary,
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    s = agent.get_summary()
    print(f"[{agent.agent_id}] {len(scored)} scored across {len(peer_groups)} groups, "
          f"{len(signals)} signals, {s['open_positions']} open. PnL: ${s['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
