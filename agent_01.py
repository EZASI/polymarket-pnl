#!/usr/bin/env python3
"""
STAT ARB AGENT 01 — Pairs Trading
===================================
Finds correlated market pairs within the same event/race (e.g., two candidates
in a presidential primary). When one deviates from the expected price ratio,
trades the mean reversion back toward the historical relationship.

Uses volume-weighted correlation and liquidity patterns to identify genuine pairs.
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_01", "stat_arb")

# Pairs config
MIN_VOLUME = 15000
MAX_POSITIONS = 8
PROFIT_EXIT = 0.07
LOSS_EXIT = -0.05
RATIO_DEVIATION_THRESHOLD = 0.12  # 12% deviation from expected ratio


def group_markets_by_event(markets):
    """Group markets that share the same event (slug prefix or groupItemTitle)."""
    groups = {}
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
        liq = float(m.get("liquidity", 0) or 0)
        if vol < MIN_VOLUME or liq < 500:
            continue

        group_key = m.get("groupItemTitle", "") or ""
        slug = m.get("slug", "")
        # Use the event slug prefix as the grouping key
        if not group_key and slug:
            parts = slug.split("-")
            group_key = "-".join(parts[:3]) if len(parts) >= 3 else slug

        if not group_key:
            continue

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append({
            "question": m.get("question", ""),
            "condition_id": m.get("conditionId", ""),
            "price": price,
            "volume": vol,
            "liquidity": liq,
            "vol_liq": vol / max(liq, 1),
        })
    return groups


def find_pair_opportunities(groups):
    """Find pairs where the price ratio has deviated from the expected norm."""
    opportunities = []
    for group_key, members in groups.items():
        if len(members) < 2:
            continue
        # Sort by volume descending to get the two most liquid members
        members.sort(key=lambda x: x["volume"], reverse=True)

        for i in range(len(members)):
            for j in range(i + 1, min(len(members), 4)):
                a, b = members[i], members[j]
                # Expected ratio: prices should sum near 1.0 for binary complements
                # or maintain a stable ratio for multi-outcome events
                price_sum = a["price"] + b["price"]

                # If these are complementary outcomes (sum near 1.0),
                # deviation from 1.0 is an arbitrage signal
                if 0.80 <= price_sum <= 1.20:
                    deviation = abs(price_sum - 1.0)
                    if deviation > RATIO_DEVIATION_THRESHOLD:
                        # Prices sum > 1: both overpriced, short the more expensive
                        # Prices sum < 1: both underpriced, buy the cheaper one
                        if price_sum > 1.0:
                            target = a if a["price"] > b["price"] else b
                            direction = "No"
                            edge = deviation / 2
                        else:
                            target = a if a["price"] < b["price"] else b
                            direction = "Yes"
                            edge = deviation / 2

                        opportunities.append({
                            "group": group_key,
                            "pair_a": a["question"][:60],
                            "pair_b": b["question"][:60],
                            "price_a": a["price"],
                            "price_b": b["price"],
                            "price_sum": round(price_sum, 4),
                            "deviation": round(deviation, 4),
                            "target_market": target["question"][:60],
                            "target_cid": target["condition_id"],
                            "target_price": target["price"],
                            "direction": direction,
                            "edge": round(edge, 4),
                        })
                    continue

                # For non-complementary pairs, check volume-weighted ratio stability
                if a["price"] > 0.05 and b["price"] > 0.05:
                    ratio = a["price"] / b["price"]
                    # Expected ratio based on volume weighting
                    vol_weight = a["volume"] / max(a["volume"] + b["volume"], 1)
                    expected_ratio = vol_weight / max(1 - vol_weight, 0.01)
                    ratio_dev = abs(ratio - expected_ratio) / max(expected_ratio, 0.01)

                    if ratio_dev > 0.20:
                        # Trade the one that deviated more
                        if ratio > expected_ratio:
                            target, direction = a, "No"
                        else:
                            target, direction = b, "No"
                        edge = ratio_dev * 0.3

                        opportunities.append({
                            "group": group_key,
                            "pair_a": a["question"][:60],
                            "pair_b": b["question"][:60],
                            "price_a": a["price"],
                            "price_b": b["price"],
                            "ratio": round(ratio, 4),
                            "expected_ratio": round(expected_ratio, 4),
                            "ratio_deviation": round(ratio_dev, 4),
                            "target_market": target["question"][:60],
                            "target_cid": target["condition_id"],
                            "target_price": target["price"],
                            "direction": direction,
                            "edge": round(edge, 4),
                        })

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


def run():
    print(f"[{agent.agent_id}] Pairs trading scan starting...")
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

    # --- Scan for new pairs ---
    markets = fetch_active_markets(500)
    groups = group_markets_by_event(markets)
    opportunities = find_pair_opportunities(groups)
    signals = []

    for opp in opportunities[:15]:
        signals.append(opp)
        bet_size = min(opp["edge"] * agent.capital * 0.4, agent.capital * 0.08)
        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                opp["target_market"], opp["target_cid"],
                opp["direction"], bet_size, opp["target_price"],
                reason=f"Pairs: dev={opp.get('deviation', opp.get('ratio_deviation', 0)):.1%} edge={opp['edge']:.1%}"
            )

    # --- Write signals ---
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy": "pairs_trading",
            "groups_found": len(groups),
            "pairs_scanned": sum(len(v) for v in groups.values()),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    s = agent.get_summary()
    print(f"[{agent.agent_id}] {len(groups)} groups, {len(signals)} pair signals, "
          f"{s['open_positions']} open. PnL: ${s['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
