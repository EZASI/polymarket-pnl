#!/usr/bin/env python3
"""
STAT ARB AGENT 02 — Cross-Category Spread
============================================
Exploits pricing inconsistencies between related markets in different categories.
Example: A "Will BTC hit $100k?" market at 0.70 while a "Will SEC approve crypto
ETF?" market sits at 0.30 — if the ETF approval would likely push BTC up, the
implied probabilities are inconsistent.

Builds a cross-category linkage map and trades the mispriced leg.
"""
import json
import re
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_02", "stat_arb")

MAX_POSITIONS = 8
PROFIT_EXIT = 0.08
LOSS_EXIT = -0.05

# Cross-category keyword linkage: if market A has key from group,
# and market B has a linked key, they are related
CATEGORY_LINKS = {
    "crypto_price": ["bitcoin", "btc", "ethereum", "eth", "crypto price", "100k", "150k"],
    "crypto_regulation": ["sec", "etf", "regulation", "crypto ban", "stablecoin", "cbdc"],
    "election_outcome": ["president", "win election", "nominee", "electoral"],
    "election_policy": ["tariff", "tax", "immigration", "policy", "executive order"],
    "ai_company": ["openai", "google ai", "nvidia", "agi", "chatgpt", "anthropic"],
    "ai_regulation": ["ai regulation", "ai ban", "ai safety", "ai executive order"],
    "fed_rates": ["fed", "interest rate", "rate cut", "rate hike", "fomc"],
    "economy": ["recession", "gdp", "inflation", "unemployment", "cpi"],
}

# Expected directional relationships between categories
# (cat_a, cat_b) -> correlation: +1 means same direction, -1 means inverse
LINK_CORRELATIONS = {
    ("crypto_regulation", "crypto_price"): 0.6,
    ("election_outcome", "election_policy"): 0.5,
    ("ai_company", "ai_regulation"): -0.3,
    ("fed_rates", "economy"): -0.4,
    ("fed_rates", "crypto_price"): 0.3,
    ("economy", "crypto_price"): 0.4,
}


def categorize_market(question):
    """Assign a market to a category based on keyword matching."""
    q = question.lower()
    scores = {}
    for cat, keywords in CATEGORY_LINKS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits > 0:
            scores[cat] = hits
    if scores:
        return max(scores, key=scores.get)
    return None


def get_correlation(cat_a, cat_b):
    """Get expected correlation between two categories."""
    key = (cat_a, cat_b)
    if key in LINK_CORRELATIONS:
        return LINK_CORRELATIONS[key]
    rev = (cat_b, cat_a)
    if rev in LINK_CORRELATIONS:
        return LINK_CORRELATIONS[rev]
    return None


def run():
    print(f"[{agent.agent_id}] Cross-category spread scan starting...")
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

    # --- Categorize all markets ---
    markets = fetch_active_markets(500)
    categorized = {}

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
        if vol < 10000:
            continue

        question = m.get("question", "")
        cat = categorize_market(question)
        if not cat:
            continue

        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append({
            "question": question,
            "condition_id": m.get("conditionId", ""),
            "price": price,
            "volume": vol,
            "category": cat,
        })

    # --- Find cross-category inconsistencies ---
    signals = []
    cats = list(categorized.keys())

    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            corr = get_correlation(cats[i], cats[j])
            if corr is None:
                continue

            for mkt_a in categorized[cats[i]]:
                for mkt_b in categorized[cats[j]]:
                    p_a = mkt_a["price"]
                    p_b = mkt_b["price"]

                    # Convert prices to implied probability displacement from 0.5
                    disp_a = p_a - 0.5
                    disp_b = p_b - 0.5

                    # If positively correlated, displacements should match direction
                    # If negatively correlated, they should oppose
                    expected_b_disp = disp_a * corr
                    inconsistency = disp_b - expected_b_disp

                    if abs(inconsistency) < 0.10:
                        continue

                    # Trade the market that is more mispriced
                    # If inconsistency > 0, market B is too high relative to A
                    if abs(inconsistency) > 0.10:
                        if inconsistency > 0:
                            target = mkt_b
                            direction = "No"
                            edge = abs(inconsistency) * 0.4
                        else:
                            target = mkt_b
                            direction = "Yes"
                            edge = abs(inconsistency) * 0.4

                        signals.append({
                            "market_a": mkt_a["question"][:50],
                            "cat_a": cats[i],
                            "price_a": round(p_a, 3),
                            "market_b": mkt_b["question"][:50],
                            "cat_b": cats[j],
                            "price_b": round(p_b, 3),
                            "correlation": corr,
                            "inconsistency": round(inconsistency, 4),
                            "target_cid": target["condition_id"],
                            "target_price": target["price"],
                            "direction": direction,
                            "edge": round(edge, 4),
                        })

    signals.sort(key=lambda x: x["edge"], reverse=True)

    # --- Open positions on top signals ---
    for sig in signals[:10]:
        bet_size = min(sig["edge"] * agent.capital * 0.35, agent.capital * 0.07)
        if bet_size >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                sig["market_b"], sig["target_cid"],
                sig["direction"], bet_size, sig["target_price"],
                reason=f"XCat: {sig['cat_a']}<>{sig['cat_b']} incon={sig['inconsistency']:+.2f}"
            )

    # --- Write signals ---
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy": "cross_category_spread",
            "categories_found": {k: len(v) for k, v in categorized.items()},
            "signals": signals[:20],
            "summary": agent.get_summary(),
        }, f, indent=2)

    s = agent.get_summary()
    print(f"[{agent.agent_id}] {len(categorized)} categories, {len(signals)} spread signals, "
          f"{s['open_positions']} open. PnL: ${s['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
