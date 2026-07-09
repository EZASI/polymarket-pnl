#!/usr/bin/env python3
"""
STAT ARB AGENT 04 — Market Maker Spread Capture
==================================================
Acts as a virtual market maker. Identifies markets with wide bid-ask spreads
(low liquidity relative to volume). Takes positions at favorable prices when
spreads are wide, exits when spreads narrow and price reverts to mid.

Core idea: in illiquid markets, the bid-ask spread is wide and the "true" price
sits near the midpoint. Buy near the bid, sell near the ask, capture the spread.
"""
import json
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, get_copy_signals, LOG_DIR, GAMMA_HOST, CLOB_HOST,
)

agent = SwarmAgent("agent_04", "stat_arb")

MAX_POSITIONS = 12   # More positions, smaller sizes — like a real MM
PROFIT_EXIT = 0.05   # Tight profit target (spread capture)
LOSS_EXIT = -0.03    # Very tight stop — MM strategy should not hold losers
MIN_SPREAD_PCT = 0.06  # 6% minimum spread to enter


def get_orderbook(token_id):
    """Fetch orderbook and extract spread metrics."""
    try:
        resp = rate_limited_get(f"{CLOB_HOST}/book", params={"token_id": token_id})
        if not resp or resp.status_code != 200:
            return None
        book = resp.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if not bids or not asks:
            return None
        best_bid = max(float(b.get("price", 0)) for b in bids)
        best_ask = min(float(a.get("price", 1)) for a in asks)
        if best_ask <= best_bid:
            return None
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / max(total_depth, 1)
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread / max(mid, 0.01),
            "mid": mid,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance,
            "total_depth": total_depth,
        }
    except:
        return None


def estimate_spread_from_liquidity(price, volume, liquidity):
    """Estimate effective spread when orderbook data is unavailable.
    Low liquidity relative to volume implies wide spreads."""
    if liquidity <= 0 or volume <= 0:
        return None
    liq_vol_ratio = liquidity / volume
    # Empirical: spread ~ 1 / sqrt(liquidity_ratio)
    estimated_spread = min(0.15 / max(liq_vol_ratio ** 0.5, 0.01), 0.30)
    mid = price
    return {
        "best_bid": max(mid - estimated_spread / 2, 0.01),
        "best_ask": min(mid + estimated_spread / 2, 0.99),
        "spread": estimated_spread,
        "spread_pct": estimated_spread / max(mid, 0.01),
        "mid": mid,
        "bid_depth": liquidity * 0.5,
        "ask_depth": liquidity * 0.5,
        "imbalance": 0,
        "total_depth": liquidity,
        "estimated": True,
    }


def run():
    print(f"[{agent.agent_id}] Market maker spread capture scan starting...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED by risk manager. Stopping.")
        return

    # --- Position management: exit when spread narrows or P&L hits threshold ---
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

    # --- Scan for wide-spread markets ---
    markets = fetch_active_markets(400)
    spread_opportunities = []

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
        if price >= 0.92 or price <= 0.08:
            continue

        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 8000:
            continue

        # Try orderbook first for token_id markets
        token_ids = m.get("clobTokenIds", "[]")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except:
                token_ids = []

        ob = None
        if token_ids:
            ob = get_orderbook(token_ids[0])

        # Fall back to liquidity-based estimation
        if ob is None and liq > 0:
            ob = estimate_spread_from_liquidity(price, vol, liq)

        if ob is None:
            continue
        if ob["spread_pct"] < MIN_SPREAD_PCT:
            continue

        # MM strategy: buy at (or near) the bid, expecting price to move to mid
        # Direction based on order book imbalance
        if ob["imbalance"] > 0.1:
            # More bids than asks — buy pressure, price likely to rise
            direction = "Yes"
            entry_price = ob["best_bid"] + ob["spread"] * 0.2  # Slightly above bid
        elif ob["imbalance"] < -0.1:
            # More asks — sell pressure, lean No
            direction = "No"
            entry_price = ob["best_bid"] + ob["spread"] * 0.2
        else:
            # Balanced book — buy at bid, target mid
            direction = "Yes" if price < ob["mid"] else "No"
            entry_price = price

        # Edge is half the spread (we enter near bid/ask, exit near mid)
        edge = ob["spread"] / 2

        spread_opportunities.append({
            "question": m.get("question", ""),
            "condition_id": m.get("conditionId", ""),
            "price": price,
            "spread": round(ob["spread"], 4),
            "spread_pct": round(ob["spread_pct"], 4),
            "mid": round(ob["mid"], 4),
            "imbalance": round(ob["imbalance"], 3),
            "depth": round(ob["total_depth"], 0),
            "direction": direction,
            "entry_price": round(entry_price, 4),
            "edge": round(edge, 4),
            "volume": vol,
            "estimated": ob.get("estimated", False),
        })

    # Sort by spread width — widest first (most profit potential)
    spread_opportunities.sort(key=lambda x: x["spread_pct"], reverse=True)
    signals = []

    for opp in spread_opportunities[:20]:
        signals.append({
            "market": opp["question"][:80],
            "condition_id": opp["condition_id"],
            "price": opp["price"],
            "spread_pct": opp["spread_pct"],
            "mid": opp["mid"],
            "imbalance": opp["imbalance"],
            "direction": opp["direction"],
            "edge": opp["edge"],
            "estimated": opp["estimated"],
        })

        # Small positions — MM style, many small bets
        bet_size = min(opp["edge"] * agent.capital * 0.25, agent.capital * 0.04)
        if bet_size >= 8 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(
                opp["question"], opp["condition_id"],
                opp["direction"], bet_size, opp["entry_price"],
                reason=f"MM: spread={opp['spread_pct']:.1%} imb={opp['imbalance']:+.2f}"
            )

    # --- Write signals ---
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({
            "time": datetime.now(timezone.utc).isoformat(),
            "strategy": "market_maker_spread",
            "markets_scanned": len(markets),
            "wide_spread_found": len(spread_opportunities),
            "signals": signals,
            "summary": agent.get_summary(),
        }, f, indent=2)

    s = agent.get_summary()
    print(f"[{agent.agent_id}] {len(spread_opportunities)} wide-spread markets, {len(signals)} signals, "
          f"{s['open_positions']} open. PnL: ${s['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
