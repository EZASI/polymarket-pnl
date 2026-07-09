#!/usr/bin/env python3
"""
SCALPER AGENT 8 — Spread Scalper
==================================
Exploits wide bid-ask spreads in low-liquidity markets.
Buys below fair value, sells above. Mean-reversion approach.
"""
import json
from datetime import datetime, timezone
from swarm_base import SwarmAgent, rate_limited_get, fetch_active_markets, fetch_market_price, LOG_DIR, CLOB_HOST

agent = SwarmAgent("scalp_spread", "scalp")

def get_orderbook(token_id):
    """Get orderbook spread for a market."""
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
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        return {
            "best_bid": best_bid, "best_ask": best_ask,
            "spread": spread, "mid": mid,
            "spread_pct": spread / max(mid, 0.01),
            "bid_depth": bid_depth, "ask_depth": ask_depth,
            "imbalance": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1)
        }
    except:
        return None

def run():
    print(f"[{agent.agent_id}] Scanning for spread opportunities...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    # Exit management
    for pos in list(agent.positions):
        if pos.get("status") != "open": continue
        current = fetch_market_price(pos["condition_id"])
        if current is None: continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        # Tight exits for spread scalping
        if pnl_pct >= 0.06 or pnl_pct <= -0.04:
            realized = agent.close_position(pos["condition_id"], current)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    markets = fetch_active_markets(300)
    signals = []

    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.90 or poly_price <= 0.10: continue

        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 10000: continue

        # Look for wide spreads (inefficient pricing)
        # We estimate spread from liquidity vs volume ratio
        # Low liquidity + decent volume = wide spread opportunity
        if liq > 0:
            liq_vol_ratio = liq / max(vol, 1)
            # Wide spread proxy: low liq relative to volume
            if liq_vol_ratio < 0.15 and liq > 1000:
                # Mean reversion: price likely to revert toward 0.5 in mid-range
                reversion_edge = (0.5 - poly_price) * 0.3
                if abs(reversion_edge) > 0.02:
                    direction = "Yes" if reversion_edge > 0 else "No"
                    bet_size = min(abs(reversion_edge) * agent.capital * 0.3, agent.capital * 0.04)

                    signals.append({
                        "market": m.get("question", "")[:80],
                        "condition_id": m.get("conditionId", ""),
                        "direction": direction,
                        "poly_price": round(poly_price, 3),
                        "liq_vol_ratio": round(liq_vol_ratio, 4),
                        "reversion_edge": round(reversion_edge, 3),
                        "volume": vol,
                        "liquidity": liq,
                    })

                    if bet_size >= 10 and len(agent.positions) < 10:
                        agent.open_position(
                            m.get("question", ""), m.get("conditionId", ""),
                            direction, bet_size, poly_price,
                            reason=f"Spread: liq/vol={liq_vol_ratio:.3f} edge={reversion_edge:+.1%}"
                        )

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals,
                    "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} spread signals, {len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
