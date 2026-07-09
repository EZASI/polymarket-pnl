#!/usr/bin/env python3
"""
AGENT 21 — Entropy Trader
==========================
Shannon entropy of market price/volume/liquidity distributions.
LOW entropy + mid-range price = "false certainty" -> contrarian trade.
HIGH entropy + high volume = breakout imminent -> follow volume direction.
H = -sum(p * log2(p)), no numpy.
"""
import json, math
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_21", "chaos")
MAX_POSITIONS = 8
TAKE_PROFIT, STOP_LOSS = 0.09, -0.06
FALSE_CERT_H = 0.35       # Entropy below = too confident
BREAKOUT_H = 0.85         # Entropy above = genuine uncertainty
BREAKOUT_VOL_X = 2.5      # vol24 must exceed avg by this multiple


def shannon_h(probs):
    """H = -sum(p * log2(p)). Max=1.0 for binary."""
    return -sum(p * math.log2(p) for p in probs if 0 < p < 1)


def price_entropy(price):
    return shannon_h([price, 1.0 - price])


def vol_entropy(vol, vol24, liq):
    """Normalized entropy from volume/vol24/liquidity distribution."""
    total = vol + vol24 + liq
    if total <= 0:
        return 1.0
    probs = [x / total for x in (vol, vol24, liq) if x > 0]
    if len(probs) < 2:
        return 1.0
    return shannon_h(probs) / math.log2(len(probs))


def run():
    print(f"[{agent.agent_id}] Entropy Trader -- scanning information disorder...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping.")
        return

    # -- Position management --
    for pos in list(agent.positions):
        if pos.get("status") != "open":
            continue
        cur = fetch_market_price(pos["condition_id"])
        if cur is None:
            continue
        pnl_pct = (cur - pos["entry_price"]) / max(pos["entry_price"], 0.01)
        if pnl_pct >= TAKE_PROFIT or pnl_pct <= STOP_LOSS:
            r = agent.close_position(pos["condition_id"], cur)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${r:+.2f}")

    # -- Score all markets --
    markets = fetch_active_markets(500)
    vol24_vals, scored = [], []
    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: continue
        if not prices:
            continue
        price = float(prices[0])
        vol = float(m.get("volume", 0) or 0)
        v24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 5000 or liq < 200:
            continue
        vol24_vals.append(v24)
        pe, ve = price_entropy(price), vol_entropy(vol, v24, liq)
        scored.append({"m": m, "price": price, "v24": v24,
                       "pe": pe, "ve": ve, "comp": 0.6 * pe + 0.4 * ve})

    avg_v24 = sum(vol24_vals) / max(len(vol24_vals), 1)
    signals = []

    for s in scored:
        p, m = s["price"], s["m"]
        if p > 0.90 or p < 0.10:
            continue
        direction, reason, strength = None, "", 0.0

        # STRATEGY 1: False Certainty -- low entropy, mid price -> contrarian
        if s["comp"] < FALSE_CERT_H and 0.15 < p < 0.85:
            direction = "No" if p > 0.5 else "Yes"
            strength = (FALSE_CERT_H - s["comp"]) * 3.0
            reason = f"FalseCert H={s['comp']:.2f} contrarian"
        # STRATEGY 2: Breakout -- high entropy + volume spike -> follow flow
        elif s["pe"] > BREAKOUT_H and s["v24"] > avg_v24 * BREAKOUT_VOL_X:
            if p > 0.52: direction = "Yes"
            elif p < 0.48: direction = "No"
            else: continue
            surge = s["v24"] / max(avg_v24, 1)
            strength = min(surge * 0.4, 2.5)
            reason = f"Breakout H={s['pe']:.2f} surge={surge:.1f}x"

        if direction is None:
            continue
        bet = min(strength * agent.capital * 0.015, agent.capital * 0.06)
        signals.append({
            "market": m.get("question", "")[:80],
            "condition_id": m.get("conditionId", ""),
            "direction": direction, "price": round(p, 3),
            "p_entropy": round(s["pe"], 3), "v_entropy": round(s["ve"], 3),
            "composite": round(s["comp"], 3), "strength": round(strength, 3),
            "strategy": reason,
        })
        if bet >= 10 and len(agent.positions) < MAX_POSITIONS:
            agent.open_position(m.get("question", ""), m.get("conditionId", ""),
                                direction, bet, p, reason=reason)

    # -- Save signals --
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(),
                    "strategy": "entropy_trader", "markets_scored": len(scored),
                    "avg_vol24": round(avg_v24, 2), "signals": signals[:30],
                    "summary": agent.get_summary()}, f, indent=2)

    sm = agent.get_summary()
    print(f"[{agent.agent_id}] {len(signals)} entropy signals from {len(scored)} mkts. "
          f"{sm['open_positions']} open. PnL: ${sm['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
