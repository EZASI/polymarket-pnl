#!/usr/bin/env python3
"""
QUANT AGENT 3 — Copy Signal Convergence
=========================================
Follows markets where 2+ top traders agree. Weighted by win rate and position size.
"""
import json, time
from datetime import datetime, timezone
from swarm_base import SwarmAgent, fetch_active_markets, LOG_DIR

agent = SwarmAgent("quant_copy", "quant")

def load_copy_signals():
    path = LOG_DIR / "copy_signals.json"
    if not path.exists(): return []
    try:
        data = json.load(open(path))
        return data.get("signals", [])
    except: return []

def run():
    print(f"[{agent.agent_id}] Running copy signal convergence strategy...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    signals = load_copy_signals()
    markets = fetch_active_markets(500)
    market_lookup = {m.get("conditionId",""): m for m in markets if m.get("conditionId")}

    trades = []
    for sig in signals:
        if sig.get("n_traders", 0) < 2: continue
        if sig.get("avg_wr", 0) < 55: continue

        cid = sig.get("conditionId", "")
        m = market_lookup.get(cid)
        if not m: continue

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.95 or poly_price <= 0.05: continue

        wr = sig["avg_wr"]
        n = sig["n_traders"]
        our_prob = min(wr / 100.0 + (n - 1) * 0.03, 0.95)
        edge = our_prob - poly_price

        if abs(edge) > 0.03:
            strength_mult = {"STRONG": 1.0, "MODERATE": 0.7, "WEAK": 0.4}.get(sig.get("strength","WEAK"), 0.4)
            bet_size = min(abs(edge) * agent.capital * strength_mult, agent.capital * 0.10)
            trades.append({"market": sig.get("title","")[:80], "condition_id": cid,
                           "consensus": sig.get("consensus",""), "edge": round(edge, 3),
                           "n_traders": n, "avg_wr": wr, "strength": sig.get("strength",""),
                           "bet_size": round(bet_size, 2)})

            if bet_size >= 10:
                agent.open_position(sig.get("title",""), cid, sig.get("consensus","Yes"),
                    bet_size, poly_price, reason=f"Copy: {n} traders, {wr:.0f}% WR, edge={edge:+.1%}")

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": trades, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(trades)} copy convergence trades. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
