#!/usr/bin/env python3
"""
NEWS HAWK AGENT 5 — Entertainment & Awards Scanner
====================================================
Specializes in Oscar/Grammy/Emmy markets. Uses market price consensus
and copy signal divergence to find mispriced entertainment bets.
"""
import json
from datetime import datetime, timezone
from swarm_base import SwarmAgent, fetch_active_markets, LOG_DIR

agent = SwarmAgent("news_entertainment", "news")

ENT_KEYWORDS = ["oscar", "academy award", "grammy", "emmy", "best picture", "best director",
    "best actor", "best actress", "best supporting", "golden globe", "bafta", "netflix",
    "box office", "album", "song", "movie", "film"]

def run():
    print(f"[{agent.agent_id}] Scanning entertainment markets...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    markets = fetch_active_markets(500)
    entertainment = []
    for m in markets:
        q = (m.get("question","") + " " + m.get("slug","")).lower()
        if any(kw in q for kw in ENT_KEYWORDS):
            entertainment.append(m)

    # Load copy signals for correlation
    copy_signals = []
    try:
        data = json.load(open(LOG_DIR / "copy_signals.json"))
        copy_signals = data.get("signals", [])
    except: pass

    signals = []
    for m in entertainment:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.95 or poly_price <= 0.05: continue

        # Check for copy signal match
        cid = m.get("conditionId", "")
        matched_signal = None
        for sig in copy_signals:
            if sig.get("conditionId","") == cid:
                matched_signal = sig
                break

        if matched_signal and matched_signal.get("avg_wr", 0) > 55:
            wr = matched_signal["avg_wr"]
            our_prob = min(wr / 100.0, 0.95)
            edge = our_prob - poly_price

            if abs(edge) > 0.05:
                bet_size = min(abs(edge) * agent.capital * 0.4, agent.capital * 0.10)
                direction = matched_signal.get("consensus", "No")
                signals.append({"market": m.get("question","")[:80], "condition_id": cid,
                    "direction": direction, "edge": round(edge, 3), "wr": wr,
                    "poly_price": round(poly_price, 3), "bet_size": round(bet_size, 2)})

                if bet_size >= 10:
                    agent.open_position(m.get("question",""), cid,
                        direction, bet_size, poly_price, reason=f"Entertainment: WR={wr:.0f}% edge={edge:+.1%}")

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} entertainment signals. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
