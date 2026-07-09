#!/usr/bin/env python3
"""
SCALPER AGENT 9 — Volume Spike Scalper
========================================
Detects abnormal volume spikes as leading indicators.
Trades the volume-implied direction before price catches up.
"""
import json
from datetime import datetime, timezone
from swarm_base import SwarmAgent, fetch_active_markets, fetch_market_price, LOG_DIR

agent = SwarmAgent("scalp_volume", "scalp")

def run():
    print(f"[{agent.agent_id}] Scanning for volume spike entries...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    # Exit management
    for pos in list(agent.positions):
        if pos.get("status") != "open": continue
        current = fetch_market_price(pos["condition_id"])
        if current is None: continue
        entry = pos["entry_price"]
        pnl_pct = (current - entry) / max(entry, 0.01)
        # Volume scalp: take quick profits or cut losses
        if pnl_pct >= 0.07 or pnl_pct <= -0.04:
            realized = agent.close_position(pos["condition_id"], current)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${realized:+.2f}")

    markets = fetch_active_markets(500)

    # Build volume profile for all markets
    volume_data = []
    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.93 or poly_price <= 0.07: continue

        vol = float(m.get("volume", 0) or 0)
        vol24 = float(m.get("volume24hr", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)

        if vol < 5000 or liq < 1000: continue

        # Volume spike detection
        # Normal daily volume estimate (total / days, assume ~30 days avg)
        avg_daily_vol = vol / 30 if vol > 0 else 1
        vol_spike_ratio = vol24 / max(avg_daily_vol, 1)

        volume_data.append({
            "market": m,
            "poly_price": poly_price,
            "vol": vol,
            "vol24": vol24,
            "liq": liq,
            "avg_daily_vol": avg_daily_vol,
            "vol_spike_ratio": vol_spike_ratio,
        })

    # Sort by spike ratio descending
    volume_data.sort(key=lambda x: x["vol_spike_ratio"], reverse=True)

    signals = []
    for vd in volume_data[:30]:  # Top 30 by volume spike
        if vd["vol_spike_ratio"] < 1.5: continue  # Need 1.5x normal volume

        m = vd["market"]
        poly_price = vd["poly_price"]
        spike = vd["vol_spike_ratio"]

        # Volume spike + directional price = strong signal
        # If price is moving away from 0.5 with high volume, follow the trend
        displacement = abs(poly_price - 0.5)
        if displacement > 0.1:  # Price has moved meaningfully
            direction = "Yes" if poly_price > 0.55 else "No"
            # Higher spike + more displacement = stronger signal
            signal_strength = min(spike * displacement, 2.0)
            bet_size = min(signal_strength * agent.capital * 0.02, agent.capital * 0.05)

            signals.append({
                "market": m.get("question", "")[:80],
                "condition_id": m.get("conditionId", ""),
                "direction": direction,
                "poly_price": round(poly_price, 3),
                "vol_spike_ratio": round(spike, 2),
                "displacement": round(displacement, 3),
                "signal_strength": round(signal_strength, 3),
                "vol24": vd["vol24"],
            })

            if bet_size >= 10 and len(agent.positions) < 8:
                agent.open_position(
                    m.get("question", ""), m.get("conditionId", ""),
                    direction, bet_size, poly_price,
                    reason=f"VolSpike: {spike:.1f}x normal, disp={displacement:.0%}"
                )

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals,
                    "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} volume spike signals, {len(agent.positions)} open. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
