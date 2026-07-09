#!/usr/bin/env python3
"""
QUANT AGENT 2 — Crypto Technical Analysis
===========================================
Trades BTC/ETH price target markets using RSI, MACD, Bollinger Bands.
"""
import json, re, time, requests
import numpy as np
from datetime import datetime, timezone
from swarm_base import SwarmAgent, rate_limited_get, fetch_active_markets, LOG_DIR

agent = SwarmAgent("quant_ta", "quant")

def get_ta(symbol="BTCUSDT"):
    resp = rate_limited_get("https://api.binance.com/api/v3/klines",
                            params={"symbol": symbol, "interval": "1h", "limit": 100})
    if not resp or resp.status_code != 200:
        return None
    klines = resp.json()
    closes = np.array([float(k[4]) for k in klines])
    if len(closes) < 26:
        return None
    current = closes[-1]
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rsi = 100 - (100 / (1 + np.mean(gains[-14:]) / max(np.mean(losses[-14:]), 1e-10)))
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    bb_pct = (current - (sma20 - 2*std20)) / max((sma20 + 2*std20) - (sma20 - 2*std20), 1e-10)
    mom_1d = (current - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0
    return {"price": current, "rsi": float(rsi), "bb_pct": float(bb_pct), "momentum_1d": float(mom_1d)}

def run():
    print(f"[{agent.agent_id}] Running crypto TA strategy...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED. Stopping."); return

    ta = get_ta("BTCUSDT")
    if not ta:
        print(f"[{agent.agent_id}] No TA data."); return

    markets = fetch_active_markets(300)
    crypto = [m for m in markets if any(w in (m.get("question","")+" "+m.get("slug","")).lower() for w in ["bitcoin","btc","$"])]
    signals = []

    for m in crypto:
        q = m.get("question", "").lower()
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: prices = []
        if not prices: continue
        poly_price = float(prices[0])
        if poly_price >= 0.98 or poly_price <= 0.02: continue

        # Find price target
        match = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(k|m)?', q)
        if not match: continue
        target = float(match.group(1).replace(",",""))
        suffix = (match.group(2) or "").lower()
        if suffix == "k": target *= 1000
        elif suffix == "m": target *= 1_000_000
        elif target < 500 and ta["price"] > 10000: target *= 1000

        distance = (target - ta["price"]) / ta["price"] * 100
        is_above = any(w in q for w in ["hit","above","reach","exceed","break","over"])

        # Simple TA-based probability
        if is_above:
            if distance <= 0: prob = 0.85
            elif distance < 10: prob = 0.50 + ta["momentum_1d"] * 0.02
            else: prob = max(0.05, 0.25 - distance * 0.005)
        else:
            if distance >= 0: prob = 0.85
            else: prob = max(0.05, 0.25 + distance * 0.005)

        # RSI adjustments
        if ta["rsi"] > 70 and is_above: prob -= 0.10
        if ta["rsi"] < 30 and not is_above: prob -= 0.10

        prob = max(0.03, min(0.97, prob))
        edge = prob - poly_price

        if abs(edge) > 0.05:
            bet_size = min(abs(edge) * agent.capital * 0.4, agent.capital * 0.08)
            direction = "Yes" if edge > 0 else "No"
            signals.append({"market": m.get("question","")[:80], "condition_id": m.get("conditionId",""),
                            "prob": round(prob, 3), "poly": round(poly_price, 3), "edge": round(edge, 3),
                            "target": target, "rsi": ta["rsi"], "direction": direction})

            if bet_size >= 10:
                agent.open_position(m.get("question",""), m.get("conditionId",""),
                    direction, bet_size, poly_price, reason=f"TA: RSI={ta['rsi']:.0f} edge={edge:+.1%}")

    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "signals": signals, "summary": agent.get_summary()}, f, indent=2)

    print(f"[{agent.agent_id}] {len(signals)} crypto TA signals. PnL: ${agent.pnl:+,.2f}")

if __name__ == "__main__":
    run()
