#!/usr/bin/env python3
"""AGENT 22 -- Fractal Pattern Scanner: cross-category lag + outlier reversion."""
import json, math
from datetime import datetime, timezone
from swarm_base import (
    SwarmAgent, fetch_active_markets, fetch_market_price,
    rate_limited_get, LOG_DIR, GAMMA_HOST, STARTING_CAPITAL,
)

agent = SwarmAgent("agent_22", "chaos")
MAX_POS, TP, SL = 10, 0.08, -0.05
LEADER_DISP, LAG_THRESH, OUTLIER_Z, MIN_CAT = 0.18, 0.10, 1.4, 3


def categorize(m):
    tags = m.get("tags", []) or []
    if isinstance(tags, str):
        try: tags = json.loads(tags)
        except: tags = []
    if tags and isinstance(tags, list) and tags:
        t = tags[0]
        return t.get("label", str(t)) if isinstance(t, dict) else str(t)
    parts = m.get("slug", "").split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else None


def _try_open(m, p, d, bet, reason):
    if bet >= 10 and len(agent.positions) < MAX_POS:
        agent.open_position(m.get("question", ""), m.get("conditionId", ""),
                            d, bet, p, reason=reason)


def run():
    print(f"[{agent.agent_id}] Fractal Scanner -- cross-category analysis...")
    if agent.is_killed():
        print(f"[{agent.agent_id}] KILLED."); return

    for pos in list(agent.positions):
        if pos.get("status") != "open": continue
        cur = fetch_market_price(pos["condition_id"])
        if cur is None: continue
        pnl_pct = (cur - pos["entry_price"]) / max(pos["entry_price"], 0.01)
        if pnl_pct >= TP or pnl_pct <= SL:
            r = agent.close_position(pos["condition_id"], cur)
            print(f"  [EXIT] {pos['market'][:40]} PnL: ${r:+.2f}")

    # -- Group markets by category --
    markets = fetch_active_markets(500)
    cats = {}
    for m in markets:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try: prices = json.loads(prices)
            except: continue
        if not prices:
            continue
        price = float(prices[0])
        vol = float(m.get("volume", 0) or 0)
        liq = float(m.get("liquidity", 0) or 0)
        if vol < 5000 or liq < 200 or price > 0.95 or price < 0.05:
            continue
        cat = categorize(m)
        if not cat:
            continue
        cats.setdefault(cat, []).append(
            {"m": m, "price": price, "vol": vol, "disp": price - 0.5})

    cats = {k: v for k, v in cats.items() if len(v) >= MIN_CAT}

    # -- Category-level stats --
    cat_stats = {}
    for cat, members in cats.items():
        disps = [x["disp"] for x in members]
        avg_d = sum(disps) / len(disps)
        tot_vol = sum(x["vol"] for x in members)
        momentum = sum(x["disp"] * x["vol"] for x in members) / max(tot_vol, 1)
        var = sum((d - avg_d) ** 2 for d in disps) / len(disps)
        cat_stats[cat] = {"avg_d": avg_d, "abs_d": abs(avg_d), "mom": momentum,
                          "avg_p": sum(x["price"] for x in members) / len(members),
                          "std": math.sqrt(var) if var > 0 else 0.001,
                          "members": members}

    sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]["abs_d"], reverse=True)
    leaders = [(k, v) for k, v in sorted_cats if v["abs_d"] >= LEADER_DISP]
    laggers = [(k, v) for k, v in sorted_cats if v["abs_d"] < LAG_THRESH]
    signals = []

    # STRATEGY 1: Category lag -- laggers should follow leaders
    if leaders and laggers:
        ldr_dir = sum(s["mom"] for _, s in leaders) / len(leaders)
        for lag_cat, ls in laggers[:5]:
            for mem in ls["members"]:
                m, p = mem["m"], mem["price"]
                if ldr_dir > 0.03 and p < 0.55: d = "Yes"
                elif ldr_dir < -0.03 and p > 0.45: d = "No"
                else: continue
                strength = min(abs(ldr_dir) * 4, 2.0)
                bet = min(strength * agent.capital * 0.012, agent.capital * 0.05)
                reason = f"FracLag: {lag_cat[:20]} ldr={ldr_dir:.2f}"
                signals.append({
                    "market": m.get("question", "")[:80], "condition_id": m.get("conditionId", ""),
                    "direction": d, "price": round(p, 3), "category": lag_cat,
                    "cat_disp": round(ls["avg_d"], 3), "leader_mom": round(ldr_dir, 3),
                    "strength": round(strength, 3), "strategy": reason})
                _try_open(m, p, d, bet, reason)

    # STRATEGY 2: Outlier reversion within categories
    for cat, st in cat_stats.items():
        for mem in st["members"]:
            z = (mem["disp"] - st["avg_d"]) / max(st["std"], 0.001)
            if abs(z) < OUTLIER_Z:
                continue
            m, p = mem["m"], mem["price"]
            if z > OUTLIER_Z and p < 0.88: d = "No"
            elif z < -OUTLIER_Z and p > 0.12: d = "Yes"
            else: continue
            strength = min(abs(z) * 0.6, 2.0)
            bet = min(strength * agent.capital * 0.01, agent.capital * 0.04)
            reason = f"FracOut: z={z:.1f} {cat[:20]}"
            signals.append({
                "market": m.get("question", "")[:80], "condition_id": m.get("conditionId", ""),
                "direction": d, "price": round(p, 3), "category": cat,
                "zscore": round(z, 2), "cat_avg": round(st["avg_p"], 3),
                "strength": round(strength, 3), "strategy": reason})
            _try_open(m, p, d, bet, reason)

    # -- Save signals --
    with open(LOG_DIR / f"signals_{agent.agent_id}.json", "w") as f:
        json.dump({"time": datetime.now(timezone.utc).isoformat(),
                    "strategy": "fractal_pattern_scanner",
                    "categories": len(cat_stats),
                    "leaders": [k for k, _ in leaders],
                    "laggers": [k for k, _ in laggers],
                    "signals": signals[:30],
                    "summary": agent.get_summary()}, f, indent=2)

    sm = agent.get_summary()
    print(f"[{agent.agent_id}] {len(cat_stats)} cats, {len(leaders)} leading, "
          f"{len(signals)} signals. {sm['open_positions']} open. PnL: ${sm['pnl']:+,.2f}")


if __name__ == "__main__":
    run()
