#!/usr/bin/env python3
"""
COPY SIGNAL BOT — Follow Top Traders' Tomorrow Bets
=====================================================

1. Scans top 10 Polymarket traders every 10 minutes
2. Finds their RECENT buys (last 24h) on UNRESOLVED markets
3. Calculates each trader's historical win rate
4. When 2+ top traders bet on the same outcome → COPY SIGNAL
5. Ranks signals by: number of traders × their combined win rate × size

Run: python3 copy_signal_bot.py
"""

import json, time, requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
API = "https://data-api.polymarket.com/v1"


def get_top_traders(n=15):
    """Fetch top traders from leaderboard."""
    traders = {}
    for period in ["WEEK", "MONTH"]:
        try:
            resp = requests.get(f"{API}/leaderboard", params={
                "limit": n, "timePeriod": period, "orderBy": "PNL"
            }, timeout=15)
            for t in resp.json():
                w = t.get("proxyWallet", "")
                if w and w not in traders:
                    traders[w] = {
                        "name": t.get("userName", w[:12]),
                        "pnl": float(t.get("pnl", 0) or 0),
                        "wallet": w,
                    }
        except: pass
        time.sleep(0.3)
    return traders


def analyze_trader_full(wallet, name, pnl):
    """Get trader's recent activity + calculate win rate."""
    try:
        resp = requests.get(f"{API}/activity", params={
            "user": wallet, "limit": 500
        }, timeout=30)
        activity = resp.json()
    except:
        return None

    if not activity:
        return None

    df = pd.DataFrame(activity)
    df["usdcSize"] = pd.to_numeric(df.get("usdcSize", 0), errors="coerce").fillna(0)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Calculate historical win rate
    markets = {}
    for cid in df["conditionId"].unique():
        m = df[df["conditionId"] == cid]
        mt = m[m["type"] == "TRADE"]
        mr = m[m["type"] == "REDEEM"]
        buys = mt[mt["side"] == "BUY"]["usdcSize"].sum()
        sells = mt[mt["side"] == "SELL"]["usdcSize"].sum()
        redeemed = mr["usdcSize"].sum()
        net = sells + redeemed - buys
        resolved = len(mr) > 0 or (sells > buys * 0.5)
        markets[cid] = {"pnl": net, "resolved": resolved}

    resolved = {k: v for k, v in markets.items() if v["resolved"]}
    wins = sum(1 for v in resolved.values() if v["pnl"] > 0)
    losses = sum(1 for v in resolved.values() if v["pnl"] < 0)
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # Find RECENT buys (last 48h) on UNRESOLVED markets
    cutoff = pd.Timestamp.now(tz=None) - timedelta(hours=48)
    # Remove timezone info for comparison
    df["timestamp_dt"] = df["timestamp_dt"].dt.tz_localize(None)
    recent_buys = df[
        (df["timestamp_dt"] > cutoff) &
        (df["type"] == "TRADE") &
        (df["side"] == "BUY")
    ]

    active_bets = []
    for _, trade in recent_buys.iterrows():
        cid = trade.get("conditionId", "")
        # Check if this market has been redeemed (resolved)
        has_redeem = len(df[(df["conditionId"] == cid) & (df["type"] == "REDEEM")]) > 0
        if has_redeem:
            continue  # already resolved, skip

        active_bets.append({
            "conditionId": cid,
            "title": str(trade.get("title", ""))[:60],
            "outcome": trade.get("outcome", ""),
            "size": float(trade.get("usdcSize", 0)),
            "price": float(trade.get("price", 0) or 0),
            "time": trade["timestamp_dt"].isoformat() if pd.notna(trade["timestamp_dt"]) else "",
            "slug": trade.get("slug", ""),
        })

    # Aggregate by market (sum sizes for same market)
    market_totals = defaultdict(lambda: {"title": "", "outcome": "", "total": 0, "trades": 0, "slug": ""})
    for bet in active_bets:
        k = bet["conditionId"]
        market_totals[k]["title"] = bet["title"]
        market_totals[k]["outcome"] = bet["outcome"]
        market_totals[k]["total"] += bet["size"]
        market_totals[k]["trades"] += 1
        market_totals[k]["slug"] = bet["slug"]
        market_totals[k]["conditionId"] = k

    return {
        "name": name,
        "pnl": pnl,
        "win_rate": round(wr, 1),
        "resolved_markets": len(resolved),
        "wins": wins,
        "losses": losses,
        "active_bets": sorted(market_totals.values(), key=lambda x: -x["total"]),
    }


def find_copy_signals(trader_analyses):
    """Find markets where multiple top traders are betting the same way."""
    # Group all bets by market
    market_signals = defaultdict(lambda: {
        "title": "", "traders": [], "total_invested": 0,
        "outcomes": defaultdict(lambda: {"traders": [], "total": 0}),
    })

    for ta in trader_analyses:
        if not ta:
            continue
        for bet in ta["active_bets"]:
            cid = bet.get("conditionId", "")
            if not cid:
                continue

            market_signals[cid]["title"] = bet["title"]
            market_signals[cid]["slug"] = bet.get("slug", "")
            market_signals[cid]["traders"].append({
                "name": ta["name"],
                "win_rate": ta["win_rate"],
                "pnl": ta["pnl"],
                "size": bet["total"],
                "outcome": bet["outcome"],
            })
            market_signals[cid]["total_invested"] += bet["total"]
            market_signals[cid]["outcomes"][bet["outcome"]]["traders"].append(ta["name"])
            market_signals[cid]["outcomes"][bet["outcome"]]["total"] += bet["total"]

    # Score each market: more traders + higher WR + more money = stronger signal
    signals = []
    for cid, ms in market_signals.items():
        n_traders = len(ms["traders"])
        avg_wr = sum(t["win_rate"] for t in ms["traders"]) / n_traders if n_traders else 0

        # Find consensus outcome
        best_outcome = max(ms["outcomes"].items(), key=lambda x: x[1]["total"])
        consensus = best_outcome[0]
        consensus_pct = best_outcome[1]["total"] / ms["total_invested"] * 100 if ms["total_invested"] > 0 else 0

        # Signal strength
        score = n_traders * (avg_wr / 100) * min(ms["total_invested"] / 10000, 10)

        strength = "STRONG" if n_traders >= 3 and avg_wr > 60 else (
            "MODERATE" if n_traders >= 2 and avg_wr > 50 else "WEAK")

        signals.append({
            "title": ms["title"],
            "slug": ms.get("slug", ""),
            "n_traders": n_traders,
            "avg_wr": round(avg_wr, 1),
            "total_invested": round(ms["total_invested"], 0),
            "consensus": consensus,
            "consensus_pct": round(consensus_pct, 1),
            "strength": strength,
            "score": round(score, 2),
            "traders": ms["traders"],
            "conditionId": cid,
        })

    signals.sort(key=lambda x: (-{"STRONG": 3, "MODERATE": 2, "WEAK": 1}[x["strength"]], -x["score"]))
    return signals


def run():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"  COPY SIGNAL BOT | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Scanning top traders' active bets for tomorrow")
    print(f"{'='*70}")

    traders = get_top_traders(15)
    print(f"\n  Tracking {len(traders)} top traders")

    analyses = []
    for i, (wallet, info) in enumerate(list(traders.items())[:8]):
        print(f"  [{i+1}/8] {info['name']}...", end=" ")
        a = analyze_trader_full(wallet, info["name"], info["pnl"])
        if a:
            n_bets = len(a["active_bets"])
            print(f"WR:{a['win_rate']}% | {n_bets} active bets")
            analyses.append(a)
        else:
            print("no data")
        time.sleep(1)

    # Find copy signals
    signals = find_copy_signals(analyses)

    print(f"\n{'='*70}")
    print(f"  COPY SIGNALS FOR TOMORROW")
    print(f"{'='*70}")

    if not signals:
        print("\n  No convergence signals found. Top traders haven't placed new bets.")
    else:
        for s in signals[:10]:
            print(f"\n  [{s['strength']}] {s['title']}")
            print(f"    Consensus: {s['consensus']} ({s['consensus_pct']:.0f}% agreement)")
            print(f"    Traders: {s['n_traders']} | Avg WR: {s['avg_wr']}% | Invested: ${s['total_invested']:,.0f}")
            for t in s["traders"]:
                print(f"      {t['name']} (WR:{t['win_rate']}%, PnL:${t['pnl']:+,.0f}) → {t['outcome']} ${t['size']:,.0f}")

    # Save for dashboard
    result = {
        "time": now.isoformat(),
        "signals": signals,
        "traders_analyzed": len(analyses),
        "trader_details": [
            {"name": a["name"], "wr": a["win_rate"], "pnl": a["pnl"],
             "active": len(a["active_bets"]), "wins": a["wins"], "losses": a["losses"]}
            for a in analyses
        ],
    }
    with open(LOG_DIR / "copy_signals.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Saved to logs/copy_signals.json")
    return result


if __name__ == "__main__":
    run()
