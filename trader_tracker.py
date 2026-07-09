#!/usr/bin/env python3
"""
TOP TRADER TRACKER
===================
Monitors the top Polymarket traders' positions and signals.
Runs every 10 minutes, saves to logs for dashboard display.
"""

import json, time, requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
API = "https://data-api.polymarket.com/v1"


def fetch_top_traders(n=10):
    """Get top traders from leaderboard."""
    traders = {}
    for period in ["WEEK", "MONTH", "ALL"]:
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
                        "volume": float(t.get("volume", 0) or 0),
                        "wallet": w,
                        "period": period,
                    }
        except:
            pass
        time.sleep(0.3)
    return traders


def analyze_trader(wallet, name):
    """Analyze a trader's recent activity and current positions."""
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
    if df.empty:
        return None

    df["usdcSize"] = pd.to_numeric(df.get("usdcSize", 0), errors="coerce").fillna(0)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    trades = df[df["type"] == "TRADE"]
    redeems = df[df["type"] == "REDEEM"]

    # Per-market P&L
    markets = {}
    for cid in df["conditionId"].unique():
        m = df[df["conditionId"] == cid]
        mt = m[m["type"] == "TRADE"]
        mr = m[m["type"] == "REDEEM"]

        buys = mt[mt["side"] == "BUY"]["usdcSize"].sum()
        sells = mt[mt["side"] == "SELL"]["usdcSize"].sum()
        redeemed = mr["usdcSize"].sum()
        pnl = sells + redeemed - buys

        title = str(m.iloc[0].get("title", m.iloc[0].get("conditionId", "")))[:60]
        last_trade = m["timestamp_dt"].max()
        last_side = mt.iloc[-1]["side"] if len(mt) > 0 else ""
        outcome = mt.iloc[-1].get("outcome", "") if len(mt) > 0 else ""

        markets[cid] = {
            "title": title,
            "pnl": round(pnl, 2),
            "buys": round(buys, 2),
            "sells": round(sells + redeemed, 2),
            "trades": len(mt),
            "last_trade": last_trade.isoformat() if pd.notna(last_trade) else "",
            "last_side": last_side,
            "outcome": outcome,
            "resolved": len(mr) > 0,
        }

    # Active positions (bought but not sold/redeemed)
    active = {k: v for k, v in markets.items() if not v["resolved"] and v["buys"] > v["sells"]}

    wins = sum(1 for v in markets.values() if v["pnl"] > 0)
    losses = sum(1 for v in markets.values() if v["pnl"] < 0)
    total_pnl = sum(v["pnl"] for v in markets.values())

    return {
        "name": name,
        "wallet": wallet[:12] + "...",
        "total_activity": len(df),
        "total_trades": len(trades),
        "markets_traded": len(markets),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "net_pnl": round(total_pnl, 0),
        "active_positions": len(active),
        "active": sorted(active.values(), key=lambda x: -abs(x["buys"]))[:10],
        "recent_trades": sorted(markets.values(), key=lambda x: x["last_trade"], reverse=True)[:5],
    }


def run():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"  TOP TRADER TRACKER | {now.strftime('%H:%M UTC')}")
    print(f"{'='*60}")

    traders = fetch_top_traders(10)
    print(f"  Found {len(traders)} top traders")

    results = []
    for i, (wallet, info) in enumerate(list(traders.items())[:5]):
        print(f"\n  Analyzing #{i+1} {info['name']}...", end=" ")
        analysis = analyze_trader(wallet, info["name"])
        if analysis:
            print(f"WR:{analysis['win_rate']}% | Active:{analysis['active_positions']} positions")
            analysis["leaderboard_pnl"] = info["pnl"]
            analysis["period"] = info["period"]
            results.append(analysis)
        else:
            print("no data")
        time.sleep(1)

    # Save
    log_path = LOG_DIR / "trader_tracker.json"
    with open(log_path, "w") as f:
        json.dump({"time": now.isoformat(), "traders": results}, f, indent=2, default=str)

    print(f"\n  Saved to {log_path}")

    # Summary
    print(f"\n  {'#':<3} {'Trader':<15} {'WR':<7} {'Markets':<9} {'Active':<8} {'P&L'}")
    print(f"  {'-'*55}")
    for i, t in enumerate(results):
        print(f"  {i+1:<3} {t['name']:<15} {t['win_rate']:<6}% {t['markets_traded']:<9} {t['active_positions']:<8} ${t['leaderboard_pnl']:+,.0f}")

        if t["active"]:
            print(f"      Active positions:")
            for pos in t["active"][:3]:
                print(f"        {pos['last_side']} {pos['outcome']} | ${pos['buys']:,.0f} invested | {pos['title']}")

    return results


if __name__ == "__main__":
    run()
