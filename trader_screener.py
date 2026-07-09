#!/usr/bin/env python3
"""
TRADER SCREENER — Copy-Trade Copyability Scoring Engine
========================================================
Screens and ranks Polymarket traders by "copyability" — identifying
which wallets are most profitable to follow with a copy-trading bot.

Scoring Algorithm (0-100):
  Win Rate (30d)          — 25% weight
  Profit Consistency      — 20% weight (Sharpe of per-market P&L)
  Recent Performance (7d) — 15% weight
  Market Type Diversity   — 15% weight (politics/weather > crypto 15-min)
  Trade Frequency         — 10% weight (sweet spot: 50-150/month)
  Position Size Compat    — 10% weight (ideal: $10K-$100K positions)
  ROI Efficiency          — 5%  weight (P&L / volume traded)

Hard Filters:
  - Trades 30d: 20-1000 (too few = no data, too many = latency bot)
  - Avg position: $500-$500K
  - Last trade within 7 days
  - Max drawdown < 50%
  - Not exclusively crypto 15-min markets

Usage:
  python3 trader_screener.py          # full screening run, saves to logs/
  from trader_screener import get_trader_screening_data  # for dashboard
"""

import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

API = "https://data-api.polymarket.com/v1"
CACHE_FILE = LOG_DIR / "trader_screening.json"
CACHE_TTL = 1800  # 30 min

COPY_CAPITAL = 50_000

# ── Hard filter thresholds ──
MIN_TRADES_30D = 20
MAX_TRADES_30D = 1000
MIN_AVG_POSITION = 500
MAX_AVG_POSITION = 500_000
MAX_INACTIVE_DAYS = 7
MAX_DRAWDOWN_PCT = 50

# ── Market classification keywords ──
MARKET_PATTERNS = {
    "crypto_15min": ["15-min", "15 min", "btc-updown", "15-minute"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "crypto",
               "token", "price of", "market cap"],
    "politics": ["president", "election", "senate", "congress", "trump", "biden",
                 "governor", "republican", "democrat", "primary", "nominee",
                 "white house", "cabinet", "impeach", "veto", "executive order"],
    "weather": ["temperature", "weather", "degrees", "highest temp", "°f", "°c",
                "forecast", "climate"],
    "sports_nba": ["nba", "lakers", "celtics", "warriors", "bucks", "nuggets",
                   "heat", "nets", "knicks", "76ers", "suns", "mavericks",
                   "clippers", "bulls", "cavaliers", "thunder", "grizzlies",
                   "timberwolves", "pelicans", "rockets", "spurs", "magic",
                   "pacers", "hawks", "raptors", "wizards", "pistons",
                   "hornets", "trail blazers", "kings", "jazz"],
    "sports_nfl": ["nfl", "super bowl", "touchdown", "quarterback", "chiefs",
                   "eagles", "cowboys", "49ers", "bills", "ravens", "bengals",
                   "dolphins", "lions", "packers"],
    "sports_other": ["mlb", "nhl", "ufc", "premier league", "champions league",
                     "world cup", "tennis", "golf", "f1", "formula 1",
                     "soccer", "cricket", "boxing", "mma"],
    "entertainment": ["oscar", "grammy", "emmy", "box office", "movie",
                      "tv show", "album", "spotify", "youtube", "tiktok",
                      "twitter", "subscriber"],
    "economics": ["fed", "interest rate", "inflation", "gdp", "unemployment",
                  "cpi", "fomc", "tariff", "recession", "stock market",
                  "s&p", "nasdaq", "dow jones"],
}

# Copyability scores by market type (slow-moving = more copyable)
MARKET_COPY_SCORES = {
    "politics": 10,
    "weather": 9,
    "economics": 8,
    "entertainment": 8,
    "sports_nba": 7,
    "sports_nfl": 7,
    "sports_other": 7,
    "crypto": 5,
    "crypto_15min": 0,  # NOT copyable
    "other": 6,
}


def classify_market(title):
    """Categorize a market by its title."""
    t = (title or "").lower()
    for cat, keywords in MARKET_PATTERNS.items():
        for kw in keywords:
            if kw in t:
                return cat
    return "other"


def fetch_leaderboard(n=50):
    """Fetch top traders from leaderboard across time periods."""
    traders = {}
    for period in ["WEEK", "MONTH", "ALL"]:
        try:
            resp = requests.get(f"{API}/leaderboard", params={
                "limit": n, "timePeriod": period, "orderBy": "PNL"
            }, timeout=15)
            if resp.status_code != 200:
                continue
            for t in resp.json():
                w = t.get("proxyWallet", "")
                if not w:
                    continue
                if w not in traders:
                    traders[w] = {
                        "name": t.get("userName") or w[:12],
                        "wallet": w,
                        "leaderboard_pnl": {},
                        "leaderboard_volume": {},
                    }
                traders[w]["leaderboard_pnl"][period] = float(t.get("pnl", 0) or 0)
                traders[w]["leaderboard_volume"][period] = float(t.get("volume", 0) or 0)
        except Exception:
            pass
        time.sleep(0.3)
    return traders


def fetch_activity(wallet):
    """Fetch 500 recent activities for a wallet."""
    try:
        resp = requests.get(f"{API}/activity", params={
            "user": wallet, "limit": 500
        }, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def compute_metrics(activity, name, wallet):
    """Compute all scoring metrics from a trader's activity data."""
    if not activity:
        return None

    now = datetime.now(timezone.utc)
    cutoff_30d = now - timedelta(days=30)
    cutoff_7d = now - timedelta(days=7)

    # Parse timestamps and amounts
    trades = []
    redeems = []
    for a in activity:
        try:
            ts = datetime.fromtimestamp(float(a.get("timestamp", 0)), tz=timezone.utc)
        except (ValueError, TypeError):
            continue
        size = float(a.get("usdcSize", 0) or 0)
        entry = {
            "timestamp": ts,
            "type": a.get("type", ""),
            "side": a.get("side", ""),
            "conditionId": a.get("conditionId", ""),
            "title": a.get("title", ""),
            "outcome": a.get("outcome", ""),
            "usdcSize": size,
        }
        if entry["type"] == "TRADE":
            trades.append(entry)
        elif entry["type"] == "REDEEM":
            redeems.append(entry)

    if not trades:
        return None

    # ── Per-market P&L ──
    markets = {}
    all_entries = trades + redeems
    for entry in all_entries:
        cid = entry["conditionId"]
        if cid not in markets:
            markets[cid] = {
                "title": entry["title"],
                "buys": 0, "sells": 0, "redeemed": 0,
                "trades": 0, "category": classify_market(entry["title"]),
                "first_trade": entry["timestamp"],
                "last_trade": entry["timestamp"],
                "is_resolved": False,
            }
        m = markets[cid]
        if entry["type"] == "TRADE":
            if entry["side"] == "BUY":
                m["buys"] += entry["usdcSize"]
            else:
                m["sells"] += entry["usdcSize"]
            m["trades"] += 1
        elif entry["type"] == "REDEEM":
            m["redeemed"] += entry["usdcSize"]
            m["is_resolved"] = True
        m["last_trade"] = max(m["last_trade"], entry["timestamp"])
        m["first_trade"] = min(m["first_trade"], entry["timestamp"])

    for m in markets.values():
        m["pnl"] = m["sells"] + m["redeemed"] - m["buys"]

    # ── Win rate (all time) ──
    resolved = [m for m in markets.values() if m["is_resolved"]]
    wins_all = sum(1 for m in resolved if m["pnl"] > 0)
    losses_all = sum(1 for m in resolved if m["pnl"] <= 0)
    wr_all = wins_all / max(wins_all + losses_all, 1) * 100

    # ── Win rate (30d) ──
    markets_30d = {k: v for k, v in markets.items() if v["last_trade"] >= cutoff_30d}
    resolved_30d = [m for m in markets_30d.values() if m["is_resolved"]]
    wins_30d = sum(1 for m in resolved_30d if m["pnl"] > 0)
    losses_30d = sum(1 for m in resolved_30d if m["pnl"] <= 0)
    wr_30d = wins_30d / max(wins_30d + losses_30d, 1) * 100

    # ── P&L calculations ──
    pnl_all = sum(m["pnl"] for m in markets.values())
    pnl_30d = sum(m["pnl"] for m in markets_30d.values())
    markets_7d = {k: v for k, v in markets.items() if v["last_trade"] >= cutoff_7d}
    pnl_7d = sum(m["pnl"] for m in markets_7d.values())

    # ── Trade frequency (30d) ──
    trades_30d = [t for t in trades if t["timestamp"] >= cutoff_30d]
    n_trades_30d = len(trades_30d)

    # ── Average position size ──
    position_sizes = [m["buys"] for m in markets.values() if m["buys"] > 0]
    avg_position = np.mean(position_sizes) if position_sizes else 0

    # ── Volume ──
    total_volume = sum(m["buys"] for m in markets.values())

    # ── ROI ──
    roi_pct = (pnl_all / total_volume * 100) if total_volume > 0 else 0

    # ── Last trade ──
    last_trade = max(t["timestamp"] for t in trades) if trades else now - timedelta(days=999)

    # ── Equity curve ──
    sorted_trades = sorted(all_entries, key=lambda x: x["timestamp"])
    cum_pnl = 0
    equity_curve = []
    for entry in sorted_trades:
        if entry["type"] == "TRADE":
            if entry["side"] == "BUY":
                cum_pnl -= entry["usdcSize"]
            else:
                cum_pnl += entry["usdcSize"]
        elif entry["type"] == "REDEEM":
            cum_pnl += entry["usdcSize"]
        equity_curve.append({
            "t": entry["timestamp"].isoformat(),
            "pnl": round(cum_pnl, 2),
        })

    # ── Max drawdown ──
    peak = 0
    max_dd = 0
    for pt in equity_curve:
        peak = max(peak, pt["pnl"])
        if peak > 0:
            dd = (peak - pt["pnl"]) / peak * 100
            max_dd = max(max_dd, dd)

    # ── Consistency (Sharpe of per-market P&L) ──
    market_pnls = [m["pnl"] for m in resolved]
    if len(market_pnls) >= 3:
        pnl_mean = np.mean(market_pnls)
        pnl_std = np.std(market_pnls)
        consistency_sharpe = pnl_mean / pnl_std if pnl_std > 0 else 0
    else:
        consistency_sharpe = 0

    # ── Market diversity ──
    category_counts = {}
    category_pnl = {}
    for m in markets.values():
        cat = m["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        category_pnl[cat] = category_pnl.get(cat, 0) + m["pnl"]

    # Weighted copyability score by market type
    total_market_trades = sum(category_counts.values())
    if total_market_trades > 0:
        diversity_score = sum(
            (count / total_market_trades) * MARKET_COPY_SCORES.get(cat, 6)
            for cat, count in category_counts.items()
        )
    else:
        diversity_score = 0

    # Main market types
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    main_markets = ", ".join(c[0] for c in sorted_cats[:3])

    # ── Hourly activity distribution (for heatmap) ──
    hour_counts = [0] * 24
    day_counts = [0] * 7
    for t in trades:
        hour_counts[t["timestamp"].hour] += 1
        day_counts[t["timestamp"].weekday()] += 1

    # ── Best/Worst trades ──
    sorted_by_pnl = sorted(markets.values(), key=lambda x: x["pnl"], reverse=True)
    best_5 = [{"title": m["title"][:60], "pnl": round(m["pnl"], 2), "category": m["category"]}
              for m in sorted_by_pnl[:5]]
    worst_5 = [{"title": m["title"][:60], "pnl": round(m["pnl"], 2), "category": m["category"]}
               for m in sorted_by_pnl[-5:] if m["pnl"] < 0]

    # ── Crypto 15-min fraction ──
    crypto_15m_count = category_counts.get("crypto_15min", 0)
    crypto_15m_frac = crypto_15m_count / total_market_trades if total_market_trades > 0 else 0

    return {
        "name": name,
        "wallet": wallet,
        "wallet_short": wallet[:8] + "..." + wallet[-4:],
        # Win rates
        "win_rate_all": round(wr_all, 1),
        "win_rate_30d": round(wr_30d, 1),
        "wins_all": wins_all, "losses_all": losses_all,
        "wins_30d": wins_30d, "losses_30d": losses_30d,
        # P&L
        "pnl_all": round(pnl_all, 0),
        "pnl_30d": round(pnl_30d, 0),
        "pnl_7d": round(pnl_7d, 0),
        # Trades
        "trades_30d": n_trades_30d,
        "trades_all": len(trades),
        "markets_traded": len(markets),
        # Position sizing
        "avg_position": round(avg_position, 0),
        "total_volume": round(total_volume, 0),
        # Performance
        "roi_pct": round(roi_pct, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "consistency_sharpe": round(consistency_sharpe, 3),
        # Diversity
        "main_markets": main_markets,
        "diversity_score": round(diversity_score, 1),
        "category_breakdown": {cat: {"count": cnt, "pnl": round(category_pnl.get(cat, 0), 0)}
                               for cat, cnt in sorted_cats},
        "crypto_15m_fraction": round(crypto_15m_frac, 2),
        # Timing
        "last_trade": last_trade.isoformat(),
        "days_since_trade": (now - last_trade).days,
        "hour_counts": hour_counts,
        "day_counts": day_counts,
        # Curves (subsample for rankings view)
        "equity_curve": equity_curve,
        "equity_curve_mini": equity_curve[::max(1, len(equity_curve) // 20)][-20:],
        # Best/worst
        "best_trades": best_5,
        "worst_trades": worst_5,
        # Trade history (last 30)
        "recent_trades": [
            {"title": m["title"][:60], "pnl": round(m["pnl"], 2),
             "buys": round(m["buys"], 0), "sells": round(m["sells"] + m["redeemed"], 0),
             "trades": m["trades"], "category": m["category"],
             "resolved": m["is_resolved"],
             "last_trade": m["last_trade"].isoformat()}
            for m in sorted(markets.values(), key=lambda x: x["last_trade"], reverse=True)[:30]
        ],
    }


def apply_hard_filters(metrics):
    """Apply hard filters. Returns (passed: bool, reasons: list)."""
    reasons = []
    if metrics["trades_30d"] < MIN_TRADES_30D:
        reasons.append(f"Too few trades: {metrics['trades_30d']} < {MIN_TRADES_30D}")
    if metrics["trades_30d"] > MAX_TRADES_30D:
        reasons.append(f"Too many trades: {metrics['trades_30d']} > {MAX_TRADES_30D} (likely bot)")
    if metrics["avg_position"] < MIN_AVG_POSITION:
        reasons.append(f"Position too small: ${metrics['avg_position']:.0f} < ${MIN_AVG_POSITION}")
    if metrics["avg_position"] > MAX_AVG_POSITION:
        reasons.append(f"Position too large: ${metrics['avg_position']:.0f} > ${MAX_AVG_POSITION:,}")
    if metrics["days_since_trade"] > MAX_INACTIVE_DAYS:
        reasons.append(f"Inactive: {metrics['days_since_trade']}d > {MAX_INACTIVE_DAYS}d")
    if metrics["max_drawdown_pct"] > MAX_DRAWDOWN_PCT:
        reasons.append(f"Drawdown too high: {metrics['max_drawdown_pct']:.0f}% > {MAX_DRAWDOWN_PCT}%")
    if metrics["crypto_15m_fraction"] > 0.8:
        reasons.append(f"Mostly 15-min crypto: {metrics['crypto_15m_fraction']:.0%}")
    return len(reasons) == 0, reasons


def score_trader(metrics):
    """Compute copyability score (0-100) with component breakdown."""
    scores = {}

    # Win Rate (25%) — 0 to 10 points
    wr = metrics["win_rate_30d"]
    if wr >= 70:
        scores["win_rate"] = 10
    elif wr >= 65:
        scores["win_rate"] = 7
    elif wr >= 60:
        scores["win_rate"] = 5
    elif wr >= 55:
        scores["win_rate"] = 3
    elif wr >= 50:
        scores["win_rate"] = 1
    else:
        scores["win_rate"] = 0

    # Consistency (20%) — Sharpe of per-market P&L
    sharpe = metrics["consistency_sharpe"]
    scores["consistency"] = min(10, max(0, sharpe * 5))

    # Recent Performance (15%) — 7d P&L relative to 30d
    pnl_7d = metrics["pnl_7d"]
    pnl_30d = metrics["pnl_30d"]
    if pnl_7d > 0 and pnl_30d > 0:
        # Good recent performance relative to monthly
        ratio = pnl_7d / max(abs(pnl_30d), 1)
        scores["recent"] = min(10, ratio * 30)
    elif pnl_7d > 0:
        scores["recent"] = 5
    elif pnl_7d > -1000:
        scores["recent"] = 2
    else:
        scores["recent"] = 0

    # Market Diversity (15%) — weighted by copyability
    scores["diversity"] = metrics["diversity_score"]

    # Trade Frequency (10%) — sweet spot
    freq = metrics["trades_30d"]
    if 50 <= freq <= 150:
        scores["frequency"] = 10
    elif 150 < freq <= 300:
        scores["frequency"] = 7
    elif 20 <= freq < 50:
        scores["frequency"] = 5
    elif 300 < freq <= 500:
        scores["frequency"] = 3
    else:
        scores["frequency"] = 1

    # Position Size Compatibility (10%) — can we copy at 0.01-0.05x?
    avg = metrics["avg_position"]
    copy_size = avg * 0.03  # 3% of their position
    if 500 <= copy_size <= 5000:
        scores["position_compat"] = 10
    elif 200 <= copy_size <= 10000:
        scores["position_compat"] = 7
    elif 100 <= copy_size <= 20000:
        scores["position_compat"] = 4
    else:
        scores["position_compat"] = 1

    # ROI Efficiency (5%)
    roi = metrics["roi_pct"]
    if roi > 5:
        scores["roi"] = 10
    elif roi > 3:
        scores["roi"] = 7
    elif roi > 1:
        scores["roi"] = 4
    elif roi > 0:
        scores["roi"] = 2
    else:
        scores["roi"] = 0

    # Weighted total
    weights = {
        "win_rate": 0.25,
        "consistency": 0.20,
        "recent": 0.15,
        "diversity": 0.15,
        "frequency": 0.10,
        "position_compat": 0.10,
        "roi": 0.05,
    }
    total = sum(scores[k] * weights[k] * 10 for k in weights)
    total = min(100, max(0, total))

    # Grade
    if total >= 80:
        grade = "S"
    elif total >= 65:
        grade = "A"
    elif total >= 50:
        grade = "B"
    elif total >= 35:
        grade = "C"
    else:
        grade = "D"

    # Copy ratio recommendation
    avg_pos = metrics["avg_position"]
    if avg_pos > 0:
        copy_ratio = min(0.1, COPY_CAPITAL * 0.1 / avg_pos)
        rec_bet = avg_pos * copy_ratio
    else:
        copy_ratio = 0
        rec_bet = 0

    # Estimated daily return
    if metrics["trades_30d"] > 0 and metrics["pnl_30d"] > 0:
        daily_pnl_trader = metrics["pnl_30d"] / 30
        est_daily = daily_pnl_trader * copy_ratio
    else:
        est_daily = 0

    return {
        "score": round(total, 1),
        "grade": grade,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "copy_ratio": round(copy_ratio, 4),
        "recommended_bet": round(rec_bet, 0),
        "est_daily_return": round(est_daily, 2),
    }


def run_screening(n=50):
    """Full screening pipeline."""
    print(f"\n{'='*60}")
    print(f"TRADER SCREENER — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # Step 1: Fetch leaderboard
    print(f"\nFetching top {n} traders from leaderboard...")
    lb = fetch_leaderboard(n)
    print(f"  Found {len(lb)} unique wallets")

    # Step 2: Analyze each trader
    results = []
    filtered_out = []
    for i, (wallet, info) in enumerate(lb.items()):
        name = info["name"]
        print(f"  [{i+1}/{len(lb)}] Analyzing {name}...", end=" ", flush=True)

        activity = fetch_activity(wallet)
        if not activity:
            print("no data")
            continue

        metrics = compute_metrics(activity, name, wallet)
        if not metrics:
            print("no trades")
            continue

        # Merge leaderboard data
        metrics["leaderboard_pnl"] = info.get("leaderboard_pnl", {})
        metrics["leaderboard_volume"] = info.get("leaderboard_volume", {})

        # Apply hard filters
        passed, reasons = apply_hard_filters(metrics)
        if not passed:
            print(f"FILTERED: {reasons[0]}")
            filtered_out.append({"name": name, "wallet": wallet[:12], "reasons": reasons})
            continue

        # Score
        scoring = score_trader(metrics)
        result = {**metrics, **scoring, "filter_reasons": []}
        # Trim equity curve for storage
        result["equity_curve_mini"] = result.pop("equity_curve_mini")
        result.pop("equity_curve")  # full curve saved only in detail view
        result.pop("hour_counts")
        result.pop("day_counts")

        results.append(result)
        print(f"Score: {scoring['score']:.0f} ({scoring['grade']})")

        time.sleep(0.3)

    # Step 3: Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Stats
    scores = [r["score"] for r in results]
    wrs = [r["win_rate_30d"] for r in results]
    grade_dist = {}
    for r in results:
        g = r["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_analyzed": len(lb),
        "n_passed": len(results),
        "n_filtered": len(filtered_out),
        "traders": results,
        "filtered": filtered_out[:10],
        "stats": {
            "avg_score": round(np.mean(scores), 1) if scores else 0,
            "avg_wr": round(np.mean(wrs), 1) if wrs else 0,
            "max_score": round(max(scores), 1) if scores else 0,
            "total_screened": len(lb),
            "grade_dist": grade_dist,
        },
    }

    # Save
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, default=str)
    print(f"\nSaved {len(results)} ranked traders to {CACHE_FILE}")
    print(f"Grade distribution: {grade_dist}")

    return data


def get_trader_screening_data():
    """Read cached screening data. Trigger fresh scan if stale."""
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            ts = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            data["_cache_age_s"] = round(age)
            if age < CACHE_TTL:
                return data
        except Exception:
            pass

    # Cache missing or stale — run screening
    return run_screening()


def get_trader_detail(wallet):
    """Fetch deep-dive data for a single trader."""
    if not wallet:
        return {"error": "wallet required"}

    activity = fetch_activity(wallet)
    if not activity:
        return {"error": "no activity found"}

    # Try to get name from cached screening data
    name = wallet[:12]
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            for t in cached.get("traders", []):
                if t.get("wallet") == wallet:
                    name = t.get("name", name)
                    break
        except Exception:
            pass

    metrics = compute_metrics(activity, name, wallet)
    if not metrics:
        return {"error": "could not compute metrics"}

    scoring = score_trader(metrics)

    return {
        **metrics,
        **scoring,
        # Full equity curve for chart
        "equity_curve": metrics["equity_curve"],
        # Drawdown series
        "drawdown_series": _compute_drawdown_series(metrics["equity_curve"]),
        # Timing heatmap (7 days x 24 hours)
        "timing_heatmap": _compute_timing_heatmap(activity),
    }


def _compute_drawdown_series(equity_curve):
    """Compute drawdown percentage series from equity curve."""
    peak = 0
    series = []
    for pt in equity_curve:
        peak = max(peak, pt["pnl"])
        dd = ((peak - pt["pnl"]) / peak * 100) if peak > 0 else 0
        series.append({"t": pt["t"], "dd_pct": round(dd, 1)})
    return series


def _compute_timing_heatmap(activity):
    """Build 7x24 trading activity heatmap."""
    grid = [[0] * 24 for _ in range(7)]
    for a in activity:
        if a.get("type") != "TRADE":
            continue
        try:
            ts = datetime.fromtimestamp(float(a.get("timestamp", 0)), tz=timezone.utc)
            grid[ts.weekday()][ts.hour] += 1
        except (ValueError, TypeError):
            pass
    return grid


def simulate_portfolio(wallets):
    """Simulate combined portfolio of 2-3 traders."""
    if not wallets or len(wallets) < 2:
        return {"error": "need at least 2 wallets"}
    if len(wallets) > 3:
        wallets = wallets[:3]

    allocation = COPY_CAPITAL / len(wallets)
    individual = {}
    all_daily = {}

    for wallet in wallets:
        activity = fetch_activity(wallet)
        if not activity:
            continue
        metrics = compute_metrics(activity, wallet[:12], wallet)
        if not metrics:
            continue

        scoring = score_trader(metrics)
        copy_ratio = scoring["copy_ratio"]

        # Build daily P&L series
        curve = metrics["equity_curve"]
        daily_pnl = {}
        prev_pnl = 0
        for pt in curve:
            day = pt["t"][:10]
            daily_pnl[day] = pt["pnl"] - prev_pnl
            prev_pnl = pt["pnl"]

        # Scale by copy ratio
        scaled_daily = {d: pnl * copy_ratio for d, pnl in daily_pnl.items()}

        individual[wallet] = {
            "name": metrics["name"],
            "score": scoring["score"],
            "grade": scoring["grade"],
            "copy_ratio": copy_ratio,
            "allocation": round(allocation, 0),
            "pnl_30d": metrics["pnl_30d"],
            "win_rate_30d": metrics["win_rate_30d"],
            "daily_pnl": scaled_daily,
            "equity_curve_mini": metrics["equity_curve_mini"],
        }
        for d, pnl in scaled_daily.items():
            all_daily[d] = all_daily.get(d, 0) + pnl

        time.sleep(0.3)

    if not individual:
        return {"error": "could not fetch trader data"}

    # Combined equity curve
    sorted_days = sorted(all_daily.keys())
    cum = 0
    combined_curve = []
    for d in sorted_days:
        cum += all_daily[d]
        combined_curve.append({"t": d, "pnl": round(cum, 2)})

    # Combined stats
    daily_vals = [all_daily[d] for d in sorted_days]
    combined_sharpe = 0
    if daily_vals and np.std(daily_vals) > 0:
        combined_sharpe = np.mean(daily_vals) / np.std(daily_vals) * math.sqrt(365)

    peak = 0
    max_dd = 0
    for pt in combined_curve:
        peak = max(peak, pt["pnl"])
        if peak > 0:
            dd = (peak - pt["pnl"]) / peak * 100
            max_dd = max(max_dd, dd)

    total_return = combined_curve[-1]["pnl"] if combined_curve else 0

    # Correlation matrix
    trader_wallets = list(individual.keys())
    n = len(trader_wallets)
    corr_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d1 = individual[trader_wallets[i]]["daily_pnl"]
            d2 = individual[trader_wallets[j]]["daily_pnl"]
            common_days = sorted(set(d1.keys()) & set(d2.keys()))
            if len(common_days) >= 5:
                v1 = [d1[d] for d in common_days]
                v2 = [d2[d] for d in common_days]
                if np.std(v1) > 0 and np.std(v2) > 0:
                    corr_matrix[i][j] = round(float(np.corrcoef(v1, v2)[0, 1]), 2)
                else:
                    corr_matrix[i][j] = 0
            else:
                corr_matrix[i][j] = 0
            if i == j:
                corr_matrix[i][j] = 1.0

    return {
        "wallets": trader_wallets,
        "names": [individual[w]["name"] for w in trader_wallets],
        "individual": individual,
        "combined_curve": combined_curve,
        "correlation_matrix": corr_matrix,
        "combined_stats": {
            "sharpe": round(combined_sharpe, 2),
            "max_drawdown_pct": round(max_dd, 1),
            "total_return": round(total_return, 2),
            "total_capital": COPY_CAPITAL,
            "est_daily": round(np.mean(daily_vals), 2) if daily_vals else 0,
            "est_monthly": round(np.mean(daily_vals) * 30, 2) if daily_vals else 0,
            "n_days": len(sorted_days),
        },
    }


if __name__ == "__main__":
    run_screening()
