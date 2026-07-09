#!/usr/bin/env python3
"""
MTY PERFORMANCE TRACKER
========================
Tracks how our predictions perform over time.
Checks resolved markets against our predictions to calculate:
  - Accuracy (overall and per-category)
  - Simulated P&L (Kelly-sized bets)
  - Win streaks and drawdowns
  - Edge decay analysis

Runs via cron every hour. Saves results to logs/performance.json

Usage:
    python3 performance_tracker.py
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("performance")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
PERF_FILE = LOG_DIR / "performance.json"
GAMMA_HOST = "https://gamma-api.polymarket.com"


def load_performance():
    """Load existing performance data."""
    if PERF_FILE.exists():
        try:
            return json.load(open(PERF_FILE))
        except:
            pass
    return {
        "tracked_predictions": [],
        "resolved": [],
        "stats": {},
        "daily_pnl": [],
        "last_check": None,
    }


def save_performance(data):
    """Save performance data."""
    with open(PERF_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_daily_predictions():
    """Load all daily prediction snapshots."""
    predictions = []
    seen_keys = set()

    for f in sorted(LOG_DIR.glob("predictions_*.json")):
        try:
            data = json.load(open(f))
            for p in data.get("predictions", []):
                # Only track predictions with actual edge
                if abs(p.get("edge", 0)) < 0.01:
                    continue
                key = f"{p.get('conditionId', '')}-{p.get('slug', '')}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    p["prediction_date"] = f.stem.replace("predictions_", "")
                    predictions.append(p)
        except:
            pass

    return predictions


def check_market_resolution(condition_id, slug):
    """Check if a market has resolved and what the outcome was."""
    # Try condition ID first
    if condition_id:
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/markets",
                params={"condition_id": condition_id},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    m = data[0] if isinstance(data, list) else data
                    return parse_resolution(m)
        except:
            pass

    # Try slug
    if slug:
        try:
            resp = requests.get(
                f"{GAMMA_HOST}/markets",
                params={"slug": slug, "limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    m = data[0] if isinstance(data, list) else data
                    return parse_resolution(m)
        except:
            pass

    return None


def parse_resolution(market):
    """Parse market resolution status."""
    prices = market.get("outcomePrices", "[]")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            prices = []

    outcomes = market.get("outcomes", "[]")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            outcomes = []

    # Check if resolved (price at 0 or 1)
    resolved = False
    winner = None
    for i, p in enumerate(prices):
        pf = float(p)
        if pf >= 0.99:
            resolved = True
            winner = outcomes[i] if i < len(outcomes) else f"Outcome_{i}"
            break
        elif pf <= 0.01:
            # The other outcome won
            resolved = True

    if resolved and winner is None and len(outcomes) == 2:
        # Binary market, find the winner
        for i, p in enumerate(prices):
            if float(p) >= 0.99:
                winner = outcomes[i] if i < len(outcomes) else "Yes"
                break

    if not resolved:
        return None

    return {
        "resolved": True,
        "winner": winner,
        "prices": [float(p) for p in prices],
        "outcomes": outcomes,
    }


def calculate_kelly_bet(our_prob, market_price, bankroll=50000, fraction=0.25):
    """Calculate Kelly Criterion bet size."""
    if our_prob <= 0 or market_price <= 0 or market_price >= 1:
        return 0
    b = (1 / market_price) - 1  # odds
    p = our_prob
    q = 1 - p
    f = (p * b - q) / b
    if f <= 0:
        return 0
    f = f * fraction  # fractional Kelly
    f = min(f, 0.10)  # cap at 10% of bankroll
    return max(50, bankroll * f)


def run_tracker():
    """Main tracking loop."""
    print(f"\n{'='*60}")
    print(f"  MTY PERFORMANCE TRACKER")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    perf = load_performance()
    predictions = load_daily_predictions()

    print(f"\n  Found {len(predictions)} edge predictions to track")

    # Track which predictions we've already resolved
    resolved_keys = {r.get("key", "") for r in perf.get("resolved", [])}

    new_resolved = 0
    checked = 0

    for pred in predictions:
        cid = pred.get("conditionId", "")
        slug = pred.get("slug", "")
        key = f"{cid}-{slug}"

        if key in resolved_keys:
            continue

        # Check if resolved (rate limit: max 30 checks per run)
        if checked >= 30:
            break
        checked += 1
        time.sleep(0.3)  # Rate limit

        result = check_market_resolution(cid, slug)
        if result is None:
            continue  # Not resolved yet

        # Determine if our prediction was correct
        our_pick = pred.get("pick", "")
        winner = result.get("winner", "")
        correct = our_pick.lower() == winner.lower() if winner else None

        # Calculate simulated P&L
        edge = abs(pred.get("edge", 0))
        market_price = pred.get("market_price", 0.5)
        our_prob = pred.get("our_prob", 0.5)
        bet_size = calculate_kelly_bet(our_prob, market_price)

        if correct:
            payout = bet_size * ((1 / market_price) - 1) * 0.98  # 2% fee
        elif correct is False:
            payout = -bet_size
        else:
            payout = 0

        resolved_entry = {
            "key": key,
            "title": pred.get("title", ""),
            "category": pred.get("category", ""),
            "our_pick": our_pick,
            "winner": winner,
            "correct": correct,
            "our_prob": our_prob,
            "market_price": market_price,
            "edge": pred.get("edge", 0),
            "confidence": pred.get("confidence", 0),
            "bet_size": round(bet_size, 2),
            "pnl": round(payout, 2),
            "source": pred.get("source", ""),
            "prediction_date": pred.get("prediction_date", ""),
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

        perf["resolved"].append(resolved_entry)
        resolved_keys.add(key)
        new_resolved += 1

        status = "WIN" if correct else ("LOSS" if correct is False else "???")
        print(f"  [{status}] {pred.get('title', '')[:50]} | {our_pick} vs {winner} | ${payout:+.0f}")

    # Calculate stats
    resolved = perf["resolved"]
    total = len(resolved)
    wins = sum(1 for r in resolved if r.get("correct") is True)
    losses = sum(1 for r in resolved if r.get("correct") is False)
    total_pnl = sum(r.get("pnl", 0) for r in resolved)

    # Per-category stats
    cat_stats = {}
    for r in resolved:
        cat = r.get("category", "other")
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "wins": 0, "pnl": 0, "bets": []}
        cat_stats[cat]["total"] += 1
        if r.get("correct"):
            cat_stats[cat]["wins"] += 1
        cat_stats[cat]["pnl"] += r.get("pnl", 0)
        cat_stats[cat]["bets"].append(r.get("pnl", 0))

    for cat in cat_stats:
        cs = cat_stats[cat]
        cs["accuracy"] = round(cs["wins"] / cs["total"] * 100, 1) if cs["total"] > 0 else 0
        cs["avg_pnl"] = round(cs["pnl"] / cs["total"], 2) if cs["total"] > 0 else 0
        del cs["bets"]  # Don't save raw bet list

    # Edge bucket analysis
    edge_buckets = {}
    for r in resolved:
        edge = abs(r.get("edge", 0))
        if edge >= 0.5:
            bucket = "50%+"
        elif edge >= 0.3:
            bucket = "30-50%"
        elif edge >= 0.1:
            bucket = "10-30%"
        elif edge >= 0.03:
            bucket = "3-10%"
        else:
            bucket = "<3%"
        if bucket not in edge_buckets:
            edge_buckets[bucket] = {"total": 0, "wins": 0, "pnl": 0}
        edge_buckets[bucket]["total"] += 1
        if r.get("correct"):
            edge_buckets[bucket]["wins"] += 1
        edge_buckets[bucket]["pnl"] += r.get("pnl", 0)

    for bucket in edge_buckets:
        eb = edge_buckets[bucket]
        eb["accuracy"] = round(eb["wins"] / eb["total"] * 100, 1) if eb["total"] > 0 else 0

    # Win/loss streaks
    streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for r in sorted(resolved, key=lambda x: x.get("resolved_at", "")):
        if r.get("correct"):
            streak = max(streak, 0) + 1
            max_win_streak = max(max_win_streak, streak)
        elif r.get("correct") is False:
            streak = min(streak, 0) - 1
            max_loss_streak = max(max_loss_streak, abs(streak))

    # Daily P&L tracking
    daily = {}
    for r in resolved:
        d = r.get("resolved_at", "")[:10]
        if d:
            if d not in daily:
                daily[d] = {"pnl": 0, "trades": 0, "wins": 0}
            daily[d]["pnl"] += r.get("pnl", 0)
            daily[d]["trades"] += 1
            if r.get("correct"):
                daily[d]["wins"] += 1

    perf["stats"] = {
        "total_predictions": total,
        "wins": wins,
        "losses": losses,
        "accuracy": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total, 2) if total > 0 else 0,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "roi": round(total_pnl / 50000 * 100, 2) if total > 0 else 0,
        "category_stats": cat_stats,
        "edge_buckets": edge_buckets,
    }
    perf["daily_pnl"] = [
        {"date": d, **v} for d, v in sorted(daily.items())
    ]
    perf["last_check"] = datetime.now(timezone.utc).isoformat()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total resolved: {total}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Accuracy: {perf['stats']['accuracy']:.1f}%")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  ROI: {perf['stats']['roi']:+.2f}%")
    print(f"  Win streak: {max_win_streak} | Loss streak: {max_loss_streak}")

    if cat_stats:
        print(f"\n  Per-Category:")
        for cat, cs in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
            print(f"    {cat:15s}: {cs['accuracy']:5.1f}% ({cs['wins']}/{cs['total']}) P&L ${cs['pnl']:+,.0f}")

    if edge_buckets:
        print(f"\n  By Edge Size:")
        for bucket in ["50%+", "30-50%", "10-30%", "3-10%", "<3%"]:
            if bucket in edge_buckets:
                eb = edge_buckets[bucket]
                print(f"    {bucket:8s}: {eb['accuracy']:5.1f}% ({eb['wins']}/{eb['total']}) P&L ${eb['pnl']:+,.0f}")

    if new_resolved:
        print(f"\n  New resolutions this run: {new_resolved}")

    save_performance(perf)
    print(f"\n  Saved to {PERF_FILE}")

    return perf


if __name__ == "__main__":
    run_tracker()
