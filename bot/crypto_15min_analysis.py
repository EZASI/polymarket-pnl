"""
15-Minute Crypto Market Analysis (Using Trade Data)

Analyzes:
1. Midpoint Cross frequency (price crossing 50c)
2. Mean Reversion Score (5c+ fluctuation from open)
3. High-Low range per 15-min interval
4. Bid-Ask spread estimation
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


def load_trade_data(trade_file: str) -> List[Dict]:
    """Load trade data with timestamps."""
    trades = []
    try:
        with open(trade_file, 'r') as f:
            content = f.read().strip()
            parts = content.replace('} {', '}\n{').split('\n')
            for part in parts:
                if part.strip():
                    try:
                        data = json.loads(part.strip())
                        ts = data.get('timestamp', data.get('t', 0))
                        price = float(data.get('price', 0))
                        if ts > 0 and 0 < price < 1:
                            trades.append({
                                'ts': ts,
                                'price': price,
                                'size': float(data.get('size', 0)),
                                'side': data.get('side', ''),
                                'outcome': data.get('outcome', '')
                            })
                    except:
                        continue
    except:
        pass
    return trades


def analyze_15min_intervals(trades: List[Dict]) -> Dict:
    """
    Analyze 15-minute intervals for:
    - Midpoint crosses (50c)
    - Mean reversion (5c+ from open)
    - High-Low range
    """
    if len(trades) < 5:
        return None

    trades = sorted(trades, key=lambda x: x['ts'])

    INTERVAL = 900  # 15 minutes
    start_ts = trades[0]['ts']

    # Group trades by 15-min interval
    intervals = defaultdict(list)
    for t in trades:
        interval_idx = (t['ts'] - start_ts) // INTERVAL
        intervals[interval_idx].append(t)

    # Analyze each interval
    interval_stats = []

    for idx, interval_trades in intervals.items():
        if len(interval_trades) < 2:
            continue

        prices = [t['price'] for t in interval_trades]
        open_price = prices[0]
        high = max(prices)
        low = min(prices)
        close = prices[-1]

        # High-Low range
        hl_range = high - low

        # Mean reversion: does price move 5c+ above AND below open?
        above_5c = high >= open_price + 0.05
        below_5c = low <= open_price - 0.05
        mean_reversion = above_5c and below_5c

        # Midpoint (50c) crosses
        crosses = 0
        prev = prices[0]
        for p in prices[1:]:
            if (prev < 0.50 and p >= 0.50) or (prev > 0.50 and p <= 0.50):
                crosses += 1
            prev = p

        # Time span of interval
        time_span_min = (interval_trades[-1]['ts'] - interval_trades[0]['ts']) / 60

        interval_stats.append({
            'interval': idx,
            'trades': len(interval_trades),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'hl_range': hl_range,
            'mean_reversion': mean_reversion,
            'midpoint_crosses': crosses,
            'time_span_min': time_span_min,
            'above_5c': above_5c,
            'below_5c': below_5c
        })

    if not interval_stats:
        return None

    # Aggregate statistics
    total_intervals = len(interval_stats)
    intervals_with_3plus_crosses = sum(1 for i in interval_stats if i['midpoint_crosses'] >= 3)
    intervals_with_mean_reversion = sum(1 for i in interval_stats if i['mean_reversion'])

    avg_hl_range = sum(i['hl_range'] for i in interval_stats) / total_intervals
    avg_crosses = sum(i['midpoint_crosses'] for i in interval_stats) / total_intervals
    total_crosses = sum(i['midpoint_crosses'] for i in interval_stats)

    # Time-based metrics
    total_hours = total_intervals * 0.25  # Each interval is 0.25 hours
    crosses_per_hour = total_crosses / total_hours if total_hours > 0 else 0

    return {
        'total_intervals': total_intervals,
        'total_trades': len(trades),
        'intervals_3plus_crosses': intervals_with_3plus_crosses,
        'pct_3plus_crosses': intervals_with_3plus_crosses / total_intervals * 100 if total_intervals > 0 else 0,
        'intervals_mean_reversion': intervals_with_mean_reversion,
        'pct_mean_reversion': intervals_with_mean_reversion / total_intervals * 100 if total_intervals > 0 else 0,
        'avg_hl_range': avg_hl_range,
        'avg_crosses_per_interval': avg_crosses,
        'crosses_per_hour': crosses_per_hour,
        'total_hours': total_hours,
        'interval_details': interval_stats
    }


def calculate_spread_from_trades(trades: List[Dict]) -> float:
    """Estimate bid-ask spread from consecutive buy/sell trades."""
    if len(trades) < 10:
        return None

    trades = sorted(trades, key=lambda x: x['ts'])
    spreads = []

    for i in range(1, len(trades)):
        t1, t2 = trades[i-1], trades[i]

        # If consecutive trades are buy then sell (or vice versa) within 60 sec
        if t2['ts'] - t1['ts'] <= 60:
            if t1['side'] != t2['side'] and t1['side'] and t2['side']:
                spread = abs(t1['price'] - t2['price'])
                if spread > 0 and spread < 0.10:  # Reasonable spread
                    spreads.append(spread)

    if spreads:
        return sum(spreads) / len(spreads)
    return None


def analyze_all_markets(max_markets: int = 500) -> List[Dict]:
    """Analyze all markets."""

    results = []
    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')

        trade_files = list(market_dir.glob('trade/*.ndjson'))

        all_trades = []
        for tf in trade_files:
            all_trades.extend(load_trade_data(str(tf)))

        if len(all_trades) < 20:
            continue

        # Check if market spans the 50c midpoint
        prices = [t['price'] for t in all_trades]
        min_p, max_p = min(prices), max(prices)

        # Only include markets that cross 50c
        spans_midpoint = min_p < 0.50 < max_p

        # Analyze intervals
        stats = analyze_15min_intervals(all_trades)

        if stats is None:
            continue

        # Calculate spread
        spread = calculate_spread_from_trades(all_trades)

        # Mean reversion score: how often does it fluctuate 5c+ from open?
        mean_reversion_score = stats['pct_mean_reversion']

        results.append({
            'market_id': market_id[:42],
            'total_trades': len(all_trades),
            'total_intervals': stats['total_intervals'],
            'price_range': f"${min_p:.2f}-${max_p:.2f}",
            'spans_midpoint': spans_midpoint,

            # Midpoint crosses
            'avg_crosses_per_15min': stats['avg_crosses_per_interval'],
            'crosses_per_hour': stats['crosses_per_hour'],
            'intervals_3plus': stats['intervals_3plus_crosses'],
            'pct_3plus': stats['pct_3plus_crosses'],

            # Mean reversion
            'mean_reversion_score': mean_reversion_score,
            'intervals_mean_reversion': stats['intervals_mean_reversion'],

            # Range
            'avg_hl_range': stats['avg_hl_range'],

            # Spread
            'est_spread': spread,
        })

    return results


def main():
    print("=" * 110)
    print("15-MINUTE CRYPTO MARKET ANALYSIS")
    print("Mean Reversion Score, Midpoint Crosses, High-Low Range")
    print("=" * 110)
    print()

    print("Analyzing markets from trade data...")
    results = analyze_all_markets(max_markets=800)

    # Filter for markets that span midpoint
    midpoint_markets = [r for r in results if r['spans_midpoint']]

    print(f"\nTotal markets analyzed: {len(results)}")
    print(f"Markets spanning 50c midpoint: {len(midpoint_markets)}")

    # 1. TOP 5 BY MIDPOINT CROSS FREQUENCY
    print("\n" + "=" * 110)
    print("TOP 5 MARKETS BY MIDPOINT (50c) CROSS FREQUENCY")
    print("Markets where price crosses 50c most frequently")
    print("=" * 110)

    midpoint_markets.sort(key=lambda x: x['crosses_per_hour'], reverse=True)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Crosses/Hr':<12} | {'Crosses/15m':<12} | {'3+ Intervals':<12} | {'Avg H-L':<10}")
    print("-" * 110)

    for i, m in enumerate(midpoint_markets[:5], 1):
        print(f"{i:<3} | {m['market_id']:<44} | {m['crosses_per_hour']:<12.2f} | "
              f"{m['avg_crosses_per_15min']:<12.2f} | {m['intervals_3plus']:<12} | {m['avg_hl_range']:.2%}")

    # 2. TOP 5 BY MEAN REVERSION SCORE
    print("\n" + "=" * 110)
    print("TOP 5 MARKETS BY MEAN REVERSION SCORE")
    print("Markets where price fluctuates 5c+ above AND below opening price within 15-min")
    print("=" * 110)

    midpoint_markets.sort(key=lambda x: x['mean_reversion_score'], reverse=True)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'MR Score':<10} | {'MR Intervals':<14} | {'Avg H-L':<10} | {'Crosses/Hr':<12}")
    print("-" * 110)

    for i, m in enumerate(midpoint_markets[:5], 1):
        print(f"{i:<3} | {m['market_id']:<44} | {m['mean_reversion_score']:<10.1f}% | "
              f"{m['intervals_mean_reversion']:<14} | {m['avg_hl_range']:.2%}   | {m['crosses_per_hour']:<12.2f}")

    # 3. HIGH-LOW RANGE ANALYSIS
    print("\n" + "=" * 110)
    print("TOP 5 MARKETS BY AVERAGE HIGH-LOW RANGE (15-min intervals)")
    print("Higher range = more trading opportunity")
    print("=" * 110)

    midpoint_markets.sort(key=lambda x: x['avg_hl_range'], reverse=True)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Avg H-L':<12} | {'Crosses/Hr':<12} | {'MR Score':<10} | {'Spread':<10}")
    print("-" * 110)

    for i, m in enumerate(midpoint_markets[:5], 1):
        spread_str = f"${m['est_spread']:.4f}" if m['est_spread'] else "N/A"
        print(f"{i:<3} | {m['market_id']:<44} | {m['avg_hl_range']:.2%}      | "
              f"{m['crosses_per_hour']:<12.2f} | {m['mean_reversion_score']:<10.1f}% | {spread_str}")

    # 4. LOWEST SPREAD (BEST FOR TRADING)
    print("\n" + "=" * 110)
    print("TOP 5 MARKETS BY LOWEST ESTIMATED SPREAD")
    print("Lower spread = better for arbitrage")
    print("=" * 110)

    markets_with_spread = [m for m in midpoint_markets if m['est_spread'] and m['est_spread'] > 0]
    markets_with_spread.sort(key=lambda x: x['est_spread'])

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Est Spread':<12} | {'Crosses/Hr':<12} | {'Avg H-L':<10}")
    print("-" * 110)

    for i, m in enumerate(markets_with_spread[:5], 1):
        print(f"{i:<3} | {m['market_id']:<44} | ${m['est_spread']:<10.4f} | "
              f"{m['crosses_per_hour']:<12.2f} | {m['avg_hl_range']:.2%}")

    # 5. COMBINED BEST FOR ARBITRAGE
    print("\n" + "=" * 110)
    print("BEST COMBINED: High Crosses + High Mean Reversion + Low Spread")
    print("Ideal for spread-based arbitrage bot")
    print("=" * 110)

    # Score markets
    for m in midpoint_markets:
        score = 0
        # More crosses = better (max 40)
        score += min(m['crosses_per_hour'] * 5, 40)
        # Higher mean reversion = better (max 30)
        score += min(m['mean_reversion_score'] * 0.3, 30)
        # Lower spread = better (max 20)
        if m['est_spread'] and m['est_spread'] > 0:
            score += max(0, 20 - m['est_spread'] * 200)
        # More trades = more liquid (max 10)
        score += min(m['total_trades'] / 100, 10)
        m['arb_score'] = score

    midpoint_markets.sort(key=lambda x: x['arb_score'], reverse=True)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Score':<8} | {'Crosses/Hr':<12} | {'MR%':<8} | {'Spread':<10} | {'H-L':<10}")
    print("-" * 120)

    for i, m in enumerate(midpoint_markets[:10], 1):
        spread_str = f"${m['est_spread']:.4f}" if m['est_spread'] else "N/A"
        print(f"{i:<3} | {m['market_id']:<44} | {m['arb_score']:<8.1f} | "
              f"{m['crosses_per_hour']:<12.2f} | {m['mean_reversion_score']:<8.1f}% | {spread_str:<10} | {m['avg_hl_range']:.2%}")

    # 6. SUMMARY TABLE
    print("\n" + "=" * 110)
    print("SUMMARY: TOP 5 FOR SPREAD-BASED ARBITRAGE BOT")
    print("=" * 110)

    for i, m in enumerate(midpoint_markets[:5], 1):
        spread_str = f"${m['est_spread']:.4f}" if m['est_spread'] else "estimated ~$0.01"
        print(f"""
{i}. Market: {m['market_id']}
   Price Range: {m['price_range']}

   MIDPOINT CROSSES:
   - {m['crosses_per_hour']:.1f} crosses per hour
   - {m['avg_crosses_per_15min']:.1f} per 15-min interval
   - {m['intervals_3plus']} intervals with 3+ crosses ({m['pct_3plus']:.1f}%)

   MEAN REVERSION:
   - Score: {m['mean_reversion_score']:.1f}%
   - {m['intervals_mean_reversion']} intervals with 5c+ fluctuation both directions

   TRADING METRICS:
   - Avg High-Low Range: {m['avg_hl_range']:.1%} per 15-min
   - Estimated Spread: {spread_str}
   - Total Trades: {m['total_trades']:,}
""")

    # Trading recommendations
    print("\n" + "=" * 110)
    print("BOT ENTRY POINT RECOMMENDATIONS")
    print("=" * 110)

    if midpoint_markets:
        best = midpoint_markets[0]
        print(f"""
FOR TOP MARKET ({best['market_id']}):

ENTRY STRATEGY:
1. Place BUY limit order at: $0.45 (5c below midpoint)
2. Place SELL limit order at: $0.55 (5c above midpoint)

EXPECTED FILLS:
- With {best['avg_hl_range']:.1%} avg range, expect fills when price swings
- {best['crosses_per_hour']:.1f} midpoint crosses per hour = opportunity
- Mean reversion in {best['mean_reversion_score']:.0f}% of intervals

FLEXIBLE ENTRY POINTS:
- Aggressive: $0.48-$0.52 (tight, more fills, less profit)
- Moderate: $0.45-$0.55 (balanced)
- Conservative: $0.40-$0.60 (fewer fills, more profit per fill)

PROFIT CALCULATION (per round trip):
- Buy at $0.45, Sell at $0.55 = $0.10 profit per share (20% return)
- Buy at $0.48, Sell at $0.52 = $0.04 profit per share (8% return)
- Minus ~1% fee = net ~19% or ~7%
""")


if __name__ == "__main__":
    main()
