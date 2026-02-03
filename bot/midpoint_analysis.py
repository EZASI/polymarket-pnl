"""
15-Minute Crypto Market Analysis

Analyzes:
1. Midpoint Cross frequency (price crossing 50c)
2. Average Bid-Ask Spread
3. Order Book Depth at 45c and 55c marks
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


def load_price_data(price_file: str) -> List[Dict]:
    """Load price time series from ndjson file."""
    prices = []
    try:
        with open(price_file, 'r') as f:
            content = f.read().strip()
            parts = content.replace('} {', '}\n{').split('\n')
            for part in parts:
                if part.strip():
                    try:
                        data = json.loads(part.strip())
                        if 't' in data and 'p' in data:
                            prices.append({
                                'timestamp': data['t'],
                                'price': float(data['p'])
                            })
                    except:
                        continue
    except:
        pass
    return prices


def load_book_data(book_file: str) -> List[Dict]:
    """Load order book snapshots from ndjson file."""
    books = []
    try:
        with open(book_file, 'r') as f:
            content = f.read().strip()
            parts = content.replace('} {', '}\n{').split('\n')
            for part in parts:
                if part.strip():
                    try:
                        data = json.loads(part.strip())
                        books.append(data)
                    except:
                        continue
    except:
        pass
    return books


def calculate_midpoint_crosses(prices: List[Dict], interval_seconds: int = 900) -> Dict:
    """
    Calculate how many times price crosses 50c within each 15-min interval.
    """
    if len(prices) < 2:
        return {'total_crosses': 0, 'intervals': 0, 'avg_crosses_per_interval': 0, 'intervals_with_3plus': 0}

    prices = sorted(prices, key=lambda x: x['timestamp'])

    # Group by 15-min intervals
    start_ts = prices[0]['timestamp']
    intervals = defaultdict(list)

    for p in prices:
        interval_idx = (p['timestamp'] - start_ts) // interval_seconds
        intervals[interval_idx].append(p['price'])

    total_crosses = 0
    intervals_with_3plus = 0

    for idx, interval_prices in intervals.items():
        if len(interval_prices) < 2:
            continue

        crosses = 0
        prev_price = interval_prices[0]

        for price in interval_prices[1:]:
            # Check if crossed 0.50
            if (prev_price < 0.50 and price >= 0.50) or (prev_price > 0.50 and price <= 0.50):
                crosses += 1
            prev_price = price

        total_crosses += crosses
        if crosses >= 3:
            intervals_with_3plus += 1

    num_intervals = len(intervals)
    avg_crosses = total_crosses / num_intervals if num_intervals > 0 else 0

    return {
        'total_crosses': total_crosses,
        'intervals': num_intervals,
        'avg_crosses_per_interval': avg_crosses,
        'intervals_with_3plus': intervals_with_3plus,
        'pct_intervals_3plus': (intervals_with_3plus / num_intervals * 100) if num_intervals > 0 else 0
    }


def calculate_bid_ask_spread(books: List[Dict]) -> Dict:
    """Calculate average bid-ask spread from order book data."""
    spreads = []

    for book in books:
        # Try different field names
        bids = book.get('bids', book.get('bid', []))
        asks = book.get('asks', book.get('ask', []))

        if bids and asks:
            # Get best bid and best ask
            if isinstance(bids[0], dict):
                best_bid = max(float(b.get('price', b.get('p', 0))) for b in bids)
                best_ask = min(float(a.get('price', a.get('p', 1))) for a in asks)
            elif isinstance(bids[0], (list, tuple)):
                best_bid = max(float(b[0]) for b in bids)
                best_ask = min(float(a[0]) for a in asks)
            else:
                continue

            if best_ask > best_bid:
                spread = best_ask - best_bid
                spreads.append(spread)

    if not spreads:
        return {'avg_spread': None, 'min_spread': None, 'max_spread': None}

    return {
        'avg_spread': sum(spreads) / len(spreads),
        'min_spread': min(spreads),
        'max_spread': max(spreads),
        'num_samples': len(spreads)
    }


def calculate_depth_at_levels(books: List[Dict], levels: List[float] = [0.45, 0.55]) -> Dict:
    """Calculate order book depth at specific price levels."""
    depth_at_levels = {level: [] for level in levels}

    for book in books:
        bids = book.get('bids', book.get('bid', []))
        asks = book.get('asks', book.get('ask', []))

        for level in levels:
            # Sum liquidity within 2c of the level
            depth = 0

            for order_list in [bids, asks]:
                for order in order_list:
                    if isinstance(order, dict):
                        price = float(order.get('price', order.get('p', 0)))
                        size = float(order.get('size', order.get('s', order.get('amount', 0))))
                    elif isinstance(order, (list, tuple)):
                        price = float(order[0])
                        size = float(order[1]) if len(order) > 1 else 0
                    else:
                        continue

                    # Within 2c of target level
                    if abs(price - level) <= 0.02:
                        depth += size

            if depth > 0:
                depth_at_levels[level].append(depth)

    result = {}
    for level, depths in depth_at_levels.items():
        if depths:
            result[f'depth_{int(level*100)}c'] = {
                'avg': sum(depths) / len(depths),
                'min': min(depths),
                'max': max(depths),
                'samples': len(depths)
            }
        else:
            result[f'depth_{int(level*100)}c'] = {'avg': None, 'min': None, 'max': None, 'samples': 0}

    return result


def estimate_spread_from_prices(prices: List[Dict]) -> float:
    """Estimate spread from price volatility when no orderbook available."""
    if len(prices) < 10:
        return None

    # Calculate tick-to-tick changes
    prices = sorted(prices, key=lambda x: x['timestamp'])
    changes = []

    for i in range(1, len(prices)):
        change = abs(prices[i]['price'] - prices[i-1]['price'])
        if change > 0:
            changes.append(change)

    if not changes:
        return None

    # Median change approximates spread
    changes.sort()
    return changes[len(changes) // 2]


def analyze_all_markets(max_markets: int = 1000) -> List[Dict]:
    """Analyze all markets for midpoint crosses, spread, and depth."""

    results = []
    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')

        # Load price data
        price_files = list(market_dir.glob('price/*.ndjson'))
        book_files = list(market_dir.glob('book/*.ndjson'))

        all_prices = []
        all_books = []

        for pf in price_files:
            all_prices.extend(load_price_data(str(pf)))

        for bf in book_files:
            all_books.extend(load_book_data(str(bf)))

        if len(all_prices) < 20:
            continue

        # Calculate metrics
        cross_stats = calculate_midpoint_crosses(all_prices)

        # Only include markets with meaningful midpoint activity
        prices_list = [p['price'] for p in all_prices]
        min_p, max_p = min(prices_list), max(prices_list)

        # Market must have prices that span the 50c mark
        if not (min_p < 0.50 < max_p):
            continue

        # Calculate spread
        if all_books:
            spread_stats = calculate_bid_ask_spread(all_books)
            depth_stats = calculate_depth_at_levels(all_books)
        else:
            # Estimate from price data
            estimated_spread = estimate_spread_from_prices(all_prices)
            spread_stats = {
                'avg_spread': estimated_spread,
                'min_spread': None,
                'max_spread': None,
                'estimated': True
            }
            depth_stats = {
                'depth_45c': {'avg': None, 'samples': 0},
                'depth_55c': {'avg': None, 'samples': 0}
            }

        # Calculate price volatility around 50c
        prices_near_50 = [p for p in prices_list if 0.40 <= p <= 0.60]
        volatility = (max(prices_near_50) - min(prices_near_50)) if prices_near_50 else 0

        results.append({
            'market_id': market_id[:42],
            'total_ticks': len(all_prices),
            'price_range': f"${min_p:.2f}-${max_p:.2f}",
            'crosses_total': cross_stats['total_crosses'],
            'intervals': cross_stats['intervals'],
            'avg_crosses_per_15min': cross_stats['avg_crosses_per_interval'],
            'intervals_with_3plus': cross_stats['intervals_with_3plus'],
            'pct_3plus': cross_stats['pct_intervals_3plus'],
            'avg_spread': spread_stats.get('avg_spread'),
            'depth_45c': depth_stats.get('depth_45c', {}).get('avg'),
            'depth_55c': depth_stats.get('depth_55c', {}).get('avg'),
            'volatility_near_50': volatility,
            'has_orderbook': len(all_books) > 0
        })

    return results


def main():
    print("=" * 100)
    print("15-MINUTE CRYPTO MARKET ANALYSIS")
    print("Midpoint Crosses, Bid-Ask Spread, Order Book Depth")
    print("=" * 100)
    print()

    print("Analyzing markets (this may take a moment)...")
    results = analyze_all_markets(max_markets=1000)

    print(f"\nMarkets analyzed: {len(results)}")

    # Filter for markets with 3+ crosses per interval
    high_cross_markets = [r for r in results if r['avg_crosses_per_15min'] >= 3]

    print(f"Markets with avg 3+ midpoint crosses per 15-min: {len(high_cross_markets)}")

    # Sort by crosses per interval
    results.sort(key=lambda x: x['avg_crosses_per_15min'], reverse=True)

    # 1. TOP 5 MIDPOINT CROSS FREQUENCY
    print("\n" + "=" * 100)
    print("TOP 5 MARKETS BY MIDPOINT (50c) CROSS FREQUENCY")
    print("=" * 100)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Crosses/15min':<14} | {'Total':<8} | {'3+ Intervals':<12} | {'Spread':<10} | {'Price Range'}")
    print("-" * 140)

    for i, m in enumerate(results[:5], 1):
        spread_str = f"${m['avg_spread']:.3f}" if m['avg_spread'] else "N/A"
        print(f"{i:<3} | {m['market_id']:<44} | {m['avg_crosses_per_15min']:<14.2f} | "
              f"{m['crosses_total']:<8} | {m['intervals_with_3plus']:<12} | {spread_str:<10} | {m['price_range']}")

    # 2. BID-ASK SPREAD ANALYSIS
    print("\n" + "=" * 100)
    print("TOP 5 MARKETS BY LOWEST BID-ASK SPREAD (Best for Trading)")
    print("=" * 100)

    markets_with_spread = [r for r in results if r['avg_spread'] is not None and r['avg_spread'] > 0]
    markets_with_spread.sort(key=lambda x: x['avg_spread'])

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Avg Spread':<12} | {'Crosses/15min':<14} | {'Volatility':<12}")
    print("-" * 100)

    for i, m in enumerate(markets_with_spread[:5], 1):
        print(f"{i:<3} | {m['market_id']:<44} | ${m['avg_spread']:<10.4f} | "
              f"{m['avg_crosses_per_15min']:<14.2f} | {m['volatility_near_50']:.2%}")

    # Calculate average spread across all markets
    all_spreads = [m['avg_spread'] for m in markets_with_spread]
    if all_spreads:
        avg_spread = sum(all_spreads) / len(all_spreads)
        print(f"\nAverage spread across all markets: ${avg_spread:.4f}")

    # 3. ORDER BOOK DEPTH ANALYSIS
    print("\n" + "=" * 100)
    print("LOWEST ORDER BOOK DEPTH AT 45c AND 55c (Thin Liquidity = Easy to Move)")
    print("=" * 100)

    markets_with_depth = [r for r in results if r['depth_45c'] is not None or r['depth_55c'] is not None]

    if markets_with_depth:
        # Sort by combined depth (lowest first = thinnest books)
        for m in markets_with_depth:
            d45 = m['depth_45c'] if m['depth_45c'] else float('inf')
            d55 = m['depth_55c'] if m['depth_55c'] else float('inf')
            m['combined_depth'] = min(d45, d55)

        markets_with_depth.sort(key=lambda x: x['combined_depth'])

        print(f"\n{'#':<3} | {'Market ID':<44} | {'Depth @45c':<12} | {'Depth @55c':<12} | {'Crosses/15min':<14}")
        print("-" * 100)

        for i, m in enumerate(markets_with_depth[:5], 1):
            d45_str = f"${m['depth_45c']:.0f}" if m['depth_45c'] else "N/A"
            d55_str = f"${m['depth_55c']:.0f}" if m['depth_55c'] else "N/A"
            print(f"{i:<3} | {m['market_id']:<44} | {d45_str:<12} | {d55_str:<12} | {m['avg_crosses_per_15min']:<14.2f}")
    else:
        print("\nNo order book depth data available in the dataset.")
        print("The dataset contains price/trade data but limited order book snapshots.")

    # 4. COMBINED BEST MARKETS
    print("\n" + "=" * 100)
    print("BEST COMBINED: High Crosses + Low Spread + Active Trading")
    print("=" * 100)

    # Score markets
    for m in results:
        score = 0
        # Higher crosses = better
        score += min(m['avg_crosses_per_15min'] * 10, 50)
        # Lower spread = better (inverse)
        if m['avg_spread'] and m['avg_spread'] > 0:
            score += max(0, 30 - m['avg_spread'] * 100)
        # Higher tick count = more active
        score += min(m['total_ticks'] / 100, 20)
        m['combined_score'] = score

    results.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"\n{'#':<3} | {'Market ID':<44} | {'Score':<8} | {'Crosses/15m':<12} | {'Spread':<10} | {'Ticks'}")
    print("-" * 110)

    for i, m in enumerate(results[:10], 1):
        spread_str = f"${m['avg_spread']:.3f}" if m['avg_spread'] else "N/A"
        print(f"{i:<3} | {m['market_id']:<44} | {m['combined_score']:<8.1f} | "
              f"{m['avg_crosses_per_15min']:<12.2f} | {spread_str:<10} | {m['total_ticks']}")

    # 5. SUMMARY
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    high_cross = [m for m in results if m['avg_crosses_per_15min'] >= 3]

    print(f"""
MARKETS WITH 3+ MIDPOINT CROSSES PER 15-MIN: {len(high_cross)}

TOP 5 BY MIDPOINT CROSS FREQUENCY:
""")

    for i, m in enumerate(results[:5], 1):
        spread_info = f", spread ${m['avg_spread']:.3f}" if m['avg_spread'] else ""
        print(f"  {i}. {m['market_id']}")
        print(f"     - {m['avg_crosses_per_15min']:.1f} crosses per 15-min interval")
        print(f"     - {m['intervals_with_3plus']} intervals with 3+ crosses ({m['pct_3plus']:.1f}%)")
        print(f"     - Price range: {m['price_range']}{spread_info}")
        print()

    if markets_with_spread:
        print(f"""
SPREAD STATISTICS:
- Lowest spread: ${min(all_spreads):.4f}
- Average spread: ${avg_spread:.4f}
- Highest spread: ${max(all_spreads):.4f}

TRADING RECOMMENDATION:
Markets with high midpoint crosses and low spreads are ideal for:
- Mean reversion strategies (buy <50c, sell >50c)
- Scalping around the 50c equilibrium
- Capturing volatility in tight ranges
""")


if __name__ == "__main__":
    main()
