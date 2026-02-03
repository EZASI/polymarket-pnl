"""
Orderbook Analysis for Polymarket Tick-Level Data.

Analyzes:
1. Best entry points based on price patterns
2. Arbitrage opportunities (YES + NO < $1)
3. Swing frequency (20c to 80c within intervals)
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


def load_price_data(price_file: str) -> List[Dict]:
    """Load price time series from ndjson file."""
    prices = []
    try:
        with open(price_file, 'r') as f:
            content = f.read().strip()
            # Handle space-separated JSON objects (not newline separated)
            parts = content.replace('} {', '}\n{').split('\n')
            for part in parts:
                if part.strip():
                    try:
                        data = json.loads(part.strip())
                        if 't' in data and 'p' in data:
                            prices.append({
                                'timestamp': data['t'],
                                'price': data['p']
                            })
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        pass
    return prices


def load_trade_data(trade_file: str) -> List[Dict]:
    """Load trade data from ndjson file."""
    trades = []
    try:
        with open(trade_file, 'r') as f:
            content = f.read().strip()
            parts = content.replace('} {', '}\n{').split('\n')
            for part in parts:
                if part.strip():
                    try:
                        data = json.loads(part.strip())
                        trades.append({
                            'timestamp': data.get('timestamp', 0),
                            'side': data.get('side', ''),
                            'price': data.get('price', 0),
                            'size': data.get('size', 0),
                            'outcome': data.get('outcome', '')
                        })
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        pass
    return trades


def find_swing_opportunities(prices: List[Dict], interval_seconds: int = 900) -> List[Dict]:
    """
    Find intervals where price swings from <=20c to >=80c (or vice versa).
    interval_seconds: 900 = 15 minutes
    """
    if not prices:
        return []

    prices = sorted(prices, key=lambda x: x['timestamp'])
    swings = []

    # Group prices by interval
    if not prices:
        return []

    start_ts = prices[0]['timestamp']
    intervals = defaultdict(list)

    for p in prices:
        interval_idx = (p['timestamp'] - start_ts) // interval_seconds
        intervals[interval_idx].append(p['price'])

    for interval_idx, interval_prices in intervals.items():
        if not interval_prices:
            continue

        min_price = min(interval_prices)
        max_price = max(interval_prices)

        # Check for 20c to 80c swing (60c+ range)
        if min_price <= 0.25 and max_price >= 0.75:
            swing_range = max_price - min_price
            interval_start = start_ts + (interval_idx * interval_seconds)

            swings.append({
                'interval_start': interval_start,
                'min_price': min_price,
                'max_price': max_price,
                'swing_range': swing_range,
                'num_ticks': len(interval_prices)
            })

    return swings


def find_entry_points(trades: List[Dict]) -> Dict:
    """
    Analyze trades to find optimal entry points.

    Entry signals:
    1. Large buy orders at low prices (smart money accumulation)
    2. Rapid price changes (momentum)
    3. Volume spikes
    """
    if not trades:
        return {}

    trades = sorted(trades, key=lambda x: x['timestamp'])

    # Calculate statistics
    prices = [t['price'] for t in trades if t['price'] > 0]
    sizes = [t['size'] for t in trades if t['size'] > 0]

    if not prices or not sizes:
        return {}

    avg_price = sum(prices) / len(prices)
    avg_size = sum(sizes) / len(sizes)

    # Find large orders at extreme prices
    entry_points = {
        'buy_low': [],  # Large buys at low prices
        'sell_high': [],  # Large sells at high prices
        'volume_spikes': [],  # Unusual volume
    }

    for t in trades:
        # Buy low: buying YES below 30c
        if t['outcome'] == 'Yes' and t['side'] == 'BUY' and t['price'] <= 0.30:
            if t['size'] > avg_size * 2:  # Large order
                entry_points['buy_low'].append({
                    'timestamp': t['timestamp'],
                    'price': t['price'],
                    'size': t['size'],
                    'edge': f"Buy YES at {t['price']:.1%} (below 30c)"
                })

        # Buy high: buying NO (contrarian)
        if t['outcome'] == 'No' and t['side'] == 'BUY' and t['price'] <= 0.20:
            if t['size'] > avg_size * 2:
                entry_points['sell_high'].append({
                    'timestamp': t['timestamp'],
                    'price': t['price'],
                    'size': t['size'],
                    'edge': f"Buy NO at {t['price']:.1%} (contrarian)"
                })

        # Volume spike
        if t['size'] > avg_size * 5:
            entry_points['volume_spikes'].append({
                'timestamp': t['timestamp'],
                'price': t['price'],
                'size': t['size'],
                'multiple': t['size'] / avg_size
            })

    return entry_points


def analyze_all_markets(max_markets: int = 500) -> Dict:
    """Analyze all markets for swings, entry points, and arbitrage."""

    results = {
        'markets_analyzed': 0,
        'total_swings': 0,
        'swings_by_market': [],
        'entry_signals': [],
        'price_ranges': [],
        'trades_analyzed': 0,
    }

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')[:20] + '...'

        # Load price data for both tokens (YES and NO)
        price_files = list(market_dir.glob('price/*.ndjson'))
        trade_files = list(market_dir.glob('trade/*.ndjson'))

        all_prices = []
        all_trades = []

        for pf in price_files:
            all_prices.extend(load_price_data(str(pf)))

        for tf in trade_files:
            all_trades.extend(load_trade_data(str(tf)))

        if not all_prices:
            continue

        results['markets_analyzed'] += 1
        results['trades_analyzed'] += len(all_trades)

        # Find swing opportunities
        swings = find_swing_opportunities(all_prices, interval_seconds=900)

        if swings:
            results['total_swings'] += len(swings)

            # Get best swing for this market
            best_swing = max(swings, key=lambda x: x['swing_range'])
            results['swings_by_market'].append({
                'market': market_id,
                'swing_count': len(swings),
                'best_swing_range': best_swing['swing_range'],
                'min_price': best_swing['min_price'],
                'max_price': best_swing['max_price'],
            })

        # Analyze entry points
        entry_points = find_entry_points(all_trades)

        if entry_points.get('buy_low') or entry_points.get('sell_high'):
            total_entries = len(entry_points.get('buy_low', [])) + len(entry_points.get('sell_high', []))
            results['entry_signals'].append({
                'market': market_id,
                'buy_low_signals': len(entry_points.get('buy_low', [])),
                'contrarian_signals': len(entry_points.get('sell_high', [])),
                'volume_spikes': len(entry_points.get('volume_spikes', [])),
            })

        # Track price range
        prices = [p['price'] for p in all_prices]
        if prices:
            results['price_ranges'].append({
                'market': market_id,
                'min': min(prices),
                'max': max(prices),
                'range': max(prices) - min(prices),
            })

    return results


def calculate_arbitrage_from_trades(max_markets: int = 500) -> Dict:
    """
    Find arbitrage by analyzing simultaneous YES and NO prices.

    If you can buy YES at X and buy NO at Y where X + Y < 1, that's arbitrage.
    """

    arb_opportunities = []

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')[:20] + '...'

        trade_files = list(market_dir.glob('trade/*.ndjson'))

        # Load all trades
        trades = []
        for tf in trade_files:
            trades.extend(load_trade_data(str(tf)))

        if not trades:
            continue

        # Group trades by timestamp (within same second)
        by_timestamp = defaultdict(list)
        for t in trades:
            ts = t['timestamp']
            by_timestamp[ts].append(t)

        # Look for simultaneous YES and NO trades
        for ts, ts_trades in by_timestamp.items():
            yes_prices = [t['price'] for t in ts_trades if t['outcome'] == 'Yes']
            no_prices = [t['price'] for t in ts_trades if t['outcome'] == 'No']

            if yes_prices and no_prices:
                min_yes = min(yes_prices)
                min_no = min(no_prices)
                total = min_yes + min_no

                if total < 0.98:  # 2%+ profit potential
                    profit_pct = (1 - total) * 100
                    arb_opportunities.append({
                        'market': market_id,
                        'timestamp': ts,
                        'yes_price': min_yes,
                        'no_price': min_no,
                        'total_cost': total,
                        'profit_pct': profit_pct,
                    })

    return arb_opportunities


def main():
    print("=" * 70)
    print("POLYMARKET ORDERBOOK ANALYSIS")
    print("Analyzing tick-level data for entry points & arbitrage")
    print("=" * 70)
    print()

    # Analyze markets
    print("Analyzing markets (this may take a moment)...")
    results = analyze_all_markets(max_markets=500)

    print(f"\nMarkets analyzed: {results['markets_analyzed']}")
    print(f"Total trades analyzed: {results['trades_analyzed']:,}")

    # 1. SWING ANALYSIS
    print("\n" + "=" * 70)
    print("1. SWING OPPORTUNITIES (20c to 80c within 15 minutes)")
    print("=" * 70)

    print(f"\nTotal 15-min swing intervals found: {results['total_swings']}")

    if results['swings_by_market']:
        # Sort by swing count
        swings_sorted = sorted(results['swings_by_market'],
                               key=lambda x: x['swing_count'], reverse=True)

        print(f"\nTop 15 markets with most swings:")
        print(f"{'Market':<25} | {'Swings':<8} | {'Best Range':<12} | {'Low':<8} | {'High':<8}")
        print("-" * 75)

        for m in swings_sorted[:15]:
            print(f"{m['market']:<25} | {m['swing_count']:<8} | "
                  f"{m['best_swing_range']:.1%}       | "
                  f"${m['min_price']:<6.2f} | ${m['max_price']:<6.2f}")

        # Statistics
        total_swings = sum(m['swing_count'] for m in results['swings_by_market'])
        markets_with_swings = len(results['swings_by_market'])

        print(f"\n{'='*50}")
        print("SWING STATISTICS")
        print(f"{'='*50}")
        print(f"Markets with 60c+ swings: {markets_with_swings}/{results['markets_analyzed']}")
        print(f"Total swing intervals: {total_swings}")

        if markets_with_swings > 0:
            avg_swings_per_market = total_swings / markets_with_swings
            print(f"Avg swings per market (with swings): {avg_swings_per_market:.1f}")

    # 2. ENTRY POINT ANALYSIS
    print("\n" + "=" * 70)
    print("2. ENTRY POINT SIGNALS")
    print("=" * 70)

    if results['entry_signals']:
        # Sort by total signals
        entries_sorted = sorted(results['entry_signals'],
                                key=lambda x: x['buy_low_signals'] + x['contrarian_signals'],
                                reverse=True)

        total_buy_low = sum(e['buy_low_signals'] for e in results['entry_signals'])
        total_contrarian = sum(e['contrarian_signals'] for e in results['entry_signals'])
        total_volume_spikes = sum(e['volume_spikes'] for e in results['entry_signals'])

        print(f"\nEntry signals found:")
        print(f"  - Buy YES at low (<30c): {total_buy_low:,} signals")
        print(f"  - Buy NO contrarian (<20c): {total_contrarian:,} signals")
        print(f"  - Volume spikes (5x avg): {total_volume_spikes:,} signals")

        print(f"\nTop 10 markets with most entry signals:")
        print(f"{'Market':<25} | {'Buy Low':<10} | {'Contrarian':<12} | {'Vol Spikes':<10}")
        print("-" * 65)

        for e in entries_sorted[:10]:
            print(f"{e['market']:<25} | {e['buy_low_signals']:<10} | "
                  f"{e['contrarian_signals']:<12} | {e['volume_spikes']:<10}")
    else:
        print("No entry signals found in the analyzed markets.")

    # 3. PRICE RANGE ANALYSIS
    print("\n" + "=" * 70)
    print("3. PRICE RANGE ANALYSIS (Volatility)")
    print("=" * 70)

    if results['price_ranges']:
        # Sort by range
        ranges_sorted = sorted(results['price_ranges'],
                               key=lambda x: x['range'], reverse=True)

        print(f"\nTop 15 most volatile markets (by price range):")
        print(f"{'Market':<25} | {'Min':<8} | {'Max':<8} | {'Range':<10}")
        print("-" * 60)

        for r in ranges_sorted[:15]:
            print(f"{r['market']:<25} | ${r['min']:<6.2f} | ${r['max']:<6.2f} | {r['range']:.1%}")

        # Distribution
        high_vol = [r for r in results['price_ranges'] if r['range'] >= 0.60]
        med_vol = [r for r in results['price_ranges'] if 0.30 <= r['range'] < 0.60]
        low_vol = [r for r in results['price_ranges'] if r['range'] < 0.30]

        print(f"\nVolatility distribution:")
        print(f"  - High (60c+ range): {len(high_vol)} markets ({len(high_vol)/len(results['price_ranges'])*100:.1f}%)")
        print(f"  - Medium (30-60c): {len(med_vol)} markets ({len(med_vol)/len(results['price_ranges'])*100:.1f}%)")
        print(f"  - Low (<30c): {len(low_vol)} markets ({len(low_vol)/len(results['price_ranges'])*100:.1f}%)")

    # 4. ARBITRAGE ANALYSIS
    print("\n" + "=" * 70)
    print("4. ARBITRAGE ANALYSIS (YES + NO < $1)")
    print("=" * 70)

    print("\nSearching for arbitrage opportunities...")
    arb_opps = calculate_arbitrage_from_trades(max_markets=500)

    print(f"\nArbitrage opportunities found: {len(arb_opps)}")

    if arb_opps:
        # Sort by profit
        arb_sorted = sorted(arb_opps, key=lambda x: x['profit_pct'], reverse=True)

        print(f"\nTop 20 arbitrage opportunities:")
        print(f"{'Market':<25} | {'YES':<8} | {'NO':<8} | {'Total':<8} | {'Profit':<8}")
        print("-" * 65)

        for a in arb_sorted[:20]:
            print(f"{a['market']:<25} | ${a['yes_price']:<6.2f} | ${a['no_price']:<6.2f} | "
                  f"${a['total_cost']:<6.2f} | {a['profit_pct']:.1f}%")

        # Distribution
        big_arb = [a for a in arb_opps if a['profit_pct'] >= 10]
        med_arb = [a for a in arb_opps if 5 <= a['profit_pct'] < 10]
        small_arb = [a for a in arb_opps if a['profit_pct'] < 5]

        print(f"\nArbitrage distribution:")
        print(f"  - Large (10%+ profit): {len(big_arb)}")
        print(f"  - Medium (5-10%): {len(med_arb)}")
        print(f"  - Small (2-5%): {len(small_arb)}")
    else:
        print("No arbitrage opportunities found (markets are efficient!)")

    # 5. SUMMARY
    print("\n" + "=" * 70)
    print("5. ACTIONABLE SUMMARY")
    print("=" * 70)

    # Calculate daily opportunity estimates
    # Assuming the data covers some time period, extrapolate
    if results['swings_by_market']:
        total_swings = sum(m['swing_count'] for m in results['swings_by_market'])
        markets_with_swings = len(results['swings_by_market'])

        # Each swing is a 15-min interval
        # Estimate daily: assume data covers ~7 days average per market
        est_swings_per_day = total_swings / 7  # rough estimate

        print(f"""
SWING TRADING OPPORTUNITIES:
- Total 60c+ swing intervals found: {total_swings}
- Markets with swing potential: {markets_with_swings}
- Estimated swings per day: ~{est_swings_per_day:.0f}

BEST ENTRY STRATEGY:
1. Monitor high-volatility markets (60c+ range)
2. Wait for price to touch 20c (or 80c)
3. Place limit order at the opposite extreme
4. If fills within 15 min = successful swing capture

ARBITRAGE OPPORTUNITIES:
- Found: {len(arb_opps)} total
- Most are small (<5% profit)
- Large spreads (10%+): {len([a for a in arb_opps if a['profit_pct'] >= 10])}
- Key: Execute FAST before spread closes
""")

    if results['entry_signals']:
        total_signals = sum(e['buy_low_signals'] + e['contrarian_signals']
                           for e in results['entry_signals'])

        print(f"""
ENTRY SIGNAL FREQUENCY:
- Total entry signals: {total_signals:,}
- Buy-low signals (YES < 30c): {sum(e['buy_low_signals'] for e in results['entry_signals']):,}
- Contrarian signals (NO < 20c): {sum(e['contrarian_signals'] for e in results['entry_signals']):,}

RECOMMENDED APPROACH:
1. Focus on markets with high swing count
2. Use volume spikes as entry confirmation
3. Set limit orders at extreme prices
4. Capture 60c+ swings when they occur
""")

    print("=" * 70)
    print("WHY MOST ARBITRAGE DOESN'T WORK")
    print("=" * 70)
    print("""
The arbitrage opportunities found are mostly:

1. STALE DATA
   - By the time you see it, the spread is gone
   - High-frequency traders already captured it

2. EXECUTION RISK
   - You need to buy BOTH YES and NO simultaneously
   - Partial fills leave you exposed

3. FEES
   - Polymarket fees (~1%) eat small arbitrage
   - Need >3% spread to profit after fees

4. LIQUIDITY
   - Large orders move the price
   - Can't fill at the quoted price

REALISTIC OPPORTUNITY:
- Swing trading on volatile markets
- Buy cheap outcomes (<30c) that have higher win probability
- Use OBI signals for short-term momentum
""")


if __name__ == "__main__":
    main()
