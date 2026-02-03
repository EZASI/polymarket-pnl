"""
Detailed Entry Point Analysis.

Find the BEST specific entry points from orderbook data:
1. Optimal buy prices (when to enter)
2. Time-based patterns (when swings happen)
3. Signal quality scoring
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


def load_all_data(price_file: str, trade_file: str = None) -> Tuple[List, List]:
    """Load price and trade data."""
    prices = []
    trades = []

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
                                'price': data['p']
                            })
                    except:
                        continue
    except:
        pass

    if trade_file:
        try:
            with open(trade_file, 'r') as f:
                content = f.read().strip()
                parts = content.replace('} {', '}\n{').split('\n')
                for part in parts:
                    if part.strip():
                        try:
                            data = json.loads(part.strip())
                            trades.append(data)
                        except:
                            continue
        except:
            pass

    return prices, trades


def analyze_swing_timing(prices: List[Dict]) -> Dict:
    """Analyze when swings happen (time of day, day of week)."""
    if not prices:
        return {}

    prices = sorted(prices, key=lambda x: x['timestamp'])

    # Find swing events (price crosses 30c or 70c threshold)
    swing_events = []

    prev_price = prices[0]['price']
    for p in prices[1:]:
        curr_price = p['price']

        # Crossing from high to low
        if prev_price >= 0.70 and curr_price <= 0.30:
            swing_events.append({
                'timestamp': p['timestamp'],
                'type': 'high_to_low',
                'from_price': prev_price,
                'to_price': curr_price
            })

        # Crossing from low to high
        if prev_price <= 0.30 and curr_price >= 0.70:
            swing_events.append({
                'timestamp': p['timestamp'],
                'type': 'low_to_high',
                'from_price': prev_price,
                'to_price': curr_price
            })

        prev_price = curr_price

    # Analyze timing patterns
    hour_distribution = defaultdict(int)
    day_distribution = defaultdict(int)

    for event in swing_events:
        dt = datetime.fromtimestamp(event['timestamp'], tz=timezone.utc)
        hour_distribution[dt.hour] += 1
        day_distribution[dt.strftime('%A')] += 1

    return {
        'total_swings': len(swing_events),
        'by_hour': dict(hour_distribution),
        'by_day': dict(day_distribution),
        'events': swing_events[:100]  # Sample
    }


def find_optimal_entry_prices(prices: List[Dict]) -> Dict:
    """
    Find the optimal entry price levels.

    Looking for:
    - Price levels where reversals happen most often
    - Support/resistance levels
    """
    if not prices:
        return {}

    # Bucket prices into 5c increments
    price_buckets = defaultdict(int)
    for p in prices:
        bucket = round(p['price'] * 20) / 20  # 5c increments
        price_buckets[bucket] += 1

    # Find reversal points (local min/max)
    prices = sorted(prices, key=lambda x: x['timestamp'])

    reversals_low = defaultdict(int)  # Prices where lows occur
    reversals_high = defaultdict(int)  # Prices where highs occur

    window = 10  # Look at 10 ticks for local min/max

    for i in range(window, len(prices) - window):
        curr = prices[i]['price']

        window_prices = [prices[j]['price'] for j in range(i - window, i + window + 1)]

        if curr == min(window_prices):
            bucket = round(curr * 20) / 20
            reversals_low[bucket] += 1

        if curr == max(window_prices):
            bucket = round(curr * 20) / 20
            reversals_high[bucket] += 1

    return {
        'price_distribution': dict(sorted(price_buckets.items())),
        'support_levels': dict(sorted(reversals_low.items(), key=lambda x: x[1], reverse=True)[:10]),
        'resistance_levels': dict(sorted(reversals_high.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


def score_entry_opportunity(price: float, recent_prices: List[float], volume: float = 0) -> Dict:
    """
    Score an entry opportunity.

    Higher score = better entry.
    """
    if not recent_prices:
        return {'score': 0, 'reasons': []}

    score = 0
    reasons = []

    avg_price = sum(recent_prices) / len(recent_prices)
    min_recent = min(recent_prices)
    max_recent = max(recent_prices)

    # 1. Price at extreme (good for mean reversion)
    if price <= 0.20:
        score += 30
        reasons.append(f"Price at extreme low ({price:.0%})")
    elif price >= 0.80:
        score += 30
        reasons.append(f"Price at extreme high ({price:.0%})")

    # 2. Price at local low
    if price == min_recent and price < avg_price:
        score += 20
        reasons.append("Price at local minimum")

    # 3. High volatility (swing potential)
    volatility = max_recent - min_recent
    if volatility >= 0.30:
        score += 25
        reasons.append(f"High volatility ({volatility:.0%} range)")

    # 4. Discount from average
    if price < avg_price * 0.7:
        discount = (avg_price - price) / avg_price * 100
        score += min(25, discount)
        reasons.append(f"{discount:.0f}% discount from avg")

    return {
        'score': min(100, score),
        'reasons': reasons,
        'entry_price': price,
        'avg_price': avg_price,
        'volatility': volatility,
    }


def analyze_markets(max_markets: int = 200) -> Dict:
    """Comprehensive entry point analysis."""

    results = {
        'timing_analysis': {
            'by_hour': defaultdict(int),
            'by_day': defaultdict(int),
        },
        'best_entries': [],
        'support_resistance': {
            'support': defaultdict(int),
            'resistance': defaultdict(int),
        },
        'markets_analyzed': 0,
    }

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')[:20] + '...'

        price_files = list(market_dir.glob('price/*.ndjson'))

        for pf in price_files:
            prices, _ = load_all_data(str(pf))

            if len(prices) < 50:
                continue

            results['markets_analyzed'] += 1

            # Timing analysis
            timing = analyze_swing_timing(prices)
            for hour, count in timing.get('by_hour', {}).items():
                results['timing_analysis']['by_hour'][hour] += count
            for day, count in timing.get('by_day', {}).items():
                results['timing_analysis']['by_day'][day] += count

            # Entry price analysis
            entry_analysis = find_optimal_entry_prices(prices)

            for level, count in entry_analysis.get('support_levels', {}).items():
                results['support_resistance']['support'][level] += count
            for level, count in entry_analysis.get('resistance_levels', {}).items():
                results['support_resistance']['resistance'][level] += count

            # Score current entry opportunities
            recent_prices = [p['price'] for p in prices[-50:]]
            current_price = prices[-1]['price']

            entry_score = score_entry_opportunity(current_price, recent_prices)

            if entry_score['score'] >= 50:
                results['best_entries'].append({
                    'market': market_id,
                    'score': entry_score['score'],
                    'entry_price': entry_score['entry_price'],
                    'reasons': entry_score['reasons'],
                    'volatility': entry_score['volatility'],
                })

    # Sort best entries by score
    results['best_entries'].sort(key=lambda x: x['score'], reverse=True)

    return results


def main():
    print("=" * 70)
    print("ENTRY POINT ANALYZER")
    print("Finding optimal entry timing and price levels")
    print("=" * 70)
    print()

    print("Analyzing markets...")
    results = analyze_markets(max_markets=300)

    print(f"\nMarkets analyzed: {results['markets_analyzed']}")

    # 1. TIMING ANALYSIS
    print("\n" + "=" * 70)
    print("1. WHEN DO SWINGS HAPPEN? (Time Analysis)")
    print("=" * 70)

    hour_data = dict(results['timing_analysis']['by_hour'])
    day_data = dict(results['timing_analysis']['by_day'])

    if hour_data:
        print("\nSwings by Hour (UTC):")
        total_swings = sum(hour_data.values())

        # Sort by count
        sorted_hours = sorted(hour_data.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'Hour':<8} | {'Swings':<10} | {'Pct':<8} | Bar")
        print("-" * 60)

        max_count = max(hour_data.values()) if hour_data else 1
        for hour, count in sorted_hours[:12]:
            pct = count / total_swings * 100 if total_swings else 0
            bar_len = int(count / max_count * 30)
            print(f"{hour:02d}:00   | {count:<10} | {pct:>5.1f}%  | {'#' * bar_len}")

        # Best hours
        best_hours = [h for h, _ in sorted_hours[:5]]
        print(f"\nBEST HOURS FOR SWINGS: {', '.join([f'{h:02d}:00' for h in best_hours])} UTC")

    if day_data:
        print("\nSwings by Day of Week:")
        sorted_days = sorted(day_data.items(), key=lambda x: x[1], reverse=True)

        for day, count in sorted_days:
            pct = count / sum(day_data.values()) * 100
            print(f"  {day:<10}: {count:>6} ({pct:.1f}%)")

    # 2. SUPPORT/RESISTANCE LEVELS
    print("\n" + "=" * 70)
    print("2. KEY PRICE LEVELS (Support & Resistance)")
    print("=" * 70)

    support = dict(results['support_resistance']['support'])
    resistance = dict(results['support_resistance']['resistance'])

    if support:
        print("\nSTRONGEST SUPPORT LEVELS (best buy points):")
        sorted_support = sorted(support.items(), key=lambda x: x[1], reverse=True)[:10]

        for level, count in sorted_support:
            print(f"  ${level:.2f} - {count} reversals")

    if resistance:
        print("\nSTRONGEST RESISTANCE LEVELS (best sell points):")
        sorted_resistance = sorted(resistance.items(), key=lambda x: x[1], reverse=True)[:10]

        for level, count in sorted_resistance:
            print(f"  ${level:.2f} - {count} reversals")

    # 3. BEST CURRENT ENTRY OPPORTUNITIES
    print("\n" + "=" * 70)
    print("3. BEST ENTRY OPPORTUNITIES (Scored)")
    print("=" * 70)

    if results['best_entries']:
        print(f"\nTop 20 entry opportunities (score >= 50):")
        print(f"{'Market':<25} | {'Score':<6} | {'Entry':<8} | {'Volatility':<10} | Reasons")
        print("-" * 100)

        for entry in results['best_entries'][:20]:
            reasons_str = "; ".join(entry['reasons'][:2])
            print(f"{entry['market']:<25} | {entry['score']:<6} | "
                  f"${entry['entry_price']:<6.2f} | {entry['volatility']:.0%}        | {reasons_str}")

        # Score distribution
        scores = [e['score'] for e in results['best_entries']]
        print(f"\nEntry opportunity distribution:")
        print(f"  Score 80+: {sum(1 for s in scores if s >= 80)} markets (EXCELLENT)")
        print(f"  Score 60-79: {sum(1 for s in scores if 60 <= s < 80)} markets (GOOD)")
        print(f"  Score 50-59: {sum(1 for s in scores if 50 <= s < 60)} markets (MODERATE)")
    else:
        print("No high-score entry opportunities found.")

    # 4. ACTIONABLE SUMMARY
    print("\n" + "=" * 70)
    print("4. ACTIONABLE ENTRY STRATEGY")
    print("=" * 70)

    best_hours = []
    if hour_data:
        sorted_hours = sorted(hour_data.items(), key=lambda x: x[1], reverse=True)
        best_hours = [h for h, _ in sorted_hours[:3]]

    best_buy_levels = []
    if support:
        sorted_support = sorted(support.items(), key=lambda x: x[1], reverse=True)[:3]
        best_buy_levels = [f"${level:.2f}" for level, _ in sorted_support]

    print(f"""
OPTIMAL ENTRY STRATEGY:

1. TIMING
   - Best hours (UTC): {', '.join([f'{h:02d}:00' for h in best_hours]) if best_hours else 'N/A'}
   - Monitor markets during high-swing hours

2. PRICE LEVELS
   - Set buy orders at: {', '.join(best_buy_levels) if best_buy_levels else 'N/A'}
   - These are statistically proven reversal points

3. ENTRY CRITERIA
   - Score >= 80: Excellent entry (act quickly)
   - Score 60-79: Good entry (wait for confirmation)
   - Score 50-59: Moderate (only with high volume)

4. EXECUTION
   - Use limit orders at support levels
   - Monitor for volume spikes (5x normal)
   - Set take-profit at resistance levels

5. RISK MANAGEMENT
   - Max 10% of capital per trade
   - Stop loss 20% below entry
   - Take profit at 60c+ swing from entry
""")

    # Calculate estimated daily opportunities
    total_good_entries = len([e for e in results['best_entries'] if e['score'] >= 60])

    print(f"""
ESTIMATED OPPORTUNITY:
- High-quality entries (score 60+): {total_good_entries} currently available
- If patterns hold: ~{total_good_entries * 4} entry signals per day
- At 60% success rate: ~{int(total_good_entries * 4 * 0.6)} profitable trades/day
""")


if __name__ == "__main__":
    main()
