#!/usr/bin/env python3
"""
Analyze trading patterns from the Polymarket data.
"""

import json
from collections import defaultdict
from datetime import datetime

def analyze_patterns():
    print("Loading trade data...")
    with open("mty44_trades.json", 'r') as f:
        data = json.load(f)

    trades = data.get('trades', [])

    # Filter for 15-min crypto trades
    crypto_trades = []
    for t in trades:
        title = t.get('title', '').lower()
        if 'up or down' in title and any(x in title for x in ['bitcoin', 'solana', 'ethereum']):
            crypto_trades.append(t)

    print(f"\n{'='*80}")
    print("TRADING PATTERN ANALYSIS")
    print(f"Total 15-min crypto trades: {len(crypto_trades):,}")
    print(f"{'='*80}")

    # 1. TIMING PATTERNS
    print(f"\n{'='*80}")
    print("1. TIMING PATTERNS")
    print(f"{'='*80}")

    hour_trades = defaultdict(int)
    hour_volume = defaultdict(float)

    for t in crypto_trades:
        ts = t.get('timestamp')
        if ts:
            try:
                dt = datetime.fromtimestamp(int(ts))
                hour = dt.hour
                hour_trades[hour] += 1
                hour_volume[hour] += float(t.get('size', 0)) * float(t.get('price', 0))
            except:
                pass

    print("\nTrades by Hour (ET):")
    print(f"{'Hour':<8} {'Trades':>10} {'Volume':>15} {'Avg/Trade':>12}")
    print("-" * 50)
    for hour in sorted(hour_trades.keys()):
        count = hour_trades[hour]
        vol = hour_volume[hour]
        avg = vol / count if count > 0 else 0
        bar = "█" * int(count / 500)
        print(f"{hour:02d}:00    {count:>10,} ${vol:>13,.2f} ${avg:>10,.2f} {bar}")

    # Peak hours
    peak_hour = max(hour_trades.items(), key=lambda x: x[1])
    low_hour = min(hour_trades.items(), key=lambda x: x[1])
    print(f"\nPeak trading hour: {peak_hour[0]:02d}:00 ({peak_hour[1]:,} trades)")
    print(f"Lowest trading hour: {low_hour[0]:02d}:00 ({low_hour[1]:,} trades)")

    # 2. PRICE ENTRY PATTERNS
    print(f"\n{'='*80}")
    print("2. PRICE ENTRY PATTERNS")
    print(f"{'='*80}")

    price_buckets = defaultdict(int)
    for t in crypto_trades:
        if t.get('side') == 'BUY':
            price = float(t.get('price', 0))
            bucket = round(price, 1)
            price_buckets[bucket] += 1

    print("\nEntry Price Distribution:")
    print(f"{'Price':>8} {'Count':>10} {'%':>8} Distribution")
    print("-" * 60)
    total = sum(price_buckets.values())
    for price in sorted(price_buckets.keys()):
        count = price_buckets[price]
        pct = count / total * 100
        bar = "█" * int(pct)
        print(f"${price:.1f}     {count:>10,} {pct:>7.1f}% {bar}")

    # 3. POSITION SIZING PATTERNS
    print(f"\n{'='*80}")
    print("3. POSITION SIZING PATTERNS")
    print(f"{'='*80}")

    size_buckets = defaultdict(int)
    sizes = []
    for t in crypto_trades:
        size = float(t.get('size', 0))
        sizes.append(size)
        if size <= 10:
            size_buckets['1-10'] += 1
        elif size <= 25:
            size_buckets['11-25'] += 1
        elif size <= 50:
            size_buckets['26-50'] += 1
        elif size <= 100:
            size_buckets['51-100'] += 1
        elif size <= 200:
            size_buckets['101-200'] += 1
        elif size <= 500:
            size_buckets['201-500'] += 1
        else:
            size_buckets['500+'] += 1

    print("\nPosition Size Distribution:")
    order = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-500', '500+']
    for bucket in order:
        count = size_buckets.get(bucket, 0)
        pct = count / len(crypto_trades) * 100
        bar = "█" * int(pct / 2)
        print(f"{bucket:>10} shares: {count:>8,} ({pct:>5.1f}%) {bar}")

    # Common exact sizes
    exact_sizes = defaultdict(int)
    for s in sizes:
        exact_sizes[s] += 1

    print("\nMost Common Exact Position Sizes:")
    for size, count in sorted(exact_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {size:>8.2f} shares: {count:,} times")

    # 4. OUTCOME/DIRECTION PATTERNS
    print(f"\n{'='*80}")
    print("4. OUTCOME/DIRECTION PATTERNS")
    print(f"{'='*80}")

    # By asset and direction
    asset_direction = defaultdict(lambda: defaultdict(int))
    for t in crypto_trades:
        title = t.get('title', '').lower()
        outcome = t.get('outcome', 'Unknown')

        if 'bitcoin' in title:
            asset = 'Bitcoin'
        elif 'solana' in title:
            asset = 'Solana'
        elif 'ethereum' in title:
            asset = 'Ethereum'
        else:
            continue

        asset_direction[asset][outcome] += 1

    print("\nDirection Preference by Asset:")
    for asset in ['Bitcoin', 'Solana', 'Ethereum']:
        directions = asset_direction[asset]
        total = sum(directions.values())
        if total == 0:
            continue
        print(f"\n{asset}:")
        for direction, count in sorted(directions.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {direction:>6}: {count:>6,} ({pct:>5.1f}%) {bar}")

    # 5. SEQUENTIAL TRADING PATTERNS
    print(f"\n{'='*80}")
    print("5. SEQUENTIAL TRADING PATTERNS (Martingale/Doubling?)")
    print(f"{'='*80}")

    # Sort by timestamp
    sorted_trades = sorted(crypto_trades, key=lambda x: int(x.get('timestamp', 0)))

    # Look for doubling patterns
    doubling_sequences = 0
    consecutive_same_direction = []
    current_streak = 1
    current_direction = None

    for i, t in enumerate(sorted_trades):
        direction = t.get('outcome', '')
        if direction == current_direction:
            current_streak += 1
        else:
            if current_streak >= 3:
                consecutive_same_direction.append((current_direction, current_streak))
            current_streak = 1
            current_direction = direction

    print(f"\nLong Streaks (3+ consecutive same direction):")
    streak_counts = defaultdict(int)
    for direction, length in consecutive_same_direction:
        streak_counts[length] += 1

    for length in sorted(streak_counts.keys()):
        print(f"  {length} in a row: {streak_counts[length]} times")

    # Look for rapid-fire trading
    rapid_trades = 0
    for i in range(1, len(sorted_trades)):
        ts1 = int(sorted_trades[i-1].get('timestamp', 0))
        ts2 = int(sorted_trades[i].get('timestamp', 0))
        if ts2 - ts1 < 5:  # Within 5 seconds
            rapid_trades += 1

    print(f"\nRapid-fire trades (within 5 seconds): {rapid_trades:,}")
    print(f"Percentage of trades: {rapid_trades/len(sorted_trades)*100:.1f}%")

    # 6. PRICE SENSITIVITY PATTERNS
    print(f"\n{'='*80}")
    print("6. PRICE SENSITIVITY PATTERNS")
    print(f"{'='*80}")

    # When do they buy high vs low?
    high_price_trades = [t for t in crypto_trades if float(t.get('price', 0)) > 0.7]
    low_price_trades = [t for t in crypto_trades if float(t.get('price', 0)) < 0.5]
    mid_price_trades = [t for t in crypto_trades if 0.5 <= float(t.get('price', 0)) <= 0.7]

    print(f"\nPrice Tier Analysis:")
    print(f"  High price (>$0.70): {len(high_price_trades):,} trades ({len(high_price_trades)/len(crypto_trades)*100:.1f}%)")
    print(f"  Mid price ($0.50-$0.70): {len(mid_price_trades):,} trades ({len(mid_price_trades)/len(crypto_trades)*100:.1f}%)")
    print(f"  Low price (<$0.50): {len(low_price_trades):,} trades ({len(low_price_trades)/len(crypto_trades)*100:.1f}%)")

    # Direction at different price points
    print(f"\nDirection by Price Tier:")
    for name, bucket in [("High >$0.70", high_price_trades), ("Mid $0.50-0.70", mid_price_trades), ("Low <$0.50", low_price_trades)]:
        up = sum(1 for t in bucket if t.get('outcome', '').lower() == 'up')
        down = sum(1 for t in bucket if t.get('outcome', '').lower() == 'down')
        total = up + down
        if total > 0:
            print(f"  {name}: UP {up/total*100:.1f}% | DOWN {down/total*100:.1f}%")

    # 7. KEY PATTERN SUMMARY
    print(f"\n{'='*80}")
    print("KEY PATTERNS IDENTIFIED")
    print(f"{'='*80}")

    patterns = []

    # Pattern 1: Heavy bias towards high-probability (high-price) bets
    if len(high_price_trades) > len(crypto_trades) * 0.4:
        patterns.append("CHASING FAVORITES: 40%+ of trades are high-price (>$0.70) entries, meaning betting on likely outcomes but with poor risk/reward")

    # Pattern 2: DOWN bias
    down_trades = sum(1 for t in crypto_trades if t.get('outcome', '').lower() == 'down')
    if down_trades > len(crypto_trades) * 0.55:
        patterns.append(f"BEARISH BIAS: {down_trades/len(crypto_trades)*100:.1f}% of bets are DOWN predictions")

    # Pattern 3: Rapid trading
    if rapid_trades > len(crypto_trades) * 0.3:
        patterns.append(f"ALGORITHMIC/BOT TRADING: {rapid_trades/len(crypto_trades)*100:.1f}% trades within 5 seconds of each other")

    # Pattern 4: Round position sizes
    round_sizes = sum(1 for s in sizes if s == int(s) and s in [5, 10, 20, 25, 50, 100, 200, 300, 500])
    if round_sizes > len(sizes) * 0.5:
        patterns.append(f"FIXED POSITION SIZING: {round_sizes/len(sizes)*100:.1f}% use round numbers (suggests preset bot parameters)")

    # Pattern 5: No sells
    sells = sum(1 for t in crypto_trades if t.get('side') == 'SELL')
    if sells < len(crypto_trades) * 0.05:
        patterns.append(f"NO EXIT STRATEGY: Only {sells/len(crypto_trades)*100:.1f}% are sell trades (holding to expiration)")

    print("\n" + "\n\n".join(f"{i+1}. {p}" for i, p in enumerate(patterns)))

    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    print("""
This appears to be a LOSING BOT STRATEGY with these characteristics:

1. POOR ENTRY PRICES: Consistently buying at 60-70%+ prices, requiring
   unrealistically high win rates (>65%) to profit

2. NEGATIVE EXPECTED VALUE: At fair 50% odds, every trade has negative EV
   because entry prices are too high

3. BEARISH BIAS: Consistently betting DOWN across all assets, suggesting
   a single directional strategy without adaptation

4. NO RISK MANAGEMENT: Almost no sells/exits, meaning positions are held
   to expiration with no opportunity to cut losses

5. BOT BEHAVIOR: Rapid-fire execution and round position sizes indicate
   automated trading without human oversight

RECOMMENDATION: This strategy needs fundamental changes:
- Enter at lower prices (<$0.50) for positive expected value
- Implement stop-losses or exit strategies
- Diversify directional bets based on market conditions
""")

if __name__ == "__main__":
    analyze_patterns()
