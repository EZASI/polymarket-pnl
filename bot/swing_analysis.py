"""
Analyze Polymarket data for high-swing intervals.
Find markets where price swings between 20c and 80c could occur.
"""

import csv
import json
from typing import Dict, List
from collections import defaultdict


def load_markets(filepath: str) -> List[Dict]:
    """Load market data."""
    markets = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            markets.append(row)

    return markets


def analyze_swing_potential(markets: List[Dict]):
    """Analyze markets for swing trading potential."""

    print("=" * 70)
    print("POLYMARKET SWING ANALYSIS")
    print("Finding high-volatility arbitrage opportunities")
    print("=" * 70)
    print()

    # Parse relevant fields
    analyzed = []

    for m in markets:
        try:
            best_bid = float(m.get('bestBid', 0) or 0)
            best_ask = float(m.get('bestAsk', 0) or 0)

            # Parse outcome prices
            prices_str = m.get('outcomePrices', '[]')
            try:
                prices = json.loads(prices_str) if prices_str else []
                prices = [float(p) for p in prices] if prices else []
            except:
                prices = []

            # Price changes
            one_day_change = abs(float(m.get('oneDayPriceChange', 0) or 0))
            one_hour_change = abs(float(m.get('oneHourPriceChange', 0) or 0))

            volume = float(m.get('volume', 0) or 0)
            liquidity = float(m.get('liquidity', 0) or 0)

            spread = best_ask - best_bid if best_ask > best_bid else 0

            if prices and volume > 0:
                analyzed.append({
                    'question': m.get('question', '')[:60],
                    'yes_price': prices[0] if prices else 0,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': spread,
                    'one_day_change': one_day_change,
                    'one_hour_change': one_hour_change,
                    'volume': volume,
                    'liquidity': liquidity,
                    'category': m.get('groupItemTitle', 'Unknown'),
                })
        except Exception as e:
            continue

    print(f"Analyzed {len(analyzed)} markets with price data")
    print()

    # 1. SPREAD ANALYSIS (immediate arbitrage)
    print("=" * 70)
    print("1. SPREAD ANALYSIS (Bid-Ask Spread)")
    print("=" * 70)

    spreads = [m['spread'] for m in analyzed if m['spread'] > 0]

    if spreads:
        print(f"\nMarkets with spread data: {len(spreads)}")
        print(f"Average spread: {sum(spreads)/len(spreads):.4f}")
        print(f"Max spread: {max(spreads):.4f}")

        # Count by spread size
        spread_buckets = {
            '0-5c': sum(1 for s in spreads if s < 0.05),
            '5-10c': sum(1 for s in spreads if 0.05 <= s < 0.10),
            '10-20c': sum(1 for s in spreads if 0.10 <= s < 0.20),
            '20-50c': sum(1 for s in spreads if 0.20 <= s < 0.50),
            '50c+': sum(1 for s in spreads if s >= 0.50),
        }

        print("\nSpread distribution:")
        for bucket, count in spread_buckets.items():
            pct = count / len(spreads) * 100
            print(f"  {bucket}: {count} markets ({pct:.1f}%)")

        # High spread markets (potential arb)
        high_spread = [m for m in analyzed if m['spread'] >= 0.20]
        print(f"\nMarkets with 20c+ spread: {len(high_spread)}")

        if high_spread:
            print("\nTop 10 highest spread markets:")
            high_spread.sort(key=lambda x: x['spread'], reverse=True)
            for m in high_spread[:10]:
                print(f"  Spread: {m['spread']:.2f} | Vol: ${m['volume']:,.0f} | {m['question'][:40]}...")

    # 2. VOLATILITY ANALYSIS (swing potential)
    print("\n" + "=" * 70)
    print("2. VOLATILITY ANALYSIS (Price Swings)")
    print("=" * 70)

    # Markets with high daily change
    volatile = [m for m in analyzed if m['one_day_change'] > 0]

    if volatile:
        daily_changes = [m['one_day_change'] for m in volatile]
        print(f"\nMarkets with daily change data: {len(volatile)}")
        print(f"Average daily change: {sum(daily_changes)/len(daily_changes):.1%}")
        print(f"Max daily change: {max(daily_changes):.1%}")

        # Count by volatility
        vol_buckets = {
            '0-5%': sum(1 for c in daily_changes if c < 0.05),
            '5-10%': sum(1 for c in daily_changes if 0.05 <= c < 0.10),
            '10-20%': sum(1 for c in daily_changes if 0.10 <= c < 0.20),
            '20-50%': sum(1 for c in daily_changes if 0.20 <= c < 0.50),
            '50%+': sum(1 for c in daily_changes if c >= 0.50),
        }

        print("\nDaily volatility distribution:")
        for bucket, count in vol_buckets.items():
            pct = count / len(volatile) * 100
            print(f"  {bucket}: {count} markets ({pct:.1f}%)")

        # High volatility markets
        high_vol = [m for m in volatile if m['one_day_change'] >= 0.30]
        print(f"\nMarkets with 30%+ daily swing: {len(high_vol)}")

    # 3. SWING POTENTIAL: 20c to 80c range
    print("\n" + "=" * 70)
    print("3. SWING POTENTIAL: 20c ↔ 80c Analysis")
    print("=" * 70)

    # Markets currently in the "swingable" range (20c-80c)
    swingable = [m for m in analyzed if 0.15 <= m['yes_price'] <= 0.85]
    extreme_low = [m for m in analyzed if m['yes_price'] < 0.20]
    extreme_high = [m for m in analyzed if m['yes_price'] > 0.80]

    print(f"\nCurrent price distribution:")
    print(f"  Price < 20c (extreme low): {len(extreme_low)} markets")
    print(f"  Price 20c-80c (mid-range): {len(swingable)} markets")
    print(f"  Price > 80c (extreme high): {len(extreme_high)} markets")

    # For a 60c swing (20c to 80c), we need markets that move a LOT
    # A market at 50c would need to move ±30c to hit both extremes

    # Estimate: how many markets have 60c+ daily range?
    # Using daily change as a proxy (this underestimates true range)
    big_swingers = [m for m in volatile if m['one_day_change'] >= 0.40]

    print(f"\nMarkets with 40%+ daily moves (potential 60c swingers): {len(big_swingers)}")

    if big_swingers:
        print("\nTop potential swing markets:")
        big_swingers.sort(key=lambda x: x['one_day_change'], reverse=True)
        for m in big_swingers[:10]:
            print(f"  Change: {m['one_day_change']:.0%} | Price: {m['yes_price']:.2f} | {m['question'][:35]}...")

    # 4. CRYPTO MARKET ANALYSIS
    print("\n" + "=" * 70)
    print("4. CRYPTO MARKET SWING ANALYSIS")
    print("=" * 70)

    crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'doge']
    crypto = [m for m in analyzed
              if any(kw in m['question'].lower() for kw in crypto_keywords)]

    print(f"\nCrypto markets found: {len(crypto)}")

    if crypto:
        crypto_volatile = [m for m in crypto if m['one_day_change'] > 0]
        if crypto_volatile:
            avg_change = sum(m['one_day_change'] for m in crypto_volatile) / len(crypto_volatile)
            print(f"Average daily change in crypto: {avg_change:.1%}")

            crypto_swings = [m for m in crypto_volatile if m['one_day_change'] >= 0.30]
            print(f"Crypto markets with 30%+ daily swing: {len(crypto_swings)}")

    # 5. ESTIMATE: How often does 20c-80c swing happen?
    print("\n" + "=" * 70)
    print("5. ESTIMATED SWING FREQUENCY")
    print("=" * 70)

    # Based on the data we have, estimate swing frequency
    # This is an ESTIMATE since we don't have intraday OHLC data

    total_markets = len(analyzed)
    markets_with_40pct_swing = len(big_swingers)

    # Assumption: if daily change is 40%+, there's ~30% chance of 60c swing intraday
    # This is conservative - actual swings may be higher

    estimated_60c_swings_per_day = markets_with_40pct_swing * 0.30

    # For 15-min intervals: 96 intervals per day
    # If a market swings 60c in a day, it might happen across multiple intervals
    # Estimate: such a market has 1-2 intervals with extreme swings

    estimated_per_15min = estimated_60c_swings_per_day / 96 * 2

    print(f"""
ESTIMATION (based on available data):

Data we have:
- Markets with 40%+ daily moves: {markets_with_40pct_swing}
- These are candidates for 60c+ swings

Estimated frequency:
- Markets with 60c swing per day: ~{estimated_60c_swings_per_day:.0f}
- Estimated 20c↔80c opportunities per day: ~{estimated_60c_swings_per_day:.0f}
- Per 15-minute interval: ~{estimated_per_15min:.1f}

⚠️  IMPORTANT CAVEATS:
1. This is an ESTIMATE - we don't have 15-min OHLC data
2. The Kaggle dataset is a SNAPSHOT, not time-series
3. True swing frequency requires tick-level data
4. High spreads make execution at both extremes difficult

To get accurate swing frequency, you would need:
- Polymarket API with historical price data
- Or real-time monitoring of orderbook changes
""")

    # 6. ACTIONABLE INSIGHT
    print("=" * 70)
    print("6. ACTIONABLE INSIGHT")
    print("=" * 70)

    print(f"""
WHAT THE DATA SHOWS:

1. High Spread Opportunities:
   - {len([m for m in analyzed if m['spread'] >= 0.20])} markets have 20c+ spread RIGHT NOW
   - This means you could potentially buy YES at Xc and NO at (100-X-spread)c
   - If spread > 50c, that's immediate arbitrage potential

2. Volatile Markets:
   - {len(big_swingers)} markets move 40%+ per day
   - These are your best candidates for swing trading

3. Crypto Specifically:
   - {len(crypto)} crypto markets
   - Typically more volatile than other categories

STRATEGY FOR FINDING 20c↔80c SWINGS:

1. Monitor high-volatility markets in real-time
2. Wait for price to touch one extreme (20c or 80c)
3. Place limit order at the other extreme
4. If it fills within 15 min = successful swing capture

ESTIMATED OPPORTUNITY:
- ~{estimated_60c_swings_per_day:.0f} potential 60c+ swings per day across all markets
- Not all are capturable (liquidity, timing, execution)
- Realistic capture rate: maybe 10-20% = {estimated_60c_swings_per_day * 0.15:.0f} trades/day
""")


if __name__ == "__main__":
    filepath = "/home/user/polymarket-pnl/bot/data/polymarket_markets.csv"
    markets = load_markets(filepath)
    analyze_swing_potential(markets)
