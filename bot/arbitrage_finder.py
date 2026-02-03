"""
Find actual arbitrage opportunities in Polymarket data.
If you can buy YES at X cents and NO at Y cents where X + Y < 100, that's free money.
"""

import csv
import json
from typing import Dict, List


def load_markets(filepath: str) -> List[Dict]:
    """Load market data."""
    markets = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            markets.append(row)
    return markets


def find_arbitrage(markets: List[Dict]):
    """Find arbitrage opportunities."""

    print("=" * 70)
    print("POLYMARKET ARBITRAGE FINDER")
    print("Looking for YES + NO < $1.00 opportunities")
    print("=" * 70)
    print()

    opportunities = []

    for m in markets:
        try:
            best_bid = float(m.get('bestBid', 0) or 0)
            best_ask = float(m.get('bestAsk', 1) or 1)
            volume = float(m.get('volume', 0) or 0)
            liquidity = float(m.get('liquidity', 0) or 0)

            # Parse outcome prices
            prices_str = m.get('outcomePrices', '[]')
            try:
                prices = json.loads(prices_str) if prices_str else []
                prices = [float(p) for p in prices] if prices else []
            except:
                prices = []

            if not prices or len(prices) < 2:
                continue

            yes_price = prices[0]
            no_price = prices[1] if len(prices) > 1 else (1 - yes_price)

            # Check for arbitrage: can we buy YES and NO for less than $1?
            # YES costs: best_ask (or yes_price as proxy)
            # NO costs: 1 - best_bid (or no_price)

            # Method 1: Using bid/ask spread
            if best_bid > 0 and best_ask > 0 and best_ask < 1:
                yes_cost = best_ask  # Buy YES at ask
                no_cost = 1 - best_bid  # Buy NO at (1 - bid)
                total_cost = yes_cost + no_cost

                if total_cost < 1.0:
                    profit = 1.0 - total_cost
                    opportunities.append({
                        'question': m.get('question', '')[:50],
                        'yes_cost': yes_cost,
                        'no_cost': no_cost,
                        'total_cost': total_cost,
                        'profit': profit,
                        'profit_pct': profit / total_cost * 100,
                        'volume': volume,
                        'liquidity': liquidity,
                        'spread': best_ask - best_bid,
                    })

        except Exception as e:
            continue

    print(f"Analyzed markets: {len(markets)}")
    print(f"Arbitrage opportunities found: {len(opportunities)}")
    print()

    if opportunities:
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)

        print("=" * 70)
        print("TOP ARBITRAGE OPPORTUNITIES")
        print("=" * 70)

        print("\nTop 20 by profit %:")
        print(f"{'Profit %':<10} | {'Profit $':<10} | {'YES':<8} | {'NO':<8} | {'Volume':<12} | Question")
        print("-" * 100)

        for opp in opportunities[:20]:
            print(f"{opp['profit_pct']:>8.1f}% | ${opp['profit']:>8.4f} | "
                  f"${opp['yes_cost']:<6.2f} | ${opp['no_cost']:<6.2f} | "
                  f"${opp['volume']:>10,.0f} | {opp['question'][:30]}...")

        # Filter to high-volume opportunities
        high_vol = [o for o in opportunities if o['volume'] > 10000]
        print(f"\nHigh-volume opportunities (>$10k volume): {len(high_vol)}")

        if high_vol:
            print("\nTop 10 high-volume arbitrage:")
            for opp in high_vol[:10]:
                print(f"  {opp['profit_pct']:.1f}% profit | ${opp['profit']:.4f} per $1 | "
                      f"Vol: ${opp['volume']:,.0f} | {opp['question'][:35]}...")

        # Statistics
        print("\n" + "=" * 70)
        print("ARBITRAGE STATISTICS")
        print("=" * 70)

        profits = [o['profit'] for o in opportunities]
        profit_pcts = [o['profit_pct'] for o in opportunities]

        print(f"""
Total opportunities: {len(opportunities)}

Profit distribution:
  Average profit per $1: ${sum(profits)/len(profits):.4f}
  Max profit per $1: ${max(profits):.4f}
  Average profit %: {sum(profit_pcts)/len(profit_pcts):.2f}%

By profit size:
  >50% profit: {len([p for p in profit_pcts if p > 50])}
  20-50% profit: {len([p for p in profit_pcts if 20 <= p < 50])}
  10-20% profit: {len([p for p in profit_pcts if 10 <= p < 20])}
  5-10% profit: {len([p for p in profit_pcts if 5 <= p < 10])}
  1-5% profit: {len([p for p in profit_pcts if 1 <= p < 5])}
  <1% profit: {len([p for p in profit_pcts if p < 1])}
""")

        # The catch
        print("=" * 70)
        print("⚠️  WHY THESE ARBITRAGES EXIST")
        print("=" * 70)
        print("""
These arbitrage opportunities exist because:

1. RESOLVED/INACTIVE MARKETS
   - Many are already resolved (winner determined)
   - Orderbook is stale, no one trading

2. ILLIQUIDITY
   - Bid/ask may exist but with tiny volume
   - Actual execution would move the price

3. DATA SNAPSHOT
   - This is a point-in-time snapshot
   - Real-time prices may have changed

4. FEES
   - Polymarket fees eat into small arbitrage
   - Need >3% profit to overcome fees

REALISTIC ARBITRAGE:
- Filter to active, high-volume markets
- Need >3% profit margin
- Execute quickly before prices move
""")

    else:
        print("No arbitrage opportunities found in the data.")


if __name__ == "__main__":
    filepath = "/home/user/polymarket-pnl/bot/data/polymarket_markets.csv"
    markets = load_markets(filepath)
    find_arbitrage(markets)
