"""
Detailed Arbitrage Analysis - Every Single Trade

Shows:
1. Every arbitrage opportunity found
2. Exact entry price for YES and NO
3. How many shares to buy
4. Expected profit per trade
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


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
                            'price': float(data.get('price', 0)),
                            'size': float(data.get('size', 0)),
                            'outcome': data.get('outcome', ''),
                            'market_id': data.get('market_id', ''),
                            'conditionId': data.get('conditionId', ''),
                        })
                    except (json.JSONDecodeError, ValueError):
                        continue
    except Exception as e:
        pass
    return trades


def load_price_data(price_file: str) -> List[Dict]:
    """Load price time series."""
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
                                'price': float(data['p']),
                                'market_id': data.get('market_id', ''),
                            })
                    except:
                        continue
    except:
        pass
    return prices


def find_all_arbitrage_trades(max_markets: int = 1000) -> List[Dict]:
    """
    Find EVERY arbitrage opportunity where YES + NO < $1.

    Returns detailed list with entry points and share quantities.
    """

    all_arb_trades = []

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')

        trade_files = list(market_dir.glob('trade/*.ndjson'))

        # Load all trades for this market
        all_trades = []
        for tf in trade_files:
            all_trades.extend(load_trade_data(str(tf)))

        if not all_trades:
            continue

        # Group trades by timestamp (within same second)
        by_timestamp = defaultdict(list)
        for t in all_trades:
            ts = t['timestamp']
            by_timestamp[ts].append(t)

        # Look for simultaneous YES and NO trades
        for ts, ts_trades in by_timestamp.items():
            yes_trades = [t for t in ts_trades if t['outcome'] == 'Yes']
            no_trades = [t for t in ts_trades if t['outcome'] == 'No']

            if not yes_trades or not no_trades:
                continue

            # Find best prices (lowest ask for each)
            best_yes = min(yes_trades, key=lambda x: x['price'])
            best_no = min(no_trades, key=lambda x: x['price'])

            yes_price = best_yes['price']
            no_price = best_no['price']
            total_cost = yes_price + no_price

            # Arbitrage if total < 1
            if total_cost < 0.99:  # At least 1% profit
                profit_pct = (1 - total_cost) * 100

                # Calculate optimal shares
                # Budget: $1000 per arbitrage trade
                budget = 1000

                # To capture arbitrage: buy equal $ value of YES and NO
                # shares_yes * yes_price + shares_no * no_price = budget
                # For guaranteed profit: buy equal shares of both
                # Cost = shares * (yes_price + no_price)
                # Payout = shares * $1

                max_shares = budget / total_cost

                # Also consider available liquidity
                yes_liquidity = best_yes['size']
                no_liquidity = best_no['size']

                available_shares = min(yes_liquidity, no_liquidity, max_shares)

                if available_shares < 1:
                    continue

                actual_cost = available_shares * total_cost
                profit_per_trade = available_shares * (1 - total_cost)

                dt = datetime.fromtimestamp(ts, tz=timezone.utc)

                all_arb_trades.append({
                    'market_id': market_id[:42],
                    'timestamp': ts,
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'yes_price': yes_price,
                    'no_price': no_price,
                    'total_cost': total_cost,
                    'profit_pct': profit_pct,
                    'shares_to_buy': int(available_shares),
                    'yes_liquidity': yes_liquidity,
                    'no_liquidity': no_liquidity,
                    'investment': actual_cost,
                    'guaranteed_payout': available_shares,
                    'profit_usd': profit_per_trade,
                })

    return all_arb_trades


def find_all_entry_points(max_markets: int = 500) -> List[Dict]:
    """
    Find every optimal entry point from trade data.

    Entry signals:
    1. Large buys at low prices (accumulation)
    2. Price touching support levels
    3. Volume spikes
    """

    all_entries = []

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        market_id = market_dir.name.replace('market=', '')

        # Load price data
        price_files = list(market_dir.glob('price/*.ndjson'))
        trade_files = list(market_dir.glob('trade/*.ndjson'))

        all_prices = []
        all_trades = []

        for pf in price_files:
            all_prices.extend(load_price_data(str(pf)))

        for tf in trade_files:
            all_trades.extend(load_trade_data(str(tf)))

        if not all_prices or len(all_prices) < 20:
            continue

        # Sort by timestamp
        all_prices = sorted(all_prices, key=lambda x: x['timestamp'])
        all_trades = sorted(all_trades, key=lambda x: x['timestamp'])

        # Calculate statistics
        prices_list = [p['price'] for p in all_prices]
        avg_price = sum(prices_list) / len(prices_list)
        min_price = min(prices_list)
        max_price = max(prices_list)
        volatility = max_price - min_price

        if not all_trades:
            continue

        sizes = [t['size'] for t in all_trades if t['size'] > 0]
        if not sizes:
            continue

        avg_size = sum(sizes) / len(sizes)

        # Find entry signals
        for trade in all_trades:
            entry_signal = None
            score = 0
            reasons = []

            price = trade['price']
            size = trade['size']
            outcome = trade['outcome']
            side = trade['side']

            # Signal 1: Buy at extreme low (<20c for YES)
            if outcome == 'Yes' and side == 'BUY' and price <= 0.20:
                score += 40
                reasons.append(f"Buy YES at extreme low ({price:.0%})")
                entry_signal = 'BUY_LOW_YES'

            # Signal 2: Buy at extreme high NO (<20c for NO = YES >80c)
            if outcome == 'No' and side == 'BUY' and price <= 0.20:
                score += 40
                reasons.append(f"Buy NO at extreme low ({price:.0%})")
                entry_signal = 'BUY_LOW_NO'

            # Signal 3: Large order (smart money)
            if size > avg_size * 3:
                score += 25
                reasons.append(f"Large order ({size:.0f} shares, {size/avg_size:.1f}x avg)")

            # Signal 4: High volatility market
            if volatility >= 0.50:
                score += 15
                reasons.append(f"High volatility ({volatility:.0%} range)")

            # Signal 5: Price discount from average
            if price < avg_price * 0.5:
                discount = (avg_price - price) / avg_price * 100
                score += min(20, discount / 2)
                reasons.append(f"{discount:.0f}% discount from avg")

            if score >= 40 and entry_signal:
                dt = datetime.fromtimestamp(trade['timestamp'], tz=timezone.utc)

                # Calculate recommended position
                # Risk: max 5% of $10k portfolio = $500
                max_risk = 500
                recommended_shares = int(max_risk / price) if price > 0 else 0

                # Potential profit if price mean-reverts
                target_price = min(0.80, avg_price * 1.5)  # Conservative target
                potential_profit = (target_price - price) * recommended_shares

                all_entries.append({
                    'market_id': market_id[:42],
                    'timestamp': trade['timestamp'],
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'signal': entry_signal,
                    'outcome': outcome,
                    'entry_price': price,
                    'size_available': size,
                    'score': score,
                    'reasons': reasons,
                    'recommended_shares': recommended_shares,
                    'investment_usd': recommended_shares * price,
                    'target_price': target_price,
                    'potential_profit': potential_profit,
                    'avg_market_price': avg_price,
                    'market_volatility': volatility,
                })

    return all_entries


def main():
    print("=" * 100)
    print("DETAILED ARBITRAGE & ENTRY POINT ANALYSIS")
    print("Every single trade opportunity with exact shares to buy")
    print("=" * 100)
    print()

    # 1. ARBITRAGE TRADES
    print("Searching for arbitrage opportunities...")
    arb_trades = find_all_arbitrage_trades(max_markets=1000)

    print(f"\n{'='*100}")
    print(f"ARBITRAGE OPPORTUNITIES FOUND: {len(arb_trades)}")
    print(f"{'='*100}")

    if arb_trades:
        # Sort by profit
        arb_trades.sort(key=lambda x: x['profit_pct'], reverse=True)

        print(f"\n{'#':<5} | {'Market ID':<44} | {'YES':<7} | {'NO':<7} | {'Total':<7} | {'Profit%':<8} | {'Shares':<8} | {'Profit$':<10} | {'DateTime'}")
        print("-" * 160)

        for i, trade in enumerate(arb_trades, 1):
            print(f"{i:<5} | {trade['market_id']:<44} | "
                  f"${trade['yes_price']:<5.3f} | ${trade['no_price']:<5.3f} | "
                  f"${trade['total_cost']:<5.3f} | {trade['profit_pct']:>6.1f}% | "
                  f"{trade['shares_to_buy']:<8} | ${trade['profit_usd']:<8.2f} | "
                  f"{trade['datetime']}")

        # Summary statistics
        total_profit = sum(t['profit_usd'] for t in arb_trades)
        avg_profit = total_profit / len(arb_trades)

        print(f"\n{'='*100}")
        print("ARBITRAGE SUMMARY")
        print(f"{'='*100}")
        print(f"""
Total arbitrage opportunities: {len(arb_trades)}
Total potential profit: ${total_profit:,.2f}
Average profit per trade: ${avg_profit:.2f}

By profit tier:
  - 50%+ profit: {len([t for t in arb_trades if t['profit_pct'] >= 50])} trades
  - 20-50% profit: {len([t for t in arb_trades if 20 <= t['profit_pct'] < 50])} trades
  - 10-20% profit: {len([t for t in arb_trades if 10 <= t['profit_pct'] < 20])} trades
  - 5-10% profit: {len([t for t in arb_trades if 5 <= t['profit_pct'] < 10])} trades
  - 2-5% profit: {len([t for t in arb_trades if 2 <= t['profit_pct'] < 5])} trades
""")

        # Bot execution instructions
        print(f"{'='*100}")
        print("BOT EXECUTION INSTRUCTIONS")
        print(f"{'='*100}")
        print("""
HOW TO EXECUTE ARBITRAGE:

For each opportunity above:

1. SIMULTANEOUSLY buy:
   - [Shares] of YES at [YES Price]
   - [Shares] of NO at [NO Price]

2. WAIT for market resolution

3. GUARANTEED PAYOUT:
   - One of YES/NO will be worth $1.00
   - The other will be worth $0.00
   - Total payout = [Shares] × $1.00

4. PROFIT = Payout - Investment
   - Investment = Shares × (YES + NO price)
   - Profit = Shares × (1 - YES - NO price)

EXAMPLE (first trade):
""")

        if arb_trades:
            ex = arb_trades[0]
            print(f"""
   Market: {ex['market_id']}

   ACTION:
   - Buy {ex['shares_to_buy']} shares of YES at ${ex['yes_price']:.3f} = ${ex['shares_to_buy'] * ex['yes_price']:.2f}
   - Buy {ex['shares_to_buy']} shares of NO at ${ex['no_price']:.3f} = ${ex['shares_to_buy'] * ex['no_price']:.2f}

   TOTAL INVESTMENT: ${ex['investment']:.2f}
   GUARANTEED PAYOUT: ${ex['shares_to_buy']:.2f}
   GUARANTEED PROFIT: ${ex['profit_usd']:.2f} ({ex['profit_pct']:.1f}%)
""")

    # 2. ENTRY POINTS
    print(f"\n{'='*100}")
    print("SEARCHING FOR ENTRY POINTS...")
    print(f"{'='*100}")

    entries = find_all_entry_points(max_markets=500)

    print(f"\nENTRY SIGNALS FOUND: {len(entries)}")

    if entries:
        # Sort by score
        entries.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'='*100}")
        print("TOP 100 ENTRY POINTS (by score)")
        print(f"{'='*100}")

        print(f"\n{'#':<5} | {'Market ID':<44} | {'Signal':<15} | {'Entry$':<8} | {'Shares':<8} | {'Invest$':<10} | {'Target$':<8} | {'Profit$':<10} | {'Score'}")
        print("-" * 170)

        for i, entry in enumerate(entries[:100], 1):
            print(f"{i:<5} | {entry['market_id']:<44} | "
                  f"{entry['signal']:<15} | ${entry['entry_price']:<6.3f} | "
                  f"{entry['recommended_shares']:<8} | ${entry['investment_usd']:<8.2f} | "
                  f"${entry['target_price']:<6.2f} | ${entry['potential_profit']:<8.2f} | "
                  f"{entry['score']}")

        # Show ALL entries to file
        print(f"\n{'='*100}")
        print("SAVING ALL ENTRIES TO FILE...")
        print(f"{'='*100}")

        # Save to CSV for bot use
        csv_file = "/home/user/polymarket-pnl/bot/entry_points.csv"
        with open(csv_file, 'w') as f:
            f.write("market_id,timestamp,datetime,signal,outcome,entry_price,size_available,score,recommended_shares,investment_usd,target_price,potential_profit,avg_market_price,market_volatility,reasons\n")
            for entry in entries:
                reasons_str = "|".join(entry['reasons'])
                f.write(f"{entry['market_id']},{entry['timestamp']},{entry['datetime']},{entry['signal']},"
                        f"{entry['outcome']},{entry['entry_price']},{entry['size_available']},{entry['score']},"
                        f"{entry['recommended_shares']},{entry['investment_usd']},{entry['target_price']},"
                        f"{entry['potential_profit']},{entry['avg_market_price']},{entry['market_volatility']},\"{reasons_str}\"\n")

        print(f"Saved {len(entries)} entries to {csv_file}")

        # Save arbitrage to CSV
        arb_csv = "/home/user/polymarket-pnl/bot/arbitrage_trades.csv"
        with open(arb_csv, 'w') as f:
            f.write("market_id,timestamp,datetime,yes_price,no_price,total_cost,profit_pct,shares_to_buy,yes_liquidity,no_liquidity,investment,guaranteed_payout,profit_usd\n")
            for trade in arb_trades:
                f.write(f"{trade['market_id']},{trade['timestamp']},{trade['datetime']},"
                        f"{trade['yes_price']},{trade['no_price']},{trade['total_cost']},"
                        f"{trade['profit_pct']},{trade['shares_to_buy']},{trade['yes_liquidity']},"
                        f"{trade['no_liquidity']},{trade['investment']},{trade['guaranteed_payout']},"
                        f"{trade['profit_usd']}\n")

        print(f"Saved {len(arb_trades)} arbitrage trades to {arb_csv}")

        # Entry summary
        print(f"\n{'='*100}")
        print("ENTRY POINT SUMMARY")
        print(f"{'='*100}")

        buy_low_yes = [e for e in entries if e['signal'] == 'BUY_LOW_YES']
        buy_low_no = [e for e in entries if e['signal'] == 'BUY_LOW_NO']

        total_potential = sum(e['potential_profit'] for e in entries)

        print(f"""
Total entry signals: {len(entries)}
  - BUY_LOW_YES (buy YES at <20c): {len(buy_low_yes)}
  - BUY_LOW_NO (buy NO at <20c): {len(buy_low_no)}

Score distribution:
  - Score 80+: {len([e for e in entries if e['score'] >= 80])} (EXCELLENT)
  - Score 60-79: {len([e for e in entries if 60 <= e['score'] < 80])} (GOOD)
  - Score 40-59: {len([e for e in entries if 40 <= e['score'] < 60])} (MODERATE)

Total potential profit (if all hit targets): ${total_potential:,.2f}
Average potential per entry: ${total_potential/len(entries):.2f}
""")

        # Bot instructions for entries
        print(f"{'='*100}")
        print("BOT ENTRY INSTRUCTIONS")
        print(f"{'='*100}")
        print("""
HOW TO EXECUTE ENTRY TRADES:

For each entry point:

1. CHECK current market price matches entry price (±2%)

2. PLACE limit order:
   - Side: BUY
   - Outcome: [YES or NO as shown]
   - Price: [Entry Price]
   - Shares: [Recommended Shares]

3. SET take-profit order at [Target Price]

4. SET stop-loss at 50% of entry price

RISK MANAGEMENT:
- Max $500 per trade
- Max 10 concurrent positions
- Daily loss limit: $1,000

EXAMPLE (best entry):
""")

        if entries:
            ex = entries[0]
            print(f"""
   Market: {ex['market_id']}
   Signal: {ex['signal']}
   Score: {ex['score']}

   ACTION:
   - BUY {ex['recommended_shares']} shares of {ex['outcome']}
   - Entry price: ${ex['entry_price']:.3f}
   - Total investment: ${ex['investment_usd']:.2f}

   TARGETS:
   - Take profit at: ${ex['target_price']:.2f}
   - Stop loss at: ${ex['entry_price'] * 0.5:.3f}

   POTENTIAL PROFIT: ${ex['potential_profit']:.2f}

   Reasons: {', '.join(ex['reasons'])}
""")

    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*100}")
    print(f"""
Files created:
1. bot/arbitrage_trades.csv - All {len(arb_trades)} arbitrage opportunities
2. bot/entry_points.csv - All {len(entries)} entry signals

Use these CSV files to feed your trading bot.
""")


if __name__ == "__main__":
    main()
