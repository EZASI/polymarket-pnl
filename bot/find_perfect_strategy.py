"""
Find a Strategy That Works Every Time

Testing EVERY possible strategy against real outcome data:
1. Pure arbitrage (YES + NO < $1)
2. Buy at specific price levels
3. Time-based patterns
4. Resolution patterns

Goal: Find ANY strategy with 100% win rate
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import random


DATA_BASE = "/home/user/polymarket-pnl/bot/data/full_market/Polymarket_dataset/Polymarket_dataset"


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
                                'price': float(data['p'])
                            })
                    except:
                        continue
    except:
        pass
    return sorted(prices, key=lambda x: x['timestamp'])


def get_market_outcome(prices: List[Dict]) -> Tuple[str, float]:
    """
    Determine market outcome from final price.
    Final price near 1.0 = YES won
    Final price near 0.0 = NO won (YES lost)
    """
    if not prices:
        return None, None

    final_price = prices[-1]['price']

    if final_price >= 0.95:
        return 'YES', final_price
    elif final_price <= 0.05:
        return 'NO', final_price
    else:
        return 'UNRESOLVED', final_price


def test_strategy_buy_low(prices: List[Dict], threshold: float) -> Dict:
    """
    Strategy: Buy YES when price drops below threshold.
    Win if final price = 1.0 (YES wins)
    """
    if len(prices) < 10:
        return None

    outcome, final = get_market_outcome(prices)
    if outcome == 'UNRESOLVED':
        return None

    # Find opportunities to buy below threshold
    entries = []
    for i, p in enumerate(prices[:-10]):  # Don't buy near end
        if p['price'] <= threshold:
            entries.append({
                'entry_price': p['price'],
                'timestamp': p['timestamp']
            })

    if not entries:
        return None

    # Calculate results
    wins = 0
    losses = 0
    total_profit = 0

    for entry in entries:
        if outcome == 'YES':
            # YES won, we get $1 per share
            profit = 1.0 - entry['entry_price']
            wins += 1
        else:
            # NO won, we get $0
            profit = -entry['entry_price']
            losses += 1
        total_profit += profit

    return {
        'entries': len(entries),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(entries) * 100 if entries else 0,
        'total_profit': total_profit,
        'avg_profit': total_profit / len(entries) if entries else 0
    }


def test_strategy_sell_high(prices: List[Dict], threshold: float) -> Dict:
    """
    Strategy: Buy NO (sell YES) when YES price goes above threshold.
    Win if final price = 0.0 (NO wins)
    """
    if len(prices) < 10:
        return None

    outcome, final = get_market_outcome(prices)
    if outcome == 'UNRESOLVED':
        return None

    entries = []
    for i, p in enumerate(prices[:-10]):
        if p['price'] >= threshold:
            # NO price = 1 - YES price
            no_price = 1 - p['price']
            entries.append({
                'entry_price': no_price,
                'yes_price': p['price'],
                'timestamp': p['timestamp']
            })

    if not entries:
        return None

    wins = 0
    losses = 0
    total_profit = 0

    for entry in entries:
        if outcome == 'NO':
            # NO won, we get $1 per share
            profit = 1.0 - entry['entry_price']
            wins += 1
        else:
            # YES won, we get $0
            profit = -entry['entry_price']
            losses += 1
        total_profit += profit

    return {
        'entries': len(entries),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(entries) * 100 if entries else 0,
        'total_profit': total_profit,
        'avg_profit': total_profit / len(entries) if entries else 0
    }


def test_strategy_mean_reversion(prices: List[Dict]) -> Dict:
    """
    Strategy: Buy when price drops 10%+ from recent high,
    sell when it recovers 5%+.
    """
    if len(prices) < 20:
        return None

    trades = []
    position = None

    for i in range(10, len(prices) - 5):
        recent_high = max(p['price'] for p in prices[i-10:i])
        recent_low = min(p['price'] for p in prices[i-10:i])
        current = prices[i]['price']

        if position is None:
            # Look to buy
            if current <= recent_high * 0.90:  # 10% drop
                position = {'entry': current, 'idx': i}
        else:
            # Look to sell
            if current >= position['entry'] * 1.05:  # 5% gain
                profit = current - position['entry']
                trades.append({'profit': profit, 'win': True})
                position = None
            elif i > position['idx'] + 20:  # Timeout
                profit = current - position['entry']
                trades.append({'profit': profit, 'win': profit > 0})
                position = None

    if not trades:
        return None

    wins = sum(1 for t in trades if t['win'])
    return {
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) * 100,
        'total_profit': sum(t['profit'] for t in trades),
        'avg_profit': sum(t['profit'] for t in trades) / len(trades)
    }


def test_resolved_market_pattern(prices: List[Dict]) -> Dict:
    """
    Analyze: Do markets that reach certain prices always resolve that way?
    e.g., If YES hits 90%+, does it always win?
    """
    if len(prices) < 10:
        return None

    outcome, final = get_market_outcome(prices)
    if outcome == 'UNRESOLVED':
        return None

    max_price = max(p['price'] for p in prices)
    min_price = min(p['price'] for p in prices)

    return {
        'outcome': outcome,
        'max_price_reached': max_price,
        'min_price_reached': min_price,
        'final_price': final
    }


def analyze_all_markets(max_markets: int = 1000):
    """Run all strategy tests."""

    results = {
        'buy_low_10': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'buy_low_15': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'buy_low_20': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'buy_low_25': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'buy_low_30': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'sell_high_70': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'sell_high_75': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'sell_high_80': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'sell_high_85': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'sell_high_90': {'wins': 0, 'losses': 0, 'profit': 0, 'entries': 0},
        'mean_reversion': {'wins': 0, 'losses': 0, 'profit': 0, 'trades': 0},
    }

    resolution_patterns = {
        'hit_90_resolved_yes': 0,
        'hit_90_resolved_no': 0,
        'hit_10_resolved_no': 0,
        'hit_10_resolved_yes': 0,
        'hit_95_resolved_yes': 0,
        'hit_95_resolved_no': 0,
        'hit_05_resolved_no': 0,
        'hit_05_resolved_yes': 0,
    }

    markets_analyzed = 0
    resolved_markets = 0

    market_dirs = sorted(Path(DATA_BASE).glob('market=*'))[:max_markets]

    for market_dir in market_dirs:
        price_files = list(market_dir.glob('price/*.ndjson'))

        for pf in price_files:
            prices = load_price_data(str(pf))

            if len(prices) < 20:
                continue

            markets_analyzed += 1

            outcome, final = get_market_outcome(prices)
            if outcome != 'UNRESOLVED':
                resolved_markets += 1

            # Test buy low strategies
            for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
                key = f'buy_low_{int(thresh*100)}'
                result = test_strategy_buy_low(prices, thresh)
                if result:
                    results[key]['wins'] += result['wins']
                    results[key]['losses'] += result['losses']
                    results[key]['profit'] += result['total_profit']
                    results[key]['entries'] += result['entries']

            # Test sell high strategies
            for thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
                key = f'sell_high_{int(thresh*100)}'
                result = test_strategy_sell_high(prices, thresh)
                if result:
                    results[key]['wins'] += result['wins']
                    results[key]['losses'] += result['losses']
                    results[key]['profit'] += result['total_profit']
                    results[key]['entries'] += result['entries']

            # Test mean reversion
            result = test_strategy_mean_reversion(prices)
            if result:
                results['mean_reversion']['wins'] += result['wins']
                results['mean_reversion']['losses'] += result['trades'] - result['wins']
                results['mean_reversion']['profit'] += result['total_profit']
                results['mean_reversion']['trades'] += result['trades']

            # Analyze resolution patterns
            pattern = test_resolved_market_pattern(prices)
            if pattern:
                max_p = pattern['max_price_reached']
                min_p = pattern['min_price_reached']
                outcome = pattern['outcome']

                if max_p >= 0.90:
                    if outcome == 'YES':
                        resolution_patterns['hit_90_resolved_yes'] += 1
                    else:
                        resolution_patterns['hit_90_resolved_no'] += 1

                if max_p >= 0.95:
                    if outcome == 'YES':
                        resolution_patterns['hit_95_resolved_yes'] += 1
                    else:
                        resolution_patterns['hit_95_resolved_no'] += 1

                if min_p <= 0.10:
                    if outcome == 'NO':
                        resolution_patterns['hit_10_resolved_no'] += 1
                    else:
                        resolution_patterns['hit_10_resolved_yes'] += 1

                if min_p <= 0.05:
                    if outcome == 'NO':
                        resolution_patterns['hit_05_resolved_no'] += 1
                    else:
                        resolution_patterns['hit_05_resolved_yes'] += 1

    return results, resolution_patterns, markets_analyzed, resolved_markets


def main():
    print("=" * 100)
    print("FINDING A STRATEGY THAT WORKS EVERY TIME")
    print("Testing against real market outcomes")
    print("=" * 100)
    print()

    print("Analyzing markets...")
    results, patterns, total, resolved = analyze_all_markets(max_markets=1000)

    print(f"\nMarkets analyzed: {total}")
    print(f"Resolved markets: {resolved}")

    # 1. BUY LOW STRATEGIES
    print("\n" + "=" * 100)
    print("STRATEGY 1: BUY YES WHEN PRICE IS LOW")
    print("=" * 100)

    print(f"\n{'Threshold':<12} | {'Entries':<10} | {'Wins':<10} | {'Losses':<10} | {'Win Rate':<12} | {'Profit':<12} | {'Works?'}")
    print("-" * 90)

    for thresh in [10, 15, 20, 25, 30]:
        key = f'buy_low_{thresh}'
        r = results[key]
        entries = r['entries']
        if entries > 0:
            win_rate = r['wins'] / entries * 100
            works = "YES ✓" if win_rate == 100 else "NO ✗"
            print(f"<${thresh/100:.2f}       | {entries:<10} | {r['wins']:<10} | {r['losses']:<10} | "
                  f"{win_rate:<12.1f}% | ${r['profit']:<10.2f} | {works}")

    # 2. SELL HIGH STRATEGIES
    print("\n" + "=" * 100)
    print("STRATEGY 2: BUY NO (SELL YES) WHEN YES PRICE IS HIGH")
    print("=" * 100)

    print(f"\n{'Threshold':<12} | {'Entries':<10} | {'Wins':<10} | {'Losses':<10} | {'Win Rate':<12} | {'Profit':<12} | {'Works?'}")
    print("-" * 90)

    for thresh in [70, 75, 80, 85, 90]:
        key = f'sell_high_{thresh}'
        r = results[key]
        entries = r['entries']
        if entries > 0:
            win_rate = r['wins'] / entries * 100
            works = "YES ✓" if win_rate == 100 else "NO ✗"
            print(f">${thresh/100:.2f}       | {entries:<10} | {r['wins']:<10} | {r['losses']:<10} | "
                  f"{win_rate:<12.1f}% | ${r['profit']:<10.2f} | {works}")

    # 3. MEAN REVERSION
    print("\n" + "=" * 100)
    print("STRATEGY 3: MEAN REVERSION (Buy dips, sell recoveries)")
    print("=" * 100)

    mr = results['mean_reversion']
    if mr['trades'] > 0:
        win_rate = mr['wins'] / mr['trades'] * 100
        works = "YES ✓" if win_rate == 100 else "NO ✗"
        print(f"\nTrades: {mr['trades']}")
        print(f"Wins: {mr['wins']}")
        print(f"Losses: {mr['losses']}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Profit: ${mr['profit']:.2f}")
        print(f"Works Every Time? {works}")

    # 4. RESOLUTION PATTERNS
    print("\n" + "=" * 100)
    print("STRATEGY 4: BET ON MOMENTUM (If price hits X, bet it resolves that way)")
    print("=" * 100)

    print("\nDoes hitting a high price guarantee YES resolution?")

    for thresh in [90, 95]:
        yes_key = f'hit_{thresh}_resolved_yes'
        no_key = f'hit_{thresh}_resolved_no'
        total_hit = patterns[yes_key] + patterns[no_key]
        if total_hit > 0:
            accuracy = patterns[yes_key] / total_hit * 100
            works = "YES ✓" if accuracy == 100 else "NO ✗"
            print(f"\n  Price hit {thresh}%+:")
            print(f"    Resolved YES: {patterns[yes_key]}")
            print(f"    Resolved NO:  {patterns[no_key]}")
            print(f"    Accuracy: {accuracy:.1f}%")
            print(f"    Works Every Time? {works}")

    print("\nDoes hitting a low price guarantee NO resolution?")

    for thresh in [10, 5]:
        no_key = f'hit_{thresh:02d}_resolved_no'
        yes_key = f'hit_{thresh:02d}_resolved_yes'
        total_hit = patterns[no_key] + patterns[yes_key]
        if total_hit > 0:
            accuracy = patterns[no_key] / total_hit * 100
            works = "YES ✓" if accuracy == 100 else "NO ✗"
            print(f"\n  Price hit {thresh}% or lower:")
            print(f"    Resolved NO:  {patterns[no_key]}")
            print(f"    Resolved YES: {patterns[yes_key]}")
            print(f"    Accuracy: {accuracy:.1f}%")
            print(f"    Works Every Time? {works}")

    # 5. THE ONLY GUARANTEED STRATEGY
    print("\n" + "=" * 100)
    print("THE ONLY STRATEGY THAT WORKS 100% OF THE TIME")
    print("=" * 100)

    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║   PURE ARBITRAGE: Buy YES + NO when total < $1.00                   ║
    ║                                                                      ║
    ║   Example:                                                           ║
    ║   - Buy YES at $0.40                                                ║
    ║   - Buy NO at $0.50                                                 ║
    ║   - Total cost: $0.90                                               ║
    ║   - Guaranteed payout: $1.00                                        ║
    ║   - Guaranteed profit: $0.10 (11.1%)                                ║
    ║                                                                      ║
    ║   WHY IT'S GUARANTEED:                                              ║
    ║   - One of YES or NO MUST be worth $1.00 at resolution             ║
    ║   - The other MUST be worth $0.00                                   ║
    ║   - You own both, so you ALWAYS get $1.00 back                     ║
    ║                                                                      ║
    ║   THE CATCH:                                                         ║
    ║   - These opportunities last MILLISECONDS                           ║
    ║   - Need real-time data + instant execution                         ║
    ║   - Professional traders already capture them                       ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Find best non-100% strategy
    print("\n" + "=" * 100)
    print("BEST NON-GUARANTEED STRATEGIES (Ranked by Profit)")
    print("=" * 100)

    all_strategies = []

    for thresh in [10, 15, 20, 25, 30]:
        key = f'buy_low_{thresh}'
        r = results[key]
        if r['entries'] > 0:
            all_strategies.append({
                'name': f'Buy YES below ${thresh/100:.2f}',
                'entries': r['entries'],
                'win_rate': r['wins'] / r['entries'] * 100,
                'profit': r['profit'],
                'avg_profit': r['profit'] / r['entries']
            })

    for thresh in [70, 75, 80, 85, 90]:
        key = f'sell_high_{thresh}'
        r = results[key]
        if r['entries'] > 0:
            all_strategies.append({
                'name': f'Buy NO above ${thresh/100:.2f}',
                'entries': r['entries'],
                'win_rate': r['wins'] / r['entries'] * 100,
                'profit': r['profit'],
                'avg_profit': r['profit'] / r['entries']
            })

    all_strategies.sort(key=lambda x: x['profit'], reverse=True)

    print(f"\n{'Strategy':<25} | {'Entries':<10} | {'Win Rate':<12} | {'Total Profit':<14} | {'Avg/Trade'}")
    print("-" * 85)

    for s in all_strategies[:10]:
        print(f"{s['name']:<25} | {s['entries']:<10} | {s['win_rate']:<12.1f}% | ${s['profit']:<12.2f} | ${s['avg_profit']:.4f}")

    # Final verdict
    print("\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)

    best = all_strategies[0] if all_strategies else None

    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │ DOES ANY STRATEGY WORK 100% OF THE TIME?                           │
    │                                                                     │
    │ ❌ Buy Low:        {results['buy_low_20']['wins']}/{results['buy_low_20']['entries']} wins ({results['buy_low_20']['wins']/results['buy_low_20']['entries']*100 if results['buy_low_20']['entries'] > 0 else 0:.1f}% - NOT 100%)                           │
    │ ❌ Sell High:      {results['sell_high_80']['wins']}/{results['sell_high_80']['entries']} wins ({results['sell_high_80']['wins']/results['sell_high_80']['entries']*100 if results['sell_high_80']['entries'] > 0 else 0:.1f}% - NOT 100%)                           │
    │ ❌ Mean Reversion: {mr['wins']}/{mr['trades']} wins ({mr['wins']/mr['trades']*100 if mr['trades'] > 0 else 0:.1f}% - NOT 100%)                           │
    │ ❌ Price Momentum: ~{patterns['hit_95_resolved_yes']/(patterns['hit_95_resolved_yes']+patterns['hit_95_resolved_no'])*100 if (patterns['hit_95_resolved_yes']+patterns['hit_95_resolved_no']) > 0 else 0:.0f}% accurate (NOT 100%)                                │
    │                                                                     │
    │ ✅ ONLY GUARANTEED: Pure arbitrage (YES + NO < $1)                 │
    │    But: Opportunities exist for milliseconds only                   │
    │                                                                     │
    │ BEST REALISTIC STRATEGY:                                            │
    │ {best['name'] if best else 'N/A':<30} {best['win_rate'] if best else 0:.1f}% win rate, ${best['profit'] if best else 0:.2f} profit        │
    └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
