"""
Working $10K Strategy - Properly Constrained
"""

import random
from typing import Dict, List


# Realistic parameters based on Polymarket observation
SIGNALS_PER_HOUR = {
    'extreme': 1,    # ~72% accuracy, very rare
    'strong': 2,     # ~65% accuracy
    'moderate': 4,   # ~58% accuracy
}

ACCURACY = {'extreme': 0.72, 'strong': 0.65, 'moderate': 0.58}

# CRITICAL: Max position per market (liquidity constraint)
# Polymarket crypto markets typically have $10k-50k in total liquidity
# Taking more than 10% causes significant slippage
MAX_BET_PER_MARKET = 2000  # $2,000 max per single market


def simulate_one_day(capital: float, min_accuracy: float, fee_rate: float) -> Dict:
    """Simulate one day of trading with proper constraints."""

    balance = capital
    trades = []

    for hour in range(24):
        # Generate signals for this hour
        for signal_type, count in SIGNALS_PER_HOUR.items():
            accuracy = ACCURACY[signal_type]

            # Skip if below our accuracy threshold
            if accuracy < min_accuracy:
                continue

            for _ in range(count):
                # Position size: capped by liquidity
                position = min(MAX_BET_PER_MARKET, capital * 0.1)  # Also cap at 10% of capital

                if position < 50:  # Minimum viable bet
                    continue

                # Determine win/loss
                won = random.random() < accuracy

                # Polymarket payoff (simplified: buy at 0.5)
                if won:
                    gross_pnl = 0.5 * position  # Win $0.50 per $1 bet
                else:
                    gross_pnl = -0.5 * position  # Lose $0.50 per $1 bet

                fees = position * fee_rate
                net_pnl = gross_pnl - fees

                balance += net_pnl

                trades.append({
                    'signal': signal_type,
                    'accuracy': accuracy,
                    'position': position,
                    'won': won,
                    'pnl': net_pnl,
                })

    return {
        'pnl': balance - capital,
        'trades': len(trades),
        'wins': sum(1 for t in trades if t['won']),
        'avg_position': sum(t['position'] for t in trades) / len(trades) if trades else 0,
    }


def run_strategy(capital: float, min_accuracy: float, fee_rate: float, days: int = 100) -> Dict:
    """Run multiple days and get stats."""
    results = []

    for _ in range(days):
        day = simulate_one_day(capital, min_accuracy, fee_rate)
        results.append(day)

    pnls = [r['pnl'] for r in results]

    return {
        'avg_pnl': sum(pnls) / len(pnls),
        'min_pnl': min(pnls),
        'max_pnl': max(pnls),
        'profitable': sum(1 for p in pnls if p > 0),
        'hit_10k': sum(1 for p in pnls if p >= 10000),
        'avg_trades': sum(r['trades'] for r in results) / len(results),
        'win_rate': sum(r['wins'] for r in results) / sum(r['trades'] for r in results) if sum(r['trades'] for r in results) > 0 else 0,
        'avg_position': sum(r['avg_position'] for r in results) / len(results),
    }


def main():
    print("=" * 70)
    print("WORKING $10K STRATEGY")
    print("=" * 70)
    print()
    print(f"MAX BET PER MARKET: ${MAX_BET_PER_MARKET:,}")
    print()
    print("SIGNALS PER HOUR:")
    for sig, count in SIGNALS_PER_HOUR.items():
        daily = count * 24
        print(f"  {sig}: {count}/hr ({daily}/day) @ {ACCURACY[sig]:.0%} accuracy")
    print()

    # Test configurations
    configs = []

    for capital in [10000, 25000, 50000, 100000]:
        for min_acc in [0.58, 0.65, 0.72]:  # moderate, strong, extreme only
            for fee in [0.005, 0.01, 0.015, 0.02]:
                configs.append({
                    'capital': capital,
                    'min_accuracy': min_acc,
                    'fee_rate': fee,
                })

    print(f"Testing {len(configs)} configurations...")
    print()

    results = []
    for cfg in configs:
        r = run_strategy(**cfg, days=100)
        r['config'] = cfg
        results.append(r)

    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    # Show results
    print("=" * 70)
    print("TOP 10 STRATEGIES")
    print("=" * 70)

    for i, r in enumerate(results[:10], 1):
        c = r['config']
        acc_name = {0.58: 'moderate', 0.65: 'strong', 0.72: 'extreme'}[c['min_accuracy']]
        print(f"\n{i}. ${c['capital']:,} | {acc_name}+ signals | {c['fee_rate']:.1%} fee")
        print(f"   Avg P&L: ${r['avg_pnl']:+,.2f}/day | Trades: {r['avg_trades']:.0f}/day")
        print(f"   Win Rate: {r['win_rate']:.1%} | Avg Position: ${r['avg_position']:,.0f}")
        print(f"   Profitable: {r['profitable']}/100 | Hit $10k: {r['hit_10k']}/100")

    # Best strategy
    best = results[0]
    c = best['config']

    print(f"\n{'*'*70}")
    print("BEST STRATEGY")
    print(f"{'*'*70}")

    acc_name = {0.58: 'moderate', 0.65: 'strong', 0.72: 'extreme'}[c['min_accuracy']]

    print(f"""
CONFIGURATION:
- Capital: ${c['capital']:,}
- Signal Filter: {acc_name}+ only ({c['min_accuracy']:.0%}+ accuracy)
- Fee Rate: {c['fee_rate']:.1%}
- Max Position: ${MAX_BET_PER_MARKET:,} per market

PERFORMANCE:
- Avg Daily P&L: ${best['avg_pnl']:+,.2f}
- Trades/Day: {best['avg_trades']:.0f}
- Win Rate: {best['win_rate']:.1%}
- Avg Position: ${best['avg_position']:,.0f}
- Profitable Days: {best['profitable']}/100
- Days Hitting $10k: {best['hit_10k']}/100
- Best Day: ${best['max_pnl']:+,.2f}
- Worst Day: ${best['min_pnl']:+,.2f}

MONTHLY: ${best['avg_pnl'] * 30:+,.2f}
""")

    # What you need for $10k
    hit_10k_configs = [r for r in results if r['hit_10k'] >= 50]  # At least 50% chance

    print("=" * 70)
    print("CONFIGURATIONS THAT CAN HIT $10K/DAY (>50% of days)")
    print("=" * 70)

    if hit_10k_configs:
        for r in hit_10k_configs[:5]:
            c = r['config']
            acc_name = {0.58: 'moderate', 0.65: 'strong', 0.72: 'extreme'}[c['min_accuracy']]
            print(f"\n${c['capital']:,} | {acc_name}+ | {c['fee_rate']:.1%} fee")
            print(f"  Avg: ${r['avg_pnl']:+,.2f} | Hit $10k: {r['hit_10k']}/100 days")
    else:
        print("\nNone with current constraints.")
        print(f"Best achieves ${results[0]['avg_pnl']:+,.2f}/day")

        if results[0]['avg_pnl'] > 0:
            needed = 10000 / results[0]['avg_pnl']
            print(f"\nTo average $10k/day, you would need to scale up {needed:.1f}x")
            print(f"This requires either:")
            print(f"  - Higher liquidity markets (increase MAX_BET)")
            print(f"  - More capital")
            print(f"  - More signal opportunities")

    # The actual math
    print(f"\n{'='*70}")
    print("THE MATH (Reality Check)")
    print(f"{'='*70}")

    # Calculate expected value with best config
    c = best['config']
    pos = min(MAX_BET_PER_MARKET, c['capital'] * 0.1)

    # Count signals per day at this accuracy threshold
    signals_day = 0
    weighted_acc = 0
    total_weight = 0

    for sig, count in SIGNALS_PER_HOUR.items():
        if ACCURACY[sig] >= c['min_accuracy']:
            signals_day += count * 24
            weighted_acc += ACCURACY[sig] * count
            total_weight += count

    if total_weight > 0:
        weighted_acc /= total_weight

    ev_per_trade = (weighted_acc * 0.5 * pos) - ((1-weighted_acc) * 0.5 * pos) - (pos * c['fee_rate'])
    ev_per_day = ev_per_trade * signals_day

    print(f"""
Capital: ${c['capital']:,}
Position per trade: ${pos:,.0f} (min of ${MAX_BET_PER_MARKET:,} liquidity cap and 10% capital)
Signals per day: {signals_day}
Weighted accuracy: {weighted_acc:.1%}
Fee rate: {c['fee_rate']:.1%}

Per Trade:
  Win (+50%): ${0.5 * pos:,.2f}
  Lose (-50%): ${0.5 * pos:,.2f}
  Fees: ${pos * c['fee_rate']:,.2f}

  EV = ({weighted_acc:.2f} x ${0.5*pos:,.0f}) - ({1-weighted_acc:.2f} x ${0.5*pos:,.0f}) - ${pos * c['fee_rate']:,.0f}
  EV = ${ev_per_trade:,.2f} per trade

Per Day ({signals_day} trades):
  Expected P&L = ${ev_per_day:,.2f}

TO REACH $10K/DAY:
  Need {10000/ev_per_trade:.0f} trades at this EV
  OR increase position size to ${10000/signals_day/(weighted_acc - 0.5 - c['fee_rate']):,.0f} per trade
  OR find more high-accuracy signals
""")


if __name__ == "__main__":
    main()
