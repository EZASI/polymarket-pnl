"""
Final Realistic $10K Strategy
Properly constrained for actual Polymarket trading:
- Max 16 opportunities/hour (often fewer with strong signals)
- Position limits based on market depth
- Realistic signal distribution
"""

import random
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Signal:
    strength: str  # weak, moderate, strong, extreme
    accuracy: float
    max_position: float  # Liquidity-based limit


# Realistic signal occurrence rates per hour (across 16 market slots)
SIGNAL_RATES = {
    'extreme': 1,     # ~1 extreme signal per hour (rare)
    'strong': 2,      # ~2 strong signals per hour
    'moderate': 4,    # ~4 moderate signals per hour
    'weak': 9,        # Rest are weak (not worth trading)
}

ACCURACY = {
    'extreme': 0.72,
    'strong': 0.65,
    'moderate': 0.58,
    'weak': 0.52,
}

# Max position per signal type (liquidity constraints)
MAX_POSITION = {
    'extreme': 3000,   # High conviction but limited size due to market impact
    'strong': 2500,
    'moderate': 2000,
    'weak': 1000,
}


def generate_hourly_signals() -> List[Signal]:
    """Generate realistic signals for 1 hour."""
    signals = []

    for strength, count in SIGNAL_RATES.items():
        for _ in range(count):
            signals.append(Signal(
                strength=strength,
                accuracy=ACCURACY[strength],
                max_position=MAX_POSITION[strength] * random.uniform(0.8, 1.2),
            ))

    random.shuffle(signals)
    return signals


def simulate_day(
    capital: float,
    position_pct: float,
    min_signal: str,  # Only trade this strength or better
    fee_rate: float,
) -> Dict:
    """Simulate one trading day."""

    # Signal strength hierarchy
    strength_order = ['weak', 'moderate', 'strong', 'extreme']
    min_idx = strength_order.index(min_signal)
    tradeable_strengths = set(strength_order[min_idx:])

    balance = capital
    trades = []

    for hour in range(24):
        signals = generate_hourly_signals()

        for signal in signals:
            if signal.strength not in tradeable_strengths:
                continue

            # Position size: min of capital %, and liquidity limit
            desired_position = capital * position_pct
            position_size = min(desired_position, signal.max_position)

            # Check if we have enough capital
            if position_size > balance * 0.5:  # Don't risk more than 50% of remaining
                position_size = balance * 0.5

            if position_size < 100:
                continue

            # Determine outcome
            won = random.random() < signal.accuracy

            # Polymarket payoff (entry at ~0.5)
            if won:
                gross_pnl = 0.5 * position_size
            else:
                gross_pnl = -0.5 * position_size

            fees = position_size * fee_rate
            net_pnl = gross_pnl - fees

            balance += net_pnl

            trades.append({
                'strength': signal.strength,
                'accuracy': signal.accuracy,
                'position': position_size,
                'won': won,
                'pnl': net_pnl,
            })

    wins = sum(1 for t in trades if t['won'])

    return {
        'pnl': balance - capital,
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) if trades else 0,
        'by_strength': {
            s: {
                'count': sum(1 for t in trades if t['strength'] == s),
                'wins': sum(1 for t in trades if t['strength'] == s and t['won']),
                'pnl': sum(t['pnl'] for t in trades if t['strength'] == s),
            }
            for s in ['moderate', 'strong', 'extreme']
        }
    }


def run_trials(
    capital: float,
    position_pct: float,
    min_signal: str,
    fee_rate: float,
    num_trials: int = 100,
) -> Dict:
    """Run multiple trials."""
    results = []

    for _ in range(num_trials):
        day = simulate_day(capital, position_pct, min_signal, fee_rate)
        results.append(day)

    pnls = [r['pnl'] for r in results]

    return {
        'avg_pnl': sum(pnls) / len(pnls),
        'min_pnl': min(pnls),
        'max_pnl': max(pnls),
        'profitable': sum(1 for p in pnls if p > 0),
        'hit_10k': sum(1 for p in pnls if p >= 10000),
        'avg_trades': sum(r['trades'] for r in results) / len(results),
        'avg_win_rate': sum(r['win_rate'] for r in results) / len(results),
    }


def main():
    print("=" * 70)
    print("FINAL REALISTIC $10K STRATEGY")
    print("=" * 70)
    print()
    print("SIGNAL DISTRIBUTION (per hour):")
    for strength, count in SIGNAL_RATES.items():
        print(f"  {strength:10}: {count} signals, {ACCURACY[strength]:.0%} accuracy, "
              f"max ${MAX_POSITION[strength]:,} position")
    print()

    configs = []

    # Test different configurations
    for capital in [10000, 25000, 50000, 100000, 150000]:
        for pos_pct in [0.05, 0.10, 0.15, 0.20]:
            for min_signal in ['moderate', 'strong', 'extreme']:
                for fee in [0.005, 0.01, 0.015]:
                    configs.append({
                        'capital': capital,
                        'position_pct': pos_pct,
                        'min_signal': min_signal,
                        'fee_rate': fee,
                    })

    print(f"Testing {len(configs)} configurations...\n")

    results = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(configs)}")

        r = run_trials(**cfg)
        r['config'] = cfg
        results.append(r)

    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    # Filter meaningful results
    profitable = [r for r in results if r['profitable'] > 50]
    hit_10k = [r for r in results if r['hit_10k'] > 0]
    avg_10k = [r for r in results if r['avg_pnl'] >= 10000]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal configs: {len(configs)}")
    print(f">50% profitable: {len(profitable)}")
    print(f"Can hit $10k: {len(hit_10k)}")
    print(f"Average $10k+: {len(avg_10k)}")

    print(f"\n{'='*70}")
    print("TOP 10 STRATEGIES")
    print(f"{'='*70}")

    for i, r in enumerate(results[:10], 1):
        c = r['config']
        print(f"\n{i}. ${c['capital']:,} | {c['position_pct']:.0%} pos | "
              f"{c['min_signal']} signals | {c['fee_rate']:.1%} fee")
        print(f"   Daily P&L: ${r['avg_pnl']:+,.2f} | Trades: {r['avg_trades']:.1f}")
        print(f"   Win Rate: {r['avg_win_rate']:.1%} | Profitable: {r['profitable']}/100")
        print(f"   Hit $10k: {r['hit_10k']}/100 | Range: ${r['min_pnl']:+,.0f} to ${r['max_pnl']:+,.0f}")

    # Best strategy
    best = results[0]
    c = best['config']

    print(f"\n{'*'*70}")
    print("RECOMMENDED STRATEGY")
    print(f"{'*'*70}")

    # Calculate daily signal counts
    if c['min_signal'] == 'moderate':
        signals_per_hour = SIGNAL_RATES['moderate'] + SIGNAL_RATES['strong'] + SIGNAL_RATES['extreme']
    elif c['min_signal'] == 'strong':
        signals_per_hour = SIGNAL_RATES['strong'] + SIGNAL_RATES['extreme']
    else:
        signals_per_hour = SIGNAL_RATES['extreme']

    signals_per_day = signals_per_hour * 24

    print(f"""
CONFIGURATION:
- Starting Capital: ${c['capital']:,}
- Position Size: {c['position_pct']:.0%} of capital (${c['capital'] * c['position_pct']:,.0f})
- Minimum Signal: {c['min_signal']} (trade only {c['min_signal']}+ signals)
- Fee Rate: {c['fee_rate']:.1%}
- Expected Trades: ~{signals_per_day}/day ({signals_per_hour}/hour)

EXPECTED PERFORMANCE:
- Average Daily P&L: ${best['avg_pnl']:+,.2f}
- Win Rate: {best['avg_win_rate']:.1%}
- Profitable Days: {best['profitable']}/100
- Days Hitting $10k: {best['hit_10k']}/100
- Best Day: ${best['max_pnl']:+,.2f}
- Worst Day: ${best['min_pnl']:+,.2f}

MONTHLY PROJECTION: ${best['avg_pnl'] * 30:+,.2f}
""")

    # Show minimum for $10k
    print(f"{'='*70}")
    print("WHAT YOU NEED FOR $10K/DAY")
    print(f"{'='*70}")

    if avg_10k:
        min_cap = min(r['config']['capital'] for r in avg_10k)
        min_cap_results = [r for r in avg_10k if r['config']['capital'] == min_cap]
        best_min = max(min_cap_results, key=lambda x: x['avg_pnl'])
        c = best_min['config']

        print(f"""
MINIMUM CONFIGURATION FOR $10K/DAY AVERAGE:

- Capital: ${c['capital']:,}
- Position: {c['position_pct']:.0%} (${c['capital'] * c['position_pct']:,.0f} per trade)
- Signal Filter: {c['min_signal']}+ only
- Fee Rate: {c['fee_rate']:.1%}

Results:
- Avg Daily P&L: ${best_min['avg_pnl']:+,.2f}
- Hit $10k: {best_min['hit_10k']}/100 days
- Win Rate: {best_min['avg_win_rate']:.1%}
""")
    else:
        # Find scaling needed
        if best['avg_pnl'] > 0:
            scale = 10000 / best['avg_pnl']
            needed = c['capital'] * scale
            print(f"""
NO CONFIG AVERAGES $10K/DAY

Best achieves ${best['avg_pnl']:+,.2f}/day

To reach $10k/day average, you need:
- Capital: ~${needed:,.0f}
  (scaling from ${c['capital']:,} with {c['position_pct']:.0%} position)
- Or better fee rates
- Or more extreme signal opportunities
""")

    # The math
    print(f"\n{'='*70}")
    print("THE MATH BEHIND IT")
    print(f"{'='*70}")

    # Calculate for best config
    c = results[0]['config']
    pos_size = c['capital'] * c['position_pct']

    # Get weighted accuracy based on signal filter
    if c['min_signal'] == 'moderate':
        weighted_acc = (SIGNAL_RATES['moderate'] * ACCURACY['moderate'] +
                       SIGNAL_RATES['strong'] * ACCURACY['strong'] +
                       SIGNAL_RATES['extreme'] * ACCURACY['extreme']) / 7
    elif c['min_signal'] == 'strong':
        weighted_acc = (SIGNAL_RATES['strong'] * ACCURACY['strong'] +
                       SIGNAL_RATES['extreme'] * ACCURACY['extreme']) / 3
    else:
        weighted_acc = ACCURACY['extreme']

    win_value = 0.5 * pos_size
    lose_value = 0.5 * pos_size
    fee_cost = pos_size * c['fee_rate']

    ev_per_trade = weighted_acc * win_value - (1 - weighted_acc) * lose_value - fee_cost

    signals_day = signals_per_day
    ev_per_day = ev_per_trade * signals_day

    print(f"""
For best config (${c['capital']:,}, {c['position_pct']:.0%}, {c['min_signal']}):

Per Trade:
- Position: ${pos_size:,.2f}
- Win (+50%): ${win_value:,.2f}
- Lose (-50%): ${lose_value:,.2f}
- Fee: ${fee_cost:,.2f}
- Weighted Accuracy: {weighted_acc:.1%}
- Expected Value: ${ev_per_trade:,.2f}

Per Day ({signals_day} trades):
- Expected P&L: ${ev_per_day:,.2f}

FORMULA:
EV = (Accuracy x Win) - ((1-Accuracy) x Loss) - Fees
EV = ({weighted_acc:.2f} x ${win_value:,.0f}) - ({1-weighted_acc:.2f} x ${lose_value:,.0f}) - ${fee_cost:,.0f}
EV = ${ev_per_trade:,.2f} per trade
""")


if __name__ == "__main__":
    main()
