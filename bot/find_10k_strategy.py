"""
Find Strategy to Make $10k
Aggressive parameter search ignoring standard fee constraints.
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass


# Academic accuracy rates
ACCURACY = {
    'weak': 0.52,
    'moderate': 0.58,
    'strong': 0.65,
    'extreme': 0.72,
}


@dataclass
class Config:
    capital: float
    position_pct: float
    obi_threshold: float
    fee_rate: float
    trades_per_hour: int
    accuracy: float


def simulate_hour(config: Config) -> Dict:
    """Simulate 1 hour of trading."""
    balance = config.capital
    position_size = config.capital * config.position_pct

    trades = []
    wins = 0

    for _ in range(config.trades_per_hour):
        # Determine if we win based on accuracy
        won = random.random() < config.accuracy

        # Polymarket payoff (simplified: entry at 0.5)
        entry_price = 0.5

        if won:
            gross_pnl = (1 - entry_price) * position_size  # Win $0.50 per $1
        else:
            gross_pnl = -entry_price * position_size  # Lose $0.50 per $1

        fees = position_size * config.fee_rate
        net_pnl = gross_pnl - fees

        balance += net_pnl
        if won:
            wins += 1

        trades.append(net_pnl)

    return {
        'final_balance': balance,
        'pnl': balance - config.capital,
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) if trades else 0,
    }


def simulate_day(config: Config, hours: int = 24) -> Dict:
    """Simulate a full day of trading."""
    balance = config.capital
    total_trades = 0
    total_wins = 0

    hourly_pnl = []

    for _ in range(hours):
        result = simulate_hour(config)
        pnl = result['pnl']
        hourly_pnl.append(pnl)
        balance += pnl
        total_trades += result['trades']
        total_wins += result['wins']

    return {
        'final_balance': balance,
        'daily_pnl': balance - config.capital,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'win_rate': total_wins / total_trades if total_trades else 0,
        'hourly_pnl': hourly_pnl,
    }


def run_trials(config: Config, num_trials: int = 100) -> Dict:
    """Run multiple trials and get statistics."""
    results = []

    for _ in range(num_trials):
        day_result = simulate_day(config)
        results.append(day_result)

    daily_pnls = [r['daily_pnl'] for r in results]

    return {
        'config': config,
        'avg_daily_pnl': sum(daily_pnls) / len(daily_pnls),
        'min_pnl': min(daily_pnls),
        'max_pnl': max(daily_pnls),
        'profitable_days': sum(1 for p in daily_pnls if p > 0),
        'hit_10k': sum(1 for p in daily_pnls if p >= 10000),
        'avg_win_rate': sum(r['win_rate'] for r in results) / len(results),
        'avg_trades': sum(r['total_trades'] for r in results) / len(results),
    }


def find_10k_strategy():
    """Find strategy configurations that can make $10k/day."""
    print("=" * 70)
    print("FINDING $10K/DAY STRATEGY")
    print("Testing aggressive configurations")
    print("=" * 70)
    print()

    # Parameters to test
    capitals = [10000, 25000, 50000, 100000, 150000]
    position_pcts = [0.05, 0.10, 0.15, 0.20, 0.25]
    obi_thresholds = [0.3, 0.4, 0.5, 0.6]  # Higher = fewer but better signals
    fee_rates = [0.005, 0.01, 0.015, 0.02]  # Ignoring 3%, testing lower

    # Trades per hour depends on threshold (higher threshold = fewer signals)
    trades_map = {0.3: 8, 0.4: 5, 0.5: 3, 0.6: 2}
    accuracy_map = {0.3: 0.58, 0.4: 0.62, 0.5: 0.68, 0.6: 0.72}

    all_results = []

    print("Testing configurations...")
    total = len(capitals) * len(position_pcts) * len(obi_thresholds) * len(fee_rates)
    tested = 0

    for capital in capitals:
        for pos_pct in position_pcts:
            for obi_thresh in obi_thresholds:
                for fee_rate in fee_rates:
                    config = Config(
                        capital=capital,
                        position_pct=pos_pct,
                        obi_threshold=obi_thresh,
                        fee_rate=fee_rate,
                        trades_per_hour=trades_map[obi_thresh],
                        accuracy=accuracy_map[obi_thresh],
                    )

                    result = run_trials(config, num_trials=50)
                    all_results.append(result)

                    tested += 1
                    if tested % 50 == 0:
                        print(f"  Tested {tested}/{total} configurations...")

    # Sort by average daily P&L
    all_results.sort(key=lambda x: x['avg_daily_pnl'], reverse=True)

    # Filter to those that can hit $10k
    can_hit_10k = [r for r in all_results if r['hit_10k'] > 0]
    consistently_10k = [r for r in all_results if r['avg_daily_pnl'] >= 10000]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Can hit $10k (at least once in 50 trials): {len(can_hit_10k)}")
    print(f"Average $10k+ daily: {len(consistently_10k)}")

    print(f"\n{'='*70}")
    print("TOP 10 STRATEGIES BY DAILY P&L")
    print(f"{'='*70}")

    for i, r in enumerate(all_results[:10], 1):
        c = r['config']
        print(f"\n{i}. Capital: ${c.capital:,} | Position: {c.position_pct:.0%} | "
              f"OBI: {c.obi_threshold} | Fee: {c.fee_rate:.1%}")
        print(f"   Accuracy: {c.accuracy:.0%} | Trades/day: {c.trades_per_hour * 24}")
        print(f"   Avg Daily P&L: ${r['avg_daily_pnl']:+,.2f}")
        print(f"   Win Rate: {r['avg_win_rate']:.1%}")
        print(f"   Profitable days: {r['profitable_days']}/50")
        print(f"   Hit $10k: {r['hit_10k']}/50 days")
        print(f"   P&L Range: ${r['min_pnl']:+,.2f} to ${r['max_pnl']:+,.2f}")

    # Find minimum requirements for $10k
    print(f"\n{'='*70}")
    print("MINIMUM REQUIREMENTS FOR $10K/DAY")
    print(f"{'='*70}")

    if consistently_10k:
        # Find minimum capital needed
        min_capital = min(r['config'].capital for r in consistently_10k)
        min_cap_results = [r for r in consistently_10k if r['config'].capital == min_capital]
        best_min_cap = max(min_cap_results, key=lambda x: x['avg_daily_pnl'])

        c = best_min_cap['config']
        print(f"""
MINIMUM CAPITAL CONFIGURATION:
- Capital Required: ${c.capital:,}
- Position Size: {c.position_pct:.0%} (${c.capital * c.position_pct:,.0f} per trade)
- OBI Threshold: {c.obi_threshold} ({c.accuracy:.0%} accuracy)
- Fee Rate: {c.fee_rate:.1%}
- Trades per day: {c.trades_per_hour * 24}

EXPECTED RESULTS:
- Average Daily P&L: ${best_min_cap['avg_daily_pnl']:+,.2f}
- Win Rate: {best_min_cap['avg_win_rate']:.1%}
- Profitable Days: {best_min_cap['profitable_days']}/50
- Days hitting $10k+: {best_min_cap['hit_10k']}/50
""")
    else:
        # Find what gets closest
        closest = all_results[0]
        c = closest['config']
        shortfall = 10000 - closest['avg_daily_pnl']

        print(f"""
NO CONFIGURATION AVERAGES $10K/DAY

Closest configuration:
- Capital: ${c.capital:,}
- Position: {c.position_pct:.0%}
- Average Daily P&L: ${closest['avg_daily_pnl']:+,.2f}
- Shortfall: ${shortfall:,.2f}

To reach $10k/day, you would need:
""")
        # Calculate what's needed
        if closest['avg_daily_pnl'] > 0:
            multiplier = 10000 / closest['avg_daily_pnl']
            needed_capital = c.capital * multiplier
            print(f"- Capital: ${needed_capital:,.0f} (at same position %)")
            print(f"- Or higher position sizing with current capital")

    # Best overall strategy
    best = all_results[0]
    c = best['config']

    print(f"\n{'*'*70}")
    print("RECOMMENDED STRATEGY")
    print(f"{'*'*70}")
    print(f"""
Configuration:
- Starting Capital: ${c.capital:,}
- Position Size: {c.position_pct:.0%} per trade (${c.capital * c.position_pct:,.0f})
- OBI Threshold: {c.obi_threshold} (only trade when |OBI| > {c.obi_threshold})
- Signal Accuracy: {c.accuracy:.0%}
- Fee Rate: {c.fee_rate:.1%}
- Trades per Hour: {c.trades_per_hour}
- Trades per Day: {c.trades_per_hour * 24}

Expected Performance:
- Daily P&L: ${best['avg_daily_pnl']:+,.2f}
- Win Rate: {best['avg_win_rate']:.1%}
- Profitable Days: {best['profitable_days']}/50 ({best['profitable_days']*2}%)
- Best Day: ${best['max_pnl']:+,.2f}
- Worst Day: ${best['min_pnl']:+,.2f}

Monthly Projection: ${best['avg_daily_pnl'] * 30:+,.2f}
""")

    # Show the math
    print(f"\n{'='*70}")
    print("THE MATH")
    print(f"{'='*70}")

    pos_size = c.capital * c.position_pct
    trades_day = c.trades_per_hour * 24

    # Expected value per trade
    win_payout = 0.5 * pos_size  # Win $0.50 per $1 at 50% entry
    lose_cost = 0.5 * pos_size   # Lose $0.50 per $1 at 50% entry
    fee_cost = pos_size * c.fee_rate

    ev_per_trade = (c.accuracy * win_payout) - ((1-c.accuracy) * lose_cost) - fee_cost
    ev_per_day = ev_per_trade * trades_day

    print(f"""
Per Trade:
- Position Size: ${pos_size:,.2f}
- Win Payout (50% of position): ${win_payout:,.2f}
- Lose Cost (50% of position): ${lose_cost:,.2f}
- Fee Cost: ${fee_cost:,.2f}
- Expected Value: ${ev_per_trade:,.2f}

Per Day ({trades_day} trades):
- Expected P&L: ${ev_per_day:,.2f}

Formula:
EV = (Accuracy × Win) - ((1-Accuracy) × Loss) - Fees
EV = ({c.accuracy:.2f} × ${win_payout:,.2f}) - ({1-c.accuracy:.2f} × ${lose_cost:,.2f}) - ${fee_cost:,.2f}
EV = ${c.accuracy * win_payout:,.2f} - ${(1-c.accuracy) * lose_cost:,.2f} - ${fee_cost:,.2f}
EV = ${ev_per_trade:,.2f} per trade
""")

    return all_results


if __name__ == "__main__":
    find_10k_strategy()
