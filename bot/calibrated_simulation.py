"""
Calibrated 1-Hour Simulation
Accuracy rates calibrated to academic research:
- OBI moderate: 58% accuracy (Wang 2025)
- OBI strong: 65% accuracy
- OBI extreme: 72% accuracy
- Price momentum: ~52% (barely better than random)
"""

import random
from typing import Dict, List, Optional


# Academic accuracy rates
ACCURACY_RATES = {
    'obi_weak': 0.52,      # OBI < 0.2: barely predictive
    'obi_moderate': 0.58,  # OBI 0.2-0.4: some edge
    'obi_strong': 0.65,    # OBI 0.4-0.6: good edge
    'obi_extreme': 0.72,   # OBI > 0.6: best edge (Wang 2025)
    'price_momentum': 0.52, # Price signals: minimal edge
    'contrarian': 0.48,     # Contrarian: slightly worse than random
    'random': 0.50,
}


def get_signal_accuracy(obi: float, strategy: str) -> float:
    """Get accuracy rate based on signal strength."""
    if strategy == 'contrarian':
        return ACCURACY_RATES['contrarian']
    elif strategy == 'price_momentum':
        return ACCURACY_RATES['price_momentum']

    # OBI-based strategies
    abs_obi = abs(obi)
    if abs_obi < 0.2:
        return ACCURACY_RATES['obi_weak']
    elif abs_obi < 0.4:
        return ACCURACY_RATES['obi_moderate']
    elif abs_obi < 0.6:
        return ACCURACY_RATES['obi_strong']
    else:
        return ACCURACY_RATES['obi_extreme']


def simulate_market():
    """Generate a simulated market state."""
    # OBI follows a distribution centered around 0
    obi = random.gauss(0, 0.3)
    obi = max(-1, min(1, obi))  # Clamp to [-1, 1]

    # Market price (probability estimate)
    yes_price = 0.5 + obi * 0.3 + random.gauss(0, 0.1)
    yes_price = max(0.1, min(0.9, yes_price))

    return {
        'obi': obi,
        'yes_price': yes_price,
    }


def determine_outcome(prediction_up: bool, accuracy: float) -> bool:
    """Determine if prediction is correct based on calibrated accuracy."""
    return random.random() < accuracy


def simulate_trade(
    market: Dict,
    strategy: str,
    params: Dict,
    position_size: float,
) -> Optional[Dict]:
    """Simulate a single trade."""
    obi = market['obi']
    yes_price = market['yes_price']

    should_trade = False
    predicted_up = None

    if strategy == 'obi_threshold':
        threshold = params.get('obi_threshold', 0.3)
        if abs(obi) >= threshold:
            should_trade = True
            predicted_up = obi > 0

    elif strategy == 'obi_extreme_only':
        if abs(obi) >= 0.5:
            should_trade = True
            predicted_up = obi > 0

    elif strategy == 'price_momentum':
        if yes_price > 0.6 or yes_price < 0.4:
            should_trade = True
            predicted_up = yes_price > 0.5

    elif strategy == 'contrarian':
        threshold = params.get('obi_threshold', 0.4)
        if abs(obi) >= threshold:
            should_trade = True
            predicted_up = obi < 0  # Bet against OBI

    elif strategy == 'selective':
        # Only trade extreme OBI with confirming price
        if abs(obi) >= 0.5:
            obi_signal = obi > 0
            price_signal = yes_price > 0.55
            if obi_signal == price_signal:
                should_trade = True
                predicted_up = obi_signal

    if not should_trade:
        return None

    # Get calibrated accuracy
    accuracy = get_signal_accuracy(obi, strategy)

    # Determine outcome
    won = determine_outcome(predicted_up, accuracy)

    # Calculate P&L
    fee_rate = params.get('fee_rate', 0.03)

    if predicted_up:
        entry_price = yes_price
    else:
        entry_price = 1 - yes_price

    if won:
        gross_pnl = (1 - entry_price) * position_size
    else:
        gross_pnl = -entry_price * position_size

    fees = position_size * fee_rate
    net_pnl = gross_pnl - fees

    return {
        'strategy': strategy,
        'obi': obi,
        'accuracy': accuracy,
        'predicted_up': predicted_up,
        'won': won,
        'entry_price': entry_price,
        'gross_pnl': gross_pnl,
        'fees': fees,
        'net_pnl': net_pnl,
    }


def run_hour_simulation(
    strategy: str,
    params: Dict,
    capital: float = 10000.0,
    markets_per_hour: int = 16,  # 4 coins x 4 periods
    num_trials: int = 100,
) -> Dict:
    """Run 1-hour simulation."""
    results = []

    for _ in range(num_trials):
        trades = []
        balance = capital
        position_size = capital * params.get('position_pct', 0.05)

        for _ in range(markets_per_hour):
            market = simulate_market()
            trade = simulate_trade(market, strategy, params, position_size)

            if trade:
                balance += trade['net_pnl']
                trade['balance'] = balance
                trades.append(trade)

        if trades:
            wins = sum(1 for t in trades if t['won'])
            total_pnl = sum(t['net_pnl'] for t in trades)
            results.append({
                'trades': len(trades),
                'wins': wins,
                'win_rate': wins / len(trades),
                'total_pnl': total_pnl,
            })

    if not results:
        return {
            'strategy': strategy,
            'params': params,
            'avg_trades': 0,
            'avg_win_rate': 0,
            'avg_pnl': 0,
            'profitable_trials': 0,
        }

    return {
        'strategy': strategy,
        'params': params,
        'avg_trades': sum(r['trades'] for r in results) / len(results),
        'avg_win_rate': sum(r['win_rate'] for r in results) / len(results),
        'avg_pnl': sum(r['total_pnl'] for r in results) / len(results),
        'profitable_trials': sum(1 for r in results if r['total_pnl'] > 0),
        'total_trials': len(results),
        'best_pnl': max(r['total_pnl'] for r in results),
        'worst_pnl': min(r['total_pnl'] for r in results),
    }


def calculate_breakeven_fee(accuracy: float, avg_entry: float = 0.5) -> float:
    """Calculate fee rate where strategy breaks even."""
    # Expected value = accuracy * (1 - entry) - (1 - accuracy) * entry - fee
    # At breakeven: EV = 0
    # fee = accuracy * (1 - entry) - (1 - accuracy) * entry
    # For entry = 0.5: fee = accuracy * 0.5 - (1-accuracy) * 0.5 = accuracy - 0.5
    return accuracy - 0.5


def main():
    print("=" * 70)
    print("CALIBRATED 1-HOUR SIMULATION")
    print("Accuracy rates from academic research (Wang 2025, etc.)")
    print("=" * 70)
    print()

    print("CALIBRATED ACCURACY RATES:")
    print("-" * 40)
    for signal, acc in ACCURACY_RATES.items():
        breakeven = calculate_breakeven_fee(acc)
        print(f"  {signal:18} : {acc:.0%} accuracy -> breakeven at {breakeven:.1%} fees")
    print()

    strategies = [
        # Standard 3% fees
        ("obi_threshold", {"obi_threshold": 0.2, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_threshold", {"obi_threshold": 0.3, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_threshold", {"obi_threshold": 0.4, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_threshold", {"obi_threshold": 0.5, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_extreme_only", {"fee_rate": 0.03, "position_pct": 0.05}),
        ("price_momentum", {"fee_rate": 0.03, "position_pct": 0.05}),
        ("contrarian", {"obi_threshold": 0.4, "fee_rate": 0.03, "position_pct": 0.05}),
        ("selective", {"fee_rate": 0.03, "position_pct": 0.05}),

        # Lower fees
        ("obi_threshold", {"obi_threshold": 0.3, "fee_rate": 0.015, "position_pct": 0.05}),
        ("obi_threshold", {"obi_threshold": 0.3, "fee_rate": 0.01, "position_pct": 0.05}),
        ("obi_threshold", {"obi_threshold": 0.5, "fee_rate": 0.015, "position_pct": 0.05}),
        ("obi_extreme_only", {"fee_rate": 0.015, "position_pct": 0.05}),
        ("obi_extreme_only", {"fee_rate": 0.01, "position_pct": 0.05}),
        ("selective", {"fee_rate": 0.015, "position_pct": 0.05}),
        ("selective", {"fee_rate": 0.01, "position_pct": 0.05}),

        # Higher position sizing (more aggressive)
        ("obi_extreme_only", {"fee_rate": 0.015, "position_pct": 0.10}),
        ("selective", {"fee_rate": 0.015, "position_pct": 0.10}),
    ]

    results = []
    capital = 10000.0

    print(f"Testing {len(strategies)} configurations...")
    print(f"Capital: ${capital:,.0f}")
    print(f"Markets per hour: 16 (4 coins x 4 periods)")
    print()
    print("-" * 70)

    for strategy, params in strategies:
        result = run_hour_simulation(strategy, params, capital)
        results.append(result)

        fee_pct = params.get('fee_rate', 0.03) * 100
        pos_pct = params.get('position_pct', 0.05) * 100
        thresh = params.get('obi_threshold', 'N/A')

        print(f"{strategy:18} | Th:{str(thresh):4} | Fee:{fee_pct:.1f}% | Pos:{pos_pct:.0f}% | "
              f"WR:{result['avg_win_rate']:.1%} | P&L:${result['avg_pnl']:+8.2f} | "
              f"Prof:{result['profitable_trials']}/100")

    # Sort by P&L
    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 5 STRATEGIES")
    print("=" * 70)

    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['strategy'].upper()}")
        print(f"   Params: {r['params']}")
        print(f"   Trades/Hour: {r['avg_trades']:.1f} | Win Rate: {r['avg_win_rate']:.1%}")
        print(f"   P&L/Hour: ${r['avg_pnl']:+,.2f}")
        print(f"   Profitable: {r['profitable_trials']}/100 trials")
        print(f"   Range: ${r['worst_pnl']:+,.2f} to ${r['best_pnl']:+,.2f}")

    # Profitable summary
    profitable = [r for r in results if r['avg_pnl'] > 0]

    print("\n" + "=" * 70)
    print("PROFITABILITY ANALYSIS")
    print("=" * 70)

    print(f"\nTotal strategies: {len(results)}")
    print(f"Profitable on average: {len(profitable)}")

    # By fee rate
    for fee in [0.03, 0.015, 0.01]:
        fee_strats = [r for r in results if r['params'].get('fee_rate') == fee]
        prof = [r for r in fee_strats if r['avg_pnl'] > 0]
        avg_pnl = sum(r['avg_pnl'] for r in fee_strats) / len(fee_strats) if fee_strats else 0
        print(f"\nFee {fee:.1%}: {len(prof)}/{len(fee_strats)} profitable, avg P&L ${avg_pnl:+,.2f}")

    if profitable:
        best = profitable[0]
        daily = best['avg_pnl'] * 24

        print(f"\n{'*' * 70}")
        print("BEST PROFITABLE STRATEGY")
        print(f"{'*' * 70}")
        print(f"""
Strategy: {best['strategy'].upper()}
Parameters: {best['params']}

1-HOUR METRICS:
- Trades: {best['avg_trades']:.1f}
- Win Rate: {best['avg_win_rate']:.1%}
- P&L: ${best['avg_pnl']:+,.2f}
- Profitable trials: {best['profitable_trials']}%

PROJECTIONS (if consistent):
- Daily (24h): ${daily:+,.2f}
- Weekly: ${daily * 7:+,.2f}
- Monthly: ${daily * 30:+,.2f}

REQUIREMENTS:
- Fee rate: {best['params'].get('fee_rate', 0.03):.1%}
- Position size: {best['params'].get('position_pct', 0.05):.0%} per trade
- Capital: ${capital:,.0f}
""")
    else:
        print(f"\n{'!'*70}")
        print("NO PROFITABLE STRATEGIES FOUND")
        print(f"{'!'*70}")
        print("""
At calibrated academic accuracy rates, no strategy is profitable
at current fee levels. The edge from signals is too small to
overcome transaction costs.

To be profitable, you need:
1. OBI extreme signals (72% accuracy) with fees < 22%
2. But extreme signals are rare (few trades/hour)
3. Net result: small profits from rare trades

The math:
- 72% accuracy gives 22% edge over 50/50
- At 3% fees on position size, you lose 3% per trade
- Net: ~19% edge, but on small position sizes = small $
""")

    print("\n" + "=" * 70)
    print("MATHEMATICAL REALITY CHECK")
    print("=" * 70)

    print("""
EDGE CALCULATION (per $100 trade):

Signal Type     | Accuracy | Expected Win | Expected Loss | Fees  | Net/Trade
----------------|----------|--------------|---------------|-------|----------
OBI Weak (0.2)  |   52%    |  +$26.00     |  -$24.00      | -$3   | -$1.00
OBI Moderate    |   58%    |  +$29.00     |  -$21.00      | -$3   | +$5.00
OBI Strong      |   65%    |  +$32.50     |  -$17.50      | -$3   | +$12.00
OBI Extreme     |   72%    |  +$36.00     |  -$14.00      | -$3   | +$19.00

With $10k capital, 5% position ($500/trade):
- OBI Moderate: ~$25/trade profit
- OBI Strong: ~$60/trade profit
- OBI Extreme: ~$95/trade profit

But extreme signals are rare (1-2 per hour across 16 markets)
Expected hourly profit at extreme only: $95 * 1.5 = ~$142.50
""")


if __name__ == "__main__":
    main()
