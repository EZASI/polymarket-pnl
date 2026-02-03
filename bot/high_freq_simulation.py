"""
High-Frequency 1-Hour Simulation
Simulates 15-minute Polymarket crypto prediction markets.
Uses realistic price dynamics based on actual volatility.
"""

import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class MarketState:
    """Current state of a 15-min crypto prediction market."""
    coin: str
    start_price: float
    current_price: float
    end_time: datetime
    yes_price: float  # Price of "will go up" token
    no_price: float   # Price of "will go down" token
    volume: float
    obi: float  # Order book imbalance


def generate_realistic_price_path(
    start_price: float,
    periods: int = 4,  # 4 x 15-min = 1 hour
    volatility: float = 0.02,  # 2% per period typical for crypto
) -> List[float]:
    """Generate realistic price path using GBM."""
    prices = [start_price]
    drift = 0.0001  # Small positive drift

    for _ in range(periods):
        returns = drift + volatility * random.gauss(0, 1)
        prices.append(prices[-1] * (1 + returns))

    return prices


class PolymarketSimulator:
    """Simulate Polymarket 15-minute crypto markets."""

    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        self.balance = capital
        self.trades = []

    def reset(self):
        self.balance = self.capital
        self.trades = []

    def calculate_obi(self, yes_price: float, volume: float) -> float:
        """Simulate order book imbalance based on price and volume."""
        # OBI tends to be correlated with price deviation from 0.5
        price_signal = (yes_price - 0.5) * 2  # -1 to +1

        # Add noise
        noise = random.gauss(0, 0.2)

        # Volume affects signal strength
        vol_factor = min(volume / 10000, 1.5)

        obi = price_signal * vol_factor + noise
        return max(-1, min(1, obi))  # Clamp to [-1, 1]

    def run_market(
        self,
        coin: str,
        start_price: float,
        actual_end_price: float,
        fee_rate: float = 0.03,
    ) -> Dict:
        """Simulate a single 15-minute market."""
        # Determine outcome
        went_up = actual_end_price > start_price
        pct_change = (actual_end_price - start_price) / start_price

        # Market prices (probability estimates)
        # Markets are somewhat efficient but have noise
        if went_up:
            true_prob = 0.5 + abs(pct_change) * 5  # Higher change = higher prob
        else:
            true_prob = 0.5 - abs(pct_change) * 5

        true_prob = max(0.2, min(0.8, true_prob))  # Clamp

        # Add market inefficiency (this is where edge exists)
        market_noise = random.gauss(0, 0.08)
        yes_price = max(0.05, min(0.95, true_prob + market_noise))
        no_price = 1 - yes_price

        # Generate OBI
        volume = random.uniform(5000, 50000)
        obi = self.calculate_obi(yes_price, volume)

        return {
            'coin': coin,
            'start_price': start_price,
            'end_price': actual_end_price,
            'went_up': went_up,
            'pct_change': pct_change,
            'yes_price': yes_price,
            'no_price': no_price,
            'volume': volume,
            'obi': obi,
        }

    def trade_market(
        self,
        market: Dict,
        strategy: str,
        params: Dict,
        position_size: float,
    ) -> Optional[Dict]:
        """Execute a trade based on strategy."""
        obi = market['obi']
        yes_price = market['yes_price']
        went_up = market['went_up']

        should_trade = False
        predicted_up = None

        if strategy == "obi_momentum":
            threshold = params.get('obi_threshold', 0.3)
            if abs(obi) > threshold:
                should_trade = True
                predicted_up = obi > 0

        elif strategy == "obi_extreme":
            threshold = params.get('obi_threshold', 0.5)
            if abs(obi) > threshold:
                should_trade = True
                predicted_up = obi > 0

        elif strategy == "price_momentum":
            # Bet that price will continue in same direction
            threshold = params.get('price_threshold', 0.55)
            if yes_price > threshold:
                should_trade = True
                predicted_up = True
            elif yes_price < (1 - threshold):
                should_trade = True
                predicted_up = False

        elif strategy == "contrarian":
            # Bet against crowd when OBI is extreme
            threshold = params.get('obi_threshold', 0.4)
            if abs(obi) > threshold:
                should_trade = True
                predicted_up = obi < 0  # Opposite of OBI

        elif strategy == "value":
            # Buy underpriced outcomes
            if yes_price < 0.35:
                should_trade = True
                predicted_up = True
            elif yes_price > 0.65:
                should_trade = True
                predicted_up = False

        elif strategy == "combined":
            # Combine OBI + price signals
            obi_threshold = params.get('obi_threshold', 0.3)
            if abs(obi) > obi_threshold:
                obi_signal = obi > 0
                price_signal = yes_price > 0.5

                # Only trade if signals agree
                if obi_signal == price_signal:
                    should_trade = True
                    predicted_up = obi_signal

        if not should_trade:
            return None

        # Execute trade
        fee_rate = params.get('fee_rate', 0.03)

        if predicted_up:
            entry_price = yes_price
            won = went_up
        else:
            entry_price = 1 - yes_price  # no_price
            won = not went_up

        if won:
            # Win: get $1 per share, minus entry cost and fees
            gross_pnl = (1 - entry_price) * position_size
        else:
            # Lose: lose entry cost
            gross_pnl = -entry_price * position_size

        fees = position_size * fee_rate
        net_pnl = gross_pnl - fees

        return {
            'market': market['coin'],
            'strategy': strategy,
            'predicted_up': predicted_up,
            'actual_up': went_up,
            'entry_price': entry_price,
            'won': won,
            'gross_pnl': gross_pnl,
            'fees': fees,
            'net_pnl': net_pnl,
        }


def run_one_hour_simulation(
    strategy: str,
    params: Dict,
    capital: float = 10000.0,
    coins: List[str] = ["BTC", "ETH", "SOL", "DOGE"],
    num_trials: int = 100,
) -> Dict:
    """Run 1-hour simulation with multiple trials."""

    all_results = []

    for trial in range(num_trials):
        sim = PolymarketSimulator(capital)
        trial_trades = []

        # Generate 4 market periods per coin (1 hour = 4 x 15-min)
        for coin in coins:
            # Starting prices (realistic)
            start_prices = {
                "BTC": 78000 + random.uniform(-2000, 2000),
                "ETH": 2300 + random.uniform(-100, 100),
                "SOL": 100 + random.uniform(-10, 10),
                "DOGE": 0.08 + random.uniform(-0.01, 0.01),
            }

            base_price = start_prices.get(coin, 1000)
            prices = generate_realistic_price_path(base_price, periods=4)

            for i in range(4):  # 4 x 15-min markets
                market = sim.run_market(
                    coin=coin,
                    start_price=prices[i],
                    actual_end_price=prices[i + 1],
                    fee_rate=params.get('fee_rate', 0.03),
                )

                position_size = capital * params.get('position_pct', 0.05)
                trade = sim.trade_market(market, strategy, params, position_size)

                if trade:
                    sim.balance += trade['net_pnl']
                    trade['balance'] = sim.balance
                    trial_trades.append(trade)

        if trial_trades:
            wins = sum(1 for t in trial_trades if t['won'])
            total_pnl = sum(t['net_pnl'] for t in trial_trades)

            all_results.append({
                'trades': len(trial_trades),
                'wins': wins,
                'win_rate': wins / len(trial_trades),
                'total_pnl': total_pnl,
                'final_balance': sim.balance,
            })

    if not all_results:
        return {
            'strategy': strategy,
            'params': params,
            'avg_trades': 0,
            'avg_win_rate': 0,
            'avg_pnl': 0,
            'profitable_trials': 0,
            'best_pnl': 0,
            'worst_pnl': 0,
        }

    return {
        'strategy': strategy,
        'params': params,
        'avg_trades': sum(r['trades'] for r in all_results) / len(all_results),
        'avg_win_rate': sum(r['win_rate'] for r in all_results) / len(all_results),
        'avg_pnl': sum(r['total_pnl'] for r in all_results) / len(all_results),
        'profitable_trials': sum(1 for r in all_results if r['total_pnl'] > 0),
        'total_trials': len(all_results),
        'best_pnl': max(r['total_pnl'] for r in all_results),
        'worst_pnl': min(r['total_pnl'] for r in all_results),
    }


def main():
    """Run comprehensive 1-hour simulation."""
    print("=" * 70)
    print("HIGH-FREQUENCY 1-HOUR SIMULATION")
    print("Simulating Polymarket 15-minute crypto prediction markets")
    print("4 coins x 4 markets/hour = 16 trading opportunities per hour")
    print("=" * 70)
    print()

    # Strategies to test
    strategies = [
        # OBI-based strategies
        ("obi_momentum", {"obi_threshold": 0.2, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_momentum", {"obi_threshold": 0.3, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_momentum", {"obi_threshold": 0.4, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_extreme", {"obi_threshold": 0.5, "fee_rate": 0.03, "position_pct": 0.05}),
        ("obi_extreme", {"obi_threshold": 0.6, "fee_rate": 0.03, "position_pct": 0.05}),

        # With lower fees
        ("obi_momentum", {"obi_threshold": 0.3, "fee_rate": 0.015, "position_pct": 0.05}),
        ("obi_momentum", {"obi_threshold": 0.3, "fee_rate": 0.01, "position_pct": 0.05}),
        ("obi_momentum", {"obi_threshold": 0.3, "fee_rate": 0.005, "position_pct": 0.05}),

        # Price-based
        ("price_momentum", {"price_threshold": 0.55, "fee_rate": 0.03, "position_pct": 0.05}),
        ("price_momentum", {"price_threshold": 0.60, "fee_rate": 0.03, "position_pct": 0.05}),

        # Contrarian
        ("contrarian", {"obi_threshold": 0.4, "fee_rate": 0.03, "position_pct": 0.05}),
        ("contrarian", {"obi_threshold": 0.5, "fee_rate": 0.03, "position_pct": 0.05}),

        # Value
        ("value", {"fee_rate": 0.03, "position_pct": 0.05}),
        ("value", {"fee_rate": 0.015, "position_pct": 0.05}),

        # Combined
        ("combined", {"obi_threshold": 0.3, "fee_rate": 0.03, "position_pct": 0.05}),
        ("combined", {"obi_threshold": 0.3, "fee_rate": 0.015, "position_pct": 0.05}),
    ]

    results = []
    capital = 10000.0

    print(f"Testing {len(strategies)} strategy configurations...")
    print(f"Capital: ${capital:,.0f}")
    print(f"Running 100 trials each...")
    print()

    for strategy, params in strategies:
        result = run_one_hour_simulation(
            strategy=strategy,
            params=params,
            capital=capital,
            num_trials=100,
        )
        results.append(result)

        fee_pct = params.get('fee_rate', 0.03) * 100
        print(f"{strategy:15} | Fee: {fee_pct:.1f}% | "
              f"Win: {result['avg_win_rate']:.1%} | "
              f"P&L: ${result['avg_pnl']:+7.2f} | "
              f"Profitable: {result['profitable_trials']}/100")

    # Sort by P&L
    results.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 5 STRATEGIES BY P&L")
    print("=" * 70)

    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['strategy'].upper()}")
        print(f"   Parameters: {r['params']}")
        print(f"   Avg Trades/Hour: {r['avg_trades']:.1f}")
        print(f"   Win Rate: {r['avg_win_rate']:.1%}")
        print(f"   Avg P&L/Hour: ${r['avg_pnl']:+,.2f}")
        print(f"   Profitable Trials: {r['profitable_trials']}/100")
        print(f"   P&L Range: ${r['worst_pnl']:+,.2f} to ${r['best_pnl']:+,.2f}")

    print("\n" + "=" * 70)
    print("WORST 3 STRATEGIES")
    print("=" * 70)

    for i, r in enumerate(results[-3:], 1):
        print(f"\n{i}. {r['strategy'].upper()}")
        print(f"   Fee: {r['params'].get('fee_rate', 0.03):.1%}")
        print(f"   Avg P&L: ${r['avg_pnl']:+,.2f}")

    # Find profitable strategies
    profitable = [r for r in results if r['avg_pnl'] > 0]

    print("\n" + "=" * 70)
    print("PROFITABILITY SUMMARY")
    print("=" * 70)

    print(f"\nTotal strategies: {len(results)}")
    print(f"Profitable (avg): {len(profitable)}")

    if profitable:
        best = profitable[0]
        daily = best['avg_pnl'] * 24

        print(f"\n{'*' * 70}")
        print("BEST PROFITABLE STRATEGY")
        print(f"{'*' * 70}")
        print(f"""
Strategy: {best['strategy'].upper()}
Parameters: {best['params']}

1-HOUR RESULTS (avg of 100 trials):
- Trades: {best['avg_trades']:.1f}
- Win Rate: {best['avg_win_rate']:.1%}
- P&L: ${best['avg_pnl']:+,.2f}
- Profitable trials: {best['profitable_trials']}%

PROJECTIONS:
- Daily (24h): ${daily:+,.2f}
- Weekly: ${daily * 7:+,.2f}
- Monthly: ${daily * 30:+,.2f}

REQUIREMENTS:
- Fee rate: {best['params'].get('fee_rate', 0.03):.1%}
- Capital: ${capital:,.0f}
- Markets: 4 coins x 4 periods = 16 opportunities/hour
""")
    else:
        print("\nNo consistently profitable strategies at current fee levels.")

    # Fee analysis
    print("\n" + "=" * 70)
    print("FEE SENSITIVITY ANALYSIS")
    print("=" * 70)

    fee_groups = {}
    for r in results:
        fee = r['params'].get('fee_rate', 0.03)
        if fee not in fee_groups:
            fee_groups[fee] = []
        fee_groups[fee].append(r['avg_pnl'])

    print("\nAverage P&L by fee rate:")
    for fee in sorted(fee_groups.keys()):
        avg = sum(fee_groups[fee]) / len(fee_groups[fee])
        profitable = sum(1 for p in fee_groups[fee] if p > 0)
        print(f"  {fee:.1%} fees: ${avg:+,.2f} avg P&L, {profitable}/{len(fee_groups[fee])} profitable")


if __name__ == "__main__":
    main()
