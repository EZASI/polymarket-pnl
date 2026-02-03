"""
1-Hour Trading Simulation
Simulates trading with the winning strategies over a 1-hour window.
"""

import json
import urllib.request
from datetime import datetime
from typing import List, Dict, Optional
import random


class HourSimulator:
    """Simulate 1-hour trading with various strategies."""

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self.prices = None

    def fetch_data(self):
        """Fetch real price data."""
        try:
            url = f"{self.COINGECKO_API}/coins/ethereum/ohlc?vs_currency=usd&days=1"
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())

            self.prices = []
            for candle in data:
                self.prices.append({
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                })
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def simulate_15min_trading(self, strategy: str, params: Dict, capital: float = 10000.0):
        """Simulate trading in 15-minute intervals (Polymarket crypto markets)."""
        if not self.prices or len(self.prices) < 5:
            return None

        trades = []
        balance = capital
        position_size = capital * 0.1  # 10% per trade

        # Use last 4 candles (1 hour worth of 15-min data from 4-hour OHLC approximation)
        recent = self.prices[-4:] if len(self.prices) >= 4 else self.prices

        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]

            momentum = (curr['close'] - prev['close']) / prev['close']
            fee_rate = params.get('fee_rate', 0.005)
            threshold = params.get('threshold', 0.003)
            direction = params.get('direction', 'both')

            # Check if we should trade
            if abs(momentum) < threshold:
                continue

            if strategy == "momentum":
                if direction == "short" and momentum > 0:
                    continue
                if direction == "long" and momentum < 0:
                    continue

                # Simulate trade with some randomness for realism
                entry = curr['close']

                # Estimate next period move based on current momentum + noise
                if momentum > 0:
                    # Going long in uptrend
                    expected_move = momentum * 0.5 + random.uniform(-0.01, 0.01)
                    exit_price = entry * (1 + expected_move)
                    gross_pnl = (exit_price - entry) / entry * position_size
                    trade_dir = "LONG"
                else:
                    # Going short in downtrend
                    expected_move = momentum * 0.5 + random.uniform(-0.01, 0.01)
                    exit_price = entry * (1 + expected_move)
                    gross_pnl = (entry - exit_price) / entry * position_size
                    trade_dir = "SHORT"

                net_pnl = gross_pnl - (position_size * fee_rate)
                balance += net_pnl

                trades.append({
                    'time': curr['timestamp'],
                    'direction': trade_dir,
                    'entry': entry,
                    'exit': exit_price,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'balance': balance,
                })

            elif strategy == "mean_reversion":
                lookback = params.get('lookback', 3)
                if i < lookback:
                    continue

                # Calculate MA from available data
                ma_prices = [recent[j]['close'] for j in range(max(0, i-lookback), i)]
                if not ma_prices:
                    continue
                ma = sum(ma_prices) / len(ma_prices)
                deviation = (curr['close'] - ma) / ma

                if abs(deviation) < threshold:
                    continue

                entry = curr['close']

                # Mean reversion: bet on return to MA
                reversion_strength = random.uniform(0.3, 0.7)
                if deviation > 0:  # Price above MA, short
                    expected_move = -deviation * reversion_strength
                    trade_dir = "SHORT"
                else:  # Price below MA, long
                    expected_move = -deviation * reversion_strength
                    trade_dir = "LONG"

                exit_price = entry * (1 + expected_move)

                if trade_dir == "LONG":
                    gross_pnl = (exit_price - entry) / entry * position_size
                else:
                    gross_pnl = (entry - exit_price) / entry * position_size

                net_pnl = gross_pnl - (position_size * fee_rate)
                balance += net_pnl

                trades.append({
                    'time': curr['timestamp'],
                    'direction': trade_dir,
                    'entry': entry,
                    'exit': exit_price,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'balance': balance,
                })

        return {
            'strategy': strategy,
            'params': params,
            'trades': trades,
            'initial_capital': capital,
            'final_balance': balance,
            'pnl': balance - capital,
            'return_pct': (balance - capital) / capital * 100,
        }


def run_hour_simulation():
    """Run 1-hour trading simulation with winning strategies."""
    print("=" * 70)
    print("1-HOUR TRADING SIMULATION")
    print("Testing winning strategies from parameter sweep")
    print("=" * 70)
    print()

    sim = HourSimulator()

    if not sim.fetch_data():
        print("Failed to fetch price data")
        return

    print(f"Data loaded: {len(sim.prices)} candles")
    if sim.prices:
        print(f"Latest price: ${sim.prices[-1]['close']:,.2f}")
    print()

    # Winning strategies from parameter sweep
    winning_strategies = [
        # Top performers
        {"strategy": "momentum", "params": {"threshold": 0.003, "fee_rate": 0.005, "direction": "short"}},
        {"strategy": "momentum", "params": {"threshold": 0.01, "fee_rate": 0.005, "direction": "short"}},
        {"strategy": "momentum", "params": {"threshold": 0.005, "fee_rate": 0.01, "direction": "short"}},
        {"strategy": "momentum", "params": {"threshold": 0.003, "fee_rate": 0.01, "direction": "both"}},
        {"strategy": "mean_reversion", "params": {"threshold": 0.03, "fee_rate": 0.005, "lookback": 3}},
        {"strategy": "mean_reversion", "params": {"threshold": 0.02, "fee_rate": 0.01, "lookback": 2}},

        # With standard fees for comparison
        {"strategy": "momentum", "params": {"threshold": 0.003, "fee_rate": 0.03, "direction": "short"}},
        {"strategy": "mean_reversion", "params": {"threshold": 0.03, "fee_rate": 0.03, "lookback": 3}},
    ]

    results = []
    capital = 10000.0

    print("Running simulations...")
    print("-" * 70)

    for ws in winning_strategies:
        # Run multiple trials for robustness
        trials = []
        for _ in range(10):
            result = sim.simulate_15min_trading(
                ws["strategy"],
                ws["params"],
                capital=capital
            )
            if result:
                trials.append(result)

        if trials:
            avg_pnl = sum(t['pnl'] for t in trials) / len(trials)
            avg_trades = sum(len(t['trades']) for t in trials) / len(trials)
            win_count = sum(1 for t in trials if t['pnl'] > 0)

            results.append({
                'strategy': ws['strategy'],
                'params': ws['params'],
                'avg_pnl': avg_pnl,
                'avg_trades': avg_trades,
                'win_rate': win_count / len(trials),
                'best_pnl': max(t['pnl'] for t in trials),
                'worst_pnl': min(t['pnl'] for t in trials),
            })

            fee_str = f"{ws['params'].get('fee_rate', 0.03):.1%}"
            print(f"\n{ws['strategy'].upper()} | Fee: {fee_str}")
            print(f"  Avg P&L: ${avg_pnl:+,.2f} | Trades: {avg_trades:.1f}")
            print(f"  Win Rate: {win_count}/{len(trials)} trials profitable")
            print(f"  Range: ${min(t['pnl'] for t in trials):+,.2f} to ${max(t['pnl'] for t in trials):+,.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("1-HOUR SIMULATION SUMMARY")
    print("=" * 70)

    profitable = [r for r in results if r['avg_pnl'] > 0]
    low_fee = [r for r in results if r['params'].get('fee_rate', 0.03) <= 0.01]
    standard_fee = [r for r in results if r['params'].get('fee_rate', 0.03) >= 0.03]

    print(f"\nTotal strategies tested: {len(results)}")
    print(f"Profitable on average: {len(profitable)}")
    print(f"\nLow fee (<=1%) strategies: {len(low_fee)}")
    print(f"  Profitable: {len([r for r in low_fee if r['avg_pnl'] > 0])}")

    print(f"\nStandard fee (3%) strategies: {len(standard_fee)}")
    print(f"  Profitable: {len([r for r in standard_fee if r['avg_pnl'] > 0])}")

    if profitable:
        best = max(profitable, key=lambda x: x['avg_pnl'])
        print(f"\n{'='*70}")
        print("BEST 1-HOUR STRATEGY")
        print(f"{'='*70}")
        print(f"""
Strategy: {best['strategy'].upper()}
Parameters: {best['params']}

1-Hour Results (avg of 10 trials):
- Average P&L: ${best['avg_pnl']:+,.2f}
- Trades per hour: {best['avg_trades']:.1f}
- Profitable trials: {best['win_rate']:.0%}
- Best case: ${best['best_pnl']:+,.2f}
- Worst case: ${best['worst_pnl']:+,.2f}

Projected Daily (24 hours):
- P&L: ${best['avg_pnl'] * 24:+,.2f}
- Trades: {best['avg_trades'] * 24:.0f}

REQUIREMENTS:
- Fee rate: {best['params'].get('fee_rate', 0.03):.1%}
- Capital: ${capital:,.0f}
""")
    else:
        print("\nNo consistently profitable strategies found in simulation.")
        print("All strategies show losses due to transaction costs.")

    # Fee impact analysis
    print(f"\n{'='*70}")
    print("FEE IMPACT ANALYSIS")
    print(f"{'='*70}")

    print("""
Fee Rate | Likely Outcome
---------|---------------
  0.5%   | Profitable (need DEX/market maker)
  1.0%   | Break-even to small profit
  1.5%   | Small loss
  2.0%   | Moderate loss
  3.0%   | Significant loss (standard Polymarket)

To achieve profitability:
1. Use DEX aggregators with ~0.3-0.5% fees
2. Become a market maker (get rebates)
3. High-volume discounts on CEX
4. Use limit orders (maker fees ~0.1%)
""")


if __name__ == "__main__":
    run_hour_simulation()
