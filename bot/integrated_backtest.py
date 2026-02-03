#!/usr/bin/env python3
"""
Integrated Realistic Backtest - No Capital Risk

Simulates market conditions where signals have academically-validated
predictive power to test if the strategy CAN work.

Key difference from previous backtests:
- Outcomes correlate with signal strength (as per academic research)
- OBI at 0.6+ -> 72% accuracy (Wang 2025)
- Liquidation cascade -> 57% accuracy (our backtest finding)
- Multiple signal confirmation -> higher accuracy

This answers: "If signals work as documented, is the strategy profitable?"
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict


@dataclass
class SimulatedTrade:
    timestamp: datetime
    coin: str
    direction: str  # "YES" or "NO"
    entry_price: float
    position_size: float
    signals: List[str]
    signal_strength: float
    predicted_win_prob: float
    actual_outcome: bool  # Did we win?
    pnl: float


class IntegratedBacktest:
    """
    Integrated backtest with realistic signal-outcome correlation.

    Academic accuracy rates used:
    - OBI extreme (>0.5): 65% base, up to 72% at >0.6
    - Liquidation cascade: 57%
    - Funding extreme: 52% (barely above random)
    - Multiple signals aligned: +5-10% boost
    - Signal disagreement: reduced to 50% (random)
    """

    # Academic accuracy rates
    ACCURACY_RATES = {
        'obi_moderate': 0.58,      # OBI 0.3-0.5
        'obi_strong': 0.65,        # OBI 0.5-0.6
        'obi_extreme': 0.72,       # OBI > 0.6 (Wang 2025)
        'liquidation': 0.57,       # From our backtest
        'funding_extreme': 0.52,   # Weak predictor
        'sentiment': 0.51,         # Nearly random
        'multi_signal_boost': 0.05, # Bonus for confirmation
    }

    # Polymarket costs
    TRANSACTION_COST = 0.03  # 3% round trip
    SLIPPAGE = 0.005  # 0.5%

    def __init__(
        self,
        initial_capital: float = 50000,
        position_size_pct: float = 0.02,
        max_position_pct: float = 0.05,
        max_daily_trades: int = 20,
        daily_loss_limit_pct: float = 0.03,
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_position_pct = max_position_pct
        self.max_daily_trades = max_daily_trades
        self.daily_loss_limit_pct = daily_loss_limit_pct

        # State
        self.capital = initial_capital
        self.trades: List[SimulatedTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_pnl: Dict[str, float] = {}

    def _generate_market_condition(self) -> Dict:
        """Generate random market conditions."""
        # OBI follows a distribution where extremes are rare
        obi = np.random.normal(0, 0.2)  # Most values near 0
        if np.random.random() < 0.15:  # 15% chance of extreme
            obi = np.random.choice([-1, 1]) * np.random.uniform(0.4, 0.8)
        obi = np.clip(obi, -1, 1)

        # Funding rate (mostly normal, occasionally extreme)
        funding_ann = np.random.normal(10, 20)  # 10% mean, 20% std
        if np.random.random() < 0.05:  # 5% chance of extreme
            funding_ann = np.random.choice([-1, 1]) * np.random.uniform(80, 150)

        # Liquidation value (power law - mostly small, rarely large)
        liquidation_value = np.random.pareto(2) * 1000  # Pareto distribution

        # Sentiment
        sentiment = np.random.normal(0, 0.3)
        sentiment = np.clip(sentiment, -1, 1)

        # Market price (probability shown on Polymarket)
        # Most markets are near 50%, some are skewed
        market_price = 0.5 + np.random.normal(0, 0.1)
        market_price = np.clip(market_price, 0.1, 0.9)

        return {
            'obi': obi,
            'funding_annualized': funding_ann,
            'liquidation_value': liquidation_value,
            'sentiment': sentiment,
            'market_yes_price': market_price,
        }

    def _evaluate_signals(self, conditions: Dict) -> Tuple[List[str], int, float]:
        """
        Evaluate market conditions and return signals.

        Returns: (signal_names, direction, predicted_accuracy)
        """
        signals = []
        direction_votes = []
        accuracy_estimates = []

        obi = conditions['obi']
        funding = conditions['funding_annualized']
        liq = conditions['liquidation_value']
        sentiment = conditions['sentiment']

        # OBI Signal
        if abs(obi) > 0.6:
            signals.append('OBI_EXTREME')
            direction_votes.append(1 if obi > 0 else -1)
            accuracy_estimates.append(self.ACCURACY_RATES['obi_extreme'])
        elif abs(obi) > 0.5:
            signals.append('OBI_STRONG')
            direction_votes.append(1 if obi > 0 else -1)
            accuracy_estimates.append(self.ACCURACY_RATES['obi_strong'])
        elif abs(obi) > 0.3:
            signals.append('OBI_MODERATE')
            direction_votes.append(1 if obi > 0 else -1)
            accuracy_estimates.append(self.ACCURACY_RATES['obi_moderate'])

        # Liquidation Signal (contrarian)
        if liq > 50000:  # $50k+ in liquidations
            signals.append('LIQUIDATION')
            # Assume longs liquidated = price dropped = bullish contrarian
            direction_votes.append(1 if np.random.random() > 0.5 else -1)
            accuracy_estimates.append(self.ACCURACY_RATES['liquidation'])

        # Funding Signal (contrarian, weak)
        if abs(funding) > 75:
            signals.append('FUNDING_EXTREME')
            direction_votes.append(-1 if funding > 0 else 1)  # Contrarian
            accuracy_estimates.append(self.ACCURACY_RATES['funding_extreme'])

        # No signals = no trade
        if not signals:
            return [], 0, 0.5

        # Calculate combined direction
        avg_direction = np.mean(direction_votes)
        if abs(avg_direction) < 0.3:
            # Signals disagree - skip
            return signals, 0, 0.5

        direction = 1 if avg_direction > 0 else -1

        # Calculate predicted accuracy
        base_accuracy = np.mean(accuracy_estimates)

        # Boost for multiple confirming signals
        if len(signals) >= 2:
            agreement = sum(1 for v in direction_votes if (v > 0) == (direction > 0)) / len(direction_votes)
            if agreement > 0.7:
                base_accuracy += self.ACCURACY_RATES['multi_signal_boost']

        # Cap accuracy at realistic levels
        predicted_accuracy = min(0.75, base_accuracy)

        return signals, direction, predicted_accuracy

    def _should_trade(
        self,
        signals: List[str],
        direction: int,
        accuracy: float,
        market_price: float,
        daily_trades: int,
        daily_pnl: float,
    ) -> Tuple[bool, str]:
        """Determine if we should take this trade."""

        # No signal
        if direction == 0:
            return False, "No clear direction"

        # Daily limits
        if daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached"

        if daily_pnl < -self.initial_capital * self.daily_loss_limit_pct:
            return False, "Daily loss limit reached"

        # Death zone (market too close to 50%)
        if 0.45 <= market_price <= 0.55:
            return False, "Market in death zone"

        # Calculate edge
        # Our predicted prob vs market price, minus costs
        our_price = market_price if direction > 0 else (1 - market_price)
        edge = accuracy - our_price - self.TRANSACTION_COST

        if edge < 0.02:  # Need 2% edge minimum
            return False, f"Edge too low: {edge:.2%}"

        # Require strong signal
        if accuracy < 0.55:
            return False, f"Accuracy too low: {accuracy:.2%}"

        return True, f"Trade: edge={edge:.2%}, accuracy={accuracy:.2%}"

    def _simulate_outcome(self, accuracy: float) -> bool:
        """Simulate trade outcome based on predicted accuracy."""
        return np.random.random() < accuracy

    def run(self, days: int = 30, coins: List[str] = None) -> Dict:
        """
        Run integrated backtest.

        Args:
            days: Number of days to simulate
            coins: List of coins to trade (default: BTC, ETH)
        """
        if coins is None:
            coins = ['BTC', 'ETH']

        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}

        start_date = datetime(2024, 1, 1)

        print("=" * 70)
        print("INTEGRATED REALISTIC BACKTEST")
        print("=" * 70)
        print(f"Initial Capital: ${self.initial_capital:,.0f} (PAPER - NO REAL MONEY)")
        print(f"Position Size: {self.position_size_pct:.1%} of capital")
        print(f"Max Daily Trades: {self.max_daily_trades}")
        print(f"Daily Loss Limit: {self.daily_loss_limit_pct:.1%}")
        print(f"Coins: {coins}")
        print(f"Days: {days}")
        print()
        print("Signal Accuracy Rates (from academic research):")
        for signal, rate in self.ACCURACY_RATES.items():
            print(f"  {signal}: {rate:.1%}")
        print("=" * 70)
        print()

        for day in range(days):
            current_date = start_date + timedelta(days=day)
            date_str = current_date.strftime("%Y-%m-%d")

            daily_trades = 0
            self.daily_pnl[date_str] = 0.0

            # 96 periods per day (15-min intervals)
            for period in range(96):
                current_time = current_date + timedelta(minutes=period * 15)

                # Check daily limits
                if daily_trades >= self.max_daily_trades:
                    break
                if self.daily_pnl[date_str] < -self.initial_capital * self.daily_loss_limit_pct:
                    break

                for coin in coins:
                    # Generate market conditions
                    conditions = self._generate_market_condition()

                    # Evaluate signals
                    signals, direction, accuracy = self._evaluate_signals(conditions)

                    # Should we trade?
                    should_trade, reason = self._should_trade(
                        signals, direction, accuracy,
                        conditions['market_yes_price'],
                        daily_trades,
                        self.daily_pnl[date_str],
                    )

                    if not should_trade:
                        continue

                    # Calculate position size
                    base_size = self.capital * self.position_size_pct
                    # Scale by signal strength
                    if 'OBI_EXTREME' in signals:
                        size_mult = 1.5
                    elif len(signals) >= 2:
                        size_mult = 1.25
                    else:
                        size_mult = 1.0

                    position_size = min(
                        base_size * size_mult,
                        self.capital * self.max_position_pct,
                        self.capital * 0.25,  # Never more than 25%
                    )

                    if position_size < 100:  # Minimum trade size
                        continue

                    # Entry price
                    market_price = conditions['market_yes_price']
                    if direction > 0:
                        entry_price = market_price * (1 + self.SLIPPAGE)
                        outcome_str = "YES"
                    else:
                        entry_price = (1 - market_price) * (1 + self.SLIPPAGE)
                        outcome_str = "NO"

                    # Simulate outcome
                    won = self._simulate_outcome(accuracy)

                    # Calculate P&L
                    if won:
                        # Win: receive $1 per share, minus entry cost
                        gross_pnl = position_size * (1.0 / entry_price - 1)
                    else:
                        # Lose: lose entire position
                        gross_pnl = -position_size

                    # Apply transaction costs
                    pnl = gross_pnl - (position_size * self.TRANSACTION_COST)

                    # Record trade
                    trade = SimulatedTrade(
                        timestamp=current_time,
                        coin=coin,
                        direction=outcome_str,
                        entry_price=entry_price,
                        position_size=position_size,
                        signals=signals,
                        signal_strength=accuracy,
                        predicted_win_prob=accuracy,
                        actual_outcome=won,
                        pnl=pnl,
                    )
                    self.trades.append(trade)

                    # Update state
                    self.capital += pnl
                    self.daily_pnl[date_str] += pnl
                    daily_trades += 1

                    self.equity_curve.append((current_time, self.capital))

            # Daily summary
            pnl = self.daily_pnl[date_str]
            status = "✓" if pnl > 0 else "✗" if pnl < 0 else "○"
            trades_today = sum(1 for t in self.trades
                             if t.timestamp.strftime("%Y-%m-%d") == date_str)

            print(f"Day {day+1:2d}: {status} Trades: {trades_today:2d} | "
                  f"P&L: ${pnl:+9,.2f} | Equity: ${self.capital:,.2f}")

        return self._calculate_results(days)

    def _calculate_results(self, days: int) -> Dict:
        """Calculate final results."""
        total_trades = len(self.trades)

        if total_trades == 0:
            print("\nNo trades executed!")
            return {'total_trades': 0}

        winners = [t for t in self.trades if t.actual_outcome]
        losers = [t for t in self.trades if not t.actual_outcome]

        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(winners) / total_trades

        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        peak = self.initial_capital
        max_dd = 0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)

        # Daily stats
        profitable_days = sum(1 for v in self.daily_pnl.values() if v > 0)
        losing_days = sum(1 for v in self.daily_pnl.values() if v < 0)
        flat_days = sum(1 for v in self.daily_pnl.values() if v == 0)

        # Signal breakdown
        signal_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        for trade in self.trades:
            for signal in trade.signals:
                if trade.actual_outcome:
                    signal_stats[signal]['wins'] += 1
                else:
                    signal_stats[signal]['losses'] += 1
                signal_stats[signal]['pnl'] += trade.pnl

        print()
        print("=" * 70)
        print("BACKTEST RESULTS (PAPER TRADING - NO REAL CAPITAL)")
        print("=" * 70)

        print(f"\n--- PERFORMANCE ---")
        print(f"Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"Final Capital:       ${self.capital:,.2f}")
        print(f"Total P&L:           ${total_pnl:+,.2f} ({total_pnl/self.initial_capital:+.2%})")
        print(f"Daily Avg P&L:       ${total_pnl/days:+,.2f}")
        print(f"Max Drawdown:        ${max_dd:,.2f} ({max_dd/self.initial_capital:.2%})")

        print(f"\n--- TRADE STATISTICS ---")
        print(f"Total Trades:        {total_trades}")
        print(f"Trades/Day:          {total_trades/days:.1f}")
        print(f"Win Rate:            {win_rate:.2%}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print(f"Avg Winner:          ${avg_winner:+,.2f}")
        print(f"Avg Loser:           ${avg_loser:+,.2f}")

        print(f"\n--- DAILY BREAKDOWN ---")
        print(f"Profitable Days:     {profitable_days}/{days} ({profitable_days/days:.1%})")
        print(f"Losing Days:         {losing_days}/{days}")
        print(f"Flat Days:           {flat_days}/{days}")
        if profitable_days > 0:
            avg_profit_day = np.mean([v for v in self.daily_pnl.values() if v > 0])
            print(f"Avg Profitable Day:  ${avg_profit_day:+,.2f}")
        if losing_days > 0:
            avg_loss_day = np.mean([v for v in self.daily_pnl.values() if v < 0])
            print(f"Avg Losing Day:      ${avg_loss_day:+,.2f}")

        print(f"\n--- SIGNAL PERFORMANCE ---")
        for signal, stats in sorted(signal_stats.items()):
            total = stats['wins'] + stats['losses']
            wr = stats['wins'] / total if total > 0 else 0
            avg_pnl = stats['pnl'] / total if total > 0 else 0
            print(f"  {signal}:")
            print(f"    Trades: {total}, Win Rate: {wr:.1%}, Avg P&L: ${avg_pnl:+,.2f}")

        print()
        print("=" * 70)

        # Projections
        if total_pnl > 0:
            daily_avg = total_pnl / days
            print(f"\n*** PROFITABLE STRATEGY ***")
            print(f"Daily Average:    ${daily_avg:+,.2f}")
            print(f"Monthly (30d):    ${daily_avg * 30:+,.2f}")
            print(f"Annual (252d):    ${daily_avg * 252:+,.2f}")
            print(f"Annual ROI:       {(daily_avg * 252) / self.initial_capital:.1%}")
        else:
            print(f"\n*** LOSING STRATEGY - NEEDS ADJUSTMENT ***")

        print("=" * 70)

        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'daily_avg_pnl': total_pnl / days,
            'trades_per_day': total_trades / days,
            'final_capital': self.capital,
        }


def main():
    """Run the integrated backtest."""
    print("\n" + "=" * 70)
    print("POLYMARKET PREDICTION BOT - INTEGRATED BACKTEST")
    print("NO REAL CAPITAL AT RISK - PAPER SIMULATION ONLY")
    print("=" * 70 + "\n")

    # Test with different configurations
    configs = [
        {
            'name': 'Conservative',
            'initial_capital': 50000,
            'position_size_pct': 0.02,
            'max_daily_trades': 10,
        },
        {
            'name': 'Moderate',
            'initial_capital': 50000,
            'position_size_pct': 0.03,
            'max_daily_trades': 15,
        },
        {
            'name': 'Aggressive',
            'initial_capital': 50000,
            'position_size_pct': 0.04,
            'max_daily_trades': 20,
        },
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*70}")
        print(f"TESTING: {config['name']} Configuration")
        print(f"{'='*70}\n")

        backtest = IntegratedBacktest(
            initial_capital=config['initial_capital'],
            position_size_pct=config['position_size_pct'],
            max_daily_trades=config['max_daily_trades'],
        )

        result = backtest.run(days=30, coins=['BTC', 'ETH'])
        results[config['name']] = result

        print("\n" + "-" * 70 + "\n")

    # Summary comparison
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Total P&L':>12} {'Win Rate':>10} {'Trades/Day':>12} {'Daily Avg':>12}")
    print("-" * 65)

    for name, result in results.items():
        print(f"{name:<15} ${result['total_pnl']:>+10,.0f} {result['win_rate']:>9.1%} "
              f"{result['trades_per_day']:>11.1f} ${result['daily_avg_pnl']:>+10,.0f}")

    print("\n" + "=" * 70)
    print("CONCLUSION: These are SIMULATED results assuming signals perform")
    print("as documented in academic literature. Real-world results may vary.")
    print("=" * 70)


if __name__ == "__main__":
    main()
