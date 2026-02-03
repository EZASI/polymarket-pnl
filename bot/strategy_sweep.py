"""
Strategy Parameter Sweep
Tests multiple strategies and parameters to find profitable configurations.
"""

import json
import urllib.request
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random


@dataclass
class TradeResult:
    strategy: str
    params: Dict
    trades: int
    wins: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    sharpe_approx: float


class RealDataFetcher:
    """Fetch real market data."""

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self._cache = {}

    def _fetch_url(self, url: str) -> Optional[Dict]:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                }
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception:
            return None

    def get_prices(self, coin: str = "bitcoin", days: int = 7) -> List[Dict]:
        """Fetch OHLC data from CoinGecko."""
        cache_key = f"{coin}_{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{self.COINGECKO_API}/coins/{coin}/ohlc?vs_currency=usd&days={days}"
        data = self._fetch_url(url)

        if not data:
            return []

        prices = []
        for candle in data:
            prices.append({
                'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
            })

        self._cache[cache_key] = prices
        return prices


class StrategyTester:
    """Test various trading strategies."""

    def __init__(self, prices: List[Dict], position_size: float = 1000.0):
        self.prices = prices
        self.position_size = position_size

    def _calculate_returns(self, trades: List[Dict]) -> TradeResult:
        """Calculate strategy metrics from trades."""
        if not trades:
            return TradeResult(
                strategy="", params={}, trades=0, wins=0, win_rate=0,
                total_pnl=0, avg_pnl=0, max_drawdown=0, sharpe_approx=0
            )

        wins = sum(1 for t in trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in trades)

        # Calculate drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        pnls = []
        for t in trades:
            cumulative += t['pnl']
            pnls.append(t['pnl'])
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Approximate Sharpe
        if len(pnls) > 1:
            avg = sum(pnls) / len(pnls)
            variance = sum((p - avg) ** 2 for p in pnls) / len(pnls)
            std = variance ** 0.5
            sharpe = (avg / std) if std > 0 else 0
        else:
            sharpe = 0

        return TradeResult(
            strategy="",
            params={},
            trades=len(trades),
            wins=wins,
            win_rate=wins / len(trades) if trades else 0,
            total_pnl=total_pnl,
            avg_pnl=total_pnl / len(trades) if trades else 0,
            max_drawdown=max_dd,
            sharpe_approx=sharpe
        )

    def momentum_strategy(
        self,
        threshold: float = 0.005,
        fee_rate: float = 0.03,
        direction: str = "both",  # "long", "short", "both"
        lookback: int = 1,
    ) -> TradeResult:
        """Momentum-following strategy."""
        trades = []

        for i in range(lookback, len(self.prices) - 1):
            # Calculate momentum
            current = self.prices[i]['close']
            past = self.prices[i - lookback]['close']
            momentum = (current - past) / past

            if abs(momentum) < threshold:
                continue

            # Direction filter
            if direction == "long" and momentum < 0:
                continue
            if direction == "short" and momentum > 0:
                continue

            # Execute trade
            entry = current
            exit_price = self.prices[i + 1]['close']

            if momentum > 0:  # Long
                gross_pnl = (exit_price - entry) / entry * self.position_size
            else:  # Short
                gross_pnl = (entry - exit_price) / entry * self.position_size

            net_pnl = gross_pnl - (self.position_size * fee_rate)

            trades.append({
                'entry': entry,
                'exit': exit_price,
                'direction': 'LONG' if momentum > 0 else 'SHORT',
                'pnl': net_pnl,
            })

        result = self._calculate_returns(trades)
        result.strategy = "momentum"
        result.params = {
            'threshold': threshold,
            'fee_rate': fee_rate,
            'direction': direction,
            'lookback': lookback,
        }
        return result

    def mean_reversion_strategy(
        self,
        threshold: float = 0.01,
        fee_rate: float = 0.03,
        lookback: int = 3,
    ) -> TradeResult:
        """Mean reversion strategy - bet on price returning to mean."""
        trades = []

        for i in range(lookback, len(self.prices) - 1):
            # Calculate deviation from moving average
            ma = sum(self.prices[i-j]['close'] for j in range(lookback)) / lookback
            current = self.prices[i]['close']
            deviation = (current - ma) / ma

            if abs(deviation) < threshold:
                continue

            entry = current
            exit_price = self.prices[i + 1]['close']

            # Bet on reversion - if price is HIGH, go SHORT expecting drop
            if deviation > 0:  # Price above MA - short
                gross_pnl = (entry - exit_price) / entry * self.position_size
                direction = 'SHORT'
            else:  # Price below MA - long
                gross_pnl = (exit_price - entry) / entry * self.position_size
                direction = 'LONG'

            net_pnl = gross_pnl - (self.position_size * fee_rate)

            trades.append({
                'entry': entry,
                'exit': exit_price,
                'direction': direction,
                'pnl': net_pnl,
            })

        result = self._calculate_returns(trades)
        result.strategy = "mean_reversion"
        result.params = {
            'threshold': threshold,
            'fee_rate': fee_rate,
            'lookback': lookback,
        }
        return result

    def breakout_strategy(
        self,
        lookback: int = 5,
        fee_rate: float = 0.03,
    ) -> TradeResult:
        """Breakout strategy - trade when price breaks recent high/low."""
        trades = []

        for i in range(lookback, len(self.prices) - 1):
            highs = [self.prices[i-j]['high'] for j in range(1, lookback + 1)]
            lows = [self.prices[i-j]['low'] for j in range(1, lookback + 1)]

            recent_high = max(highs)
            recent_low = min(lows)
            current = self.prices[i]['close']

            entry = current
            exit_price = self.prices[i + 1]['close']

            if current > recent_high:  # Breakout up - long
                gross_pnl = (exit_price - entry) / entry * self.position_size
                direction = 'LONG'
            elif current < recent_low:  # Breakout down - short
                gross_pnl = (entry - exit_price) / entry * self.position_size
                direction = 'SHORT'
            else:
                continue

            net_pnl = gross_pnl - (self.position_size * fee_rate)

            trades.append({
                'entry': entry,
                'exit': exit_price,
                'direction': direction,
                'pnl': net_pnl,
            })

        result = self._calculate_returns(trades)
        result.strategy = "breakout"
        result.params = {
            'lookback': lookback,
            'fee_rate': fee_rate,
        }
        return result

    def volatility_strategy(
        self,
        vol_threshold: float = 0.02,
        fee_rate: float = 0.03,
        lookback: int = 3,
    ) -> TradeResult:
        """Trade during high volatility periods."""
        trades = []

        for i in range(lookback, len(self.prices) - 1):
            # Calculate recent volatility
            returns = []
            for j in range(lookback):
                if i - j - 1 >= 0:
                    r = (self.prices[i-j]['close'] - self.prices[i-j-1]['close']) / self.prices[i-j-1]['close']
                    returns.append(abs(r))

            if not returns:
                continue

            avg_vol = sum(returns) / len(returns)

            if avg_vol < vol_threshold:
                continue

            # During high vol, follow the trend
            current = self.prices[i]['close']
            prev = self.prices[i-1]['close']
            entry = current
            exit_price = self.prices[i + 1]['close']

            if current > prev:  # Trending up
                gross_pnl = (exit_price - entry) / entry * self.position_size
                direction = 'LONG'
            else:  # Trending down
                gross_pnl = (entry - exit_price) / entry * self.position_size
                direction = 'SHORT'

            net_pnl = gross_pnl - (self.position_size * fee_rate)

            trades.append({
                'entry': entry,
                'exit': exit_price,
                'direction': direction,
                'pnl': net_pnl,
            })

        result = self._calculate_returns(trades)
        result.strategy = "volatility"
        result.params = {
            'vol_threshold': vol_threshold,
            'fee_rate': fee_rate,
            'lookback': lookback,
        }
        return result

    def selective_momentum(
        self,
        threshold: float = 0.015,
        fee_rate: float = 0.015,  # Lower fees for DEX/better execution
        min_strength: float = 0.02,
    ) -> TradeResult:
        """Only trade strong momentum signals."""
        trades = []

        for i in range(2, len(self.prices) - 1):
            # Require consistent direction over 2 periods
            m1 = (self.prices[i]['close'] - self.prices[i-1]['close']) / self.prices[i-1]['close']
            m2 = (self.prices[i-1]['close'] - self.prices[i-2]['close']) / self.prices[i-2]['close']

            # Both must be in same direction and strong
            if m1 * m2 < 0:  # Different directions
                continue

            if abs(m1) < threshold or abs(m2) < threshold:
                continue

            combined = abs(m1) + abs(m2)
            if combined < min_strength:
                continue

            entry = self.prices[i]['close']
            exit_price = self.prices[i + 1]['close']

            if m1 > 0:  # Long
                gross_pnl = (exit_price - entry) / entry * self.position_size
            else:  # Short
                gross_pnl = (entry - exit_price) / entry * self.position_size

            net_pnl = gross_pnl - (self.position_size * fee_rate)

            trades.append({
                'entry': entry,
                'exit': exit_price,
                'direction': 'LONG' if m1 > 0 else 'SHORT',
                'pnl': net_pnl,
            })

        result = self._calculate_returns(trades)
        result.strategy = "selective_momentum"
        result.params = {
            'threshold': threshold,
            'fee_rate': fee_rate,
            'min_strength': min_strength,
        }
        return result


def run_parameter_sweep():
    """Run comprehensive parameter sweep."""
    print("=" * 70)
    print("STRATEGY PARAMETER SWEEP")
    print("Testing multiple strategies and parameters")
    print("=" * 70)
    print()

    # Fetch real data
    print("Fetching real market data...")
    fetcher = RealDataFetcher()

    btc_prices = fetcher.get_prices("bitcoin", days=7)
    eth_prices = fetcher.get_prices("ethereum", days=7)

    if not btc_prices:
        print("Failed to fetch price data")
        return

    print(f"BTC: {len(btc_prices)} candles")
    print(f"ETH: {len(eth_prices)} candles")
    print(f"Date range: {btc_prices[0]['timestamp']} to {btc_prices[-1]['timestamp']}")
    print()

    all_results = []

    # Test on both BTC and ETH
    for coin, prices in [("BTC", btc_prices), ("ETH", eth_prices)]:
        if not prices:
            continue

        print(f"\n{'='*70}")
        print(f"TESTING ON {coin}")
        print(f"{'='*70}")

        tester = StrategyTester(prices, position_size=1000.0)

        # 1. MOMENTUM STRATEGIES
        print("\n--- MOMENTUM STRATEGIES ---")
        for threshold in [0.003, 0.005, 0.01, 0.015, 0.02]:
            for fee in [0.03, 0.02, 0.015, 0.01, 0.005]:
                for direction in ["both", "long", "short"]:
                    result = tester.momentum_strategy(
                        threshold=threshold,
                        fee_rate=fee,
                        direction=direction,
                    )
                    result.params['coin'] = coin
                    all_results.append(result)

        # 2. MEAN REVERSION
        print("--- MEAN REVERSION STRATEGIES ---")
        for threshold in [0.005, 0.01, 0.015, 0.02, 0.03]:
            for fee in [0.03, 0.02, 0.015, 0.01, 0.005]:
                for lookback in [2, 3, 5]:
                    result = tester.mean_reversion_strategy(
                        threshold=threshold,
                        fee_rate=fee,
                        lookback=lookback,
                    )
                    result.params['coin'] = coin
                    all_results.append(result)

        # 3. BREAKOUT
        print("--- BREAKOUT STRATEGIES ---")
        for lookback in [3, 5, 7, 10]:
            for fee in [0.03, 0.02, 0.015, 0.01, 0.005]:
                result = tester.breakout_strategy(
                    lookback=lookback,
                    fee_rate=fee,
                )
                result.params['coin'] = coin
                all_results.append(result)

        # 4. VOLATILITY
        print("--- VOLATILITY STRATEGIES ---")
        for vol_thresh in [0.01, 0.015, 0.02, 0.03]:
            for fee in [0.03, 0.02, 0.015, 0.01, 0.005]:
                result = tester.volatility_strategy(
                    vol_threshold=vol_thresh,
                    fee_rate=fee,
                )
                result.params['coin'] = coin
                all_results.append(result)

        # 5. SELECTIVE MOMENTUM
        print("--- SELECTIVE MOMENTUM ---")
        for thresh in [0.01, 0.015, 0.02]:
            for fee in [0.03, 0.02, 0.015, 0.01, 0.005]:
                for min_str in [0.015, 0.02, 0.03]:
                    result = tester.selective_momentum(
                        threshold=thresh,
                        fee_rate=fee,
                        min_strength=min_str,
                    )
                    result.params['coin'] = coin
                    all_results.append(result)

    # Filter to only strategies with trades
    results_with_trades = [r for r in all_results if r.trades > 0]

    print(f"\n\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal configurations tested: {len(all_results)}")
    print(f"Configurations with trades: {len(results_with_trades)}")

    # Find profitable strategies
    profitable = [r for r in results_with_trades if r.total_pnl > 0]
    print(f"Profitable configurations: {len(profitable)}")

    # Sort by P&L
    results_with_trades.sort(key=lambda x: x.total_pnl, reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 STRATEGIES BY P&L")
    print("='*70")

    for i, r in enumerate(results_with_trades[:10], 1):
        print(f"\n{i}. {r.strategy.upper()} ({r.params.get('coin', 'N/A')})")
        print(f"   Params: {r.params}")
        print(f"   Trades: {r.trades} | Win Rate: {r.win_rate:.1%}")
        print(f"   P&L: ${r.total_pnl:+,.2f} | Avg: ${r.avg_pnl:+,.2f}")
        print(f"   Max DD: ${r.max_drawdown:,.2f} | Sharpe: {r.sharpe_approx:.2f}")

    print(f"\n{'='*70}")
    print("BOTTOM 5 (WORST) STRATEGIES")
    print("='*70")

    for i, r in enumerate(results_with_trades[-5:], 1):
        print(f"\n{i}. {r.strategy.upper()} ({r.params.get('coin', 'N/A')})")
        print(f"   P&L: ${r.total_pnl:+,.2f} | Win Rate: {r.win_rate:.1%}")

    # Analyze what makes strategies profitable
    print(f"\n{'='*70}")
    print("PROFITABILITY ANALYSIS")
    print("='*70")

    if profitable:
        # Average fee rate of profitable strategies
        avg_fee = sum(r.params.get('fee_rate', 0.03) for r in profitable) / len(profitable)
        avg_wr = sum(r.win_rate for r in profitable) / len(profitable)

        print(f"\nProfitable strategies characteristics:")
        print(f"  Average fee rate: {avg_fee:.1%}")
        print(f"  Average win rate: {avg_wr:.1%}")

        # Count by strategy type
        by_type = {}
        for r in profitable:
            by_type[r.strategy] = by_type.get(r.strategy, 0) + 1
        print(f"  By strategy type: {by_type}")

        # Fee breakdown
        low_fee = [r for r in profitable if r.params.get('fee_rate', 0.03) <= 0.01]
        print(f"  With fees <= 1%: {len(low_fee)}")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print("='*70")

    best = results_with_trades[0] if results_with_trades else None
    if best and best.total_pnl > 0:
        print(f"""
PROFITABLE STRATEGY FOUND!

Strategy: {best.strategy.upper()}
Coin: {best.params.get('coin', 'N/A')}
Parameters: {best.params}

Results:
- Trades: {best.trades}
- Win Rate: {best.win_rate:.1%}
- Total P&L: ${best.total_pnl:+,.2f}
- Average P&L/Trade: ${best.avg_pnl:+,.2f}

NOTE: This requires fee rate of {best.params.get('fee_rate', 0.03):.1%}
Standard Polymarket fees are ~3%, so this needs:
- DEX with lower fees, or
- Market maker rebates, or
- Higher volume for fee discounts
""")
    else:
        print("""
NO PROFITABLE STRATEGY at standard 3% fees.

The data shows that transaction costs eliminate all edge.
Profitability requires:
1. Lower fees (< 1.5%)
2. Higher frequency data (sub-minute)
3. Better signal accuracy (OBI at 500ms windows)
""")

    return results_with_trades


if __name__ == "__main__":
    run_parameter_sweep()
