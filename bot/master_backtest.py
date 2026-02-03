"""
MASTER BACKTEST - Find THE Strategy That Actually Works

Tests ALL strategies against ALL available datasets to find the ONE
that consistently makes money with real backtesting.
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import statistics


@dataclass
class Trade:
    """Single trade result."""
    market_id: str
    strategy: str
    entry_price: float
    exit_price: float
    outcome: str
    profit: float
    timestamp: float


@dataclass
class StrategyResult:
    """Results for a single strategy."""
    name: str
    trades: int
    wins: int
    losses: int
    total_profit: float
    win_rate: float
    avg_profit_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    profitable: bool


class DataLoader:
    """Load data from all available sources."""
    
    def __init__(self):
        self.kaggle_csv_path = "bot/data/polymarket_markets.csv"
        self.full_market_path = "bot/data/full_market/Polymarket_dataset/Polymarket_dataset"
        self.markets = []
        
    def load_all_data(self) -> List[Dict]:
        """Load from all available sources."""
        markets = []
        
        # Try Kaggle CSV
        if os.path.exists(self.kaggle_csv_path):
            print(f"Loading Kaggle CSV: {self.kaggle_csv_path}")
            markets.extend(self._load_kaggle_csv())
        
        # Try full market dataset
        if os.path.exists(self.full_market_path):
            print(f"Loading full market dataset: {self.full_market_path}")
            markets.extend(self._load_full_market_data())
        
        print(f"Total markets loaded: {len(markets)}")
        return markets
    
    def _load_kaggle_csv(self) -> List[Dict]:
        """Load from Kaggle CSV."""
        markets = []
        try:
            with open(self.kaggle_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        prices_str = row.get('outcomePrices', '[]')
                        prices = json.loads(prices_str) if prices_str else []
                        prices = [float(p) for p in prices] if prices else [0.5, 0.5]
                        
                        markets.append({
                            'id': row.get('id', ''),
                            'yes_price': prices[0] if len(prices) > 0 else 0.5,
                            'no_price': prices[1] if len(prices) > 1 else 0.5,
                            'volume': float(row.get('volume', 0) or 0),
                            'liquidity': float(row.get('liquidity', 0) or 0),
                            'resolved': row.get('resolved', '').lower() == 'true',
                            'closed': row.get('closed', '').lower() == 'true',
                            'final_price': prices[0] if prices[0] >= 0.95 or prices[0] <= 0.05 else None,
                            'category': row.get('groupItemTitle', 'Unknown'),
                        })
                    except:
                        continue
        except Exception as e:
            print(f"Error loading Kaggle CSV: {e}")
        
        return markets
    
    def _load_full_market_data(self, max_markets=5000) -> List[Dict]:
        """Load from full market dataset."""
        markets = []
        market_dirs = sorted(Path(self.full_market_path).glob('market=*'))[:max_markets]
        
        for md in market_dirs:
            market_id = md.name.replace('market=', '')
            prices = []
            
            # Load price data
            for pf in md.glob('price/*.ndjson'):
                try:
                    with open(pf, 'r') as f:
                        content = f.read().strip()
                        for part in content.replace('} {', '}\n{').split('\n'):
                            if part.strip():
                                try:
                                    data = json.loads(part.strip())
                                    if 't' in data and 'p' in data:
                                        prices.append({'ts': data['t'], 'p': float(data['p'])})
                                except:
                                    pass
                except:
                    pass
            
            if not prices:
                continue
            
            prices = sorted(prices, key=lambda x: x['ts'])
            final_price = prices[-1]['p']
            
            # Only include resolved markets
            if not (final_price >= 0.95 or final_price <= 0.05):
                continue
            
            # Calculate volume from trades
            trades = []
            for tf in md.glob('trade/*.ndjson'):
                try:
                    with open(tf, 'r') as f:
                        content = f.read().strip()
                        for part in content.replace('} {', '}\n{').split('\n'):
                            if part.strip():
                                try:
                                    data = json.loads(part.strip())
                                    trades.append({
                                        'ts': data.get('timestamp', 0),
                                        'price': float(data.get('price', 0)),
                                        'size': float(data.get('size', 0)),
                                    })
                                except:
                                    pass
                except:
                    pass
            
            total_volume = sum(t['size'] * t['price'] for t in trades) if trades else 0
            
            markets.append({
                'id': market_id[:42],
                'yes_price': prices[-1]['p'],
                'no_price': 1 - prices[-1]['p'],
                'prices': prices,
                'trades': trades,
                'volume': total_volume,
                'liquidity': total_volume * 0.1,  # Estimate
                'resolved': True,
                'closed': True,
                'final_price': final_price,
                'min_price': min(p['p'] for p in prices),
                'max_price': max(p['p'] for p in prices),
                'price_range': max(p['p'] for p in prices) - min(p['p'] for p in prices),
            })
        
        return markets


class StrategyTester:
    """Test all strategies systematically."""
    
    def __init__(self, markets: List[Dict]):
        self.markets = markets
        self.results = []
    
    def test_all_strategies(self) -> List[StrategyResult]:
        """Test every strategy."""
        print("\n" + "=" * 80)
        print("TESTING ALL STRATEGIES")
        print("=" * 80)
        
        strategies = [
            self._test_arbitrage,
            self._test_extreme_price_yes,
            self._test_extreme_price_no,
            self._test_volume_spike_follow,
            self._test_mean_reversion,
            self._test_momentum_95,
            self._test_momentum_05,
            self._test_price_gap,
            self._test_contrarian_extreme,
            self._test_liquidity_arbitrage,
        ]
        
        for strategy_func in strategies:
            try:
                result = strategy_func()
                if result:
                    self.results.append(result)
            except Exception as e:
                print(f"Error testing {strategy_func.__name__}: {e}")
        
        return sorted(self.results, key=lambda x: x.total_profit, reverse=True)
    
    def _test_arbitrage(self) -> Optional[StrategyResult]:
        """Pure arbitrage: YES + NO < $1."""
        trades = []
        
        for m in self.markets:
            if m.get('resolved'):
                continue  # Can't arbitrage resolved markets
            
            yes_price = m.get('yes_price', 0.5)
            no_price = m.get('no_price', 0.5)
            
            if yes_price + no_price < 0.99:  # Arbitrage exists
                profit = 1.0 - (yes_price + no_price)
                trades.append(Trade(
                    market_id=m['id'],
                    strategy='arbitrage',
                    entry_price=yes_price + no_price,
                    exit_price=1.0,
                    outcome='BOTH',
                    profit=profit,
                    timestamp=0
                ))
        
        return self._calculate_result('Pure Arbitrage', trades)
    
    def _test_extreme_price_yes(self) -> Optional[StrategyResult]:
        """Buy YES when price <= $0.05."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved'):
                continue
            
            yes_price = m.get('yes_price', 0.5)
            final = m.get('final_price', 0.5)
            
            if yes_price <= 0.05:
                outcome = 'YES' if final >= 0.95 else 'NO'
                if outcome == 'YES':
                    profit = 1.0 - yes_price
                else:
                    profit = -yes_price
                
                trades.append(Trade(
                    market_id=m['id'],
                    strategy='extreme_price_yes',
                    entry_price=yes_price,
                    exit_price=final,
                    outcome=outcome,
                    profit=profit,
                    timestamp=0
                ))
        
        return self._calculate_result('Extreme Price YES (≤$0.05)', trades)
    
    def _test_extreme_price_no(self) -> Optional[StrategyResult]:
        """Buy NO when YES price >= $0.95."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved'):
                continue
            
            yes_price = m.get('yes_price', 0.5)
            final = m.get('final_price', 0.5)
            
            if yes_price >= 0.95:
                no_price = 1 - yes_price
                outcome = 'NO' if final <= 0.05 else 'YES'
                if outcome == 'NO':
                    profit = 1.0 - no_price
                else:
                    profit = -no_price
                
                trades.append(Trade(
                    market_id=m['id'],
                    strategy='extreme_price_no',
                    entry_price=no_price,
                    exit_price=1 - final,
                    outcome=outcome,
                    profit=profit,
                    timestamp=0
                ))
        
        return self._calculate_result('Extreme Price NO (YES≥$0.95)', trades)
    
    def _test_volume_spike_follow(self) -> Optional[StrategyResult]:
        """Follow volume spikes."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved') or not m.get('trades'):
                continue
            
            trades_data = m.get('trades', [])
            if len(trades_data) < 10:
                continue
            
            # Calculate average volume
            volumes = [t['size'] for t in trades_data]
            avg_volume = statistics.mean(volumes) if volumes else 0
            
            if avg_volume == 0:
                continue
            
            # Find volume spikes (3x average)
            spike_trades = [t for t in trades_data if t['size'] >= avg_volume * 3]
            
            if not spike_trades:
                continue
            
            # Follow the direction of the spike
            final = m.get('final_price', 0.5)
            spike_price = spike_trades[-1]['price']  # Last spike
            
            # If spike was buying YES (price going up), bet YES
            if spike_price > 0.5:
                entry_price = spike_price
                outcome = 'YES' if final >= 0.95 else 'NO'
                if outcome == 'YES':
                    profit = 1.0 - entry_price
                else:
                    profit = -entry_price
            else:
                entry_price = 1 - spike_price
                outcome = 'NO' if final <= 0.05 else 'YES'
                if outcome == 'NO':
                    profit = 1.0 - entry_price
                else:
                    profit = -entry_price
            
            trades.append(Trade(
                market_id=m['id'],
                strategy='volume_spike',
                entry_price=entry_price,
                exit_price=final if spike_price > 0.5 else 1 - final,
                outcome=outcome,
                profit=profit,
                timestamp=spike_trades[-1]['ts']
            ))
        
        return self._calculate_result('Volume Spike Follow', trades)
    
    def _test_mean_reversion(self) -> Optional[StrategyResult]:
        """Mean reversion: Buy when price deviates from midpoint."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved') or not m.get('prices'):
                continue
            
            prices = m.get('prices', [])
            if len(prices) < 5:
                continue
            
            # Find price far from 0.5
            for i, p in enumerate(prices[:-1]):  # Don't use last price
                price = p['p']
                if price < 0.3:  # Buy YES (expect reversion up)
                    final = m.get('final_price', 0.5)
                    outcome = 'YES' if final >= 0.95 else 'NO'
                    if outcome == 'YES':
                        profit = 1.0 - price
                    else:
                        profit = -price
                    
                    trades.append(Trade(
                        market_id=m['id'],
                        strategy='mean_reversion',
                        entry_price=price,
                        exit_price=final,
                        outcome=outcome,
                        profit=profit,
                        timestamp=p['ts']
                    ))
                    break  # One trade per market
        
        return self._calculate_result('Mean Reversion', trades)
    
    def _test_momentum_95(self) -> Optional[StrategyResult]:
        """Momentum: Buy YES when price hits 95%."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved') or not m.get('prices'):
                continue
            
            prices = m.get('prices', [])
            final = m.get('final_price', 0.5)
            
            # Find first time price hits 0.95+
            for p in prices:
                if p['p'] >= 0.95:
                    entry_price = p['p']
                    outcome = 'YES' if final >= 0.95 else 'NO'
                    if outcome == 'YES':
                        profit = 1.0 - entry_price
                    else:
                        profit = -entry_price
                    
                    trades.append(Trade(
                        market_id=m['id'],
                        strategy='momentum_95',
                        entry_price=entry_price,
                        exit_price=final,
                        outcome=outcome,
                        profit=profit,
                        timestamp=p['ts']
                    ))
                    break
        
        return self._calculate_result('Momentum (Hit 95%)', trades)
    
    def _test_momentum_05(self) -> Optional[StrategyResult]:
        """Momentum: Buy NO when YES price hits 5%."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved') or not m.get('prices'):
                continue
            
            prices = m.get('prices', [])
            final = m.get('final_price', 0.5)
            
            # Find first time price hits 0.05 or lower
            for p in prices:
                if p['p'] <= 0.05:
                    entry_price = 1 - p['p']  # NO price
                    outcome = 'NO' if final <= 0.05 else 'YES'
                    if outcome == 'NO':
                        profit = 1.0 - entry_price
                    else:
                        profit = -entry_price
                    
                    trades.append(Trade(
                        market_id=m['id'],
                        strategy='momentum_05',
                        entry_price=entry_price,
                        exit_price=1 - final,
                        outcome=outcome,
                        profit=profit,
                        timestamp=p['ts']
                    ))
                    break
        
        return self._calculate_result('Momentum (Hit 5%)', trades)
    
    def _test_price_gap(self) -> Optional[StrategyResult]:
        """Price gap: Buy when large gap appears."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved') or not m.get('prices'):
                continue
            
            prices = m.get('prices', [])
            if len(prices) < 2:
                continue
            
            # Find large gaps
            for i in range(len(prices) - 1):
                gap = abs(prices[i+1]['p'] - prices[i]['p'])
                if gap > 0.2:  # 20% gap
                    # Buy the cheaper side
                    if prices[i]['p'] < 0.4:
                        entry_price = prices[i]['p']
                        final = m.get('final_price', 0.5)
                        outcome = 'YES' if final >= 0.95 else 'NO'
                        if outcome == 'YES':
                            profit = 1.0 - entry_price
                        else:
                            profit = -entry_price
                        
                        trades.append(Trade(
                            market_id=m['id'],
                            strategy='price_gap',
                            entry_price=entry_price,
                            exit_price=final,
                            outcome=outcome,
                            profit=profit,
                            timestamp=prices[i]['ts']
                        ))
                        break
        
        return self._calculate_result('Price Gap', trades)
    
    def _test_contrarian_extreme(self) -> Optional[StrategyResult]:
        """Contrarian: Bet against extreme prices."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved'):
                continue
            
            yes_price = m.get('yes_price', 0.5)
            final = m.get('final_price', 0.5)
            
            # Bet AGAINST extreme confidence
            if yes_price >= 0.90:  # Market very confident YES
                # Bet NO
                no_price = 1 - yes_price
                outcome = 'NO' if final <= 0.05 else 'YES'
                if outcome == 'NO':
                    profit = 1.0 - no_price
                else:
                    profit = -no_price
                
                trades.append(Trade(
                    market_id=m['id'],
                    strategy='contrarian',
                    entry_price=no_price,
                    exit_price=1 - final,
                    outcome=outcome,
                    profit=profit,
                    timestamp=0
                ))
        
        return self._calculate_result('Contrarian Extreme', trades)
    
    def _test_liquidity_arbitrage(self) -> Optional[StrategyResult]:
        """Liquidity arbitrage: Buy when liquidity is low."""
        trades = []
        
        for m in self.markets:
            if not m.get('resolved'):
                continue
            
            liquidity = m.get('liquidity', 0)
            yes_price = m.get('yes_price', 0.5)
            
            # Low liquidity might mean mispricing
            if liquidity < 1000 and yes_price < 0.3:
                final = m.get('final_price', 0.5)
                outcome = 'YES' if final >= 0.95 else 'NO'
                if outcome == 'YES':
                    profit = 1.0 - yes_price
                else:
                    profit = -yes_price
                
                trades.append(Trade(
                    market_id=m['id'],
                    strategy='liquidity_arb',
                    entry_price=yes_price,
                    exit_price=final,
                    outcome=outcome,
                    profit=profit,
                    timestamp=0
                ))
        
        return self._calculate_result('Liquidity Arbitrage', trades)
    
    def _calculate_result(self, name: str, trades: List[Trade]) -> Optional[StrategyResult]:
        """Calculate statistics for a strategy."""
        if not trades:
            return None
        
        wins = sum(1 for t in trades if t.profit > 0)
        losses = sum(1 for t in trades if t.profit <= 0)
        total_profit = sum(t.profit for t in trades)
        win_rate = wins / len(trades) if trades else 0
        avg_profit = total_profit / len(trades) if trades else 0
        
        # Calculate Sharpe ratio (simplified)
        profits = [t.profit for t in trades]
        if len(profits) > 1:
            mean_profit = statistics.mean(profits)
            std_profit = statistics.stdev(profits) if len(profits) > 1 else 1
            sharpe = mean_profit / std_profit if std_profit > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = 0
        max_dd = 0
        peak = 0
        for profit in profits:
            cumulative += profit
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        return StrategyResult(
            name=name,
            trades=len(trades),
            wins=wins,
            losses=losses,
            total_profit=total_profit,
            win_rate=win_rate,
            avg_profit_per_trade=avg_profit,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            profitable=total_profit > 0
        )


def main():
    """Run master backtest."""
    print("=" * 80)
    print("MASTER BACKTEST - Finding THE Strategy That Works")
    print("=" * 80)
    
    # Load data
    loader = DataLoader()
    markets = loader.load_all_data()
    
    if not markets:
        print("\n❌ No data found!")
        print("\nTo run this backtest, you need:")
        print("1. Kaggle dataset: bot/data/polymarket_markets.csv")
        print("   OR")
        print("2. Full market dataset: bot/data/full_market/")
        return
    
    print(f"\n✅ Loaded {len(markets)} markets")
    
    # Test all strategies
    tester = StrategyTester(markets)
    results = tester.test_all_strategies()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS - ALL STRATEGIES RANKED BY PROFIT")
    print("=" * 80)
    
    print(f"\n{'Strategy':<30} | {'Trades':<8} | {'Win%':<8} | {'Total $':<12} | {'Avg $':<10} | {'Profitable'}")
    print("-" * 100)
    
    for r in results:
        status = "✅ YES" if r.profitable else "❌ NO"
        print(f"{r.name:<30} | {r.trades:<8} | {r.win_rate*100:>6.1f}% | ${r.total_profit:>10,.2f} | ${r.avg_profit_per_trade:>8.2f} | {status}")
    
    # Find the winner
    if results:
        winner = results[0]
        print("\n" + "=" * 80)
        print("🏆 WINNING STRATEGY")
        print("=" * 80)
        print(f"""
Strategy: {winner.name}
Total Profit: ${winner.total_profit:,.2f}
Trades: {winner.trades}
Win Rate: {winner.win_rate*100:.1f}%
Average Profit/Trade: ${winner.avg_profit_per_trade:.2f}
Sharpe Ratio: {winner.sharpe_ratio:.2f}
Max Drawdown: ${winner.max_drawdown:.2f}

PROJECTION FOR $8K/DAY:
- Trades needed: {8000 / winner.avg_profit_per_trade:.0f} trades/day
- Capital per trade: ${1000:.0f} (assuming ${winner.avg_profit_per_trade:.2f} profit per $1000)
- Total capital needed: ${1000 * 8000 / winner.avg_profit_per_trade:,.0f}
        """)


if __name__ == '__main__':
    main()
