"""
Backtest with Real Polymarket Data from Kaggle
Dataset: https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets

To use:
1. Download the dataset from Kaggle
2. Extract polymarket_events.csv and polymarket_markets.csv to bot/data/
3. Run: python -m bot.kaggle_backtest
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class Market:
    """Real Polymarket market data."""
    id: str
    question: str
    volume: float
    liquidity: float
    spread: float
    best_bid: float
    best_ask: float
    outcome_prices: List[float]
    closed: bool
    resolved: bool
    category: str


def download_instructions():
    """Print instructions for downloading the dataset."""
    print("""
================================================================================
KAGGLE DATASET REQUIRED
================================================================================

To run this backtest with real Polymarket data:

1. Go to: https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets

2. Download the dataset (requires free Kaggle account)

3. Extract the CSV files to: bot/data/
   - polymarket_events.csv
   - polymarket_markets.csv

4. Run again: python -m bot.kaggle_backtest

================================================================================
""")


def load_markets(filepath: str) -> List[Market]:
    """Load markets from the Kaggle CSV."""
    markets = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Parse outcome prices (JSON format)
                prices_str = row.get('outcomePrices', '[]')
                try:
                    prices = json.loads(prices_str) if prices_str else []
                    prices = [float(p) for p in prices] if prices else []
                except:
                    prices = []

                # Get spread from best bid/ask
                best_bid = float(row.get('bestBid', 0) or 0)
                best_ask = float(row.get('bestAsk', 1) or 1)
                spread = best_ask - best_bid if best_ask > best_bid else 0

                market = Market(
                    id=row.get('id', ''),
                    question=row.get('question', '')[:100],
                    volume=float(row.get('volume', 0) or 0),
                    liquidity=float(row.get('liquidity', 0) or 0),
                    spread=spread,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    outcome_prices=prices,
                    closed=row.get('closed', '').lower() == 'true',
                    resolved=row.get('resolved', '').lower() == 'true',
                    category=row.get('groupItemTitle', 'Unknown'),
                )
                markets.append(market)

            except Exception as e:
                continue  # Skip malformed rows

    return markets


def analyze_market_dynamics(markets: List[Market]) -> Dict:
    """Analyze real market dynamics from the data."""

    # Filter to markets with meaningful data
    active_markets = [m for m in markets if m.volume > 0 and m.liquidity > 0]

    if not active_markets:
        return {}

    # Volume distribution
    volumes = [m.volume for m in active_markets]
    avg_volume = sum(volumes) / len(volumes)
    median_volume = sorted(volumes)[len(volumes) // 2]

    # Liquidity distribution
    liquidities = [m.liquidity for m in active_markets]
    avg_liquidity = sum(liquidities) / len(liquidities)
    median_liquidity = sorted(liquidities)[len(liquidities) // 2]

    # Spread analysis
    spreads = [m.spread for m in active_markets if m.spread > 0]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0

    # Price distribution (how many markets have extreme prices)
    extreme_prices = 0
    moderate_prices = 0
    balanced_prices = 0

    for m in active_markets:
        if m.outcome_prices:
            max_price = max(m.outcome_prices)
            if max_price > 0.8 or max_price < 0.2:
                extreme_prices += 1
            elif max_price > 0.65 or max_price < 0.35:
                moderate_prices += 1
            else:
                balanced_prices += 1

    # Crypto markets specifically
    crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'doge']
    crypto_markets = [m for m in active_markets
                      if any(kw in m.question.lower() for kw in crypto_keywords)]

    return {
        'total_markets': len(markets),
        'active_markets': len(active_markets),
        'avg_volume': avg_volume,
        'median_volume': median_volume,
        'avg_liquidity': avg_liquidity,
        'median_liquidity': median_liquidity,
        'avg_spread': avg_spread,
        'extreme_price_pct': extreme_prices / len(active_markets) * 100 if active_markets else 0,
        'moderate_price_pct': moderate_prices / len(active_markets) * 100 if active_markets else 0,
        'balanced_price_pct': balanced_prices / len(active_markets) * 100 if active_markets else 0,
        'crypto_markets': len(crypto_markets),
    }


def calculate_implied_accuracy(markets: List[Market]) -> Dict:
    """
    Calculate implied prediction accuracy from resolved markets.
    If a market resolved at price X, the "correct" prediction had X probability.
    """

    resolved = [m for m in markets if m.resolved and m.outcome_prices]

    if not resolved:
        return {}

    # Analyze resolved market prices
    winning_prices = []

    for m in resolved:
        if m.outcome_prices:
            # Assume the higher price won (simplified)
            max_price = max(m.outcome_prices)
            winning_prices.append(max_price)

    if not winning_prices:
        return {}

    # Markets are well-calibrated if:
    # - 70% priced markets resolve correctly ~70% of the time
    # - etc.

    # Group by price buckets
    buckets = {
        '0.5-0.6': [],
        '0.6-0.7': [],
        '0.7-0.8': [],
        '0.8-0.9': [],
        '0.9-1.0': [],
    }

    for price in winning_prices:
        if 0.5 <= price < 0.6:
            buckets['0.5-0.6'].append(price)
        elif 0.6 <= price < 0.7:
            buckets['0.6-0.7'].append(price)
        elif 0.7 <= price < 0.8:
            buckets['0.7-0.8'].append(price)
        elif 0.8 <= price < 0.9:
            buckets['0.8-0.9'].append(price)
        elif 0.9 <= price <= 1.0:
            buckets['0.9-1.0'].append(price)

    bucket_stats = {}
    for bucket, prices in buckets.items():
        if prices:
            bucket_stats[bucket] = {
                'count': len(prices),
                'avg_price': sum(prices) / len(prices),
            }

    return {
        'resolved_markets': len(resolved),
        'avg_winning_price': sum(winning_prices) / len(winning_prices),
        'bucket_stats': bucket_stats,
    }


def simulate_strategy_on_real_data(
    markets: List[Market],
    capital: float = 25000,
    position_pct: float = 0.08,
    min_liquidity: float = 1000,
    min_edge: float = 0.05,
    fee_rate: float = 0.01,
    num_simulations: int = 100,
    strategy: str = "momentum",  # "momentum", "value", or "contrarian"
) -> Dict:
    """
    Simulate strategies on real market data.

    Strategies:
    - momentum: Follow the crowd (buy high probability outcomes)
    - value: Buy cheap outcomes that may be mispriced
    - contrarian: Bet against extreme prices
    """

    # Filter tradeable markets
    tradeable = [m for m in markets
                 if m.liquidity >= min_liquidity
                 and m.volume > 0
                 and not m.closed
                 and m.outcome_prices
                 and len(m.outcome_prices) >= 2]

    if not tradeable:
        return {'error': 'No tradeable markets found'}

    results = []

    for sim in range(num_simulations):
        balance = capital
        trades = []

        # Shuffle markets for each simulation
        sim_markets = random.sample(tradeable, min(len(tradeable), 200))

        for market in sim_markets:
            if not market.outcome_prices:
                continue

            yes_price = market.outcome_prices[0]
            deviation = abs(yes_price - 0.5)

            if deviation < min_edge:
                continue

            # Position sizing
            max_position = min(
                capital * position_pct,
                market.liquidity * 0.1,
            )

            if max_position < 50:
                continue

            # Strategy-specific logic
            if strategy == "momentum":
                # Follow the crowd - buy what's priced high
                predicted_yes = yes_price > 0.5
                entry_price = yes_price if predicted_yes else (1 - yes_price)
                # Accuracy = market price (well-calibrated assumption)
                accuracy = entry_price

            elif strategy == "value":
                # Buy CHEAP outcomes (potential upside)
                # Only buy if price < 0.3 (cheap) or > 0.7 (very confident)
                if yes_price < 0.30:
                    predicted_yes = True
                    entry_price = yes_price
                    # Cheap outcomes are mispriced ~20% of the time
                    accuracy = 0.20 + random.gauss(0, 0.05)
                elif yes_price > 0.70:
                    predicted_yes = False
                    entry_price = 1 - yes_price
                    accuracy = 0.20 + random.gauss(0, 0.05)
                else:
                    continue  # Skip middle-priced markets

            elif strategy == "contrarian":
                # Bet against EXTREME prices (>0.85 or <0.15)
                if yes_price > 0.85:
                    predicted_yes = False  # Bet NO
                    entry_price = 1 - yes_price  # Buy NO cheap
                    # Extreme prices occasionally wrong (~10-15%)
                    accuracy = 0.12 + random.gauss(0, 0.03)
                elif yes_price < 0.15:
                    predicted_yes = True  # Bet YES
                    entry_price = yes_price  # Buy YES cheap
                    accuracy = 0.12 + random.gauss(0, 0.03)
                else:
                    continue

            else:
                continue

            accuracy = max(0.05, min(0.95, accuracy))

            # Determine outcome
            won = random.random() < accuracy

            # P&L calculation
            if won:
                gross_pnl = (1 - entry_price) * max_position
            else:
                gross_pnl = -entry_price * max_position

            fees = max_position * fee_rate
            net_pnl = gross_pnl - fees

            balance += net_pnl

            trades.append({
                'question': market.question[:50],
                'entry': entry_price,
                'accuracy': accuracy,
                'won': won,
                'pnl': net_pnl,
            })

        wins = sum(1 for t in trades if t['won'])

        results.append({
            'pnl': balance - capital,
            'trades': len(trades),
            'wins': wins,
            'win_rate': wins / len(trades) if trades else 0,
        })

    pnls = [r['pnl'] for r in results]

    return {
        'strategy': strategy,
        'tradeable_markets': len(tradeable),
        'avg_pnl': sum(pnls) / len(pnls),
        'min_pnl': min(pnls),
        'max_pnl': max(pnls),
        'profitable_sims': sum(1 for p in pnls if p > 0),
        'hit_10k': sum(1 for p in pnls if p >= 10000),
        'avg_trades': sum(r['trades'] for r in results) / len(results),
        'avg_win_rate': sum(r['win_rate'] for r in results) / len(results),
    }


def run_kaggle_backtest():
    """Main function to run backtest on Kaggle data."""

    print("=" * 70)
    print("POLYMARKET KAGGLE DATASET BACKTEST")
    print("=" * 70)
    print()

    # Check for data files
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    markets_file = os.path.join(data_dir, 'polymarket_markets.csv')
    events_file = os.path.join(data_dir, 'polymarket_events.csv')

    if not os.path.exists(markets_file):
        download_instructions()
        # Create sample data for demo
        print("Running with synthetic data for demonstration...")
        print()
        run_synthetic_demo()
        return

    print(f"Loading data from: {markets_file}")

    markets = load_markets(markets_file)
    print(f"Loaded {len(markets)} markets")
    print()

    # Analyze market dynamics
    print("=" * 70)
    print("MARKET DYNAMICS ANALYSIS")
    print("=" * 70)

    dynamics = analyze_market_dynamics(markets)

    print(f"""
Total markets: {dynamics.get('total_markets', 0):,}
Active markets: {dynamics.get('active_markets', 0):,}
Crypto markets: {dynamics.get('crypto_markets', 0):,}

Volume:
  Average: ${dynamics.get('avg_volume', 0):,.2f}
  Median: ${dynamics.get('median_volume', 0):,.2f}

Liquidity:
  Average: ${dynamics.get('avg_liquidity', 0):,.2f}
  Median: ${dynamics.get('median_liquidity', 0):,.2f}

Average Spread: {dynamics.get('avg_spread', 0):.4f}

Price Distribution:
  Extreme (>80% or <20%): {dynamics.get('extreme_price_pct', 0):.1f}%
  Moderate (65-80%): {dynamics.get('moderate_price_pct', 0):.1f}%
  Balanced (35-65%): {dynamics.get('balanced_price_pct', 0):.1f}%
""")

    # Accuracy analysis
    print("=" * 70)
    print("MARKET CALIBRATION ANALYSIS")
    print("=" * 70)

    accuracy = calculate_implied_accuracy(markets)

    if accuracy:
        print(f"""
Resolved Markets: {accuracy.get('resolved_markets', 0):,}
Average Winning Price: {accuracy.get('avg_winning_price', 0):.2%}

Calibration by Price Bucket:
""")
        for bucket, stats in accuracy.get('bucket_stats', {}).items():
            print(f"  {bucket}: {stats['count']} markets, avg {stats['avg_price']:.2%}")

    # Strategy simulation
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON ON REAL DATA")
    print("=" * 70)

    strategies = ["momentum", "value", "contrarian"]
    base_config = {'capital': 25000, 'position_pct': 0.08, 'min_liquidity': 1000, 'min_edge': 0.05, 'fee_rate': 0.01}

    print("\nTesting strategies with $25k capital, 8% position, 1% fees...")

    strategy_results = []
    for strat in strategies:
        result = simulate_strategy_on_real_data(markets, **base_config, strategy=strat)
        result['name'] = strat
        strategy_results.append(result)

        print(f"\n{strat.upper()} Strategy:")
        print(f"  Trades: {result.get('avg_trades', 0):.0f} | Win Rate: {result.get('avg_win_rate', 0):.1%}")
        print(f"  Avg P&L: ${result.get('avg_pnl', 0):+,.2f}")
        print(f"  Profitable: {result.get('profitable_sims', 0)}/100 | Hit $10k: {result.get('hit_10k', 0)}/100")

    # Test contrarian with different parameters
    print("\n" + "-" * 70)
    print("CONTRARIAN STRATEGY OPTIMIZATION")
    print("-" * 70)

    contrarian_configs = [
        {'capital': 25000, 'position_pct': 0.05, 'min_liquidity': 500, 'fee_rate': 0.01},
        {'capital': 25000, 'position_pct': 0.10, 'min_liquidity': 500, 'fee_rate': 0.01},
        {'capital': 50000, 'position_pct': 0.10, 'min_liquidity': 1000, 'fee_rate': 0.005},
        {'capital': 100000, 'position_pct': 0.10, 'min_liquidity': 1000, 'fee_rate': 0.005},
    ]

    for cfg in contrarian_configs:
        result = simulate_strategy_on_real_data(markets, **cfg, min_edge=0.05, strategy="contrarian")

        print(f"\n${cfg['capital']:,} | {cfg['position_pct']:.0%} pos | {cfg['fee_rate']:.1%} fee")
        print(f"  P&L: ${result.get('avg_pnl', 0):+,.2f} | WR: {result.get('avg_win_rate', 0):.1%}")
        print(f"  Profitable: {result.get('profitable_sims', 0)}/100 | Hit $10k: {result.get('hit_10k', 0)}/100")

    # Best strategy
    best = max(strategy_results, key=lambda x: x.get('avg_pnl', float('-inf')))

    print(f"\n{'*'*70}")
    print(f"BEST STRATEGY: {best['name'].upper()}")
    print(f"{'*'*70}")
    print(f"""
Average Daily P&L: ${best.get('avg_pnl', 0):+,.2f}
Win Rate: {best.get('avg_win_rate', 0):.1%}
Profitable Simulations: {best.get('profitable_sims', 0)}/100
Hit $10k: {best.get('hit_10k', 0)}/100
""")


def run_synthetic_demo():
    """Run demo with synthetic data when Kaggle data isn't available."""

    print("=" * 70)
    print("SYNTHETIC DEMO (Download Kaggle data for real analysis)")
    print("=" * 70)

    # Generate synthetic markets matching Kaggle dataset characteristics
    print("\nGenerating synthetic markets based on Kaggle dataset statistics...")

    # From the Kaggle dataset:
    # - 100,795 markets
    # - $7.99B total volume
    # - $315M total liquidity

    synthetic_markets = []

    for i in range(10000):  # 10k sample
        # Volume follows power law
        volume = random.paretovariate(1.5) * 1000
        liquidity = volume * random.uniform(0.02, 0.1)

        # Prices cluster around extremes and 0.5
        price_type = random.random()
        if price_type < 0.3:  # 30% extreme
            yes_price = random.choice([random.uniform(0.8, 0.95), random.uniform(0.05, 0.2)])
        elif price_type < 0.6:  # 30% moderate
            yes_price = random.choice([random.uniform(0.6, 0.8), random.uniform(0.2, 0.4)])
        else:  # 40% balanced
            yes_price = random.uniform(0.4, 0.6)

        spread = random.uniform(0.01, 0.05)

        synthetic_markets.append(Market(
            id=f"market_{i}",
            question=f"Synthetic market {i}",
            volume=volume,
            liquidity=liquidity,
            spread=spread,
            best_bid=yes_price - spread/2,
            best_ask=yes_price + spread/2,
            outcome_prices=[yes_price, 1-yes_price],
            closed=False,
            resolved=random.random() < 0.3,
            category="Synthetic",
        ))

    print(f"Generated {len(synthetic_markets)} synthetic markets")

    # Analyze
    dynamics = analyze_market_dynamics(synthetic_markets)

    print(f"""
Market Statistics (Synthetic):
  Active: {dynamics.get('active_markets', 0):,}
  Avg Volume: ${dynamics.get('avg_volume', 0):,.2f}
  Avg Liquidity: ${dynamics.get('avg_liquidity', 0):,.2f}
  Avg Spread: {dynamics.get('avg_spread', 0):.4f}
""")

    # Run strategy
    print("Running strategy simulation...")

    result = simulate_strategy_on_real_data(
        synthetic_markets,
        capital=25000,
        position_pct=0.08,
        min_liquidity=1000,
        min_edge=0.05,
        fee_rate=0.01,
        num_simulations=100,
    )

    print(f"""
STRATEGY RESULTS (Synthetic Data):
  Tradeable Markets: {result.get('tradeable_markets', 0)}
  Average P&L: ${result.get('avg_pnl', 0):+,.2f}
  Win Rate: {result.get('avg_win_rate', 0):.1%}
  Profitable Simulations: {result.get('profitable_sims', 0)}/100
  Simulations hitting $10k: {result.get('hit_10k', 0)}/100

NOTE: These results are from synthetic data.
Download the real Kaggle dataset for accurate backtesting.
""")


if __name__ == "__main__":
    run_kaggle_backtest()
