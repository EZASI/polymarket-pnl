"""
Realistic $10K Strategy
Accounts for actual Polymarket constraints:
- 15-minute crypto markets (BTC, ETH, SOL, DOGE)
- 4 markets per hour per coin = 16 total opportunities/hour
- Typical liquidity limits (~$5-50k per market)
- Not every market has strong signals
"""

import random
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MarketOpportunity:
    coin: str
    obi: float
    accuracy: float
    liquidity: float  # Max position possible


def generate_market_opportunities(num_markets: int = 16) -> List[MarketOpportunity]:
    """Generate realistic market opportunities for 1 hour."""
    coins = ["BTC", "ETH", "SOL", "DOGE"]
    opportunities = []

    for i in range(num_markets):
        coin = coins[i % 4]

        # OBI follows realistic distribution (most are weak)
        obi_roll = random.random()
        if obi_roll < 0.50:
            obi = random.uniform(0, 0.2)  # Weak signal (50%)
            accuracy = 0.52
        elif obi_roll < 0.75:
            obi = random.uniform(0.2, 0.4)  # Moderate (25%)
            accuracy = 0.58
        elif obi_roll < 0.90:
            obi = random.uniform(0.4, 0.6)  # Strong (15%)
            accuracy = 0.65
        else:
            obi = random.uniform(0.6, 0.8)  # Extreme (10%)
            accuracy = 0.72

        # Random direction
        if random.random() < 0.5:
            obi = -obi

        # Liquidity varies by coin and time
        base_liquidity = {"BTC": 50000, "ETH": 30000, "SOL": 15000, "DOGE": 10000}
        liquidity = base_liquidity[coin] * random.uniform(0.5, 1.5)

        opportunities.append(MarketOpportunity(
            coin=coin,
            obi=obi,
            accuracy=accuracy,
            liquidity=liquidity,
        ))

    return opportunities


def simulate_realistic_day(
    capital: float,
    position_pct: float,
    obi_threshold: float,
    fee_rate: float,
    hours: int = 24,
) -> Dict:
    """Simulate a realistic trading day."""
    balance = capital
    trades = []

    for hour in range(hours):
        markets = generate_market_opportunities(16)

        for market in markets:
            # Only trade if signal is strong enough
            if abs(market.obi) < obi_threshold:
                continue

            # Position size limited by capital % AND liquidity
            max_position = capital * position_pct
            position_size = min(max_position, market.liquidity * 0.1)  # Max 10% of liquidity

            if position_size < 100:  # Minimum viable trade
                continue

            # Determine outcome
            won = random.random() < market.accuracy

            # P&L calculation
            entry_price = 0.5  # Simplified

            if won:
                gross_pnl = 0.5 * position_size
            else:
                gross_pnl = -0.5 * position_size

            fees = position_size * fee_rate
            net_pnl = gross_pnl - fees

            balance += net_pnl

            trades.append({
                'coin': market.coin,
                'obi': market.obi,
                'accuracy': market.accuracy,
                'position': position_size,
                'won': won,
                'net_pnl': net_pnl,
            })

    wins = sum(1 for t in trades if t['won'])

    return {
        'final_balance': balance,
        'daily_pnl': balance - capital,
        'trades': len(trades),
        'wins': wins,
        'win_rate': wins / len(trades) if trades else 0,
        'trade_details': trades,
    }


def run_simulation(
    capital: float,
    position_pct: float,
    obi_threshold: float,
    fee_rate: float,
    num_days: int = 100,
) -> Dict:
    """Run multiple day simulation."""
    results = []

    for _ in range(num_days):
        day = simulate_realistic_day(capital, position_pct, obi_threshold, fee_rate)
        results.append(day)

    daily_pnls = [r['daily_pnl'] for r in results]

    return {
        'avg_daily_pnl': sum(daily_pnls) / len(daily_pnls),
        'min_pnl': min(daily_pnls),
        'max_pnl': max(daily_pnls),
        'profitable_days': sum(1 for p in daily_pnls if p > 0),
        'hit_10k': sum(1 for p in daily_pnls if p >= 10000),
        'avg_trades': sum(r['trades'] for r in results) / len(results),
        'avg_win_rate': sum(r['win_rate'] for r in results) / len(results),
    }


def find_10k_config():
    """Find configuration that can make $10k/day."""
    print("=" * 70)
    print("REALISTIC $10K/DAY STRATEGY FINDER")
    print("=" * 70)
    print()
    print("Constraints:")
    print("- 16 market opportunities per hour (4 coins x 4 periods)")
    print("- Position limited by liquidity (max 10% of market liquidity)")
    print("- Signal distribution: 50% weak, 25% moderate, 15% strong, 10% extreme")
    print()

    # Test configurations
    configs = []

    for capital in [10000, 25000, 50000, 100000, 150000, 200000]:
        for pos_pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for obi_thresh in [0.2, 0.3, 0.4, 0.5]:
                for fee in [0.005, 0.01, 0.015]:
                    configs.append({
                        'capital': capital,
                        'position_pct': pos_pct,
                        'obi_threshold': obi_thresh,
                        'fee_rate': fee,
                    })

    print(f"Testing {len(configs)} configurations...")

    results = []
    for i, cfg in enumerate(configs):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(configs)}")

        result = run_simulation(
            capital=cfg['capital'],
            position_pct=cfg['position_pct'],
            obi_threshold=cfg['obi_threshold'],
            fee_rate=cfg['fee_rate'],
            num_days=50,
        )
        result['config'] = cfg
        results.append(result)

    # Sort by daily P&L
    results.sort(key=lambda x: x['avg_daily_pnl'], reverse=True)

    # Find those that hit $10k
    hit_10k = [r for r in results if r['hit_10k'] > 0]
    consistent_10k = [r for r in results if r['avg_daily_pnl'] >= 10000]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nConfigurations tested: {len(results)}")
    print(f"Can hit $10k (at least 1 day): {len(hit_10k)}")
    print(f"Average $10k+/day: {len(consistent_10k)}")

    print(f"\n{'='*70}")
    print("TOP 10 STRATEGIES")
    print(f"{'='*70}")

    for i, r in enumerate(results[:10], 1):
        c = r['config']
        print(f"\n{i}. Capital: ${c['capital']:,} | Pos: {c['position_pct']:.0%} | "
              f"OBI: {c['obi_threshold']} | Fee: {c['fee_rate']:.1%}")
        print(f"   Avg P&L: ${r['avg_daily_pnl']:+,.2f} | Trades: {r['avg_trades']:.1f}/day")
        print(f"   Win Rate: {r['avg_win_rate']:.1%} | Profitable: {r['profitable_days']}/50")
        print(f"   Hit $10k: {r['hit_10k']}/50 days")

    # Best strategy
    best = results[0]
    c = best['config']

    print(f"\n{'*'*70}")
    print("BEST STRATEGY FOR $10K TARGET")
    print(f"{'*'*70}")

    print(f"""
CONFIGURATION:
- Capital: ${c['capital']:,}
- Position Size: {c['position_pct']:.0%} per trade
- OBI Threshold: {c['obi_threshold']} (trade only strong signals)
- Fee Rate: {c['fee_rate']:.1%}

EXPECTED RESULTS:
- Daily P&L: ${best['avg_daily_pnl']:+,.2f}
- Trades/Day: {best['avg_trades']:.1f}
- Win Rate: {best['avg_win_rate']:.1%}
- Profitable Days: {best['profitable_days']}/50 ({best['profitable_days']*2}%)
- Days hitting $10k+: {best['hit_10k']}/50

PROJECTIONS:
- Weekly: ${best['avg_daily_pnl'] * 7:+,.2f}
- Monthly: ${best['avg_daily_pnl'] * 30:+,.2f}
""")

    # What's needed for $10k if not achieved
    if best['avg_daily_pnl'] < 10000:
        gap = 10000 - best['avg_daily_pnl']
        multiplier = 10000 / best['avg_daily_pnl'] if best['avg_daily_pnl'] > 0 else float('inf')

        print(f"\nGAP TO $10K: ${gap:,.2f}/day")
        print(f"\nTO REACH $10K, YOU NEED:")

        if multiplier != float('inf') and multiplier < 10:
            needed_capital = c['capital'] * multiplier
            print(f"- Capital: ${needed_capital:,.0f} (current ${c['capital']:,})")
        else:
            print("- Significantly more capital OR")
            print("- Better signal accuracy OR")
            print("- Lower fees")

    # Show capital scaling
    print(f"\n{'='*70}")
    print("CAPITAL VS DAILY P&L (Best Config)")
    print(f"{'='*70}")

    best_cfg = results[0]['config']

    print("\nCapital     | Position  | Est. Daily P&L | Days to $10k")
    print("-" * 60)

    for cap in [10000, 25000, 50000, 100000, 150000, 200000, 500000]:
        # Scale P&L roughly with capital
        scale = cap / best_cfg['capital']
        est_pnl = best['avg_daily_pnl'] * scale * 0.8  # 0.8 for liquidity constraints

        if est_pnl > 0:
            days_to_10k = 10000 / est_pnl
            days_str = f"{days_to_10k:.1f}" if days_to_10k < 100 else ">100"
        else:
            days_str = "N/A"

        pos_size = cap * best_cfg['position_pct']
        print(f"${cap:>9,} | ${pos_size:>8,.0f} | ${est_pnl:>13,.2f} | {days_str}")

    # Minimum for $10k
    print(f"\n{'='*70}")
    print("MINIMUM REQUIREMENTS FOR $10K/DAY")
    print(f"{'='*70}")

    # Calculate based on best strategy's efficiency
    if best['avg_daily_pnl'] > 0:
        efficiency = best['avg_daily_pnl'] / best['config']['capital']
        min_capital = 10000 / efficiency / 0.8  # Account for liquidity limits

        print(f"""
Based on best strategy efficiency ({efficiency:.2%}/day):

Minimum Capital Needed: ~${min_capital:,.0f}

With this capital and:
- Position size: {best_cfg['position_pct']:.0%}
- OBI threshold: {best_cfg['obi_threshold']}
- Fee rate: {best_cfg['fee_rate']:.1%}

You can expect ~$10,000/day average
""")


if __name__ == "__main__":
    find_10k_config()
