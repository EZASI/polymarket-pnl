#!/usr/bin/env python3
"""
Analyze profitability of 15-minute crypto prediction trades by asset (BTC, SOL, ETH).
"""

import json
from collections import defaultdict

def analyze_15min_profitability():
    print("Loading trade data...")
    with open("mty44_trades.json", 'r') as f:
        data = json.load(f)

    trades = data.get('trades', [])

    # Filter for 15-minute crypto trades only
    crypto_trades = {
        'Bitcoin': [],
        'Solana': [],
        'Ethereum': []
    }

    for t in trades:
        title = t.get('title', '').lower()
        if '15' not in title and 'up or down' not in title.lower():
            continue

        if 'bitcoin' in title:
            crypto_trades['Bitcoin'].append(t)
        elif 'solana' in title:
            crypto_trades['Solana'].append(t)
        elif 'ethereum' in title:
            crypto_trades['Ethereum'].append(t)

    print(f"\n{'='*80}")
    print("15-MINUTE CRYPTO PREDICTION PROFITABILITY ANALYSIS")
    print(f"{'='*80}")

    # Analyze each asset
    results = {}

    for asset, asset_trades in crypto_trades.items():
        if not asset_trades:
            continue

        buy_trades = [t for t in asset_trades if t.get('side') == 'BUY']
        sell_trades = [t for t in asset_trades if t.get('side') == 'SELL']

        # Calculate costs and revenue
        total_buy_cost = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in buy_trades)
        total_buy_shares = sum(float(t.get('size', 0)) for t in buy_trades)

        total_sell_revenue = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in sell_trades)
        total_sell_shares = sum(float(t.get('size', 0)) for t in sell_trades)

        # Outcome analysis - for binary prediction markets:
        # If you BUY at price P and the outcome is YES (wins), you get $1 per share
        # If you BUY at price P and the outcome is NO (loses), you get $0
        # Profit on winning trade = (1 - P) * shares
        # Loss on losing trade = P * shares

        # Group trades by market/event to track outcomes
        markets = defaultdict(lambda: {'buys': [], 'sells': []})

        for t in asset_trades:
            condition_id = t.get('conditionId', t.get('title', 'unknown'))
            if t.get('side') == 'BUY':
                markets[condition_id]['buys'].append(t)
            else:
                markets[condition_id]['sells'].append(t)

        # Analyze by outcome preference (Up vs Down bets)
        up_trades = [t for t in buy_trades if t.get('outcome', '').lower() == 'up']
        down_trades = [t for t in buy_trades if t.get('outcome', '').lower() == 'down']

        up_cost = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in up_trades)
        up_shares = sum(float(t.get('size', 0)) for t in up_trades)

        down_cost = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in down_trades)
        down_shares = sum(float(t.get('size', 0)) for t in down_trades)

        # Average prices
        avg_up_price = up_cost / up_shares if up_shares > 0 else 0
        avg_down_price = down_cost / down_shares if down_shares > 0 else 0

        # For estimation: assuming 50% win rate on binary predictions
        # Expected value = (win_rate * payout) - (1 - win_rate) * cost
        # At fair odds (50%), price should be 0.50
        # If buying at higher prices (>0.50), expected loss
        # If buying at lower prices (<0.50), expected gain

        # Calculate implied win rate needed to break even
        # breakeven_win_rate = avg_price (since payout is $1)

        results[asset] = {
            'total_trades': len(asset_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_buy_cost': total_buy_cost,
            'total_buy_shares': total_buy_shares,
            'total_sell_revenue': total_sell_revenue,
            'total_sell_shares': total_sell_shares,
            'up_trades': len(up_trades),
            'down_trades': len(down_trades),
            'up_cost': up_cost,
            'up_shares': up_shares,
            'down_cost': down_cost,
            'down_shares': down_shares,
            'avg_up_price': avg_up_price,
            'avg_down_price': avg_down_price,
            'unique_markets': len(markets)
        }

    # Print results
    for asset in ['Bitcoin', 'Solana', 'Ethereum']:
        if asset not in results:
            print(f"\n{asset}: No 15-minute trades found")
            continue

        r = results[asset]
        print(f"\n{'='*80}")
        print(f"{asset.upper()} 15-MINUTE PREDICTIONS")
        print(f"{'='*80}")

        print(f"\n--- TRADE VOLUME ---")
        print(f"Total trades: {r['total_trades']:,}")
        print(f"Buy trades: {r['buy_trades']:,}")
        print(f"Sell trades: {r['sell_trades']:,}")
        print(f"Unique market events: {r['unique_markets']:,}")

        print(f"\n--- CAPITAL DEPLOYED ---")
        print(f"Total buy cost: ${r['total_buy_cost']:,.2f}")
        print(f"Total shares bought: {r['total_buy_shares']:,.2f}")
        print(f"Average buy price: ${r['total_buy_cost']/r['total_buy_shares']:.4f}" if r['total_buy_shares'] > 0 else "N/A")

        print(f"\n--- UP vs DOWN BETS ---")
        print(f"UP predictions: {r['up_trades']:,} trades, {r['up_shares']:,.2f} shares, ${r['up_cost']:,.2f} cost")
        print(f"DOWN predictions: {r['down_trades']:,} trades, {r['down_shares']:,.2f} shares, ${r['down_cost']:,.2f} cost")

        if r['up_shares'] > 0:
            print(f"Average UP entry price: ${r['avg_up_price']:.4f} (need {r['avg_up_price']*100:.1f}% win rate to break even)")
        if r['down_shares'] > 0:
            print(f"Average DOWN entry price: ${r['avg_down_price']:.4f} (need {r['avg_down_price']*100:.1f}% win rate to break even)")

        print(f"\n--- SELLS (exits/hedges) ---")
        print(f"Sell revenue: ${r['total_sell_revenue']:,.2f}")
        print(f"Shares sold: {r['total_sell_shares']:,.2f}")

        # Estimate P&L
        # Net exposure = shares bought - shares sold
        net_shares = r['total_buy_shares'] - r['total_sell_shares']
        net_cost = r['total_buy_cost'] - r['total_sell_revenue']

        print(f"\n--- NET EXPOSURE ---")
        print(f"Net shares held: {net_shares:,.2f}")
        print(f"Net capital at risk: ${net_cost:,.2f}")

        if net_shares > 0:
            avg_net_price = net_cost / net_shares
            print(f"Average cost basis: ${avg_net_price:.4f}")

            # Estimated P&L scenarios
            print(f"\n--- ESTIMATED P&L SCENARIOS ---")
            for win_rate in [0.40, 0.45, 0.50, 0.55, 0.60]:
                # Expected value per share = win_rate * $1 - cost_per_share
                expected_pnl = (win_rate * net_shares) - net_cost
                print(f"  At {win_rate*100:.0f}% win rate: ${expected_pnl:,.2f} ({'profit' if expected_pnl > 0 else 'loss'})")

    # Summary comparison
    print(f"\n{'='*80}")
    print("PROFITABILITY COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Asset':<12} {'Trades':>10} {'Buy Cost':>15} {'Avg Price':>12} {'Down%':>10} {'Risk Score':>12}")
    print("-" * 80)

    for asset in ['Bitcoin', 'Solana', 'Ethereum']:
        if asset not in results:
            continue
        r = results[asset]
        avg_price = r['total_buy_cost']/r['total_buy_shares'] if r['total_buy_shares'] > 0 else 0
        down_pct = r['down_trades'] / r['buy_trades'] * 100 if r['buy_trades'] > 0 else 0

        # Risk score: lower avg price = better odds, higher is worse
        # Score from 1-10 where lower is better
        risk_score = avg_price * 10

        print(f"{asset:<12} {r['total_trades']:>10,} ${r['total_buy_cost']:>13,.2f} ${avg_price:>10.4f} {down_pct:>9.1f}% {risk_score:>11.1f}")

    print(f"\n--- KEY INSIGHTS ---")

    # Find best asset
    best_asset = None
    best_avg_price = 1.0

    for asset, r in results.items():
        if r['total_buy_shares'] > 0:
            avg = r['total_buy_cost'] / r['total_buy_shares']
            if avg < best_avg_price:
                best_avg_price = avg
                best_asset = asset

    if best_asset:
        print(f"1. LOWEST AVG ENTRY PRICE: {best_asset} (${best_avg_price:.4f})")
        print(f"   - Lower entry prices mean better odds and lower breakeven win rate required")

    # Most traded
    most_traded = max(results.items(), key=lambda x: x[1]['total_trades'])
    print(f"\n2. MOST ACTIVELY TRADED: {most_traded[0]} ({most_traded[1]['total_trades']:,} trades)")

    # Highest capital deployed
    highest_capital = max(results.items(), key=lambda x: x[1]['total_buy_cost'])
    print(f"\n3. HIGHEST CAPITAL DEPLOYED: {highest_capital[0]} (${highest_capital[1]['total_buy_cost']:,.2f})")

    # Direction preference
    print(f"\n4. TRADING DIRECTION PREFERENCES:")
    for asset, r in results.items():
        if r['buy_trades'] > 0:
            down_pct = r['down_trades'] / r['buy_trades'] * 100
            up_pct = r['up_trades'] / r['buy_trades'] * 100
            bias = "DOWN" if down_pct > up_pct else "UP"
            print(f"   {asset}: {bias} bias ({down_pct:.1f}% down, {up_pct:.1f}% up)")

if __name__ == "__main__":
    analyze_15min_profitability()
