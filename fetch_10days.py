#!/usr/bin/env python3
"""
Fetch trades for the past 10 days and analyze patterns.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

WALLET_ADDRESS = "0x1831d1abf62dfadabcabb3d1c2bf4476146a6c99"
BASE_URL = "https://data-api.polymarket.com"

def fetch_trades_before(wallet_address, before_timestamp, limit=500):
    """Fetch trades before a certain timestamp."""
    url = f"{BASE_URL}/trades?user={wallet_address}&limit={limit}&before={before_timestamp}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return []

def fetch_all_trades_for_period(wallet_address, days=10):
    """Fetch all trades for the specified number of days."""
    all_trades = []

    # Start from now and work backwards
    current_ts = int(time.time() * 1000)  # milliseconds
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_ts = int(cutoff_date.timestamp() * 1000)

    print(f"Fetching trades from {cutoff_date.strftime('%Y-%m-%d')} to now...")
    print(f"Cutoff timestamp: {cutoff_ts}")

    batch = 0
    while True:
        batch += 1
        print(f"Batch {batch}: fetching before {current_ts}...")

        trades = fetch_trades_before(wallet_address, current_ts)

        if not trades:
            print("No more trades returned")
            break

        # Filter trades within our date range
        valid_trades = []
        oldest_ts = current_ts

        for t in trades:
            ts = int(t.get('timestamp', 0))
            if ts >= cutoff_ts:
                valid_trades.append(t)
            if ts < oldest_ts:
                oldest_ts = ts

        all_trades.extend(valid_trades)
        print(f"  Got {len(trades)} trades, {len(valid_trades)} within range (total: {len(all_trades)})")

        # Check if we've gone past our cutoff
        if oldest_ts < cutoff_ts:
            print(f"Reached cutoff date")
            break

        # Move cursor back
        current_ts = oldest_ts - 1
        time.sleep(0.3)

        # Safety limit
        if batch > 500:
            print("Safety limit reached")
            break

    return all_trades

def analyze_10day_patterns(trades):
    """Analyze patterns for the 10-day period."""

    if not trades:
        print("No trades to analyze")
        return

    # Filter for 15-min crypto trades
    crypto_trades = []
    for t in trades:
        title = t.get('title', '').lower()
        if 'up or down' in title and any(x in title for x in ['bitcoin', 'solana', 'ethereum']):
            crypto_trades.append(t)

    print(f"\n{'='*80}")
    print("10-DAY TRADING ANALYSIS")
    print(f"{'='*80}")

    # Date range
    timestamps = [int(t.get('timestamp', 0)) for t in trades]
    if timestamps:
        min_date = datetime.fromtimestamp(min(timestamps))
        max_date = datetime.fromtimestamp(max(timestamps))
        print(f"Period: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Days: {(max_date - min_date).days + 1}")

    print(f"Total trades: {len(trades):,}")
    print(f"15-min crypto trades: {len(crypto_trades):,}")

    # Daily breakdown
    print(f"\n{'='*80}")
    print("DAILY BREAKDOWN")
    print(f"{'='*80}")

    daily_stats = defaultdict(lambda: {
        'trades': 0, 'btc': 0, 'sol': 0, 'eth': 0,
        'up': 0, 'down': 0, 'cost': 0.0, 'shares': 0.0
    })

    for t in crypto_trades:
        ts = int(t.get('timestamp', 0))
        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        title = t.get('title', '').lower()
        outcome = t.get('outcome', '').lower()

        daily_stats[date]['trades'] += 1
        daily_stats[date]['cost'] += float(t.get('size', 0)) * float(t.get('price', 0))
        daily_stats[date]['shares'] += float(t.get('size', 0))

        if 'bitcoin' in title:
            daily_stats[date]['btc'] += 1
        elif 'solana' in title:
            daily_stats[date]['sol'] += 1
        elif 'ethereum' in title:
            daily_stats[date]['eth'] += 1

        if outcome == 'up':
            daily_stats[date]['up'] += 1
        elif outcome == 'down':
            daily_stats[date]['down'] += 1

    print(f"\n{'Date':<12} {'Trades':>8} {'BTC':>7} {'SOL':>7} {'ETH':>6} {'UP%':>7} {'DOWN%':>7} {'Cost':>14}")
    print("-" * 80)

    total_cost = 0
    for date in sorted(daily_stats.keys()):
        d = daily_stats[date]
        total = d['up'] + d['down']
        up_pct = d['up'] / total * 100 if total > 0 else 0
        down_pct = d['down'] / total * 100 if total > 0 else 0
        total_cost += d['cost']
        print(f"{date:<12} {d['trades']:>8,} {d['btc']:>7,} {d['sol']:>7,} {d['eth']:>6,} {up_pct:>6.1f}% {down_pct:>6.1f}% ${d['cost']:>12,.2f}")

    print("-" * 80)
    print(f"{'TOTAL':<12} {len(crypto_trades):>8,} {'':<7} {'':<7} {'':<6} {'':<7} {'':<7} ${total_cost:>12,.2f}")

    # Asset comparison
    print(f"\n{'='*80}")
    print("ASSET PROFITABILITY COMPARISON (10 DAYS)")
    print(f"{'='*80}")

    asset_stats = defaultdict(lambda: {'trades': 0, 'cost': 0.0, 'shares': 0.0, 'avg_price': 0.0})

    for t in crypto_trades:
        title = t.get('title', '').lower()
        cost = float(t.get('size', 0)) * float(t.get('price', 0))
        shares = float(t.get('size', 0))

        if 'bitcoin' in title:
            asset = 'Bitcoin'
        elif 'solana' in title:
            asset = 'Solana'
        elif 'ethereum' in title:
            asset = 'Ethereum'
        else:
            continue

        asset_stats[asset]['trades'] += 1
        asset_stats[asset]['cost'] += cost
        asset_stats[asset]['shares'] += shares

    print(f"\n{'Asset':<12} {'Trades':>10} {'Total Cost':>15} {'Avg Price':>12} {'Breakeven WR':>14}")
    print("-" * 70)

    for asset in ['Bitcoin', 'Solana', 'Ethereum']:
        s = asset_stats[asset]
        if s['shares'] > 0:
            avg_price = s['cost'] / s['shares']
            print(f"{asset:<12} {s['trades']:>10,} ${s['cost']:>13,.2f} ${avg_price:>10.4f} {avg_price*100:>13.1f}%")

    # Best/Worst days
    print(f"\n{'='*80}")
    print("BEST & WORST TRADING DAYS")
    print(f"{'='*80}")

    # Sort by cost (capital deployed)
    sorted_days = sorted(daily_stats.items(), key=lambda x: x[1]['cost'], reverse=True)

    print("\nHighest Capital Deployed:")
    for date, d in sorted_days[:3]:
        print(f"  {date}: ${d['cost']:,.2f} ({d['trades']:,} trades)")

    print("\nLowest Capital Deployed:")
    for date, d in sorted_days[-3:]:
        print(f"  {date}: ${d['cost']:,.2f} ({d['trades']:,} trades)")

    # Direction changes
    print(f"\n{'='*80}")
    print("DIRECTION BIAS BY DAY")
    print(f"{'='*80}")

    for date in sorted(daily_stats.keys()):
        d = daily_stats[date]
        total = d['up'] + d['down']
        if total > 0:
            down_pct = d['down'] / total * 100
            bias = "BEARISH" if down_pct > 55 else "BULLISH" if down_pct < 45 else "NEUTRAL"
            bar_down = "▼" * int(down_pct / 5)
            bar_up = "▲" * int((100 - down_pct) / 5)
            print(f"{date}: {bar_up}{bar_down} {bias} ({down_pct:.0f}% DOWN)")

def main():
    # Try to load existing data first
    try:
        with open('mty44_trades.json', 'r') as f:
            data = json.load(f)
        trades = data.get('trades', [])
        print(f"Loaded {len(trades):,} existing trades")

        # Check date range
        timestamps = [int(t.get('timestamp', 0)) for t in trades]
        if timestamps:
            min_date = datetime.fromtimestamp(min(timestamps))
            days_covered = (datetime.now() - min_date).days
            print(f"Existing data covers {days_covered} days")

            if days_covered >= 10:
                print("Already have 10+ days of data, analyzing...")
                analyze_10day_patterns(trades)
                return
    except:
        trades = []

    # Need to fetch more data
    print("\nFetching additional historical data...")
    new_trades = fetch_all_trades_for_period(WALLET_ADDRESS, days=10)

    if new_trades:
        # Merge with existing, removing duplicates
        existing_hashes = set(t.get('transactionHash', '') for t in trades)
        for t in new_trades:
            if t.get('transactionHash', '') not in existing_hashes:
                trades.append(t)

        print(f"\nTotal trades after merge: {len(trades):,}")

        # Save updated data
        data = {
            'wallet_address': WALLET_ADDRESS,
            'fetched_at': datetime.utcnow().isoformat(),
            'total_trades': len(trades),
            'trades': trades
        }

        with open('mty44_trades_10days.json', 'w') as f:
            json.dump(data, f)
        print("Saved to mty44_trades_10days.json")

    # Analyze
    analyze_10day_patterns(trades)

if __name__ == "__main__":
    main()
