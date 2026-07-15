#!/usr/bin/env python3
"""
Analyze trading data from the initial fetch and provide comprehensive statistics.
"""

import json
from collections import defaultdict
from datetime import datetime

def analyze_trades():
    print("Loading trade data...")
    with open("mty44_trades.json", 'r') as f:
        data = json.load(f)

    trades = data.get('trades', [])
    positions = data.get('positions', [])

    print(f"\n{'='*70}")
    print(f"POLYMARKET TRADING DATA ANALYSIS")
    print(f"Wallet: {data.get('wallet_address')}")
    print(f"Fetched at: {data.get('fetched_at')}")
    print(f"{'='*70}")

    print(f"\n--- OVERVIEW ---")
    print(f"Total trades in file: {len(trades):,}")
    print(f"Total positions: {len(positions)}")

    if not trades:
        print("No trades found.")
        return

    # Calculate statistics
    buy_trades = [t for t in trades if t.get('side') == 'BUY']
    sell_trades = [t for t in trades if t.get('side') == 'SELL']

    print(f"\n--- TRADE BREAKDOWN ---")
    print(f"Buy trades: {len(buy_trades):,}")
    print(f"Sell trades: {len(sell_trades):,}")

    # Volume calculation
    total_volume = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in trades)
    buy_volume = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in buy_trades)
    sell_volume = sum(float(t.get('size', 0)) * float(t.get('price', 0)) for t in sell_trades)

    print(f"\n--- VOLUME ---")
    print(f"Total Volume: ${total_volume:,.2f}")
    print(f"Buy Volume: ${buy_volume:,.2f}")
    print(f"Sell Volume: ${sell_volume:,.2f}")

    # Market analysis
    markets = defaultdict(int)
    market_volume = defaultdict(float)
    outcomes = defaultdict(int)

    for t in trades:
        title = t.get('title', 'Unknown')
        outcome = t.get('outcome', 'Unknown')
        vol = float(t.get('size', 0)) * float(t.get('price', 0))

        markets[title] += 1
        market_volume[title] += vol
        outcomes[outcome] += 1

    print(f"\n--- MARKET DIVERSITY ---")
    print(f"Unique markets traded: {len(markets):,}")

    # Top markets by trade count
    print(f"\n--- TOP 10 MARKETS BY TRADE COUNT ---")
    for market, count in sorted(markets.items(), key=lambda x: -x[1])[:10]:
        vol = market_volume.get(market, 0)
        print(f"  {count:>6} trades | ${vol:>12,.2f} | {market[:60]}")

    # Outcome distribution
    print(f"\n--- OUTCOME DISTRIBUTION ---")
    for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1])[:10]:
        pct = count / len(trades) * 100
        print(f"  {outcome}: {count:,} trades ({pct:.1f}%)")

    # Time analysis
    timestamps = []
    for t in trades:
        ts = t.get('timestamp')
        if ts:
            try:
                timestamps.append(int(ts))
            except:
                pass

    if timestamps:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        # Convert epoch seconds to datetime
        try:
            min_date = datetime.fromtimestamp(min_ts)
            max_date = datetime.fromtimestamp(max_ts)
            print(f"\n--- TIME RANGE ---")
            print(f"First trade: {min_date}")
            print(f"Last trade: {max_date}")
            duration = max_date - min_date
            print(f"Trading period: {duration.days} days")
            if duration.days > 0:
                trades_per_day = len(trades) / duration.days
                print(f"Average trades per day: {trades_per_day:,.1f}")
        except:
            pass

    # Asset analysis
    print(f"\n--- ASSET CATEGORIES ---")
    asset_types = defaultdict(int)
    for t in trades:
        title = t.get('title', '').lower()
        if 'bitcoin' in title or 'btc' in title:
            asset_types['Bitcoin'] += 1
        elif 'ethereum' in title or 'eth' in title:
            asset_types['Ethereum'] += 1
        elif 'solana' in title or 'sol' in title:
            asset_types['Solana'] += 1
        elif any(sport in title for sport in ['nhl', 'nba', 'nfl', 'premier', 'serie', 'champions', 'vs.']):
            asset_types['Sports'] += 1
        else:
            asset_types['Other'] += 1

    for asset, count in sorted(asset_types.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"  {asset}: {count:,} trades ({pct:.1f}%)")

    # Price distribution
    prices = [float(t.get('price', 0)) for t in trades if t.get('price')]
    if prices:
        avg_price = sum(prices) / len(prices)
        print(f"\n--- PRICE STATISTICS ---")
        print(f"Average entry price: ${avg_price:.4f}")
        print(f"Min price: ${min(prices):.4f}")
        print(f"Max price: ${max(prices):.4f}")

    # Position sizes
    sizes = [float(t.get('size', 0)) for t in trades if t.get('size')]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        print(f"\n--- POSITION SIZE STATISTICS ---")
        print(f"Average position size: {avg_size:,.2f} contracts")
        print(f"Min size: {min(sizes):,.2f}")
        print(f"Max size: {max(sizes):,.2f}")
        print(f"Total contracts traded: {sum(sizes):,.2f}")

    # Current positions
    if positions:
        print(f"\n--- CURRENT POSITIONS ---")
        for p in positions:
            title = p.get('title', 'Unknown')
            outcome = p.get('outcome', 'Unknown')
            size = float(p.get('size', 0))
            entry = float(p.get('avgSharePrice', 0))
            curr = float(p.get('curPrice', 0))
            value = float(p.get('value', 0))
            pnl = float(p.get('pnl', 0))
            print(f"  {title[:50]}...")
            print(f"    Outcome: {outcome} | Size: {size:,.2f} | Entry: ${entry:.4f} | Current: ${curr:.4f}")
            print(f"    Value: ${value:,.2f} | P&L: ${pnl:,.2f}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

    # Note about additional data
    print(f"\n*** NOTE ***")
    print(f"This analysis is based on {len(trades):,} trades from the initial fetch.")
    print(f"The full trading history contains approximately 2,200,000+ trades.")
    print(f"The API returned trades from multiple fetches before being stopped.")

if __name__ == "__main__":
    analyze_trades()
