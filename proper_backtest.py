#!/usr/bin/env python3
"""
PROPER Polymarket Backtester

Key insight: Each 15-minute window is a SEPARATE market.
- Progress goes 0→1 within each market
- At progress=1, market resolves (price → 0 or 1)
- We must trade WITHIN a single market, not across

This backtester:
1. Segments by progress resets (new markets)
2. Only trades when progress < 0.9 (before resolution)
3. Forces close at progress = 0.85
4. Properly handles each market independently
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Costs
TAKER_FEE = 0.005
SLIPPAGE = 0.003

# Close position before this progress
MAX_PROGRESS = 0.85

def load_and_segment():
    """Load data and segment by 15-minute markets."""
    import os
    
    path = 'polymarket_combined.parquet' if os.path.exists('polymarket_combined.parquet') else 'polymarket_chunk.parquet'
    df = pd.read_parquet(path)
    print(f'📊 Loaded {len(df):,} rows')
    
    # YES side only
    df = df[df['outcome_up'] == 1].copy()
    df = df.sort_values('ts')
    
    # Identify market boundaries (when progress resets)
    df['progress_diff'] = df['progress'].diff()
    df['new_market'] = (df['progress_diff'] < -0.3).astype(int)
    df['market_id'] = df['new_market'].cumsum()
    
    num_markets = df['market_id'].nunique()
    print(f'   Found {num_markets} distinct 15-minute markets')
    
    # Prepare
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    df['best_bid'] = df['best_bid'].ffill()
    df['best_ask'] = df['best_ask'].ffill()
    df = df.dropna(subset=['best_bid', 'best_ask'])
    
    df['mid'] = (df['best_bid'] + df['best_ask']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_pct'] = df['spread'] / df['mid']
    
    df['obi'] = (df['best_bid_size'] - df['best_ask_size']) / (df['best_bid_size'] + df['best_ask_size'] + 1e-10)
    df['obi'] = df['obi'].fillna(0).clip(-1, 1)
    
    return df


def backtest_single_market(df_market, entry_fn, exit_fn):
    """
    Backtest within a SINGLE 15-minute market.
    
    Returns PnL in dollars for this market.
    """
    capital = 1000  # Per-market allocation
    position = 0
    entry_price = 0
    trades = []
    
    # Only trade when progress < MAX_PROGRESS
    df_tradeable = df_market[df_market['progress'] < MAX_PROGRESS].copy()
    
    if len(df_tradeable) < 10:
        return 0, []
    
    # Resample to 100ms bars for speed
    df_bars = df_tradeable.resample('100ms').agg({
        'mid': 'last',
        'best_bid': 'last',
        'best_ask': 'last',
        'spread_pct': 'mean',
        'obi': 'mean',
        'progress': 'last'
    }).dropna()
    
    if len(df_bars) < 5:
        return 0, []
    
    # Calculate signals
    entries = entry_fn(df_bars)
    exits = exit_fn(df_bars)
    
    for i in range(len(df_bars)):
        row = df_bars.iloc[i]
        
        # Force close near end
        if row['progress'] > MAX_PROGRESS - 0.05 and position > 0:
            fill_price = row['best_bid'] * (1 - SLIPPAGE)
            proceeds = position * fill_price * (1 - TAKER_FEE)
            pnl = proceeds - position * entry_price
            capital += proceeds
            trades.append({'type': 'SELL', 'pnl': pnl, 'reason': 'force_close'})
            position = 0
            break
        
        # Exit signal
        if position > 0 and exits.iloc[i]:
            fill_price = row['best_bid'] * (1 - SLIPPAGE)
            proceeds = position * fill_price * (1 - TAKER_FEE)
            pnl = proceeds - position * entry_price
            capital += proceeds
            trades.append({'type': 'SELL', 'pnl': pnl, 'reason': 'signal'})
            position = 0
        
        # Entry signal
        if position == 0 and entries.iloc[i]:
            fill_price = row['best_ask'] * (1 + SLIPPAGE)
            cost = capital * 0.1 * (1 + TAKER_FEE)  # 10% of capital
            if cost < capital:
                position = (capital * 0.1) / fill_price
                entry_price = fill_price
                capital -= cost
                trades.append({'type': 'BUY'})
    
    # Calculate total PnL
    total_pnl = sum(t.get('pnl', 0) for t in trades if t['type'] == 'SELL')
    
    return total_pnl, trades


def strategy_spread_entry(df):
    return df['spread_pct'] > 0.03

def strategy_spread_exit(df):
    return df['spread_pct'] < 0.015

def strategy_mr_entry(df):
    df = df.copy()
    df['ma'] = df['mid'].rolling(20).mean()
    df['std'] = df['mid'].rolling(20).std()
    return ((df['mid'] - df['ma']) / (df['std'] + 1e-10)) < -1.5

def strategy_mr_exit(df):
    df = df.copy()
    df['ma'] = df['mid'].rolling(20).mean()
    df['std'] = df['mid'].rolling(20).std()
    return ((df['mid'] - df['ma']) / (df['std'] + 1e-10)) > 0.5

def strategy_random_entry(df):
    return pd.Series(np.random.random(len(df)) > 0.98, index=df.index)

def strategy_random_exit(df):
    return pd.Series(np.random.random(len(df)) > 0.95, index=df.index)


def main():
    print('=' * 70)
    print('📊 PROPER PER-MARKET BACKTEST')
    print('   Treating each 15-minute window as separate market')
    print('=' * 70 + '\n')
    
    df = load_and_segment()
    
    market_ids = df['market_id'].unique()
    
    # Split markets into train/test
    n_train = int(len(market_ids) * 0.7)
    train_markets = market_ids[:n_train]
    test_markets = market_ids[n_train:]
    
    print(f'\n📈 Train: {len(train_markets)} markets | Test: {len(test_markets)} markets')
    
    strategies = {
        'Spread Capture': (strategy_spread_entry, strategy_spread_exit),
        'Mean Reversion': (strategy_mr_entry, strategy_mr_exit),
        'RANDOM': (strategy_random_entry, strategy_random_exit),
    }
    
    print('\n' + '=' * 70)
    print('📊 RESULTS')
    print('=' * 70)
    
    for name, (entry_fn, exit_fn) in strategies.items():
        print(f'\n{name}:')
        
        # Train
        train_pnl = 0
        train_trades = 0
        train_wins = 0
        
        for mid in train_markets:
            df_m = df[df['market_id'] == mid]
            pnl, trades = backtest_single_market(df_m, entry_fn, exit_fn)
            train_pnl += pnl
            train_trades += len([t for t in trades if t['type'] == 'SELL'])
            train_wins += len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        
        train_wr = train_wins / train_trades * 100 if train_trades > 0 else 0
        
        # Test
        test_pnl = 0
        test_trades = 0
        test_wins = 0
        
        for mid in test_markets:
            df_m = df[df['market_id'] == mid]
            pnl, trades = backtest_single_market(df_m, entry_fn, exit_fn)
            test_pnl += pnl
            test_trades += len([t for t in trades if t['type'] == 'SELL'])
            test_wins += len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        
        test_wr = test_wins / test_trades * 100 if test_trades > 0 else 0
        
        # Per-market returns (assuming $1000 per market)
        train_ret = train_pnl / (1000 * len(train_markets)) * 100
        test_ret = test_pnl / (1000 * len(test_markets)) * 100
        
        print(f'   Train: PnL=${train_pnl:.2f} ({train_ret:.2f}%), Win Rate={train_wr:.1f}%, Trades={train_trades}')
        print(f'   Test:  PnL=${test_pnl:.2f} ({test_ret:.2f}%), Win Rate={test_wr:.1f}%, Trades={test_trades}')
    
    # Sanity check interpretation
    print('\n' + '=' * 70)
    print('🔍 INTERPRETATION')
    print('=' * 70)
    
    print('''
Key metrics to evaluate:
- Random strategy should have ~0% return and ~50% win rate
- Profitable strategies should show consistency between train/test
- Returns per market should be small (0.1-1% per 15-min window is good)
- A Sharpe > 2 annualized is still excellent at this scale

If random shows significant positive/negative returns, the sample
size (33 markets) is too small for statistical significance.
''')


if __name__ == '__main__':
    main()
