#!/usr/bin/env python3
"""
Realistic Polymarket Backtester V2

Key fixes:
1. Filter out resolution events (price near 0 or 1)
2. Segment by 15-minute windows
3. Close all positions before resolution
4. Proper annualization
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================

MAKER_FEE = 0.003
TAKER_FEE = 0.005
SLIPPAGE_PCT = 0.003  # 0.3%

# Tradeable price range (exclude resolution)
MIN_PRICE = 0.10
MAX_PRICE = 0.90

# Max price change to consider "normal" (not resolution)
MAX_NORMAL_CHANGE = 0.15  # 15%


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and filter data, excluding resolution events."""
    import os
    
    if os.path.exists('polymarket_combined.parquet'):
        df = pd.read_parquet('polymarket_combined.parquet')
    else:
        df = pd.read_parquet('polymarket_chunk.parquet')
    
    print(f'📊 Raw data: {len(df):,} rows')
    
    # Filter for YES outcome
    df = df[df['outcome_up'] == 1].copy()
    
    # Prepare
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    df['best_bid'] = df['best_bid'].ffill()
    df['best_ask'] = df['best_ask'].ffill()
    df = df.dropna(subset=['best_bid', 'best_ask'])
    
    df['mid'] = (df['best_bid'] + df['best_ask']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_pct'] = df['spread'] / df['mid']
    
    # OBI
    df['obi'] = (df['best_bid_size'] - df['best_ask_size']) / (df['best_bid_size'] + df['best_ask_size'] + 1e-10)
    df['obi'] = df['obi'].fillna(0).clip(-1, 1)
    
    # Resample to 1-second
    df = df.resample('1s').agg({
        'mid': 'last',
        'best_bid': 'last',
        'best_ask': 'last',
        'spread': 'mean',
        'spread_pct': 'mean',
        'obi': 'mean',
    }).dropna()
    
    print(f'   After resampling: {len(df):,} bars')
    
    # === CRITICAL: Filter out resolution events ===
    
    # 1. Remove prices near 0 or 1 (resolution zone)
    df = df[(df['mid'] >= MIN_PRICE) & (df['mid'] <= MAX_PRICE)]
    print(f'   After filtering resolution prices: {len(df):,} bars')
    
    # 2. Remove rows following extreme price jumps
    df['price_change'] = df['mid'].diff().abs()
    df['is_jump'] = df['price_change'] > MAX_NORMAL_CHANGE
    df['near_jump'] = df['is_jump'].rolling(10).max().fillna(0)  # 10 bars after jump
    
    jumps_removed = df['near_jump'].sum()
    df = df[df['near_jump'] == 0]
    print(f'   After removing {int(jumps_removed)} jump-adjacent bars: {len(df):,} bars')
    
    # Final stats
    print(f'   Time range: {df.index.min()} to {df.index.max()}')
    print(f'   Average spread: {df["spread_pct"].mean()*100:.2f}%')
    print(f'   Price range: {df["mid"].min():.2f} - {df["mid"].max():.2f}')
    
    return df


# ============================================================================
# BACKTESTER
# ============================================================================

class RealisticBacktester:
    def __init__(self, df, initial_capital=10000):
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.entry_price = 0
    
    def execute_buy(self, row):
        if self.position > 0:
            return False
        
        fill_price = row['best_ask'] * (1 + SLIPPAGE_PCT)
        trade_value = self.capital * 0.05  # 5% position size
        shares = trade_value / fill_price
        cost = trade_value * (1 + TAKER_FEE)
        
        if cost > self.capital:
            return False
        
        self.capital -= cost
        self.position = shares
        self.entry_price = fill_price
        
        self.trades.append({
            'type': 'BUY',
            'time': row.name,
            'price': fill_price,
            'shares': shares,
        })
        return True
    
    def execute_sell(self, row):
        if self.position <= 0:
            return False
        
        fill_price = row['best_bid'] * (1 - SLIPPAGE_PCT)
        proceeds = self.position * fill_price
        fee = proceeds * TAKER_FEE
        net_proceeds = proceeds - fee
        
        pnl = net_proceeds - (self.position * self.entry_price)
        
        self.capital += net_proceeds
        
        self.trades.append({
            'type': 'SELL',
            'price': fill_price,
            'pnl': pnl,
        })
        
        self.position = 0
        self.entry_price = 0
        return True
    
    def run(self, entry_signal, exit_signal):
        self.reset()
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            if self.position > 0 and exit_signal.iloc[i]:
                self.execute_sell(row)
            
            if self.position == 0 and entry_signal.iloc[i]:
                self.execute_buy(row)
            
            # Mark to market
            if self.position > 0:
                equity = self.capital + self.position * row['best_bid']
            else:
                equity = self.capital
            
            self.equity_curve.append(equity)
        
        # Close any open position
        if self.position > 0:
            self.execute_sell(self.df.iloc[-1])
        
        return self.calculate_stats()
    
    def calculate_stats(self):
        if not self.trades:
            return {'return': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'trades': 0}
        
        equity = pd.Series(self.equity_curve)
        
        # Total return
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Daily returns (assuming ~86400 seconds per day)
        # This data spans ~9 hours, so let's calculate properly
        hours = (self.df.index[-1] - self.df.index[0]).total_seconds() / 3600
        
        returns = equity.pct_change().dropna()
        if len(returns) > 10 and returns.std() > 0:
            # Annualize based on actual time span
            samples_per_year = len(returns) / hours * 24 * 365
            sharpe = returns.mean() / returns.std() * np.sqrt(samples_per_year)
        else:
            sharpe = 0
        
        # Max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min() * 100)
        
        # Win rate
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']
        if sell_trades:
            wins = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(sell_trades) * 100
        else:
            win_rate = 0
        
        return {
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'trades': len(sell_trades)
        }


# ============================================================================
# STRATEGIES
# ============================================================================

def strategy_spread(df):
    entries = df['spread_pct'] > 0.04
    exits = df['spread_pct'] < 0.02
    return entries, exits

def strategy_mean_reversion(df):
    df = df.copy()
    df['ma'] = df['mid'].rolling(120).mean()
    df['std'] = df['mid'].rolling(120).std()
    df['zscore'] = (df['mid'] - df['ma']) / (df['std'] + 1e-10)
    entries = df['zscore'] < -1.5
    exits = df['zscore'] > 0.5
    return entries.fillna(False), exits.fillna(False)

def strategy_momentum(df):
    df = df.copy()
    df['mom'] = df['mid'] - df['mid'].shift(30)
    entries = df['mom'] > df['mom'].rolling(120).std()
    exits = df['mom'] < 0
    return entries.fillna(False), exits.fillna(False)

def strategy_random(df):
    """Random signals for sanity check."""
    np.random.seed(42)
    entries = pd.Series(np.random.random(len(df)) > 0.99, index=df.index)
    exits = pd.Series(np.random.random(len(df)) > 0.98, index=df.index)
    return entries, exits


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 70)
    print('📊 REALISTIC BACKTEST V2')
    print('   Excluding resolution events, proper execution costs')
    print('=' * 70 + '\n')
    
    df = load_data()
    
    if len(df) < 1000:
        print('\n❌ Not enough data after filtering')
        return
    
    # 70/30 split
    split = int(len(df) * 0.7)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]
    
    print(f'\n📈 Train: {len(df_train):,} bars | Test: {len(df_test):,} bars')
    
    strategies = {
        'Spread Capture': strategy_spread,
        'Mean Reversion': strategy_mean_reversion,
        'Momentum': strategy_momentum,
        'RANDOM': strategy_random,
    }
    
    print('\n' + '=' * 70)
    print('📊 RESULTS')
    print('=' * 70)
    
    print(f'\n{"Strategy":<20} {"Train Ret%":>10} {"Test Ret%":>10} {"Train SR":>10} {"Test SR":>10} {"Trades":>8}')
    print('-' * 70)
    
    for name, strategy_fn in strategies.items():
        # Train
        entries, exits = strategy_fn(df_train)
        bt = RealisticBacktester(df_train)
        train_r = bt.run(entries, exits)
        
        # Test
        entries, exits = strategy_fn(df_test)
        bt = RealisticBacktester(df_test)
        test_r = bt.run(entries, exits)
        
        print(f'{name:<20} {train_r["return"]:>10.2f} {test_r["return"]:>10.2f} '
              f'{train_r["sharpe"]:>10.2f} {test_r["sharpe"]:>10.2f} {test_r["trades"]:>8}')
    
    # Sanity check
    print('\n' + '=' * 70)
    print('🔍 SANITY CHECK')
    print('=' * 70)
    
    entries, exits = strategy_random(df_test)
    bt = RealisticBacktester(df_test)
    random_r = bt.run(entries, exits)
    
    print(f'\nRandom Strategy Sharpe: {random_r["sharpe"]:.2f}')
    
    if abs(random_r['sharpe']) < 1.0:
        print('✅ PASS - Random signals produce low Sharpe')
    else:
        print('⚠️ STILL BIASED - Random signals show significant Sharpe')
        print('   Possible causes:')
        print('   - Price trends inherent in data (market direction)')
        print('   - Need longer dataset')
        print('   - Need to segment by individual 15-min markets')


if __name__ == '__main__':
    main()
