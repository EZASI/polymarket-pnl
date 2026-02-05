#!/usr/bin/env python3
"""
Fast CEX Price Predictor - Uses sample data for quick iteration
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_sample(path: str, n_rows: int = 100000) -> pd.DataFrame:
    """Load only last N rows for fast iteration."""
    table = pq.read_table(path)
    total = table.num_rows
    start = max(0, total - n_rows)
    df = table.slice(start, n_rows).to_pandas()
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'MA_20', 'MA_50', 'MA_200', 'RSI', '%K', '%D', 
                    'ADX', 'ATR', 'MACD', 'BL_Upper', 'BL_Lower', 
                    'Signal', 'Histogram', 'MN_Upper', 'MN_Lower']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def create_features(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    """Create target and features."""
    # Target: 15-min ahead direction
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df['price_change_pct'] = (df['future_close'] - df['close']) / df['close'] * 100
    
    # Momentum
    df['mom_5'] = df['close'].pct_change(5) * 100
    df['mom_15'] = df['close'].pct_change(15) * 100
    
    # Volatility
    df['vol_15'] = df['close'].rolling(15).std() / df['close'].rolling(15).mean() * 100
    
    # Price vs MA
    df['price_vs_ma20'] = (df['close'] - df['MA_20']) / df['MA_20'] * 100
    df['price_vs_ma50'] = (df['close'] - df['MA_50']) / df['MA_50'] * 100
    
    # RSI zones
    df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
    df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
    
    # MACD signal
    df['macd_positive'] = (df['MACD'] > 0).astype(int)
    df['macd_above_signal'] = (df['MACD'] > df['Signal']).astype(int)
    
    # Recent direction
    df['up_ratio_10'] = (df['close'].diff() > 0).rolling(10).mean()
    
    # Trend strength
    df['trend_strong'] = (df['ADX'] > 25).astype(int)
    
    return df


def train_model(df: pd.DataFrame):
    """Train and evaluate."""
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'MA_20', 'MA_50', 'MA_200', 'RSI', '%K', '%D', 'ADX', 'ATR',
        'MACD', 'Signal', 'Histogram',
        'mom_5', 'mom_15', 'vol_15',
        'price_vs_ma20', 'price_vs_ma50',
        'rsi_oversold', 'rsi_overbought',
        'macd_positive', 'macd_above_signal',
        'up_ratio_10', 'trend_strong'
    ]
    
    # Clean data
    df_clean = df.dropna(subset=feature_cols + ['target'])
    print(f"Clean samples: {len(df_clean):,}")
    
    # Time split
    split = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split]
    test = df_clean.iloc[split:]
    
    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test = test[feature_cols].values
    y_test = test['target'].values
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    overall = accuracy_score(y_test, y_pred)
    print(f"\nOVERALL ACCURACY: {overall:.1%}")
    
    # Confidence filtering
    print("\n" + "="*60)
    print("CONFIDENCE-FILTERED ACCURACY")
    print("="*60)
    
    confidence = np.max(y_prob, axis=1)
    
    print(f"\n{'Threshold':<12} | {'Accuracy':<10} | {'Trades':<10} | {'Per Day':<10}")
    print("-" * 50)
    
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = confidence >= thresh
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            trades = mask.sum()
            trades_day = trades / (len(test) / 1440)
            marker = "✓ 90%+" if acc >= 0.90 else ""
            print(f"{thresh:<12.0%} | {acc:<10.1%} | {trades:<10} | {trades_day:<10.0f} {marker}")
    
    # Find 90% threshold
    print("\n" + "="*60)
    print("SEARCHING FOR 90% ACCURACY")
    print("="*60)
    
    for thresh in np.arange(0.50, 0.99, 0.01):
        mask = confidence >= thresh
        if mask.sum() >= 10:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            if acc >= 0.90:
                trades = mask.sum()
                trades_day = trades / (len(test) / 1440)
                print(f"\n✓ FOUND 90%+ at threshold {thresh:.0%}")
                print(f"  Accuracy: {acc:.1%}")
                print(f"  Trades: {trades}")
                print(f"  Trades/day: {trades_day:.0f}")
                
                # Calculate expected profit
                win_rate = acc
                # Assuming buy at 0.50, win pays $1.00
                ev_per_trade = win_rate * 0.50 - (1 - win_rate) * 0.50
                position = 2000  # $2000 per trade
                profit_trade = ev_per_trade * position
                daily_profit = profit_trade * trades_day
                
                print(f"\n  EXPECTED PROFIT:")
                print(f"  - Win rate: {win_rate:.1%}")
                print(f"  - Position: ${position:,}")
                print(f"  - Profit/trade: ${profit_trade:.0f}")
                print(f"  - Trades/day: {trades_day:.0f}")
                print(f"  - Daily profit: ${daily_profit:,.0f}")
                break
    else:
        print("\n✗ Could not achieve 90% with confidence alone")
        
        # Try extreme confidence
        for thresh in np.arange(0.95, 0.999, 0.01):
            mask = confidence >= thresh
            if mask.sum() >= 5:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  Trying {thresh:.1%}: {acc:.1%} ({mask.sum()} trades)")
    
    return model, scaler


def main():
    print("="*60)
    print("FAST CEX PREDICTOR FOR POLYMARKET")
    print("="*60)
    
    path = "data/WinkingFace_CryptoLM-Bitcoin-BTC-USDT_train.parquet"
    
    print("\nLoading 100k sample rows...")
    df = load_sample(path, n_rows=100000)
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\nCreating features...")
    df = create_features(df, horizon=15)
    
    print("\nTraining model...")
    model, scaler = train_model(df)


if __name__ == "__main__":
    main()
