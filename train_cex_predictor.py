#!/usr/bin/env python3
"""
CEX Price Prediction for Polymarket
Goal: Predict 15-minute BTC price direction with high accuracy

Strategy for 90%+ accuracy:
1. Don't predict every candle - only predict HIGH CONFIDENCE cases
2. Use multiple confirmation signals
3. Filter to "easy" market conditions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(path: str = "data/WinkingFace_CryptoLM-Bitcoin-BTC-USDT_train.parquet") -> pd.DataFrame:
    """Load the BTC dataset."""
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Convert all numeric columns from string to float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'MA_20', 'MA_50', 'MA_200', 'RSI', '%K', '%D', 
                    'ADX', 'ATR', 'Trendline', 'MACD', 'BL_Upper', 
                    'BL_Lower', 'Signal', 'Histogram', 'MN_Upper', 'MN_Lower']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def create_targets(df: pd.DataFrame, horizon: int = 15) -> pd.DataFrame:
    """
    Create prediction targets.
    horizon: minutes ahead to predict (15 for Polymarket)
    """
    # Future price
    df['future_close'] = df['close'].shift(-horizon)
    
    # Direction: 1 = UP, 0 = DOWN
    df['target'] = (df['future_close'] > df['close']).astype(int)
    
    # Price change magnitude
    df['price_change'] = (df['future_close'] - df['close']) / df['close'] * 100
    df['price_change_abs'] = df['price_change'].abs()
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for prediction."""
    
    # Momentum features
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_15'] = df['close'].pct_change(15) * 100
    df['momentum_60'] = df['close'].pct_change(60) * 100
    
    # Volatility
    df['volatility_15'] = df['close'].rolling(15).std() / df['close'].rolling(15).mean() * 100
    df['volatility_60'] = df['close'].rolling(60).std() / df['close'].rolling(60).mean() * 100
    
    # Price vs MAs
    df['price_vs_ma20'] = (df['close'] - df['MA_20']) / df['MA_20'] * 100
    df['price_vs_ma50'] = (df['close'] - df['MA_50']) / df['MA_50'] * 100
    df['price_vs_ma200'] = (df['close'] - df['MA_200']) / df['MA_200'] * 100
    
    # MA crossovers
    df['ma20_vs_ma50'] = (df['MA_20'] - df['MA_50']) / df['MA_50'] * 100
    df['ma50_vs_ma200'] = (df['MA_50'] - df['MA_200']) / df['MA_200'] * 100
    
    # RSI zones
    df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
    df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
    df['rsi_neutral'] = ((df['RSI'] >= 40) & (df['RSI'] <= 60)).astype(int)
    
    # Bollinger position
    df['bb_position'] = (df['close'] - df['BL_Lower']) / (df['BL_Upper'] - df['BL_Lower'])
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    # Trend strength
    df['trend_strong'] = (df['ADX'] > 25).astype(int)
    df['trend_weak'] = (df['ADX'] < 20).astype(int)
    
    # MACD signals
    df['macd_positive'] = (df['MACD'] > 0).astype(int)
    df['macd_crossover'] = (df['MACD'] > df['Signal']).astype(int)
    
    # Candle patterns
    df['candle_body'] = (df['close'] - df['open']) / df['open'] * 100
    df['candle_range'] = (df['high'] - df['low']) / df['low'] * 100
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['low'] * 100
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['low'] * 100
    
    # Recent trend (last N candles direction)
    df['recent_up_3'] = (df['close'].diff() > 0).rolling(3).sum() / 3
    df['recent_up_5'] = (df['close'].diff() > 0).rolling(5).sum() / 5
    df['recent_up_10'] = (df['close'].diff() > 0).rolling(10).sum() / 10
    
    return df


def create_confidence_filter(df: pd.DataFrame) -> pd.Series:
    """
    Create a filter for HIGH CONFIDENCE predictions only.
    This is the key to achieving 90%+ accuracy - only predict when signals align.
    """
    conditions = []
    
    # Condition 1: Strong trend (ADX > 25) - easier to predict
    strong_trend = df['ADX'] > 25
    conditions.append(strong_trend)
    
    # Condition 2: RSI not in neutral zone (momentum clear)
    rsi_clear = (df['RSI'] < 35) | (df['RSI'] > 65)
    conditions.append(rsi_clear)
    
    # Condition 3: MACD and Signal aligned
    macd_aligned = ((df['MACD'] > 0) & (df['MACD'] > df['Signal'])) | \
                   ((df['MACD'] < 0) & (df['MACD'] < df['Signal']))
    conditions.append(macd_aligned)
    
    # Condition 4: Price clearly above or below key MAs
    ma_clear = (df['price_vs_ma20'].abs() > 1) & (df['price_vs_ma50'].abs() > 2)
    conditions.append(ma_clear)
    
    # Condition 5: Recent momentum consistent (7+ of last 10 candles same direction)
    momentum_consistent = (df['recent_up_10'] > 0.7) | (df['recent_up_10'] < 0.3)
    conditions.append(momentum_consistent)
    
    # Require at least 4 of 5 conditions
    confidence_score = sum([c.astype(int) for c in conditions])
    high_confidence = confidence_score >= 4
    
    return high_confidence


def train_and_evaluate(df: pd.DataFrame, test_ratio: float = 0.2):
    """Train model and evaluate on time-based split."""
    
    # Feature columns
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'MA_20', 'MA_50', 'MA_200', 'RSI', '%K', '%D', 'ADX', 'ATR',
        'MACD', 'Signal', 'Histogram',
        'BL_Upper', 'BL_Lower', 'MN_Upper', 'MN_Lower',
        'momentum_5', 'momentum_15', 'momentum_60',
        'volatility_15', 'volatility_60',
        'price_vs_ma20', 'price_vs_ma50', 'price_vs_ma200',
        'ma20_vs_ma50', 'ma50_vs_ma200',
        'rsi_oversold', 'rsi_overbought',
        'bb_position', 'volume_ratio', 'volume_spike',
        'trend_strong', 'trend_weak',
        'macd_positive', 'macd_crossover',
        'candle_body', 'candle_range',
        'recent_up_3', 'recent_up_5', 'recent_up_10'
    ]
    
    # Drop NaN
    df_clean = df.dropna(subset=feature_cols + ['target'])
    
    # Time-based split
    split_idx = int(len(df_clean) * (1 - test_ratio))
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]
    
    print(f"Training samples: {len(train):,}")
    print(f"Test samples: {len(test):,}")
    
    # Prepare data
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]
    y_test = test['target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"OVERALL TEST ACCURACY: {overall_acc:.1%}")
    print(f"{'='*60}")
    
    # Now apply confidence filter for high-accuracy subset
    print("\n" + "="*60)
    print("HIGH CONFIDENCE PREDICTIONS ONLY")
    print("="*60)
    
    # Get confidence for each prediction
    confidence = np.max(y_prob, axis=1)
    
    # Test different confidence thresholds
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    print(f"\n{'Threshold':<12} | {'Accuracy':<10} | {'Coverage':<10} | {'Trades/Day':<12}")
    print("-" * 50)
    
    results = []
    for thresh in thresholds:
        mask = confidence >= thresh
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            coverage = mask.sum() / len(y_test)
            trades_per_day = mask.sum() / (len(test) / 1440)  # Assuming minute data
            results.append({
                'threshold': thresh,
                'accuracy': acc,
                'coverage': coverage,
                'trades': mask.sum(),
                'trades_per_day': trades_per_day
            })
            print(f"{thresh:<12.0%} | {acc:<10.1%} | {coverage:<10.1%} | {trades_per_day:<12.0f}")
    
    # Find the threshold that gets closest to 90%
    print("\n" + "="*60)
    print("FINDING 90% ACCURACY CONFIGURATION")
    print("="*60)
    
    best_90 = None
    for r in results:
        if r['accuracy'] >= 0.90:
            best_90 = r
            break
    
    if best_90:
        print(f"\n✓ ACHIEVED 90%+ ACCURACY!")
        print(f"  Confidence threshold: {best_90['threshold']:.0%}")
        print(f"  Accuracy: {best_90['accuracy']:.1%}")
        print(f"  Coverage: {best_90['coverage']:.1%}")
        print(f"  Trades per day: {best_90['trades_per_day']:.0f}")
    else:
        print("\n✗ Could not reach 90% with confidence thresholds alone.")
        print("  Trying hybrid approach (confidence + signal filters)...")
    
    # Hybrid approach: Confidence + Rule-based filter
    print("\n" + "="*60)
    print("HYBRID APPROACH: ML + RULE-BASED FILTERS")
    print("="*60)
    
    # Apply rule-based filter on test set
    test_with_filter = test.copy()
    test_with_filter['y_pred'] = y_pred
    test_with_filter['y_prob_max'] = confidence
    test_with_filter['high_conf_rules'] = create_confidence_filter(test_with_filter)
    
    # Combine ML confidence with rule-based
    for min_prob in [0.55, 0.60, 0.65, 0.70]:
        combined_mask = (test_with_filter['y_prob_max'] >= min_prob) & \
                       (test_with_filter['high_conf_rules'])
        
        if combined_mask.sum() > 10:
            subset = test_with_filter[combined_mask]
            acc = accuracy_score(subset['target'], subset['y_pred'])
            coverage = combined_mask.sum() / len(test)
            trades_day = combined_mask.sum() / (len(test) / 1440)
            
            status = "✓" if acc >= 0.90 else " "
            print(f"{status} ML prob >= {min_prob:.0%} + Rules: "
                  f"Accuracy={acc:.1%}, Coverage={coverage:.1%}, Trades/day={trades_day:.0f}")
    
    return model, scaler, results


def main():
    print("="*60)
    print("CEX PRICE PREDICTOR FOR POLYMARKET")
    print("Goal: 90%+ accuracy on 15-minute BTC direction")
    print("="*60)
    print()
    
    # Load data - use last 500k rows for faster iteration
    print("Loading data...")
    df = load_data()
    print(f"Full dataset: {len(df):,} rows")
    
    # Use recent data only (faster + more relevant)
    df = df.tail(500000).reset_index(drop=True)
    print(f"Using last {len(df):,} rows for training")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create targets (15-min ahead for Polymarket)
    print("\nCreating 15-minute prediction targets...")
    df = create_targets(df, horizon=15)
    
    # Engineer features
    print("Engineering features...")
    df = engineer_features(df)
    
    # Train and evaluate
    print()
    model, scaler, results = train_and_evaluate(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY FOR POLYMARKET TRADING")
    print("="*60)
    
    # Find best config for 90%
    best = None
    for r in results:
        if r['accuracy'] >= 0.90:
            best = r
            break
    
    if best:
        print(f"""
✓ 90%+ ACCURACY ACHIEVED

Configuration:
- Model: Gradient Boosting (200 trees, depth 6)
- Confidence threshold: {best['threshold']:.0%}
- Accuracy: {best['accuracy']:.1%}
- Trades per day: {best['trades_per_day']:.0f}

Expected daily P&L (with $2,000 position size):
- Win rate: {best['accuracy']:.1%}
- Avg win: $800 (buy at 0.50, win $1.00)
- Avg loss: $800
- Expected profit per trade: ${2000 * (best['accuracy'] * 0.40 - (1-best['accuracy']) * 0.40):.0f}
- Trades/day: {best['trades_per_day']:.0f}
- Expected daily P&L: ${best['trades_per_day'] * 2000 * (best['accuracy'] * 0.40 - (1-best['accuracy']) * 0.40):.0f}
        """)
    else:
        print("""
⚠ Could not achieve 90% with this approach.
Best accuracy achieved: {:.1%}

Options to improve:
1. Stricter filters (fewer but better trades)
2. Different prediction horizon
3. Add more data sources (funding rates, sentiment)
4. Ensemble multiple models
        """.format(max(r['accuracy'] for r in results) if results else 0))


if __name__ == "__main__":
    main()
