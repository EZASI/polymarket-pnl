#!/usr/bin/env python3
"""
BINANCE MULTI-TIMEFRAME PRICE PREDICTOR
========================================
Predicts BTC-USDT price direction at 1m, 5m, 10m, 15m horizons.

Data Sources:
  1. Live Binance API (free, no API key needed)
  2. HuggingFace: 123olp/binance-futures-ohlcv-2018-2026 (425M rows)
  3. Local parquet files (your existing data)

Architecture:
  - Feature engineering: 80+ features (momentum, volatility, microstructure, time)
  - Models: LightGBM ensemble + XGBoost + confidence calibration
  - Per-horizon models: separate model per timeframe
  - Confidence filter: only signal when confidence >= threshold
  - Walk-forward validation: realistic backtesting

Usage:
  # 1. Train on live Binance data (fetches last N days)
  python binance_predictor.py --mode train --days 30

  # 2. Train on HuggingFace dataset (massive historical data)
  python binance_predictor.py --mode train --source huggingface

  # 3. Train on your existing local parquet
  python binance_predictor.py --mode train --source local --data-path data/your_file.parquet

  # 4. Live prediction mode (fetches current data, predicts next moves)
  python binance_predictor.py --mode predict

  # 5. Backtest with walk-forward validation
  python binance_predictor.py --mode backtest --days 60

Requirements:
  pip install requests pandas numpy scikit-learn lightgbm xgboost joblib
  pip install datasets  # only for HuggingFace source
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "BTCUSDT"
HORIZONS = [1, 5, 10, 15]  # minutes ahead to predict
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

# Binance public API (no key needed)
BINANCE_API = "https://api.binance.com/api/v3"
BINANCE_KLINES = f"{BINANCE_API}/klines"

# Feature engineering periods
MOMENTUM_PERIODS = [1, 2, 3, 5, 10, 15, 20, 30, 60]
VOLATILITY_PERIODS = [5, 10, 15, 30, 60, 120]
MA_PERIODS = [5, 10, 20, 50, 100, 200]


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_binance_klines(
    symbol: str = SYMBOL,
    interval: str = "1m",
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Fetch klines from Binance public API (no auth needed)."""
    import requests

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    resp = requests.get(BINANCE_KLINES, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time", "close_time", "ignore"])

    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_binance_history(
    symbol: str = SYMBOL,
    interval: str = "1m",
    days: int = 7,
) -> pd.DataFrame:
    """Fetch multiple days of Binance kline data (paginated)."""
    import requests

    all_dfs = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
    current = start_ms

    total_expected = days * 24 * 60  # for 1m candles
    fetched = 0

    print(f"Fetching {days} days of {symbol} {interval} data from Binance...")

    while current < end_ms:
        try:
            df = fetch_binance_klines(
                symbol=symbol, interval=interval, limit=1000, start_time=current
            )
            if len(df) == 0:
                break

            all_dfs.append(df)
            fetched += len(df)

            # Move start to after the last candle
            last_ts = int(df["timestamp"].iloc[-1].timestamp() * 1000)
            current = last_ts + 60000  # +1 minute

            pct = min(100, fetched / total_expected * 100)
            print(f"\r  Fetched {fetched:,} candles ({pct:.0f}%)...", end="", flush=True)

            time.sleep(0.1)  # Rate limit respect
        except Exception as e:
            print(f"\n  Error: {e}, retrying in 2s...")
            time.sleep(2)

    print(f"\n  Done! Total: {fetched:,} candles")

    if not all_dfs:
        raise ValueError("No data fetched from Binance")

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return result


def load_huggingface_data(n_rows: int = 1_000_000) -> pd.DataFrame:
    """Load BTC data from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets library: pip install datasets")
        sys.exit(1)

    print(f"Loading HuggingFace dataset (last {n_rows:,} rows)...")
    print("  Dataset: 123olp/binance-futures-ohlcv-2018-2026")

    ds = load_dataset(
        "123olp/binance-futures-ohlcv-2018-2026",
        split="train",
        streaming=True,
    )

    # Filter for BTCUSDT only
    rows = []
    count = 0
    for row in ds:
        if row.get("symbol") == "BTCUSDT" or "BTC" in str(row.get("symbol", "")):
            rows.append(row)
            count += 1
            if count % 100000 == 0:
                print(f"  Loaded {count:,} rows...")
            if count >= n_rows:
                break

    if not rows:
        print("  BTCUSDT not found, loading any data...")
        ds = load_dataset(
            "123olp/binance-futures-ohlcv-2018-2026",
            split="train",
            streaming=True,
        )
        for row in ds:
            rows.append(row)
            count += 1
            if count >= n_rows:
                break

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df):,} rows")

    # Normalize columns
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "open" in cl and "time" not in cl:
            col_map[col] = "open"
        elif "high" in cl:
            col_map[col] = "high"
        elif "low" in cl:
            col_map[col] = "low"
        elif "close" in cl and "time" not in cl:
            col_map[col] = "close"
        elif "volume" in cl and "quote" not in cl and "taker" not in cl:
            col_map[col] = "volume"
        elif "quote" in cl and "volume" in cl:
            col_map[col] = "quote_volume"
        elif "trade" in cl or "count" in cl:
            col_map[col] = "trades"
        elif "time" in cl or "date" in cl:
            col_map[col] = "timestamp"

    df = df.rename(columns=col_map)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "open_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_local_parquet(path: str) -> pd.DataFrame:
    """Load local parquet file."""
    print(f"Loading local data: {path}")
    df = pd.read_parquet(path)

    # Normalize columns
    for col in df.columns:
        if col.lower() in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD, Signal, Histogram."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Compute Bollinger Bands."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Compute Stochastic %K, %D."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()
    return adx


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 80+ features for prediction.
    Features are computed from OHLCV data only (no external dependencies).
    """
    df = df.copy()

    # ── Price-based features ──
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Momentum at multiple scales
    for p in MOMENTUM_PERIODS:
        df[f"mom_{p}"] = df["close"].pct_change(p) * 100
        df[f"log_mom_{p}"] = np.log(df["close"] / df["close"].shift(p))

    # Moving averages and distances
    for p in MA_PERIODS:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()
        df[f"dist_ma_{p}"] = (df["close"] - df[f"ma_{p}"]) / df[f"ma_{p}"] * 100

    # MA crossover signals
    df["ma_5_20_cross"] = (df["ma_5"] > df["ma_20"]).astype(int)
    df["ma_10_50_cross"] = (df["ma_10"] > df["ma_50"]).astype(int)
    df["ma_20_100_cross"] = (df["ma_20"] > df["ma_100"]).astype(int)
    df["ma_50_200_cross"] = (df["ma_50"] > df["ma_200"]).astype(int)

    # Trend alignment score (0-4)
    df["trend_score"] = (
        df["ma_5_20_cross"] + df["ma_10_50_cross"] +
        df["ma_20_100_cross"] + df["ma_50_200_cross"]
    )

    # ── Volatility features ──
    for p in VOLATILITY_PERIODS:
        df[f"vol_{p}"] = df["log_return"].rolling(p).std() * 100
        df[f"range_{p}"] = (
            (df["high"].rolling(p).max() - df["low"].rolling(p).min())
            / df["close"] * 100
        )

    # ATR
    df["atr_14"] = compute_atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["close"] * 100
    df["atr_ratio"] = df["atr_pct"] / df["atr_pct"].rolling(100).mean()

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = compute_bollinger(df["close"], 20, 2.0)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid * 100
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ── Oscillators ──
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_7"] = compute_rsi(df["close"], 7)
    df["rsi_21"] = compute_rsi(df["close"], 21)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df, 14, 3)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["macd_bull"] = (df["macd"] > df["macd_signal"]).astype(int)

    # ADX
    df["adx"] = compute_adx(df, 14)

    # ── Candlestick features ──
    df["candle_body"] = (df["close"] - df["open"]) / df["open"] * 100
    df["candle_range"] = (df["high"] - df["low"]) / df["low"] * 100
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"] * 100
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"] * 100
    df["body_to_range"] = df["candle_body"].abs() / df["candle_range"].replace(0, np.nan)

    # ── Volume features ──
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_20"].replace(0, np.nan)
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)

    if "quote_volume" in df.columns:
        df["avg_trade_size"] = df["quote_volume"] / df["trades"].replace(0, np.nan)
    if "taker_buy_base" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)

    # ── Microstructure features ──
    if "trades" in df.columns:
        df["trades_ma"] = df["trades"].rolling(20).mean()
        df["trades_ratio"] = df["trades"] / df["trades_ma"].replace(0, np.nan)

    # ── Streak / Persistence features ──
    df["direction"] = np.sign(df["close"].diff())
    for n in [3, 5, 10]:
        df[f"up_ratio_{n}"] = (df["close"].diff() > 0).rolling(n).mean()

    # Consecutive same direction streak
    direction_change = (df["direction"] != df["direction"].shift()).cumsum()
    df["streak_len"] = df.groupby(direction_change).cumcount() + 1

    # ── Time features ──
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ── Composite signals ──
    # Signal alignment count
    df["bullish_signals"] = (
        (df["rsi_14"] > 50).astype(int) +
        df["macd_bull"] +
        (df["stoch_k"] > 50).astype(int) +
        (df["mom_5"] > 0).astype(int) +
        (df["mom_15"] > 0).astype(int) +
        df["trend_score"]
    )
    df["bearish_signals"] = 8 - df["bullish_signals"]

    # Extreme zones
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns (everything except targets, metadata, and leaky columns)."""
    exclude = {
        "timestamp", "open_time", "close_time",
        "target", "future_close", "change_pct",
        "direction", "quote_volume", "taker_buy_base", "taker_buy_quote",
    }
    # Exclude any target-related or future-leaking columns
    exclude.update(c for c in df.columns if "target_" in c or "future_" in c or "change_" in c)

    features = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]
    ]
    return features


# ============================================================
# MODEL TRAINING
# ============================================================
def create_targets(df: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Create target variables for each horizon."""
    df = df.copy()
    for h in horizons:
        df[f"future_{h}"] = df["close"].shift(-h)
        df[f"target_{h}"] = (df[f"future_{h}"] > df["close"]).astype(int)
        df[f"change_{h}"] = (df[f"future_{h}"] - df["close"]) / df["close"] * 100
    return df


def train_single_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    test_ratio: float = 0.2,
) -> dict:
    """Train model for a single prediction horizon."""
    try:
        import lightgbm as lgb
        USE_LGB = True
    except ImportError:
        USE_LGB = False

    try:
        import xgboost as xgb
        USE_XGB = True
    except ImportError:
        USE_XGB = False

    target_col = f"target_{horizon}"
    df_clean = df.dropna(subset=feature_cols + [target_col])

    if len(df_clean) < 1000:
        print(f"  ⚠ Not enough data for {horizon}m horizon ({len(df_clean)} rows)")
        return None

    # Time-based split
    split_idx = int(len(df_clean) * (1 - test_ratio))
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = []
    probas = []

    # Model 1: LightGBM (if available) - FAST and powerful
    if USE_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(X_train_s, y_train)
        prob = lgb_model.predict_proba(X_test_s)
        models.append(("lgb", lgb_model))
        probas.append(prob)

    # Model 2: XGBoost (if available)
    if USE_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb_model.fit(X_train_s, y_train)
        prob = xgb_model.predict_proba(X_test_s)
        models.append(("xgb", xgb_model))
        probas.append(prob)

    # Model 3: Only add sklearn GB if LightGBM and XGBoost are both missing
    if not USE_LGB and not USE_XGB:
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_train_s, y_train)
        prob = gb_model.predict_proba(X_test_s)
        models.append(("gb", gb_model))
        probas.append(prob)

    # Ensemble: average probabilities
    avg_prob = np.mean(probas, axis=0)
    y_pred = (avg_prob[:, 1] > 0.5).astype(int)
    confidence = np.max(avg_prob, axis=1)

    # Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)

    # Confidence-bucketed accuracy
    conf_results = []
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = confidence >= thresh
        if mask.sum() >= 10:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            trades = mask.sum()
            trades_day = trades / (len(test) / 1440)
            conf_results.append({
                "threshold": thresh,
                "accuracy": acc,
                "trades": int(trades),
                "trades_per_day": round(trades_day, 1),
            })

    # Feature importance (from best single model)
    if USE_LGB:
        imp_model = models[0][1]
        importances = imp_model.feature_importances_
    else:
        imp_model = models[-1][1]
        importances = imp_model.feature_importances_

    top_features = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    return {
        "horizon": horizon,
        "models": models,
        "scaler": scaler,
        "overall_accuracy": overall_acc,
        "confidence_results": conf_results,
        "top_features": top_features,
        "train_size": len(train),
        "test_size": len(test),
        "test_up_rate": float(y_test.mean()),
    }


def train_all_horizons(df: pd.DataFrame) -> dict:
    """Train models for all prediction horizons."""
    print("\n" + "=" * 70)
    print("TRAINING MULTI-TIMEFRAME PREDICTION MODELS")
    print("=" * 70)

    # Create targets
    df = create_targets(df, HORIZONS)

    # Engineer features (if not already done)
    if "rsi_14" not in df.columns:
        print("\nEngineering features...")
        df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Total rows: {len(df):,}")

    results = {}

    for h in HORIZONS:
        print(f"\n{'─' * 50}")
        print(f"Training {h}-minute horizon model...")
        print(f"{'─' * 50}")

        result = train_single_horizon(df, h, feature_cols)
        if result is None:
            continue

        # Store the feature columns used so live prediction matches
        result["feature_cols"] = feature_cols
        results[h] = result

        print(f"\n  Overall accuracy: {result['overall_accuracy']:.1%}")
        print(f"  Test up-rate: {result['test_up_rate']:.1%}")
        print(f"  Train: {result['train_size']:,} | Test: {result['test_size']:,}")

        print(f"\n  Confidence-filtered accuracy:")
        print(f"  {'Threshold':<12} {'Accuracy':<10} {'Trades':<10} {'Per Day':<10}")
        print(f"  {'─' * 45}")
        for cr in result["confidence_results"]:
            marker = " ✓" if cr["accuracy"] >= 0.70 else ""
            print(
                f"  {cr['threshold']:<12.0%} {cr['accuracy']:<10.1%} "
                f"{cr['trades']:<10} {cr['trades_per_day']:<10.1f}{marker}"
            )

        print(f"\n  Top features:")
        for fname, fimp in result["top_features"][:8]:
            print(f"    {fname:<30} {fimp:>8.0f}")

    return results


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================
def walk_forward_backtest(
    df: pd.DataFrame,
    horizon: int = 15,
    train_window: int = 20000,
    test_window: int = 5000,
    step: int = 5000,
) -> dict:
    """
    Walk-forward backtest: train on window, test on next window, slide forward.
    This gives the most realistic accuracy estimate.
    """
    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD BACKTEST ({horizon}-minute horizon)")
    print(f"{'=' * 70}")

    df = create_targets(df, [horizon])
    if "rsi_14" not in df.columns:
        df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    target_col = f"target_{horizon}"
    df_clean = df.dropna(subset=feature_cols + [target_col])

    print(f"  Clean data: {len(df_clean):,} rows")
    print(f"  Train window: {train_window:,} | Test window: {test_window:,} | Step: {step:,}")

    from sklearn.ensemble import GradientBoostingClassifier

    all_preds = []
    all_true = []
    all_conf = []
    fold = 0

    for start in range(0, len(df_clean) - train_window - test_window, step):
        train_end = start + train_window
        test_end = min(train_end + test_window, len(df_clean))

        train = df_clean.iloc[start:train_end]
        test = df_clean.iloc[train_end:test_end]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
        confidence = np.max(y_prob, axis=1)

        all_preds.extend(y_pred)
        all_true.extend(y_test)
        all_conf.extend(confidence)

        fold += 1
        acc = accuracy_score(y_test, y_pred)
        print(f"  Fold {fold}: Accuracy = {acc:.1%} ({len(test):,} samples)")

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_conf = np.array(all_conf)

    overall = accuracy_score(all_true, all_preds)
    print(f"\n  Walk-Forward Overall Accuracy: {overall:.1%}")

    print(f"\n  Confidence-filtered:")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'Trades':<10} {'Coverage':<10}")
    print(f"  {'─' * 45}")

    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = all_conf >= thresh
        if mask.sum() >= 10:
            acc = accuracy_score(all_true[mask], all_preds[mask])
            coverage = mask.mean()
            marker = " ✓" if acc >= 0.70 else ""
            print(f"  {thresh:<12.0%} {acc:<10.1%} {mask.sum():<10} {coverage:<10.1%}{marker}")

    return {
        "horizon": horizon,
        "overall_accuracy": overall,
        "folds": fold,
        "total_predictions": len(all_preds),
    }


# ============================================================
# LIVE PREDICTION
# ============================================================
def predict_live(results: dict | None = None):
    """
    Fetch current Binance data and make live predictions.
    """
    print("\n" + "=" * 70)
    print("LIVE PREDICTION MODE")
    print("=" * 70)

    # Fetch recent data
    print("\nFetching latest data from Binance...")
    df = fetch_binance_klines(symbol=SYMBOL, interval="1m", limit=1000)
    print(f"  Got {len(df)} candles")
    print(f"  Latest: {df['timestamp'].iloc[-1]}")
    print(f"  Price: ${df['close'].iloc[-1]:,.2f}")

    # Engineer features
    df = engineer_features(df)

    current_price = float(df["close"].iloc[-1])

    print(f"\n  Current BTC Price: ${current_price:,.2f}")
    print(f"  Time: {df['timestamp'].iloc[-1]}")

    if results:
        print(f"\n  {'Horizon':<10} {'Direction':<12} {'Confidence':<12} {'Signal'}")
        print(f"  {'─' * 50}")

        for h in HORIZONS:
            if h not in results:
                continue

            r = results[h]
            scaler = r["scaler"]

            # Use the same feature columns as training
            feature_cols = r.get("feature_cols", get_feature_columns(df))

            # Ensure all features exist, fill missing with 0
            latest = df.iloc[-1:].copy()
            for col in feature_cols:
                if col not in latest.columns:
                    latest[col] = 0.0

            X_latest = latest[feature_cols].values
            X_scaled = scaler.transform(X_latest)

            # Average predictions from all models
            probas = []
            for name, model in r["models"]:
                prob = model.predict_proba(X_scaled)
                probas.append(prob)

            avg_prob = np.mean(probas, axis=0)
            up_prob = avg_prob[0, 1]
            confidence = max(up_prob, 1 - up_prob)
            direction = "UP ↑" if up_prob > 0.5 else "DOWN ↓"

            # Signal strength
            if confidence >= 0.75:
                signal = "🟢 STRONG"
            elif confidence >= 0.65:
                signal = "🟡 MODERATE"
            elif confidence >= 0.55:
                signal = "⚪ WEAK"
            else:
                signal = "⚫ NO SIGNAL"

            print(f"  {h}m{'':<8} {direction:<12} {confidence:<12.1%} {signal}")
    else:
        # Quick prediction without trained models - use rule-based
        print("\n  No trained models loaded. Using rule-based analysis...")

        rsi = float(df["rsi_14"].iloc[-1]) if "rsi_14" in df.columns else 50
        mom_5 = float(df["mom_5"].iloc[-1]) if "mom_5" in df.columns else 0
        mom_15 = float(df["mom_15"].iloc[-1]) if "mom_15" in df.columns else 0
        trend = int(df["trend_score"].iloc[-1]) if "trend_score" in df.columns else 2
        macd_b = int(df["macd_bull"].iloc[-1]) if "macd_bull" in df.columns else 0
        adx = float(df["adx"].iloc[-1]) if "adx" in df.columns else 20

        print(f"\n  Indicators:")
        print(f"    RSI(14):  {rsi:.1f}")
        print(f"    Mom(5):   {mom_5:+.2f}%")
        print(f"    Mom(15):  {mom_15:+.2f}%")
        print(f"    Trend:    {trend}/4")
        print(f"    MACD:     {'Bullish' if macd_b else 'Bearish'}")
        print(f"    ADX:      {adx:.1f}")

        # Simple signal
        bullish = sum([
            rsi > 50, mom_5 > 0, mom_15 > 0, trend >= 3, macd_b == 1, adx > 25
        ])
        bearish = 6 - bullish

        if bullish >= 5:
            print(f"\n  Signal: 🟢 BULLISH ({bullish}/6 indicators)")
        elif bearish >= 5:
            print(f"\n  Signal: 🔴 BEARISH ({bearish}/6 indicators)")
        else:
            print(f"\n  Signal: ⚪ NEUTRAL ({bullish}/6 bullish)")


# ============================================================
# SAVE / LOAD MODELS
# ============================================================
def save_models(results: dict, path: Path = MODEL_DIR):
    """Save trained models to disk."""
    import joblib

    path.mkdir(parents=True, exist_ok=True)

    for h, r in results.items():
        model_path = path / f"horizon_{h}m.joblib"
        save_data = {
            "models": r["models"],
            "scaler": r["scaler"],
            "overall_accuracy": r["overall_accuracy"],
            "confidence_results": r["confidence_results"],
            "top_features": r["top_features"],
            "feature_cols": r.get("feature_cols", []),
        }
        joblib.dump(save_data, model_path)
        print(f"  Saved {h}m model to {model_path}")


def load_models(path: Path = MODEL_DIR) -> dict:
    """Load trained models from disk."""
    import joblib

    results = {}
    for h in HORIZONS:
        model_path = path / f"horizon_{h}m.joblib"
        if model_path.exists():
            data = joblib.load(model_path)
            results[h] = data
            print(f"  Loaded {h}m model from {model_path}")

    return results


# ============================================================
# MAIN
# ============================================================
def print_summary(results: dict):
    """Print a nice summary of all results."""
    print("\n" + "=" * 70)
    print("📊 PREDICTION SUMMARY")
    print("=" * 70)

    print(f"\n{'Horizon':<10} {'Overall':<12} {'Best Filtered':<15} {'Trades/Day':<12} {'Threshold':<10}")
    print("─" * 62)

    for h in HORIZONS:
        if h not in results:
            continue

        r = results[h]
        overall = r["overall_accuracy"]

        # Find best filtered accuracy with reasonable trade count
        best = None
        for cr in r["confidence_results"]:
            if cr["trades_per_day"] >= 5 and cr["accuracy"] > (best or {}).get("accuracy", 0):
                best = cr

        if best:
            print(
                f"{h}m{'':<8} {overall:<12.1%} {best['accuracy']:<15.1%} "
                f"{best['trades_per_day']:<12.1f} {best['threshold']:<10.0%}"
            )
        else:
            print(f"{h}m{'':<8} {overall:<12.1%} {'N/A':<15} {'N/A':<12} {'N/A':<10}")

    # Profit estimation
    print(f"\n{'─' * 62}")
    print("ESTIMATED DAILY PROFIT (per horizon, $1000 position)")
    print(f"{'─' * 62}")

    for h in HORIZONS:
        if h not in results:
            continue
        r = results[h]
        for cr in r["confidence_results"]:
            if cr["accuracy"] >= 0.65 and cr["trades_per_day"] >= 5:
                win_rate = cr["accuracy"]
                # Polymarket: buy at ~$0.50, win pays $1.00
                ev = win_rate * 0.50 - (1 - win_rate) * 0.50
                position = 1000
                daily = ev * position * cr["trades_per_day"]
                print(
                    f"  {h}m @ {cr['threshold']:.0%} conf: "
                    f"WR={win_rate:.0%}, {cr['trades_per_day']:.0f} trades/day, "
                    f"~${daily:,.0f}/day"
                )
                break


def main():
    parser = argparse.ArgumentParser(description="Binance Multi-Timeframe Predictor")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "backtest"],
        default="train",
        help="Operation mode",
    )
    parser.add_argument(
        "--source",
        choices=["binance", "huggingface", "local"],
        default="binance",
        help="Data source",
    )
    parser.add_argument("--days", type=int, default=14, help="Days of Binance data to fetch")
    parser.add_argument("--data-path", type=str, default=None, help="Local parquet path")
    parser.add_argument("--rows", type=int, default=500000, help="Max rows from HuggingFace")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--save", action="store_true", help="Save trained models")

    args = parser.parse_args()

    global SYMBOL
    SYMBOL = args.symbol

    print("=" * 70)
    print("BINANCE MULTI-TIMEFRAME PRICE PREDICTOR")
    print(f"Symbol: {SYMBOL} | Horizons: {HORIZONS}")
    print("=" * 70)

    if args.mode == "predict":
        # Try to load saved models
        if MODEL_DIR.exists():
            print("\nLoading saved models...")
            results = load_models()
        else:
            results = None

        predict_live(results)
        return

    # Load data
    if args.source == "binance":
        df = fetch_binance_history(symbol=SYMBOL, days=args.days)
    elif args.source == "huggingface":
        df = load_huggingface_data(n_rows=args.rows)
    elif args.source == "local":
        if not args.data_path:
            # Try to find existing parquet files
            parquets = list(DATA_DIR.glob("*.parquet"))
            if parquets:
                args.data_path = str(parquets[0])
                print(f"Found local data: {args.data_path}")
            else:
                print("ERROR: No --data-path specified and no parquet files in data/")
                sys.exit(1)
        df = load_local_parquet(args.data_path)

    # Validate data
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"\nData loaded: {len(df):,} rows")
    if "timestamp" in df.columns:
        print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():,.2f} → ${df['close'].max():,.2f}")

    # Engineer features
    print("\nEngineering 80+ features...")
    df = engineer_features(df)

    if args.mode == "train":
        results = train_all_horizons(df)
        print_summary(results)

        if args.save:
            print("\nSaving models...")
            save_models(results)

        # Also do a live prediction with the fresh models
        predict_live(results)

    elif args.mode == "backtest":
        for h in HORIZONS:
            walk_forward_backtest(df, horizon=h)


if __name__ == "__main__":
    main()
