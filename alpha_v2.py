#!/usr/bin/env python3
"""
ALPHA v2: The 61% Machine
===========================

Thesis from stress test v2:
  - 1m is the ONLY horizon with edge (54.5% WR, 0.79 Sharpe)
  - Edge lives in LOW VOLATILITY regimes (62-65% WR when ATR < 0.05%)
  - Microstructure signals (taker imbalance) decay within seconds
  - 5m/10m/15m all LOSE money — the signal is ultra-short-lived

Attack plan to reach 61%:

  1. CROSS-ASSET LEAD-LAG
     ETH moves before BTC by 1-3 bars at the 1m level.
     SOL is even noisier — its spikes are a canary.
     Compute: ETH mom → BTC future direction correlation.

  2. ORDERBOOK IMBALANCE VELOCITY
     Not the level of imbalance, but the CHANGE in imbalance.
     A rising bid wall is more predictive than a static one.

  3. INVERTED VOLATILITY GATE
     Only trade when ATR < 50th percentile.
     In low-vol, small signals have high signal-to-noise ratio.

  4. PSYCHOLOGICAL PRICE LEVELS
     BTC respects round numbers ($60k, $65k, $70k).
     Distance to nearest $1000 level affects mean-reversion.

  5. MICROSTRUCTURE DELTA FEATURES
     Change in taker_ratio over 1, 3, 5 bars — acceleration of buying/selling.
     Trade count acceleration — "something is happening" detector.

  6. STACKING ENSEMBLE
     Layer 1: LightGBM + XGBoost (diverse tree models)
     Layer 2: Logistic regression on Layer 1 probabilities + top features
     This captures non-linear interactions the trees miss.

  7. REGIME-AWARE FEATURE SELECTION
     Remove features that only work in one regime.
     Keep features with stable importance across folds.

Usage:
  python3 alpha_v2.py --days 30
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

from ultra_predictor import (
    BinanceDataCollector,
    engineer_all_features,
    create_targets,
    get_feature_cols,
)

# ============================================================
# CONFIG
# ============================================================
TRAIN_DAYS = 7
TEST_DAYS = 2
TRAIN_W = TRAIN_DAYS * 1440
TEST_W = TEST_DAYS * 1440

COST_SLIP = 0.02
COST_SPREAD = 0.01
BUY_PRICE = 0.50
PNL_WIN = 1.00 - (BUY_PRICE * (1 + COST_SLIP) + COST_SPREAD)    # +0.48
PNL_LOSS = -(BUY_PRICE * (1 + COST_SLIP) + COST_SPREAD)          # -0.52


# ============================================================
# MULTI-ASSET DATA COLLECTION
# ============================================================
def fetch_multi_asset(days: int = 30) -> dict[str, pd.DataFrame]:
    """Fetch BTC, ETH, SOL kline data in parallel-ish fashion."""
    assets = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"  Fetching {symbol}...")
        c = BinanceDataCollector(symbol)
        df = c.get_klines_history(days=days)
        if not df.empty:
            assets[symbol] = df
            print(f"    {len(df):,} candles")
        else:
            print(f"    FAILED")
    return assets


def fetch_derivatives(collector: BinanceDataCollector) -> dict:
    """Fetch all derivatives data."""
    import requests

    data = {}

    # Funding rate
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                            params={"symbol": "BTCUSDT", "limit": 1000}, timeout=15)
        if resp.ok:
            data["funding"] = pd.DataFrame(resp.json())
            data["funding"]["fundingRate"] = pd.to_numeric(data["funding"]["fundingRate"])
            data["funding"]["fundingTime"] = pd.to_datetime(data["funding"]["fundingTime"], unit="ms", utc=True)
    except:
        pass

    # Long/Short ratio
    try:
        resp = requests.get("https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                            params={"symbol": "BTCUSDT", "period": "5m", "limit": 500}, timeout=15)
        if resp.ok:
            data["ls_ratio"] = pd.DataFrame(resp.json())
            for c in ["longShortRatio", "longAccount", "shortAccount"]:
                if c in data["ls_ratio"].columns:
                    data["ls_ratio"][c] = pd.to_numeric(data["ls_ratio"][c])
            data["ls_ratio"]["timestamp"] = pd.to_datetime(data["ls_ratio"]["timestamp"], unit="ms", utc=True)
    except:
        pass

    # Taker buy/sell
    try:
        resp = requests.get("https://fapi.binance.com/futures/data/takerlongshortRatio",
                            params={"symbol": "BTCUSDT", "period": "5m", "limit": 500}, timeout=15)
        if resp.ok:
            data["taker"] = pd.DataFrame(resp.json())
            for c in ["buySellRatio", "buyVol", "sellVol"]:
                if c in data["taker"].columns:
                    data["taker"][c] = pd.to_numeric(data["taker"][c])
            data["taker"]["timestamp"] = pd.to_datetime(data["taker"]["timestamp"], unit="ms", utc=True)
    except:
        pass

    # Open interest
    try:
        resp = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
                            params={"symbol": "BTCUSDT", "period": "5m", "limit": 500}, timeout=15)
        if resp.ok:
            data["oi"] = pd.DataFrame(resp.json())
            for c in ["sumOpenInterest", "sumOpenInterestValue"]:
                if c in data["oi"].columns:
                    data["oi"][c] = pd.to_numeric(data["oi"][c])
            data["oi"]["timestamp"] = pd.to_datetime(data["oi"]["timestamp"], unit="ms", utc=True)
    except:
        pass

    return data


# ============================================================
# ALPHA v2 FEATURE ENGINEERING
# ============================================================
def build_alpha_v2_features(btc: pd.DataFrame, eth: pd.DataFrame,
                             sol: pd.DataFrame, deriv: dict) -> pd.DataFrame:
    """
    Build the full alpha v2 feature set.
    Returns BTC dataframe with all features merged.
    """
    df = btc.copy()

    # ── 1. BASE FEATURES (from ultra_predictor) ──
    df = engineer_all_features(df)

    # ── 2. CROSS-ASSET LEAD-LAG ──
    # Align ETH and SOL by timestamp
    if eth is not None and len(eth) > 0:
        eth = eth.set_index("timestamp").sort_index()
        eth_close = eth["close"].rename("eth_close")
        eth_vol = eth["volume"].rename("eth_volume")

        df = df.set_index("timestamp").sort_index()
        df = df.join(eth_close, how="left")
        df = df.join(eth_vol, how="left")
        df["eth_close"] = df["eth_close"].ffill()
        df["eth_volume"] = df["eth_volume"].ffill()
        df = df.reset_index()

        # ETH momentum (LAGGED — ETH's recent move predicts BTC's next move)
        for p in [1, 2, 3, 5]:
            df[f"eth_mom_{p}"] = df["eth_close"].pct_change(p) * 100

        # ETH-BTC spread momentum (relative value)
        df["eth_btc_ratio"] = df["eth_close"] / df["close"]
        df["eth_btc_ratio_change"] = df["eth_btc_ratio"].pct_change(3) * 100

        # ETH leads BTC: shift ETH momentum forward by 1 bar
        # If ETH moved up 1 bar ago, does BTC follow?
        for p in [1, 2, 3]:
            df[f"eth_lead_{p}"] = df[f"eth_mom_{p}"].shift(1)

        # Cross-asset volume divergence
        df["eth_vol_ratio"] = df["eth_volume"] / df["eth_volume"].rolling(20).mean()
        df["vol_diverge"] = df["vol_ratio"] - df["eth_vol_ratio"]  # BTC vol vs ETH vol

    if sol is not None and len(sol) > 0:
        sol = sol.set_index("timestamp").sort_index()
        sol_close = sol["close"].rename("sol_close")

        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()
        df = df.join(sol_close, how="left")
        df["sol_close"] = df["sol_close"].ffill()
        if "timestamp" not in df.columns:
            df = df.reset_index()

        for p in [1, 2, 3]:
            df[f"sol_mom_{p}"] = df["sol_close"].pct_change(p) * 100
            df[f"sol_lead_{p}"] = df[f"sol_mom_{p}"].shift(1)

    if "timestamp" in df.index.names:
        df = df.reset_index()

    # ── 3. MICROSTRUCTURE DELTA FEATURES ──
    # Change in taker ratio (acceleration of buying/selling)
    if "taker_ratio" in df.columns:
        for p in [1, 2, 3, 5]:
            df[f"taker_delta_{p}"] = df["taker_ratio"].diff(p)
        df["taker_accel"] = df["taker_delta_1"].diff(1)  # 2nd derivative

    # Trade count acceleration
    if "trades" in df.columns:
        df["trades_delta_1"] = df["trades"].diff(1)
        df["trades_delta_3"] = df["trades"].diff(3)
        df["trades_accel"] = df["trades_delta_1"].diff(1)

    # Volume acceleration
    df["vol_delta_1"] = df["volume"].diff(1)
    df["vol_accel"] = df["vol_delta_1"].diff(1)

    # ── 4. PSYCHOLOGICAL PRICE LEVELS ──
    df["dist_1000"] = (df["close"] % 1000) / 1000  # distance to nearest $1k (0-1)
    df["dist_500"] = (df["close"] % 500) / 500
    df["near_round"] = ((df["dist_1000"] < 0.02) | (df["dist_1000"] > 0.98)).astype(int)

    # Distance from session (24h rolling) high/low
    df["session_high"] = df["high"].rolling(1440).max()
    df["session_low"] = df["low"].rolling(1440).min()
    df["dist_session_high"] = (df["session_high"] - df["close"]) / df["close"] * 100
    df["dist_session_low"] = (df["close"] - df["session_low"]) / df["close"] * 100

    # ── 5. REGIME FEATURES ──
    # ATR percentile (for inverted gate)
    df["atr_pct_rank"] = df["atr_pct"].rolling(1440).rank(pct=True)
    df["low_vol_regime"] = (df["atr_pct_rank"] < 0.50).astype(int)
    df["ultra_low_vol"] = (df["atr_pct_rank"] < 0.25).astype(int)

    # Trend clarity: are all MAs aligned clearly?
    df["trend_clarity"] = (
        (df["trend_score"] >= 3).astype(int) +
        (df["trend_score"] <= 1).astype(int)  # clear up OR clear down
    )

    # Range contraction (Bollinger squeeze → breakout anticipation)
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(100).quantile(0.2)).astype(int)

    # ── 6. INFORMATION FLOW FEATURES ──
    # Shannon entropy of recent returns (high entropy = random, low = predictable)
    def rolling_entropy(series, window=20):
        def entropy(x):
            # Discretize into 5 bins
            counts = np.histogram(x, bins=5)[0]
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        return series.rolling(window).apply(entropy, raw=True)

    df["return_entropy_20"] = rolling_entropy(df["log_ret"].fillna(0), 20)
    df["return_entropy_60"] = rolling_entropy(df["log_ret"].fillna(0), 60)
    df["low_entropy"] = (df["return_entropy_20"] < df["return_entropy_20"].rolling(200).quantile(0.3)).astype(int)

    # ── 7. MEAN REVERSION SIGNALS ──
    # Z-score of price vs recent distribution
    for p in [20, 60, 120]:
        ma = df["close"].rolling(p).mean()
        std = df["close"].rolling(p).std()
        df[f"zscore_{p}"] = (df["close"] - ma) / std.replace(0, np.nan)

    # Overextension: consecutive bars in same direction beyond threshold
    df["overextended_up"] = (
        (df["mom_5"] > 0.3) & (df["rsi"] > 70) & (df["bb_pos"] > 0.95)
    ).astype(int)
    df["overextended_down"] = (
        (df["mom_5"] < -0.3) & (df["rsi"] < 30) & (df["bb_pos"] < 0.05)
    ).astype(int)

    # ── 8. MERGE DERIVATIVES (forward-fill) ──
    if "funding" in deriv and len(deriv["funding"]) > 0:
        fd = deriv["funding"].sort_values("fundingTime")
        df["fund_rate"] = np.nan
        for _, row in fd.iterrows():
            mask = df["timestamp"] >= row["fundingTime"]
            if mask.any():
                df.loc[mask.idxmax(), "fund_rate"] = row["fundingRate"]
        df["fund_rate"] = df["fund_rate"].ffill().bfill().fillna(0)
        df["fund_rate_pct"] = df["fund_rate"] * 100
        df["fund_extreme_long"] = (df["fund_rate_pct"] > 0.05).astype(int)
        df["fund_extreme_short"] = (df["fund_rate_pct"] < -0.01).astype(int)

    if "ls_ratio" in deriv and len(deriv["ls_ratio"]) > 0:
        ls = deriv["ls_ratio"].sort_values("timestamp")
        df["ls_ratio"] = np.nan
        for _, row in ls.iterrows():
            mask = df["timestamp"] >= row["timestamp"]
            if mask.any():
                df.loc[mask.idxmax(), "ls_ratio"] = row.get("longShortRatio", np.nan)
        df["ls_ratio"] = df["ls_ratio"].ffill().bfill().fillna(1.0)
        df["crowd_extreme"] = ((df["ls_ratio"] > 2.5) | (df["ls_ratio"] < 0.5)).astype(int)

    if "taker" in deriv and len(deriv["taker"]) > 0:
        tk = deriv["taker"].sort_values("timestamp")
        df["taker_ls_fut"] = np.nan
        for _, row in tk.iterrows():
            mask = df["timestamp"] >= row["timestamp"]
            if mask.any():
                df.loc[mask.idxmax(), "taker_ls_fut"] = row.get("buySellRatio", np.nan)
        df["taker_ls_fut"] = df["taker_ls_fut"].ffill().bfill().fillna(1.0)

    if "oi" in deriv and len(deriv["oi"]) > 0:
        oi = deriv["oi"].sort_values("timestamp")
        df["oi_val"] = np.nan
        for _, row in oi.iterrows():
            mask = df["timestamp"] >= row["timestamp"]
            if mask.any():
                df.loc[mask.idxmax(), "oi_val"] = row.get("sumOpenInterestValue", np.nan)
        df["oi_val"] = df["oi_val"].ffill().bfill().fillna(0)
        df["oi_change_5m"] = df["oi_val"].pct_change(5) * 100

    # Fill any remaining NaN in non-essential columns
    for c in df.columns:
        if df[c].dtype in [np.float64, np.int64, float, int] and df[c].isna().any():
            df[c] = df[c].ffill().bfill().fillna(0)

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================
def get_alpha_features(df: pd.DataFrame) -> list[str]:
    """Get feature columns, excluding targets and leaky columns."""
    exclude_patterns = ["target", "future", "change_", "timestamp", "open_time", "close_time"]
    exclude_exact = {"quote_volume", "taker_buy_base", "taker_buy_quote",
                     "eth_close", "sol_close", "eth_volume", "session_high", "session_low"}

    cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(p in c for p in exclude_patterns):
            continue
        if df[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]:
            # Drop features with zero variance
            if df[c].std() > 0:
                cols.append(c)
    return cols


# ============================================================
# SELECTIVE FILTER v2
# ============================================================
def filter_v2(df, y_pred, confidence):
    """
    v2 filters: STRICT selection. Trade LESS but BETTER.

    Key insight from stress test:
      - Folds with ATR < 0.05% → 57-60% WR
      - Folds with ATR > 0.10% → 49-51% WR (noise)
      - MORE trades = LOWER quality
      - The path to 61% is FEWER, HIGHER-QUALITY trades

    Rules:
      1. HARD ATR gate: only trade when atr_pct_rank < 0.50
      2. ML confidence >= 0.60 minimum
      3. At least ONE of: VolSpike+Mom, Trend+RSI, ETH confirmation
      4. ETH must not contradict (if available)
    """
    n = len(y_pred)

    def col(name):
        return df[name].values[:n] if name in df.columns else np.zeros(n)

    vs = col("vol_spike")
    m3 = col("mom_3")
    m5 = col("mom_5")
    ts = col("trend_score")
    rsi = col("rsi")
    vr = col("vol_ratio")
    ti = col("taker_imbalance") if "taker_imbalance" in df.columns else col("taker_ratio") - 0.5
    atr_rank = col("atr_pct_rank")
    entropy = col("low_entropy")

    # ETH lead-lag
    eth1 = col("eth_lead_1")
    eth2 = col("eth_lead_2")
    has_eth = "eth_lead_1" in df.columns

    # ═══ HARD GATE: Low volatility regime only ═══
    vol_gate = atr_rank < 0.50

    # ═══ MINIMUM CONFIDENCE ═══
    conf_gate = confidence >= 0.60

    # ═══ SIGNAL CONFIRMATIONS ═══

    # Signal A: Volume Spike + Momentum
    sig_a = (vs == 1) & (
        ((y_pred == 1) & (m3 > 0.1)) | ((y_pred == 0) & (m3 < -0.1)))

    # Signal B: Trend + RSI (tighter than v1)
    sig_b = (
        ((y_pred == 1) & (ts >= 2) & (rsi > 48) & (rsi < 68)) |
        ((y_pred == 0) & (ts <= 1) & (rsi > 32) & (rsi < 52)))

    # Signal C: Taker imbalance aligned
    sig_c = (
        ((y_pred == 1) & (ti > 0.015)) |
        ((y_pred == 0) & (ti < -0.015)))

    # Signal D: ETH confirms direction
    if has_eth:
        sig_d = (
            ((y_pred == 1) & (eth1 > 0.03) & (eth2 > 0.02)) |
            ((y_pred == 0) & (eth1 < -0.03) & (eth2 < -0.02)))

        # ETH VETO: if ETH strongly contradicts, don't trade
        eth_veto = (
            ((y_pred == 1) & (eth1 < -0.1)) |
            ((y_pred == 0) & (eth1 > 0.1)))
    else:
        sig_d = np.ones(n, dtype=bool)
        eth_veto = np.zeros(n, dtype=bool)

    # Signal E: Low entropy (market is predictable, not random)
    sig_e = entropy == 1

    # ═══ COMBINE: gate + confidence + at least 2 signals + no ETH veto ═══
    signal_count = (
        sig_a.astype(int) + sig_b.astype(int) + sig_c.astype(int) +
        sig_d.astype(int) + sig_e.astype(int)
    )

    mask = vol_gate & conf_gate & (signal_count >= 2) & (~eth_veto)

    return mask


# ============================================================
# STACKING ENSEMBLE
# ============================================================
def train_stacking_ensemble(X_train, y_train, X_test, feat_cols):
    """
    Layer 1: LightGBM + XGBoost with different hyperparams
    Layer 2: Logistic Regression on Layer 1 probabilities
    """
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_ts = sc.transform(X_test)

    l1_train = []
    l1_test = []

    # LightGBM variant 1 (deep, slow learning)
    if lgb:
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.04,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=30, random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_train)
        l1_train.append(m.predict_proba(X_tr)[:, 1])
        l1_test.append(m.predict_proba(X_ts)[:, 1])

    # LightGBM variant 2 (shallow, fast learning)
    if lgb:
        m = lgb.LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            num_leaves=8, subsample=0.9, colsample_bytree=0.6,
            min_child_samples=50, random_state=123, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_train)
        l1_train.append(m.predict_proba(X_tr)[:, 1])
        l1_test.append(m.predict_proba(X_ts)[:, 1])

    # XGBoost
    if xgb:
        m = xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_tr, y_train)
        l1_train.append(m.predict_proba(X_tr)[:, 1])
        l1_test.append(m.predict_proba(X_ts)[:, 1])

    if not l1_train:
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        m.fit(X_tr, y_train)
        l1_train.append(m.predict_proba(X_tr)[:, 1])
        l1_test.append(m.predict_proba(X_ts)[:, 1])

    # Layer 2: Stack L1 outputs
    L1_tr = np.column_stack(l1_train)
    L1_ts = np.column_stack(l1_test)

    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(L1_tr, y_train)

    final_prob = lr.predict_proba(L1_ts)
    y_pred = (final_prob[:, 1] > 0.5).astype(int)
    confidence = np.max(final_prob, axis=1)

    return y_pred, confidence, final_prob[:, 1]


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_v2(df, feat_cols, verbose=True):
    target = "target_1"
    clean = df.dropna(subset=feat_cols + [target]).reset_index(drop=True)
    n = len(clean)

    if n < TRAIN_W + TEST_W:
        print(f"  ERROR: Need {TRAIN_W + TEST_W:,}, have {n:,}")
        return None

    all_trades = []
    fold = 0
    start = 0

    while start + TRAIN_W + TEST_W <= n:
        te = start + TRAIN_W
        oe = min(te + TEST_W, n)

        tr = clean.iloc[start:te]
        ts_df = clean.iloc[te:oe].reset_index(drop=True)

        X_train = tr[feat_cols].values
        y_train = tr[target].values
        X_test = ts_df[feat_cols].values
        y_test = ts_df[target].values

        # Stacking ensemble
        y_pred, confidence, up_prob = train_stacking_ensemble(X_train, y_train, X_test, feat_cols)

        # Selective filter v2
        mask = filter_v2(ts_df, y_pred, confidence)
        idxs = np.where(mask)[0]

        # Fold regime stats
        fold_atr = float(ts_df["atr_pct"].mean()) if "atr_pct" in ts_df.columns else 0
        price_chg = 0
        if "close" in ts_df.columns and len(ts_df) > 1:
            price_chg = (ts_df["close"].iloc[-1] - ts_df["close"].iloc[0]) / ts_df["close"].iloc[0] * 100

        for idx in idxs:
            correct = y_pred[idx] == y_test[idx]
            all_trades.append({
                "fold": fold,
                "correct": correct,
                "confidence": float(confidence[idx]),
                "fold_atr": fold_atr,
                "fold_price_chg": price_chg,
            })

        if verbose:
            wr = accuracy_score(y_test[idxs], y_pred[idxs]) if len(idxs) > 0 else 0
            overall = accuracy_score(y_test, y_pred)
            print(
                f"  Fold {fold:>2d} | All {overall:.1%} | "
                f"Filtered: {len(idxs):>4d} trades, WR {wr:.1%} | "
                f"ATR {fold_atr:.3f}% | BTC {price_chg:+.1f}%")

        fold += 1
        start += TEST_W

    return {"trades": all_trades, "folds": fold}


# ============================================================
# ANALYSIS
# ============================================================
def sharpe(rets, per=252):
    return float(np.mean(rets) / np.std(rets) * np.sqrt(per)) if len(rets) > 1 and np.std(rets) > 0 else 0

def max_dd(eq):
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / np.where(pk > 0, pk, 1)
    return float(np.max(dd)) if len(dd) > 0 else 0

def monte_carlo(pnls, n_runs=1000):
    rng = np.random.default_rng(42)
    n = len(pnls)
    cap = abs(pnls).sum() * 0.1
    finals = np.zeros(n_runs)
    max_dds = np.zeros(n_runs)
    sharpes = np.zeros(n_runs)
    ruin = 0
    for i in range(n_runs):
        s = rng.permutation(pnls)
        eq = np.cumsum(s)
        finals[i] = eq[-1]
        sharpes[i] = sharpe(s)
        max_dds[i] = max_dd(np.concatenate([[0], eq]) + cap)
        if np.min(eq) < -cap * 0.5:
            ruin += 1
    return {
        "pnl_5": np.percentile(finals, 5), "pnl_50": np.median(finals),
        "pnl_95": np.percentile(finals, 95),
        "sharpe_5": np.percentile(sharpes, 5), "sharpe_50": np.median(sharpes),
        "mdd_50": np.median(max_dds), "mdd_95": np.percentile(max_dds, 95),
        "p_profit": np.mean(finals > 0), "p_ruin": ruin / n_runs,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--mc-runs", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 72)
    print("  ALPHA v2: The 61% Machine")
    print("=" * 72)

    # ── Fetch multi-asset data ──
    print(f"\n[1/5] Fetching {args.days} days of multi-asset data...")
    assets = fetch_multi_asset(days=args.days)

    if "BTCUSDT" not in assets:
        print("ERROR: No BTC data"); sys.exit(1)

    btc = assets["BTCUSDT"]
    eth = assets.get("ETHUSDT")
    sol = assets.get("SOLUSDT")

    print(f"\n  BTC: {len(btc):,} candles | "
          f"${btc['close'].min():,.0f} → ${btc['close'].max():,.0f}")

    # ── Fetch derivatives ──
    print(f"\n[2/5] Fetching derivatives data...")
    collector = BinanceDataCollector("BTCUSDT")
    deriv = fetch_derivatives(collector)
    print(f"  Funding: {len(deriv.get('funding', []))} | "
          f"L/S: {len(deriv.get('ls_ratio', []))} | "
          f"Taker: {len(deriv.get('taker', []))} | "
          f"OI: {len(deriv.get('oi', []))}")

    # ── Build alpha v2 features ──
    print(f"\n[3/5] Building alpha v2 features...")
    df = build_alpha_v2_features(btc, eth, sol, deriv)
    df = create_targets(df)

    feat_cols = get_alpha_features(df)
    print(f"  Total features: {len(feat_cols)}")

    # Verify clean data
    clean_n = len(df.dropna(subset=feat_cols + ["target_1"]))
    print(f"  Clean rows: {clean_n:,} / {len(df):,}")

    # ── Walk-Forward v2 ──
    print(f"\n[4/5] Walk-Forward Backtest (7d train / 2d OOS, stacking ensemble)")
    print("─" * 72)

    result = walk_forward_v2(df, feat_cols, verbose=True)

    if not result or not result["trades"]:
        print("  No trades produced."); sys.exit(1)

    trades = result["trades"]
    correct = np.array([t["correct"] for t in trades])
    pnls = np.where(correct, PNL_WIN, PNL_LOSS) * 100  # $100 per trade
    wr = correct.mean()

    n_folds = result["folds"]
    test_days = n_folds * TEST_DAYS
    tpd = len(trades) / test_days if test_days > 0 else 0

    eq = np.cumsum(pnls)
    eq_c = np.concatenate([[0], eq]) + abs(pnls).sum() * 0.1

    sr = sharpe(pnls)
    mdd = max_dd(eq_c)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    pf = wins.sum() / abs(losses).sum() if abs(losses).sum() > 0 else float("inf")

    # Equity curve quality
    n_trades = len(pnls)
    q_size = n_trades // 4
    q_pnls = [pnls[i*q_size:(i+1)*q_size].sum() for i in range(4)]

    print(f"""
{'═' * 72}
  ALPHA v2 RESULTS (1m horizon, $100/trade, 2% slippage + $0.01 spread)
{'═' * 72}

  Total trades:     {len(trades):,}
  Trades/day:       {tpd:.1f}
  WIN RATE:         {wr:.2%}  {'✓ TARGET HIT' if wr >= 0.61 else f'(target: 61%, gap: {0.61-wr:.1%})'}

  Total P&L:        ${pnls.sum():+,.0f}
  Avg trade P&L:    ${pnls.mean():+.2f}
  Sharpe Ratio:     {sr:.3f}
  Max Drawdown:     {mdd:.1%}
  Profit Factor:    {pf:.2f}

  Kelly fraction:   {max(0, (wr * abs(PNL_WIN) - (1-wr) * abs(PNL_LOSS)) / abs(PNL_WIN)):.1%}

  Equity curve by quartile:
    Q1: ${q_pnls[0]:>+10,.0f}
    Q2: ${q_pnls[1]:>+10,.0f}
    Q3: ${q_pnls[2]:>+10,.0f}
    Q4: ${q_pnls[3]:>+10,.0f}
""")

    # Per-fold detail
    folds = sorted(set(t["fold"] for t in trades))
    print(f"  {'Fold':<6} {'Trades':<8} {'WR':<10} {'ATR%':<8} {'BTC Δ%':<10}")
    print(f"  {'─' * 50}")
    for f in folds:
        ft = [t for t in trades if t["fold"] == f]
        fwr = np.mean([t["correct"] for t in ft])
        atr = ft[0]["fold_atr"]
        btc_chg = ft[0]["fold_price_chg"]
        marker = " ★" if fwr >= 0.61 else (" ✗" if fwr < 0.48 else "")
        print(f"  {f:<6} {len(ft):<8} {fwr:<10.1%} {atr:<8.3f} {btc_chg:<+10.1f}{marker}")

    # ── Monte Carlo ──
    print(f"\n[5/5] Monte Carlo ({args.mc_runs:,} shuffles)")
    print("─" * 72)

    mc = monte_carlo(pnls, n_runs=args.mc_runs)
    print(f"""
  Sharpe:    5th={mc['sharpe_5']:+.3f}  med={mc['sharpe_50']:+.3f}
  MaxDD:     med={mc['mdd_50']:.1%}  95th={mc['mdd_95']:.1%}
  P&L:       5th=${mc['pnl_5']:+,.0f}  med=${mc['pnl_50']:+,.0f}  95th=${mc['pnl_95']:+,.0f}
  P(profit): {mc['p_profit']:.1%}
  P(ruin):   {mc['p_ruin']:.1%}
""")

    # ── COMPARISON vs BASELINE ──
    print(f"{'═' * 72}")
    print(f"  COMPARISON: ALPHA v2 vs BASELINE (v1)")
    print(f"{'═' * 72}")
    print(f"""
  Metric          BASELINE (v1)    ALPHA v2        Change
  ─────────────────────────────────────────────────────────
  Win Rate         54.5%            {wr:.1%}            {(wr - 0.545)*100:+.1f}pp
  Sharpe           0.793            {sr:.3f}           {sr - 0.793:+.3f}
  Max DD           11.9%            {mdd:.1%}            {(mdd - 0.119)*100:+.1f}pp
  Profit Factor    1.11             {pf:.2f}            {pf - 1.11:+.2f}
  Trades/day       125              {tpd:.0f}              {tpd - 125:+.0f}

  DATA SOURCES ADDED in v2:
    ✓ ETH-BTC lead-lag (1-3 bar lag)
    ✓ SOL momentum (canary signal)
    ✓ Microstructure deltas (taker accel, trade accel)
    ✓ Psychological price levels ($1000, $500 round numbers)
    ✓ Return entropy (predictability detector)
    ✓ Inverted ATR gate (low-vol regime filter)
    ✓ Z-score mean reversion signals
    ✓ Bollinger squeeze detector
    ✓ Stacking ensemble (LGBx2 + XGB → LogReg)
""")

    if wr >= 0.61:
        print("  🎯 TARGET ACHIEVED: 61%+ Win Rate after full stress testing!")
    elif wr >= 0.58:
        print("  ↗ SIGNIFICANT IMPROVEMENT. Close to 61% target.")
        print("  Next: try 60-day data window for more stable features.")
    elif wr > 0.545:
        print("  ↗ IMPROVED over baseline. More data or feature tuning needed.")
    else:
        print("  → No improvement. The new features may need different horizons.")


if __name__ == "__main__":
    main()
