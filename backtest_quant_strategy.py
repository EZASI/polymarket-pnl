#!/usr/bin/env python3
"""
backtest_quant_strategy.py — Full backtest of the Polymarket Quant Trading
research brief strategies on REAL Polymarket + Binance historical data.

Fetches resolved 5-minute BTC Up/Down markets from Polymarket Gamma API
(~17,000 markets available since Dec 17 2025) + Binance 1m klines for
feature engineering, then tests the strategies from the research brief:

  1. Lead-lag signal: CEX return + OFI → p_theo via Φ(drift_z / (vol * √T))
  2. Ensemble ML: XGBoost, LightGBM, HistGradientBoosting, ExtraTrees, RF
     with double calibration (Platt sigmoid + isotonic regression)
  3. Kelly sizing: f* = (P - q) / (1 - q) with fractional scaling
  4. Edge tracking: Brier score, D_KL, reliability diagrams, Sharpe

Data sources:
  - Polymarket Gamma API: real resolved 5m BTC markets with outcomes + volume
  - Binance REST API: 1m BTCUSDT klines for feature engineering

Usage:
    python backtest_quant_strategy.py [--days 7] [--interval 5] [--bankroll 10000]
    python backtest_quant_strategy.py --use-real-pm --pm-rounds 2000
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtest_quant")

CACHE_DIR = Path("/Users/eiji/polymarket-pnl/data/cache")


# ============================================================================
# 0. Local Data Cache
# ============================================================================


def save_cache(df: pd.DataFrame, name: str) -> Path:
    """Save DataFrame to local cache as parquet (fast) or CSV (fallback)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = CACHE_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    log.info("Cached %d rows → %s", len(df), path)
    return path


def load_cache(name: str) -> pd.DataFrame | None:
    """Load DataFrame from local cache. Returns None if not found."""
    for ext in ("parquet", "csv"):
        path = CACHE_DIR / f"{name}.{ext}"
        if path.exists():
            try:
                df = pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path)
                log.info("Loaded %d rows from cache: %s", len(df), path)
                return df
            except Exception as e:
                log.warning("Cache load failed for %s: %s", path, e)
    return None


def cache_key_pm(n_rounds: int) -> str:
    return f"pm_5m_markets_{n_rounds}"


def cache_key_binance(days: int) -> str:
    return f"binance_1m_klines_{days}d"


# ============================================================================
# 1. Data Fetching — Binance 1m klines
# ============================================================================


def fetch_binance_klines(days: int = 7) -> pd.DataFrame:
    """Fetch historical 1m BTC/USDT klines from Binance."""
    log.info("Fetching %d days of 1m BTCUSDT klines from Binance...", days)

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    all_klines = []
    current = start_ms

    while current < end_ms:
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": current,
                    "endTime": min(current + 1000 * 60 * 1000, end_ms),
                    "limit": 1000,
                },
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_klines.extend(batch)
            current = batch[-1][0] + 60_000
            if len(all_klines) % 5000 == 0:
                log.info("  Fetched %d candles...", len(all_klines))
        except Exception as e:
            log.warning("Binance fetch error: %s — retrying", e)
            time.sleep(2)
            current += 60_000 * 100
        time.sleep(0.1)  # rate limit

    df = pd.DataFrame(
        all_klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "_",
        ],
    )
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["ts"] = df["open_time"] // 1000
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Fetched %d total 1m candles (%.1f days)", len(df), len(df) / 1440)
    return df


# ============================================================================
# 2a. REAL Polymarket Data — Fetch resolved 5m BTC markets from Gamma API
# ============================================================================


def fetch_polymarket_5m_markets(n_rounds: int = 2000,
                                 start_ts: int = None) -> pd.DataFrame:
    """
    Fetch real resolved 5-minute BTC Up/Down markets from Polymarket Gamma API.

    Markets launched Dec 17, 2025 (ts ~1766031900). Each market's slug follows
    the pattern: btc-updown-5m-{unix_timestamp} where timestamp is a multiple
    of 300 (5 minutes).

    Returns DataFrame with: round_ts, outcome (1=UP, 0=DOWN), volume,
    last_trade_price, up_resolved_price.
    """
    log.info("Fetching %d resolved 5m BTC markets from Polymarket Gamma API...", n_rounds)

    if start_ts is None:
        # Start from 2 hours ago and go backwards
        now = int(time.time())
        start_ts = ((now - 7200) // 300) * 300

    results = []
    miss_streak = 0
    checked = 0

    # Go backwards from start_ts
    current_ts = start_ts
    while len(results) < n_rounds and miss_streak < 50:
        slug = f"btc-updown-5m-{current_ts}"
        try:
            resp = requests.get(
                f"https://gamma-api.polymarket.com/events?slug={slug}",
                timeout=5,
            )
            data = resp.json()
            if data:
                miss_streak = 0
                evt = data[0]
                markets = evt.get("markets", [])
                if markets:
                    mk = markets[0]
                    closed = mk.get("closed", False)
                    if not closed:
                        current_ts -= 300
                        continue

                    prices = mk.get("outcomePrices", "[]")
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    up_price = float(prices[0]) if prices else 0.5
                    outcome = 1 if up_price > 0.5 else 0
                    volume = float(mk.get("volume", 0))
                    last_trade = float(mk.get("lastTradePrice", 0.5))

                    results.append({
                        "round_ts": current_ts,
                        "outcome": outcome,
                        "volume": volume,
                        "last_trade_price": last_trade,
                        "up_resolved_price": up_price,
                    })
            else:
                miss_streak += 1
        except Exception as e:
            miss_streak += 1
            if miss_streak % 10 == 0:
                log.warning("  Miss streak %d at ts=%d: %s", miss_streak, current_ts, e)

        checked += 1
        current_ts -= 300

        if checked % 3 == 0:
            time.sleep(0.35)  # rate limit

        if len(results) % 200 == 0 and len(results) > 0:
            log.info("  Fetched %d/%d markets...", len(results), n_rounds)

    df = pd.DataFrame(results)
    if df.empty:
        log.error("No Polymarket markets found!")
        return df

    df.sort_values("round_ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info("Fetched %d resolved markets (checked %d timestamps)", len(df), checked)
    log.info("  Date range: %s to %s",
             datetime.fromtimestamp(df["round_ts"].min(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
             datetime.fromtimestamp(df["round_ts"].max(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
    log.info("  Outcomes: UP=%d (%.1f%%) DOWN=%d (%.1f%%)",
             df["outcome"].sum(), 100 * df["outcome"].mean(),
             len(df) - df["outcome"].sum(), 100 * (1 - df["outcome"].mean()))
    log.info("  Avg volume: $%.0f/market", df["volume"].mean())
    return df


# ============================================================================
# 2b. Simulated Binary Markets — "Will BTC be up in N minutes?"
# ============================================================================


def create_binary_markets(klines: pd.DataFrame, interval_min: int = 5) -> pd.DataFrame:
    """
    Create simulated binary prediction markets from kline data.

    Each 'market' asks: "Will BTC close higher than the open of this
    N-minute interval?"

    Returns DataFrame with:
      - round_ts: start timestamp of the interval
      - open_price: BTC price at interval start
      - close_price: BTC price at interval end
      - outcome: 1 if close > open, 0 otherwise
      - simulated_market_price: what the 'market' would price YES at
        (based on recent momentum — simulates a naive market)
    """
    # Group klines into N-minute intervals
    klines = klines.copy()
    klines["interval_group"] = klines["ts"] // (interval_min * 60)

    markets = []
    groups = klines.groupby("interval_group")

    for gid, group in groups:
        if len(group) < interval_min:
            continue
        open_price = group.iloc[0]["open"]
        close_price = group.iloc[-1]["close"]
        high = group["high"].max()
        low = group["low"].min()
        volume = group["volume"].sum()
        taker_buy = group["taker_buy_base"].sum()
        n_trades = group["trades"].sum()

        outcome = 1 if close_price > open_price else 0

        # Simulate what a naive market would price this at
        # Use trailing momentum as a proxy for market consensus
        markets.append({
            "round_ts": int(group.iloc[0]["ts"]),
            "open_price": open_price,
            "close_price": close_price,
            "high": high,
            "low": low,
            "volume": volume,
            "taker_buy": taker_buy,
            "n_trades": n_trades,
            "outcome": outcome,
        })

    df = pd.DataFrame(markets)
    df.sort_values("round_ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Created %d simulated %dm binary markets", len(df), interval_min)
    log.info("  Outcome distribution: UP=%d (%.1f%%) DOWN=%d (%.1f%%)",
             df["outcome"].sum(), 100 * df["outcome"].mean(),
             len(df) - df["outcome"].sum(), 100 * (1 - df["outcome"].mean()))
    return df


# ============================================================================
# 3. Feature Engineering (from the research brief)
# ============================================================================


def build_features(klines: pd.DataFrame, markets: pd.DataFrame,
                   lookback_minutes: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix for each market round using preceding klines.

    Features (aligned with research brief Section 6.2 + Section 4.3):
      - Log returns at multiple horizons (1m, 3m, 5m, 10m, 15m, 30m)
      - Realized volatility (5m, 10m, 30m windows)
      - Volume ratios and taker buy pressure
      - VWAP deviation
      - RSI(14)
      - Candle body/wick features
      - Order book imbalance proxy (taker buy ratio)
      - Mean reversion signal
      - Consecutive up/down count
      - Volume trend slope
    """
    X_list, y_list, ts_list = [], [], []

    for _, row in markets.iterrows():
        round_ts = int(row["round_ts"])

        # Get klines up to (but not including) this round
        mask = klines["ts"] < round_ts
        available = klines[mask]

        if len(available) < lookback_minutes:
            continue

        window = available.iloc[-lookback_minutes:]
        closes = window["close"].values
        volumes = window["volume"].values
        highs = window["high"].values
        lows = window["low"].values
        opens = window["open"].values
        taker_buy = window["taker_buy_base"].values
        trades_arr = window["trades"].values

        log_returns = np.diff(np.log(closes))
        if len(log_returns) < 1:
            continue

        # -- Returns at multiple horizons --
        lr_1m = log_returns[-1]
        lr_3m = np.log(closes[-1] / closes[-4]) if len(closes) >= 4 else 0.0
        lr_5m = np.log(closes[-1] / closes[-6]) if len(closes) >= 6 else 0.0
        lr_10m = np.log(closes[-1] / closes[-11]) if len(closes) >= 11 else 0.0
        lr_15m = np.log(closes[-1] / closes[-16]) if len(closes) >= 16 else 0.0
        lr_30m = np.log(closes[-1] / closes[0])

        # -- Realized volatility --
        vol_5m = np.std(log_returns[-5:]) if len(log_returns) >= 5 else 0.0
        vol_10m = np.std(log_returns[-10:]) if len(log_returns) >= 10 else 0.0
        vol_30m = np.std(log_returns)

        # -- Volume features --
        vol_5 = volumes[-5:].sum()
        vol_15 = volumes[-15:].sum()
        vol_30 = volumes.sum()
        volume_ratio_5_15 = vol_5 / vol_15 if vol_15 > 0 else 1.0
        volume_ratio_5_30 = vol_5 / vol_30 if vol_30 > 0 else 1.0

        # -- Taker buy ratio (order flow imbalance proxy) --
        tbr_5 = taker_buy[-5:].sum() / volumes[-5:].sum() if volumes[-5:].sum() > 0 else 0.5
        tbr_15 = taker_buy[-15:].sum() / volumes[-15:].sum() if volumes[-15:].sum() > 0 else 0.5
        tbr_30 = taker_buy.sum() / volumes.sum() if volumes.sum() > 0 else 0.5

        # -- Price range features --
        hl_range_5 = (highs[-5:].max() - lows[-5:].min()) / closes[-1]
        hl_range_15 = (highs[-15:].max() - lows[-15:].min()) / closes[-1]
        hl_range_30 = (highs.max() - lows.min()) / closes[-1]

        # -- VWAP deviation --
        vwap_5 = (closes[-5:] * volumes[-5:]).sum() / volumes[-5:].sum() if volumes[-5:].sum() > 0 else closes[-1]
        vwap_15 = (closes[-15:] * volumes[-15:]).sum() / volumes[-15:].sum() if volumes[-15:].sum() > 0 else closes[-1]
        close_vs_vwap_5 = (closes[-1] - vwap_5) / closes[-1]
        close_vs_vwap_15 = (closes[-1] - vwap_15) / closes[-1]

        # -- RSI(14) --
        if len(log_returns) >= 14:
            gains = np.maximum(log_returns[-14:], 0)
            losses = np.maximum(-log_returns[-14:], 0)
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0
        else:
            rsi = 50.0

        # -- Consecutive up/down candles --
        recent_dirs = np.sign(log_returns[-10:])
        consec_up = 0
        for d in reversed(recent_dirs):
            if d > 0:
                consec_up += 1
            else:
                break
        consec_down = 0
        for d in reversed(recent_dirs):
            if d < 0:
                consec_down += 1
            else:
                break

        # -- Candle body features (last candle) --
        last_body = (closes[-1] - opens[-1]) / closes[-1]
        last_range = (highs[-1] - lows[-1]) / closes[-1]
        body_ratio = abs(last_body) / last_range if last_range > 0 else 0.0

        # -- Wick ratios --
        if last_range > 0:
            upper_wick = (highs[-1] - max(closes[-1], opens[-1])) / (highs[-1] - lows[-1])
            lower_wick = (min(closes[-1], opens[-1]) - lows[-1]) / (highs[-1] - lows[-1])
        else:
            upper_wick = 0.0
            lower_wick = 0.0

        # -- Volume trend (slope of last 10 bars) --
        if len(volumes) >= 10:
            x = np.arange(10, dtype=float)
            v = volumes[-10:]
            slope = np.polyfit(x, v, 1)[0]
            vol_trend = slope / v.mean() if v.mean() > 0 else 0.0
        else:
            vol_trend = 0.0

        # -- Trade intensity --
        trade_count_5 = float(trades_arr[-5:].sum())
        trade_intensity = trade_count_5 / float(trades_arr[-15:].sum()) if trades_arr[-15:].sum() > 0 else 1.0

        # -- Mean reversion signal --
        mean_30m = closes.mean()
        dist_from_mean = (closes[-1] - mean_30m) / closes[-1]

        # -- Hour of day (cyclical encoding) --
        hour = (round_ts % 86400) / 3600.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        features = np.array([
            lr_1m, lr_3m, lr_5m, lr_10m, lr_15m, lr_30m,           # 0-5
            vol_5m, vol_10m, vol_30m,                                # 6-8
            volume_ratio_5_15, volume_ratio_5_30,                    # 9-10
            tbr_5, tbr_15, tbr_30,                                   # 11-13
            hl_range_5, hl_range_15, hl_range_30,                    # 14-16
            close_vs_vwap_5, close_vs_vwap_15,                      # 17-18
            rsi,                                                      # 19
            float(consec_up), float(consec_down),                    # 20-21
            last_body, body_ratio,                                   # 22-23
            upper_wick, lower_wick,                                  # 24-25
            vol_trend, trade_intensity, dist_from_mean,              # 26-28
            hour_sin, hour_cos,                                      # 29-30
        ], dtype=np.float64)

        X_list.append(features)
        y_list.append(int(row["outcome"]))
        ts_list.append(round_ts)

    return np.array(X_list), np.array(y_list), np.array(ts_list)


FEATURE_NAMES = [
    "lr_1m", "lr_3m", "lr_5m", "lr_10m", "lr_15m", "lr_30m",
    "vol_5m", "vol_10m", "vol_30m",
    "vol_ratio_5_15", "vol_ratio_5_30",
    "tbr_5", "tbr_15", "tbr_30",
    "hl_range_5", "hl_range_15", "hl_range_30",
    "vwap_dev_5", "vwap_dev_15",
    "rsi_14",
    "consec_up", "consec_down",
    "last_body", "body_ratio",
    "upper_wick", "lower_wick",
    "vol_trend", "trade_intensity", "dist_from_mean",
    "hour_sin", "hour_cos",
]


# ============================================================================
# 4. Lead-Lag Signal (from research brief Section 3.4 / state.py model)
# ============================================================================


def lead_lag_p_theo(klines: pd.DataFrame, round_ts: int,
                    interval_min: int = 5) -> float:
    """
    Compute theoretical probability using the drift + vol model
    from the HFT bot (state.py):

        drift_z = (return_30s + ofi_weight * ofi_norm) / (vol * sqrt(T))
        p_theo  = Φ(drift_z)

    Since we don't have tick data, we approximate:
      - return_30s → 30-second return from last kline's close
      - vol → 1-minute realized vol from last 60 candles
      - OFI → taker buy ratio as proxy
      - T → interval_min
    """
    mask = klines["ts"] < round_ts
    available = klines[mask]

    if len(available) < 60:
        return 0.5

    window = available.iloc[-60:]
    closes = window["close"].values
    taker_buy = window["taker_buy_base"].values
    volumes = window["volume"].values

    # 30-second return (approximate using last candle's close-to-close)
    if len(closes) >= 2:
        ret_30s = np.log(closes[-1] / closes[-2])
    else:
        ret_30s = 0.0

    # Realized vol (annualized from 1m returns)
    log_rets = np.diff(np.log(closes))
    if len(log_rets) >= 2:
        vol_1m = np.std(log_rets)
        vol_ann = vol_1m * np.sqrt(525960)  # minutes per year
    else:
        vol_ann = 0.001

    # OFI proxy: net taker buy direction
    total_vol = volumes[-5:].sum()
    ofi = (taker_buy[-5:].sum() / total_vol - 0.5) * 2.0 if total_vol > 0 else 0.0
    ofi_norm = max(-1.0, min(1.0, ofi))

    ofi_weight = 0.3
    drift = ret_30s + ofi_weight * ofi_norm * vol_1m * 0.01
    drift_z = drift / (max(vol_1m, 1e-8) * np.sqrt(max(interval_min, 0.01)))

    # CDF of standard normal
    p_theo = 0.5 * (1.0 + math.erf(drift_z / np.sqrt(2.0)))
    return max(0.01, min(0.99, p_theo))


# ============================================================================
# 5. Ensemble Model with Double Calibration (Bawa v2 architecture)
# ============================================================================


def build_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Build the 5-model ensemble with double calibration per Bawa v2.

    Pass 1: Platt scaling (sigmoid)
    Pass 2: Isotonic regression on top of Pass 1 output

    Returns dict of {name: fitted_pipeline}
    """
    models = {}

    base_models = {
        "HistGBM": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        ),
    }

    if HAS_XGB:
        base_models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )

    if HAS_LGB:
        base_models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1,
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    for name, model in base_models.items():
        log.info("  Training %s...", name)
        try:
            # Pass 1: Platt scaling
            cal1 = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            cal1.fit(X_scaled, y_train)

            # Pass 2: Isotonic on top of sigmoid output
            cal2 = CalibratedClassifierCV(cal1, method="isotonic", cv=2)
            cal2.fit(X_scaled, y_train)

            models[name] = cal2
        except Exception as e:
            log.warning("  Failed to train %s: %s — skipping", name, e)

    return models, scaler


def ensemble_predict(models: dict, scaler: StandardScaler,
                     X: np.ndarray) -> np.ndarray:
    """
    Performance-weighted ensemble prediction.

    Each model outputs calibrated P(UP), combined via weighted average
    where weights are proportional to each model's training Brier score.
    """
    X_scaled = scaler.transform(X)
    probas = []

    for name, model in models.items():
        try:
            p = model.predict_proba(X_scaled)[:, 1]
            probas.append(p)
        except Exception:
            pass

    if not probas:
        return np.full(len(X), 0.5)

    # Equal weights for simplicity (Brier-based weights require validation set)
    return np.mean(probas, axis=0)


# ============================================================================
# 6. Kelly Criterion for Bounded Binary Contracts
# ============================================================================


def kelly_fraction(p_model: float, market_price: float,
                   alpha: float = 0.25) -> float:
    """
    Kelly fraction for bounded binary contract.

    f* = (P - q) / (1 - q)
    f_practical = alpha * f*

    From research brief Section 3.3 (arXiv:2412.14144v1).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0

    edge = p_model - market_price
    if edge <= 0:
        return 0.0

    f_star = edge / (1 - market_price)
    return alpha * f_star


# ============================================================================
# 7. Brier Score & Calibration Metrics
# ============================================================================


def brier_score(probas: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    return np.mean((probas - outcomes) ** 2)


def brier_decomposition(probas: np.ndarray, outcomes: np.ndarray,
                        n_bins: int = 10) -> dict:
    """
    Brier score decomposition into Reliability, Resolution, Uncertainty.
    From research brief Section 3.1.
    """
    N = len(probas)
    base_rate = outcomes.mean()
    uncertainty = base_rate * (1 - base_rate)

    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    calibration_data = []

    for k in range(n_bins):
        mask = (probas >= bins[k]) & (probas < bins[k + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue

        p_k = probas[mask].mean()
        o_k = outcomes[mask].mean()

        reliability += n_k * (p_k - o_k) ** 2
        resolution += n_k * (o_k - base_rate) ** 2
        calibration_data.append({
            "bin": f"[{bins[k]:.1f},{bins[k+1]:.1f})",
            "n": int(n_k),
            "mean_pred": round(p_k, 4),
            "mean_obs": round(o_k, 4),
            "gap": round(p_k - o_k, 4),
        })

    reliability /= N
    resolution /= N

    return {
        "brier_score": brier_score(probas, outcomes),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "calibration_bins": calibration_data,
    }


def kl_divergence(p_model: np.ndarray, p_market: np.ndarray) -> float:
    """
    D_KL(P_model || P_market) — information-theoretic edge measure.
    From research brief Section 3.5.
    """
    eps = 1e-10
    p = np.clip(p_model, eps, 1 - eps)
    q = np.clip(p_market, eps, 1 - eps)
    return np.mean(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


# ============================================================================
# 8. Backtest Engine
# ============================================================================


def simulate_market_price(markets: pd.DataFrame, idx: int,
                          klines: pd.DataFrame) -> float:
    """
    Simulate what a naive market would price YES at.

    Uses trailing 10-bar momentum as a proxy for market consensus.
    A slightly overconfident market with longshot bias (per Becker 2025).
    """
    row = markets.iloc[idx]
    round_ts = int(row["round_ts"])

    mask = klines["ts"] < round_ts
    available = klines[mask]

    if len(available) < 10:
        return 0.50

    closes = available.iloc[-10:]["close"].values
    ret = np.log(closes[-1] / closes[0])

    # Market uses simple momentum → sigmoid → price
    # Add noise + slight favorite-longshot bias
    raw_signal = ret * 100  # scale up
    p_market = 1.0 / (1.0 + np.exp(-raw_signal))

    # Add noise (simulates market microstructure)
    np.random.seed(round_ts % (2**31))
    noise = np.random.normal(0, 0.03)
    p_market += noise

    # Longshot bias: compress toward 0.5 (markets underreact)
    p_market = 0.5 + 0.85 * (p_market - 0.5)

    return max(0.05, min(0.95, p_market))


def run_backtest(
    klines: pd.DataFrame,
    interval_min: int = 5,
    bankroll: float = 10000.0,
    kelly_alpha: float = 0.25,
    edge_threshold: float = 0.03,
    max_bet: float = 500.0,
    train_pct: float = 0.60,
    real_pm_markets: pd.DataFrame = None,
    maker_only: bool = False,
):
    """
    Full backtest implementing the research brief architecture:

    Layer 2: Signal Generation (lead-lag + ensemble ML)
    Layer 3: Risk & Position Sizing (Kelly + drawdown circuit breaker)
    Layer 4: Execution (simulated maker-only)
    Layer 5: Monitoring (Brier, Sharpe, D_KL)
    """
    using_real = real_pm_markets is not None
    log.info("=" * 80)
    log.info("POLYMARKET QUANT STRATEGY BACKTEST")
    log.info("  Data source: %s", "REAL Polymarket 5m markets" if using_real else "Simulated from Binance")
    log.info("  Interval: %dm | Bankroll: $%.0f | Kelly alpha: %.2f",
             interval_min, bankroll, kelly_alpha)
    log.info("  Edge threshold: %.0f%% | Max bet: $%.0f",
             edge_threshold * 100, max_bet)
    fee_mode = "MAKER (rebate 0.5%)" if maker_only else "TAKER (fee 2%)"
    log.info("  Fee mode: %s", fee_mode)
    log.info("=" * 80)

    # -- Get markets --
    if using_real:
        markets = real_pm_markets
        log.info("Using %d real Polymarket markets", len(markets))
    else:
        markets = create_binary_markets(klines, interval_min)

    # -- Build features --
    log.info("Building features...")
    X, y, timestamps = build_features(klines, markets, lookback_minutes=30)
    log.info("Feature matrix: %d rounds x %d features", X.shape[0], X.shape[1])

    if len(X) < 200:
        log.error("Not enough data (%d rounds). Need at least 200. Try --days 7 or more.", len(X))
        sys.exit(1)

    # -- Train/test split (chronological) --
    split = int(len(X) * train_pct)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    ts_train = timestamps[:split]
    ts_test = timestamps[split:]

    log.info("Train: %d rounds | Test: %d rounds", len(X_train), len(X_test))
    log.info("Train class balance: UP=%.1f%%", 100 * y_train.mean())
    log.info("Test class balance:  UP=%.1f%%", 100 * y_test.mean())

    # -- Train ensemble --
    log.info("\nTraining ensemble with double calibration...")
    models, scaler = build_ensemble(X_train, y_train)
    log.info("Trained %d models: %s", len(models), list(models.keys()))

    # -- Generate predictions --
    log.info("\nGenerating predictions on test set...")
    ensemble_probas = ensemble_predict(models, scaler, X_test)

    # -- Also compute lead-lag p_theo for comparison --
    ll_probas = np.array([
        lead_lag_p_theo(klines, int(ts), interval_min)
        for ts in ts_test
    ])

    # -- Combined signal: 60% ensemble + 40% lead-lag --
    combined_probas = 0.6 * ensemble_probas + 0.4 * ll_probas

    # -- Market prices --
    if using_real:
        # For 5m BTC Up/Down markets, the opening market price is always ~0.50
        # because the market is essentially a fair coin flip at open.
        # The REAL edge comes from the lead-lag: CEX moves first, then the
        # Polymarket price adjusts. We model the market price at the time
        # our signal fires (30-60 seconds after market open) as 0.50 plus
        # a small adjustment based on early CEX movement.
        test_markets = markets.iloc[split:split + len(X_test)].reset_index(drop=True)

        market_prices = np.zeros(len(X_test))
        for i in range(len(X_test)):
            if i >= len(test_markets):
                market_prices[i] = 0.50
                continue
            rts = int(test_markets.iloc[i]["round_ts"])
            # The market opens at ~0.50. After 30-60s of trading,
            # it adjusts based on early CEX price action.
            # Model: market_price = 0.50 + 0.3 * (early CEX signal)
            mask = (klines["ts"] >= rts) & (klines["ts"] < rts + 1)
            early = klines[mask]
            if len(early) > 0:
                # First 1-minute candle's movement as proxy
                first_ret = (early.iloc[0]["close"] - early.iloc[0]["open"]) / early.iloc[0]["open"]
                # Market adjusts ~30% of the way toward the CEX signal
                market_adjustment = 0.30 * np.clip(first_ret * 100, -0.4, 0.4)
                market_prices[i] = 0.50 + market_adjustment
            else:
                market_prices[i] = 0.50
        market_prices = np.clip(market_prices, 0.10, 0.90)
        log.info("Market prices at entry (modeled from opening + early CEX signal)")
        log.info("  Market price stats: mean=%.3f std=%.3f min=%.3f max=%.3f",
                 market_prices.mean(), market_prices.std(),
                 market_prices.min(), market_prices.max())
    else:
        test_markets = markets.iloc[split:split + len(X_test)].reset_index(drop=True)
        market_prices = np.array([
            simulate_market_price(markets, split + i, klines)
            for i in range(len(X_test))
        ])

    # ================================================================
    # TRADING SIMULATION
    # ================================================================

    strategies = {
        "Ensemble Only": ensemble_probas,
        "Lead-Lag Only": ll_probas,
        "Combined (60/40)": combined_probas,
    }

    results = {}

    for strat_name, probas in strategies.items():
        log.info("\n" + "=" * 70)
        log.info("STRATEGY: %s", strat_name)
        log.info("=" * 70)

        current_bankroll = bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0
        trades = []
        daily_pnl = {}

        for i in range(len(y_test)):
            p_model = probas[i]
            q = market_prices[i]
            outcome = y_test[i]
            ts = ts_test[i]

            # Determine trade direction
            edge_up = p_model - q
            edge_down = (1 - p_model) - (1 - q)

            trade_side = None
            edge = 0
            buy_price = 0

            if edge_up > edge_threshold:
                trade_side = "YES"
                edge = edge_up
                buy_price = q
            elif edge_down > edge_threshold:
                trade_side = "NO"
                edge = edge_down
                buy_price = 1 - q

            if trade_side is None:
                continue

            # Kelly sizing
            if trade_side == "YES":
                f = kelly_fraction(p_model, q, kelly_alpha)
            else:
                f = kelly_fraction(1 - p_model, 1 - q, kelly_alpha)

            bet_size = min(f * current_bankroll, max_bet)
            if bet_size < 1.0:
                continue

            # Drawdown circuit breaker (from research brief Section 9)
            dd_pct = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd_pct > 0.15:  # 15% max drawdown
                continue

            # Execute trade
            n_shares = bet_size / buy_price
            if trade_side == "YES":
                won = outcome == 1
            else:
                won = outcome == 0

            if won:
                pnl = n_shares * (1.0 - buy_price)  # pay buy_price, receive $1
            else:
                pnl = -n_shares * buy_price  # lose the cost

            # Apply fee/rebate
            if maker_only:
                # Maker rebate: you RECEIVE 0.5% on notional
                fee = -0.005 * bet_size  # negative = income
            else:
                # Taker fee: pay 2%
                fee = 0.02 * bet_size
            pnl -= fee

            current_bankroll += pnl
            peak_bankroll = max(peak_bankroll, current_bankroll)
            max_drawdown = max(max_drawdown, (peak_bankroll - current_bankroll) / peak_bankroll)

            # Track daily PnL
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0) + pnl

            trades.append({
                "ts": ts,
                "side": trade_side,
                "p_model": round(p_model, 4),
                "q_market": round(q, 4),
                "edge": round(edge, 4),
                "kelly_f": round(f, 4),
                "bet_size": round(bet_size, 2),
                "buy_price": round(buy_price, 4),
                "won": won,
                "pnl": round(pnl, 2),
                "bankroll": round(current_bankroll, 2),
            })

        # -- Compute metrics --
        n_trades = len(trades)
        if n_trades == 0:
            log.info("  No trades taken.")
            results[strat_name] = {"n_trades": 0}
            continue

        wins = sum(1 for t in trades if t["won"])
        total_pnl = current_bankroll - bankroll
        win_rate = wins / n_trades
        pnl_series = [t["pnl"] for t in trades]
        avg_pnl = np.mean(pnl_series)
        std_pnl = np.std(pnl_series) if len(pnl_series) > 1 else 1.0

        # Sharpe (annualized, assuming 288 rounds/day for 5m intervals)
        rounds_per_day = 1440 / interval_min
        sharpe = (avg_pnl / std_pnl) * np.sqrt(rounds_per_day * 365) if std_pnl > 0 else 0

        # Profit factor
        gross_profit = sum(p for p in pnl_series if p > 0)
        gross_loss = abs(sum(p for p in pnl_series if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Brier score
        bs = brier_score(probas, y_test)
        bs_decomp = brier_decomposition(probas, y_test)
        d_kl = kl_divergence(probas, market_prices)

        # Average win/loss
        win_pnls = [t["pnl"] for t in trades if t["won"]]
        loss_pnls = [t["pnl"] for t in trades if not t["won"]]
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0

        log.info("\n  --- Performance ---")
        log.info("  Trades:           %d", n_trades)
        log.info("  Win Rate:         %.1f%% (%d/%d)", win_rate * 100, wins, n_trades)
        log.info("  Total PnL:        $%.2f (%.1f%%)", total_pnl, total_pnl / bankroll * 100)
        log.info("  Final Bankroll:   $%.2f", current_bankroll)
        log.info("  Avg PnL/Trade:    $%.2f", avg_pnl)
        log.info("  Avg Win:          $%.2f", avg_win)
        log.info("  Avg Loss:         $%.2f", avg_loss)
        log.info("  Max Drawdown:     %.1f%%", max_drawdown * 100)
        log.info("  Profit Factor:    %.2f", profit_factor)
        log.info("  Sharpe Ratio:     %.2f", sharpe)

        log.info("\n  --- Model Quality ---")
        log.info("  Brier Score:      %.4f", bs)
        log.info("  Reliability:      %.4f (lower = better calibrated)", bs_decomp["reliability"])
        log.info("  Resolution:       %.4f (higher = more informative)", bs_decomp["resolution"])
        log.info("  Uncertainty:      %.4f", bs_decomp["uncertainty"])
        log.info("  D_KL(model||mkt): %.6f nats", d_kl)

        log.info("\n  --- Calibration Diagram ---")
        log.info("  %-15s %6s %8s %8s %8s", "Bin", "N", "PredAvg", "ObsFreq", "Gap")
        for b in bs_decomp["calibration_bins"]:
            log.info("  %-15s %6d %8.4f %8.4f %+8.4f",
                     b["bin"], b["n"], b["mean_pred"], b["mean_obs"], b["gap"])

        log.info("\n  --- Daily PnL ---")
        for day, dpnl in sorted(daily_pnl.items()):
            log.info("  %s: $%+.2f", day, dpnl)

        # -- Kelly statistics --
        kelly_fracs = [t["kelly_f"] for t in trades]
        log.info("\n  --- Kelly Stats ---")
        log.info("  Avg Kelly f:      %.4f", np.mean(kelly_fracs))
        log.info("  Max Kelly f:      %.4f", np.max(kelly_fracs))
        log.info("  Avg Bet Size:     $%.2f", np.mean([t["bet_size"] for t in trades]))
        log.info("  Max Bet Size:     $%.2f", np.max([t["bet_size"] for t in trades]))

        # -- Edge distribution --
        edges = [t["edge"] for t in trades]
        log.info("\n  --- Edge Distribution ---")
        log.info("  Mean Edge:        %.2f%%", np.mean(edges) * 100)
        log.info("  Median Edge:      %.2f%%", np.median(edges) * 100)
        log.info("  Max Edge:         %.2f%%", np.max(edges) * 100)

        results[strat_name] = {
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "final_bankroll": round(current_bankroll, 2),
            "max_drawdown": round(max_drawdown, 4),
            "profit_factor": round(profit_factor, 4),
            "sharpe": round(sharpe, 2),
            "brier_score": round(bs, 4),
            "d_kl": round(d_kl, 6),
            "avg_edge": round(np.mean(edges), 4),
            "trades": trades,
        }

    # ================================================================
    # BASELINE COMPARISON
    # ================================================================

    log.info("\n" + "=" * 70)
    log.info("BASELINES")
    log.info("=" * 70)

    # Fee per $100 trade
    fee_per_100 = -0.005 * 100 if maker_only else 0.02 * 100

    # Always buy YES at market price
    always_yes_pnl = 0
    for i in range(len(y_test)):
        q = market_prices[i]
        shares = 100 / q
        if y_test[i] == 1:
            always_yes_pnl += shares * (1.0 - q) - fee_per_100
        else:
            always_yes_pnl -= shares * q + fee_per_100

    log.info("  Always YES:    PnL=$%.2f (win rate=%.1f%%)",
             always_yes_pnl, y_test.mean() * 100)

    # Random
    rng = np.random.default_rng(42)
    random_pnl = 0
    for i in range(len(y_test)):
        q = market_prices[i]
        if rng.random() > 0.5:
            shares = 100 / q
            if y_test[i] == 1:
                random_pnl += shares * (1.0 - q) - fee_per_100
            else:
                random_pnl -= shares * q + fee_per_100
        else:
            shares = 100 / (1 - q)
            if y_test[i] == 0:
                random_pnl += shares * q - fee_per_100
            else:
                random_pnl -= shares * (1 - q) + fee_per_100

    log.info("  Random:        PnL=$%.2f", random_pnl)

    # Market Brier score (how well-calibrated is the simulated market?)
    market_bs = brier_score(market_prices, y_test)
    log.info("  Market Brier:  %.4f", market_bs)
    for name, probas in strategies.items():
        log.info("  %s Brier: %.4f", name, brier_score(probas, y_test))

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    output_path = Path("/Users/eiji/polymarket-pnl/backtest_results_quant.json")
    save_data = {
        "config": {
            "interval_min": interval_min,
            "bankroll": bankroll,
            "kelly_alpha": kelly_alpha,
            "edge_threshold": edge_threshold,
            "max_bet": max_bet,
            "train_pct": train_pct,
            "n_total_rounds": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "results": {
            name: {k: v for k, v in r.items() if k != "trades"}
            for name, r in results.items()
        },
        "baselines": {
            "always_yes_pnl": round(always_yes_pnl, 2),
            "random_pnl": round(random_pnl, 2),
            "market_brier": round(market_bs, 4),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    log.info("\nResults saved to %s", output_path)

    log.info("\n" + "=" * 80)
    log.info("BACKTEST COMPLETE")
    log.info("=" * 80)

    return results


# ============================================================================
# Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Polymarket Quant Trading strategies on BTC data"
    )
    parser.add_argument("--days", type=int, default=7,
                        help="Days of Binance kline data (default: 7)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Market interval in minutes (default: 5)")
    parser.add_argument("--bankroll", type=float, default=10000.0,
                        help="Starting bankroll in USD (default: 10000)")
    parser.add_argument("--kelly-alpha", type=float, default=0.25,
                        help="Kelly fraction scaling (default: 0.25 = quarter Kelly)")
    parser.add_argument("--edge-threshold", type=float, default=0.03,
                        help="Minimum edge to trade (default: 0.03 = 3%%)")
    parser.add_argument("--max-bet", type=float, default=500.0,
                        help="Maximum bet size in USD (default: 500)")
    parser.add_argument("--use-real-pm", action="store_true",
                        help="Use real Polymarket 5m market data instead of simulated")
    parser.add_argument("--pm-rounds", type=int, default=2000,
                        help="Number of Polymarket rounds to fetch (default: 2000)")
    parser.add_argument("--maker-only", action="store_true",
                        help="Use maker-only fees (0.5%% rebate) instead of taker (2%% fee)")
    parser.add_argument("--cache", action="store_true",
                        help="Cache fetched data locally for fast re-runs")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only fetch and cache data, don't run backtest")
    args = parser.parse_args()

    if args.use_real_pm:
        log.info("MODE: Real Polymarket 5m BTC markets")
        log.info("  Fee mode: %s", "MAKER (rebate 0.5%%)" if args.maker_only else "TAKER (fee 2%%)")

        # --- Polymarket data: load from cache or fetch ---
        pm_cache_name = cache_key_pm(args.pm_rounds)
        pm_markets = load_cache(pm_cache_name) if args.cache else None

        if pm_markets is None:
            pm_markets = fetch_polymarket_5m_markets(n_rounds=args.pm_rounds)
            if pm_markets.empty:
                log.error("No Polymarket data found. Exiting.")
                sys.exit(1)
            if args.cache:
                save_cache(pm_markets, pm_cache_name)
        else:
            log.info("Using cached Polymarket data (%d markets)", len(pm_markets))

        # --- Binance klines: load from cache or fetch ---
        ts_min = int(pm_markets["round_ts"].min())
        ts_max = int(pm_markets["round_ts"].max())
        days_needed = max(1, (ts_max - ts_min) // 86400 + 2)

        binance_cache_name = cache_key_binance(days_needed)
        klines = load_cache(binance_cache_name) if args.cache else None

        if klines is None:
            log.info("Fetching %d days of Binance klines to cover Polymarket range...", days_needed)
            klines = fetch_binance_klines(days=days_needed)
            if args.cache:
                save_cache(klines, binance_cache_name)
        else:
            log.info("Using cached Binance klines (%d candles, %.1f days)",
                     len(klines), len(klines) / 1440)

        if args.cache_only:
            log.info("Cache-only mode: data saved. Exiting.")
            sys.exit(0)

        # Enrich pm_markets with Binance price data for feature engineering
        pm_markets["open_price"] = 0.0
        pm_markets["close_price"] = 0.0
        pm_markets["high"] = 0.0
        pm_markets["low"] = 0.0
        pm_markets["n_trades"] = 0
        pm_markets["taker_buy"] = 0.0

        for idx, row in pm_markets.iterrows():
            rts = int(row["round_ts"])
            mask = (klines["ts"] >= rts) & (klines["ts"] < rts + 300)
            interval_klines = klines[mask]
            if len(interval_klines) >= 1:
                pm_markets.at[idx, "open_price"] = interval_klines.iloc[0]["open"]
                pm_markets.at[idx, "close_price"] = interval_klines.iloc[-1]["close"]
                pm_markets.at[idx, "high"] = interval_klines["high"].max()
                pm_markets.at[idx, "low"] = interval_klines["low"].min()
                pm_markets.at[idx, "n_trades"] = interval_klines["trades"].sum()
                pm_markets.at[idx, "taker_buy"] = interval_klines["taker_buy_base"].sum()
            else:
                pm_markets.at[idx, "open_price"] = 0
                pm_markets.at[idx, "close_price"] = 0

        # Filter out rows with missing price data
        pm_markets = pm_markets[pm_markets["open_price"] > 0].reset_index(drop=True)
        log.info("Markets with Binance price data: %d", len(pm_markets))

        run_backtest(
            klines,
            interval_min=args.interval,
            bankroll=args.bankroll,
            kelly_alpha=args.kelly_alpha,
            edge_threshold=args.edge_threshold,
            max_bet=args.max_bet,
            real_pm_markets=pm_markets,
            maker_only=args.maker_only,
        )
    else:
        log.info("MODE: Simulated markets from Binance data")
        klines = fetch_binance_klines(days=args.days)
        run_backtest(
            klines,
            interval_min=args.interval,
            bankroll=args.bankroll,
            kelly_alpha=args.kelly_alpha,
            edge_threshold=args.edge_threshold,
            max_bet=args.max_bet,
            maker_only=args.maker_only,
        )


if __name__ == "__main__":
    main()
