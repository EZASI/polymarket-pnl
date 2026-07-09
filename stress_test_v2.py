#!/usr/bin/env python3
"""
STRESS TEST v2: The Truth Machine
===================================

Improvements over v1:
  1. TIMEFRAME SWEEP  — 1m, 5m, 10m, 15m side-by-side comparison
  2. EQUITY CURVE     — Cumulative P&L with flatness detection
  3. VOLATILITY GATE  — Only trade when ATR > threshold ("meat on the bone")
  4. KELLY CRITERION  — Dynamic position sizing based on edge + variance
  5. LIMIT ORDER SIM  — Compare 2% slippage (market) vs 0.5% (limit)
  6. REGIME DIAGNOSIS — Per-fold volatility/trend analysis to explain outliers
  7. MONTE CARLO      — 1,000 shuffles with proper per-trade P&L variance

Usage:
  python3 stress_test_v2.py --days 30
  python3 stress_test_v2.py --days 60 --mc-runs 2000
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

warnings.filterwarnings("ignore")

from ultra_predictor import (
    BinanceDataCollector,
    engineer_all_features,
    merge_futures_data,
    create_targets,
    get_feature_cols,
)

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "BTCUSDT"
HORIZONS = [1, 5, 10, 15]

TRAIN_DAYS = 7
TEST_DAYS = 2
TRAIN_W = TRAIN_DAYS * 1440
TEST_W = TEST_DAYS * 1440

# Cost scenarios
SCENARIOS = {
    "market_2pct": {"slippage": 0.02, "spread": 0.01, "label": "Market Order (2% slip + $0.01)"},
    "limit_05pct": {"slippage": 0.005, "spread": 0.005, "label": "Limit Order  (0.5% slip + $0.005)"},
    "zero_cost":   {"slippage": 0.0,   "spread": 0.0,   "label": "Zero Cost    (theoretical edge)"},
}
BUY_PRICE = 0.50
WIN_PAYOUT = 1.00


def pnl_per_dollar(correct: bool, slippage: float, spread: float) -> float:
    cost = BUY_PRICE * (1 + slippage) + spread
    return (WIN_PAYOUT - cost) if correct else -cost


# ============================================================
# STRATEGY FILTERS (with optional ATR gate)
# ============================================================
def apply_filters(df, y_pred, confidence, atr_gate: float = 0.0):
    """
    Selective filters from ultra_predictor + optional ATR gate.
    atr_gate: minimum ATR ratio (atr / atr_ma) to allow trading.
              0.0 = no filter, 1.2 = only trade when vol is 20% above average.
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
    ti = col("taker_imbalance")
    ar = col("atr_ratio")

    # Strategy A: Volume Spike + Momentum
    a = (confidence >= 0.55) & (vs == 1) & (
        ((y_pred == 1) & (m3 > 0.1)) | ((y_pred == 0) & (m3 < -0.1)))

    # Strategy B: Trend + RSI
    b = (confidence >= 0.60) & (
        ((y_pred == 1) & (ts >= 2) & (rsi > 45) & (rsi < 70)) |
        ((y_pred == 0) & (ts <= 1) & (rsi > 30) & (rsi < 55)))

    # Strategy C: ULTRA — Taker + Trend + Volume
    c = (confidence >= 0.60) & (vr > 1.3) & (
        ((y_pred == 1) & (ti > 0.01) & (ts >= 2) & (m5 > 0)) |
        ((y_pred == 0) & (ti < -0.01) & (ts <= 1) & (m5 < 0)))

    mask = a | b | c

    # ATR gate: only trade when volatility is above threshold
    if atr_gate > 0:
        mask = mask & (ar >= atr_gate)

    return mask


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================
def walk_forward(df, horizon, feat_cols, atr_gate=0.0, verbose=False):
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

    target = f"target_{horizon}"
    clean = df.dropna(subset=feat_cols + [target]).reset_index(drop=True)
    n = len(clean)

    if n < TRAIN_W + TEST_W:
        return None

    trades = []
    fold = 0
    start = 0

    while start + TRAIN_W + TEST_W <= n:
        te = start + TRAIN_W
        oe = min(te + TEST_W, n)

        tr = clean.iloc[start:te]
        ts_df = clean.iloc[te:oe].reset_index(drop=True)

        Xtr = tr[feat_cols].values
        ytr = tr[target].values
        Xts = ts_df[feat_cols].values
        yts = ts_df[target].values

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xts_s = sc.transform(Xts)

        probas = []
        if USE_LGB:
            m = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=30, random_state=42, verbose=-1, n_jobs=-1)
            m.fit(Xtr_s, ytr)
            probas.append(m.predict_proba(Xts_s))

        if USE_XGB:
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1)
            m.fit(Xtr_s, ytr)
            probas.append(m.predict_proba(Xts_s))

        if not probas:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.06,
                subsample=0.8, random_state=42)
            m.fit(Xtr_s, ytr)
            probas.append(m.predict_proba(Xts_s))

        avg = np.mean(probas, axis=0)
        yp = (avg[:, 1] > 0.5).astype(int)
        conf = np.max(avg, axis=1)

        mask = apply_filters(ts_df, yp, conf, atr_gate=atr_gate)
        idxs = np.where(mask)[0]

        # Fold-level regime stats
        fold_atr = float(ts_df["atr_pct"].mean()) if "atr_pct" in ts_df.columns else 0
        fold_trend = float(ts_df["trend_score"].mean()) if "trend_score" in ts_df.columns else 0
        fold_vol = float(ts_df["vol_10"].mean()) if "vol_10" in ts_df.columns else 0
        price_change = 0
        if "close" in ts_df.columns and len(ts_df) > 1:
            price_change = (ts_df["close"].iloc[-1] - ts_df["close"].iloc[0]) / ts_df["close"].iloc[0] * 100

        for idx in idxs:
            trades.append({
                "fold": fold,
                "pred": int(yp[idx]),
                "actual": int(yts[idx]),
                "correct": yp[idx] == yts[idx],
                "confidence": float(conf[idx]),
                "fold_atr": fold_atr,
                "fold_trend": fold_trend,
                "fold_vol": fold_vol,
                "fold_price_chg": price_change,
            })

        if verbose:
            wr = accuracy_score(yts[idxs], yp[idxs]) if len(idxs) > 0 else 0
            print(
                f"  Fold {fold:>2d} | {len(idxs):>4d} trades | WR {wr:.1%} | "
                f"ATR {fold_atr:.3f}% | Trend {fold_trend:.1f} | "
                f"BTC {price_change:+.1f}%")

        fold += 1
        start += TEST_W

    return {"trades": trades, "folds": fold}


# ============================================================
# RISK METRICS
# ============================================================
def sharpe(returns, periods=252):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods))


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def kelly_fraction(win_rate, win_size, loss_size):
    """Kelly Criterion: optimal fraction of bankroll to risk."""
    if loss_size == 0:
        return 0
    b = win_size / loss_size  # odds
    p = win_rate
    q = 1 - p
    f = (b * p - q) / b
    return max(0, min(f, 0.25))  # cap at 25%


# ============================================================
# EQUITY CURVE ANALYSIS
# ============================================================
def analyze_equity_curve(pnls):
    """Detect if equity curve is 'streaky' vs 'steady'."""
    if len(pnls) < 20:
        return {"steady": False, "reason": "too few trades"}

    equity = np.cumsum(pnls)
    n = len(equity)

    # Split into quartiles and check if profit is concentrated
    q_size = n // 4
    q_pnls = [pnls[i * q_size:(i + 1) * q_size].sum() for i in range(4)]
    total = sum(q_pnls)

    # If >70% of profit comes from one quartile, it's "streaky"
    if total > 0:
        max_q_frac = max(q_pnls) / total if total > 0 else 0
    else:
        max_q_frac = 0

    # Longest drawdown (consecutive losing stretch in equity terms)
    underwater = equity < np.maximum.accumulate(equity)
    if underwater.any():
        groups = np.diff(np.where(np.concatenate(([False], underwater, [False])))[0])
        longest_dd_bars = int(groups.max()) if len(groups) > 0 else 0
    else:
        longest_dd_bars = 0

    # Linearity: R² of equity curve vs straight line
    x = np.arange(n)
    if n > 1 and np.std(equity) > 0:
        corr = np.corrcoef(x, equity)[0, 1]
        r_squared = corr ** 2
    else:
        r_squared = 0

    steady = r_squared > 0.7 and max_q_frac < 0.5

    return {
        "steady": steady,
        "r_squared": r_squared,
        "max_quartile_fraction": max_q_frac,
        "quartile_pnls": q_pnls,
        "longest_dd_bars": longest_dd_bars,
        "total_pnl": total,
    }


# ============================================================
# MONTE CARLO
# ============================================================
def monte_carlo(pnls, n_runs=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnls)
    if n == 0:
        return None

    cap = abs(pnls).sum() * 0.1
    sharpes = np.zeros(n_runs)
    max_dds = np.zeros(n_runs)
    finals = np.zeros(n_runs)
    ruin = 0

    for i in range(n_runs):
        shuf = rng.permutation(pnls)
        eq = np.cumsum(shuf)

        sharpes[i] = sharpe(shuf, 252)
        eq_c = np.concatenate([[0], eq]) + cap
        max_dds[i] = max_dd(eq_c)
        finals[i] = eq[-1]
        if np.min(eq) < -cap * 0.5:
            ruin += 1

    return {
        "sharpe_5": float(np.percentile(sharpes, 5)),
        "sharpe_50": float(np.median(sharpes)),
        "sharpe_95": float(np.percentile(sharpes, 95)),
        "mdd_5": float(np.percentile(max_dds, 5)),
        "mdd_50": float(np.median(max_dds)),
        "mdd_95": float(np.percentile(max_dds, 95)),
        "mdd_worst": float(np.max(max_dds)),
        "pnl_5": float(np.percentile(finals, 5)),
        "pnl_50": float(np.median(finals)),
        "pnl_95": float(np.percentile(finals, 95)),
        "p_profit": float(np.mean(finals > 0)),
        "p_ruin": ruin / n_runs,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--mc-runs", type=int, default=1000)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    global SYMBOL
    SYMBOL = args.symbol

    print("=" * 72)
    print("  STRESS TEST v2: The Truth Machine")
    print("=" * 72)
    print(f"  Symbol: {SYMBOL}  |  Data: {args.days}d  |  MC: {args.mc_runs:,} runs")
    print(f"  WF: {TRAIN_DAYS}d train / {TEST_DAYS}d OOS  |  Horizons: {HORIZONS}")
    print("=" * 72)

    # ── Fetch data ──
    print(f"\n[1/6] Fetching {args.days} days of data...")
    collector = BinanceDataCollector(SYMBOL)
    df = collector.get_klines_history(days=args.days)
    if df.empty:
        print("ERROR: no data"); sys.exit(1)

    print(f"  {len(df):,} candles | "
          f"${df['close'].min():,.0f} → ${df['close'].max():,.0f}")

    # ── Features ──
    print(f"\n[2/6] Engineering features + futures data...")
    df = engineer_all_features(df)
    df = merge_futures_data(df, collector)
    df = create_targets(df)

    # Fix sparse columns
    for c in df.columns:
        if df[c].isna().mean() > 0.3:
            df[c] = df[c].ffill().bfill().fillna(0)

    feat_cols = get_feature_cols(df)
    for c in list(feat_cols):
        if df[c].isna().mean() > 0.5:
            feat_cols.remove(c)
    for c in feat_cols:
        df[c] = df[c].fillna(df[c].median())

    clean_n = len(df.dropna(subset=feat_cols + ["target_1"]))
    print(f"  Features: {len(feat_cols)} | Clean rows: {clean_n:,}")

    # ── Timeframe Sweep ──
    print(f"\n[3/6] TIMEFRAME SWEEP (no ATR gate)")
    print("─" * 72)

    sweep_results = {}
    for h in HORIZONS:
        print(f"\n  ── {h}-MINUTE HORIZON ──")
        res = walk_forward(df, h, feat_cols, atr_gate=0.0, verbose=True)
        if res and res["trades"]:
            sweep_results[h] = res
        else:
            print(f"    No trades for {h}m")

    # ── ATR-Gated Sweep ──
    print(f"\n[4/6] VOLATILITY-GATED SWEEP (ATR ratio >= 1.2)")
    print("─" * 72)

    gated_results = {}
    for h in HORIZONS:
        print(f"\n  ── {h}-MINUTE HORIZON (ATR gate) ──")
        res = walk_forward(df, h, feat_cols, atr_gate=1.2, verbose=True)
        if res and res["trades"]:
            gated_results[h] = res
        else:
            print(f"    No trades for {h}m (gated)")

    # ── Analysis ──
    print(f"\n[5/6] ANALYSIS")
    print("=" * 72)

    def analyze(trades, scenario_key, label_prefix=""):
        sc = SCENARIOS[scenario_key]
        slp, spd = sc["slippage"], sc["spread"]
        pnl_w = pnl_per_dollar(True, slp, spd)
        pnl_l = pnl_per_dollar(False, slp, spd)

        correct = np.array([t["correct"] for t in trades])
        pnls = np.where(correct, pnl_w, pnl_l) * 100  # $100 per trade
        wr = correct.mean()

        n_folds = len(set(t["fold"] for t in trades))
        test_days = n_folds * TEST_DAYS
        tpd = len(trades) / test_days if test_days else 0

        eq = np.cumsum(pnls)
        eq_c = np.concatenate([[0], eq]) + abs(pnls).sum() * 0.1

        sr = sharpe(pnls, 252)
        mdd = max_dd(eq_c)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        pf = wins.sum() / abs(losses).sum() if abs(losses).sum() > 0 else float("inf")

        # Kelly
        kf = kelly_fraction(wr, abs(pnl_w), abs(pnl_l))

        # Equity curve analysis
        ec = analyze_equity_curve(pnls)

        return {
            "trades": len(trades), "wr": wr, "tpd": tpd,
            "total_pnl": pnls.sum(), "avg_pnl": pnls.mean(),
            "sharpe": sr, "mdd": mdd, "pf": pf,
            "kelly": kf, "equity_analysis": ec,
            "pnls": pnls, "pnl_w": pnl_w, "pnl_l": pnl_l,
        }

    # ────────────────────────────────────────────
    # Comparison table header
    # ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TIMEFRAME SWEEP COMPARISON (Market Order: 2% slippage + $0.01)")
    print(f"{'─'*72}")
    print(f"  {'Hz':<5} {'Trades':<8} {'T/Day':<7} {'WR':<8} {'PnL':<12} "
          f"{'Sharpe':<8} {'MaxDD':<8} {'PF':<6} {'Kelly':<7} {'Curve'}")
    print(f"  {'─'*72}")

    all_analyses = {}
    for h in HORIZONS:
        for tag, results in [("", sweep_results), ("_gated", gated_results)]:
            if h not in results:
                continue
            a = analyze(results[h]["trades"], "market_2pct")
            key = f"{h}m{tag}"
            all_analyses[key] = a
            ec = a["equity_analysis"]
            curve_tag = "STEADY" if ec["steady"] else f"STREAKY (R²={ec['r_squared']:.2f})"
            if tag == "_gated":
                key_label = f"{h}m*"
            else:
                key_label = f"{h}m "
            print(
                f"  {key_label:<5} {a['trades']:<8} {a['tpd']:<7.0f} {a['wr']:<8.1%} "
                f"${a['total_pnl']:<11,.0f} {a['sharpe']:<8.3f} {a['mdd']:<8.1%} "
                f"{a['pf']:<6.2f} {a['kelly']:<7.1%} {curve_tag}")

    print(f"\n  * = ATR-gated (only trades when volatility > 20% above average)")

    # ────────────────────────────────────────────
    # Cost scenario comparison for best horizon
    # ────────────────────────────────────────────
    # Find best ungated horizon by Sharpe
    best_h = max(
        [h for h in HORIZONS if h in sweep_results],
        key=lambda h: analyze(sweep_results[h]["trades"], "market_2pct")["sharpe"],
        default=1,
    )

    print(f"\n{'─'*72}")
    print(f"  COST SCENARIO COMPARISON (best horizon: {best_h}m)")
    print(f"{'─'*72}")
    print(f"  {'Scenario':<40} {'WR':<8} {'PnL':<12} {'Sharpe':<8} {'PF':<6} {'Kelly'}")
    print(f"  {'─'*72}")

    if best_h in sweep_results:
        for sk, sv in SCENARIOS.items():
            a = analyze(sweep_results[best_h]["trades"], sk)
            print(
                f"  {sv['label']:<40} {a['wr']:<8.1%} ${a['total_pnl']:<11,.0f} "
                f"{a['sharpe']:<8.3f} {a['pf']:<6.2f} {a['kelly']:.1%}")

    # ────────────────────────────────────────────
    # Regime diagnosis per fold
    # ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  REGIME DIAGNOSIS: Why do some folds win and others lose?")
    print(f"{'─'*72}")

    for h in HORIZONS:
        if h not in sweep_results:
            continue
        trades = sweep_results[h]["trades"]
        folds = sorted(set(t["fold"] for t in trades))

        print(f"\n  ── {h}m Horizon ──")
        print(f"  {'Fold':<6} {'Trades':<8} {'WR':<8} {'ATR%':<8} {'Trend':<8} "
              f"{'BTC Δ%':<9} {'Regime'}")
        print(f"  {'─'*65}")

        for f in folds:
            ft = [t for t in trades if t["fold"] == f]
            wr = np.mean([t["correct"] for t in ft])
            atr = ft[0]["fold_atr"]
            trend = ft[0]["fold_trend"]
            btc_chg = ft[0]["fold_price_chg"]

            # Classify regime
            if abs(btc_chg) > 3 and atr > 0.15:
                regime = "HIGH-VOL TREND"
            elif abs(btc_chg) > 1.5:
                regime = "TRENDING"
            elif atr > 0.12:
                regime = "CHOPPY/HIGH-VOL"
            else:
                regime = "LOW-VOL RANGE"

            marker = " ★" if wr > 0.60 else (" ✗" if wr < 0.48 else "")
            print(
                f"  {f:<6} {len(ft):<8} {wr:<8.1%} {atr:<8.3f} {trend:<8.1f} "
                f"{btc_chg:<+9.1f} {regime}{marker}")

    # ────────────────────────────────────────────
    # Equity curve detail for best horizon
    # ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  EQUITY CURVE DETAIL ({best_h}m, Market Order)")
    print(f"{'─'*72}")

    if best_h in sweep_results:
        a = all_analyses.get(f"{best_h}m")
        if a:
            ec = a["equity_analysis"]
            q_pnls = ec["quartile_pnls"]
            total = ec["total_pnl"]

            print(f"\n  Equity R² (linearity):    {ec['r_squared']:.3f}  "
                  f"({'GOOD' if ec['r_squared'] > 0.7 else 'POOR — concentrated gains'})")
            print(f"  Longest drawdown:         {ec['longest_dd_bars']} trades")
            print(f"\n  Profit by quartile of trades:")
            for i, qp in enumerate(q_pnls):
                bar_len = int(abs(qp) / max(abs(x) for x in q_pnls) * 30) if max(abs(x) for x in q_pnls) > 0 else 0
                bar = "█" * bar_len if qp > 0 else "░" * bar_len
                pct = qp / total * 100 if total != 0 else 0
                print(f"    Q{i+1}: ${qp:>+10,.0f}  ({pct:>+5.0f}%)  {bar}")

    # ────────────────────────────────────────────
    # Kelly-Criterion position sizing simulation
    # ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  KELLY CRITERION: Optimal Position Sizing")
    print(f"{'─'*72}")

    print(f"\n  {'Hz':<5} {'WR':<8} {'Win$':<8} {'Loss$':<8} {'Kelly f*':<10} "
          f"{'½Kelly':<10} {'Suggested$/trade'}")
    print(f"  {'─'*65}")

    for h in HORIZONS:
        if h not in sweep_results:
            continue
        a = analyze(sweep_results[h]["trades"], "market_2pct")
        kf = a["kelly"]
        half_k = kf / 2
        # With $50k bankroll, suggested trade size
        suggested = 50000 * half_k
        print(
            f"  {h}m{'':<3} {a['wr']:<8.1%} ${a['pnl_w']:<7.2f} ${a['pnl_l']:<7.2f} "
            f"{kf:<10.2%} {half_k:<10.2%} ${suggested:>,.0f}")

    # ────────────────────────────────────────────
    # Monte Carlo on best horizon
    # ────────────────────────────────────────────
    print(f"\n[6/6] MONTE CARLO ({args.mc_runs:,} shuffles)")
    print("=" * 72)

    for h in HORIZONS:
        if h not in sweep_results:
            continue
        a = analyze(sweep_results[h]["trades"], "market_2pct")
        pnls = a["pnls"]

        mc = monte_carlo(pnls, n_runs=args.mc_runs)
        if not mc:
            continue

        print(f"\n  ── {h}m Horizon ({len(pnls):,} trades) ──")
        print(f"    Sharpe:  5th={mc['sharpe_5']:+.3f}  med={mc['sharpe_50']:+.3f}  "
              f"95th={mc['sharpe_95']:+.3f}")
        print(f"    MaxDD:   5th={mc['mdd_5']:.1%}  med={mc['mdd_50']:.1%}  "
              f"95th={mc['mdd_95']:.1%}  worst={mc['mdd_worst']:.1%}")
        print(f"    P&L:     5th=${mc['pnl_5']:+,.0f}  med=${mc['pnl_50']:+,.0f}  "
              f"95th=${mc['pnl_95']:+,.0f}")
        print(f"    P(profit)={mc['p_profit']:.1%}  P(ruin)={mc['p_ruin']:.1%}")

    # ────────────────────────────────────────────
    # FINAL VERDICT
    # ────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  FINAL VERDICT")
    print(f"{'═'*72}")

    # Find best overall config
    best_key = None
    best_sharpe = -999
    for key, a in all_analyses.items():
        if a["sharpe"] > best_sharpe and a["trades"] > 20:
            best_sharpe = a["sharpe"]
            best_key = key

    if best_key:
        a = all_analyses[best_key]
        ec = a["equity_analysis"]

        print(f"""
  Best configuration: {best_key} horizon
  ─────────────────────────────────
  Win Rate:       {a['wr']:.1%}
  Sharpe:         {a['sharpe']:.3f}
  Max Drawdown:   {a['mdd']:.1%}
  Profit Factor:  {a['pf']:.2f}
  Kelly fraction: {a['kelly']:.1%} (half-Kelly: {a['kelly']/2:.1%})
  Equity curve:   {'STEADY' if ec['steady'] else 'STREAKY'}  (R²={ec['r_squared']:.2f})
  Total P&L:      ${a['total_pnl']:+,.0f}  ({a['trades']} trades, {a['tpd']:.0f}/day)
""")

        if a["sharpe"] > 0.8 and a["wr"] > 0.55 and ec["steady"]:
            print("  VERDICT: ✓ HIGH-CONFIDENCE EDGE")
            print("  The strategy shows consistent, cost-adjusted alpha.")
            daily = a["avg_pnl"] * a["tpd"]
            print(f"  Est. daily P&L ($100/trade): ${daily:+,.0f}")
            print(f"  To hit $10k/day: trade ${10000/daily*100:,.0f}/trade" if daily > 0 else "")
        elif a["sharpe"] > 0.3 and a["wr"] > 0.52:
            print("  VERDICT: ~ MARGINAL EDGE")
            print("  Profitable after costs, but unstable across regimes.")
            print("  RECOMMENDATIONS:")
            print("    1. Use LIMIT ORDERS to cut slippage (0.5% vs 2%)")
            print("    2. Add ATR gate — only trade high-volatility windows")
            print("    3. Use HALF-KELLY sizing to survive drawdowns")
            print("    4. Prefer LONGER horizons (5m/15m) for better cost ratio")
        else:
            print("  VERDICT: ✗ NO RELIABLE EDGE")
            print("  The signal-to-cost ratio is too thin for production trading.")
    else:
        print("\n  No viable configuration found.")


if __name__ == "__main__":
    main()
