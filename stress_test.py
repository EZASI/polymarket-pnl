#!/usr/bin/env python3
"""
STRESS TEST: Walk-Forward + Monte Carlo Simulation of Ultra Predictor
======================================================================

Tests the ultra_predictor logic under realistic Polymarket conditions:

  1. WALK-FORWARD OPTIMIZATION
     - 7-day training window, 2-day out-of-sample testing
     - Slides forward 2 days at a time across full dataset
     - Separate model retrained every fold (no lookahead bias)

  2. REALISTIC COST MODEL
     - 2% slippage on every trade (Polymarket low-liquidity penalty)
     - $0.01 spread penalty per contract (bid-ask crossing)
     - Models buying at $0.50 (fair-odds), payout $1.00 on win

  3. RISK METRICS
     - Sharpe Ratio (annualised)
     - Maximum Drawdown (peak-to-trough)
     - Calmar Ratio (annualised return / max drawdown)
     - Win rate, profit factor, average trade P&L

  4. MONTE CARLO SIMULATION (1,000 iterations)
     - Shuffles the order of OOS trades
     - Rebuilds equity curve each time
     - Reports distribution of Sharpe, MaxDD, final P&L
     - Shows probability of ruin and confidence intervals

Usage:
  python3 stress_test.py                        # default 30 days
  python3 stress_test.py --days 60              # more data
  python3 stress_test.py --mc-runs 5000         # more MC iterations
  python3 stress_test.py --horizon 5            # 5-min horizon
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

# Import feature engineering from ultra_predictor
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

# Polymarket cost model
SLIPPAGE_PCT = 0.02           # 2% slippage
SPREAD_PENALTY = 0.01         # $0.01 per contract
BUY_PRICE = 0.50              # buying at ~50c (fair odds)
WIN_PAYOUT = 1.00             # payout on win
POSITION_SIZE = 100.0         # dollars per trade

# Walk-forward windows (in 1-min candles)
TRAIN_WINDOW_DAYS = 7
TEST_WINDOW_DAYS = 2
TRAIN_WINDOW = TRAIN_WINDOW_DAYS * 1440   # 10,080
TEST_WINDOW = TEST_WINDOW_DAYS * 1440     # 2,880

# Monte Carlo
MC_DEFAULT_RUNS = 1000

# Trading hours per year for annualisation
TRADING_MINUTES_PER_YEAR = 365.25 * 1440


# ============================================================
# STRATEGY FILTERS (mirror ultra_predictor logic)
# ============================================================
def apply_strategy_filters(df: pd.DataFrame, y_pred: np.ndarray,
                           confidence: np.ndarray) -> pd.Series:
    """
    Return a boolean mask of trades that pass ALL selective filters.
    Mirrors the best strategies from ultra_predictor:
      - ML confidence >= 0.55
      - Volume spike + momentum alignment
      OR
      - ML confidence >= 0.60 + trend alignment + RSI confirm
    """
    n = len(df)
    df = df.iloc[:n].reset_index(drop=True)
    mask = pd.Series(False, index=df.index[:n])

    # Helper to safely get column values as numpy array
    def col(name):
        if name in df.columns:
            return df[name].values[:n]
        return np.zeros(n)

    vs = col("vol_spike")
    m3 = col("mom_3")
    m5 = col("mom_5")
    ts = col("trend_score")
    rsi = col("rsi")
    vr = col("vol_ratio")
    ti = col("taker_imbalance")

    # Strategy A: ML conf >= 0.55 + Volume Spike + Momentum
    strat_a = (
        (confidence >= 0.55) & (vs == 1) &
        (((y_pred == 1) & (m3 > 0.1)) | ((y_pred == 0) & (m3 < -0.1)))
    )

    # Strategy B: ML conf >= 0.60 + Trend + RSI
    strat_b = (
        (confidence >= 0.60) &
        (
            ((y_pred == 1) & (ts >= 2) & (rsi > 45) & (rsi < 70)) |
            ((y_pred == 0) & (ts <= 1) & (rsi > 30) & (rsi < 55))
        )
    )

    # Strategy C: ULTRA — ML + Taker + Trend + Volume
    strat_c = (
        (confidence >= 0.60) & (vr > 1.3) &
        (
            ((y_pred == 1) & (ti > 0.01) & (ts >= 2) & (m5 > 0)) |
            ((y_pred == 0) & (ti < -0.01) & (ts <= 1) & (m5 < 0))
        )
    )

    return strat_a | strat_b | strat_c


# ============================================================
# TRADE P&L with Polymarket cost model
# ============================================================
def compute_trade_pnl(correct: bool) -> float:
    """
    Compute P&L for a single Polymarket-style trade.

    Buy at BUY_PRICE ($0.50), with slippage and spread penalty.
      - Effective cost = BUY_PRICE * (1 + SLIPPAGE_PCT) + SPREAD_PENALTY
      - If correct:  receive WIN_PAYOUT, profit = payout - effective_cost
      - If wrong:    lose effective_cost
    Returns: dollar P&L per $1 notional
    """
    effective_cost = BUY_PRICE * (1 + SLIPPAGE_PCT) + SPREAD_PENALTY
    if correct:
        return WIN_PAYOUT - effective_cost   # ~$0.48 profit per contract
    else:
        return -effective_cost                # ~-$0.52 loss per contract


# Pre-compute for speed
PNL_WIN = compute_trade_pnl(True)
PNL_LOSS = compute_trade_pnl(False)


def trade_pnl_array(correct_mask: np.ndarray) -> np.ndarray:
    """Vectorised P&L: array of per-trade dollar returns (per $1 notional)."""
    return np.where(correct_mask, PNL_WIN, PNL_LOSS)


# ============================================================
# RISK METRICS
# ============================================================
def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252) -> float:
    """Annualised Sharpe ratio (assumes 0 risk-free rate)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def calmar_ratio(total_return: float, mdd: float) -> float:
    if mdd == 0:
        return 0.0
    return total_return / mdd


def profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    total_loss = np.abs(losses).sum()
    if total_loss == 0:
        return float("inf") if wins.sum() > 0 else 0.0
    return wins.sum() / total_loss


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================
def walk_forward_backtest(df: pd.DataFrame, horizon: int = 1,
                          verbose: bool = True,
                          feature_cols: list[str] | None = None) -> dict:
    """
    Walk-forward optimisation with 7-day training / 2-day OOS windows.
    Returns all OOS trades with their P&L.
    """
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
    feat_cols = feature_cols if feature_cols else get_feature_cols(df)
    df_clean = df.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)

    n = len(df_clean)
    if n < TRAIN_WINDOW + TEST_WINDOW:
        print(f"  ERROR: Need {TRAIN_WINDOW + TEST_WINDOW:,} rows, have {n:,}")
        return {"trades": [], "error": True}

    all_trades = []
    fold = 0
    start = 0

    while start + TRAIN_WINDOW + TEST_WINDOW <= n:
        train_end = start + TRAIN_WINDOW
        test_end = min(train_end + TEST_WINDOW, n)

        train_df = df_clean.iloc[start:train_end]
        test_df = df_clean.iloc[train_end:test_end]

        X_train = train_df[feat_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feat_cols].values
        y_test = test_df[target_col].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ── Train ensemble ──
        models = []
        probas = []

        if USE_LGB:
            m = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=30, random_state=42, verbose=-1, n_jobs=-1,
            )
            m.fit(X_train_s, y_train)
            models.append(m)
            probas.append(m.predict_proba(X_test_s))

        if USE_XGB:
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, n_jobs=-1,
            )
            m.fit(X_train_s, y_train)
            models.append(m)
            probas.append(m.predict_proba(X_test_s))

        if not models:
            from sklearn.ensemble import GradientBoostingClassifier
            m = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.06,
                subsample=0.8, random_state=42,
            )
            m.fit(X_train_s, y_train)
            models.append(m)
            probas.append(m.predict_proba(X_test_s))

        avg_prob = np.mean(probas, axis=0)
        y_pred = (avg_prob[:, 1] > 0.5).astype(int)
        confidence = np.max(avg_prob, axis=1)

        # ── Apply selective filters ──
        test_df_reset = test_df.reset_index(drop=True)
        trade_mask = apply_strategy_filters(test_df_reset, y_pred, confidence)
        if isinstance(trade_mask, pd.Series):
            trade_indices = np.where(trade_mask.values)[0]
        else:
            trade_indices = np.where(trade_mask)[0]

        if len(trade_indices) > 0:
            for idx in trade_indices:
                correct = y_pred[idx] == y_test[idx]
                pnl = PNL_WIN if correct else PNL_LOSS
                all_trades.append({
                    "fold": fold,
                    "timestamp": test_df.iloc[idx].get("timestamp", None),
                    "pred": int(y_pred[idx]),
                    "actual": int(y_test[idx]),
                    "correct": correct,
                    "confidence": float(confidence[idx]),
                    "pnl_per_dollar": pnl,
                    "pnl_usd": pnl * POSITION_SIZE,
                })

        fold_acc = accuracy_score(y_test, y_pred)
        filtered_acc = accuracy_score(
            y_test[trade_indices], y_pred[trade_indices]
        ) if len(trade_indices) > 0 else 0

        if verbose:
            print(
                f"  Fold {fold:>2d} | "
                f"Train {start:>6,}-{train_end:>6,} | "
                f"Test {train_end:>6,}-{test_end:>6,} | "
                f"All acc {fold_acc:.1%} | "
                f"Filtered: {len(trade_indices):>4d} trades, "
                f"acc {filtered_acc:.1%}"
            )

        fold += 1
        start += TEST_WINDOW  # slide forward by test window

    return {"trades": all_trades, "folds": fold, "error": False}


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
def monte_carlo_simulation(trades: list[dict], n_runs: int = MC_DEFAULT_RUNS,
                           seed: int = 42) -> dict:
    """
    Shuffle trade order 1,000 times, rebuild equity curve each time.
    This tests whether the results depend on lucky sequencing.
    """
    rng = np.random.default_rng(seed)
    pnls = np.array([t["pnl_usd"] for t in trades])
    n_trades = len(pnls)

    if n_trades == 0:
        return {"error": True}

    sharpes = []
    max_dds = []
    final_pnls = []
    ruin_count = 0  # equity drops below -50% of starting capital
    starting_capital = POSITION_SIZE * n_trades * 0.1  # 10% of total exposure

    for i in range(n_runs):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled)

        # Sharpe: treat each trade as one "period"
        sr = sharpe_ratio(shuffled, periods_per_year=252)
        sharpes.append(sr)

        # Max drawdown on equity curve
        equity_with_start = np.concatenate([[0], equity])
        mdd = max_drawdown(equity_with_start + starting_capital)
        max_dds.append(mdd)

        # Final P&L
        final_pnls.append(float(equity[-1]))

        # Ruin: equity ever drops below -50% of starting capital
        if np.min(equity) < -starting_capital * 0.5:
            ruin_count += 1

    return {
        "n_runs": n_runs,
        "n_trades": n_trades,
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "sharpe_5th": float(np.percentile(sharpes, 5)),
        "sharpe_50th": float(np.percentile(sharpes, 50)),
        "sharpe_95th": float(np.percentile(sharpes, 95)),
        "maxdd_mean": float(np.mean(max_dds)),
        "maxdd_std": float(np.std(max_dds)),
        "maxdd_5th": float(np.percentile(max_dds, 5)),
        "maxdd_50th": float(np.percentile(max_dds, 50)),
        "maxdd_95th": float(np.percentile(max_dds, 95)),
        "maxdd_worst": float(np.max(max_dds)),
        "final_pnl_mean": float(np.mean(final_pnls)),
        "final_pnl_std": float(np.std(final_pnls)),
        "final_pnl_5th": float(np.percentile(final_pnls, 5)),
        "final_pnl_50th": float(np.percentile(final_pnls, 50)),
        "final_pnl_95th": float(np.percentile(final_pnls, 95)),
        "prob_profit": float(np.mean(np.array(final_pnls) > 0)),
        "prob_ruin": ruin_count / n_runs,
        "starting_capital": starting_capital,
    }


# ============================================================
# REPORTING
# ============================================================
def print_trade_stats(trades: list[dict], label: str = ""):
    """Print detailed trade statistics."""
    if not trades:
        print(f"  No trades to report. {label}")
        return {}

    pnls = np.array([t["pnl_usd"] for t in trades])
    correct = np.array([t["correct"] for t in trades])
    confs = np.array([t["confidence"] for t in trades])

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    equity = np.cumsum(pnls)
    equity_with_start = np.concatenate([[0], equity])

    wr = correct.mean()
    sr = sharpe_ratio(pnls, periods_per_year=252)
    mdd = max_drawdown(equity_with_start + POSITION_SIZE * len(trades) * 0.1)
    pf = profit_factor(wins, losses)
    total_pnl = pnls.sum()

    n_folds = len(set(t["fold"] for t in trades))
    test_days = n_folds * TEST_WINDOW_DAYS
    trades_per_day = len(trades) / test_days if test_days > 0 else 0

    stats = {
        "total_trades": len(trades),
        "win_rate": wr,
        "sharpe": sr,
        "max_drawdown": mdd,
        "profit_factor": pf,
        "total_pnl": total_pnl,
        "avg_pnl": pnls.mean(),
        "trades_per_day": trades_per_day,
        "avg_confidence": confs.mean(),
        "folds": n_folds,
        "test_days": test_days,
    }

    print(f"""
{'═' * 70}
  {label}WALK-FORWARD OUT-OF-SAMPLE RESULTS
{'═' * 70}

  Cost Model:
    Slippage:        {SLIPPAGE_PCT:.0%}
    Spread penalty:  ${SPREAD_PENALTY:.2f} per contract
    Buy price:       ${BUY_PRICE:.2f}
    Win payout:      ${WIN_PAYOUT:.2f}
    Effective cost:  ${BUY_PRICE * (1 + SLIPPAGE_PCT) + SPREAD_PENALTY:.4f}
    P&L if correct:  ${PNL_WIN:+.4f} per $1
    P&L if wrong:    ${PNL_LOSS:+.4f} per $1
    Position size:   ${POSITION_SIZE:.0f} per trade

  Walk-Forward:
    Training window: {TRAIN_WINDOW_DAYS} days ({TRAIN_WINDOW:,} candles)
    Testing window:  {TEST_WINDOW_DAYS} days ({TEST_WINDOW:,} candles)
    Total folds:     {n_folds}
    Total OOS days:  {test_days}

  ── TRADE STATISTICS ──
    Total trades:        {len(trades):,}
    Trades / day:        {trades_per_day:.1f}
    Win rate:            {wr:.2%}
    Avg confidence:      {confs.mean():.2%}

  ── P&L (${POSITION_SIZE:.0f} per trade) ──
    Total P&L:           ${total_pnl:+,.2f}
    Avg trade P&L:       ${pnls.mean():+.2f}
    Best trade:          ${pnls.max():+.2f}
    Worst trade:         ${pnls.min():+.2f}
    Winning trades:      {len(wins):,} (avg ${wins.mean():+.2f})
    Losing trades:       {len(losses):,} (avg ${losses.mean():.2f})
    Profit factor:       {pf:.2f}

  ── RISK METRICS ──
    Sharpe Ratio:        {sr:.3f}
    Max Drawdown:        {mdd:.2%}
    Calmar Ratio:        {calmar_ratio(total_pnl / (POSITION_SIZE * len(trades)), mdd):.3f}
""")

    return stats


def print_monte_carlo_results(mc: dict):
    """Print Monte Carlo simulation results."""
    if mc.get("error"):
        print("  Monte Carlo: No trades to simulate.")
        return

    print(f"""
{'═' * 70}
  MONTE CARLO SIMULATION ({mc['n_runs']:,} iterations, {mc['n_trades']:,} trades)
{'═' * 70}

  Method: Shuffle trade order, rebuild equity curve each iteration.
  Purpose: Test whether win rate holds under different market sequences.

  ── SHARPE RATIO DISTRIBUTION ──
     5th percentile:   {mc['sharpe_5th']:+.3f}
    Median:            {mc['sharpe_50th']:+.3f}
    Mean:              {mc['sharpe_mean']:+.3f}  (std: {mc['sharpe_std']:.3f})
    95th percentile:   {mc['sharpe_95th']:+.3f}

  ── MAX DRAWDOWN DISTRIBUTION ──
    Best case (5th):   {mc['maxdd_5th']:.2%}
    Median:            {mc['maxdd_50th']:.2%}
    Mean:              {mc['maxdd_mean']:.2%}  (std: {mc['maxdd_std']:.2%})
    95th percentile:   {mc['maxdd_95th']:.2%}
    Worst case:        {mc['maxdd_worst']:.2%}

  ── FINAL P&L DISTRIBUTION (${POSITION_SIZE:.0f}/trade) ──
    5th percentile:    ${mc['final_pnl_5th']:+,.2f}
    Median:            ${mc['final_pnl_50th']:+,.2f}
    Mean:              ${mc['final_pnl_mean']:+,.2f}  (std: ${mc['final_pnl_std']:,.2f})
    95th percentile:   ${mc['final_pnl_95th']:+,.2f}

  ── PROBABILITIES ──
    P(profitable):     {mc['prob_profit']:.1%}
    P(ruin):           {mc['prob_ruin']:.1%}
    (ruin = equity drops > 50% of ${mc['starting_capital']:,.0f} starting capital)

  ── VERDICT ──""")

    # Interpret results
    if mc["sharpe_5th"] > 0.5:
        print("    ✓ ROBUST: Even worst-case Sharpe > 0.5 across all shuffles.")
    elif mc["sharpe_50th"] > 0.3:
        print("    ~ MARGINAL: Median Sharpe positive but tail risk exists.")
    elif mc["sharpe_50th"] > 0:
        print("    ⚠ FRAGILE: Positive median but 5th percentile is concerning.")
    else:
        print("    ✗ UNPROFITABLE: Strategy does not survive shuffle test.")

    if mc["prob_ruin"] > 0.1:
        print(f"    ⚠ HIGH RUIN RISK: {mc['prob_ruin']:.0%} of paths hit ruin.")
    elif mc["prob_ruin"] > 0.01:
        print(f"    ~ MODERATE RUIN RISK: {mc['prob_ruin']:.1%} of paths hit ruin.")
    else:
        print(f"    ✓ LOW RUIN RISK: {mc['prob_ruin']:.2%} of paths hit ruin.")

    if mc["prob_profit"] > 0.90:
        print(f"    ✓ {mc['prob_profit']:.0%} of shuffled paths are profitable.")
    elif mc["prob_profit"] > 0.60:
        print(f"    ~ {mc['prob_profit']:.0%} of shuffled paths are profitable.")
    else:
        print(f"    ✗ Only {mc['prob_profit']:.0%} of shuffled paths are profitable.")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Stress Test: Walk-Forward + Monte Carlo")
    parser.add_argument("--days", type=int, default=30, help="Days of Binance data")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (minutes)")
    parser.add_argument("--mc-runs", type=int, default=MC_DEFAULT_RUNS, help="Monte Carlo iterations")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--position", type=float, default=100.0, help="Position size per trade ($)")
    args = parser.parse_args()

    global SYMBOL, POSITION_SIZE
    SYMBOL = args.symbol
    POSITION_SIZE = args.position

    print("=" * 70)
    print("  STRESS TEST: Walk-Forward + Monte Carlo")
    print("=" * 70)
    print(f"  Symbol:       {SYMBOL}")
    print(f"  Horizon:      {args.horizon}m")
    print(f"  Data:         {args.days} days from Binance")
    print(f"  WF Windows:   {TRAIN_WINDOW_DAYS}d train / {TEST_WINDOW_DAYS}d OOS")
    print(f"  Slippage:     {SLIPPAGE_PCT:.0%}")
    print(f"  Spread:       ${SPREAD_PENALTY}")
    print(f"  MC Runs:      {args.mc_runs:,}")
    print(f"  Position:     ${POSITION_SIZE:.0f}/trade")
    print("=" * 70)

    # ── Step 1: Fetch data ──
    print(f"\n[1/5] Fetching {args.days} days of {SYMBOL} data from Binance...")
    collector = BinanceDataCollector(SYMBOL)
    df = collector.get_klines_history(days=args.days)
    if df.empty:
        print("ERROR: No data fetched.")
        sys.exit(1)

    print(f"\n  Rows:  {len(df):,}")
    print(f"  Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Price: ${df['close'].min():,.2f} → ${df['close'].max():,.2f}")

    # ── Step 2: Feature engineering ──
    print(f"\n[2/5] Engineering features...")
    df = engineer_all_features(df)

    print(f"  Merging futures data (funding, L/S ratio, OI, taker volume)...")
    df = merge_futures_data(df, collector)

    df = create_targets(df)

    # Forward-fill AND backfill sparse futures columns so we don't lose rows
    sparse_cols = [c for c in df.columns if c in [
        "fund_rate", "fund_rate_pct", "fund_extreme_long", "fund_extreme_short",
        "ls_ratio", "long_pct", "short_pct", "crowd_long", "crowd_short",
        "taker_ls", "taker_buy_dominant", "oi", "oi_change", "oi_rising",
        "oi_falling", "basis", "basis_bps",
    ]]
    for c in sparse_cols:
        if c in df.columns:
            df[c] = df[c].ffill().bfill().fillna(0)

    feat_cols = get_feature_cols(df)
    print(f"  Features: {len(feat_cols)}")

    # Check how many clean rows we have
    clean_count = len(df.dropna(subset=feat_cols + [f"target_{args.horizon}"]))
    print(f"  Clean rows: {clean_count:,} / {len(df):,}")
    if clean_count < TRAIN_WINDOW + TEST_WINDOW:
        print(f"  ⚠ Still sparse. Dropping columns with >50% NaN...")
        for c in list(feat_cols):
            if df[c].isna().mean() > 0.5:
                feat_cols.remove(c)
                print(f"    Dropped: {c} ({df[c].isna().mean():.0%} NaN)")
        # Fill remaining NaNs with column median
        for c in feat_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        clean_count = len(df.dropna(subset=feat_cols + [f"target_{args.horizon}"]))
        print(f"  Clean rows after fix: {clean_count:,}")

    # ── Step 3: Walk-Forward Backtest ──
    print(f"\n[3/5] Running Walk-Forward Backtest ({args.horizon}m horizon)...")
    print(f"  Window: {TRAIN_WINDOW_DAYS}d train → {TEST_WINDOW_DAYS}d OOS, slide {TEST_WINDOW_DAYS}d")
    print()

    wf_result = walk_forward_backtest(df, horizon=args.horizon, verbose=True,
                                      feature_cols=feat_cols)

    if wf_result.get("error") or not wf_result["trades"]:
        print("\n  ERROR: Walk-forward produced no trades.")
        print("  Try increasing --days or lowering filter thresholds.")
        sys.exit(1)

    trades = wf_result["trades"]

    # ── Step 4: Trade Statistics ──
    print(f"\n[4/5] Computing trade statistics...")
    stats = print_trade_stats(trades, label=f"{args.horizon}m | ")

    # Per-fold breakdown
    folds = sorted(set(t["fold"] for t in trades))
    print(f"  ── PER-FOLD BREAKDOWN ──")
    print(f"  {'Fold':<6} {'Trades':<8} {'WinRate':<10} {'P&L':<12} {'Sharpe':<10}")
    print(f"  {'─' * 50}")
    for f in folds:
        fold_trades = [t for t in trades if t["fold"] == f]
        fold_pnls = np.array([t["pnl_usd"] for t in fold_trades])
        fold_wr = np.mean([t["correct"] for t in fold_trades])
        fold_sr = sharpe_ratio(fold_pnls, 252)
        print(
            f"  {f:<6} {len(fold_trades):<8} {fold_wr:<10.1%} "
            f"${fold_pnls.sum():<11,.2f} {fold_sr:<10.3f}"
        )

    # ── Step 5: Monte Carlo ──
    print(f"\n[5/5] Running Monte Carlo simulation ({args.mc_runs:,} iterations)...")
    t0 = time.time()
    mc = monte_carlo_simulation(trades, n_runs=args.mc_runs)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    print_monte_carlo_results(mc)

    # ── Final Summary ──
    print(f"""
{'═' * 70}
  FINAL STRESS TEST SUMMARY
{'═' * 70}

  Horizon:             {args.horizon}m
  Walk-Forward Folds:  {wf_result['folds']}
  Total OOS Trades:    {len(trades):,}
  Win Rate (OOS):      {stats['win_rate']:.2%}

  After costs (2% slippage + $0.01 spread):
    Total P&L:         ${stats['total_pnl']:+,.2f}
    Sharpe Ratio:      {stats['sharpe']:.3f}
    Max Drawdown:      {stats['max_drawdown']:.2%}
    Profit Factor:     {stats['profit_factor']:.2f}

  Monte Carlo ({args.mc_runs:,} shuffles):
    Median Sharpe:     {mc['sharpe_50th']:+.3f}
    Median MaxDD:      {mc['maxdd_50th']:.2%}
    P(Profitable):     {mc['prob_profit']:.1%}
    P(Ruin):           {mc['prob_ruin']:.1%}

  BOTTOM LINE:""")

    # Final verdict
    if stats["win_rate"] > 0.55 and mc["prob_profit"] > 0.80 and stats["sharpe"] > 0.3:
        print("    ✓ Strategy shows EDGE after costs and stress testing.")
        print(f"    Estimated daily P&L: ${stats['avg_pnl'] * stats['trades_per_day']:+,.2f}")
    elif stats["win_rate"] > 0.52 and mc["prob_profit"] > 0.60:
        print("    ~ Strategy shows MARGINAL edge. Needs refinement.")
        print("    Consider: higher confidence thresholds, fewer trades, larger position on strongest signals.")
    else:
        print("    ✗ Strategy does NOT show reliable edge after costs.")
        print("    The slippage and spread penalties consume the thin ML edge.")
        print("    Consider: reducing costs (limit orders), adding orderbook data, or longer horizons.")


if __name__ == "__main__":
    main()
