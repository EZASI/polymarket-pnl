#!/usr/bin/env python3
"""
BACKTEST: 10 Revenue Bots on Historical 15-Minute Windows
============================================================

Fetches 30 days of 1-minute Binance data, simulates every 15-minute
window as if it were a Polymarket "Bitcoin Up or Down" market, and
runs all 10 bot strategies against it.

This gives ~2,880 windows (30 days * 96 windows/day) per bot.

Usage:
  python3 backtest_bots.py --days 30
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ultra_predictor import (
    BinanceDataCollector,
    engineer_all_features,
    create_targets,
    get_feature_cols,
)

# Polymarket cost model
EFFECTIVE_COST = 0.52  # buy at $0.52 after spread+slippage
TRADE_SIZE = 2000.0


def compute_features_at(df: pd.DataFrame, idx: int, lookback: int = 60) -> dict:
    """
    Compute streaming-like features at a specific index using surrounding data.
    Simulates what the streaming engine would see at window start.
    """
    start = max(0, idx - lookback)
    window = df.iloc[start:idx + 1]

    if len(window) < 10:
        return {}

    last = window.iloc[-1]
    features = {"btc_price": float(last["close"])}

    # Momentum (using 1-min bars as ~seconds equivalent)
    for p, label in [(1, "5s"), (3, "15s"), (5, "30s")]:
        if len(window) > p:
            ret = (window["close"].iloc[-1] - window["close"].iloc[-1 - p]) / window["close"].iloc[-1 - p] * 100
            features[f"btc_mom_{label}"] = float(ret)
            if label == "5s":
                features[f"btc_ret_{label}"] = float(ret)

    # Trade flow (taker buy ratio as proxy)
    if "taker_buy_base" in window.columns and "volume" in window.columns:
        recent = window.tail(5)
        total_vol = recent["volume"].sum()
        buy_vol = recent["taker_buy_base"].sum()
        if total_vol > 0:
            buy_ratio = buy_vol / total_vol
            features["btc_imb_1s"] = float(buy_ratio - 0.5) * 2
            features["btc_imb_5s"] = float(buy_ratio - 0.5) * 2
            features["btc_imb_roll_5s"] = float(buy_ratio - 0.5) * 2

    # Micro volatility
    if len(window) >= 10:
        recent = window.tail(10)
        highs = recent["high"].values
        lows = recent["low"].values
        if lows.min() > 0:
            features["micro_volatility"] = float((highs.max() - lows.min()) / lows.min() * 100)
            features["low_vol"] = 1 if features["micro_volatility"] < 0.05 else 0

    # Whale proxy (large volume bars)
    if "volume" in window.columns and "quote_volume" in window.columns:
        recent = window.tail(5)
        avg_qv = window["quote_volume"].mean()
        large = recent[recent["quote_volume"] > avg_qv * 3]
        features["btc_whale_count_10s"] = len(large)
        features["btc_whale_count_60s"] = len(large)

        if len(large) > 0:
            buy_vol = large["taker_buy_base"].sum() if "taker_buy_base" in large.columns else 0
            sell_vol = (large["volume"] - large["taker_buy_base"]).sum() if "taker_buy_base" in large.columns else 0
            total = buy_vol + sell_vol
            features["btc_whale_imb"] = float((buy_vol - sell_vol) / total) if total > 0 else 0
            features["btc_whale_buy_usd"] = float(buy_vol * last["close"])
            features["btc_whale_sell_usd"] = float(sell_vol * last["close"])
        else:
            features["btc_whale_imb"] = 0
            features["btc_whale_buy_usd"] = 0
            features["btc_whale_sell_usd"] = 0

    # ETH proxy (use BTC data shifted as fake ETH — not ideal but functional)
    # In live trading we have real ETH data, but for backtesting we approximate
    if len(window) > 3:
        eth_ret = (window["close"].iloc[-2] - window["close"].iloc[-3]) / window["close"].iloc[-3] * 100
        btc_ret = features.get("btc_ret_5s", 0)
        features["eth_ret_5s"] = float(eth_ret * 1.2)  # ETH is ~1.2x BTC volatility
        features["eth_btc_lead"] = features["eth_ret_5s"] - btc_ret
        features["eth_imb_5s"] = features.get("btc_imb_5s", 0) * 0.9

    # Simulated Polymarket data (50/50 market)
    features["poly_up_price"] = 0.50
    features["poly_down_price"] = 0.50
    features["poly_spread"] = 0.02
    features["poly_liquidity"] = 20000
    features["poly_up_buy_price"] = 0.51
    features["poly_down_buy_price"] = 0.51

    return features


# ============================================================
# BOT STRATEGIES (same logic as tournament.py)
# ============================================================

def bot1_trend(f):
    m5 = f.get("btc_mom_5s", 0); m15 = f.get("btc_mom_15s", 0)
    m30 = f.get("btc_mom_30s", 0); flow = f.get("btc_imb_roll_5s", 0)
    vol = f.get("micro_volatility", 0)
    if vol > 0.12: return None
    if m15 != 0 and m30 != 0 and ((m15 > 0) != (m30 > 0)): return None
    if m5 > 0.01 and m15 > 0.02 and m30 > 0.02 and flow > 0.05: return ("UP", 0.70)
    elif m5 < -0.01 and m15 < -0.02 and m30 < -0.02 and flow < -0.05: return ("DOWN", 0.70)
    return None

def bot2_whale(f):
    wc = f.get("btc_whale_count_10s", 0) + f.get("btc_whale_count_60s", 0)
    wi = f.get("btc_whale_imb", 0)
    wb = f.get("btc_whale_buy_usd", 0); ws = f.get("btc_whale_sell_usd", 0)
    if wc >= 2 and wi > 0.5 and wb > 300_000: return ("UP", 0.80)
    elif wc >= 2 and wi < -0.5 and ws > 300_000: return ("DOWN", 0.80)
    return None

def bot3_flow(f):
    flow = f.get("btc_imb_5s", 0); roll = f.get("btc_imb_roll_5s", 0)
    m5 = f.get("btc_mom_5s", 0)
    if flow > 0.20 and roll > 0.15 and m5 > 0: return ("UP", 0.68)
    elif flow < -0.20 and roll < -0.15 and m5 < 0: return ("DOWN", 0.68)
    return None

def bot4_eth(f):
    er = f.get("eth_ret_5s", 0); br = f.get("btc_ret_5s", 0)
    ef = f.get("eth_imb_5s", 0)
    if er and br is not None:
        if er > 0.08 and br < 0.03 and ef > 0.05: return ("UP", 0.65)
        elif er < -0.08 and br > -0.03 and ef < -0.05: return ("DOWN", 0.65)
    return None

def bot5_revert(f):
    m15 = f.get("btc_mom_15s", 0); m30 = f.get("btc_mom_30s", 0)
    m5 = f.get("btc_mom_5s", 0)
    if m30 > 0.12 and m15 > 0.06 and m5 < m15: return ("DOWN", 0.60)
    elif m30 < -0.12 and m15 < -0.06 and m5 > m15: return ("UP", 0.60)
    return None

def bot6_liquidity(f):
    up_ask = f.get("poly_up_buy_price", 0.51)
    down_ask = f.get("poly_down_buy_price", 0.51)
    m15 = f.get("btc_mom_15s", 0); flow = f.get("btc_imb_5s", 0)
    if up_ask < 0.48 and m15 > 0.01 and flow > 0.05: return ("UP", 0.70)
    elif down_ask < 0.48 and m15 < -0.01 and flow < -0.05: return ("DOWN", 0.70)
    return None

def bot7_pattern(f, hist):
    if len(hist) < 3: return None
    recent = [h["actual"] for h in hist[-3:]]
    m15 = f.get("btc_mom_15s", 0); flow = f.get("btc_imb_5s", 0)
    if all(d == "UP" for d in recent) and m15 > 0 and flow > 0: return ("UP", 0.65)
    elif all(d == "DOWN" for d in recent) and m15 < 0 and flow < 0: return ("DOWN", 0.65)
    if len(set(recent)) > 1 and recent[-1] != recent[-2]:
        nd = "DOWN" if recent[-1] == "UP" else "UP"
        if (nd == "UP" and flow > 0) or (nd == "DOWN" and flow < 0): return (nd, 0.55)
    return None

def bot8_lowvol(f):
    lv = f.get("low_vol", 0); vol = f.get("micro_volatility", 0)
    m15 = f.get("btc_mom_15s", 0); flow = f.get("btc_imb_5s", 0)
    if not lv or vol > 0.06: return None
    if m15 > 0.015 and flow > 0.03: return ("UP", 0.72)
    elif m15 < -0.015 and flow < -0.03: return ("DOWN", 0.72)
    return None

def bot9_ml(f, model_data):
    if not model_data: return None
    models, scaler, feat_cols = model_data
    try:
        fv = np.zeros(len(feat_cols))
        for i, col in enumerate(feat_cols):
            fv[i] = f.get(col, 0.0)
        fv_s = scaler.transform(fv.reshape(1, -1))
        probas = [m.predict_proba(fv_s) for _, m in models]
        avg = np.mean(probas, axis=0)
        up = float(avg[0, 1]); conf = max(up, 1 - up)
        if conf >= 0.57:
            m15 = f.get("btc_mom_15s", 0); flow = f.get("btc_imb_5s", 0)
            d = "UP" if up > 0.5 else "DOWN"
            if (d == "UP" and m15 > 0 and flow > 0) or (d == "DOWN" and m15 < 0 and flow < 0):
                return (d, min(conf * 1.1, 0.90))
    except: pass
    return None

def bot10_consensus(f, hist, model_data):
    results = [bot1_trend(f), bot2_whale(f), bot3_flow(f), bot4_eth(f),
               bot5_revert(f), bot6_liquidity(f), bot7_pattern(f, hist),
               bot8_lowvol(f), bot9_ml(f, model_data)]
    up = sum(1 for r in results if r and r[0] == "UP")
    dn = sum(1 for r in results if r and r[0] == "DOWN")
    if up >= 4: return ("UP", 0.85)
    elif dn >= 4: return ("DOWN", 0.85)
    return None

BOT_FUNCS = {
    "B1: Trend Surfer": lambda f, h, m: bot1_trend(f),
    "B2: Whale Hunter": lambda f, h, m: bot2_whale(f),
    "B3: Flow Sniper": lambda f, h, m: bot3_flow(f),
    "B4: ETH Arbitrage": lambda f, h, m: bot4_eth(f),
    "B5: Mean Reversion": lambda f, h, m: bot5_revert(f),
    "B6: Liquidity Sentry": lambda f, h, m: bot6_liquidity(f),
    "B7: Pattern Memory": lambda f, h, m: bot7_pattern(f, h),
    "B8: Volatility Edge": lambda f, h, m: bot8_lowvol(f),
    "B9: ML Hybrid": lambda f, h, m: bot9_ml(f, m),
    "B10: Consensus": lambda f, h, m: bot10_consensus(f, h, m),
}


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df: pd.DataFrame, model_data=None):
    """Run all 10 bots across every 15-minute window in the dataset."""

    # Group data into 15-minute windows
    df["window"] = (df.index // 15) * 15  # 15 bars = 15 minutes
    windows = sorted(df["window"].unique())

    print(f"\n  Total 1-min bars: {len(df):,}")
    print(f"  Total 15-min windows: {len(windows):,}")
    print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # Results per bot
    bot_results = {name: [] for name in BOT_FUNCS}
    history = []  # shared window history for pattern bot

    for i, w_start_idx in enumerate(windows[:-1]):  # skip last (incomplete)
        w_end_idx = windows[i + 1] if i + 1 < len(windows) else w_start_idx + 15

        # Get bars in this window
        w_bars = df[df["window"] == w_start_idx]
        if len(w_bars) < 10:
            continue

        btc_start = float(w_bars.iloc[0]["close"])
        btc_end = float(w_bars.iloc[-1]["close"])
        actual = "UP" if btc_end >= btc_start else "DOWN"
        change_pct = (btc_end - btc_start) / btc_start * 100

        # Compute features at window start (using preceding data)
        feat_idx = w_bars.index[0]
        features = compute_features_at(df, feat_idx, lookback=60)

        if not features:
            continue

        # Run all bots
        for name, func in BOT_FUNCS.items():
            try:
                result = func(features, history, model_data)
            except Exception:
                result = None

            if result:
                direction, confidence = result
                correct = direction == actual
                buy_price = EFFECTIVE_COST
                pnl = (1.0 - buy_price) * TRADE_SIZE if correct else -buy_price * TRADE_SIZE

                bot_results[name].append({
                    "window": i,
                    "direction": direction,
                    "actual": actual,
                    "correct": correct,
                    "pnl": pnl,
                    "btc_change": change_pct,
                    "confidence": confidence,
                })

        history.append({"actual": actual, "btc_change_pct": change_pct})

        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(windows)} windows...")

    return bot_results, len(windows) - 1


def print_results(bot_results: dict, total_windows: int):
    """Print backtest results for all bots."""

    print(f"\n{'=' * 80}")
    print(f"  BACKTEST RESULTS — {total_windows:,} windows (15-min each)")
    print(f"  Trade size: ${TRADE_SIZE:,.0f} | Buy at: ${EFFECTIVE_COST} | Win: $1.00")
    print(f"{'=' * 80}")

    summaries = []
    for name, trades in bot_results.items():
        total = len(trades)
        wins = sum(1 for t in trades if t["correct"])
        pnl = sum(t["pnl"] for t in trades)
        pnls = [t["pnl"] for t in trades]

        wr = wins / total if total > 0 else 0
        avg = pnl / total if total > 0 else 0
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(96)) if len(pnls) >= 2 and np.std(pnls) > 0 else 0

        # Selectivity: what fraction of windows did this bot trade?
        selectivity = total / total_windows if total_windows > 0 else 0
        trades_per_day = total / (total_windows / 96) if total_windows > 0 else 0

        # Daily P&L estimate
        days = total_windows / 96
        daily_pnl = pnl / days if days > 0 else 0

        # Max drawdown
        if pnls:
            equity = np.cumsum(pnls)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity)
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        else:
            max_dd = 0

        # Profit factor
        win_total = sum(t["pnl"] for t in trades if t["correct"])
        loss_total = abs(sum(t["pnl"] for t in trades if not t["correct"]))
        pf = win_total / loss_total if loss_total > 0 else 0

        summaries.append({
            "name": name, "trades": total, "wins": wins, "losses": total - wins,
            "wr": wr, "pnl": pnl, "avg": avg, "sharpe": sharpe,
            "selectivity": selectivity, "tpd": trades_per_day,
            "daily_pnl": daily_pnl, "max_dd": max_dd, "pf": pf,
        })

    summaries.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n  {'#':<3} {'BOT':<22} {'TRADES':<7} {'T/Day':<6} {'WR':<7} "
          f"{'TOTAL P&L':<12} {'$/DAY':<10} {'SHARPE':<8} {'MaxDD':<10} {'PF':<6} {'SELECT'}")
    print(f"  {'─' * 100}")

    for i, s in enumerate(summaries):
        wr_str = f"{s['wr']:.0%}" if s['trades'] > 0 else "--"
        marker = " ***" if s['daily_pnl'] >= 5000 else (" **" if s['daily_pnl'] >= 1000 else (" *" if s['pnl'] > 0 else ""))
        print(
            f"  {i + 1:<3} {s['name']:<22} {s['trades']:<7} {s['tpd']:<6.0f} {wr_str:<7} "
            f"${s['pnl']:<11,.0f} ${s['daily_pnl']:<9,.0f} {s['sharpe']:<8.2f} "
            f"${s['max_dd']:<9,.0f} {s['pf']:<6.2f} {s['selectivity']:.0%}{marker}"
        )

    # Detailed analysis for top 3
    print(f"\n{'=' * 80}")
    print(f"  TOP PERFORMERS — DETAILED ANALYSIS")
    print(f"{'=' * 80}")

    for s in summaries[:3]:
        if s["trades"] == 0:
            continue
        trades = bot_results[s["name"]]
        pnls = [t["pnl"] for t in trades]
        equity = np.cumsum(pnls)

        # Win/loss streaks
        max_win_streak = max_loss_streak = current = 0
        last_correct = None
        for t in trades:
            if t["correct"] == last_correct:
                current += 1
            else:
                current = 1
            if t["correct"]:
                max_win_streak = max(max_win_streak, current)
            else:
                max_loss_streak = max(max_loss_streak, current)
            last_correct = t["correct"]

        # Equity curve quality
        if len(equity) > 4:
            q = len(equity) // 4
            q_pnls = [sum(pnls[i * q:(i + 1) * q]) for i in range(4)]
        else:
            q_pnls = [sum(pnls)]

        print(f"\n  {s['name']}")
        print(f"  {'─' * 50}")
        print(f"  Win Rate:          {s['wr']:.1%}")
        print(f"  Total P&L:         ${s['pnl']:+,.0f}")
        print(f"  Daily estimate:    ${s['daily_pnl']:+,.0f}")
        print(f"  Trades/day:        {s['tpd']:.1f}")
        print(f"  Selectivity:       {s['selectivity']:.0%} of windows")
        print(f"  Max win streak:    {max_win_streak}")
        print(f"  Max loss streak:   {max_loss_streak}")
        print(f"  Profit factor:     {s['pf']:.2f}")
        print(f"  Sharpe:            {s['sharpe']:.2f}")
        print(f"  Max drawdown:      ${s['max_dd']:,.0f}")
        print(f"  Equity by quartile:")
        for i, qp in enumerate(q_pnls):
            bar = "█" * int(abs(qp) / max(abs(x) for x in q_pnls) * 20) if max(abs(x) for x in q_pnls) > 0 else ""
            print(f"    Q{i + 1}: ${qp:>+10,.0f}  {'▓' if qp > 0 else '░'}{bar}")

    # $5k/day analysis
    print(f"\n{'=' * 80}")
    print(f"  PATH TO $5,000/DAY")
    print(f"{'=' * 80}")

    for s in summaries:
        if s["trades"] == 0 or s["wr"] <= 0.50:
            continue
        edge = s["wr"] * 0.48 - (1 - s["wr"]) * 0.52
        if edge <= 0:
            continue
        needed_vol = 5000 / edge
        per_trade = needed_vol / s["tpd"] if s["tpd"] > 0 else 0
        bankroll = per_trade * 5  # 5x single trade as bankroll

        print(f"\n  {s['name']}:")
        print(f"    WR: {s['wr']:.1%} | Edge: ${edge:.3f} per $1")
        print(f"    At current rate ({s['tpd']:.0f} trades/day):")
        print(f"      Trade size needed: ${per_trade:,.0f}")
        print(f"      Daily volume: ${needed_vol:,.0f}")
        print(f"      Bankroll needed: ${bankroll:,.0f}")
        if s['daily_pnl'] > 0:
            scale = 5000 / s['daily_pnl']
            print(f"      Scale factor: {scale:.1f}x current size")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    print("=" * 80)
    print("  BACKTESTING 10 BOTS ON HISTORICAL 15-MIN WINDOWS")
    print("=" * 80)

    # Fetch data
    print(f"\n[1/3] Fetching {args.days} days of BTC data from Binance...")
    collector = BinanceDataCollector("BTCUSDT")
    df = collector.get_klines_history(days=args.days)

    if df.empty:
        print("ERROR: No data")
        sys.exit(1)

    print(f"  {len(df):,} candles fetched")
    print(f"  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  ${df['close'].min():,.0f} → ${df['close'].max():,.0f}")

    # Engineer features
    print(f"\n[2/3] Engineering features...")
    df = engineer_all_features(df)

    # Fill NaNs
    for c in df.columns:
        if df[c].dtype in [np.float64, np.int64] and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Train ML model for B9
    print(f"\n  Training ML model for B9...")
    model_data = None
    try:
        feat_cols = get_feature_cols(df)
        df_t = create_targets(df)
        clean = df_t.dropna(subset=feat_cols + ["target_1"])

        # Use first 70% for training, test on rest
        split = int(len(clean) * 0.7)
        train = clean.iloc[:split]
        X_train = train[feat_cols].values
        y_train = train["target_1"].values

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_train)

        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.06,
                               verbose=-1, n_jobs=-1, random_state=42)
        m.fit(X_s, y_train)
        model_data = ([("lgb", m)], scaler, feat_cols)
        print(f"  ML trained on {len(train):,} rows")
    except Exception as e:
        print(f"  ML training failed: {e}")

    # Run backtest
    print(f"\n[3/3] Running backtest...")
    bot_results, total_windows = run_backtest(df, model_data)

    # Print results
    print_results(bot_results, total_windows)


if __name__ == "__main__":
    main()
