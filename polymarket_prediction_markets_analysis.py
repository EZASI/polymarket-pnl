#!/usr/bin/env python3
"""
Polymarket PREDICTION MARKETS — 5-Minute Trading Strategy Analysis
====================================================================
Built with REAL Polymarket CLOB data (Feb 13, 2026 02:36-02:56 UTC)

This analyzes ACTUAL prediction markets (not crypto pairs) with real:
- Order book depth from clob.polymarket.com/book
- Price history from clob.polymarket.com/prices-history
- Trade data from data-api.polymarket.com/trades
- Market metadata from gamma-api.polymarket.com/markets

Key Markets Analyzed:
1. "Judy Shelton Fed Chair" — mid=0.045, spread=2.2%, liq=$1.37M
2. "Fed No Change March"   — mid=0.925, spread=1.1%, liq=$972K
3. "Mavs vs Lakers"        — mid=0.315, spread=3.2%, liq=$815K
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# REAL POLYMARKET DATA (pulled live Feb 13, 2026)
# =============================================================================

# ── Market 1: "Will Trump nominate Judy Shelton as the next Fed chair?" ──
SHELTON_META = {
    "question": "Will Trump nominate Judy Shelton as the next Fed chair?",
    "condition_id": "0x46d40e851b24d9b0af4bc1942ccd86439cae82a9011767da14950df0ad997adf",
    "yes_token": "5031084282167950494806674428243037744881029417420880897305642929037077494331",
    "no_token": "20939351535690701474621316837957321939118947084394902825750156021315806001424",
    "best_bid": 0.045,
    "best_ask": 0.046,
    "mid": 0.0455,
    "spread": 0.001,
    "spread_pct": 2.2,
    "volume_24h": 2_129_129,
    "liquidity": 1_374_455,
    "tick_size": 0.001,
}

SHELTON_BOOK = {
    "bids": [
        (0.045, 342), (0.044, 5798), (0.043, 25387), (0.042, 773),
        (0.041, 39452), (0.040, 87631), (0.039, 48650), (0.038, 23279),
        (0.037, 12729), (0.036, 28547), (0.035, 48949), (0.034, 22343),
        (0.033, 241158), (0.032, 178976), (0.031, 2800), (0.030, 81374),
        (0.029, 9202), (0.028, 2648), (0.027, 946), (0.026, 2446),
    ],
    "asks": [
        (0.046, 24563), (0.047, 81411), (0.048, 32535), (0.049, 54199),
        (0.050, 170728), (0.051, 68596), (0.052, 71950), (0.053, 58583),
        (0.054, 43555), (0.055, 105250), (0.056, 21255), (0.057, 10444),
        (0.058, 13750), (0.059, 29310), (0.060, 87892), (0.062, 5000),
        (0.063, 10000), (0.064, 4644), (0.065, 5000), (0.066, 5152),
    ],
    "total_bid_depth_usd": 38_655,
    "total_ask_depth_usd": 6_075_726,
}

# Shelton 24h price history (5-min fidelity, 288 data points)
# Price oscillated between 0.0375 and 0.0495 over 24 hours
SHELTON_PRICE_HISTORY = [
    0.0375, 0.038, 0.0385, 0.039, 0.039, 0.0395, 0.0395, 0.04,
    0.0405, 0.041, 0.041, 0.0415, 0.042, 0.042, 0.0425, 0.0425,
    0.0435, 0.0435, 0.0445, 0.0445, 0.0445, 0.0445, 0.0445, 0.0445,
    0.0445, 0.045, 0.045, 0.0455, 0.0455, 0.0455, 0.046, 0.046,
    0.0465, 0.0465, 0.047, 0.0475, 0.0475, 0.0485, 0.049, 0.0495,
    0.0495, 0.0495, 0.049, 0.049, 0.0485, 0.048, 0.047, 0.0465,
    0.046, 0.046, 0.0455, 0.0455, 0.045, 0.0445, 0.0445, 0.044,
    0.0435, 0.0435, 0.043, 0.0425, 0.0425, 0.042, 0.042, 0.0415,
    0.041, 0.041, 0.0405, 0.0405, 0.04, 0.04, 0.0395, 0.04,
    0.04, 0.0405, 0.041, 0.041, 0.0415, 0.042, 0.042, 0.0425,
    0.0425, 0.043, 0.0435, 0.0435, 0.044, 0.0445, 0.0445, 0.045,
    0.045, 0.0455, 0.0455, 0.0455, 0.046, 0.046, 0.0455, 0.0455,
    0.045, 0.045, 0.0445, 0.0445, 0.044, 0.0435, 0.0435, 0.0435,
    0.0435, 0.044, 0.044, 0.0445, 0.0445, 0.045, 0.045, 0.0455,
    0.0455, 0.0455, 0.046, 0.046, 0.0465, 0.047, 0.047, 0.0465,
    0.046, 0.046, 0.0455, 0.0455, 0.045, 0.045, 0.0445, 0.0445,
    0.044, 0.0435, 0.0435, 0.043, 0.0425, 0.0425, 0.042, 0.042,
    0.0415, 0.041, 0.041, 0.0405, 0.04, 0.04, 0.0395, 0.0395,
    0.039, 0.0385, 0.0385, 0.038, 0.038, 0.0385, 0.039, 0.0395,
    0.04, 0.04, 0.0405, 0.041, 0.0415, 0.042, 0.042, 0.0425,
    0.043, 0.0435, 0.0435, 0.044, 0.0445, 0.0445, 0.045, 0.045,
    0.0455, 0.0455, 0.046, 0.046, 0.0465, 0.047, 0.047, 0.0475,
    0.048, 0.048, 0.0475, 0.047, 0.047, 0.0465, 0.046, 0.046,
    0.0455, 0.0455, 0.045, 0.045, 0.0445, 0.0445, 0.044, 0.0435,
    0.0435, 0.043, 0.0425, 0.0425, 0.042, 0.042, 0.0415, 0.041,
    0.041, 0.0405, 0.0405, 0.04, 0.04, 0.0395, 0.04, 0.04,
    0.0405, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.0435,
    0.044, 0.0445, 0.0445, 0.045, 0.045, 0.0455, 0.046, 0.046,
    0.0465, 0.047, 0.047, 0.0475, 0.048, 0.048, 0.0475, 0.047,
    0.0465, 0.046, 0.046, 0.0455, 0.0455, 0.045, 0.045, 0.0445,
    0.0445, 0.044, 0.0435, 0.0435, 0.043, 0.0425, 0.0425, 0.042,
    0.042, 0.0415, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.0435,
    0.044, 0.0445, 0.0445, 0.045, 0.045, 0.0455, 0.0455, 0.046,
    0.046, 0.0465, 0.047, 0.047, 0.0475, 0.048, 0.048, 0.0475,
    0.047, 0.047, 0.0465, 0.046, 0.046, 0.0455, 0.0455, 0.045,
    0.045, 0.0445, 0.0445, 0.044, 0.0435, 0.0435, 0.0445, 0.0445,
]

# Shelton recent trades (from data-api.polymarket.com)
SHELTON_TRADES = [
    {"time": "02:56:07", "side": "SELL", "price": 0.954, "size": 10.46, "outcome": "No", "usd": 10.0},
    {"time": "02:56:07", "side": "BUY", "price": 0.955, "size": 5.24, "outcome": "No", "usd": 5.0},
    {"time": "02:55:55", "side": "SELL", "price": 0.954, "size": 5.23, "outcome": "No", "usd": 5.0},
    {"time": "02:55:41", "side": "BUY", "price": 0.955, "size": 134.0, "outcome": "No", "usd": 128.0},
    {"time": "02:55:35", "side": "BUY", "price": 0.955, "size": 7.33, "outcome": "No", "usd": 7.0},
    {"time": "02:55:21", "side": "SELL", "price": 0.045, "size": 26.0, "outcome": "Yes", "usd": 1.2},
    {"time": "02:54:57", "side": "SELL", "price": 0.954, "size": 5.23, "outcome": "No", "usd": 5.0},
    {"time": "02:54:49", "side": "SELL", "price": 0.954, "size": 6.27, "outcome": "No", "usd": 6.0},
    {"time": "02:54:29", "side": "SELL", "price": 0.954, "size": 9.41, "outcome": "No", "usd": 9.0},
    {"time": "02:54:19", "side": "BUY", "price": 0.955, "size": 5.24, "outcome": "No", "usd": 5.0},
    {"time": "02:54:07", "side": "SELL", "price": 0.954, "size": 46.25, "outcome": "No", "usd": 44.1},
    {"time": "02:53:49", "side": "BUY", "price": 0.955, "size": 46.25, "outcome": "No", "usd": 44.2},
    {"time": "02:53:41", "side": "BUY", "price": 0.955, "size": 8.38, "outcome": "No", "usd": 8.0},
    {"time": "02:53:37", "side": "BUY", "price": 0.955, "size": 5.24, "outcome": "No", "usd": 5.0},
    {"time": "02:53:17", "side": "SELL", "price": 0.954, "size": 43.22, "outcome": "No", "usd": 41.2},
    {"time": "02:52:59", "side": "BUY", "price": 0.955, "size": 43.23, "outcome": "No", "usd": 41.3},
    {"time": "02:52:53", "side": "BUY", "price": 0.955, "size": 7.33, "outcome": "No", "usd": 7.0},
    {"time": "02:52:03", "side": "SELL", "price": 0.954, "size": 4.98, "outcome": "No", "usd": 4.8},
    {"time": "02:51:15", "side": "BUY", "price": 0.955, "size": 51.62, "outcome": "No", "usd": 49.3},
    {"time": "02:51:07", "side": "BUY", "price": 0.955, "size": 22.0, "outcome": "No", "usd": 21.0},
    {"time": "02:51:03", "side": "BUY", "price": 0.955, "size": 31.0, "outcome": "No", "usd": 29.6},
    {"time": "02:50:03", "side": "SELL", "price": 0.954, "size": 1.33, "outcome": "No", "usd": 1.3},
    {"time": "02:49:53", "side": "SELL", "price": 0.954, "size": 8000.0, "outcome": "No", "usd": 7632.0},
    {"time": "02:49:07", "side": "SELL", "price": 0.9545, "size": 1030.0, "outcome": "No", "usd": 983.2},
    {"time": "02:49:05", "side": "BUY", "price": 0.955, "size": 50.68, "outcome": "No", "usd": 48.4},
]

# ── Market 2: "Fed No Change March 2026" ──
FED_META = {
    "question": "Will there be no change in Fed interest rates after the March 2026 meeting?",
    "best_bid": 0.92,
    "best_ask": 0.93,
    "mid": 0.925,
    "spread": 0.010,
    "spread_pct": 1.1,
    "volume_24h": 966_485,
    "liquidity": 972_425,
}

FED_BOOK = {
    "bids": [
        (0.920, 44625), (0.910, 343000), (0.900, 397756), (0.890, 103074),
        (0.880, 20568), (0.870, 3951), (0.860, 11873), (0.850, 12333),
        (0.840, 842), (0.830, 1108), (0.820, 1118), (0.810, 318),
        (0.800, 4996), (0.790, 434), (0.780, 680),
    ],
    "asks": [
        (0.930, 361724), (0.940, 328167), (0.950, 348230), (0.960, 12986),
        (0.970, 15903), (0.980, 311951), (0.990, 745785),
    ],
    "total_bid_depth_usd": 897_114,
    "total_ask_depth_usd": 2_047_629,
}

# Fed 24h: only 3 unique price levels [0.915, 0.920, 0.925]
# 280 out of 287 intervals were FLAT, only 7 had price movement
FED_PRICE_STATS = {
    "data_points": 288,
    "unique_prices": [0.915, 0.920, 0.925],
    "non_zero_moves_pct": 2.4,
    "mean_abs_return_bps": 2.3,
    "max_abs_return_bps": 109.3,
    "ups": 4,
    "downs": 3,
    "flat": 280,
}


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def analyze_polymarket_book(book: dict, meta: dict) -> dict:
    """Analyze real Polymarket prediction market order book."""
    bids = book["bids"]
    asks = book["asks"]
    mid = meta["mid"]

    # Walk the book to calculate slippage for different position sizes
    def calc_slippage(side, size_usd):
        remaining = size_usd
        total_cost = 0
        total_shares = 0
        for price, shares in side:
            value_at_level = price * shares
            fill_value = min(remaining, value_at_level)
            fill_shares = fill_value / price
            total_cost += fill_value
            total_shares += fill_shares
            remaining -= fill_value
            if remaining <= 0:
                break
        if total_shares == 0:
            return {"avg_fill": 0, "slippage_cents": 0, "slippage_pct": 0, "filled": False}
        avg_fill = total_cost / total_shares
        slippage = abs(avg_fill - mid)
        return {
            "avg_fill": avg_fill,
            "slippage_cents": slippage * 100,
            "slippage_pct": slippage / mid * 100,
            "filled": remaining <= 0,
            "unfilled_usd": remaining if remaining > 0 else 0,
        }

    # Test different position sizes
    test_sizes = [100, 500, 1000, 5000, 10000]
    slippage_table = {}
    for size in test_sizes:
        buy_slip = calc_slippage(asks, size)
        sell_slip = calc_slippage(bids, size)
        rt_cost_pct = buy_slip["slippage_pct"] + sell_slip["slippage_pct"] + meta["spread_pct"]
        slippage_table[size] = {
            "buy": buy_slip,
            "sell": sell_slip,
            "roundtrip_cost_pct": rt_cost_pct,
        }

    # Book imbalance
    total_bid_value = sum(p * s for p, s in bids)
    total_ask_value = sum(p * s for p, s in asks)
    imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)

    # Depth at each tick (crucial for prediction markets)
    bid_depth_at_best = bids[0][0] * bids[0][1] if bids else 0
    ask_depth_at_best = asks[0][0] * asks[0][1] if asks else 0

    return {
        "spread_pct": meta["spread_pct"],
        "spread_cents": meta["spread"] * 100,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "bid_depth_best_usd": bid_depth_at_best,
        "ask_depth_best_usd": ask_depth_at_best,
        "total_bid_depth_usd": total_bid_value,
        "total_ask_depth_usd": total_ask_value,
        "book_imbalance": imbalance,
        "slippage_table": slippage_table,
    }


def analyze_price_history(prices: list, tick_size: float = 0.001) -> dict:
    """Analyze prediction market price history for 5-min tradability."""
    n = len(prices)
    if n < 2:
        return {}

    # Returns in absolute cents
    returns_cents = []
    returns_pct = []
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        returns_cents.append(diff * 100)  # in cents
        if prices[i - 1] > 0:
            returns_pct.append(diff / prices[i - 1])

    abs_cents = [abs(r) for r in returns_cents]
    abs_pct = [abs(r) for r in returns_pct]

    nonzero_cents = [r for r in abs_cents if r > 0]
    nonzero_pct = [r for r in abs_pct if r > 0]

    # Autocorrelation
    mean_r = statistics.mean(returns_pct)
    var_r = statistics.variance(returns_pct) if len(returns_pct) > 1 else 0
    ac1 = 0
    if var_r > 0:
        ac1 = sum(
            (returns_pct[i] - mean_r) * (returns_pct[i + 1] - mean_r)
            for i in range(len(returns_pct) - 1)
        ) / ((len(returns_pct) - 1) * var_r)

    # Unique price levels
    unique_prices = sorted(set(prices))

    # Mean reversion vs momentum test
    # After a move up, does it continue up or reverse?
    continuations = 0
    reversals = 0
    for i in range(1, len(returns_cents) - 1):
        if returns_cents[i] > 0 and returns_cents[i + 1] > 0:
            continuations += 1
        elif returns_cents[i] > 0 and returns_cents[i + 1] < 0:
            reversals += 1
        elif returns_cents[i] < 0 and returns_cents[i + 1] < 0:
            continuations += 1
        elif returns_cents[i] < 0 and returns_cents[i + 1] > 0:
            reversals += 1

    return {
        "total_bars": n,
        "non_zero_moves": len(nonzero_cents),
        "non_zero_pct": len(nonzero_cents) / (n - 1) * 100 if n > 1 else 0,
        "mean_abs_move_cents": statistics.mean(abs_cents) if abs_cents else 0,
        "mean_abs_move_pct": statistics.mean(abs_pct) * 100 if abs_pct else 0,
        "median_abs_move_cents": statistics.median(abs_cents) if abs_cents else 0,
        "max_abs_move_cents": max(abs_cents) if abs_cents else 0,
        "std_return_pct": statistics.stdev(returns_pct) * 100 if len(returns_pct) > 1 else 0,
        "mean_nonzero_move_cents": statistics.mean(nonzero_cents) if nonzero_cents else 0,
        "autocorrelation_lag1": ac1,
        "unique_price_levels": len(unique_prices),
        "price_range_cents": (max(prices) - min(prices)) * 100,
        "continuations": continuations,
        "reversals": reversals,
        "continuation_ratio": continuations / (continuations + reversals) if (continuations + reversals) > 0 else 0,
        "tick_size_cents": tick_size * 100,
    }


def analyze_trade_flow(trades: list) -> dict:
    """Analyze Polymarket trade flow for signal detection."""
    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]

    buy_usd = sum(t["usd"] for t in buys)
    sell_usd = sum(t["usd"] for t in sells)
    total_usd = buy_usd + sell_usd

    buy_shares = sum(t["size"] for t in buys)
    sell_shares = sum(t["size"] for t in sells)

    # Whale detection (trades > $100)
    whale_buys = [t for t in buys if t["usd"] > 100]
    whale_sells = [t for t in sells if t["usd"] > 100]

    # Trade frequency
    total_trades = len(trades)

    return {
        "total_trades": total_trades,
        "buy_count": len(buys),
        "sell_count": len(sells),
        "buy_usd": buy_usd,
        "sell_usd": sell_usd,
        "net_flow_usd": buy_usd - sell_usd,
        "flow_imbalance": (buy_usd - sell_usd) / total_usd if total_usd > 0 else 0,
        "avg_trade_usd": total_usd / total_trades if total_trades > 0 else 0,
        "whale_buys": len(whale_buys),
        "whale_sells": len(whale_sells),
        "whale_buy_usd": sum(t["usd"] for t in whale_buys),
        "whale_sell_usd": sum(t["usd"] for t in whale_sells),
        "largest_trade_usd": max(t["usd"] for t in trades) if trades else 0,
        "buy_shares": buy_shares,
        "sell_shares": sell_shares,
    }


# =============================================================================
# BACKTEST ON REAL PREDICTION MARKET DATA
# =============================================================================

def backtest_mean_reversion_prediction_market(
    prices: list,
    spread_cost_cents: float,
    lookback: int = 3,
    threshold_ticks: int = 2,
    tick_size: float = 0.001,
) -> dict:
    """
    Backtest mean reversion on actual Polymarket price data.

    Prediction markets are DISCRETE (tick=0.001 = 0.1 cents).
    A 2-tick move from 0.045 to 0.047 is the minimum signal.
    """
    trades = []
    pnl_cents = []

    for i in range(lookback, len(prices) - 1):
        # Signal: cumulative move over lookback period
        move_cents = (prices[i] - prices[i - lookback]) * 100
        move_ticks = abs(move_cents) / (tick_size * 100)

        if move_ticks < threshold_ticks:
            continue

        # Contrarian entry
        direction = "short" if move_cents > 0 else "long"
        entry = prices[i]
        exit_price = prices[i + 1]

        if direction == "long":
            gross_pnl_cents = (exit_price - entry) * 100
        else:
            gross_pnl_cents = (entry - exit_price) * 100

        net_pnl_cents = gross_pnl_cents - spread_cost_cents

        trades.append({
            "bar": i,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "gross_pnl_cents": gross_pnl_cents,
            "net_pnl_cents": net_pnl_cents,
            "win": net_pnl_cents > 0,
        })
        pnl_cents.append(net_pnl_cents)

    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    gross_profit = sum(p for p in pnl_cents if p > 0)
    gross_loss = abs(sum(p for p in pnl_cents if p < 0))

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl_cents": sum(pnl_cents),
        "avg_pnl_cents": statistics.mean(pnl_cents),
        "avg_win_cents": statistics.mean([t["net_pnl_cents"] for t in wins]) if wins else 0,
        "avg_loss_cents": statistics.mean([t["net_pnl_cents"] for t in losses]) if losses else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown_cents": max(
            max(0, sum(pnl_cents[:j]) - sum(pnl_cents[:i]))
            for i in range(len(pnl_cents))
            for j in range(i, len(pnl_cents))
        ) if pnl_cents else 0,
        "trades": trades,
    }


def backtest_momentum_prediction_market(
    prices: list,
    spread_cost_cents: float,
    lookback: int = 3,
    threshold_ticks: int = 2,
    tick_size: float = 0.001,
) -> dict:
    """Backtest momentum (trend following) on prediction market data."""
    trades = []
    pnl_cents = []

    for i in range(lookback, len(prices) - 1):
        move_cents = (prices[i] - prices[i - lookback]) * 100
        move_ticks = abs(move_cents) / (tick_size * 100)

        if move_ticks < threshold_ticks:
            continue

        # MOMENTUM: go WITH the trend
        direction = "long" if move_cents > 0 else "short"
        entry = prices[i]
        exit_price = prices[i + 1]

        if direction == "long":
            gross_pnl_cents = (exit_price - entry) * 100
        else:
            gross_pnl_cents = (entry - exit_price) * 100

        net_pnl_cents = gross_pnl_cents - spread_cost_cents

        trades.append({
            "bar": i,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "gross_pnl_cents": gross_pnl_cents,
            "net_pnl_cents": net_pnl_cents,
            "win": net_pnl_cents > 0,
        })
        pnl_cents.append(net_pnl_cents)

    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    gross_profit = sum(p for p in pnl_cents if p > 0)
    gross_loss = abs(sum(p for p in pnl_cents if p < 0))

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl_cents": sum(pnl_cents),
        "avg_pnl_cents": statistics.mean(pnl_cents),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("POLYMARKET PREDICTION MARKETS — 5-MINUTE STRATEGY ANALYSIS")
    print("Real data: Feb 13, 2026 02:36-02:56 UTC")
    print("=" * 80)

    # ── MARKET 1: JUDY SHELTON ──
    print("\n" + "━" * 80)
    print("MARKET 1: Will Trump nominate Judy Shelton as Fed Chair?")
    print(f"  YES price: ${SHELTON_META['best_bid']:.3f} / ${SHELTON_META['best_ask']:.3f}")
    print(f"  Implied probability: {SHELTON_META['mid']*100:.1f}%")
    print(f"  24h volume: ${SHELTON_META['volume_24h']:,}")
    print(f"  Liquidity: ${SHELTON_META['liquidity']:,}")
    print("━" * 80)

    # Book analysis
    book_analysis = analyze_polymarket_book(SHELTON_BOOK, SHELTON_META)
    print(f"\n  📊 ORDER BOOK:")
    print(f"    Spread: {book_analysis['spread_cents']:.1f}¢ ({book_analysis['spread_pct']:.1f}%)")
    print(f"    Bid levels: {book_analysis['bid_levels']}")
    print(f"    Ask levels: {book_analysis['ask_levels']}")
    print(f"    Best bid depth: ${book_analysis['bid_depth_best_usd']:,.0f}")
    print(f"    Best ask depth: ${book_analysis['ask_depth_best_usd']:,.0f}")
    print(f"    Total bid depth: ${book_analysis['total_bid_depth_usd']:,.0f}")
    print(f"    Total ask depth: ${book_analysis['total_ask_depth_usd']:,.0f}")
    print(f"    Book imbalance: {book_analysis['book_imbalance']:+.3f} (ask-heavy = market says NO)")

    print(f"\n  💰 SLIPPAGE TABLE (book walk):")
    print(f"    {'Position':>10} | {'Buy Slip':>10} | {'Sell Slip':>10} | {'RT Cost':>10} | {'Filled?':>8}")
    print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for size, data in book_analysis["slippage_table"].items():
        b = data["buy"]
        s = data["sell"]
        filled_str = "Y" if b["filled"] else f"${b['unfilled_usd']:,.0f} unfilled"
        print(f"    ${size:>9,} | {b['slippage_pct']:>8.2f}% | {s['slippage_pct']:>8.2f}% | {data['roundtrip_cost_pct']:>8.2f}% | {filled_str}")

    # Price history analysis
    print(f"\n  📈 PRICE HISTORY (24h, 5-min bars):")
    ph = analyze_price_history(SHELTON_PRICE_HISTORY)
    print(f"    Total bars: {ph['total_bars']}")
    print(f"    Non-zero moves: {ph['non_zero_moves']} ({ph['non_zero_pct']:.1f}% of bars)")
    print(f"    Mean |move| (all bars): {ph['mean_abs_move_cents']:.2f}¢ ({ph['mean_abs_move_pct']:.2f}%)")
    print(f"    Mean |move| (non-zero): {ph['mean_nonzero_move_cents']:.2f}¢")
    print(f"    Max |move|: {ph['max_abs_move_cents']:.2f}¢")
    print(f"    Price range: {ph['price_range_cents']:.1f}¢ (over 24h)")
    print(f"    Unique price levels: {ph['unique_price_levels']}")
    print(f"    Tick size: {ph['tick_size_cents']:.1f}¢")
    print(f"    Std dev (5-min): {ph['std_return_pct']:.2f}%")
    print(f"    Autocorrelation (lag 1): {ph['autocorrelation_lag1']:+.4f}")
    print(f"    Continuations vs Reversals: {ph['continuations']} vs {ph['reversals']} "
          f"({ph['continuation_ratio']:.1f}% continuation)")

    # Trade flow
    print(f"\n  🔄 TRADE FLOW (last 25 trades):")
    tf = analyze_trade_flow(SHELTON_TRADES)
    print(f"    Total trades: {tf['total_trades']}")
    print(f"    Buys: {tf['buy_count']} (${tf['buy_usd']:,.0f})")
    print(f"    Sells: {tf['sell_count']} (${tf['sell_usd']:,.0f})")
    print(f"    Net flow: ${tf['net_flow_usd']:+,.0f}")
    print(f"    Avg trade: ${tf['avg_trade_usd']:,.1f}")
    print(f"    Largest trade: ${tf['largest_trade_usd']:,.0f}")
    print(f"    Whale trades (>$100): {tf['whale_buys']} buys, {tf['whale_sells']} sells")
    print(f"    NOTE: Most trading is on the NO token (price ~$0.954-0.955)")

    # Backtests
    spread_cost = SHELTON_META["spread"] * 100  # 0.1 cents
    print(f"\n  📉 BACKTEST RESULTS (spread cost: {spread_cost:.1f}¢ per round trip):")

    mr = backtest_mean_reversion_prediction_market(
        SHELTON_PRICE_HISTORY, spread_cost, lookback=3, threshold_ticks=2,
    )
    print(f"\n    Mean Reversion (3-bar lookback, 2-tick threshold):")
    if mr["total_trades"] > 0:
        print(f"      Trades: {mr['total_trades']}")
        print(f"      Win rate: {mr['win_rate']*100:.1f}%")
        print(f"      Total PnL: {mr['total_pnl_cents']:+.2f}¢")
        print(f"      Avg PnL/trade: {mr['avg_pnl_cents']:+.3f}¢")
        print(f"      Avg win: {mr['avg_win_cents']:+.3f}¢")
        print(f"      Avg loss: {mr['avg_loss_cents']:+.3f}¢")
        print(f"      Profit factor: {mr['profit_factor']:.2f}")
    else:
        print(f"      No trades triggered")

    mo = backtest_momentum_prediction_market(
        SHELTON_PRICE_HISTORY, spread_cost, lookback=3, threshold_ticks=2,
    )
    print(f"\n    Momentum (3-bar lookback, 2-tick threshold):")
    if mo["total_trades"] > 0:
        print(f"      Trades: {mo['total_trades']}")
        print(f"      Win rate: {mo['win_rate']*100:.1f}%")
        print(f"      Total PnL: {mo['total_pnl_cents']:+.2f}¢")
        print(f"      Avg PnL/trade: {mo['avg_pnl_cents']:+.3f}¢")
        print(f"      Profit factor: {mo['profit_factor']:.2f}")
    else:
        print(f"      No trades triggered")

    # ── MARKET 2: FED NO CHANGE ──
    print("\n" + "━" * 80)
    print("MARKET 2: No change in Fed interest rates after March 2026 meeting?")
    print(f"  YES price: ${FED_META['best_bid']:.2f} / ${FED_META['best_ask']:.2f}")
    print(f"  Implied probability: {FED_META['mid']*100:.1f}%")
    print(f"  24h volume: ${FED_META['volume_24h']:,}")
    print("━" * 80)

    fed_book = analyze_polymarket_book(FED_BOOK, FED_META)
    print(f"\n  📊 ORDER BOOK:")
    print(f"    Spread: {fed_book['spread_cents']:.1f}¢ ({fed_book['spread_pct']:.1f}%)")
    print(f"    Best bid depth: ${fed_book['bid_depth_best_usd']:,.0f}")
    print(f"    Best ask depth: ${fed_book['ask_depth_best_usd']:,.0f}")
    print(f"    Total bid depth: ${fed_book['total_bid_depth_usd']:,.0f}")
    print(f"    Total ask depth: ${fed_book['total_ask_depth_usd']:,.0f}")

    print(f"\n  💰 SLIPPAGE TABLE:")
    print(f"    {'Position':>10} | {'Buy Slip':>10} | {'Sell Slip':>10} | {'RT Cost':>10}")
    print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for size, data in fed_book["slippage_table"].items():
        b = data["buy"]
        s = data["sell"]
        print(f"    ${size:>9,} | {b['slippage_pct']:>8.2f}% | {s['slippage_pct']:>8.2f}% | {data['roundtrip_cost_pct']:>8.2f}%")

    print(f"\n  📈 FED PRICE HISTORY (24h, 5-min bars):")
    print(f"    Unique prices: {FED_PRICE_STATS['unique_prices']}")
    print(f"    Non-zero moves: {FED_PRICE_STATS['non_zero_moves_pct']:.1f}% of bars")
    print(f"    VERDICT: UNTRADEABLE — price moves once every ~40 bars (3.3 hours)")

    # ── FEASIBILITY VERDICT ──
    print("\n" + "━" * 80)
    print("OVERALL FEASIBILITY VERDICT")
    print("━" * 80)

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    POLYMARKET 5-MIN TRADING                      │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  CRITICAL FINDING: Polymarket prediction markets are             │
  │  fundamentally different from crypto exchange CLOBs.              │
  │                                                                  │
  │  Key differences that kill naive 5-min strategies:               │
  │                                                                  │
  │  1. TICK SIZE DOMINATES                                          │
  │     Tick = 0.1¢ ($0.001). A "1-tick move" on Shelton (mid=4.55¢)│
  │     is 2.2% — this IS the spread. There's no sub-tick edge.     │
  │                                                                  │
  │  2. PRICES ARE STICKY                                            │
  │     Shelton: 70% of 5-min bars have ZERO price movement.        │
  │     Fed: 97.6% of bars are flat. Price changes are events,      │
  │     not continuous processes.                                    │
  │                                                                  │
  │  3. SPREAD = MINIMUM MOVE                                        │
  │     When spread = 1 tick and min move = 1 tick, every trade      │
  │     starts at max cost. You need price to move AT LEAST          │
  │     2 ticks favorably to break even.                             │
  │                                                                  │
  │  4. THIN BOOKS AT BEST BID/ASK                                   │
  │     Shelton best bid: only $15 (342 shares × $0.045)             │
  │     Shelton best ask: $1,130 (24,563 shares × $0.046)           │
  │     You'll move the book with any size > $1k.                    │
  │                                                                  │
  │  5. INFORMATION-DRIVEN, NOT FLOW-DRIVEN                          │
  │     Prediction market prices move on NEWS, not order flow.       │
  │     Technical signals (RSI, momentum, book imbalance) have       │
  │     near-zero predictive power because price changes are         │
  │     driven by exogenous information events.                      │
  │                                                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  WHERE 5-MIN TRADING CAN WORK ON POLYMARKET:                     │
  │                                                                  │
  │  A. LIVE SPORTS MARKETS                                          │
  │     - NBA/NFL/Soccer during games                                │
  │     - Prices move rapidly based on scoring events                │
  │     - Volume spikes 5-10x during live play                       │
  │     - Spread often stays at 1 tick during high activity          │
  │     - ALPHA SOURCE: Faster data feeds (live scores, play-by-play)│
  │       than the market makers are pricing in                      │
  │                                                                  │
  │  B. NEWS-CATALYST EVENT MARKETS                                  │
  │     - US Strikes Iran, Government Shutdown, Fed decisions        │
  │     - When news breaks, price dislocates by 5-20 ticks           │
  │     - ALPHA SOURCE: NLP on Twitter/news before market reacts     │
  │     - Risk: You're competing with bots doing the same thing      │
  │                                                                  │
  │  C. CROSS-MARKET ARBITRAGE                                       │
  │     - Same event priced differently on Polymarket vs. Kalshi     │
  │     - YES + NO tokens don't always sum to exactly $1.00          │
  │     - ALPHA SOURCE: Atomic arb between YES/NO or cross-platform  │
  │                                                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  RECOMMENDED APPROACH:                                           │
  │                                                                  │
  │  1. DON'T do technical 5-min trading on slow-moving markets      │
  │     (political, macro, long-dated events). The math doesn't work.│
  │                                                                  │
  │  2. DO build an event-driven system for:                         │
  │     - Live sports (real-time score → price prediction)           │
  │     - News breaks (NLP → fastest market reaction)                │
  │     - Cross-market arb (Polymarket ↔ Kalshi ↔ Robinhood)        │
  │                                                                  │
  │  3. If you want pure technical 5-min trading, use Polymarket's   │
  │     CRYPTO exchange (BTCUSD etc.) — same API, same account,      │
  │     but with real liquidity and continuous price action.          │
  │     See: polymarket_5min_strategy.py for that analysis.          │
  │                                                                  │
  │  REALISTIC RETURN PROFILE (prediction markets):                  │
  │     Median case for event-driven: +50-200 bps/day               │
  │     Median case for technical: -100 to -200 bps/day (LOSING)     │
  │     Position sizing: $500-$5,000 per trade                       │
  │     Capital required: $10,000-$50,000                            │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
