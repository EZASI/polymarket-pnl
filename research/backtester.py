"""
Backtest Engine
===============

Replays OHLCV data from TimescaleDB through any BaseStrategy subclass.
Tracks positions, PnL, equity curve, and computes Sharpe / max drawdown.

Usage:
    from research.backtester import run_backtest
    from strategies.btc_mean_reversion import BtcMeanReversion

    result = await run_backtest(
        db=db,
        strategy=BtcMeanReversion(),
        symbol="BTCUSDT",
        exchange="binance",
        interval="1m",
        days=30,
    )
    print(result.sharpe, result.win_rate, result.max_drawdown)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from data_pipeline.db import Database
from strategies.base import (
    BacktestResult,
    BaseStrategy,
    Candle,
    Direction,
    Signal,
    Trade,
)

log = logging.getLogger(__name__)

# Trading cost assumptions (configurable)
DEFAULT_FEE_BPS = 2.0           # 0.02% per side (maker on Binance)
DEFAULT_POSITION_SIZE = 10_000  # USD per trade


async def run_backtest(
    db: Database,
    strategy: BaseStrategy,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    interval: str = "1m",
    days: int = 30,
    start: datetime | None = None,
    end: datetime | None = None,
    fee_bps: float = DEFAULT_FEE_BPS,
    position_size_usd: float = DEFAULT_POSITION_SIZE,
) -> BacktestResult:
    """
    Run a full backtest of `strategy` against OHLCV data from TimescaleDB.

    Returns a BacktestResult with all metrics.
    """
    # Load data
    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - timedelta(days=days)

    rows = await db.query_ohlcv(
        exchange=exchange,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        limit=days * 1440 + 100,  # 1m candles per day + buffer
    )

    if not rows:
        log.warning("No data for %s %s %s", exchange, symbol, interval)
        return _empty_result(strategy.name, symbol, interval, start, end)

    candles = [
        Candle(
            ts=r["ts"],
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
            quote_volume=float(r.get("quote_volume", 0)),
            trades_count=int(r.get("trades_count", 0)),
        )
        for r in rows
    ]

    log.info(
        "Backtesting %s on %s %s (%d candles, %s → %s)",
        strategy.name,
        symbol,
        interval,
        len(candles),
        candles[0].ts.isoformat(),
        candles[-1].ts.isoformat(),
    )

    # Run simulation
    trades, equity_curve = _simulate(
        strategy=strategy,
        candles=candles,
        fee_bps=fee_bps,
        position_size_usd=position_size_usd,
    )

    # Compute metrics
    result = _compute_metrics(
        strategy_name=strategy.name,
        symbol=symbol,
        interval=interval,
        candles=candles,
        trades=trades,
        equity_curve=equity_curve,
        params=strategy.default_params,
    )

    log.info(
        "Backtest complete: %d trades, Sharpe=%.2f, Win=%.1f%%, DD=%.2f%%, PnL=$%.2f",
        result.n_trades,
        result.sharpe,
        result.win_rate * 100,
        result.max_drawdown * 100,
        result.pnl_total,
    )

    return result


def _simulate(
    strategy: BaseStrategy,
    candles: list[Candle],
    fee_bps: float,
    position_size_usd: float,
) -> tuple[list[Trade], list[float]]:
    """Core simulation loop. No look-ahead."""
    trades: list[Trade] = []
    equity_curve: list[float] = [0.0]  # cumulative PnL

    # Position state
    in_position = False
    position_dir = Direction.FLAT
    entry_price = 0.0
    entry_ts: datetime | None = None
    entry_idx = 0
    target_exit_price = 0.0  # SMA at entry for mean-reversion exit

    warmup = strategy.warmup_periods()
    fee_mult = fee_bps / 10_000  # per-side fee as fraction

    for i in range(warmup, len(candles)):
        price = candles[i].close
        signal = strategy.compute_signal(candles, i)

        if not in_position:
            # Check for entry
            if signal.direction in (Direction.LONG, Direction.SHORT):
                in_position = True
                position_dir = signal.direction
                entry_price = price
                entry_ts = candles[i].ts
                entry_idx = i
                # Store SMA as target exit (from signal metadata)
                target_exit_price = signal.metadata.get("sma", price)

        else:
            # Check for exit
            should_exit = False

            # Exit condition 1: price returns to SMA (mean reversion target)
            if position_dir == Direction.LONG and price >= target_exit_price:
                should_exit = True
            elif position_dir == Direction.SHORT and price <= target_exit_price:
                should_exit = True

            # Exit condition 2: max hold period
            max_hold = getattr(strategy, "p", None)
            max_bars = max_hold.max_hold_bars if max_hold and hasattr(max_hold, "max_hold_bars") else 60
            if (i - entry_idx) >= max_bars:
                should_exit = True

            # Exit condition 3: opposing signal
            if signal.direction != Direction.FLAT and signal.direction != position_dir:
                should_exit = True

            if should_exit:
                # Compute PnL
                size_units = position_size_usd / entry_price
                if position_dir == Direction.LONG:
                    raw_pnl = (price - entry_price) * size_units
                else:
                    raw_pnl = (entry_price - price) * size_units

                # Fees: entry + exit
                fee = position_size_usd * fee_mult * 2
                net_pnl = raw_pnl - fee

                trade = Trade(
                    entry_ts=entry_ts,
                    exit_ts=candles[i].ts,
                    direction=position_dir,
                    entry_price=entry_price,
                    exit_price=price,
                    size=size_units,
                    pnl=net_pnl,
                    pnl_pct=net_pnl / position_size_usd,
                    fee=fee,
                    metadata={
                        "hold_bars": i - entry_idx,
                        "target_exit": target_exit_price,
                    },
                )
                trades.append(trade)

                # Update equity curve
                cum_pnl = equity_curve[-1] + net_pnl
                equity_curve.append(cum_pnl)

                # Reset
                in_position = False
                position_dir = Direction.FLAT
                entry_price = 0.0
                entry_ts = None

    return trades, equity_curve


def _compute_metrics(
    strategy_name: str,
    symbol: str,
    interval: str,
    candles: list[Candle],
    trades: list[Trade],
    equity_curve: list[float],
    params: dict[str, Any],
) -> BacktestResult:
    """Compute Sharpe, max drawdown, win rate, etc."""
    n = len(trades)
    if n == 0:
        return _empty_result(
            strategy_name, symbol, interval,
            candles[0].ts, candles[-1].ts,
        )

    wins = sum(1 for t in trades if t.pnl > 0)
    pnl_total = sum(t.pnl for t in trades)
    pnl_per_trade = pnl_total / n

    # Sharpe ratio (annualized)
    returns = [t.pnl_pct for t in trades]
    avg_ret = sum(returns) / len(returns)
    if len(returns) > 1:
        std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1))
    else:
        std_ret = 0.0

    # Annualization factor: estimate trades per year from data
    data_days = (candles[-1].ts - candles[0].ts).total_seconds() / 86400
    if data_days > 0 and n > 0:
        trades_per_year = (n / data_days) * 365
    else:
        trades_per_year = 252

    if std_ret > 0:
        sharpe = (avg_ret / std_ret) * math.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    max_dd = _max_drawdown(equity_curve)

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        period_start=candles[0].ts,
        period_end=candles[-1].ts,
        n_trades=n,
        win_rate=wins / n if n > 0 else 0.0,
        pnl_total=pnl_total,
        pnl_per_trade=pnl_per_trade,
        sharpe=sharpe,
        max_drawdown=max_dd,
        trades=trades,
        equity_curve=equity_curve,
        params=params,
    )


def _max_drawdown(equity_curve: list[float]) -> float:
    """Compute maximum peak-to-trough drawdown as a fraction of peak equity."""
    if not equity_curve:
        return 0.0

    # Shift to make all values positive (add initial capital)
    capital = 100_000  # notional starting capital
    values = [capital + e for e in equity_curve]

    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _empty_result(
    name: str, symbol: str, interval: str,
    start: datetime, end: datetime,
) -> BacktestResult:
    return BacktestResult(
        strategy_name=name,
        symbol=symbol,
        interval=interval,
        period_start=start,
        period_end=end,
        n_trades=0,
        win_rate=0.0,
        pnl_total=0.0,
        pnl_per_trade=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
    )


async def save_backtest_result(db: Database, result: BacktestResult) -> int:
    """Persist backtest result to TimescaleDB. Returns the row ID."""
    import json

    sql = """
        INSERT INTO backtest_results
            (strategy_name, underlying, period_start, period_end,
             n_trades, hit_rate, sharpe, max_drawdown, pnl_total, pnl_per_trade,
             params_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """
    async with db.acquire() as conn:
        row_id = await conn.fetchval(
            sql,
            result.strategy_name,
            result.symbol,
            result.period_start,
            result.period_end,
            result.n_trades,
            result.win_rate,
            result.sharpe,
            result.max_drawdown,
            result.pnl_total,
            result.pnl_per_trade,
            json.dumps(result.params),
        )
    return row_id
