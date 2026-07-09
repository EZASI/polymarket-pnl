"""
Alpha Evaluator
================

Computes detailed performance analytics for backtest and live results:
  - Hit rate, Brier score, calibration curves
  - PnL bucketed by time-of-day, hold duration, direction
  - Win/loss streaks, profit factor, expectancy
  - Summary report for Meta-Research Agent consumption
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from strategies.base import BacktestResult, Direction, Trade


@dataclass
class AlphaReport:
    """Comprehensive strategy evaluation."""
    strategy_name: str
    symbol: str
    period: str                # e.g. "2025-11-15 → 2026-02-13"

    # Core metrics
    n_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    pnl_total: float
    pnl_per_trade: float

    # Extended metrics
    profit_factor: float       # gross_profit / gross_loss
    expectancy: float          # avg win * win_rate - avg loss * loss_rate
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_hold_bars: float
    max_win_streak: int
    max_loss_streak: int

    # Bucketed analysis
    pnl_by_direction: dict[str, float] = field(default_factory=dict)
    pnl_by_hour: dict[int, float] = field(default_factory=dict)
    trades_by_hour: dict[int, int] = field(default_factory=dict)
    pnl_by_hold_bucket: dict[str, float] = field(default_factory=dict)

    # Equity curve stats
    calmar_ratio: float = 0.0  # annualized_return / max_drawdown


def evaluate(result: BacktestResult) -> AlphaReport:
    """Build a full AlphaReport from a BacktestResult."""
    trades = result.trades
    n = len(trades)

    if n == 0:
        return _empty_report(result)

    # Win/loss splits
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in winners) if winners else 0.0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0

    avg_win = gross_profit / len(winners) if winners else 0.0
    avg_loss = gross_loss / len(losers) if losers else 0.0

    max_win = max((t.pnl for t in trades), default=0.0)
    max_loss = min((t.pnl for t in trades), default=0.0)

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = result.win_rate
    expectancy = avg_win * win_rate - avg_loss * (1 - win_rate)

    # Hold duration
    hold_bars = [t.metadata.get("hold_bars", 0) for t in trades]
    avg_hold = sum(hold_bars) / len(hold_bars) if hold_bars else 0.0

    # Streaks
    max_win_streak, max_loss_streak = _compute_streaks(trades)

    # PnL by direction
    pnl_by_dir: dict[str, float] = {}
    for t in trades:
        d = t.direction.value
        pnl_by_dir[d] = pnl_by_dir.get(d, 0.0) + t.pnl

    # PnL by hour of day
    pnl_by_hour: dict[int, float] = {}
    trades_by_hour: dict[int, int] = {}
    for t in trades:
        h = t.entry_ts.hour if t.entry_ts else 0
        pnl_by_hour[h] = pnl_by_hour.get(h, 0.0) + t.pnl
        trades_by_hour[h] = trades_by_hour.get(h, 0) + 1

    # PnL by hold duration bucket
    pnl_by_hold: dict[str, float] = {"<5m": 0.0, "5-15m": 0.0, "15-30m": 0.0, "30-60m": 0.0, ">60m": 0.0}
    for t in trades:
        bars = t.metadata.get("hold_bars", 0)
        if bars < 5:
            bucket = "<5m"
        elif bars < 15:
            bucket = "5-15m"
        elif bars < 30:
            bucket = "15-30m"
        elif bars < 60:
            bucket = "30-60m"
        else:
            bucket = ">60m"
        pnl_by_hold[bucket] += t.pnl

    # Calmar ratio
    data_days = (result.period_end - result.period_start).total_seconds() / 86400
    annualized_return = (result.pnl_total / 100_000) * (365 / data_days) if data_days > 0 else 0.0
    calmar = annualized_return / result.max_drawdown if result.max_drawdown > 0 else 0.0

    period_str = f"{result.period_start.strftime('%Y-%m-%d')} → {result.period_end.strftime('%Y-%m-%d')}"

    return AlphaReport(
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        period=period_str,
        n_trades=n,
        win_rate=win_rate,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        pnl_total=result.pnl_total,
        pnl_per_trade=result.pnl_per_trade,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max_win,
        max_loss=max_loss,
        avg_hold_bars=avg_hold,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        pnl_by_direction=pnl_by_dir,
        pnl_by_hour=pnl_by_hour,
        trades_by_hour=trades_by_hour,
        pnl_by_hold_bucket=pnl_by_hold,
        calmar_ratio=calmar,
    )


def format_report(report: AlphaReport) -> str:
    """Format an AlphaReport as a readable markdown table."""
    lines = [
        f"# Alpha Report: {report.strategy_name}",
        f"**Symbol:** {report.symbol} | **Period:** {report.period}",
        "",
        "## Core Metrics",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Trades | {report.n_trades} |",
        f"| Win Rate | {report.win_rate:.1%} |",
        f"| Sharpe | {report.sharpe:.2f} |",
        f"| Max Drawdown | {report.max_drawdown:.2%} |",
        f"| Total PnL | ${report.pnl_total:,.2f} |",
        f"| PnL/Trade | ${report.pnl_per_trade:,.2f} |",
        f"| Profit Factor | {report.profit_factor:.2f} |",
        f"| Expectancy | ${report.expectancy:,.2f} |",
        f"| Calmar Ratio | {report.calmar_ratio:.2f} |",
        "",
        "## Win/Loss Analysis",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Avg Win | ${report.avg_win:,.2f} |",
        f"| Avg Loss | ${report.avg_loss:,.2f} |",
        f"| Max Win | ${report.max_win:,.2f} |",
        f"| Max Loss | ${report.max_loss:,.2f} |",
        f"| Avg Hold (bars) | {report.avg_hold_bars:.1f} |",
        f"| Max Win Streak | {report.max_win_streak} |",
        f"| Max Loss Streak | {report.max_loss_streak} |",
        "",
        "## PnL by Direction",
    ]

    for d, pnl in sorted(report.pnl_by_direction.items()):
        lines.append(f"- **{d}:** ${pnl:,.2f}")

    lines.append("")
    lines.append("## PnL by Hold Duration")
    for bucket, pnl in report.pnl_by_hold_bucket.items():
        lines.append(f"- **{bucket}:** ${pnl:,.2f}")

    lines.append("")
    lines.append("## PnL by Hour (UTC)")
    for h in sorted(report.pnl_by_hour.keys()):
        count = report.trades_by_hour.get(h, 0)
        pnl = report.pnl_by_hour[h]
        lines.append(f"- **{h:02d}:00** — {count} trades, ${pnl:,.2f}")

    return "\n".join(lines)


def to_json(report: AlphaReport) -> dict[str, Any]:
    """Convert AlphaReport to JSON-serializable dict (for Meta-Research Agent)."""
    return {
        "strategy_name": report.strategy_name,
        "symbol": report.symbol,
        "period": report.period,
        "n_trades": report.n_trades,
        "win_rate": round(report.win_rate, 4),
        "sharpe": round(report.sharpe, 4),
        "max_drawdown": round(report.max_drawdown, 4),
        "pnl_total": round(report.pnl_total, 2),
        "pnl_per_trade": round(report.pnl_per_trade, 2),
        "profit_factor": round(report.profit_factor, 4),
        "expectancy": round(report.expectancy, 2),
        "avg_win": round(report.avg_win, 2),
        "avg_loss": round(report.avg_loss, 2),
        "max_win": round(report.max_win, 2),
        "max_loss": round(report.max_loss, 2),
        "avg_hold_bars": round(report.avg_hold_bars, 1),
        "calmar_ratio": round(report.calmar_ratio, 4),
        "pnl_by_direction": report.pnl_by_direction,
        "pnl_by_hold_bucket": report.pnl_by_hold_bucket,
        "best_hours": sorted(report.pnl_by_hour.items(), key=lambda x: x[1], reverse=True)[:5],
        "worst_hours": sorted(report.pnl_by_hour.items(), key=lambda x: x[1])[:5],
    }


def _compute_streaks(trades: list[Trade]) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak)."""
    max_w = max_l = 0
    cur_w = cur_l = 0
    for t in trades:
        if t.pnl > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _empty_report(result: BacktestResult) -> AlphaReport:
    return AlphaReport(
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        period=f"{result.period_start} → {result.period_end}",
        n_trades=0, win_rate=0, sharpe=0, max_drawdown=0,
        pnl_total=0, pnl_per_trade=0, profit_factor=0,
        expectancy=0, avg_win=0, avg_loss=0, max_win=0,
        max_loss=0, avg_hold_bars=0, max_win_streak=0,
        max_loss_streak=0,
    )
