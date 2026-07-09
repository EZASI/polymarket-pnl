"""
Base strategy interface and shared data structures.

Every strategy implements BaseStrategy and can be:
  - Backtested via research/backtester.py
  - Evaluated via research/alpha_evaluator.py
  - Registered in config/strategies.yaml for live trading
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Signal:
    """Output of a strategy's compute_signal()."""
    direction: Direction
    confidence: float          # 0.0–1.0
    sizing_hint: float = 1.0   # multiplier on base position size
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """A completed trade (entry + exit)."""
    entry_ts: datetime
    exit_ts: datetime
    direction: Direction
    entry_price: float
    exit_price: float
    size: float               # in base currency units
    pnl: float                # absolute PnL
    pnl_pct: float            # return percentage
    fee: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregated backtest metrics."""
    strategy_name: str
    symbol: str
    interval: str
    period_start: datetime
    period_end: datetime
    n_trades: int
    win_rate: float            # fraction of winning trades
    pnl_total: float
    pnl_per_trade: float
    sharpe: float              # annualized Sharpe ratio
    max_drawdown: float        # max peak-to-trough drawdown (fraction)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candle:
    """Single OHLCV candle."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades_count: int = 0


class BaseStrategy(ABC):
    """
    Abstract base for all strategies.

    Subclass and implement:
      - name (property)
      - compute_signal(candles, index) -> Signal
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def default_params(self) -> dict[str, Any]:
        """Default parameters. Override in subclass."""
        return {}

    @abstractmethod
    def compute_signal(self, candles: list[Candle], index: int) -> Signal:
        """
        Compute trading signal at position `index` in the candle series.
        `candles[:index+1]` are visible (no look-ahead).
        """
        ...

    def warmup_periods(self) -> int:
        """Number of candles needed before first valid signal."""
        return 0
