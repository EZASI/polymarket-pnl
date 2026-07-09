"""
BTC Mean-Reversion Strategy — Bollinger Bands + RSI
====================================================

Entry logic:
  LONG:  Price touches lower Bollinger Band AND RSI < oversold threshold
  SHORT: Price touches upper Bollinger Band AND RSI > overbought threshold

Exit logic:
  Close when price returns to the SMA (Bollinger midline), or after max_hold bars.

This is a reference strategy for validating the backtest pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from strategies.base import BaseStrategy, Candle, Direction, Signal


@dataclass
class BollingerRSIParams:
    bb_period: int = 20         # Bollinger Band lookback
    bb_std_mult: float = 2.0    # Standard deviation multiplier
    rsi_period: int = 14        # RSI lookback
    rsi_oversold: float = 30.0  # RSI oversold threshold
    rsi_overbought: float = 70.0
    max_hold_bars: int = 60     # Force exit after N bars (1 hour at 1m)
    min_confidence: float = 0.5


class BtcMeanReversion(BaseStrategy):
    """Bollinger Band + RSI mean-reversion on BTC/USDT 1m."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = {**BollingerRSIParams().__dict__, **(params or {})}
        self.p = BollingerRSIParams(**p)

    @property
    def name(self) -> str:
        return "btc_mean_reversion"

    @property
    def default_params(self) -> dict[str, Any]:
        return BollingerRSIParams().__dict__

    def warmup_periods(self) -> int:
        return max(self.p.bb_period, self.p.rsi_period) + 1

    def compute_signal(self, candles: list[Candle], index: int) -> Signal:
        if index < self.warmup_periods():
            return Signal(direction=Direction.FLAT, confidence=0.0)

        # Only extract the window we need (avoid O(n) copy each bar)
        bb_start = index - self.p.bb_period + 1
        bb_slice = [candles[j].close for j in range(bb_start, index + 1)]
        sma = sum(bb_slice) / len(bb_slice)
        std = math.sqrt(sum((x - sma) ** 2 for x in bb_slice) / len(bb_slice))
        upper_band = sma + self.p.bb_std_mult * std
        lower_band = sma - self.p.bb_std_mult * std

        # RSI — only need rsi_period + 1 closes
        rsi_start = max(0, index - self.p.rsi_period)
        rsi_closes = [candles[j].close for j in range(rsi_start, index + 1)]
        rsi = self._compute_rsi(rsi_closes, self.p.rsi_period)

        price = candles[index].close

        # Long: price at or below lower band + RSI oversold
        if price <= lower_band and rsi <= self.p.rsi_oversold:
            # Confidence scales with how deep into oversold territory
            depth = (self.p.rsi_oversold - rsi) / self.p.rsi_oversold
            conf = min(1.0, self.p.min_confidence + depth * 0.5)
            return Signal(
                direction=Direction.LONG,
                confidence=conf,
                metadata={"rsi": rsi, "bb_lower": lower_band, "sma": sma},
            )

        # Short: price at or above upper band + RSI overbought
        if price >= upper_band and rsi >= self.p.rsi_overbought:
            depth = (rsi - self.p.rsi_overbought) / (100 - self.p.rsi_overbought)
            conf = min(1.0, self.p.min_confidence + depth * 0.5)
            return Signal(
                direction=Direction.SHORT,
                confidence=conf,
                metadata={"rsi": rsi, "bb_upper": upper_band, "sma": sma},
            )

        return Signal(
            direction=Direction.FLAT,
            confidence=0.0,
            metadata={"rsi": rsi, "sma": sma, "upper": upper_band, "lower": lower_band},
        )

    @staticmethod
    def _compute_rsi(closes: list[float], period: int) -> float:
        """Compute Wilder's RSI."""
        if len(closes) < period + 1:
            return 50.0  # neutral

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]

        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]

        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
