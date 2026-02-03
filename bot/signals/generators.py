"""
Signal generators based on academically validated methods.

Each signal includes:
- Academic citation and validation level
- Signal strength calculation
- Confidence estimation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
import numpy as np

from ..config import (
    OBIConfig, LeadLagConfig, FundingRateConfig,
    SentimentConfig, LiquidationConfig, SignalType
)
from ..data.fetchers import (
    OrderBook, OHLCV, FundingRate, SentimentData, LiquidationEvent
)


@dataclass
class Signal:
    """Base signal output."""
    signal_type: SignalType
    timestamp: datetime
    direction: int  # -1 (bearish), 0 (neutral), 1 (bullish)
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    metadata: Dict = None

    def weighted_signal(self) -> float:
        """Return direction * strength * confidence."""
        return self.direction * self.strength * self.confidence


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""

    @abstractmethod
    def update(self, data: any) -> Optional[Signal]:
        """Update with new data and optionally emit signal."""
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state."""
        pass


class OBISignalGenerator(SignalGenerator):
    """
    Order Book Imbalance signal generator.

    Academic basis:
    - Cont, Kukanov, Stoikov (2014): R² ~65% for price changes
    - Wang (2025): 72% accuracy at 500ms horizons
    - Brazilian BMF&Bovespa study: ±0.6 threshold validated
    """

    def __init__(self, config: OBIConfig):
        self.config = config
        self._history: deque = deque(maxlen=100)
        self._last_signal: Optional[Signal] = None

    def update(self, order_book: OrderBook) -> Optional[Signal]:
        """
        Update with new order book and generate signal if threshold crossed.
        """
        if not self.config.enabled:
            return None

        imbalance = order_book.imbalance(levels=self.config.levels)
        self._history.append({
            'timestamp': order_book.timestamp,
            'imbalance': imbalance,
            'mid_price': order_book.mid_price(),
        })

        # Only signal when threshold is crossed
        if abs(imbalance) < self.config.threshold:
            return Signal(
                signal_type=SignalType.OBI,
                timestamp=order_book.timestamp,
                direction=0,
                strength=abs(imbalance),
                confidence=0.5,  # Below threshold = low confidence
                metadata={'imbalance': imbalance, 'levels': self.config.levels}
            )

        # Strong signal above threshold
        direction = 1 if imbalance > 0 else -1

        # Confidence based on how far above threshold
        excess = abs(imbalance) - self.config.threshold
        confidence = min(0.95, 0.72 + excess)  # Base 72% from Wang (2025)

        # Strength is the imbalance magnitude
        strength = min(1.0, abs(imbalance))

        signal = Signal(
            signal_type=SignalType.OBI,
            timestamp=order_book.timestamp,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'imbalance': imbalance,
                'levels': self.config.levels,
                'threshold': self.config.threshold,
            }
        )

        self._last_signal = signal
        return signal

    def reset(self):
        """Reset internal state."""
        self._history.clear()
        self._last_signal = None


class LeadLagSignalGenerator(SignalGenerator):
    """
    Lead-lag signal generator using Lévy Area approximation.

    Academic basis:
    - Cartea, Cucuringu, Jin (2023): >20% annualized returns
    - Schei (2019): 70% directional accuracy, up to 15s lead
    - Bitwise SEC filing: 6.5-17s lead from futures to spot

    The Lévy Area captures nonlinear lead-lag relationships that
    correlation-based methods miss.
    """

    def __init__(self, config: LeadLagConfig):
        self.config = config
        self._price_history: Dict[str, deque] = {}
        self._levy_areas: deque = deque(maxlen=50)

        # Initialize history for each exchange
        for exchange in config.reference_exchanges:
            self._price_history[exchange] = deque(maxlen=100)

    def update(
        self,
        prices: Dict[str, Tuple[datetime, float]],
        target_price: Tuple[datetime, float] = None,
    ) -> Optional[Signal]:
        """
        Update with new prices from multiple exchanges.

        prices: Dict mapping exchange name to (timestamp, price)
        target_price: The target market (e.g., Polymarket) price
        """
        if not self.config.enabled:
            return None

        # Store prices
        for exchange, (timestamp, price) in prices.items():
            if exchange in self._price_history:
                self._price_history[exchange].append({
                    'timestamp': timestamp,
                    'price': price,
                })

        # Need sufficient history
        min_history = 10
        if any(len(h) < min_history for h in self._price_history.values()):
            return None

        # Calculate Lévy Area between leading exchange and others
        levy_area = self._calculate_levy_area()

        if levy_area is None:
            return None

        self._levy_areas.append(levy_area)

        # Signal based on Lévy Area sign and magnitude
        if abs(levy_area) < self.config.levy_area_threshold:
            return Signal(
                signal_type=SignalType.LEAD_LAG,
                timestamp=datetime.utcnow(),
                direction=0,
                strength=abs(levy_area),
                confidence=0.5,
                metadata={'levy_area': levy_area}
            )

        # Positive Lévy Area indicates first series leads
        # (price going up on leading exchange before lagging)
        direction = 1 if levy_area > 0 else -1

        # Confidence based on consistency of recent Lévy Areas
        if len(self._levy_areas) >= 5:
            recent = list(self._levy_areas)[-5:]
            consistency = sum(1 for la in recent if (la > 0) == (levy_area > 0)) / 5
            confidence = min(0.95, 0.70 * consistency)  # Base 70% from Schei
        else:
            confidence = 0.60

        strength = min(1.0, abs(levy_area) / self.config.levy_area_threshold)

        return Signal(
            signal_type=SignalType.LEAD_LAG,
            timestamp=datetime.utcnow(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'levy_area': levy_area,
                'leading_exchange': self.config.reference_exchanges[0],
            }
        )

    def _calculate_levy_area(self) -> Optional[float]:
        """
        Calculate Lévy Area between two price series.

        Lévy Area = 0.5 * ∫(X dY - Y dX)

        This is approximated using the trapezoidal rule.
        """
        if len(self.config.reference_exchanges) < 2:
            return None

        # Get two series
        series1 = list(self._price_history[self.config.reference_exchanges[0]])
        series2 = list(self._price_history[self.config.reference_exchanges[1]])

        if len(series1) < 5 or len(series2) < 5:
            return None

        # Normalize prices
        x = np.array([s['price'] for s in series1[-20:]])
        y = np.array([s['price'] for s in series2[-20:]])

        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[-min_len:]
            y = y[-min_len:]

        # Standardize
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        # Calculate Lévy Area using trapezoidal approximation
        # A = 0.5 * Σ(x_i * (y_{i+1} - y_{i-1}) - y_i * (x_{i+1} - x_{i-1}))
        levy_area = 0.0
        for i in range(1, len(x) - 1):
            levy_area += 0.5 * (
                x[i] * (y[i+1] - y[i-1]) -
                y[i] * (x[i+1] - x[i-1])
            )

        return levy_area / len(x)  # Normalize by length

    def reset(self):
        """Reset internal state."""
        for exchange in self._price_history:
            self._price_history[exchange].clear()
        self._levy_areas.clear()


class FundingRateSignalGenerator(SignalGenerator):
    """
    Funding rate signal generator.

    Academic basis:
    - He et al.: Sharpe ratios 1.8-3.5 for deviation strategies
    - Presto Research: R² ~0 for single-asset prediction
    - Use as contrarian indicator, NOT direct predictor

    IMPORTANT: Funding rates have near-zero predictive power for
    future price movements of a single asset. Use only for:
    - Regime detection (extreme funding = crowded trade)
    - Contrarian signal (extreme = potential reversal setup)
    """

    def __init__(self, config: FundingRateConfig):
        self.config = config
        self._history: deque = deque(maxlen=100)
        self._extreme_count: int = 0

    def update(self, funding_rate: FundingRate) -> Optional[Signal]:
        """
        Update with new funding rate data.
        """
        if not self.config.enabled:
            return None

        self._history.append({
            'timestamp': funding_rate.timestamp,
            'rate': funding_rate.rate,
            'annualized': funding_rate.annualized_rate,
        })

        # Check for extreme funding
        is_extreme = abs(funding_rate.annualized_rate) > self.config.extreme_threshold_annualized

        if is_extreme:
            self._extreme_count += 1
        else:
            self._extreme_count = max(0, self._extreme_count - 1)

        # Only signal on extreme funding (contrarian)
        if not is_extreme:
            return Signal(
                signal_type=SignalType.FUNDING_RATE,
                timestamp=funding_rate.timestamp,
                direction=0,
                strength=abs(funding_rate.annualized_rate) / self.config.extreme_threshold_annualized,
                confidence=0.3,  # Low confidence for non-extreme
                metadata={
                    'rate': funding_rate.rate,
                    'annualized': funding_rate.annualized_rate,
                    'is_extreme': False,
                }
            )

        # Contrarian signal: extreme positive funding = bearish setup
        # (crowded long, potential liquidation cascade)
        if self.config.use_as_contrarian:
            direction = -1 if funding_rate.annualized_rate > 0 else 1
        else:
            direction = 1 if funding_rate.annualized_rate > 0 else -1

        # Confidence is LOW because extreme funding doesn't reliably predict
        # reversals (2021 bull run had 109%+ for extended periods)
        confidence = min(0.6, 0.4 + 0.1 * self._extreme_count)

        # Strength based on how extreme
        strength = min(1.0, abs(funding_rate.annualized_rate) / (2 * self.config.extreme_threshold_annualized))

        return Signal(
            signal_type=SignalType.FUNDING_RATE,
            timestamp=funding_rate.timestamp,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'rate': funding_rate.rate,
                'annualized': funding_rate.annualized_rate,
                'is_extreme': True,
                'extreme_count': self._extreme_count,
                'warning': 'Extreme funding does NOT guarantee reversal',
            }
        )

    def reset(self):
        """Reset internal state."""
        self._history.clear()
        self._extreme_count = 0


class SentimentSignalGenerator(SignalGenerator):
    """
    Social sentiment signal generator.

    Academic basis:
    - Renault et al.: Significant relationship up to 15 minutes
    - Jeleskovic & Mackay: 15-minute impact, 3-minute lag
    - Effect magnitude is SMALL - not profitable alone

    IMPORTANT: Academic studies note the effect is "too small for
    economic profits" - use only as supplementary signal.
    """

    def __init__(self, config: SentimentConfig):
        self.config = config
        self._history: deque = deque(maxlen=100)
        self._sentiment_ma: float = 0.0

    def update(self, sentiment: SentimentData) -> Optional[Signal]:
        """
        Update with new sentiment data.
        """
        if not self.config.enabled:
            return None

        self._history.append({
            'timestamp': sentiment.timestamp,
            'score': sentiment.sentiment_score,
            'volume': sentiment.volume,
        })

        # Update moving average
        alpha = 0.1
        self._sentiment_ma = alpha * sentiment.sentiment_score + (1 - alpha) * self._sentiment_ma

        # Filter old data (only use last 15 minutes)
        cutoff = datetime.utcnow() - timedelta(minutes=self.config.lookback_minutes)
        recent = [h for h in self._history if h['timestamp'] > cutoff]

        if len(recent) < 3:
            return None

        # Calculate weighted sentiment (volume-weighted)
        total_volume = sum(h['volume'] for h in recent)
        if total_volume == 0:
            return None

        weighted_sentiment = sum(
            h['score'] * h['volume'] for h in recent
        ) / total_volume

        # Direction based on sentiment
        if abs(weighted_sentiment) < 0.1:
            direction = 0
        else:
            direction = 1 if weighted_sentiment > 0 else -1

        # Confidence is LOW (academic finding: effect too small for profits)
        # Slightly higher confidence with higher volume
        volume_factor = min(1.5, total_volume / 1000)
        confidence = min(0.5, 0.3 * volume_factor)  # Cap at 50%

        # Strength based on sentiment magnitude
        strength = min(1.0, abs(weighted_sentiment))

        return Signal(
            signal_type=SignalType.SENTIMENT,
            timestamp=sentiment.timestamp,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'weighted_sentiment': weighted_sentiment,
                'raw_sentiment': sentiment.sentiment_score,
                'volume': sentiment.volume,
                'warning': 'Effect magnitude too small for standalone profits',
            }
        )

    def reset(self):
        """Reset internal state."""
        self._history.clear()
        self._sentiment_ma = 0.0


class LiquidationSignalGenerator(SignalGenerator):
    """
    Liquidation cascade signal generator.

    Academic basis:
    - Cheng et al. (2021): 3.51% daily forced liquidations, 60X leverage
    - CMU team: Derivatives exacerbate large moves
    - Ali (2025): 86x acceleration during cascades

    Use as CONTRARIAN indicator - large liquidations signal
    position removal, not future direction.
    """

    def __init__(self, config: LiquidationConfig):
        self.config = config
        self._history: deque = deque(maxlen=1000)
        self._cascade_detected: bool = False
        self._cascade_start: Optional[datetime] = None

    def update(self, liquidations: List[LiquidationEvent]) -> Optional[Signal]:
        """
        Update with new liquidation events.
        """
        if not self.config.enabled:
            return None

        # Store liquidations
        for liq in liquidations:
            self._history.append(liq)

        # Calculate recent liquidation volume
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent = [l for l in self._history if l.timestamp > cutoff]

        if not recent:
            return None

        total_value = sum(l.value_usd for l in recent)
        long_value = sum(l.value_usd for l in recent if l.side == 'long')
        short_value = sum(l.value_usd for l in recent if l.side == 'short')

        # Detect cascade
        is_cascade = total_value > self.config.cascade_threshold_usd

        if is_cascade and not self._cascade_detected:
            self._cascade_detected = True
            self._cascade_start = datetime.utcnow()
        elif not is_cascade:
            self._cascade_detected = False
            self._cascade_start = None

        # Determine dominant side
        if long_value > short_value * 1.5:
            dominant_side = 'long'
            # Longs being liquidated = downward pressure
            # Contrarian: potential bounce setup
            direction = 1 if self.config.use_as_contrarian else -1
        elif short_value > long_value * 1.5:
            dominant_side = 'short'
            # Shorts being liquidated = upward pressure
            # Contrarian: potential pullback setup
            direction = -1 if self.config.use_as_contrarian else 1
        else:
            dominant_side = 'mixed'
            direction = 0

        # Confidence based on cascade magnitude
        if is_cascade:
            magnitude = total_value / self.config.cascade_threshold_usd
            confidence = min(0.7, 0.4 + 0.1 * magnitude)
        else:
            confidence = 0.3

        # Strength based on value
        strength = min(1.0, total_value / (2 * self.config.cascade_threshold_usd))

        return Signal(
            signal_type=SignalType.LIQUIDATION,
            timestamp=datetime.utcnow(),
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'total_value_usd': total_value,
                'long_value_usd': long_value,
                'short_value_usd': short_value,
                'dominant_side': dominant_side,
                'is_cascade': is_cascade,
                'cascade_duration': (
                    (datetime.utcnow() - self._cascade_start).seconds
                    if self._cascade_start else 0
                ),
            }
        )

    def reset(self):
        """Reset internal state."""
        self._history.clear()
        self._cascade_detected = False
        self._cascade_start = None


class SignalAggregator:
    """
    Aggregates multiple signals into a combined signal.

    Weights signals based on:
    - Configured weights (from academic validation strength)
    - Signal confidence
    - Recent signal accuracy (adaptive)
    """

    def __init__(self, signal_weights: Dict[SignalType, float] = None):
        self.signal_weights = signal_weights or {
            SignalType.OBI: 0.25,
            SignalType.LEAD_LAG: 0.25,
            SignalType.FUNDING_RATE: 0.15,
            SignalType.SENTIMENT: 0.10,
            SignalType.LIQUIDATION: 0.10,
        }
        self._signal_accuracy: Dict[SignalType, deque] = {
            st: deque(maxlen=100) for st in SignalType
        }

    def aggregate(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Aggregate multiple signals into a combined signal.
        """
        if not signals:
            return None

        # Filter out None signals
        valid_signals = [s for s in signals if s is not None]
        if not valid_signals:
            return None

        # Calculate weighted combination
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_strength = 0.0
        max_confidence = 0.0

        for signal in valid_signals:
            weight = self.signal_weights.get(signal.signal_type, 0.1)

            # Adjust weight by confidence
            adjusted_weight = weight * signal.confidence

            # Adjust by historical accuracy
            accuracy_history = self._signal_accuracy.get(signal.signal_type)
            if accuracy_history and len(accuracy_history) > 10:
                recent_accuracy = sum(accuracy_history) / len(accuracy_history)
                adjusted_weight *= (0.5 + recent_accuracy)

            total_weight += adjusted_weight
            weighted_direction += signal.direction * adjusted_weight
            weighted_strength += signal.strength * adjusted_weight
            max_confidence = max(max_confidence, signal.confidence)

        if total_weight == 0:
            return None

        # Normalize
        final_direction = weighted_direction / total_weight
        final_strength = weighted_strength / total_weight

        # Discretize direction
        if final_direction > 0.3:
            direction = 1
        elif final_direction < -0.3:
            direction = -1
        else:
            direction = 0

        # Combined confidence (lower when signals disagree)
        direction_agreement = abs(final_direction)
        combined_confidence = max_confidence * direction_agreement

        return Signal(
            signal_type=SignalType.OBI,  # Use as placeholder for "combined"
            timestamp=datetime.utcnow(),
            direction=direction,
            strength=final_strength,
            confidence=combined_confidence,
            metadata={
                'source_signals': len(valid_signals),
                'raw_direction': final_direction,
                'direction_agreement': direction_agreement,
            }
        )

    def record_outcome(self, signal_type: SignalType, was_correct: bool):
        """Record signal outcome for adaptive weighting."""
        if signal_type in self._signal_accuracy:
            self._signal_accuracy[signal_type].append(1.0 if was_correct else 0.0)
