"""
Strategy engine combining signals and ML predictions.

Implements trading logic with risk management based on academic findings.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from .config import BotConfig, RiskConfig, SignalType
from .signals.generators import (
    Signal, SignalAggregator,
    OBISignalGenerator, LeadLagSignalGenerator,
    FundingRateSignalGenerator, SentimentSignalGenerator,
    LiquidationSignalGenerator
)
from .models.ml_models import EnsemblePredictor, FeatureEngineering, Prediction


class TradeAction(Enum):
    """Possible trade actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeSignal:
    """Output from strategy engine."""
    timestamp: datetime
    action: TradeAction
    market_id: str
    outcome: str  # "Yes" or "No"
    size_usd: float
    confidence: float
    edge_estimate: float  # Estimated edge over market
    signals_used: List[str]
    ml_prediction: Optional[Prediction]
    reasons: List[str]


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    outcome: str
    entry_price: float
    size_usd: float
    entry_time: datetime
    unrealized_pnl: float = 0.0


@dataclass
class StrategyState:
    """Current state of the strategy."""
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_trade_time: Optional[datetime] = None
    total_capital: float = 10000.0
    available_capital: float = 10000.0


class PolymarketStrategy:
    """
    Main strategy engine for Polymarket crypto prediction.

    Combines:
    - Order Book Imbalance (OBI) signals
    - Lead-lag relationships
    - Funding rate regime detection
    - Sentiment signals
    - Liquidation cascade detection
    - ML ensemble predictions

    Risk management:
    - Position sizing based on edge and confidence
    - Avoids 45-55% probability "death zone"
    - Daily loss limits
    - Maximum position count
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.risk_config = config.risk

        # Initialize signal generators
        self.obi_generator = OBISignalGenerator(config.obi)
        self.lead_lag_generator = LeadLagSignalGenerator(config.lead_lag)
        self.funding_generator = FundingRateSignalGenerator(config.funding_rate)
        self.sentiment_generator = SentimentSignalGenerator(config.sentiment)
        self.liquidation_generator = LiquidationSignalGenerator(config.liquidation)

        # Signal aggregator with weights from academic validation
        self.signal_aggregator = SignalAggregator({
            SignalType.OBI: config.obi.weight,
            SignalType.LEAD_LAG: config.lead_lag.weight,
            SignalType.FUNDING_RATE: config.funding_rate.weight,
            SignalType.SENTIMENT: config.sentiment.weight,
            SignalType.LIQUIDATION: config.liquidation.weight,
        })

        # ML ensemble predictor
        self.ml_predictor = EnsemblePredictor()
        self._ml_trained = False

        # Feature history for ML
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._imbalance_history: List[float] = []
        self._funding_history: List[float] = []
        self._sentiment_history: List[float] = []

        # State
        self.state = StrategyState(total_capital=config.backtest.initial_capital)
        self.state.available_capital = self.state.total_capital

    def update_data(
        self,
        price: float,
        volume: float = None,
        imbalance: float = None,
        funding_rate: float = None,
        sentiment: float = None,
    ):
        """Update historical data for feature engineering."""
        self._price_history.append(price)
        if volume is not None:
            self._volume_history.append(volume)
        if imbalance is not None:
            self._imbalance_history.append(imbalance)
        if funding_rate is not None:
            self._funding_history.append(funding_rate)
        if sentiment is not None:
            self._sentiment_history.append(sentiment)

        # Keep limited history
        max_history = 500
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]
        if len(self._volume_history) > max_history:
            self._volume_history = self._volume_history[-max_history:]
        if len(self._imbalance_history) > max_history:
            self._imbalance_history = self._imbalance_history[-max_history:]
        if len(self._funding_history) > max_history:
            self._funding_history = self._funding_history[-max_history:]
        if len(self._sentiment_history) > max_history:
            self._sentiment_history = self._sentiment_history[-max_history:]

    def train_ml_models(
        self,
        historical_prices: List[float],
        historical_volumes: List[float] = None,
    ):
        """Train ML models on historical data."""
        if len(historical_prices) < 200:
            return False

        # Prepare features and targets
        X_list = []
        y_direction = []
        y_magnitude = []

        lookback = 60
        for i in range(lookback, len(historical_prices) - 1):
            prices = historical_prices[i-lookback:i+1]
            volumes = historical_volumes[i-lookback:i+1] if historical_volumes else None

            features, _ = FeatureEngineering.prepare_features(
                prices=prices,
                volumes=volumes,
            )

            if len(features) > 0:
                X_list.append(features)
                # Target: next period return
                ret = (historical_prices[i+1] - historical_prices[i]) / historical_prices[i]
                y_direction.append(1 if ret > 0 else 0)
                y_magnitude.append(ret)

        if len(X_list) < 100:
            return False

        X = np.array(X_list)
        y_dir = np.array(y_direction)
        y_mag = np.array(y_magnitude)

        # Train ensemble
        metrics = self.ml_predictor.train(X, y_dir, y_mag)
        self._ml_trained = True

        return metrics

    def generate_signals(
        self,
        order_book=None,
        prices_dict: Dict[str, Tuple[datetime, float]] = None,
        funding_rate=None,
        sentiment=None,
        liquidations: List = None,
    ) -> List[Signal]:
        """Generate signals from all sources."""
        signals = []

        # OBI Signal
        if order_book and self.config.obi.enabled:
            obi_signal = self.obi_generator.update(order_book)
            if obi_signal:
                signals.append(obi_signal)

        # Lead-lag Signal
        if prices_dict and self.config.lead_lag.enabled:
            lead_lag_signal = self.lead_lag_generator.update(prices_dict)
            if lead_lag_signal:
                signals.append(lead_lag_signal)

        # Funding Rate Signal
        if funding_rate and self.config.funding_rate.enabled:
            funding_signal = self.funding_generator.update(funding_rate)
            if funding_signal:
                signals.append(funding_signal)

        # Sentiment Signal
        if sentiment and self.config.sentiment.enabled:
            sentiment_signal = self.sentiment_generator.update(sentiment)
            if sentiment_signal:
                signals.append(sentiment_signal)

        # Liquidation Signal
        if liquidations and self.config.liquidation.enabled:
            liq_signal = self.liquidation_generator.update(liquidations)
            if liq_signal:
                signals.append(liq_signal)

        return signals

    def get_ml_prediction(self) -> Optional[Prediction]:
        """Get ML ensemble prediction."""
        if not self._ml_trained:
            return None

        if len(self._price_history) < 60:
            return None

        features, _ = FeatureEngineering.prepare_features(
            prices=self._price_history,
            volumes=self._volume_history if self._volume_history else None,
            imbalances=self._imbalance_history if self._imbalance_history else None,
            funding_rates=self._funding_history if self._funding_history else None,
            sentiment_scores=self._sentiment_history if self._sentiment_history else None,
        )

        if len(features) == 0:
            return None

        return self.ml_predictor.predict(features)

    def evaluate_market(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        signals: List[Signal],
        ml_prediction: Optional[Prediction] = None,
    ) -> Optional[TradeSignal]:
        """
        Evaluate a market and generate trade signal if opportunity exists.
        """
        reasons = []

        # Check death zone (45-55% probability)
        death_zone = self.risk_config.avoid_probability_range
        if death_zone[0] <= yes_price <= death_zone[1]:
            reasons.append(f"Price {yes_price:.2f} in death zone ({death_zone[0]}-{death_zone[1]})")
            return None

        # Aggregate signals
        combined_signal = self.signal_aggregator.aggregate(signals)

        if combined_signal is None:
            return None

        # Determine base direction from signals
        signal_direction = combined_signal.direction
        signal_confidence = combined_signal.confidence

        # Incorporate ML prediction
        if ml_prediction and ml_prediction.direction != 0:
            # Weight ML prediction
            ml_weight = 0.4
            signal_weight = 0.6

            combined_direction = (
                signal_direction * signal_weight * signal_confidence +
                ml_prediction.direction * ml_weight * ml_prediction.direction_probability
            )

            if combined_direction > 0.2:
                direction = 1
            elif combined_direction < -0.2:
                direction = -1
            else:
                direction = 0

            confidence = (
                signal_confidence * signal_weight +
                ml_prediction.direction_probability * ml_weight
            )
        else:
            direction = signal_direction
            confidence = signal_confidence

        # No clear direction
        if direction == 0:
            reasons.append("No clear directional signal")
            return None

        # Determine which outcome to trade
        # direction > 0 means bullish on underlying crypto
        # For "Will BTC be above $X?" markets:
        # - Bullish = buy Yes
        # - Bearish = buy No
        if direction > 0:
            outcome = "Yes"
            current_price = yes_price
        else:
            outcome = "No"
            current_price = no_price

        # Calculate edge
        # Edge = (our predicted probability - market price) - transaction costs
        predicted_prob = 0.5 + direction * confidence * 0.3  # Scale to probability
        edge = predicted_prob - current_price - self.risk_config.transaction_cost_pct

        # Need minimum edge
        if edge < self.risk_config.min_edge_required:
            reasons.append(f"Edge {edge:.2%} below minimum {self.risk_config.min_edge_required:.2%}")
            return None

        # Position sizing (Kelly-inspired but conservative)
        # f* = (bp - q) / b where b=odds, p=prob of win, q=1-p
        # Simplified: size proportional to edge * confidence
        base_size = self.risk_config.max_position_size_usd
        size_factor = min(1.0, edge / 0.1) * confidence
        size_usd = base_size * size_factor

        # Check available capital
        if size_usd > self.state.available_capital:
            size_usd = self.state.available_capital * 0.9

        if size_usd < 10:  # Minimum trade size
            reasons.append("Position size too small")
            return None

        # Check daily loss limit
        if self.state.daily_pnl < -self.risk_config.max_daily_loss_usd:
            reasons.append("Daily loss limit reached")
            return None

        # Check max positions
        if len(self.state.positions) >= self.risk_config.max_positions:
            reasons.append("Maximum positions reached")
            return None

        signals_used = [s.signal_type.value for s in signals if s.direction != 0]

        return TradeSignal(
            timestamp=datetime.utcnow(),
            action=TradeAction.BUY,
            market_id=market_id,
            outcome=outcome,
            size_usd=size_usd,
            confidence=confidence,
            edge_estimate=edge,
            signals_used=signals_used,
            ml_prediction=ml_prediction,
            reasons=[
                f"Direction: {'bullish' if direction > 0 else 'bearish'}",
                f"Edge: {edge:.2%}",
                f"Confidence: {confidence:.2%}",
                f"Signals: {', '.join(signals_used)}",
            ]
        )

    def execute_trade(self, trade_signal: TradeSignal, current_price: float) -> bool:
        """
        Execute a trade (in simulation or dry run mode).

        Returns True if trade was executed.
        """
        if self.config.dry_run:
            # Simulate trade
            position = Position(
                market_id=trade_signal.market_id,
                outcome=trade_signal.outcome,
                entry_price=current_price,
                size_usd=trade_signal.size_usd,
                entry_time=trade_signal.timestamp,
            )

            self.state.positions[trade_signal.market_id] = position
            self.state.available_capital -= trade_signal.size_usd
            self.state.daily_trades += 1
            self.state.last_trade_time = trade_signal.timestamp

            return True

        # Live trading would go here
        return False

    def close_position(
        self,
        market_id: str,
        exit_price: float,
        resolved_outcome: str = None,
    ) -> Optional[float]:
        """
        Close a position and calculate P&L.

        Returns realized P&L.
        """
        if market_id not in self.state.positions:
            return None

        position = self.state.positions[market_id]

        # Calculate P&L
        if resolved_outcome:
            # Market resolved
            if position.outcome == resolved_outcome:
                # Won: receive $1 per share
                pnl = position.size_usd * (1.0 / position.entry_price - 1)
            else:
                # Lost: lose stake
                pnl = -position.size_usd
        else:
            # Early exit
            pnl = position.size_usd * (exit_price / position.entry_price - 1)

        # Apply transaction costs
        pnl -= position.size_usd * self.risk_config.transaction_cost_pct

        # Update state
        self.state.daily_pnl += pnl
        self.state.available_capital += position.size_usd + pnl
        del self.state.positions[market_id]

        return pnl

    def update_unrealized_pnl(self, market_prices: Dict[str, float]):
        """Update unrealized P&L for all positions."""
        for market_id, position in self.state.positions.items():
            if market_id in market_prices:
                current_price = market_prices[market_id]
                position.unrealized_pnl = position.size_usd * (
                    current_price / position.entry_price - 1
                )

    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new day)."""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0

    def get_state_summary(self) -> Dict:
        """Get summary of current strategy state."""
        total_unrealized = sum(p.unrealized_pnl for p in self.state.positions.values())

        return {
            'total_capital': self.state.total_capital,
            'available_capital': self.state.available_capital,
            'positions_count': len(self.state.positions),
            'daily_pnl': self.state.daily_pnl,
            'daily_trades': self.state.daily_trades,
            'unrealized_pnl': total_unrealized,
            'total_pnl': self.state.daily_pnl + total_unrealized,
        }
