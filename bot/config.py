"""
Configuration for Polymarket Prediction Bot

Based on academically validated signals from research analysis.
See: Academic Validation of Polymarket Crypto Prediction Strategy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class SignalType(Enum):
    """Types of signals with their academic validation levels."""
    OBI = "order_book_imbalance"  # Strong: 72% accuracy at 500ms-2s
    LEAD_LAG = "lead_lag"  # Strong: Lévy Area validated
    FUNDING_RATE = "funding_rate"  # Moderate: concurrent correlation only
    SENTIMENT = "sentiment"  # Moderate: 3-15 min window, small effect
    WHALE_FLOW = "whale_flow"  # Weak: only works at hours, not minutes
    LIQUIDATION = "liquidation"  # Moderate: contrarian indicator


class TimeFrame(Enum):
    """Supported timeframes for analysis."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class OBIConfig:
    """
    Order Book Imbalance configuration.

    Academic basis:
    - Cont, Kukanov, Stoikov (2014): R² ~65% for price changes
    - Wang (2025): 72% accuracy at 500ms on Bybit BTC/USDT
    - Optimal threshold: ±0.6 (Brazilian BMF&Bovespa study)
    """
    enabled: bool = True
    threshold: float = 0.6  # Validated threshold from academic research
    levels: int = 5  # 5-level aggregate imbalance
    lookback_ms: int = 500  # Optimal: 100ms-2s for crypto
    weight: float = 0.25  # Signal weight in ensemble


@dataclass
class LeadLagConfig:
    """
    Lead-lag relationship configuration using Lévy Area.

    Academic basis:
    - Cartea, Cucuringu, Jin (2023): >20% annualized returns
    - Bitwise SEC filing: 6.5-17s lead from futures to spot
    - Schei (2019): 70% directional accuracy
    """
    enabled: bool = True
    reference_exchanges: List[str] = field(default_factory=lambda: [
        "binance", "coinbase", "kraken"
    ])
    lead_time_seconds: float = 10.0  # Conservative: academic range 6.5-17s
    levy_area_threshold: float = 0.1  # Minimum significance
    weight: float = 0.25


@dataclass
class FundingRateConfig:
    """
    Funding rate signal configuration.

    Academic basis:
    - He et al.: Sharpe ratios 1.8-3.5 for deviation strategies
    - Presto Research: R² ~0 for single-asset prediction
    - Use for regime detection, not direct prediction
    """
    enabled: bool = True
    extreme_threshold_annualized: float = 100.0  # 100%+ as warning
    exchanges: List[str] = field(default_factory=lambda: [
        "binance", "bybit", "okx"
    ])
    use_as_contrarian: bool = True  # Not predictive, use as regime indicator
    weight: float = 0.15


@dataclass
class SentimentConfig:
    """
    Social sentiment configuration.

    Academic basis:
    - Renault et al.: Significant up to 15 minutes
    - Jeleskovic & Mackay: 15-minute impact, 3-min lag
    - Effect magnitude is small - not profitable alone
    """
    enabled: bool = True
    lookback_minutes: int = 15  # Validated window
    reaction_lag_minutes: int = 3  # Documented lag
    platforms: List[str] = field(default_factory=lambda: [
        "twitter"  # TikTok claim is arXiv preprint only
    ])
    weight: float = 0.10  # Low weight due to small effect size


@dataclass
class WhaleFlowConfig:
    """
    Whale/on-chain flow configuration.

    Academic basis:
    - Chi, Hou, Trinh (2024): Only valid at 1-4 hour intervals
    - Lima & Souza (2025): Significant at 6-24 hour windows
    - NOT validated for minutes-level signals
    """
    enabled: bool = False  # Disabled by default - timeframe mismatch
    min_interval_hours: int = 1  # Minimum validated interval
    weight: float = 0.05


@dataclass
class LiquidationConfig:
    """
    Liquidation cascade detection.

    Academic basis:
    - Cheng et al. (2021): 3.51% daily forced liquidations
    - CMU team: Derivatives exacerbate large moves
    - Ali (2025): 86x acceleration during cascades
    """
    enabled: bool = True
    cascade_threshold_usd: float = 100_000_000  # $100M+ signals cascade
    use_as_contrarian: bool = True
    weight: float = 0.10


@dataclass
class MLConfig:
    """
    Machine Learning model configuration.

    Academic basis:
    - XGBoost: 97% accuracy (Springer 2025)
    - Bi-LSTM: MAE 0.633 (Tripathy et al. 2024)
    - Recommendation: XGBoost for direction, LSTM for magnitude
    """
    use_xgboost_direction: bool = True
    use_lstm_magnitude: bool = True
    xgboost_params: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
    })
    lstm_units: int = 64
    lstm_lookback: int = 60  # 60 periods lookback
    retrain_interval_hours: int = 24
    min_training_samples: int = 1000


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_usd: float = 1000.0
    max_daily_loss_usd: float = 500.0
    max_positions: int = 5
    min_edge_required: float = 0.02  # 2% minimum edge
    transaction_cost_pct: float = 0.03  # ~3% at ATM after Jan 2026 fees
    avoid_probability_range: tuple = (0.45, 0.55)  # "Death zone"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 10000.0
    slippage_pct: float = 0.005  # 0.5% slippage
    use_realistic_fills: bool = True
    include_fees: bool = True


@dataclass
class AggressiveConfig:
    """
    Aggressive trading configuration for high-frequency Polymarket trading.

    Targets: 96 trades/day across 4 coins (BTC, ETH, SOL, MATIC)
    Resolution: 15-minute markets
    """
    coins: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "MATIC"])
    trades_per_coin_per_day: int = 24  # One per hour minimum, up to 96
    target_daily_profit_usd: float = 15000.0
    position_size_usd: float = 2500.0  # Per trade
    max_concurrent_positions: int = 8  # 2 per coin
    max_daily_loss_usd: float = 7500.0  # Stop loss for the day (50% of target)
    min_confidence_threshold: float = 0.52  # Lower threshold = more trades


@dataclass
class BotConfig:
    """Main bot configuration combining all components."""
    # API Configuration
    polymarket_api_url: str = "https://data-api.polymarket.com"
    wallet_address: str = ""  # User must provide

    # Signal configurations
    obi: OBIConfig = field(default_factory=OBIConfig)
    lead_lag: LeadLagConfig = field(default_factory=LeadLagConfig)
    funding_rate: FundingRateConfig = field(default_factory=FundingRateConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    whale_flow: WhaleFlowConfig = field(default_factory=WhaleFlowConfig)
    liquidation: LiquidationConfig = field(default_factory=LiquidationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    aggressive: AggressiveConfig = field(default_factory=AggressiveConfig)

    # Execution settings
    dry_run: bool = True  # Always start in dry run mode
    log_level: str = "INFO"
    update_interval_seconds: int = 60

    # Alpha decay awareness (McLean & Pontiff 2016: ~50% decay post-publication)
    expected_alpha_decay_months: int = 12


def get_default_config() -> BotConfig:
    """Get default bot configuration with academically validated settings."""
    return BotConfig()


def get_conservative_config() -> BotConfig:
    """
    Get conservative configuration prioritizing only strongly validated signals.
    """
    config = BotConfig()

    # Only use strongly validated signals
    config.obi.enabled = True
    config.obi.weight = 0.40
    config.obi.threshold = 0.3  # Lower threshold for demo (academic: 0.6)

    config.lead_lag.enabled = True
    config.lead_lag.weight = 0.40
    config.lead_lag.levy_area_threshold = 0.05  # Lower for demo

    config.liquidation.enabled = True
    config.liquidation.weight = 0.20

    # Disable weakly validated signals
    config.funding_rate.enabled = False
    config.sentiment.enabled = False
    config.whale_flow.enabled = False

    # Risk management for demo
    config.risk.min_edge_required = 0.02  # 2% minimum edge for demo
    config.risk.max_position_size_usd = 500.0
    config.risk.avoid_probability_range = (0.47, 0.53)  # Narrower death zone

    return config


def get_aggressive_config(target_daily_profit: float = 15000.0, capital: float = 150000.0) -> BotConfig:
    """
    Get aggressive configuration for high-frequency trading targeting $15k/day.

    WARNING: HIGH RISK configuration. Requires significant capital.

    Math breakdown:
    - 96 potential trades/day (4 coins × 24 15-min periods)
    - Realistically ~60 tradeable (others in death zone or no signal)
    - With 57% win rate, need ~$2,500 position size
    - Capital needed: ~$150k for proper risk management

    Args:
        target_daily_profit: Target profit per day in USD
        capital: Available trading capital
    """
    config = BotConfig()

    # Calculate position sizing
    # Assuming 60 tradeable opportunities per day
    # With 57% win rate: EV per trade = 0.14 * position_size (after costs)
    estimated_trades_per_day = 60
    estimated_edge_per_trade = 0.10  # 10% edge after costs (optimistic)
    position_size = target_daily_profit / (estimated_trades_per_day * estimated_edge_per_trade)

    # Cap position size at 2% of capital per trade
    max_position = capital * 0.02
    position_size = min(position_size, max_position)

    # Enable ALL signals for maximum opportunity detection
    config.obi.enabled = True
    config.obi.weight = 0.30
    config.obi.threshold = 0.20  # Lower threshold = more signals

    config.lead_lag.enabled = True
    config.lead_lag.weight = 0.30
    config.lead_lag.levy_area_threshold = 0.03

    config.funding_rate.enabled = True
    config.funding_rate.weight = 0.15

    config.sentiment.enabled = True
    config.sentiment.weight = 0.10

    config.liquidation.enabled = True
    config.liquidation.weight = 0.15

    # Aggressive risk settings
    config.risk.max_position_size_usd = position_size
    config.risk.max_daily_loss_usd = target_daily_profit * 0.5  # Stop at 50% of target as loss
    config.risk.max_positions = 8  # Multiple concurrent positions
    config.risk.min_edge_required = 0.01  # Lower edge threshold = more trades
    config.risk.avoid_probability_range = (0.48, 0.52)  # Tighter death zone

    # Aggressive config
    config.aggressive.target_daily_profit_usd = target_daily_profit
    config.aggressive.position_size_usd = position_size
    config.aggressive.max_concurrent_positions = 8
    config.aggressive.max_daily_loss_usd = target_daily_profit * 0.5
    config.aggressive.min_confidence_threshold = 0.52  # Trade on slight edge

    # Backtest with appropriate capital
    config.backtest.initial_capital = capital

    return config
