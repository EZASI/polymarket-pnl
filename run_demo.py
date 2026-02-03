#!/usr/bin/env python3
"""
Demo runner for Polymarket Prediction Bot.

This script provides easy access to all bot functionality:
- Backtesting with detailed reports
- Signal monitoring
- Paper trading simulation

Run with: python run_demo.py
"""

import sys
import os

# Add bot module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import get_default_config, get_conservative_config
from bot.backtest.engine import BacktestEngine, print_backtest_report
from bot.strategy import PolymarketStrategy
from bot.data.fetchers import HistoricalDataLoader
import asyncio


def run_quick_backtest():
    """Run a quick backtest demonstration."""
    print("\n" + "="*60)
    print("POLYMARKET CRYPTO PREDICTION BOT - DEMO BACKTEST")
    print("="*60)
    print("\nBased on academically validated signals:")
    print("  - Order Book Imbalance (72% accuracy, Wang 2025)")
    print("  - Lévy Area lead-lag (>20% annual returns, Cartea et al. 2023)")
    print("  - Funding rate regime detection")
    print("  - Liquidation cascade contrarian signals")
    print("  - XGBoost + LSTM ensemble")
    print("\n" + "-"*60)

    # Use conservative config for demo
    config = get_conservative_config()
    config.backtest.initial_capital = 10000
    config.backtest.start_date = "2024-01-01"
    config.backtest.end_date = "2024-06-30"  # 6 months for quick demo

    print(f"\nConfiguration:")
    print(f"  Initial Capital: ${config.backtest.initial_capital:,.2f}")
    print(f"  Period: {config.backtest.start_date} to {config.backtest.end_date}")
    print(f"  Risk per trade: ${config.risk.max_position_size_usd:,.2f}")
    print(f"  Death zone: {config.risk.avoid_probability_range}")

    print("\nEnabled Signals:")
    if config.obi.enabled:
        print(f"  - OBI (weight={config.obi.weight}, threshold=±{config.obi.threshold})")
    if config.lead_lag.enabled:
        print(f"  - Lead-Lag (weight={config.lead_lag.weight})")
    if config.funding_rate.enabled:
        print(f"  - Funding Rate (weight={config.funding_rate.weight})")
    if config.liquidation.enabled:
        print(f"  - Liquidation (weight={config.liquidation.weight})")

    print("\n" + "-"*60)
    print("Running backtest (this may take a moment)...")
    print("-"*60)

    # Run backtest
    engine = BacktestEngine(config)

    def progress(current, total):
        if current % 200 == 0:
            pct = current / total * 100
            print(f"\rProgress: {pct:.0f}%", end="", flush=True)

    result = engine.run(progress_callback=progress)
    print("\r" + " "*30)  # Clear progress line

    # Print results
    print_backtest_report(result)

    # Additional analysis
    print("\n--- ACADEMIC VALIDATION NOTES ---")
    print("""
Key findings from academic literature:

1. ORDER BOOK IMBALANCE (OBI):
   - Academic: 72% accuracy at 500ms (Wang 2025)
   - Backtest validation: Check signal_performance above

2. LEAD-LAG (Lévy Area):
   - Academic: >20% annualized, 70% directional (Cartea et al. 2023)
   - Note: Signal decays over time (~12 month half-life)

3. CRITICAL GAPS IDENTIFIED:
   - On-chain whale signals only valid at 1-4 hour intervals (disabled)
   - Funding rates have ~0 predictive power for single assets
   - Social sentiment effect "too small for economic profits"

4. EXPECTED ALPHA DECAY:
   - McLean & Pontiff (2016): ~50% decay post-publication
   - These signals may already be arbitraged
    """)

    return result


def run_signal_demo():
    """Demonstrate signal generation."""
    print("\n" + "="*60)
    print("SIGNAL GENERATION DEMO")
    print("="*60)

    from bot.signals.generators import (
        OBISignalGenerator, LeadLagSignalGenerator,
        FundingRateSignalGenerator, SentimentSignalGenerator,
        SignalAggregator
    )
    from bot.data.fetchers import (
        SimulatedOrderBookFetcher, SimulatedFundingRateFetcher,
        SimulatedSentimentFetcher
    )
    from bot.config import OBIConfig, LeadLagConfig, FundingRateConfig, SentimentConfig, SignalType
    from datetime import datetime

    # Initialize generators
    obi_gen = OBISignalGenerator(OBIConfig())
    funding_gen = FundingRateSignalGenerator(FundingRateConfig())
    sentiment_gen = SentimentSignalGenerator(SentimentConfig())

    # Initialize fetchers
    ob_fetcher = SimulatedOrderBookFetcher(base_price=50000)
    funding_fetcher = SimulatedFundingRateFetcher()
    sentiment_fetcher = SimulatedSentimentFetcher()

    # Signal aggregator
    aggregator = SignalAggregator()

    print("\nGenerating 10 signal snapshots...\n")

    async def generate_signals():
        for i in range(10):
            # Fetch data
            order_book = await ob_fetcher.fetch()
            funding = await funding_fetcher.fetch()
            sentiment = await sentiment_fetcher.fetch()

            # Generate signals
            obi_signal = obi_gen.update(order_book)
            funding_signal = funding_gen.update(funding)
            sentiment_signal = sentiment_gen.update(sentiment)

            # Aggregate
            signals = [s for s in [obi_signal, funding_signal, sentiment_signal] if s]
            combined = aggregator.aggregate(signals)

            # Print
            print(f"Snapshot {i+1}:")
            print(f"  Price: ${order_book.mid_price():,.2f}")
            print(f"  OBI: {order_book.imbalance():.3f}", end="")
            if obi_signal and obi_signal.direction != 0:
                print(f" -> {'BULLISH' if obi_signal.direction > 0 else 'BEARISH'} (conf={obi_signal.confidence:.2f})")
            else:
                print(" -> neutral")

            print(f"  Funding: {funding.annualized_rate:.1f}% ann.", end="")
            if funding_signal and funding_signal.direction != 0:
                print(f" -> {'BULLISH' if funding_signal.direction > 0 else 'BEARISH'} (contrarian)")
            else:
                print(" -> neutral")

            if combined and combined.direction != 0:
                print(f"  COMBINED: {'BULLISH' if combined.direction > 0 else 'BEARISH'} "
                      f"(strength={combined.strength:.2f}, conf={combined.confidence:.2f})")
            print()

            await asyncio.sleep(0.1)

    asyncio.run(generate_signals())


def run_ml_demo():
    """Demonstrate ML model training and prediction."""
    print("\n" + "="*60)
    print("ML MODEL DEMO")
    print("="*60)

    from bot.models.ml_models import (
        SimpleXGBoostDirectional, SimpleLSTMMagnitude,
        EnsemblePredictor, FeatureEngineering
    )
    import numpy as np

    print("\nTraining ensemble model (XGBoost + LSTM)...")
    print("Academic basis:")
    print("  - XGBoost: 97% accuracy (Springer 2025)")
    print("  - Bi-LSTM: MAE 0.633 (Tripathy et al. 2024)")
    print()

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 500

    # Simulated features
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    volumes = np.abs(np.random.randn(n_samples) * 1000 + 5000)

    # Prepare features
    X_list = []
    y_direction = []
    y_magnitude = []

    for i in range(60, len(prices) - 1):
        features, names = FeatureEngineering.prepare_features(
            prices=prices[i-60:i+1].tolist(),
            volumes=volumes[i-60:i+1].tolist(),
        )
        if len(features) > 0:
            X_list.append(features)
            ret = (prices[i+1] - prices[i]) / prices[i]
            y_direction.append(1 if ret > 0 else 0)
            y_magnitude.append(ret)

    X = np.array(X_list)
    y_dir = np.array(y_direction)
    y_mag = np.array(y_magnitude)

    print(f"Training data: {len(X)} samples, {X.shape[1]} features")

    # Train ensemble
    ensemble = EnsemblePredictor()
    metrics = ensemble.train(X, y_dir, y_mag, names)

    print(f"\nDirection Model (XGBoost-style):")
    print(f"  Training Accuracy: {metrics['direction']['accuracy']:.2%}")
    print(f"  Trees Used: {metrics['direction']['n_trees']}")

    print(f"\nMagnitude Model (LSTM-style):")
    print(f"  Training MAE: {metrics['magnitude']['mae']:.6f}")
    print(f"  Training RMSE: {metrics['magnitude']['rmse']:.6f}")

    # Make prediction
    print("\nMaking prediction on latest data...")
    prediction = ensemble.predict(X[-1])

    print(f"\nPrediction:")
    print(f"  Direction: {'UP' if prediction.direction > 0 else 'DOWN' if prediction.direction < 0 else 'NEUTRAL'}")
    print(f"  Direction Probability: {prediction.direction_probability:.2%}")
    print(f"  Magnitude: {prediction.magnitude:.4%}")
    print(f"  Model: {prediction.model_name}")


def main():
    """Main demo entry point."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║       POLYMARKET CRYPTO PREDICTION BOT - DEMO                ║
║                                                              ║
║   Based on Academic Validation of Prediction Strategies      ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("Select demo mode:")
    print("  1. Quick Backtest (6 months, ~30 seconds)")
    print("  2. Signal Generation Demo")
    print("  3. ML Model Demo")
    print("  4. Run All Demos")
    print("  5. Full Backtest (command line mode)")
    print()

    try:
        choice = input("Enter choice (1-5): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"  # Default to quick backtest

    if choice == "1":
        run_quick_backtest()
    elif choice == "2":
        run_signal_demo()
    elif choice == "3":
        run_ml_demo()
    elif choice == "4":
        run_signal_demo()
        run_ml_demo()
        run_quick_backtest()
    elif choice == "5":
        print("\nUse command line mode for full backtest:")
        print("  python -m bot.main backtest --start 2024-01-01 --end 2025-12-31")
        print("  python -m bot.main backtest --conservative --output results.json")
        print("  python -m bot.main monitor --duration 60")
        print("  python -m bot.main paper --hours 4")
    else:
        print("Invalid choice. Running quick backtest...")
        run_quick_backtest()


if __name__ == "__main__":
    main()
