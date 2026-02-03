"""
Main bot orchestrator for Polymarket crypto prediction.

This module provides:
- Live trading mode (dry run by default)
- Backtesting mode
- Signal monitoring mode

Based on academically validated signals with proper risk management.
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional
import sys

from .config import BotConfig, get_default_config, get_conservative_config
from .strategy import PolymarketStrategy, TradeAction
from .backtest.engine import BacktestEngine, print_backtest_report, export_results_json
from .data.fetchers import (
    SimulatedOrderBookFetcher, SimulatedFundingRateFetcher,
    SimulatedSentimentFetcher, SimulatedLiquidationFetcher,
    HistoricalDataLoader
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolymarketBot:
    """
    Main bot class orchestrating all components.

    Modes:
    - backtest: Run historical simulation
    - monitor: Monitor signals without trading
    - paper: Paper trading with simulated execution
    """

    def __init__(self, config: BotConfig = None):
        self.config = config or get_default_config()
        self.strategy = PolymarketStrategy(self.config)
        self._running = False

        # Data fetchers
        self.orderbook_fetcher = SimulatedOrderBookFetcher()
        self.funding_fetcher = SimulatedFundingRateFetcher()
        self.sentiment_fetcher = SimulatedSentimentFetcher()
        self.liquidation_fetcher = SimulatedLiquidationFetcher()

    async def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        output_file: str = None,
    ):
        """
        Run backtest simulation.

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            output_file: Optional JSON file for results export
        """
        logger.info("Starting backtest...")

        # Update config dates if provided
        if start_date:
            self.config.backtest.start_date = start_date
        if end_date:
            self.config.backtest.end_date = end_date

        # Create backtest engine
        engine = BacktestEngine(self.config)

        # Progress callback
        def progress(current, total):
            if current % 100 == 0:
                pct = current / total * 100
                print(f"\rBacktest progress: {pct:.1f}%", end="", flush=True)

        # Run backtest
        result = engine.run(progress_callback=progress)
        print()  # New line after progress

        # Print report
        print_backtest_report(result)

        # Export if requested
        if output_file:
            export_results_json(result, output_file)
            logger.info(f"Results exported to {output_file}")

        return result

    async def run_monitor(self, duration_minutes: int = 60):
        """
        Monitor signals without trading.

        Useful for:
        - Understanding signal behavior
        - Validating signal generation
        - Pre-deployment testing
        """
        logger.info(f"Starting signal monitor for {duration_minutes} minutes...")
        logger.info("Press Ctrl+C to stop")

        self._running = True
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        current_price = 50000.0  # Starting price

        try:
            while self._running and datetime.utcnow() < end_time:
                # Fetch data
                order_book = await self.orderbook_fetcher.fetch()
                funding_rate = await self.funding_fetcher.fetch()
                sentiment = await self.sentiment_fetcher.fetch()
                liquidations = await self.liquidation_fetcher.fetch(current_price, 0.001)

                # Update strategy data
                current_price = order_book.mid_price()
                self.strategy.update_data(
                    price=current_price,
                    volume=1000,
                    imbalance=order_book.imbalance(),
                    funding_rate=funding_rate.rate,
                    sentiment=sentiment.sentiment_score,
                )

                # Generate signals
                prices_dict = {
                    'binance': (datetime.utcnow(), current_price),
                    'coinbase': (datetime.utcnow(), current_price * 1.0001),
                    'kraken': (datetime.utcnow(), current_price * 0.9999),
                }

                signals = self.strategy.generate_signals(
                    order_book=order_book,
                    prices_dict=prices_dict,
                    funding_rate=funding_rate,
                    sentiment=sentiment,
                    liquidations=liquidations,
                )

                # Print signal summary
                if signals:
                    active_signals = [s for s in signals if s.direction != 0]
                    if active_signals:
                        logger.info(f"Price: ${current_price:,.2f}")
                        for signal in active_signals:
                            direction_str = "BULLISH" if signal.direction > 0 else "BEARISH"
                            logger.info(
                                f"  {signal.signal_type.value}: {direction_str} "
                                f"(strength={signal.strength:.2f}, conf={signal.confidence:.2f})"
                            )

                # Get ML prediction if available
                ml_pred = self.strategy.get_ml_prediction()
                if ml_pred and ml_pred.direction != 0:
                    logger.info(
                        f"  ML: {'UP' if ml_pred.direction > 0 else 'DOWN'} "
                        f"(prob={ml_pred.direction_probability:.2f})"
                    )

                await asyncio.sleep(self.config.update_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")

        self._running = False
        logger.info("Signal monitor complete")

    async def run_paper_trading(self, duration_hours: int = 24):
        """
        Run paper trading (simulated execution).

        Executes trades in simulation mode to validate strategy
        without risking real capital.
        """
        logger.info(f"Starting paper trading for {duration_hours} hours...")
        logger.info("Press Ctrl+C to stop")

        self._running = True
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)

        current_price = 50000.0
        market_counter = 0

        try:
            while self._running and datetime.utcnow() < end_time:
                # Fetch data
                order_book = await self.orderbook_fetcher.fetch()
                funding_rate = await self.funding_fetcher.fetch()
                sentiment = await self.sentiment_fetcher.fetch()

                price_change = (order_book.mid_price() - current_price) / current_price
                current_price = order_book.mid_price()

                liquidations = await self.liquidation_fetcher.fetch(current_price, price_change)

                # Update strategy data
                self.strategy.update_data(
                    price=current_price,
                    volume=1000,
                    imbalance=order_book.imbalance(),
                    funding_rate=funding_rate.rate,
                    sentiment=sentiment.sentiment_score,
                )

                # Generate signals
                prices_dict = {
                    'binance': (datetime.utcnow(), current_price),
                    'coinbase': (datetime.utcnow(), current_price * 1.0001),
                    'kraken': (datetime.utcnow(), current_price * 0.9999),
                }

                signals = self.strategy.generate_signals(
                    order_book=order_book,
                    prices_dict=prices_dict,
                    funding_rate=funding_rate,
                    sentiment=sentiment,
                    liquidations=liquidations,
                )

                ml_pred = self.strategy.get_ml_prediction()

                # Create simulated market
                market_counter += 1
                market_id = f"btc_sim_{market_counter}"
                yes_price = 0.5 + (order_book.imbalance() * 0.1)
                yes_price = max(0.1, min(0.9, yes_price))

                # Evaluate and potentially trade
                trade_signal = self.strategy.evaluate_market(
                    market_id=market_id,
                    yes_price=yes_price,
                    no_price=1 - yes_price,
                    signals=signals,
                    ml_prediction=ml_pred,
                )

                if trade_signal and trade_signal.action == TradeAction.BUY:
                    entry_price = yes_price if trade_signal.outcome == "Yes" else 1 - yes_price
                    self.strategy.execute_trade(trade_signal, entry_price)

                    logger.info(
                        f"TRADE: {trade_signal.outcome} @ {entry_price:.3f} "
                        f"(size=${trade_signal.size_usd:.2f}, edge={trade_signal.edge_estimate:.2%})"
                    )

                # Print state periodically
                state = self.strategy.get_state_summary()
                logger.info(
                    f"Price: ${current_price:,.2f} | "
                    f"P&L: ${state['daily_pnl']:.2f} | "
                    f"Positions: {state['positions_count']}"
                )

                # Simulate market resolution (simplified)
                if self.strategy.state.positions:
                    # Close oldest position
                    oldest_market = next(iter(self.strategy.state.positions))
                    # Random resolution for demo
                    import random
                    resolved = "Yes" if random.random() > 0.5 else "No"
                    pnl = self.strategy.close_position(oldest_market, 1.0 if resolved == self.strategy.state.positions[oldest_market].outcome else 0.0, resolved)
                    if pnl is not None:
                        logger.info(f"RESOLVED: {oldest_market} -> {resolved}, P&L: ${pnl:.2f}")

                await asyncio.sleep(self.config.update_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Paper trading stopped by user")

        self._running = False

        # Final summary
        state = self.strategy.get_state_summary()
        logger.info("\n=== PAPER TRADING SUMMARY ===")
        logger.info(f"Total P&L: ${state['total_pnl']:.2f}")
        logger.info(f"Daily Trades: {state['daily_trades']}")
        logger.info(f"Final Capital: ${state['available_capital']:.2f}")

    def stop(self):
        """Stop the bot."""
        self._running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket Crypto Prediction Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest
  python -m bot.main backtest --start 2024-06-01 --end 2024-12-31

  # Monitor signals
  python -m bot.main monitor --duration 30

  # Paper trading
  python -m bot.main paper --hours 4

  # Use conservative config
  python -m bot.main backtest --conservative
        """
    )

    parser.add_argument(
        'mode',
        choices=['backtest', 'monitor', 'paper'],
        help='Operating mode'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2024-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2025-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Monitor duration in minutes'
    )

    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Paper trading duration in hours'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file for backtest results (JSON)'
    )

    parser.add_argument(
        '--conservative',
        action='store_true',
        help='Use conservative config (only strongly validated signals)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Initial capital for backtest'
    )

    args = parser.parse_args()

    # Select configuration
    if args.conservative:
        config = get_conservative_config()
        logger.info("Using conservative configuration (strongly validated signals only)")
    else:
        config = get_default_config()

    config.backtest.initial_capital = args.capital

    # Create bot
    bot = PolymarketBot(config)

    # Run selected mode
    if args.mode == 'backtest':
        asyncio.run(bot.run_backtest(
            start_date=args.start,
            end_date=args.end,
            output_file=args.output,
        ))

    elif args.mode == 'monitor':
        asyncio.run(bot.run_monitor(duration_minutes=args.duration))

    elif args.mode == 'paper':
        asyncio.run(bot.run_paper_trading(duration_hours=args.hours))


if __name__ == '__main__':
    main()
