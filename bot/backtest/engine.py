"""
Backtesting engine for Polymarket prediction strategy.

Features:
- Event-driven simulation
- Realistic execution with slippage and fees
- Performance metrics calculation
- Trade logging and analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import defaultdict
import json

from ..config import BotConfig, BacktestConfig
from ..strategy import PolymarketStrategy, TradeSignal, TradeAction
from ..data.fetchers import (
    OHLCV, HistoricalDataLoader, SimulatedOrderBookFetcher,
    SimulatedFundingRateFetcher, SimulatedSentimentFetcher,
    SimulatedLiquidationFetcher
)


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: datetime
    exit_time: datetime
    market_id: str
    outcome: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    pnl_pct: float
    signals_used: List[str]
    resolved: bool  # True if market resolved, False if early exit


@dataclass
class BacktestResult:
    """Results from backtest run."""
    # Summary metrics
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float

    # Time series
    equity_curve: List[Tuple[datetime, float]]
    drawdown_curve: List[Tuple[datetime, float]]

    # Trade details
    trades: List[Trade]

    # By signal breakdown
    signal_performance: Dict[str, Dict[str, float]]

    # Additional stats
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trading_days: int


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates:
    - Market data feed
    - Signal generation
    - Trade execution with realistic costs
    - Position management
    - Performance tracking
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.backtest_config = config.backtest

        # Initialize strategy
        self.strategy = PolymarketStrategy(config)

        # Data sources (simulated)
        self.orderbook_fetcher = SimulatedOrderBookFetcher()
        self.funding_fetcher = SimulatedFundingRateFetcher()
        self.sentiment_fetcher = SimulatedSentimentFetcher()
        self.liquidation_fetcher = SimulatedLiquidationFetcher()

        # Historical data loader
        self.data_loader = HistoricalDataLoader()

        # State
        self._trades: List[Trade] = []
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._peak_equity: float = config.backtest.initial_capital

    def run(
        self,
        ohlcv_data: List[OHLCV] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        progress_callback: Callable[[int, int], None] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            ohlcv_data: Historical price data. If None, generates simulated data.
            start_date: Start date for backtest.
            end_date: End date for backtest.
            progress_callback: Optional callback for progress updates.

        Returns:
            BacktestResult with performance metrics.
        """
        # Generate data if not provided
        if ohlcv_data is None:
            start = datetime.strptime(self.backtest_config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.backtest_config.end_date, "%Y-%m-%d")
            ohlcv_data = self.data_loader.generate_historical_ohlcv(
                start_date=start,
                end_date=end,
                timeframe="1h",
            )

        if not ohlcv_data:
            raise ValueError("No data for backtesting")

        # Extract prices for ML training
        prices = [c.close for c in ohlcv_data]
        volumes = [c.volume for c in ohlcv_data]

        # Train ML models on first portion of data
        train_size = min(500, len(prices) // 3)
        if train_size > 100:
            self.strategy.train_ml_models(
                prices[:train_size],
                volumes[:train_size],
            )

        # Initialize equity tracking
        current_equity = self.backtest_config.initial_capital
        self._equity_curve = [(ohlcv_data[0].timestamp, current_equity)]
        self._peak_equity = current_equity
        self._trades = []

        # Active markets (simulated Polymarket markets)
        active_markets: Dict[str, Dict] = {}

        # Main simulation loop
        total_candles = len(ohlcv_data)
        for i, candle in enumerate(ohlcv_data[train_size:], start=train_size):
            if progress_callback:
                progress_callback(i, total_candles)

            current_time = candle.timestamp
            current_price = candle.close
            price_change_pct = (candle.close - candle.open) / candle.open

            # Update strategy data
            self.strategy.update_data(
                price=current_price,
                volume=candle.volume,
            )

            # Generate simulated market data
            self.orderbook_fetcher.set_price(current_price)

            # Run async fetchers synchronously for backtest
            import asyncio
            loop = asyncio.new_event_loop()

            order_book = loop.run_until_complete(self.orderbook_fetcher.fetch())
            funding_rate = loop.run_until_complete(self.funding_fetcher.fetch())
            sentiment = loop.run_until_complete(self.sentiment_fetcher.fetch())
            liquidations = loop.run_until_complete(
                self.liquidation_fetcher.fetch(current_price, price_change_pct)
            )

            loop.close()

            # Update strategy with additional data
            self.strategy.update_data(
                price=current_price,
                imbalance=order_book.imbalance(),
                funding_rate=funding_rate.rate,
                sentiment=sentiment.sentiment_score,
            )

            # Generate signals
            prices_dict = {
                'binance': (current_time, current_price),
                'coinbase': (current_time, current_price * (1 + np.random.normal(0, 0.0001))),
                'kraken': (current_time, current_price * (1 + np.random.normal(0, 0.0002))),
            }

            signals = self.strategy.generate_signals(
                order_book=order_book,
                prices_dict=prices_dict,
                funding_rate=funding_rate,
                sentiment=sentiment,
                liquidations=liquidations,
            )

            # Get ML prediction
            ml_prediction = self.strategy.get_ml_prediction()

            # Create simulated Polymarket market
            strike_price = current_price * 1.002  # Market: "Will BTC be above X+0.2%?"
            market_id = f"btc_{int(strike_price)}_{i}"

            # Calculate market prices (simplified)
            # Yes price = probability of going up
            base_yes_price = 0.5 + np.random.normal(0, 0.05)
            yes_price = max(0.01, min(0.99, base_yes_price))
            no_price = 1 - yes_price

            # Evaluate market
            trade_signal = self.strategy.evaluate_market(
                market_id=market_id,
                yes_price=yes_price,
                no_price=no_price,
                signals=signals,
                ml_prediction=ml_prediction,
            )

            # Execute trade if signal generated
            if trade_signal and trade_signal.action == TradeAction.BUY:
                entry_price = yes_price if trade_signal.outcome == "Yes" else no_price

                # Apply slippage
                slippage = self.backtest_config.slippage_pct
                entry_price *= (1 + slippage)

                self.strategy.execute_trade(trade_signal, entry_price)

                # Store market for resolution
                active_markets[market_id] = {
                    'strike_price': strike_price,
                    'current_price': current_price,
                    'entry_time': current_time,
                    'entry_candle': i,
                    'signal': trade_signal,
                    'entry_price': entry_price,
                }

            # Resolve markets (after N candles, simulating 15-min resolution)
            resolution_delay = 1  # Resolve after 1 candle (1 hour in this sim)
            markets_to_close = []

            for mid, market_info in active_markets.items():
                if i - market_info['entry_candle'] >= resolution_delay:
                    # Determine resolution
                    resolved_yes = current_price > market_info['strike_price']
                    resolved_outcome = "Yes" if resolved_yes else "No"

                    # Close position
                    pnl = self.strategy.close_position(
                        market_id=mid,
                        exit_price=1.0 if resolved_outcome == market_info['signal'].outcome else 0.0,
                        resolved_outcome=resolved_outcome,
                    )

                    if pnl is not None:
                        # Record trade
                        signal = market_info['signal']
                        won = resolved_outcome == signal.outcome

                        trade = Trade(
                            entry_time=market_info['entry_time'],
                            exit_time=current_time,
                            market_id=mid,
                            outcome=signal.outcome,
                            entry_price=market_info['entry_price'],
                            exit_price=1.0 if won else 0.0,
                            size_usd=signal.size_usd,
                            pnl=pnl,
                            pnl_pct=pnl / signal.size_usd if signal.size_usd > 0 else 0,
                            signals_used=signal.signals_used,
                            resolved=True,
                        )
                        self._trades.append(trade)

                        # Update equity
                        current_equity = self.strategy.state.total_capital + \
                                        self.strategy.state.daily_pnl

                        # Record signal accuracy
                        for sig_type in signal.signals_used:
                            self.strategy.signal_aggregator.record_outcome(
                                signal_type=next(
                                    (st for st in self.strategy.signal_aggregator.signal_weights.keys()
                                     if st.value == sig_type),
                                    None
                                ),
                                was_correct=won,
                            )

                    markets_to_close.append(mid)

            # Remove closed markets
            for mid in markets_to_close:
                del active_markets[mid]

            # Update equity curve
            current_equity = (
                self.strategy.state.available_capital +
                sum(p.size_usd for p in self.strategy.state.positions.values())
            )
            self._equity_curve.append((current_time, current_equity))
            self._peak_equity = max(self._peak_equity, current_equity)

            # Daily reset check
            if i > 0 and ohlcv_data[i-1].timestamp.date() != current_time.date():
                self.strategy.reset_daily_metrics()

        # Calculate final results
        return self._calculate_results(
            start_date=ohlcv_data[train_size].timestamp,
            end_date=ohlcv_data[-1].timestamp,
        )

    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        initial = self.backtest_config.initial_capital
        final = self._equity_curve[-1][1] if self._equity_curve else initial

        # Total return
        total_return = final - initial
        total_return_pct = total_return / initial

        # Calculate returns series for Sharpe
        equity_values = [e[1] for e in self._equity_curve]
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized hourly
            else:
                sharpe_ratio = 0.0
        else:
            returns = []
            sharpe_ratio = 0.0

        # Drawdown calculation
        drawdown_curve = []
        peak = initial
        max_dd = 0
        max_dd_pct = 0

        for timestamp, equity in self._equity_curve:
            peak = max(peak, equity)
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0
            drawdown_curve.append((timestamp, dd_pct))
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)

        # Trade statistics
        total_trades = len(self._trades)
        if total_trades > 0:
            winners = [t for t in self._trades if t.pnl > 0]
            losers = [t for t in self._trades if t.pnl <= 0]

            win_rate = len(winners) / total_trades
            avg_trade_pnl = sum(t.pnl for t in self._trades) / total_trades
            avg_winner = sum(t.pnl for t in winners) / len(winners) if winners else 0
            avg_loser = sum(t.pnl for t in losers) / len(losers) if losers else 0

            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_trade_pnl = 0
            avg_winner = 0
            avg_loser = 0
            profit_factor = 0

        # Signal performance breakdown
        signal_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        for trade in self._trades:
            for signal in trade.signals_used:
                if trade.pnl > 0:
                    signal_performance[signal]['wins'] += 1
                else:
                    signal_performance[signal]['losses'] += 1
                signal_performance[signal]['total_pnl'] += trade.pnl

        # Calculate win rate per signal
        for signal, perf in signal_performance.items():
            total = perf['wins'] + perf['losses']
            perf['win_rate'] = perf['wins'] / total if total > 0 else 0
            perf['avg_pnl'] = perf['total_pnl'] / total if total > 0 else 0

        trading_days = (end_date - start_date).days

        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            equity_curve=self._equity_curve,
            drawdown_curve=drawdown_curve,
            trades=self._trades,
            signal_performance=dict(signal_performance),
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.backtest_config.initial_capital,
            final_capital=final,
            trading_days=trading_days,
        )


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\nPeriod: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Trading Days: {result.trading_days}")

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Initial Capital:    ${result.initial_capital:,.2f}")
    print(f"Final Capital:      ${result.final_capital:,.2f}")
    print(f"Total Return:       ${result.total_return:,.2f} ({result.total_return_pct:.2%})")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:       ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2%})")

    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:       {result.total_trades}")
    print(f"Win Rate:           {result.win_rate:.2%}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")
    print(f"Avg Trade P&L:      ${result.avg_trade_pnl:.2f}")
    print(f"Avg Winner:         ${result.avg_winner:.2f}")
    print(f"Avg Loser:          ${result.avg_loser:.2f}")

    if result.signal_performance:
        print("\n--- SIGNAL PERFORMANCE ---")
        for signal, perf in sorted(result.signal_performance.items()):
            print(f"  {signal}:")
            print(f"    Win Rate: {perf['win_rate']:.2%}")
            print(f"    Avg P&L:  ${perf['avg_pnl']:.2f}")
            print(f"    Total:    {perf['wins']} wins, {perf['losses']} losses")

    print("\n" + "="*60)


def export_results_json(result: BacktestResult, filepath: str):
    """Export backtest results to JSON."""
    data = {
        'summary': {
            'total_return': result.total_return,
            'total_return_pct': result.total_return_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_pct': result.max_drawdown_pct,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
        },
        'trades': [
            {
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'market_id': t.market_id,
                'outcome': t.outcome,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size_usd': t.size_usd,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'signals_used': t.signals_used,
            }
            for t in result.trades
        ],
        'equity_curve': [
            {'timestamp': ts.isoformat(), 'equity': eq}
            for ts, eq in result.equity_curve
        ],
        'signal_performance': result.signal_performance,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


class AggressiveBacktestEngine:
    """
    High-frequency backtesting engine for 4-coin, 15-minute Polymarket markets.

    Simulates:
    - 4 coins: BTC, ETH, SOL, MATIC
    - 96 markets per day (4 coins × 24 hours × 1 per 15-min period overlap)
    - More aggressive signal thresholds
    - Concurrent position management
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.backtest_config = config.backtest
        self.aggressive_config = config.aggressive

        # Initialize strategy per coin
        self.strategies: Dict[str, PolymarketStrategy] = {}
        for coin in self.aggressive_config.coins:
            self.strategies[coin] = PolymarketStrategy(config)

        # Data fetchers per coin
        self.fetchers: Dict[str, Dict] = {}
        base_prices = {"BTC": 50000, "ETH": 3000, "SOL": 100, "MATIC": 0.80}

        for coin in self.aggressive_config.coins:
            self.fetchers[coin] = {
                'orderbook': SimulatedOrderBookFetcher(
                    symbol=f"{coin}-USD",
                    base_price=base_prices.get(coin, 1000),
                ),
                'funding': SimulatedFundingRateFetcher(symbol=f"{coin}-PERP"),
                'sentiment': SimulatedSentimentFetcher(symbol=coin),
                'liquidation': SimulatedLiquidationFetcher(symbol=f"{coin}-PERP"),
            }

        self._trades: List[Trade] = []
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._daily_pnl: Dict[str, float] = {}  # Track daily P&L

    def run(
        self,
        days: int = 30,
        progress_callback: Callable[[int, int], None] = None,
    ) -> BacktestResult:
        """
        Run aggressive backtest simulation.

        Args:
            days: Number of days to simulate
            progress_callback: Optional progress callback

        Returns:
            BacktestResult with performance metrics
        """
        import asyncio

        initial_capital = self.backtest_config.initial_capital
        current_capital = initial_capital
        available_capital = initial_capital

        start_date = datetime.strptime(self.backtest_config.start_date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=days)

        self._trades = []
        self._equity_curve = [(start_date, current_capital)]
        self._daily_pnl = {}

        # Active positions across all coins
        active_positions: Dict[str, Dict] = {}

        # Simulate each 15-minute period
        total_periods = days * 24 * 4  # 4 periods per hour
        current_time = start_date

        # Track prices for each coin
        prices: Dict[str, List[float]] = {coin: [] for coin in self.aggressive_config.coins}

        for period in range(total_periods):
            if progress_callback and period % 100 == 0:
                progress_callback(period, total_periods)

            current_date = current_time.strftime("%Y-%m-%d")

            # Initialize daily P&L tracking
            if current_date not in self._daily_pnl:
                self._daily_pnl[current_date] = 0.0
                # Reset daily loss check
                for strategy in self.strategies.values():
                    strategy.reset_daily_metrics()

            # Check daily loss limit
            if self._daily_pnl[current_date] < -self.aggressive_config.max_daily_loss_usd:
                current_time += timedelta(minutes=15)
                continue

            # Process each coin
            for coin in self.aggressive_config.coins:
                strategy = self.strategies[coin]
                fetchers = self.fetchers[coin]

                # Fetch simulated data
                loop = asyncio.new_event_loop()
                try:
                    order_book = loop.run_until_complete(fetchers['orderbook'].fetch())
                    funding = loop.run_until_complete(fetchers['funding'].fetch())
                    sentiment = loop.run_until_complete(fetchers['sentiment'].fetch())

                    current_price = order_book.mid_price()
                    prices[coin].append(current_price)

                    # Keep price history manageable
                    if len(prices[coin]) > 500:
                        prices[coin] = prices[coin][-500:]

                    price_change = 0
                    if len(prices[coin]) > 1:
                        price_change = (prices[coin][-1] - prices[coin][-2]) / prices[coin][-2]

                    liquidations = loop.run_until_complete(
                        fetchers['liquidation'].fetch(current_price, price_change)
                    )
                finally:
                    loop.close()

                # Update strategy
                strategy.update_data(
                    price=current_price,
                    volume=1000,
                    imbalance=order_book.imbalance(),
                    funding_rate=funding.rate,
                    sentiment=sentiment.sentiment_score,
                )

                # Generate signals
                prices_dict = {
                    'binance': (current_time, current_price),
                    'coinbase': (current_time, current_price * 1.0001),
                    'kraken': (current_time, current_price * 0.9999),
                }

                signals = strategy.generate_signals(
                    order_book=order_book,
                    prices_dict=prices_dict,
                    funding_rate=funding,
                    sentiment=sentiment,
                    liquidations=liquidations,
                )

                # Create market for this 15-min period
                strike_offset = np.random.choice([0.998, 0.999, 1.001, 1.002])
                strike_price = current_price * strike_offset
                market_id = f"{coin}_{int(current_time.timestamp())}_{int(strike_price)}"

                # Calculate market probability based on imbalance and signals
                base_prob = 0.5
                signal_adjustment = sum(s.weighted_signal() for s in signals if s) * 0.1
                yes_price = max(0.05, min(0.95, base_prob + signal_adjustment + np.random.normal(0, 0.05)))
                no_price = 1 - yes_price

                # Check if we can open position
                active_for_coin = sum(1 for k in active_positions if k.startswith(coin))
                max_per_coin = self.aggressive_config.max_concurrent_positions // len(self.aggressive_config.coins)

                if (active_for_coin < max_per_coin and
                    len(active_positions) < self.aggressive_config.max_concurrent_positions and
                    available_capital >= self.aggressive_config.position_size_usd):

                    # Evaluate market
                    trade_signal = strategy.evaluate_market(
                        market_id=market_id,
                        yes_price=yes_price,
                        no_price=no_price,
                        signals=signals,
                    )

                    if trade_signal and trade_signal.action == TradeAction.BUY:
                        entry_price = yes_price if trade_signal.outcome == "Yes" else no_price
                        entry_price *= (1 + self.backtest_config.slippage_pct)

                        position_size = min(
                            trade_signal.size_usd,
                            self.aggressive_config.position_size_usd,
                            available_capital * 0.25  # Max 25% per position
                        )

                        active_positions[market_id] = {
                            'coin': coin,
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'strike_price': strike_price,
                            'outcome': trade_signal.outcome,
                            'size_usd': position_size,
                            'signals_used': trade_signal.signals_used,
                            'current_price': current_price,
                        }

                        available_capital -= position_size

            # Resolve markets (15-minute resolution)
            markets_to_close = []
            for market_id, pos in active_positions.items():
                # Check if 15 minutes have passed
                if (current_time - pos['entry_time']).total_seconds() >= 900:  # 15 minutes
                    coin = pos['coin']
                    current_price = prices[coin][-1] if prices[coin] else pos['current_price']

                    # Determine resolution
                    resolved_yes = current_price > pos['strike_price']
                    resolved_outcome = "Yes" if resolved_yes else "No"
                    won = resolved_outcome == pos['outcome']

                    # Calculate P&L
                    if won:
                        pnl = pos['size_usd'] * (1.0 / pos['entry_price'] - 1)
                    else:
                        pnl = -pos['size_usd']

                    # Apply transaction costs
                    pnl -= pos['size_usd'] * self.config.risk.transaction_cost_pct

                    # Record trade
                    trade = Trade(
                        entry_time=pos['entry_time'],
                        exit_time=current_time,
                        market_id=market_id,
                        outcome=pos['outcome'],
                        entry_price=pos['entry_price'],
                        exit_price=1.0 if won else 0.0,
                        size_usd=pos['size_usd'],
                        pnl=pnl,
                        pnl_pct=pnl / pos['size_usd'] if pos['size_usd'] > 0 else 0,
                        signals_used=pos['signals_used'],
                        resolved=True,
                    )
                    self._trades.append(trade)

                    # Update capital
                    available_capital += pos['size_usd'] + pnl
                    current_capital += pnl
                    self._daily_pnl[current_date] = self._daily_pnl.get(current_date, 0) + pnl

                    markets_to_close.append(market_id)

            # Close resolved markets
            for market_id in markets_to_close:
                del active_positions[market_id]

            # Record equity
            total_in_positions = sum(p['size_usd'] for p in active_positions.values())
            equity = available_capital + total_in_positions
            self._equity_curve.append((current_time, equity))

            current_time += timedelta(minutes=15)

        # Calculate results
        return self._calculate_results(start_date, end_date)

    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate backtest results."""
        initial = self.backtest_config.initial_capital
        final = self._equity_curve[-1][1] if self._equity_curve else initial

        total_return = final - initial
        total_return_pct = total_return / initial

        # Sharpe calculation
        equity_values = [e[1] for e in self._equity_curve]
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / np.maximum(equity_values[:-1], 1)
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Drawdown
        drawdown_curve = []
        peak = initial
        max_dd = 0
        max_dd_pct = 0

        for timestamp, equity in self._equity_curve:
            peak = max(peak, equity)
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0
            drawdown_curve.append((timestamp, dd_pct))
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)

        # Trade stats
        total_trades = len(self._trades)
        if total_trades > 0:
            winners = [t for t in self._trades if t.pnl > 0]
            losers = [t for t in self._trades if t.pnl <= 0]

            win_rate = len(winners) / total_trades
            avg_trade_pnl = sum(t.pnl for t in self._trades) / total_trades
            avg_winner = sum(t.pnl for t in winners) / len(winners) if winners else 0
            avg_loser = sum(t.pnl for t in losers) / len(losers) if losers else 0

            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = avg_trade_pnl = avg_winner = avg_loser = profit_factor = 0

        # Signal performance
        signal_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        for trade in self._trades:
            for signal in trade.signals_used:
                if trade.pnl > 0:
                    signal_performance[signal]['wins'] += 1
                else:
                    signal_performance[signal]['losses'] += 1
                signal_performance[signal]['total_pnl'] += trade.pnl

        for signal, perf in signal_performance.items():
            total = perf['wins'] + perf['losses']
            perf['win_rate'] = perf['wins'] / total if total > 0 else 0
            perf['avg_pnl'] = perf['total_pnl'] / total if total > 0 else 0

        # Daily P&L summary
        daily_profits = [v for v in self._daily_pnl.values() if v > 0]
        daily_losses = [v for v in self._daily_pnl.values() if v <= 0]

        print(f"\n--- DAILY P&L SUMMARY ---")
        print(f"Profitable days: {len(daily_profits)}")
        print(f"Losing days: {len(daily_losses)}")
        if daily_profits:
            print(f"Avg profitable day: ${np.mean(daily_profits):,.2f}")
        if daily_losses:
            print(f"Avg losing day: ${np.mean(daily_losses):,.2f}")
        print(f"Best day: ${max(self._daily_pnl.values()):,.2f}")
        print(f"Worst day: ${min(self._daily_pnl.values()):,.2f}")

        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            equity_curve=self._equity_curve,
            drawdown_curve=drawdown_curve,
            trades=self._trades,
            signal_performance=dict(signal_performance),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial,
            final_capital=final,
            trading_days=(end_date - start_date).days,
        )
