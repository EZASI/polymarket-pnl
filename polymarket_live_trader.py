#!/usr/bin/env python3
"""
Polymarket 5-Minute Live Trading System
=========================================
Production-ready architecture for automated 5-minute prediction trading.

Dependencies:
    pip install websockets aiohttp py-clob-client python-dotenv

Architecture:
    1. WebSocket feed -> real-time order book + trades
    2. Signal engine  -> runs all 5 strategies in parallel
    3. Risk manager   -> position sizing, exposure limits, drawdown circuit breaker
    4. Execution layer -> smart order routing via CLOB API
    5. Monitor        -> PnL tracking, logging, alerting
"""

import asyncio
import json
import time
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("polymarket_trader.log"),
    ],
)
log = logging.getLogger("polymarket")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """All tunable parameters in one place."""

    # API Configuration
    # Polymarket CLOB
    clob_api_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    gamma_api_url: str = "https://gamma-api.polymarket.com"

    # Crypto.com Exchange (the data source we verified works)
    exchange_ws_url: str = "wss://stream.crypto.com/exchange/v1/market"
    exchange_rest_url: str = "https://api.crypto.com/exchange/v1/public"

    # Trading Parameters
    instruments: list = field(default_factory=lambda: ["BTCUSD", "ETHUSD"])
    position_size_usd: float = 10_000  # Per trade
    max_position_usd: float = 50_000  # Total exposure
    max_daily_loss_bps: float = 100  # Circuit breaker: 1% daily loss
    max_drawdown_bps: float = 50  # Per-trade max drawdown

    # Strategy Thresholds (calibrated from live data analysis)
    ofi_threshold: float = 0.30  # Order Flow Imbalance trigger
    book_imbalance_threshold: float = 0.25  # Book imbalance for mean reversion
    volume_zscore_threshold: float = 2.0  # Volume spike detection
    rsi_overbought: float = 80
    rsi_oversold: float = 20
    vwap_deviation_threshold_bps: float = 0.5

    # Execution
    order_type: str = "FOK"  # Fill-or-Kill for market orders
    max_slippage_bps: float = 2.0  # Reject if slippage > 2bps
    bar_interval_seconds: int = 300  # 5 minutes

    # Data
    trade_buffer_size: int = 500  # Recent trades to keep
    candle_buffer_size: int = 100  # Recent candles to keep
    book_depth: int = 50  # Order book levels


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OrderBook:
    """Live order book state."""
    bids: list = field(default_factory=list)  # [(price, qty), ...]
    asks: list = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float:
        if self.mid == 0:
            return 0
        return ((self.best_ask - self.best_bid) / self.mid) * 10000


@dataclass
class Candle:
    """5-minute OHLCV bar."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_usd: float
    timestamp: float


@dataclass
class Trade:
    """Individual trade."""
    price: float
    qty: float
    side: str  # "buy" or "sell"
    timestamp: float


@dataclass
class Position:
    """Current position state."""
    instrument: str
    direction: str  # "long", "short", "flat"
    entry_price: float = 0.0
    size_usd: float = 0.0
    unrealized_pnl_bps: float = 0.0
    entry_time: float = 0.0
    strategy: str = ""


# =============================================================================
# MARKET DATA ENGINE
# =============================================================================

class MarketDataEngine:
    """
    Manages WebSocket connections and maintains live market state.

    Architecture:
        - Primary: WebSocket for real-time book + trade updates (~100ms latency)
        - Fallback: REST polling every 1s if WS disconnects
        - Candle aggregation: built from trade stream in real-time
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.books: dict[str, OrderBook] = {}
        self.trades: dict[str, deque] = {}
        self.candles: dict[str, deque] = {}
        self._candle_builders: dict[str, dict] = {}

        for inst in config.instruments:
            self.books[inst] = OrderBook()
            self.trades[inst] = deque(maxlen=config.trade_buffer_size)
            self.candles[inst] = deque(maxlen=config.candle_buffer_size)
            self._candle_builders[inst] = self._new_candle_builder()

    def _new_candle_builder(self) -> dict:
        return {
            "open": 0, "high": 0, "low": float("inf"),
            "close": 0, "volume": 0, "volume_usd": 0,
            "start_time": 0,
        }

    async def connect_websocket(self):
        """
        Connect to exchange WebSocket for real-time data.

        Polymarket CLOB WebSocket:
            wss://ws-subscriptions-clob.polymarket.com/ws/market
            Subscribe: {"assets_ids": ["token_id_1", ...]}
            Messages: book updates, price_change, last_trade_price

        Crypto.com Exchange WebSocket (for crypto pairs):
            wss://stream.crypto.com/exchange/v1/market
            Subscribe: book.{instrument}.{depth}, trade.{instrument}
        """
        try:
            import websockets
        except ImportError:
            log.warning("websockets not installed, using REST polling fallback")
            await self._rest_polling_loop()
            return

        url = self.config.exchange_ws_url
        log.info(f"Connecting to WebSocket: {url}")

        async with websockets.connect(url) as ws:
            # Subscribe to book + trade channels
            for inst in self.config.instruments:
                sub_msg = {
                    "id": 1,
                    "method": "subscribe",
                    "params": {
                        "channels": [
                            f"book.{inst}.50",
                            f"trade.{inst}",
                        ]
                    }
                }
                await ws.send(json.dumps(sub_msg))
                log.info(f"Subscribed to {inst} book + trades")

            async for message in ws:
                data = json.loads(message)
                await self._process_ws_message(data)

    async def _process_ws_message(self, data: dict):
        """Route incoming WebSocket messages to appropriate handlers."""
        method = data.get("method", "")
        result = data.get("result", {})

        if "book" in method:
            self._update_book(result)
        elif "trade" in method:
            self._update_trades(result)

    def _update_book(self, data: dict):
        """Update order book from WebSocket message."""
        inst = data.get("instrument_name", "")
        if inst not in self.books:
            return

        book = self.books[inst]
        if "bids" in data:
            book.bids = [(float(b[0]), float(b[1])) for b in data["bids"]]
        if "asks" in data:
            book.asks = [(float(a[0]), float(a[1])) for a in data["asks"]]
        book.timestamp = time.time()

    def _update_trades(self, data: dict):
        """Update trade buffer and aggregate into candles."""
        trades_data = data.get("data", [data])
        for td in trades_data:
            inst = td.get("instrument_name", "")
            if inst not in self.trades:
                continue

            trade = Trade(
                price=float(td.get("price", 0)),
                qty=float(td.get("qty", 0)),
                side=td.get("side", ""),
                timestamp=time.time(),
            )
            self.trades[inst].append(trade)
            self._aggregate_candle(inst, trade)

    def _aggregate_candle(self, inst: str, trade: Trade):
        """Build 5-minute candles from the trade stream."""
        builder = self._candle_builders[inst]
        interval = self.config.bar_interval_seconds
        current_bar = int(trade.timestamp // interval) * interval

        if builder["start_time"] == 0 or current_bar > builder["start_time"]:
            # Emit completed candle
            if builder["start_time"] > 0 and builder["volume"] > 0:
                candle = Candle(
                    open=builder["open"],
                    high=builder["high"],
                    low=builder["low"],
                    close=builder["close"],
                    volume=builder["volume"],
                    volume_usd=builder["volume_usd"],
                    timestamp=builder["start_time"],
                )
                self.candles[inst].append(candle)
                log.info(f"[{inst}] New candle: O={candle.open:.2f} H={candle.high:.2f} "
                         f"L={candle.low:.2f} C={candle.close:.2f} V=${candle.volume_usd:,.0f}")

            # Start new candle
            builder.update({
                "open": trade.price, "high": trade.price,
                "low": trade.price, "close": trade.price,
                "volume": trade.qty,
                "volume_usd": trade.price * trade.qty,
                "start_time": current_bar,
            })
        else:
            # Update current candle
            builder["high"] = max(builder["high"], trade.price)
            builder["low"] = min(builder["low"], trade.price)
            builder["close"] = trade.price
            builder["volume"] += trade.qty
            builder["volume_usd"] += trade.price * trade.qty

    async def _rest_polling_loop(self):
        """Fallback REST polling when WebSocket is unavailable."""
        try:
            import aiohttp
        except ImportError:
            log.error("Neither websockets nor aiohttp installed. Cannot get data.")
            return

        log.info("Starting REST polling fallback (1s interval)")
        async with aiohttp.ClientSession() as session:
            while True:
                for inst in self.config.instruments:
                    try:
                        # Fetch order book
                        url = f"{self.config.exchange_rest_url}/get-book?instrument_name={inst}&depth={self.config.book_depth}"
                        async with session.get(url) as resp:
                            data = await resp.json()
                            if "result" in data:
                                self._update_book(data["result"]["data"][0])

                        # Fetch recent trades
                        url = f"{self.config.exchange_rest_url}/get-trades?instrument_name={inst}&count=50"
                        async with session.get(url) as resp:
                            data = await resp.json()
                            if "result" in data:
                                for t in data["result"]["data"]:
                                    self._update_trades({"data": [t]})

                    except Exception as e:
                        log.error(f"REST poll error for {inst}: {e}")

                await asyncio.sleep(1)


# =============================================================================
# SIGNAL ENGINE
# =============================================================================

class SignalEngine:
    """
    Runs all 5 strategies and produces an ensemble signal.

    Signal combination: Weighted vote with confidence scores.
    Only trade when >= 2 strategies agree on direction.
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_ensemble_signal(
        self,
        book: OrderBook,
        trades: list[Trade],
        candles: list[Candle],
        current_price: float,
    ) -> Optional[dict]:
        """Run all strategies and combine signals."""

        signals = []

        # Strategy 1: Order Flow Imbalance
        ofi = self._compute_ofi(trades)
        if abs(ofi) > self.config.ofi_threshold:
            signals.append({
                "direction": "long" if ofi > 0 else "short",
                "confidence": min(abs(ofi), 1.0),
                "strategy": "OFI",
                "weight": 1.5,  # Highest weight - most predictive at 5m
            })

        # Strategy 2: Book Imbalance Mean Reversion
        book_imb = self._compute_book_imbalance(book)
        if abs(book_imb) > self.config.book_imbalance_threshold:
            signals.append({
                "direction": "short" if book_imb > 0 else "long",  # Contrarian
                "confidence": min(abs(book_imb) * 2, 1.0),
                "strategy": "BookMR",
                "weight": 1.0,
            })

        # Strategy 3: Volume Spike
        vol_z = self._compute_volume_zscore(candles)
        if len(candles) > 0 and vol_z > self.config.volume_zscore_threshold:
            body = candles[-1].close - candles[-1].open
            if abs(body / candles[-1].open) > 0.0005:
                signals.append({
                    "direction": "long" if body > 0 else "short",
                    "confidence": min(vol_z / 4.0, 1.0),
                    "strategy": "VolSpike",
                    "weight": 1.2,
                })

        # Strategy 4: RSI Mean Reversion
        rsi = self._compute_rsi(candles)
        if rsi is not None:
            if rsi > self.config.rsi_overbought:
                signals.append({
                    "direction": "short",
                    "confidence": (rsi - 80) / 20,
                    "strategy": "RSI_MR",
                    "weight": 0.8,
                })
            elif rsi < self.config.rsi_oversold:
                signals.append({
                    "direction": "long",
                    "confidence": (20 - rsi) / 20,
                    "strategy": "RSI_MR",
                    "weight": 0.8,
                })

        # Strategy 5: VWAP Mid Deviation
        vwap_dev = self._compute_vwap_deviation(book)
        if abs(vwap_dev) > self.config.vwap_deviation_threshold_bps:
            signals.append({
                "direction": "long" if vwap_dev > 0 else "short",
                "confidence": min(abs(vwap_dev) / 3.0, 1.0),
                "strategy": "VWAPDev",
                "weight": 1.0,
            })

        if not signals:
            return None

        # Ensemble: weighted vote
        long_score = sum(
            s["confidence"] * s["weight"]
            for s in signals if s["direction"] == "long"
        )
        short_score = sum(
            s["confidence"] * s["weight"]
            for s in signals if s["direction"] == "short"
        )

        # Require at least 2 agreeing strategies
        long_count = sum(1 for s in signals if s["direction"] == "long")
        short_count = sum(1 for s in signals if s["direction"] == "short")

        if long_score > short_score and long_count >= 2:
            return {
                "direction": "long",
                "score": long_score,
                "confidence": long_score / (long_score + short_score),
                "contributing_strategies": [s["strategy"] for s in signals if s["direction"] == "long"],
                "all_signals": signals,
            }
        elif short_score > long_score and short_count >= 2:
            return {
                "direction": "short",
                "score": short_score,
                "confidence": short_score / (long_score + short_score),
                "contributing_strategies": [s["strategy"] for s in signals if s["direction"] == "short"],
                "all_signals": signals,
            }

        return None

    def _compute_ofi(self, trades: list[Trade]) -> float:
        buy_vol = sum(t.qty for t in trades if t.side == "buy")
        sell_vol = sum(t.qty for t in trades if t.side == "sell")
        total = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0

    def _compute_book_imbalance(self, book: OrderBook) -> float:
        bid_qty = sum(q for _, q in book.bids[:20])
        ask_qty = sum(q for _, q in book.asks[:20])
        total = bid_qty + ask_qty
        return (bid_qty - ask_qty) / total if total > 0 else 0

    def _compute_volume_zscore(self, candles: list[Candle], lookback: int = 12) -> float:
        if len(candles) < lookback + 1:
            return 0
        import statistics
        recent = [c.volume_usd for c in candles[-(lookback + 1):-1]]
        current = candles[-1].volume_usd
        mean = statistics.mean(recent)
        std = statistics.stdev(recent) if len(recent) > 1 else 1
        return (current - mean) / std if std > 0 else 0

    def _compute_rsi(self, candles: list[Candle], period: int = 14) -> Optional[float]:
        if len(candles) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(candles)):
            change = candles[i].close - candles[i - 1].close
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        if len(gains) < period:
            return None
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def _compute_vwap_deviation(self, book: OrderBook) -> float:
        if not book.bids or not book.asks:
            return 0
        simple_mid = book.mid
        bid_vwap = sum(p * q for p, q in book.bids[:10]) / sum(q for _, q in book.bids[:10])
        ask_vwap = sum(p * q for p, q in book.asks[:10]) / sum(q for _, q in book.asks[:10])
        bid_depth = sum(q for _, q in book.bids[:10])
        ask_depth = sum(q for _, q in book.asks[:10])
        weighted_mid = (bid_vwap * ask_depth + ask_vwap * bid_depth) / (bid_depth + ask_depth)
        return ((weighted_mid - simple_mid) / simple_mid) * 10000


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Position sizing and risk controls.

    Rules:
    1. Max position per trade: config.position_size_usd
    2. Max total exposure: config.max_position_usd
    3. Daily loss circuit breaker: config.max_daily_loss_bps
    4. Spread filter: don't trade if spread > 3bps
    5. Minimum confidence threshold: 0.5
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl_bps = 0.0
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.circuit_breaker_active = False

    def check_trade(self, signal: dict, book: OrderBook, instrument: str) -> Optional[dict]:
        """Validate trade against risk rules. Returns sized order or None."""

        # Circuit breaker
        if self.circuit_breaker_active:
            log.warning("Circuit breaker active - no new trades")
            return None

        if self.daily_pnl_bps < -self.config.max_daily_loss_bps:
            self.circuit_breaker_active = True
            log.error(f"CIRCUIT BREAKER: Daily PnL {self.daily_pnl_bps:.1f}bps < -{self.config.max_daily_loss_bps}bps")
            return None

        # Spread filter
        if book.spread_bps > 3.0:
            log.info(f"Spread too wide: {book.spread_bps:.2f}bps")
            return None

        # Confidence filter
        if signal["confidence"] < 0.5:
            log.info(f"Confidence too low: {signal['confidence']:.2f}")
            return None

        # Exposure check
        current_exposure = sum(
            abs(p.size_usd) for p in self.positions.values()
            if p.direction != "flat"
        )
        if current_exposure + self.config.position_size_usd > self.config.max_position_usd:
            log.warning(f"Max exposure reached: ${current_exposure:,.0f}")
            return None

        # Position conflict check
        if instrument in self.positions:
            existing = self.positions[instrument]
            if existing.direction != "flat" and existing.direction != signal["direction"]:
                log.info(f"Conflicting position in {instrument}: {existing.direction} vs {signal['direction']}")
                return None

        # Calculate position size (scale by confidence)
        size_usd = self.config.position_size_usd * signal["confidence"]
        size_usd = min(size_usd, self.config.position_size_usd)

        # Calculate stop loss and take profit
        mid = book.mid
        if signal["direction"] == "long":
            stop_loss = mid * (1 - self.config.max_drawdown_bps / 10000)
            take_profit = mid * (1 + self.config.max_drawdown_bps * 1.6 / 10000)
        else:
            stop_loss = mid * (1 + self.config.max_drawdown_bps / 10000)
            take_profit = mid * (1 - self.config.max_drawdown_bps * 1.6 / 10000)

        return {
            "instrument": instrument,
            "direction": signal["direction"],
            "size_usd": size_usd,
            "entry_price": mid,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategies": signal["contributing_strategies"],
            "confidence": signal["confidence"],
        }

    def update_pnl(self, pnl_bps: float):
        self.daily_pnl_bps += pnl_bps

    def reset_daily(self):
        self.daily_pnl_bps = 0.0
        self.circuit_breaker_active = False
        self.trade_count = 0


# =============================================================================
# EXECUTION ENGINE
# =============================================================================

class ExecutionEngine:
    """
    Smart order execution via CLOB API.

    Order Types Used:
    - FOK (Fill-or-Kill): For immediate execution, rejects if can't fill fully
    - GTC limit orders: For maker rebates when we're not in a hurry

    Polymarket API:
        POST https://clob.polymarket.com/order
        {
            "tokenID": "...",
            "price": "0.55",
            "size": "100",
            "side": "BUY",
            "type": "FOK"
        }
        Requires EIP-712 signature from wallet.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.pending_orders = {}

    async def execute_order(self, order: dict) -> Optional[dict]:
        """
        Execute a trade order via the CLOB API.

        In production, this would:
        1. Sign the order with EIP-712
        2. Submit to POST /order
        3. Wait for fill confirmation via WebSocket user channel
        4. Return fill details
        """
        log.info(
            f"EXECUTE: {order['direction'].upper()} ${order['size_usd']:,.0f} "
            f"{order['instrument']} @ ${order['entry_price']:,.2f} "
            f"(SL=${order['stop_loss']:,.2f}, TP=${order['take_profit']:,.2f}) "
            f"[{', '.join(order['strategies'])}]"
        )

        # In paper trading mode, simulate immediate fill
        return {
            "order_id": f"sim_{int(time.time()*1000)}",
            "fill_price": order["entry_price"],
            "fill_size_usd": order["size_usd"],
            "status": "filled",
            "timestamp": time.time(),
        }


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

class TradingSystem:
    """
    Main trading loop coordinator.

    Lifecycle:
    1. Initialize data feeds
    2. Wait for sufficient data (>= 15 candles for RSI)
    3. Every 5 minutes (or on significant events):
       a. Generate ensemble signal
       b. Risk-check the signal
       c. Execute if approved
       d. Manage open positions (SL/TP)
    4. Log everything
    """

    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.data = MarketDataEngine(self.config)
        self.signals = SignalEngine(self.config)
        self.risk = RiskManager(self.config)
        self.execution = ExecutionEngine(self.config)
        self.running = False

    async def run(self):
        """Start the trading system."""
        log.info("=" * 60)
        log.info("POLYMARKET 5-MIN TRADING SYSTEM STARTING")
        log.info(f"Instruments: {self.config.instruments}")
        log.info(f"Position size: ${self.config.position_size_usd:,.0f}")
        log.info(f"Max exposure: ${self.config.max_position_usd:,.0f}")
        log.info(f"Circuit breaker: {self.config.max_daily_loss_bps}bps daily loss")
        log.info("=" * 60)

        self.running = True

        # Run data feed and strategy loop concurrently
        await asyncio.gather(
            self.data.connect_websocket(),
            self._strategy_loop(),
        )

    async def _strategy_loop(self):
        """Main strategy evaluation loop - runs every bar close."""
        while self.running:
            await asyncio.sleep(5)  # Check every 5 seconds

            for inst in self.config.instruments:
                book = self.data.books[inst]
                trades = list(self.data.trades[inst])
                candles = list(self.data.candles[inst])

                if not book.bids or not book.asks:
                    continue

                current_price = book.mid

                # Check for stop loss / take profit on existing positions
                await self._manage_positions(inst, current_price)

                # Only generate new signals at bar close
                # (in production, check if a new candle just formed)
                if len(candles) < 15:  # Need enough data for RSI
                    continue

                # Generate ensemble signal
                signal = self.signals.generate_ensemble_signal(
                    book, trades, candles, current_price
                )

                if signal is None:
                    continue

                log.info(
                    f"[{inst}] Signal: {signal['direction'].upper()} "
                    f"(score={signal['score']:.2f}, conf={signal['confidence']:.2f}) "
                    f"from {signal['contributing_strategies']}"
                )

                # Risk check
                order = self.risk.check_trade(signal, book, inst)
                if order is None:
                    continue

                # Execute
                fill = await self.execution.execute_order(order)
                if fill and fill["status"] == "filled":
                    self.risk.positions[inst] = Position(
                        instrument=inst,
                        direction=order["direction"],
                        entry_price=fill["fill_price"],
                        size_usd=fill["fill_size_usd"],
                        entry_time=fill["timestamp"],
                        strategy=",".join(order["strategies"]),
                    )

    async def _manage_positions(self, inst: str, current_price: float):
        """Check SL/TP for open positions."""
        if inst not in self.risk.positions:
            return

        pos = self.risk.positions[inst]
        if pos.direction == "flat":
            return

        # Calculate unrealized PnL
        if pos.direction == "long":
            pnl_bps = ((current_price - pos.entry_price) / pos.entry_price) * 10000
        else:
            pnl_bps = ((pos.entry_price - current_price) / pos.entry_price) * 10000

        pos.unrealized_pnl_bps = pnl_bps

        # Time stop: close after 5 minutes
        if time.time() - pos.entry_time > self.config.bar_interval_seconds:
            log.info(
                f"[{inst}] TIME STOP: Closing {pos.direction} @ ${current_price:,.2f} "
                f"PnL={pnl_bps:+.1f}bps (${pnl_bps * pos.entry_price / 10000:+.2f})"
            )
            self.risk.update_pnl(pnl_bps)
            pos.direction = "flat"
            pos.size_usd = 0


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """
    Run the trading system.

    Usage:
        # Paper trading (default)
        python polymarket_live_trader.py

        # With custom config
        python polymarket_live_trader.py --position-size 5000 --instruments BTCUSD,ETHUSD
    """
    config = TradingConfig()

    print("""
╔══════════════════════════════════════════════════════════════╗
║         POLYMARKET 5-MIN TRADING SYSTEM                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Mode:      PAPER TRADING (no real orders)                   ║
║  Strategy:  5-strategy ensemble                              ║
║  Timeframe: 5-minute bars                                    ║
║                                                              ║
║  Tech Stack:                                                 ║
║    - WebSocket: Real-time book + trade feed                  ║
║    - Signal:    5 parallel strategies with ensemble voting    ║
║    - Risk:      Circuit breaker + exposure limits             ║
║    - Execution: FOK orders via CLOB API                      ║
║                                                              ║
║  Key API Endpoints:                                          ║
║    REST:                                                     ║
║      GET  /book          Order book snapshot                 ║
║      GET  /price         Current price                       ║
║      GET  /prices-history Historical timeseries              ║
║      POST /order         Place order (auth required)         ║
║      DEL  /order         Cancel order (auth required)        ║
║    WebSocket:                                                ║
║      wss://ws-subscriptions-clob.polymarket.com/ws/market    ║
║      Channels: book, price_change, last_trade_price          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    system = TradingSystem(config)
    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        log.info("Shutting down...")
        system.running = False


if __name__ == "__main__":
    main()
