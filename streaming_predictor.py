#!/usr/bin/env python3
"""
STREAMING ALPHA ENGINE: Sub-Second BTC Predictor
==================================================

Real-time data streams:
  1. Binance Orderbook WSS (100ms depth diffs)
  2. Binance Trade WSS (tick-by-tick BTC + ETH)
  3. Mempool.space WSS (pending BTC transactions)

Computes ~40 streaming features every 1 second, scores them against
pre-trained ML models, and emits BUY/SELL/HOLD signals.

Usage:
  # First train models (uses ultra_predictor)
  python3 ultra_predictor.py --mode train --days 14 --save

  # Then run streaming predictor
  python3 streaming_predictor.py

  # Or run standalone (trains quick models on recent data first)
  python3 streaming_predictor.py --self-train
"""

import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import websockets

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "btcusdt"
SYMBOL_UPPER = "BTCUSDT"
ETH_SYMBOL = "ethusdt"
SOL_SYMBOL = "solusdt"

BINANCE_WSS = "wss://stream.binance.com:9443/ws"
BINANCE_REST = "https://api.binance.com/api/v3"
BINANCE_FAPI = "https://fapi.binance.com"
MEMPOOL_WSS = "wss://mempool.space/api/v1/ws"

PREDICTION_INTERVAL = 1.0        # seconds between predictions
WHALE_THRESHOLD_USD = 100_000    # trades above this = whale
MEMPOOL_WHALE_BTC = 10.0         # BTC txns above this = whale
FEATURE_HISTORY = 60             # seconds of feature history to keep

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stream")


# ============================================================
# COMPONENT 1: OrderbookStream
# ============================================================
class OrderbookStream:
    """
    Maintains a local orderbook from Binance 100ms depth diffs.

    Flow:
      1. Fetch REST snapshot for initial state
      2. Connect to btcusdt@depth@100ms websocket
      3. Apply diffs to local book
      4. Compute imbalance features every call to get_features()
    """

    def __init__(self):
        self.bids: dict[float, float] = {}   # price -> qty
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.snapshot_loaded = False
        self.connected = False

        # Feature history for velocity computation
        self._imb_history: deque[tuple[float, float]] = deque(maxlen=30)  # (timestamp, imbalance)
        self._spread_history: deque[tuple[float, float]] = deque(maxlen=30)

    async def load_snapshot(self):
        """Fetch REST orderbook snapshot."""
        url = f"{BINANCE_REST}/depth?symbol={SYMBOL_UPPER}&limit=100"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()

            self.bids = {float(p): float(q) for p, q in data["bids"]}
            self.asks = {float(p): float(q) for p, q in data["asks"]}
            self.last_update_id = data["lastUpdateId"]
            self.snapshot_loaded = True
            log.info(f"Orderbook snapshot: {len(self.bids)} bids, {len(self.asks)} asks")
        except Exception as e:
            log.error(f"Orderbook snapshot failed: {e}")

    def apply_diff(self, msg: dict):
        """Apply depth diff update to local book."""
        if not self.snapshot_loaded:
            return

        # Process bid updates
        for price_s, qty_s in msg.get("b", []):
            price, qty = float(price_s), float(qty_s)
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        # Process ask updates
        for price_s, qty_s in msg.get("a", []):
            price, qty = float(price_s), float(qty_s)
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_features(self) -> dict:
        """Compute orderbook features from current state."""
        if not self.bids or not self.asks:
            return {}

        now = time.time()

        bid_prices = sorted(self.bids.keys(), reverse=True)
        ask_prices = sorted(self.asks.keys())

        if not bid_prices or not ask_prices:
            return {}

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = spread / mid * 10000

        features = {
            "ob_mid": mid,
            "ob_spread": spread,
            "ob_spread_bps": spread_bps,
        }

        # Imbalance at different depths
        for depth in [5, 10, 20, 50]:
            top_bids = bid_prices[:depth]
            top_asks = ask_prices[:depth]

            bid_vol = sum(self.bids[p] for p in top_bids)
            ask_vol = sum(self.asks[p] for p in top_asks)
            total = bid_vol + ask_vol

            if total > 0:
                imb = (bid_vol - ask_vol) / total
            else:
                imb = 0.0

            features[f"ob_imb_{depth}"] = imb

            # Wall detection: largest single level vs average
            if top_bids:
                bid_vals = [self.bids[p] for p in top_bids]
                features[f"ob_bid_wall_{depth}"] = max(bid_vals) / (sum(bid_vals) / len(bid_vals))
            if top_asks:
                ask_vals = [self.asks[p] for p in top_asks]
                features[f"ob_ask_wall_{depth}"] = max(ask_vals) / (sum(ask_vals) / len(ask_vals))

        # Orderbook slope: how quickly depth drops off
        if len(bid_prices) >= 10:
            near_bid = sum(self.bids[p] for p in bid_prices[:5])
            far_bid = sum(self.bids[p] for p in bid_prices[5:10])
            features["ob_bid_slope"] = near_bid / max(far_bid, 1e-8)

        if len(ask_prices) >= 10:
            near_ask = sum(self.asks[p] for p in ask_prices[:5])
            far_ask = sum(self.asks[p] for p in ask_prices[5:10])
            features["ob_ask_slope"] = near_ask / max(far_ask, 1e-8)

        # Imbalance velocity (change over time)
        imb_10 = features.get("ob_imb_10", 0)
        self._imb_history.append((now, imb_10))
        self._spread_history.append((now, spread_bps))

        for lookback_s in [1, 3, 5]:
            cutoff = now - lookback_s
            past = [v for t, v in self._imb_history if t <= cutoff]
            if past:
                features[f"ob_imb_vel_{lookback_s}s"] = imb_10 - past[-1]
            else:
                features[f"ob_imb_vel_{lookback_s}s"] = 0.0

        for lookback_s in [1, 3, 5]:
            cutoff = now - lookback_s
            past = [v for t, v in self._spread_history if t <= cutoff]
            if past:
                features[f"ob_spread_vel_{lookback_s}s"] = spread_bps - past[-1]

        return features

    async def run(self):
        """Connect to depth stream and maintain local book."""
        await self.load_snapshot()

        url = f"{BINANCE_WSS}/{SYMBOL}@depth@100ms"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self.connected = True
                    log.info("Orderbook stream connected")
                    async for raw in ws:
                        msg = json.loads(raw)
                        self.apply_diff(msg)
            except Exception as e:
                self.connected = False
                log.warning(f"Orderbook stream error: {e}, reconnecting...")
                await asyncio.sleep(2)


# ============================================================
# COMPONENT 2: TradeStream
# ============================================================
@dataclass
class MicroBar:
    """Aggregated trade data over a micro time window."""
    open: float = 0
    high: float = 0
    low: float = float("inf")
    close: float = 0
    volume: float = 0
    buy_volume: float = 0
    sell_volume: float = 0
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    vwap_sum: float = 0
    max_trade_usd: float = 0


class TradeStream:
    """
    Tick-by-tick trade aggregation for BTC and ETH.

    Computes:
      - 1s, 5s, 15s micro-bars
      - Taker buy/sell imbalance
      - Whale detection (trades > $100k)
      - Trade arrival rate
      - ETH-BTC lead-lag
    """

    def __init__(self, symbol: str = SYMBOL, is_eth: bool = False):
        self.symbol = symbol
        self.is_eth = is_eth
        self.prefix = "eth" if is_eth else "btc"
        self.connected = False

        # Current 1-second bar
        self._current_bar = MicroBar()
        self._bar_start = time.time()

        # History of completed bars
        self._bars_1s: deque[MicroBar] = deque(maxlen=60)
        self._bars_5s: deque[MicroBar] = deque(maxlen=30)
        self._bars_15s: deque[MicroBar] = deque(maxlen=20)

        # Whale trades (last 60 seconds)
        self._whales: deque[dict] = deque(maxlen=100)

        # Raw tick buffer for aggregation
        self._tick_buffer_5s: list[MicroBar] = []
        self._tick_buffer_15s: list[MicroBar] = []

        self.last_price: float = 0.0

    def _process_trade(self, msg: dict):
        """Process a single trade tick."""
        price = float(msg["p"])
        qty = float(msg["q"])
        is_buyer_maker = msg.get("m", False)
        trade_usd = price * qty

        self.last_price = price

        bar = self._current_bar
        if bar.trade_count == 0:
            bar.open = price
            bar.high = price
            bar.low = price
        else:
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)

        bar.close = price
        bar.volume += qty
        bar.trade_count += 1
        bar.vwap_sum += trade_usd
        bar.max_trade_usd = max(bar.max_trade_usd, trade_usd)

        if is_buyer_maker:
            # Buyer is maker = seller is taker = sell pressure
            bar.sell_volume += qty
            bar.sell_count += 1
        else:
            bar.buy_volume += qty
            bar.buy_count += 1

        # Whale detection
        if trade_usd >= WHALE_THRESHOLD_USD:
            direction = "SELL" if is_buyer_maker else "BUY"
            self._whales.append({
                "time": time.time(),
                "price": price,
                "qty": qty,
                "usd": trade_usd,
                "direction": direction,
            })
            if not self.is_eth:
                log.info(f"WHALE {direction}: ${trade_usd:,.0f} @ ${price:,.2f}")

    def _close_bar(self):
        """Close current 1s bar and aggregate into 5s/15s."""
        bar = self._current_bar
        if bar.trade_count > 0:
            self._bars_1s.append(bar)
            self._tick_buffer_5s.append(bar)
            self._tick_buffer_15s.append(bar)

        # Aggregate 5s bars
        if len(self._tick_buffer_5s) >= 5:
            merged = self._merge_bars(self._tick_buffer_5s[:5])
            self._bars_5s.append(merged)
            self._tick_buffer_5s = self._tick_buffer_5s[5:]

        # Aggregate 15s bars
        if len(self._tick_buffer_15s) >= 15:
            merged = self._merge_bars(self._tick_buffer_15s[:15])
            self._bars_15s.append(merged)
            self._tick_buffer_15s = self._tick_buffer_15s[15:]

        # Reset
        self._current_bar = MicroBar()
        self._bar_start = time.time()

    @staticmethod
    def _merge_bars(bars: list[MicroBar]) -> MicroBar:
        m = MicroBar()
        m.open = bars[0].open
        m.close = bars[-1].close
        m.high = max(b.high for b in bars)
        m.low = min(b.low for b in bars if b.low < float("inf"))
        m.volume = sum(b.volume for b in bars)
        m.buy_volume = sum(b.buy_volume for b in bars)
        m.sell_volume = sum(b.sell_volume for b in bars)
        m.trade_count = sum(b.trade_count for b in bars)
        m.buy_count = sum(b.buy_count for b in bars)
        m.sell_count = sum(b.sell_count for b in bars)
        m.vwap_sum = sum(b.vwap_sum for b in bars)
        m.max_trade_usd = max(b.max_trade_usd for b in bars)
        return m

    def get_features(self) -> dict:
        """Compute trade-based features."""
        p = self.prefix
        features = {f"{p}_price": self.last_price}

        # 1-second bar features
        if self._bars_1s:
            last = self._bars_1s[-1]
            total_vol = last.buy_volume + last.sell_volume
            features[f"{p}_tps"] = last.trade_count                    # trades per second
            features[f"{p}_vol_1s"] = last.volume
            features[f"{p}_buy_ratio_1s"] = last.buy_volume / max(total_vol, 1e-8)
            features[f"{p}_imb_1s"] = (last.buy_volume - last.sell_volume) / max(total_vol, 1e-8)
            features[f"{p}_max_trade_1s"] = last.max_trade_usd

            if last.volume > 0:
                features[f"{p}_vwap_1s"] = last.vwap_sum / last.volume / max(last.close, 1e-8) - 1
            else:
                features[f"{p}_vwap_1s"] = 0

        # 5-second bar features
        if self._bars_5s:
            last5 = self._bars_5s[-1]
            total5 = last5.buy_volume + last5.sell_volume
            features[f"{p}_vol_5s"] = last5.volume
            features[f"{p}_imb_5s"] = (last5.buy_volume - last5.sell_volume) / max(total5, 1e-8)
            features[f"{p}_tps_5s"] = last5.trade_count / 5
            if last5.open > 0:
                features[f"{p}_ret_5s"] = (last5.close - last5.open) / last5.open * 100

        # 15-second bar features
        if self._bars_15s:
            last15 = self._bars_15s[-1]
            total15 = last15.buy_volume + last15.sell_volume
            features[f"{p}_vol_15s"] = last15.volume
            features[f"{p}_imb_15s"] = (last15.buy_volume - last15.sell_volume) / max(total15, 1e-8)
            if last15.open > 0:
                features[f"{p}_ret_15s"] = (last15.close - last15.open) / last15.open * 100

        # Rolling momentum from 1s bars
        if len(self._bars_1s) >= 5:
            bars = list(self._bars_1s)
            recent = bars[-5:]
            if recent[0].close > 0:
                features[f"{p}_mom_5s"] = (recent[-1].close - recent[0].close) / recent[0].close * 100
            total_buy = sum(b.buy_volume for b in recent)
            total_sell = sum(b.sell_volume for b in recent)
            total = total_buy + total_sell
            features[f"{p}_imb_roll_5s"] = (total_buy - total_sell) / max(total, 1e-8)

        if len(self._bars_1s) >= 15:
            bars = list(self._bars_1s)
            recent = bars[-15:]
            if recent[0].close > 0:
                features[f"{p}_mom_15s"] = (recent[-1].close - recent[0].close) / recent[0].close * 100

        if len(self._bars_1s) >= 30:
            bars = list(self._bars_1s)
            recent = bars[-30:]
            if recent[0].close > 0:
                features[f"{p}_mom_30s"] = (recent[-1].close - recent[0].close) / recent[0].close * 100

        # Whale activity (last 60 seconds)
        now = time.time()
        recent_whales = [w for w in self._whales if now - w["time"] < 60]
        features[f"{p}_whale_count_60s"] = len(recent_whales)
        features[f"{p}_whale_buy_usd"] = sum(w["usd"] for w in recent_whales if w["direction"] == "BUY")
        features[f"{p}_whale_sell_usd"] = sum(w["usd"] for w in recent_whales if w["direction"] == "SELL")

        whale_buy = features[f"{p}_whale_buy_usd"]
        whale_sell = features[f"{p}_whale_sell_usd"]
        whale_total = whale_buy + whale_sell
        features[f"{p}_whale_imb"] = (whale_buy - whale_sell) / max(whale_total, 1e-8) if whale_total > 0 else 0

        # Last 10 seconds whale activity
        recent_10 = [w for w in self._whales if now - w["time"] < 10]
        features[f"{p}_whale_count_10s"] = len(recent_10)

        # Trade arrival rate change (acceleration)
        if len(self._bars_1s) >= 6:
            bars = list(self._bars_1s)
            recent_rate = np.mean([b.trade_count for b in bars[-3:]])
            prior_rate = np.mean([b.trade_count for b in bars[-6:-3]])
            features[f"{p}_tps_accel"] = recent_rate - prior_rate

        return features

    async def run(self):
        """Connect to trade stream."""
        url = f"{BINANCE_WSS}/{self.symbol}@trade"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self.connected = True
                    log.info(f"Trade stream connected: {self.symbol}")
                    async for raw in ws:
                        msg = json.loads(raw)
                        self._process_trade(msg)

                        # Close 1s bar if time elapsed
                        if time.time() - self._bar_start >= 1.0:
                            self._close_bar()
            except Exception as e:
                self.connected = False
                log.warning(f"Trade stream ({self.symbol}) error: {e}, reconnecting...")
                await asyncio.sleep(2)


# ============================================================
# COMPONENT 3: MempoolStream
# ============================================================
class MempoolStream:
    """
    Connects to mempool.space WebSocket for real-time Bitcoin mempool data.

    Tracks:
      - Mempool size and transaction count
      - Large pending transactions (>10 BTC)
      - Fee rate changes (congestion signal)
    """

    def __init__(self):
        self.connected = False
        self.mempool_count: int = 0
        self.mempool_vsize: int = 0
        self.fee_rate_fast: float = 0
        self.fee_rate_medium: float = 0

        self._large_txs: deque[dict] = deque(maxlen=50)
        self._mempool_history: deque[tuple[float, int]] = deque(maxlen=60)

    def get_features(self) -> dict:
        now = time.time()
        features = {
            "mp_tx_count": self.mempool_count,
            "mp_vsize_mb": self.mempool_vsize / 1_000_000 if self.mempool_vsize else 0,
            "mp_fee_fast": self.fee_rate_fast,
            "mp_fee_medium": self.fee_rate_medium,
        }

        # Large tx activity
        recent_large = [t for t in self._large_txs if now - t["time"] < 300]
        features["mp_whale_count_5m"] = len(recent_large)
        features["mp_whale_btc_5m"] = sum(t.get("btc", 0) for t in recent_large)

        # Mempool size change (congestion acceleration)
        self._mempool_history.append((now, self.mempool_count))
        if len(self._mempool_history) >= 2:
            oldest = self._mempool_history[0]
            if now - oldest[0] > 0:
                features["mp_growth_rate"] = (self.mempool_count - oldest[1]) / (now - oldest[0])
        else:
            features["mp_growth_rate"] = 0

        return features

    async def run(self):
        """Connect to mempool.space WebSocket."""
        while True:
            try:
                async with websockets.connect(MEMPOOL_WSS, ping_interval=30) as ws:
                    self.connected = True
                    log.info("Mempool stream connected")

                    # Subscribe to mempool blocks and stats
                    await ws.send(json.dumps({
                        "action": "init",
                    }))
                    await ws.send(json.dumps({
                        "action": "want",
                        "data": ["stats", "mempool-blocks"],
                    }))

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            self._process_message(msg)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                self.connected = False
                log.debug(f"Mempool stream error: {e}, reconnecting in 10s...")
                await asyncio.sleep(10)

    def _process_message(self, msg: dict):
        """Process mempool.space message."""
        # Mempool info
        if "mempoolInfo" in msg:
            info = msg["mempoolInfo"]
            self.mempool_count = info.get("size", 0)
            self.mempool_vsize = info.get("vsize", 0)

        # Fee estimates
        if "fees" in msg:
            fees = msg["fees"]
            self.fee_rate_fast = fees.get("fastestFee", 0)
            self.fee_rate_medium = fees.get("halfHourFee", 0)

        # Transactions (if provided)
        if "transactions" in msg:
            for tx in msg.get("transactions", []):
                value_btc = tx.get("value", 0) / 1e8 if isinstance(tx.get("value"), (int, float)) else 0
                if value_btc >= MEMPOOL_WHALE_BTC:
                    self._large_txs.append({
                        "time": time.time(),
                        "btc": value_btc,
                        "fee": tx.get("fee", 0),
                    })


# ============================================================
# COMPONENT 3b: DerivativesPoller (Funding + OI + Long/Short)
# ============================================================
class DerivativesPoller:
    """Polls Binance Futures REST APIs every 60s for derivatives data."""

    def __init__(self):
        self.funding_rate: float = 0
        self.funding_rate_pct: float = 0
        self.mark_price: float = 0
        self.index_price: float = 0
        self.open_interest: float = 0
        self.oi_value: float = 0
        self._prev_oi: float = 0
        self.oi_change_pct: float = 0
        self.connected = False

    def get_features(self) -> dict:
        return {
            "funding_rate": self.funding_rate,
            "funding_rate_pct": self.funding_rate_pct,
            "funding_extreme_long": 1 if self.funding_rate_pct > 0.03 else 0,
            "funding_extreme_short": 1 if self.funding_rate_pct < -0.01 else 0,
            "funding_positive": 1 if self.funding_rate > 0 else 0,
            "mark_price": self.mark_price,
            "basis": self.mark_price - self.index_price if self.index_price > 0 else 0,
            "open_interest": self.open_interest,
            "oi_value": self.oi_value,
            "oi_change_pct": self.oi_change_pct,
            "oi_rising": 1 if self.oi_change_pct > 1 else 0,
            "oi_falling": 1 if self.oi_change_pct < -1 else 0,
        }

    async def run(self):
        """Poll funding rate + open interest every 60 seconds."""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Funding rate / premium index
                    try:
                        async with session.get(
                            f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
                            params={"symbol": SYMBOL_UPPER}, timeout=aiohttp.ClientTimeout(total=10)
                        ) as resp:
                            data = await resp.json()
                            self.funding_rate = float(data.get("lastFundingRate", 0))
                            self.funding_rate_pct = self.funding_rate * 100
                            self.mark_price = float(data.get("markPrice", 0))
                            self.index_price = float(data.get("indexPrice", 0))
                    except Exception:
                        pass

                    # Open interest
                    try:
                        async with session.get(
                            f"{BINANCE_FAPI}/fapi/v1/openInterest",
                            params={"symbol": SYMBOL_UPPER}, timeout=aiohttp.ClientTimeout(total=10)
                        ) as resp:
                            data = await resp.json()
                            new_oi = float(data.get("openInterest", 0))
                            if self._prev_oi > 0:
                                self.oi_change_pct = (new_oi - self._prev_oi) / self._prev_oi * 100
                            self._prev_oi = self.open_interest
                            self.open_interest = new_oi
                            self.oi_value = new_oi * self.mark_price if self.mark_price > 0 else 0
                    except Exception:
                        pass

                    self.connected = True

            except Exception:
                self.connected = False

            await asyncio.sleep(60)


# ============================================================
# COMPONENT 4: StreamingFeatureEngine
# ============================================================
class StreamingFeatureEngine:
    """
    Merges features from all streams every 1 second.
    Computes cross-stream features (e.g., ETH-BTC lead-lag).
    Maintains feature history for delta/velocity computation.
    """

    def __init__(self, orderbook: OrderbookStream, btc_trades: TradeStream,
                 eth_trades: TradeStream, mempool: MempoolStream,
                 sol_trades: TradeStream = None,
                 derivatives: DerivativesPoller = None):
        self.orderbook = orderbook
        self.btc_trades = btc_trades
        self.eth_trades = eth_trades
        self.sol_trades = sol_trades
        self.mempool = mempool
        self.derivatives = derivatives

        self._feature_history: deque[dict] = deque(maxlen=FEATURE_HISTORY)
        self._prediction_log: list[dict] = []

    def compute_features(self) -> dict:
        """Compute all features from all streams."""
        features = {}

        # Orderbook features
        ob = self.orderbook.get_features()
        features.update(ob)

        # BTC trade features
        btc = self.btc_trades.get_features()
        features.update(btc)

        # ETH trade features
        eth = self.eth_trades.get_features()
        features.update(eth)

        # SOL trade features
        if self.sol_trades:
            sol = self.sol_trades.get_features()
            features.update(sol)

        # Mempool features
        mp = self.mempool.get_features()
        features.update(mp)

        # Derivatives features (funding + OI)
        if self.derivatives:
            deriv = self.derivatives.get_features()
            features.update(deriv)

        # ── CROSS-STREAM FEATURES ──

        # ETH-BTC lead-lag: if ETH moved, BTC might follow
        eth_ret_5s = features.get("eth_ret_5s", 0)
        btc_ret_5s = features.get("btc_ret_5s", 0)
        if eth_ret_5s and btc_ret_5s is not None:
            features["eth_btc_lead"] = eth_ret_5s - btc_ret_5s
            features["eth_leading_up"] = 1 if (eth_ret_5s > 0.05 and btc_ret_5s < 0.02) else 0
            features["eth_leading_down"] = 1 if (eth_ret_5s < -0.05 and btc_ret_5s > -0.02) else 0

        # Trade imbalance (defined early for use in cross-features)
        trade_imb = features.get("btc_imb_5s", 0)

        # SOL-BTC lead-lag
        sol_ret_5s = features.get("sol_ret_5s", 0)
        if sol_ret_5s and btc_ret_5s is not None:
            features["sol_btc_lead"] = sol_ret_5s - btc_ret_5s
            features["sol_leading_up"] = 1 if (sol_ret_5s > 0.08 and btc_ret_5s < 0.03) else 0
            features["sol_leading_down"] = 1 if (sol_ret_5s < -0.08 and btc_ret_5s > -0.03) else 0

        # ETH+SOL sector agreement
        if eth_ret_5s and sol_ret_5s:
            features["sector_bullish"] = 1 if (eth_ret_5s > 0.03 and sol_ret_5s > 0.05) else 0
            features["sector_bearish"] = 1 if (eth_ret_5s < -0.03 and sol_ret_5s < -0.05) else 0

        # Funding rate + flow divergence
        funding = features.get("funding_rate_pct", 0)
        if abs(funding) > 0.01:
            features["funding_flow_diverge"] = 1 if (
                (funding > 0.02 and trade_imb < -0.05) or
                (funding < -0.01 and trade_imb > 0.05)
            ) else 0

        # OI divergence from price
        oi_chg = features.get("oi_change_pct", 0)
        btc_mom = features.get("btc_mom_15s", 0)
        features["oi_price_diverge"] = 1 if (
            (btc_mom > 0.03 and oi_chg < -0.5) or
            (btc_mom < -0.03 and oi_chg > 0.5)
        ) else 0

        # Orderbook + Trade alignment
        ob_imb = features.get("ob_imb_10", 0)
        trade_imb = features.get("btc_imb_5s", 0)
        features["ob_trade_agree"] = 1 if (ob_imb > 0.1 and trade_imb > 0.1) or \
                                          (ob_imb < -0.1 and trade_imb < -0.1) else 0
        features["ob_trade_diverge"] = 1 if (ob_imb > 0.1 and trade_imb < -0.1) or \
                                            (ob_imb < -0.1 and trade_imb > 0.1) else 0

        # Whale + Orderbook alignment
        whale_imb = features.get("btc_whale_imb", 0)
        features["whale_ob_agree"] = 1 if (whale_imb > 0.3 and ob_imb > 0.1) or \
                                          (whale_imb < -0.3 and ob_imb < -0.1) else 0

        # Volatility proxy (range of recent 1s bars)
        if len(self.btc_trades._bars_1s) >= 10:
            bars = list(self.btc_trades._bars_1s)[-10:]
            highs = [b.high for b in bars if b.high > 0]
            lows = [b.low for b in bars if b.low < float("inf")]
            if highs and lows:
                range_pct = (max(highs) - min(lows)) / min(lows) * 100
                features["micro_volatility"] = range_pct
                features["low_vol"] = 1 if range_pct < 0.05 else 0

        # Feature deltas (velocity of features over time)
        self._feature_history.append(features.copy())

        if len(self._feature_history) >= 3:
            prev = self._feature_history[-3]
            for key in ["ob_imb_10", "btc_imb_5s", "btc_tps"]:
                if key in features and key in prev:
                    features[f"{key}_delta_3s"] = features[key] - prev[key]

        if len(self._feature_history) >= 10:
            prev10 = self._feature_history[-10]
            for key in ["ob_imb_10", "btc_imb_5s"]:
                if key in features and key in prev10:
                    features[f"{key}_delta_10s"] = features[key] - prev10[key]

        features["timestamp"] = time.time()

        return features


# ============================================================
# COMPONENT 5: StreamingPredictor
# ============================================================
class StreamingPredictor:
    """
    Loads pre-trained models and makes real-time predictions.
    Applies signal filters with streaming modifiers.
    Logs predictions and tracks live accuracy.
    """

    def __init__(self, feature_engine: StreamingFeatureEngine):
        self.engine = feature_engine
        self.models: list = []
        self.scaler = None
        self.feature_cols: list[str] = []
        self.model_loaded = False

        # Prediction tracking
        self._predictions: deque[dict] = deque(maxlen=1000)
        self._signal_count = 0
        self._correct_count = 0
        self._total_checked = 0

        # Log file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = LOG_DIR / f"signals_{ts}.jsonl"

    def load_models(self):
        """Try to load pre-trained models from ultra_predictor."""
        model_dir = Path("models")
        if not model_dir.exists():
            log.warning("No models/ directory found. Will use streaming-only signals.")
            return

        try:
            import joblib
            path = model_dir / "horizon_1m.joblib"
            if path.exists():
                data = joblib.load(path)
                self.models = data.get("models", [])
                self.scaler = data.get("scaler")
                self.feature_cols = data.get("feature_cols", [])
                self.model_loaded = True
                log.info(f"Loaded {len(self.models)} models, {len(self.feature_cols)} features")
            else:
                log.warning("No horizon_1m.joblib found")
        except Exception as e:
            log.warning(f"Could not load models: {e}")

    def predict(self, features: dict) -> dict:
        """
        Make a prediction from current streaming features.

        Strategy:
          1. If ML models are loaded, get base prediction
          2. Apply streaming modifiers (orderbook, whales, ETH lead)
          3. Apply signal filter (low vol, confirmations)
          4. Emit signal if confidence is high enough
        """
        result = {
            "time": datetime.now(timezone.utc).isoformat(),
            "price": features.get("btc_price", 0),
            "direction": "HOLD",
            "confidence": 0.0,
            "signal": "NONE",
            "reasons": [],
        }

        # ── Streaming-only signals (no ML model needed) ──
        ob_imb = features.get("ob_imb_10", 0)
        ob_vel = features.get("ob_imb_vel_3s", 0)
        trade_imb = features.get("btc_imb_5s", 0)
        whale_imb = features.get("btc_whale_imb", 0)
        eth_lead = features.get("eth_btc_lead", 0)
        low_vol = features.get("low_vol", 0)
        whale_count_10s = features.get("btc_whale_count_10s", 0)
        mom_5s = features.get("btc_mom_5s", 0)

        # Build a directional score from streaming signals
        # LESSON LEARNED: Don't rely on OB imbalance alone.
        #   - OB inversion works in ranges but fails in trends
        #   - Trade flow (taker aggression) is the GROUND TRUTH
        #   - Momentum confirms regime — never fight the trend
        #   - Require MULTIPLE signals to agree before trading

        score = 0.0
        reasons = []
        mom_5s = features.get("btc_mom_5s", 0)
        mom_15s = features.get("btc_mom_15s", 0)
        mom_30s = features.get("btc_mom_30s", 0)
        micro_vol = features.get("micro_volatility", 0)
        trade_imb_roll = features.get("btc_imb_roll_5s", 0)
        whale_buy = features.get("btc_whale_buy_usd", 0)
        whale_sell = features.get("btc_whale_sell_usd", 0)

        # ══════════════════════════════════════════════════
        # MAXIMUM EDGE ENGINE v3
        # ══════════════════════════════════════════════════
        # Philosophy: ONLY trade the highest-conviction setups.
        # Fewer trades, bigger edge, larger size.
        #
        # Three requirements to signal:
        #   1. NO CHOP (market is calm and directional)
        #   2. MOMENTUM ALIGNMENT (5s, 15s, 30s all agree)
        #   3. FLOW CONFIRMATION (takers pushing same direction)
        # ══════════════════════════════════════════════════

        # ── GATE 1: CHOP DETECTOR ──
        is_choppy = False

        # Flash move (too violent)
        if abs(mom_15s) > 0.15:
            is_choppy = True

        # 5s and 15s disagree (reversal in progress)
        if mom_5s != 0 and mom_15s != 0:
            if (mom_5s > 0.02 and mom_15s < -0.02) or (mom_5s < -0.02 and mom_15s > 0.02):
                is_choppy = True

        # Extreme micro-volatility
        if micro_vol > 0.12:
            is_choppy = True

        # 15s and 30s disagree (larger-scale reversal)
        if mom_15s != 0 and mom_30s != 0:
            if (mom_15s > 0.03 and mom_30s < -0.03) or (mom_15s < -0.03 and mom_30s > 0.03):
                is_choppy = True

        if is_choppy:
            result["direction"] = "HOLD"
            result["confidence"] = 0
            result["signal"] = "NONE"
            result["reasons"] = [f"CHOP: m5={mom_5s:+.3f} m15={mom_15s:+.3f} m30={mom_30s:+.3f} v={micro_vol:.3f}"]
            return result

        # ── GATE 2: MOMENTUM ALIGNMENT ──
        # All timeframes must agree on direction.
        # This is the single most important filter.
        up_votes = 0
        down_votes = 0
        neutral = 0

        for m, thresh in [(mom_5s, 0.01), (mom_15s, 0.02), (mom_30s, 0.03)]:
            if m > thresh:
                up_votes += 1
            elif m < -thresh:
                down_votes += 1
            else:
                neutral += 1

        # Require at least 2 of 3 timeframes to agree
        if up_votes >= 2:
            mom_direction = "UP"
            reasons.append(f"Mom:{up_votes}/3 UP (5s={mom_5s:+.3f} 15s={mom_15s:+.3f} 30s={mom_30s:+.3f})")
        elif down_votes >= 2:
            mom_direction = "DOWN"
            reasons.append(f"Mom:{down_votes}/3 DN (5s={mom_5s:+.3f} 15s={mom_15s:+.3f} 30s={mom_30s:+.3f})")
        else:
            # No clear momentum — don't trade
            result["direction"] = "HOLD"
            result["confidence"] = 0
            result["signal"] = "NONE"
            result["reasons"] = [f"NO MOMENTUM: up={up_votes} dn={down_votes} neutral={neutral}"]
            return result

        score = 2.0 if mom_direction == "UP" else -2.0

        # ── SIGNAL A: Trade flow confirmation (REQUIRED) ──
        flow_confirms = False
        if mom_direction == "UP" and trade_imb > 0.05:
            flow_confirms = True
            score += trade_imb * 1.5
            reasons.append(f"Flow={trade_imb:+.2f}")
        elif mom_direction == "DOWN" and trade_imb < -0.05:
            flow_confirms = True
            score += trade_imb * 1.5
            reasons.append(f"Flow={trade_imb:+.2f}")

        # Also check rolling 5s flow
        if mom_direction == "UP" and trade_imb_roll > 0.05:
            flow_confirms = True
            if f"Flow=" not in str(reasons):
                reasons.append(f"FlowRoll={trade_imb_roll:+.2f}")
        elif mom_direction == "DOWN" and trade_imb_roll < -0.05:
            flow_confirms = True
            if f"Flow=" not in str(reasons):
                reasons.append(f"FlowRoll={trade_imb_roll:+.2f}")

        if not flow_confirms:
            # Momentum without flow = weak signal, reduce confidence heavily
            score *= 0.3

        # ── SIGNAL B: Whale activity (bonus, not required) ──
        if whale_count_10s > 0:
            if mom_direction == "UP" and whale_imb > 0.3:
                score += 1.5
                reasons.append(f"Whale BUY ${whale_buy:,.0f}")
            elif mom_direction == "DOWN" and whale_imb < -0.3:
                score += -1.5
                reasons.append(f"Whale SELL ${whale_sell:,.0f}")
            elif abs(whale_imb) > 0.3:
                # Whale going AGAINST our direction — danger, reduce
                score *= 0.5
                reasons.append(f"Whale CONTRA {whale_imb:+.2f}")

        # ── SIGNAL C: ETH leading (bonus) ──
        if mom_direction == "UP" and eth_lead > 0.05:
            score += 0.8
            reasons.append(f"ETH leads UP {eth_lead:+.3f}")
        elif mom_direction == "DOWN" and eth_lead < -0.05:
            score -= 0.8
            reasons.append(f"ETH leads DN {eth_lead:+.3f}")

        # ── CONVERT TO CONFIDENCE ──
        direction = mom_direction
        raw_confidence = min(abs(score) / 5.0, 0.95)  # normalize, cap at 95%

        # ── ML model ──
        ml_confidence = 0.0
        if self.model_loaded and self.scaler is not None:
            try:
                fv = np.zeros(len(self.feature_cols))
                for i, col in enumerate(self.feature_cols):
                    fv[i] = features.get(col, 0.0)
                fv_scaled = self.scaler.transform(fv.reshape(1, -1))
                probas = []
                for name, model in self.models:
                    probas.append(model.predict_proba(fv_scaled))
                avg_prob = np.mean(probas, axis=0)
                up_prob = float(avg_prob[0, 1])
                ml_confidence = max(up_prob, 1 - up_prob)
                ml_direction = "UP" if up_prob > 0.5 else "DOWN"

                if ml_direction == direction:
                    raw_confidence = min(raw_confidence * 1.15, 0.95)
                    reasons.append(f"ML={ml_direction} {ml_confidence:.0%}")
                else:
                    # ML disagrees — only trade if streaming is very strong
                    raw_confidence *= 0.7
                    reasons.append(f"ML contra={ml_direction}")
            except Exception:
                pass

        # ── FINAL SIGNAL GATE ──
        # STRICT: only emit STRONG or MODERATE signals
        confidence = raw_confidence

        if confidence >= 0.65 and flow_confirms and len(reasons) >= 3:
            signal = "STRONG"
        elif confidence >= 0.50 and flow_confirms and len(reasons) >= 2:
            signal = "MODERATE"
        else:
            signal = "NONE"
            direction = "HOLD"

        result["direction"] = direction
        result["confidence"] = confidence
        result["signal"] = signal
        result["reasons"] = reasons
        result["streaming_score"] = score
        result["ml_confidence"] = ml_confidence

        # Only emit actionable signals
        if signal in ("STRONG", "MODERATE"):
            self._signal_count += 1

            # Log for accuracy tracking
            self._predictions.append({
                "time": time.time(),
                "price": features.get("btc_price", 0),
                "direction": direction,
                "confidence": confidence,
                "signal": signal,
            })

            # Write to log file
            with open(self._log_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

        return result

    def check_accuracy(self):
        """Check accuracy of past predictions."""
        now = time.time()
        current_price = self.engine.btc_trades.last_price
        if current_price <= 0:
            return

        checked = 0
        correct = 0

        for pred in list(self._predictions):
            age = now - pred["time"]
            if 55 <= age <= 65 and "checked" not in pred:  # check after ~60 seconds
                pred_price = pred["price"]
                if pred_price > 0:
                    actual_up = current_price > pred_price
                    predicted_up = pred["direction"] == "UP"
                    is_correct = actual_up == predicted_up
                    pred["checked"] = True
                    pred["correct"] = is_correct
                    pred["actual_price"] = current_price
                    checked += 1
                    if is_correct:
                        correct += 1
                        self._correct_count += 1
                    self._total_checked += 1

        if checked > 0:
            recent_wr = self._correct_count / self._total_checked if self._total_checked > 0 else 0
            log.info(
                f"ACCURACY CHECK: {correct}/{checked} correct | "
                f"Running: {self._correct_count}/{self._total_checked} "
                f"({recent_wr:.1%})"
            )

    def get_stats(self) -> dict:
        return {
            "signals": self._signal_count,
            "checked": self._total_checked,
            "correct": self._correct_count,
            "win_rate": self._correct_count / self._total_checked if self._total_checked > 0 else 0,
        }


# ============================================================
# MAIN: Orchestrate everything
# ============================================================
async def main_loop(self_train: bool = False):
    """Main async loop: start all streams and prediction engine."""

    # Initialize components
    orderbook = OrderbookStream()
    btc_trades = TradeStream(SYMBOL, is_eth=False)
    eth_trades = TradeStream(ETH_SYMBOL, is_eth=True)
    mempool = MempoolStream()

    engine = StreamingFeatureEngine(orderbook, btc_trades, eth_trades, mempool)
    predictor = StreamingPredictor(engine)

    # Load pre-trained models
    if self_train:
        log.info("Self-training mode: training models on recent data...")
        try:
            from ultra_predictor import BinanceDataCollector, engineer_all_features, create_targets
            collector = BinanceDataCollector(SYMBOL_UPPER)
            df = collector.get_klines_history(days=7)
            if not df.empty:
                df = engineer_all_features(df)
                df = create_targets(df)

                from ultra_predictor import get_feature_cols
                feat_cols = get_feature_cols(df)
                for c in feat_cols:
                    df[c] = df[c].fillna(df[c].median())

                target = "target_1"
                clean = df.dropna(subset=feat_cols + [target])

                from sklearn.preprocessing import StandardScaler
                X = clean[feat_cols].values
                y = clean[target].values

                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)

                try:
                    import lightgbm as lgb
                    m = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.06,
                        verbose=-1, n_jobs=-1, random_state=42)
                    m.fit(X_s, y)
                    predictor.models = [("lgb", m)]
                    predictor.scaler = scaler
                    predictor.feature_cols = feat_cols
                    predictor.model_loaded = True
                    log.info(f"Self-trained LightGBM on {len(clean):,} rows, {len(feat_cols)} features")
                except ImportError:
                    log.warning("LightGBM not installed, streaming-only mode")
        except Exception as e:
            log.error(f"Self-training failed: {e}")
    else:
        predictor.load_models()

    # Start all streams as concurrent tasks
    tasks = [
        asyncio.create_task(orderbook.run()),
        asyncio.create_task(btc_trades.run()),
        asyncio.create_task(eth_trades.run()),
        asyncio.create_task(mempool.run()),
    ]

    log.info("Starting all streams...")
    await asyncio.sleep(3)  # Let streams connect and buffer initial data

    print("\n" + "=" * 72)
    print("  STREAMING ALPHA ENGINE — LIVE")
    print("=" * 72)
    print(f"  Streams: Orderbook(100ms) + BTC Trades + ETH Trades + Mempool")
    print(f"  ML Model: {'LOADED' if predictor.model_loaded else 'STREAMING-ONLY'}")
    print(f"  Prediction interval: {PREDICTION_INTERVAL}s")
    print(f"  Signal log: {predictor._log_path}")
    print("=" * 72)
    print()

    tick = 0
    try:
        while True:
            await asyncio.sleep(PREDICTION_INTERVAL)
            tick += 1

            # Compute features
            features = engine.compute_features()

            # Make prediction
            pred = predictor.predict(features)

            # Check accuracy of old predictions
            predictor.check_accuracy()

            # Display
            price = features.get("btc_price", 0)
            ob_imb = features.get("ob_imb_10", 0)
            trade_imb = features.get("btc_imb_5s", 0)
            spread = features.get("ob_spread_bps", 0)
            stats = predictor.get_stats()

            status_parts = [
                f"OB:{'▲' if orderbook.connected else '✗'}",
                f"BTC:{'▲' if btc_trades.connected else '✗'}",
                f"ETH:{'▲' if eth_trades.connected else '✗'}",
                f"MP:{'▲' if mempool.connected else '✗'}",
            ]

            if pred["signal"] in ("STRONG", "MODERATE"):
                emoji = "🟢" if pred["signal"] == "STRONG" else "🟡"
                reasons_str = " | ".join(pred["reasons"][:3])
                print(
                    f"  {emoji} {pred['direction']:<4} {pred['confidence']:.0%} "
                    f"${price:>10,.2f} | {reasons_str}"
                )
            elif tick % 10 == 0:
                # Periodic status every 10 seconds
                wr_str = f"{stats['win_rate']:.1%}" if stats['checked'] > 0 else "--"
                print(
                    f"  ⚫ HOLD  ${price:>10,.2f} | "
                    f"OB:{ob_imb:+.2f} Trd:{trade_imb:+.2f} Sprd:{spread:.1f}bps | "
                    f"Signals:{stats['signals']} WR:{wr_str} ({stats['correct']}/{stats['checked']}) | "
                    f"{' '.join(status_parts)}"
                )

    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()

        stats = predictor.get_stats()
        print(f"\n{'=' * 72}")
        print(f"  SESSION SUMMARY")
        print(f"{'=' * 72}")
        print(f"  Signals emitted:  {stats['signals']}")
        print(f"  Checked:          {stats['checked']}")
        print(f"  Correct:          {stats['correct']}")
        print(f"  Win Rate:         {stats['win_rate']:.1%}" if stats['checked'] > 0 else "  Win Rate: N/A")
        print(f"  Log: {predictor._log_path}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Streaming Alpha Engine")
    parser.add_argument("--self-train", action="store_true",
                        help="Train models on recent data before starting")
    args = parser.parse_args()

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        print("\n\nShutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(main_loop(self_train=args.self_train))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
