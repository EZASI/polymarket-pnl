"""
Data fetchers for various market data sources.

Provides both live API fetching and simulated data for backtesting.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
import json


@dataclass
class OrderBookLevel:
    """Single level in an order book."""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    exchange: str

    def mid_price(self) -> float:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2

    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    def imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.

        OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        Returns value in [-1, 1]

        Academic basis: Cont, Kukanov, Stoikov (2014)
        """
        bid_vol = sum(b.quantity for b in self.bids[:levels])
        ask_vol = sum(a.quantity for a in self.asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    exchange: str


@dataclass
class FundingRate:
    """Perpetual futures funding rate data."""
    timestamp: datetime
    symbol: str
    exchange: str
    rate: float  # Per-period rate (usually 8h)
    annualized_rate: float  # Annualized for comparison
    next_funding_time: datetime


@dataclass
class PolymarketOutcome:
    """Polymarket market outcome."""
    token_id: str
    outcome: str  # e.g., "Yes", "No", "Up", "Down"
    price: float  # 0-1 probability
    volume_24h: float


@dataclass
class PolymarketMarket:
    """Polymarket market data."""
    condition_id: str
    question: str
    outcomes: List[PolymarketOutcome]
    end_date: datetime
    volume_total: float
    liquidity: float
    category: str


@dataclass
class SentimentData:
    """Social sentiment data."""
    timestamp: datetime
    symbol: str
    platform: str
    sentiment_score: float  # -1 to 1
    volume: int  # Number of mentions
    influential_accounts: int


@dataclass
class LiquidationEvent:
    """Liquidation event data."""
    timestamp: datetime
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    quantity: float
    price: float
    value_usd: float


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""

    @abstractmethod
    async def fetch(self) -> any:
        """Fetch data from source."""
        pass


class SimulatedOrderBookFetcher(DataFetcher):
    """
    Simulated order book data for backtesting.

    Generates realistic order book dynamics with:
    - Mean-reverting mid price
    - Time-varying volatility
    - Realistic bid-ask spreads
    - Order book imbalance that correlates with future price moves
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        exchange: str = "binance",
        base_price: float = 50000.0,
        volatility: float = 0.02,
        spread_bps: float = 5.0,
        levels: int = 10,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.base_price = base_price
        self.current_price = base_price
        self.volatility = volatility
        self.spread_bps = spread_bps
        self.levels = levels
        self._last_imbalance = 0.0

    async def fetch(self) -> OrderBook:
        """Generate simulated order book."""
        # Random walk with mean reversion
        drift = -0.01 * (self.current_price - self.base_price) / self.base_price
        shock = random.gauss(0, self.volatility)
        self.current_price *= (1 + drift + shock)

        spread = self.current_price * self.spread_bps / 10000
        mid = self.current_price

        # Generate levels with realistic depth
        bids = []
        asks = []

        # Occasionally generate extreme imbalances (realistic market dynamics)
        extreme_imbalance = 0
        if random.random() < 0.15:  # 15% chance of extreme imbalance
            extreme_imbalance = random.choice([-1, 1]) * random.uniform(0.5, 0.9)

        for i in range(self.levels):
            # Price levels
            bid_price = mid - spread/2 - i * spread * 0.5
            ask_price = mid + spread/2 + i * spread * 0.5

            # Quantity with some randomness (larger at better prices)
            base_qty = random.uniform(0.5, 2.0)
            bid_qty = base_qty * (1 + 0.2 * i + random.uniform(-0.3, 0.3))
            ask_qty = base_qty * (1 + 0.2 * i + random.uniform(-0.3, 0.3))

            # Add imbalance tendency (for realistic dynamics)
            imbalance_factor = 1 + 0.3 * self._last_imbalance
            bid_qty *= imbalance_factor
            ask_qty *= (2 - imbalance_factor)

            # Apply extreme imbalance when triggered
            if extreme_imbalance > 0:
                bid_qty *= (1 + extreme_imbalance)
            elif extreme_imbalance < 0:
                ask_qty *= (1 - extreme_imbalance)

            bids.append(OrderBookLevel(bid_price, max(0.1, bid_qty), 'bid'))
            asks.append(OrderBookLevel(ask_price, max(0.1, ask_qty), 'ask'))

        book = OrderBook(
            symbol=self.symbol,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            exchange=self.exchange,
        )

        # Store imbalance for next iteration (persistence)
        self._last_imbalance = 0.9 * self._last_imbalance + 0.1 * book.imbalance()

        return book

    def set_price(self, price: float):
        """Set current price (for backtesting)."""
        self.current_price = price


class SimulatedOHLCVFetcher(DataFetcher):
    """
    Simulated OHLCV data for backtesting.

    Generates realistic price dynamics with:
    - Trending and mean-reverting regimes
    - Volatility clustering
    - Volume-price correlation
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        exchange: str = "binance",
        base_price: float = 50000.0,
        base_volatility: float = 0.02,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.base_price = base_price
        self.current_price = base_price
        self.base_volatility = base_volatility
        self.current_volatility = base_volatility
        self._regime = "neutral"  # trending_up, trending_down, neutral

    async def fetch(self, periods: int = 100, timeframe: str = "1h") -> List[OHLCV]:
        """Generate simulated OHLCV data."""
        candles = []
        price = self.base_price
        timestamp = datetime.utcnow() - timedelta(hours=periods)

        # Timeframe to timedelta
        tf_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        delta = tf_map.get(timeframe, timedelta(hours=1))

        for i in range(periods):
            # Regime switching
            if random.random() < 0.05:
                self._regime = random.choice(["trending_up", "trending_down", "neutral"])

            # Volatility clustering (GARCH-like)
            vol_shock = random.gauss(0, 0.3)
            self.current_volatility = 0.9 * self.current_volatility + \
                                      0.1 * self.base_volatility * (1 + vol_shock)

            # Price movement based on regime
            if self._regime == "trending_up":
                drift = 0.001
            elif self._regime == "trending_down":
                drift = -0.001
            else:
                drift = 0

            returns = drift + random.gauss(0, self.current_volatility)
            close = price * (1 + returns)

            # Generate OHLC from close
            intra_vol = self.current_volatility * 0.5
            high = close * (1 + abs(random.gauss(0, intra_vol)))
            low = close * (1 - abs(random.gauss(0, intra_vol)))
            open_price = price

            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Volume correlated with volatility and price change
            base_volume = 1000
            vol_factor = 1 + 2 * (self.current_volatility / self.base_volatility - 1)
            price_change_factor = 1 + 5 * abs(returns)
            volume = base_volume * vol_factor * price_change_factor * random.uniform(0.5, 1.5)

            candles.append(OHLCV(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                symbol=self.symbol,
                exchange=self.exchange,
            ))

            price = close
            timestamp += delta

        self.current_price = price
        return candles


class SimulatedFundingRateFetcher(DataFetcher):
    """
    Simulated funding rate data.

    Academic basis:
    - He et al.: Mean absolute deviations 60-90% annualized
    - Extreme rates (100%+) don't reliably predict reversals
    """

    def __init__(
        self,
        symbol: str = "BTC-PERP",
        exchange: str = "binance",
    ):
        self.symbol = symbol
        self.exchange = exchange
        self._base_rate = 0.0001  # 0.01% per 8h = ~10% annualized
        self._current_rate = self._base_rate

    async def fetch(self) -> FundingRate:
        """Generate simulated funding rate."""
        # Mean-reverting funding rate with occasional extremes
        mean_reversion = -0.1 * (self._current_rate - self._base_rate)
        shock = random.gauss(0, 0.0002)

        # Occasional extreme rates
        if random.random() < 0.02:
            shock += random.choice([-1, 1]) * random.uniform(0.001, 0.005)

        self._current_rate = self._current_rate + mean_reversion + shock

        # Annualize (3 funding periods per day * 365)
        annualized = self._current_rate * 3 * 365 * 100

        now = datetime.utcnow()
        # Next funding at 00:00, 08:00, or 16:00 UTC
        hour = now.hour
        next_funding_hour = ((hour // 8) + 1) * 8
        if next_funding_hour >= 24:
            next_funding_hour = 0
            next_day = now.date() + timedelta(days=1)
        else:
            next_day = now.date()

        next_funding = datetime.combine(
            next_day,
            datetime.min.time().replace(hour=next_funding_hour)
        )

        return FundingRate(
            timestamp=now,
            symbol=self.symbol,
            exchange=self.exchange,
            rate=self._current_rate,
            annualized_rate=annualized,
            next_funding_time=next_funding,
        )

    def set_rate(self, rate: float):
        """Set funding rate (for backtesting scenarios)."""
        self._current_rate = rate


class SimulatedPolymarketFetcher(DataFetcher):
    """
    Simulated Polymarket data for crypto prediction markets.

    Generates realistic prediction market dynamics with:
    - Probability drift toward resolution
    - Correlation with underlying crypto prices
    - Realistic liquidity and volume
    """

    def __init__(
        self,
        crypto_symbol: str = "BTC",
        timeframe: str = "15m",
    ):
        self.crypto_symbol = crypto_symbol
        self.timeframe = timeframe
        self._markets: Dict[str, PolymarketMarket] = {}
        self._price_history: List[float] = []

    async def fetch(self, crypto_price: float = None) -> List[PolymarketMarket]:
        """
        Generate simulated Polymarket markets.

        Creates markets like "Will BTC be above $X in 15 minutes?"
        """
        if crypto_price:
            self._price_history.append(crypto_price)

        # Keep last 100 prices
        if len(self._price_history) > 100:
            self._price_history = self._price_history[-100:]

        current_price = crypto_price or 50000.0
        markets = []

        # Generate multiple strike prices around current price
        strikes = [
            current_price * 0.995,
            current_price * 0.998,
            current_price * 1.002,
            current_price * 1.005,
        ]

        for strike in strikes:
            # Base probability on distance from current price
            distance_pct = (strike - current_price) / current_price

            # Simple probability model
            # P(price > strike) decreases with higher strike
            base_prob = 0.5 - distance_pct * 10

            # Add noise
            noise = random.gauss(0, 0.05)
            yes_prob = max(0.01, min(0.99, base_prob + noise))
            no_prob = 1 - yes_prob

            # Generate realistic volume and liquidity
            base_volume = 10000 * random.uniform(0.5, 2.0)
            # Higher volume near 50% probability
            prob_factor = 1 + 2 * (0.25 - (yes_prob - 0.5) ** 2)
            volume = base_volume * prob_factor

            market = PolymarketMarket(
                condition_id=f"btc_above_{int(strike)}_{self.timeframe}",
                question=f"Will {self.crypto_symbol} be above ${strike:,.0f} in {self.timeframe}?",
                outcomes=[
                    PolymarketOutcome(
                        token_id=f"yes_{int(strike)}",
                        outcome="Yes",
                        price=yes_prob,
                        volume_24h=volume * yes_prob,
                    ),
                    PolymarketOutcome(
                        token_id=f"no_{int(strike)}",
                        outcome="No",
                        price=no_prob,
                        volume_24h=volume * no_prob,
                    ),
                ],
                end_date=datetime.utcnow() + timedelta(minutes=15),
                volume_total=volume,
                liquidity=volume * 0.1,
                category="crypto",
            )
            markets.append(market)

        return markets


class SimulatedSentimentFetcher(DataFetcher):
    """
    Simulated social sentiment data.

    Academic basis:
    - Renault et al.: Significant relationship up to 15 minutes
    - Effect magnitude is small
    """

    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol
        self._base_sentiment = 0.0
        self._current_sentiment = 0.0

    async def fetch(self, platform: str = "twitter") -> SentimentData:
        """Generate simulated sentiment data."""
        # Mean-reverting sentiment with occasional spikes
        mean_reversion = -0.1 * self._current_sentiment
        noise = random.gauss(0, 0.1)

        # Occasional sentiment spikes
        if random.random() < 0.05:
            noise += random.choice([-1, 1]) * random.uniform(0.3, 0.6)

        self._current_sentiment = max(-1, min(1,
            self._current_sentiment + mean_reversion + noise
        ))

        # Volume correlates with absolute sentiment
        base_volume = 1000
        volume = int(base_volume * (1 + 2 * abs(self._current_sentiment)) *
                     random.uniform(0.5, 1.5))

        return SentimentData(
            timestamp=datetime.utcnow(),
            symbol=self.symbol,
            platform=platform,
            sentiment_score=self._current_sentiment,
            volume=volume,
            influential_accounts=int(volume * 0.01 * random.uniform(0.5, 1.5)),
        )

    def set_sentiment(self, sentiment: float):
        """Set sentiment (for backtesting scenarios)."""
        self._current_sentiment = max(-1, min(1, sentiment))


class SimulatedLiquidationFetcher(DataFetcher):
    """
    Simulated liquidation data.

    Academic basis:
    - Cheng et al. (2021): 3.51% daily forced liquidations
    - Ali (2025): 86x acceleration during cascades
    """

    def __init__(
        self,
        symbol: str = "BTC-PERP",
        exchange: str = "binance",
    ):
        self.symbol = symbol
        self.exchange = exchange
        self._cascade_mode = False
        self._cascade_remaining = 0

    async def fetch(
        self,
        price: float = 50000.0,
        price_change_pct: float = 0.0,
    ) -> List[LiquidationEvent]:
        """Generate simulated liquidation events."""
        events = []

        # Base liquidation rate
        base_rate = 0.01  # 1% chance per period

        # Increase rate with price volatility
        vol_factor = 1 + 10 * abs(price_change_pct)

        # Cascade dynamics
        if self._cascade_remaining > 0:
            self._cascade_remaining -= 1
            vol_factor *= 10  # 10x during cascade

        # Trigger new cascade on large moves
        if abs(price_change_pct) > 0.03 and not self._cascade_mode:
            self._cascade_mode = True
            self._cascade_remaining = random.randint(3, 10)

        if self._cascade_remaining == 0:
            self._cascade_mode = False

        # Generate liquidations
        num_liquidations = int(random.expovariate(1 / (base_rate * vol_factor * 5)))

        for _ in range(num_liquidations):
            # Liquidation side opposite to price direction
            side = "long" if price_change_pct < 0 else "short"

            # Liquidation size follows power law
            base_size = 10000  # $10k base
            size = base_size * (random.paretovariate(1.5))

            events.append(LiquidationEvent(
                timestamp=datetime.utcnow(),
                symbol=self.symbol,
                exchange=self.exchange,
                side=side,
                quantity=size / price,
                price=price,
                value_usd=size,
            ))

        return events


class HistoricalDataLoader:
    """
    Loads and manages historical data for backtesting.

    Can load from:
    - CSV files
    - JSON files
    - Simulated data generation
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._cache: Dict[str, any] = {}

    def generate_historical_ohlcv(
        self,
        symbol: str = "BTC-USD",
        start_date: datetime = None,
        end_date: datetime = None,
        timeframe: str = "1h",
    ) -> List[OHLCV]:
        """Generate historical OHLCV data for backtesting."""
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        # Timeframe to periods
        tf_hours = {
            "1m": 1/60,
            "5m": 5/60,
            "15m": 0.25,
            "1h": 1,
            "4h": 4,
            "1d": 24,
        }
        hours_per_period = tf_hours.get(timeframe, 1)
        total_hours = (end_date - start_date).total_seconds() / 3600
        periods = int(total_hours / hours_per_period)

        fetcher = SimulatedOHLCVFetcher(symbol=symbol)

        # Run synchronously for historical data
        loop = asyncio.new_event_loop()
        candles = loop.run_until_complete(fetcher.fetch(periods=periods, timeframe=timeframe))
        loop.close()

        # Adjust timestamps to match date range
        delta = timedelta(hours=hours_per_period)
        current_time = start_date
        for candle in candles:
            candle.timestamp = current_time
            current_time += delta

        return candles

    def generate_correlated_polymarket_data(
        self,
        ohlcv_data: List[OHLCV],
        resolution_minutes: int = 15,
    ) -> List[Tuple[datetime, PolymarketMarket, bool]]:
        """
        Generate Polymarket data correlated with OHLCV.

        Returns: List of (timestamp, market, resolved_yes)
        """
        results = []
        fetcher = SimulatedPolymarketFetcher(timeframe=f"{resolution_minutes}m")

        for i, candle in enumerate(ohlcv_data):
            if i + 1 >= len(ohlcv_data):
                break

            # Create market at this candle
            loop = asyncio.new_event_loop()
            markets = loop.run_until_complete(fetcher.fetch(crypto_price=candle.close))
            loop.close()

            # Determine resolution based on next candle
            future_price = ohlcv_data[i + 1].close

            for market in markets:
                # Extract strike from market question
                strike = float(market.question.split("$")[1].split(" ")[0].replace(",", ""))
                resolved_yes = future_price > strike

                results.append((candle.timestamp, market, resolved_yes))

        return results
