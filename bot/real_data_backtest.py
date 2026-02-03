#!/usr/bin/env python3
"""
Real Data Backtest - Fetch actual Polymarket and CEX data

This module fetches real historical data to validate the strategy
against actual market outcomes.
"""

import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class CryptoPrice:
    """Historical crypto price data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PolymarketMarket:
    """Polymarket market data."""
    condition_id: str
    question: str
    outcome_yes_price: float
    outcome_no_price: float
    volume: float
    end_date: datetime
    resolved: bool
    resolution: Optional[str]


class RealDataFetcher:
    """Fetch real market data from APIs."""

    # Free crypto price API (CoinGecko)
    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Polymarket API
    POLYMARKET_API = "https://data-api.polymarket.com"

    # Polymarket CLOB API (for orderbook data)
    POLYMARKET_CLOB_API = "https://clob.polymarket.com"

    # Alternative: Binance public API
    BINANCE_API = "https://api.binance.com/api/v3"

    def __init__(self):
        self._cache = {}

    def _fetch_url(self, url: str, retries: int = 3) -> Optional[Dict]:
        """Fetch JSON from URL with retries."""
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/json',
                    }
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as e:
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None
            except urllib.error.URLError as e:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None
        return None

    def get_binance_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        limit: int = 1000,
    ) -> List[CryptoPrice]:
        """Fetch historical klines from Binance."""
        url = f"{self.BINANCE_API}/klines?symbol={symbol}&interval={interval}&limit={limit}"

        data = self._fetch_url(url)
        if not data:
            return []

        prices = []
        for candle in data:
            prices.append(CryptoPrice(
                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
            ))

        return prices

    def get_binance_orderbook(self, symbol: str = "BTCUSDT", limit: int = 20) -> Optional[Dict]:
        """Fetch current orderbook from Binance."""
        url = f"{self.BINANCE_API}/depth?symbol={symbol}&limit={limit}"
        return self._fetch_url(url)

    def get_coingecko_prices(self, coin_id: str = "bitcoin", days: int = 7) -> List[CryptoPrice]:
        """Fetch historical prices from CoinGecko (free, no API key required)."""
        url = f"{self.COINGECKO_API}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"

        data = self._fetch_url(url)
        if not data:
            return []

        prices = []
        for candle in data:
            # CoinGecko OHLC format: [timestamp, open, high, low, close]
            prices.append(CryptoPrice(
                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=0.0,  # CoinGecko OHLC doesn't include volume
            ))

        return prices

    def get_coingecko_market_data(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """Fetch current market data from CoinGecko."""
        url = f"{self.COINGECKO_API}/coins/{coin_id}?localization=false&tickers=false&community_data=false&developer_data=false"
        return self._fetch_url(url)

    def calculate_obi(self, orderbook: Dict, levels: int = 5) -> float:
        """Calculate Order Book Imbalance from Binance orderbook."""
        if not orderbook:
            return 0.0

        bids = orderbook.get('bids', [])[:levels]
        asks = orderbook.get('asks', [])[:levels]

        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def get_polymarket_markets(self) -> List[Dict]:
        """Fetch active Polymarket markets."""
        url = f"{self.POLYMARKET_API}/markets"
        data = self._fetch_url(url)
        return data if data else []

    def get_polymarket_crypto_markets(self) -> List[Dict]:
        """Filter for crypto-related prediction markets."""
        markets = self.get_polymarket_markets()

        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol']
        crypto_markets = []

        for market in markets:
            question = market.get('question', '').lower()
            if any(kw in question for kw in crypto_keywords):
                crypto_markets.append(market)

        return crypto_markets

    def get_polymarket_clob_orderbook(self, token_id: str) -> Optional[Dict]:
        """Fetch orderbook from Polymarket CLOB API."""
        # Try different endpoint patterns
        urls = [
            f"{self.POLYMARKET_CLOB_API}/book?token_id={token_id}",
            f"{self.POLYMARKET_CLOB_API}/orderbook/{token_id}",
            f"{self.POLYMARKET_CLOB_API}/order-book?market={token_id}",
        ]
        for url in urls:
            result = self._fetch_url(url)
            if result and (result.get('bids') or result.get('asks')):
                return result
        return None

    def get_polymarket_clob_markets(self) -> List[Dict]:
        """Fetch active markets from Polymarket CLOB API."""
        url = f"{self.POLYMARKET_CLOB_API}/markets"
        data = self._fetch_url(url)
        return data if data else []

    def calculate_polymarket_obi(self, orderbook: Dict, levels: int = 5) -> float:
        """Calculate Order Book Imbalance from Polymarket CLOB orderbook."""
        if not orderbook:
            return 0.0

        bids = orderbook.get('bids', [])[:levels]
        asks = orderbook.get('asks', [])[:levels]

        # Polymarket format: {"price": "0.50", "size": "100"}
        bid_vol = sum(float(b.get('size', 0)) for b in bids)
        ask_vol = sum(float(a.get('size', 0)) for a in asks)

        total = bid_vol + ask_vol
        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total


def run_real_data_test():
    """Test with real market data."""
    print("=" * 70)
    print("REAL DATA BACKTEST")
    print("Fetching actual market data from public APIs")
    print("=" * 70)
    print()

    fetcher = RealDataFetcher()

    # 1. Fetch real BTC price data (try Binance first, then CoinGecko)
    print("1. Fetching BTC price history...")
    btc_prices = fetcher.get_binance_klines("BTCUSDT", "15m", 100)

    if not btc_prices:
        print("   Binance unavailable, trying CoinGecko...")
        btc_prices = fetcher.get_coingecko_prices("bitcoin", days=7)

    if btc_prices:
        print(f"   Got {len(btc_prices)} candles")
        print(f"   Latest: ${btc_prices[-1].close:,.2f} at {btc_prices[-1].timestamp}")
        print(f"   Range: {btc_prices[0].timestamp} to {btc_prices[-1].timestamp}")
    else:
        print("   Failed to fetch BTC prices from all sources")
        return

    # 2. Fetch current orderbook and calculate OBI
    print("\n2. Fetching current BTC orderbook...")
    orderbook = fetcher.get_binance_orderbook("BTCUSDT", 20)
    obi = 0.0  # Default if unavailable

    if orderbook:
        obi = fetcher.calculate_obi(orderbook, levels=5)
        print(f"   Current OBI (5 levels): {obi:.4f}")

        if abs(obi) > 0.3:
            direction = "BULLISH" if obi > 0 else "BEARISH"
            print(f"   Signal: {direction} (OBI > 0.3 threshold)")
        else:
            print(f"   Signal: NEUTRAL (OBI below threshold)")
    else:
        print("   Orderbook unavailable (API restricted)")
        print("   Using price momentum as OBI proxy for backtest")

    # 3. Fetch ETH data
    print("\n3. Fetching ETH price history...")
    eth_prices = fetcher.get_binance_klines("ETHUSDT", "15m", 100)

    if not eth_prices:
        print("   Binance unavailable, trying CoinGecko...")
        eth_prices = fetcher.get_coingecko_prices("ethereum", days=7)

    if eth_prices:
        print(f"   Got {len(eth_prices)} candles")
        print(f"   Latest: ${eth_prices[-1].close:,.2f}")
    else:
        print("   ETH prices unavailable")

    # 4. Calculate historical OBI signals and outcomes
    print("\n4. Backtesting OBI signals on historical data...")
    print("-" * 50)

    # For each 15-minute period, simulate what would have happened
    # if we traded based on the previous candle's price movement

    wins = 0
    losses = 0
    total_pnl = 0.0
    trades = []

    # Use price momentum as proxy for OBI (since we don't have historical orderbook)
    for i in range(10, len(btc_prices) - 1):
        # Calculate momentum (proxy for OBI)
        recent_returns = []
        for j in range(i-5, i):
            ret = (btc_prices[j].close - btc_prices[j].open) / btc_prices[j].open
            recent_returns.append(ret)

        momentum = sum(recent_returns)

        # Only trade on strong momentum (proxy for high OBI)
        if abs(momentum) < 0.002:  # 0.2% threshold
            continue

        direction = 1 if momentum > 0 else -1
        entry_price = btc_prices[i].close
        exit_price = btc_prices[i + 1].close

        actual_direction = 1 if exit_price > entry_price else -1
        won = direction == actual_direction

        # Calculate P&L (simplified)
        position_size = 1000  # $1000 per trade
        if won:
            pnl = position_size * abs(exit_price - entry_price) / entry_price
        else:
            pnl = -position_size * 0.03  # Lose 3% on loss (simplified)

        pnl -= position_size * 0.03  # Transaction costs

        if won:
            wins += 1
        else:
            losses += 1
        total_pnl += pnl

        trades.append({
            'time': btc_prices[i].timestamp,
            'momentum': momentum,
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'entry': entry_price,
            'exit': exit_price,
            'won': won,
            'pnl': pnl,
        })

    # Results
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    if total_trades > 0:
        print(f"\n   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total P&L: ${total_pnl:+,.2f}")
        print(f"   Avg P&L/Trade: ${total_pnl/total_trades:+,.2f}")

        print(f"\n   Sample Trades:")
        for t in trades[:5]:
            status = "WIN" if t['won'] else "LOSS"
            print(f"   {status} {t['time']} | {t['direction']} | "
                  f"Entry: ${t['entry']:,.0f} -> Exit: ${t['exit']:,.0f} | "
                  f"P&L: ${t['pnl']:+,.2f}")
    else:
        print("   No trades generated (momentum too low)")

    # 5. Try to fetch Polymarket markets
    print("\n5. Fetching Polymarket crypto markets...")
    try:
        crypto_markets = fetcher.get_polymarket_crypto_markets()
        print(f"   Found {len(crypto_markets)} crypto-related markets from data API")

        if crypto_markets:
            print("\n   Sample Markets:")
            for market in crypto_markets[:5]:
                question = market.get('question', 'N/A')[:60]
                print(f"   - {question}...")
    except Exception as e:
        print(f"   Could not fetch Polymarket data API: {e}")

    # 6. Try Polymarket CLOB API for orderbook data
    print("\n6. Fetching Polymarket CLOB markets...")
    try:
        clob_response = fetcher.get_polymarket_clob_markets()

        # Handle different response formats
        clob_markets = []
        if isinstance(clob_response, list):
            clob_markets = clob_response
        elif isinstance(clob_response, dict):
            # Could be paginated or nested
            clob_markets = clob_response.get('data', clob_response.get('markets', []))
            if not clob_markets and 'next_cursor' in clob_response:
                # It's paginated, markets are at top level
                clob_markets = [v for k, v in clob_response.items()
                               if isinstance(v, dict) and 'question' in v]

        if clob_markets:
            print(f"   Found {len(clob_markets)} active CLOB markets")

            # Find crypto markets and analyze their orderbooks
            crypto_clob = []
            for m in clob_markets[:100]:
                if not isinstance(m, dict):
                    continue
                question = str(m.get('question', '')).lower()
                if any(kw in question for kw in ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto']):
                    crypto_clob.append(m)

            if crypto_clob:
                print(f"   Found {len(crypto_clob)} crypto CLOB markets")
                markets_to_analyze = crypto_clob[:5]
            else:
                print("   No crypto markets - analyzing other active markets")
                markets_to_analyze = clob_markets[:5]

            print("\n   Sample Polymarket Orderbook Analysis:")
            analyzed = 0
            for market in clob_markets[:50]:  # Try more markets
                if not market.get('enable_order_book') or not market.get('active'):
                    continue
                if market.get('closed') or market.get('archived'):
                    continue

                question = market.get('question', 'N/A')
                if not question or question == 'N/A':
                    continue

                tokens = market.get('tokens', [])
                if not tokens or not isinstance(tokens, list):
                    continue

                # Skip resolved markets (any token has winner=True)
                is_resolved = any(t.get('winner') for t in tokens if isinstance(t, dict))
                if is_resolved:
                    continue

                # Get token_id from first token
                first_token = tokens[0] if tokens else {}
                token_id = first_token.get('token_id', '') if isinstance(first_token, dict) else ''

                if token_id:
                    ob = fetcher.get_polymarket_clob_orderbook(token_id)
                    if ob and (ob.get('bids') or ob.get('asks')):
                        poly_obi = fetcher.calculate_polymarket_obi(ob, 5)
                        signal_str = '(SIGNAL!)' if abs(poly_obi) > 0.3 else ''
                        print(f"   - {question[:50]}...")
                        print(f"     OBI: {poly_obi:+.4f} {signal_str}")
                        analyzed += 1
                        if analyzed >= 5:
                            break

            if analyzed == 0:
                # Debug: check the tokens structure
                for market in clob_markets[:5]:
                    if market.get('active') and market.get('tokens'):
                        tokens = market.get('tokens', [])
                        print(f"   Tokens structure: {tokens[:2] if isinstance(tokens, list) else tokens}")
                        if isinstance(tokens, list) and len(tokens) > 0:
                            first_token = tokens[0]
                            if isinstance(first_token, dict):
                                print(f"   Token keys: {list(first_token.keys())}")
                                token_id = first_token.get('token_id', '')
                                if token_id:
                                    print(f"   Trying token_id: {token_id[:50]}...")
                        break
                print("   Could not retrieve orderbook data for active markets")
        else:
            print(f"   CLOB API returned: {list(clob_response.keys()) if isinstance(clob_response, dict) else type(clob_response)}")
    except Exception as e:
        print(f"   Could not fetch CLOB data: {e}")

    print("\n" + "=" * 70)
    print("REAL DATA ANALYSIS COMPLETE")
    print("=" * 70)

    # Summary
    obi_status = 'Signal' if abs(obi) > 0.3 else 'No signal'
    if total_trades > 0:
        win_status = 'exceeds' if win_rate > 0.53 else 'is below'
        profit_status = 'PROFITABLE' if total_pnl > 0 else 'NOT PROFITABLE'
        print(f"""
KEY FINDINGS FROM REAL DATA:

1. BTC Current Price: ${btc_prices[-1].close:,.2f}
2. Current OBI: {obi:.4f} ({obi_status})
3. Historical Momentum Trading:
   - Win Rate: {win_rate:.1%} on {total_trades} trades
   - Net P&L: ${total_pnl:+,.2f}

INTERPRETATION:
- Win rate of {win_rate:.1%} {win_status} breakeven (~53% needed)
- {profit_status} after 3% transaction costs

NOTE: This uses momentum as a proxy for OBI since we don't have
historical orderbook data. Real OBI signals may perform differently.
""")
    else:
        print(f"""
KEY FINDINGS FROM REAL DATA:

1. BTC Current Price: ${btc_prices[-1].close:,.2f}
2. Current OBI: {obi:.4f} ({obi_status})
3. Historical Momentum Trading: No trades generated

NOTE: Insufficient momentum signals in the data to generate trades.
Try with a longer time period or lower momentum threshold.
""")

    return {
        'btc_price': btc_prices[-1].close,
        'obi': obi,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'trades': total_trades,
    }


def continuous_signal_monitor(duration_minutes: int = 5):
    """Monitor real signals in real-time."""
    print("=" * 70)
    print("REAL-TIME SIGNAL MONITOR")
    print(f"Monitoring for {duration_minutes} minutes...")
    print("=" * 70)
    print()

    fetcher = RealDataFetcher()
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    signals_detected = []

    while datetime.now() < end_time:
        # Get current orderbook
        btc_ob = fetcher.get_binance_orderbook("BTCUSDT", 20)
        eth_ob = fetcher.get_binance_orderbook("ETHUSDT", 20)

        if btc_ob:
            btc_obi = fetcher.calculate_obi(btc_ob, 5)
            btc_bid = float(btc_ob['bids'][0][0]) if btc_ob['bids'] else 0

            signal = None
            if btc_obi > 0.3:
                signal = "BTC BULLISH"
            elif btc_obi < -0.3:
                signal = "BTC BEARISH"

            timestamp = datetime.now().strftime("%H:%M:%S")

            if signal:
                signals_detected.append({
                    'time': timestamp,
                    'asset': 'BTC',
                    'obi': btc_obi,
                    'signal': signal,
                    'price': btc_bid,
                })
                print(f"[{timestamp}] 🚨 SIGNAL: {signal} | OBI: {btc_obi:+.4f} | Price: ${btc_bid:,.2f}")
            else:
                print(f"[{timestamp}] BTC: OBI={btc_obi:+.4f} (no signal) | ${btc_bid:,.2f}")

        if eth_ob:
            eth_obi = fetcher.calculate_obi(eth_ob, 5)
            eth_bid = float(eth_ob['bids'][0][0]) if eth_ob['bids'] else 0

            if abs(eth_obi) > 0.3:
                signal = "ETH BULLISH" if eth_obi > 0 else "ETH BEARISH"
                signals_detected.append({
                    'time': timestamp,
                    'asset': 'ETH',
                    'obi': eth_obi,
                    'signal': signal,
                    'price': eth_bid,
                })
                print(f"[{timestamp}] 🚨 SIGNAL: {signal} | OBI: {eth_obi:+.4f} | Price: ${eth_bid:,.2f}")

        # Wait before next check
        time.sleep(15)  # Check every 15 seconds

    print("\n" + "-" * 50)
    print(f"Monitoring complete. Signals detected: {len(signals_detected)}")

    if signals_detected:
        print("\nSignal Summary:")
        for s in signals_detected:
            print(f"  [{s['time']}] {s['signal']} @ ${s['price']:,.2f}")

    return signals_detected


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        continuous_signal_monitor(duration)
    else:
        run_real_data_test()
