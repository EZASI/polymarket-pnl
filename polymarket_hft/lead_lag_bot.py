"""
Lead–Lag Bot — maker-only quoting on Polymarket crypto binary markets.
======================================================================

Architecture
------------
1. **Binance aggTrade WebSocket** — receives real-time BTC/ETH/SOL trades,
   computes rolling 30s return, OFI, and 1m realized vol.
2. **Polymarket Gamma poller** — discovers active 5m/15m crypto markets,
   fetches CLOB orderbook snapshots every few seconds.
3. **State engine** — for each active market, maintains a ``MarketState``
   and recomputes p_theo + net EV on every tick.
4. **Quote decision** — when net EV exceeds threshold, logs a quote
   intention (actual Polymarket order submission is wired in a later phase
   when API keys are configured).

The bot stores all state in a shared dict (``active_states``) that the
dashboard reads via ``/api/hft/states``.

Usage
-----
    python -m polymarket_hft.lead_lag_bot           # default: BTC
    python -m polymarket_hft.lead_lag_bot --underlying BTC ETH SOL
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_hft.market_model import MarketMeta, get_taker_fee, get_maker_rebate
from polymarket_hft.state import MarketState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lead_lag_bot")

# ---------------------------------------------------------------------------
# Globals (shared with dashboard)
# ---------------------------------------------------------------------------

active_states: dict[str, MarketState] = {}   # market_id → MarketState
bot_stats: dict[str, Any] = {
    "started_at": None,
    "ticks_processed": 0,
    "markets_tracked": 0,
    "quotes_generated": 0,
    "errors": 0,
    "running": False,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINANCE_WS = "wss://fstream.binance.com/ws"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

MIN_EDGE_BPS = 50        # minimum net EV (bps) to generate a quote
POLL_INTERVAL_S = 8       # Gamma / orderbook poll
RETURN_WINDOW_S = 30      # rolling return window
VOL_WINDOW_S = 60         # realised vol window
MAX_MARKETS = 6           # track at most N markets

SYMBOL_MAP = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
}

CRYPTO_KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
SHORT_TERM_KEYWORDS = ["5 min", "5-min", "15 min", "15-min", "5m", "15m", "minute"]


# ---------------------------------------------------------------------------
# CEX price tracker (Binance aggTrade)
# ---------------------------------------------------------------------------

class CexTracker:
    """Tracks rolling return, OFI, and realised vol from Binance aggTrades."""

    def __init__(self) -> None:
        # {symbol: deque of (unix_ts, price, qty, is_buyer_maker)}
        self.ticks: dict[str, collections.deque] = {}
        self.latest_mid: dict[str, float] = {}

    def on_agg_trade(self, symbol: str, price: float, qty: float, is_buyer_maker: bool) -> None:
        symbol = symbol.upper()
        now = time.time()
        if symbol not in self.ticks:
            self.ticks[symbol] = collections.deque(maxlen=5000)
        self.ticks[symbol].append((now, price, qty, is_buyer_maker))
        self.latest_mid[symbol] = price

    def _prune(self, symbol: str, window_s: float) -> None:
        cutoff = time.time() - window_s
        dq = self.ticks.get(symbol)
        if dq:
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def rolling_return(self, symbol: str, window_s: float = RETURN_WINDOW_S) -> float:
        """Log return over the last `window_s` seconds."""
        self._prune(symbol, window_s * 2)
        dq = self.ticks.get(symbol)
        if not dq or len(dq) < 2:
            return 0.0
        cutoff = time.time() - window_s
        # Find earliest tick within window
        first_price = dq[0][1]
        for ts, p, _, _ in dq:
            if ts >= cutoff:
                first_price = p
                break
        last_price = dq[-1][1]
        if first_price <= 0:
            return 0.0
        return math.log(last_price / first_price)

    def ofi(self, symbol: str, window_s: float = RETURN_WINDOW_S) -> float:
        """Order flow imbalance: net buy volume over window."""
        self._prune(symbol, window_s * 2)
        dq = self.ticks.get(symbol)
        if not dq:
            return 0.0
        cutoff = time.time() - window_s
        net = 0.0
        for ts, price, qty, is_buyer_maker in dq:
            if ts < cutoff:
                continue
            # is_buyer_maker=True → sell aggressed → negative OFI
            if is_buyer_maker:
                net -= qty
            else:
                net += qty
        return net

    def realized_vol(self, symbol: str, window_s: float = VOL_WINDOW_S) -> float:
        """1-minute realised vol (annualised) using 1-second buckets."""
        self._prune(symbol, window_s * 2)
        dq = self.ticks.get(symbol)
        if not dq or len(dq) < 10:
            return 0.001  # fallback tiny vol

        cutoff = time.time() - window_s
        # Bucket into 1-second intervals
        buckets: dict[int, float] = {}
        for ts, price, _, _ in dq:
            if ts < cutoff:
                continue
            sec = int(ts)
            buckets[sec] = price  # last price in second

        if len(buckets) < 3:
            return 0.001

        sorted_secs = sorted(buckets.keys())
        log_rets = []
        for i in range(1, len(sorted_secs)):
            p0 = buckets[sorted_secs[i - 1]]
            p1 = buckets[sorted_secs[i]]
            if p0 > 0:
                log_rets.append(math.log(p1 / p0))

        if len(log_rets) < 2:
            return 0.001

        mean = sum(log_rets) / len(log_rets)
        var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
        # Annualise: per-second vol → per-year
        vol_per_sec = math.sqrt(var)
        vol_annualised = vol_per_sec * math.sqrt(365.25 * 24 * 3600)
        return max(vol_annualised, 1e-6)


cex_tracker = CexTracker()


# ---------------------------------------------------------------------------
# Binance WebSocket listener
# ---------------------------------------------------------------------------

async def binance_ws_loop(underlyings: list[str]) -> None:
    """Connect to Binance Futures aggTrade streams."""
    streams = "/".join(f"{SYMBOL_MAP[u]}@aggTrade" for u in underlyings if u in SYMBOL_MAP)
    url = f"{BINANCE_WS}/{streams}"
    log.info("Connecting to Binance WS: %s", url)

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, heartbeat=20) as ws:
                    log.info("Binance WS connected")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            symbol = data.get("s", "").upper()
                            price = float(data.get("p", 0))
                            qty = float(data.get("q", 0))
                            is_buyer_maker = data.get("m", False)
                            cex_tracker.on_agg_trade(symbol, price, qty, is_buyer_maker)
                            bot_stats["ticks_processed"] += 1
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception as exc:
            bot_stats["errors"] += 1
            log.error("Binance WS error: %s — reconnecting in 5s", exc)
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Polymarket market discovery + orderbook polling
# ---------------------------------------------------------------------------

def _classify_underlying(text: str) -> str | None:
    t = text.lower()
    if any(kw in t for kw in ["bitcoin", "btc"]):
        return "BTC"
    if any(kw in t for kw in ["ethereum", "eth"]):
        return "ETH"
    if any(kw in t for kw in ["solana", "sol"]):
        return "SOL"
    return None


def _is_crypto_short_term(m: dict[str, Any]) -> bool:
    text = (m.get("question", "") + " " + m.get("description", "")).lower()
    has_crypto = any(kw in text for kw in CRYPTO_KEYWORDS)
    has_short = any(kw in text for kw in SHORT_TERM_KEYWORDS)
    return has_crypto and has_short


def _parse_expiry(m: dict[str, Any]) -> datetime | None:
    for field in ("endDate", "end_date_iso", "expirationDate"):
        val = m.get(field)
        if val:
            try:
                if isinstance(val, str):
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                if isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val, tz=timezone.utc)
            except (ValueError, OSError):
                continue
    return None


async def discover_markets(
    session: aiohttp.ClientSession,
    underlyings: list[str],
) -> list[MarketMeta]:
    """Fetch active crypto short-term markets from Gamma API."""
    out: list[MarketMeta] = []
    try:
        params = {"active": "true", "closed": "false", "limit": 100}
        async with session.get(f"{GAMMA_API}/markets", params=params) as resp:
            if resp.status != 200:
                log.warning("Gamma API returned %d", resp.status)
                return out
            markets = await resp.json()
    except Exception as exc:
        log.error("Gamma API error: %s", exc)
        return out

    for m in markets:
        if not _is_crypto_short_term(m):
            continue
        underlying = _classify_underlying(m.get("question", ""))
        if not underlying or underlying not in underlyings:
            continue

        meta = MarketMeta(
            market_id=m.get("id", m.get("conditionId", "")),
            condition_id=m.get("conditionId", ""),
            question=m.get("question", ""),
            underlying=underlying,
            expiry_ts=_parse_expiry(m),
            strike_price=None,
            strike_type="ATM",
            fee_tier=get_taker_fee(0.5),
            maker_rebate=get_maker_rebate(),
            status="active",
        )
        # Only accept short-term markets (< 24h to expiry)
        tte = meta.time_to_expiry_s
        if tte <= 0 or tte > 86400:
            continue
        out.append(meta)

    # Sort by time-to-expiry (nearest first), cap at MAX_MARKETS
    out.sort(key=lambda m: m.time_to_expiry_s)
    return out[:MAX_MARKETS]


async def fetch_pm_book(
    session: aiohttp.ClientSession,
    token_id: str,
) -> dict[str, Any] | None:
    """Fetch CLOB orderbook for a token."""
    try:
        async with session.get(f"{CLOB_API}/book", params={"token_id": token_id}) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        return None


def _update_pm_state(state: MarketState, book: dict[str, Any] | None) -> None:
    """Update Polymarket-side fields from an orderbook snapshot."""
    if not book:
        return
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    state.pm_yes_bid = float(bids[0]["price"]) if bids else 0.0
    state.pm_yes_ask = float(asks[0]["price"]) if asks else 1.0
    state.pm_no_bid = 1.0 - state.pm_yes_ask
    state.pm_no_ask = 1.0 - state.pm_yes_bid if state.pm_yes_bid > 0 else 1.0
    state.pm_mid = (state.pm_yes_bid + state.pm_yes_ask) / 2.0 if state.pm_yes_bid > 0 else 0.5
    state.pm_depth_bid = sum(float(b.get("size", 0)) for b in bids[:5])
    state.pm_depth_ask = sum(float(a.get("size", 0)) for a in asks[:5])
    state.pm_last_trade_ts = time.time()


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

async def poll_loop(underlyings: list[str]) -> None:
    """Discover markets, fetch orderbooks, update state, generate quotes."""
    log.info("Starting poll loop for %s (interval=%ds)", underlyings, POLL_INTERVAL_S)

    async with aiohttp.ClientSession() as session:
        while bot_stats["running"]:
            try:
                # Discover active markets
                markets = await discover_markets(session, underlyings)
                log.info("Active markets: %d", len(markets))

                # Remove expired/settled states
                live_ids = {m.market_id for m in markets}
                for mid in list(active_states.keys()):
                    if mid not in live_ids:
                        del active_states[mid]

                # Create / update state for each market
                for meta in markets:
                    if meta.market_id not in active_states:
                        active_states[meta.market_id] = MarketState(market=meta)

                    state = active_states[meta.market_id]
                    state.market = meta  # refresh metadata

                    # Fetch PM orderbook
                    book = await fetch_pm_book(session, meta.market_id)
                    _update_pm_state(state, book)

                    # Update CEX state
                    sym = f"{meta.underlying}USDT"
                    state.cex_mid = cex_tracker.latest_mid.get(sym, 0.0)
                    state.cex_return_30s = cex_tracker.rolling_return(sym)
                    state.cex_ofi = cex_tracker.ofi(sym)
                    state.cex_realized_vol_1m = cex_tracker.realized_vol(sym)
                    state.cex_last_trade_ts = time.time()

                    # Recompute p_theo + net EV
                    state.recompute()

                    # Quote decision (log only for now — no actual order submission)
                    if state.edge_yes_bps > MIN_EDGE_BPS:
                        log.info(
                            "QUOTE YES | %s | p_theo=%.3f pm_mid=%.3f edge=%.1f bps | ask=%.3f",
                            meta.underlying, state.p_theo, state.pm_mid,
                            state.edge_yes_bps, state.pm_yes_ask,
                        )
                        bot_stats["quotes_generated"] += 1
                    elif state.edge_no_bps > MIN_EDGE_BPS:
                        log.info(
                            "QUOTE NO  | %s | p_theo=%.3f pm_mid=%.3f edge=%.1f bps | bid=%.3f",
                            meta.underlying, state.p_theo, state.pm_mid,
                            state.edge_no_bps, state.pm_yes_bid,
                        )
                        bot_stats["quotes_generated"] += 1

                    await asyncio.sleep(0.5)  # rate limit between market fetches

                bot_stats["markets_tracked"] = len(active_states)

            except Exception as exc:
                bot_stats["errors"] += 1
                log.error("Poll loop error: %s", exc, exc_info=True)

            await asyncio.sleep(POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_bot(underlyings: list[str]) -> None:
    """Start the bot: Binance WS + Polymarket poll loop."""
    bot_stats["running"] = True
    bot_stats["started_at"] = datetime.now(timezone.utc).isoformat()
    log.info("Lead-lag bot starting for underlyings: %s", underlyings)

    tasks = [
        asyncio.create_task(binance_ws_loop(underlyings)),
        asyncio.create_task(poll_loop(underlyings)),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log.info("Bot shutting down")
    finally:
        bot_stats["running"] = False
        for t in tasks:
            t.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket lead-lag HFT bot")
    parser.add_argument(
        "--underlying", nargs="+", default=["BTC"],
        help="Underlyings to track (BTC, ETH, SOL)",
    )
    args = parser.parse_args()
    asyncio.run(run_bot([u.upper() for u in args.underlying]))


if __name__ == "__main__":
    main()
