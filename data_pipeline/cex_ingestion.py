"""
CEX data ingestion — Binance & Bybit OHLCV + aggTrades for BTC/ETH/SOL.

Provides:
    - REST-based OHLCV backfill (Binance klines endpoint)
    - Batch insert into TimescaleDB via data_pipeline.db
    - Idempotent: safe to run repeatedly (ON CONFLICT DO NOTHING)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import aiohttp

from data_pipeline.db import Database

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance REST klines
# ---------------------------------------------------------------------------

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_KLINE_LIMIT = 1500  # max per request


async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    limit: int = BINANCE_KLINE_LIMIT,
) -> list[dict[str, Any]]:
    """
    Fetch klines from Binance Futures API.
    Returns list of OHLCV dicts in our canonical format.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": limit,
    }
    if end_ms is not None:
        params["endTime"] = end_ms

    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        raw = await resp.json()

    rows: list[dict[str, Any]] = []
    for k in raw:
        # Binance kline format: [open_time, o, h, l, c, vol, close_time, quote_vol, trades, ...]
        rows.append(
            {
                "ts": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "exchange": "binance",
                "symbol": symbol,
                "interval": interval,
                "open": k[1],
                "high": k[2],
                "low": k[3],
                "close": k[4],
                "volume": k[5],
                "quote_volume": k[7],
                "trades_count": k[8],
            }
        )
    return rows


async def backfill_binance_ohlcv(
    db: Database,
    symbol: str,
    interval: str = "1m",
    days: int = 90,
    batch_size: int = BINANCE_KLINE_LIMIT,
) -> int:
    """
    Backfill OHLCV from Binance Futures into TimescaleDB.
    Resumes from the latest stored candle (idempotent).
    Returns total rows inserted.
    """
    # Determine start point
    latest_ts = await db.get_latest_ohlcv_ts("binance", symbol, interval)
    if latest_ts:
        start_dt = latest_ts + timedelta(minutes=1)  # next candle after last stored
        log.info(
            "Resuming %s %s backfill from %s", symbol, interval, start_dt.isoformat()
        )
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        log.info(
            "Starting fresh %s %s backfill from %s",
            symbol,
            interval,
            start_dt.isoformat(),
        )

    end_dt = datetime.now(timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    total_inserted = 0
    cursor_ms = start_ms

    async with aiohttp.ClientSession() as session:
        while cursor_ms < end_ms:
            try:
                rows = await fetch_binance_klines(
                    session,
                    symbol=symbol,
                    interval=interval,
                    start_ms=cursor_ms,
                    end_ms=end_ms,
                    limit=batch_size,
                )
            except aiohttp.ClientError as exc:
                log.error("Binance API error for %s: %s", symbol, exc)
                await asyncio.sleep(2)
                continue

            if not rows:
                break

            inserted = await db.upsert_ohlcv_batch(rows)
            total_inserted += inserted

            # Advance cursor past the last returned candle
            last_ts_ms = int(rows[-1]["ts"].timestamp() * 1000)
            cursor_ms = last_ts_ms + 60_000  # +1 minute for 1m candles

            log.info(
                "Fetched %d candles for %s (up to %s), total inserted: %d",
                len(rows),
                symbol,
                rows[-1]["ts"].isoformat(),
                total_inserted,
            )

            # Rate limit: Binance allows ~2400 weight/min; klines = 5 weight
            await asyncio.sleep(0.25)

    log.info("Backfill complete for %s %s: %d rows", symbol, interval, total_inserted)
    return total_inserted


# ---------------------------------------------------------------------------
# Bybit REST klines (same pattern, different API)
# ---------------------------------------------------------------------------

BYBIT_BASE = "https://api.bybit.com"


async def fetch_bybit_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Fetch klines from Bybit V5 API."""
    url = f"{BYBIT_BASE}/v5/market/kline"
    # Bybit interval mapping: "1" = 1m, "5" = 5m, etc.
    bybit_interval = interval.replace("m", "")
    params: dict[str, Any] = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval,
        "start": start_ms,
        "limit": limit,
    }
    if end_ms is not None:
        params["end"] = end_ms

    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json()

    if data.get("retCode") != 0:
        log.error("Bybit API error: %s", data.get("retMsg"))
        return []

    rows: list[dict[str, Any]] = []
    for k in data.get("result", {}).get("list", []):
        # Bybit format: [startTime, open, high, low, close, volume, turnover]
        rows.append(
            {
                "ts": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc),
                "exchange": "bybit",
                "symbol": symbol,
                "interval": interval,
                "open": k[1],
                "high": k[2],
                "low": k[3],
                "close": k[4],
                "volume": k[5],
                "quote_volume": k[6] if len(k) > 6 else 0,
                "trades_count": 0,
            }
        )
    # Bybit returns newest first; reverse to chronological
    rows.reverse()
    return rows


async def backfill_bybit_ohlcv(
    db: Database,
    symbol: str,
    interval: str = "1m",
    days: int = 90,
) -> int:
    """Backfill OHLCV from Bybit into TimescaleDB. Same pattern as Binance."""
    latest_ts = await db.get_latest_ohlcv_ts("bybit", symbol, interval)
    if latest_ts:
        start_dt = latest_ts + timedelta(minutes=1)
    else:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)

    end_dt = datetime.now(timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    total_inserted = 0
    cursor_ms = start_ms

    async with aiohttp.ClientSession() as session:
        while cursor_ms < end_ms:
            try:
                rows = await fetch_bybit_klines(
                    session,
                    symbol=symbol,
                    interval=interval,
                    start_ms=cursor_ms,
                    end_ms=end_ms,
                    limit=200,
                )
            except aiohttp.ClientError as exc:
                log.error("Bybit API error for %s: %s", symbol, exc)
                await asyncio.sleep(2)
                continue

            if not rows:
                break

            inserted = await db.upsert_ohlcv_batch(rows)
            total_inserted += inserted

            last_ts_ms = int(rows[-1]["ts"].timestamp() * 1000)
            cursor_ms = last_ts_ms + 60_000
            await asyncio.sleep(0.3)

    log.info("Bybit backfill complete for %s %s: %d rows", symbol, interval, total_inserted)
    return total_inserted
