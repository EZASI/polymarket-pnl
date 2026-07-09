#!/usr/bin/env python3
"""
Nightly data refresh script.

Usage:
    python scripts/refresh_data.py                    # backfill all symbols, 90 days
    python scripts/refresh_data.py --symbol BTCUSDT   # single symbol
    python scripts/refresh_data.py --days 30          # last 30 days only
    python scripts/refresh_data.py --schema-only      # just create/update schema

Idempotent: safe to run repeatedly. Resumes from the last stored candle.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db import Database
from data_pipeline.cex_ingestion import backfill_binance_ohlcv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("refresh_data")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_DAYS = 90
DEFAULT_INTERVAL = "1m"


async def main(
    symbols: list[str],
    days: int,
    interval: str,
    schema_only: bool,
) -> None:
    db = Database()
    await db.connect()

    # Always ensure schema is up to date
    log.info("Creating / updating schema...")
    await db.create_schema()
    log.info("Schema ready.")

    if schema_only:
        await db.close()
        return

    total = 0
    for symbol in symbols:
        log.info("=" * 60)
        log.info("Backfilling %s %s (last %d days)", symbol, interval, days)
        log.info("=" * 60)
        count = await backfill_binance_ohlcv(
            db,
            symbol=symbol,
            interval=interval,
            days=days,
        )
        total += count
        log.info("Inserted %d candles for %s", count, symbol)

    log.info("=" * 60)
    log.info("DONE — total candles inserted: %d", total)
    log.info("=" * 60)

    await db.close()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill CEX OHLCV data into TimescaleDB"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol to backfill (e.g., BTCUSDT). Default: all configured symbols.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days to backfill (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help=f"Candle interval (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only create/update the DB schema, skip data backfill",
    )
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    asyncio.run(main(symbols, args.days, args.interval, args.schema_only))


if __name__ == "__main__":
    cli()
