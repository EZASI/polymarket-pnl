"""
Tests for data_pipeline.db — schema creation and OHLCV read/write.

These tests require a running TimescaleDB instance.
Set TEST_TIMESCALEDB_DSN env var, or they default to:
    postgresql://postgres:postgres@localhost:5432/mty_hft_test

Skip gracefully if DB is unavailable.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone, timedelta

import pytest

# Check if asyncpg is available and DB is reachable
try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

# Adjust path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db import Database

TEST_DSN = os.environ.get(
    "TEST_TIMESCALEDB_DSN",
    "postgresql://postgres:postgres@localhost:5432/mty_hft_test",
)


async def _db_reachable() -> bool:
    """Check if we can connect to the test database."""
    try:
        conn = await asyncpg.connect(TEST_DSN, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


# Determine at import time whether to skip DB tests
_DB_AVAILABLE = False
if HAS_ASYNCPG:
    try:
        _DB_AVAILABLE = asyncio.get_event_loop().run_until_complete(_db_reachable())
    except RuntimeError:
        _DB_AVAILABLE = asyncio.run(_db_reachable())

requires_db = pytest.mark.skipif(
    not _DB_AVAILABLE,
    reason=f"TimescaleDB not reachable at {TEST_DSN}",
)


@pytest.fixture
async def db():
    """Provide a connected Database instance, clean up after test."""
    database = Database(dsn=TEST_DSN)
    await database.connect()
    yield database
    # Clean up test data
    try:
        async with database.acquire() as conn:
            await conn.execute(
                "DELETE FROM ohlcv WHERE exchange = 'test_exchange'"
            )
    except Exception:
        pass
    await database.close()


@requires_db
@pytest.mark.asyncio
async def test_schema_creation(db: Database) -> None:
    """Verify schema creation runs without errors (idempotent)."""
    await db.create_schema()

    # Verify key tables exist
    async with db.acquire() as conn:
        tables = await conn.fetch(
            """
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename IN ('ohlcv', 'trades_raw', 'pm_markets',
                              'pm_orderbook_snapshots', 'pm_fills',
                              'backtest_results', 'live_metrics')
            """
        )
    table_names = {r["tablename"] for r in tables}
    assert "ohlcv" in table_names
    assert "pm_markets" in table_names
    assert "backtest_results" in table_names
    assert "live_metrics" in table_names


@requires_db
@pytest.mark.asyncio
async def test_ohlcv_write_and_read(db: Database) -> None:
    """Verify we can write a batch of OHLCV rows and read them back correctly."""
    await db.create_schema()

    base_ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "ts": base_ts + timedelta(minutes=i),
            "exchange": "test_exchange",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "open": 65000.0 + i * 10,
            "high": 65050.0 + i * 10,
            "low": 64950.0 + i * 10,
            "close": 65020.0 + i * 10,
            "volume": 100.5 + i,
            "quote_volume": 6500000.0 + i * 1000,
            "trades_count": 500 + i,
        }
        for i in range(5)
    ]

    # Write
    inserted = await db.upsert_ohlcv_batch(rows)
    assert inserted == 5

    # Read back
    result = await db.query_ohlcv(
        exchange="test_exchange",
        symbol="BTCUSDT",
        interval="1m",
        start=base_ts,
        end=base_ts + timedelta(minutes=10),
        limit=100,
    )
    assert len(result) == 5
    assert float(result[0]["open"]) == 65000.0
    assert float(result[4]["open"]) == 65040.0
    assert result[0]["ts"] == base_ts

    # Verify idempotency: re-insert same rows, count should not change
    inserted2 = await db.upsert_ohlcv_batch(rows)
    result2 = await db.query_ohlcv(
        exchange="test_exchange",
        symbol="BTCUSDT",
        interval="1m",
        start=base_ts,
        end=base_ts + timedelta(minutes=10),
        limit=100,
    )
    assert len(result2) == 5  # no duplicates


@requires_db
@pytest.mark.asyncio
async def test_get_latest_ohlcv_ts(db: Database) -> None:
    """Verify get_latest_ohlcv_ts returns correct timestamp."""
    await db.create_schema()

    base_ts = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "ts": base_ts + timedelta(minutes=i),
            "exchange": "test_exchange",
            "symbol": "ETHUSDT",
            "interval": "1m",
            "open": 3000.0,
            "high": 3010.0,
            "low": 2990.0,
            "close": 3005.0,
            "volume": 50.0,
            "quote_volume": 150000.0,
            "trades_count": 200,
        }
        for i in range(3)
    ]
    await db.upsert_ohlcv_batch(rows)

    latest = await db.get_latest_ohlcv_ts("test_exchange", "ETHUSDT", "1m")
    assert latest == base_ts + timedelta(minutes=2)

    # Non-existent symbol
    none_ts = await db.get_latest_ohlcv_ts("test_exchange", "XYZUSDT", "1m")
    assert none_ts is None

    # Cleanup
    async with db.acquire() as conn:
        await conn.execute(
            "DELETE FROM ohlcv WHERE exchange = 'test_exchange' AND symbol = 'ETHUSDT'"
        )


# ---------------------------------------------------------------------------
# Tests that don't require DB
# ---------------------------------------------------------------------------

def test_sql_split() -> None:
    """Verify _split_sql handles multi-statement SQL correctly."""
    from data_pipeline.db import _split_sql

    sql = """
    -- comment
    CREATE TABLE IF NOT EXISTS foo (id INT);
    SELECT create_hypertable('foo', 'id', if_not_exists => TRUE);
    """
    stmts = _split_sql(sql)
    non_empty = [s.strip() for s in stmts if s.strip()]
    assert len(non_empty) == 2
    assert "CREATE TABLE" in non_empty[0]
    assert "create_hypertable" in non_empty[1]


def test_database_dsn_override() -> None:
    """Verify Database accepts custom DSN."""
    db = Database(dsn="postgresql://custom:custom@localhost:9999/testdb")
    assert db.dsn == "postgresql://custom:custom@localhost:9999/testdb"
