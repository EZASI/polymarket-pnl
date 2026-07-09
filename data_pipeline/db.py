"""
TimescaleDB connection management and schema creation.

Provides:
    - async connection pool via asyncpg
    - DDL for all hypertables (ohlcv, trades_raw, pm_*, backtest_results, live_metrics)
    - helpers for batch insert / query
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import asyncpg
import yaml

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "trading.yaml"


def _load_dsn() -> str:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["database"]["timescaledb_dsn"]


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Extension (idempotent)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. OHLCV candles (CEX)
CREATE TABLE IF NOT EXISTS ohlcv (
    ts           TIMESTAMPTZ  NOT NULL,
    exchange     TEXT         NOT NULL,  -- 'binance', 'bybit'
    symbol       TEXT         NOT NULL,  -- 'BTCUSDT'
    interval     TEXT         NOT NULL,  -- '1m', '5m'
    open         NUMERIC      NOT NULL,
    high         NUMERIC      NOT NULL,
    low          NUMERIC      NOT NULL,
    close        NUMERIC      NOT NULL,
    volume       NUMERIC      NOT NULL,
    quote_volume NUMERIC,
    trades_count INTEGER,
    UNIQUE (ts, exchange, symbol, interval)
);
SELECT create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);

-- 2. Raw CEX trades
CREATE TABLE IF NOT EXISTS trades_raw (
    ts         TIMESTAMPTZ  NOT NULL,
    exchange   TEXT         NOT NULL,
    symbol     TEXT         NOT NULL,
    price      NUMERIC      NOT NULL,
    qty        NUMERIC      NOT NULL,
    side       TEXT         NOT NULL,  -- 'buy' | 'sell'
    trade_id   TEXT
);
SELECT create_hypertable('trades_raw', 'ts', if_not_exists => TRUE);

-- 3. Polymarket market metadata
CREATE TABLE IF NOT EXISTS pm_markets (
    market_id       TEXT         PRIMARY KEY,
    condition_id    TEXT,
    question        TEXT,
    underlying      TEXT,          -- 'BTC', 'ETH', 'SOL'
    expiry_ts       TIMESTAMPTZ,
    strike_price    NUMERIC,
    strike_type     TEXT         DEFAULT 'ATM',
    fee_tier        NUMERIC,
    maker_rebate_pct NUMERIC    DEFAULT 0,
    status          TEXT         DEFAULT 'active',
    discovered_at   TIMESTAMPTZ  DEFAULT NOW(),
    settled_outcome BOOLEAN,
    settled_at      TIMESTAMPTZ,
    raw_json        JSONB
);

-- 4. Polymarket orderbook snapshots
CREATE TABLE IF NOT EXISTS pm_orderbook_snapshots (
    ts             TIMESTAMPTZ  NOT NULL,
    market_id      TEXT         NOT NULL,
    yes_best_bid   NUMERIC,
    yes_best_ask   NUMERIC,
    no_best_bid    NUMERIC,
    no_best_ask    NUMERIC,
    yes_bid_depth  NUMERIC,
    yes_ask_depth  NUMERIC,
    spread_bps     NUMERIC,
    mid            NUMERIC
);
SELECT create_hypertable('pm_orderbook_snapshots', 'ts', if_not_exists => TRUE);

-- 5. Polymarket fills / position tracking
CREATE TABLE IF NOT EXISTS pm_fills (
    fill_id        SERIAL,
    ts             TIMESTAMPTZ  NOT NULL,
    market_id      TEXT         NOT NULL,
    strategy_name  TEXT         NOT NULL,
    side           TEXT         NOT NULL,  -- 'YES' | 'NO'
    size_usd       NUMERIC      NOT NULL,
    price          NUMERIC      NOT NULL,
    fee_paid       NUMERIC      DEFAULT 0,
    rebate_earned  NUMERIC      DEFAULT 0,
    is_maker       BOOLEAN      DEFAULT TRUE,
    pnl_net        NUMERIC
);
SELECT create_hypertable('pm_fills', 'ts', if_not_exists => TRUE);

-- 6. Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL       PRIMARY KEY,
    run_ts          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    strategy_name   TEXT         NOT NULL,
    underlying      TEXT,
    period_start    TIMESTAMPTZ,
    period_end      TIMESTAMPTZ,
    n_trades        INTEGER,
    hit_rate        NUMERIC,
    brier_score     NUMERIC,
    sharpe          NUMERIC,
    max_drawdown    NUMERIC,
    pnl_total       NUMERIC,
    pnl_per_trade   NUMERIC,
    params_json     JSONB,
    calibration_json JSONB
);

-- 7. Live metrics (time-series)
CREATE TABLE IF NOT EXISTS live_metrics (
    ts              TIMESTAMPTZ  NOT NULL,
    strategy_name   TEXT         NOT NULL,
    underlying      TEXT,
    metric_name     TEXT         NOT NULL,  -- 'pnl', 'sharpe', 'fill_rate', 'latency_p50', etc.
    metric_value    NUMERIC      NOT NULL
);
SELECT create_hypertable('live_metrics', 'ts', if_not_exists => TRUE);
"""


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

class Database:
    """Thin async wrapper around asyncpg pool."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or _load_dsn()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        log.info("Connecting to TimescaleDB: %s", self.dsn.split("@")[-1])
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        assert self._pool is not None, "call connect() first"
        async with self._pool.acquire() as conn:
            yield conn

    async def create_schema(self) -> None:
        """Run full DDL (idempotent)."""
        async with self.acquire() as conn:
            # Execute statements one by one (asyncpg doesn't like multi-statement with hypertable calls mixed in)
            for statement in _split_sql(SCHEMA_SQL):
                statement = statement.strip()
                if not statement:
                    continue
                try:
                    await conn.execute(statement)
                except asyncpg.DuplicateTableError:
                    pass  # hypertable already exists
                except Exception as exc:
                    # Ignore "already a hypertable" warnings
                    if "already a hypertable" in str(exc):
                        continue
                    log.warning("DDL statement warning: %s — %s", statement[:80], exc)

    # ------------------------------------------------------------------
    # OHLCV helpers
    # ------------------------------------------------------------------

    async def upsert_ohlcv_batch(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> int:
        """
        Insert OHLCV rows. Uses ON CONFLICT DO NOTHING for idempotency.
        Each row: {ts, exchange, symbol, interval, open, high, low, close, volume, quote_volume, trades_count}
        Returns number of rows inserted.
        """
        if not rows:
            return 0
        sql = """
            INSERT INTO ohlcv (ts, exchange, symbol, interval, open, high, low, close, volume, quote_volume, trades_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (ts, exchange, symbol, interval) DO NOTHING
        """
        async with self.acquire() as conn:
            records = [
                (
                    r["ts"],
                    r["exchange"],
                    r["symbol"],
                    r["interval"],
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                    float(r["volume"]),
                    float(r.get("quote_volume", 0)),
                    int(r.get("trades_count", 0)),
                )
                for r in rows
            ]
            result = await conn.executemany(sql, records)
        return len(rows)

    async def query_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query OHLCV candles, returns list of dicts."""
        clauses = ["exchange = $1", "symbol = $2", "interval = $3"]
        params: list[Any] = [exchange, symbol, interval]
        idx = 4
        if start:
            clauses.append(f"ts >= ${idx}")
            params.append(start)
            idx += 1
        if end:
            clauses.append(f"ts <= ${idx}")
            params.append(end)
            idx += 1
        sql = f"""
            SELECT ts, exchange, symbol, interval, open, high, low, close, volume, quote_volume, trades_count
            FROM ohlcv
            WHERE {' AND '.join(clauses)}
            ORDER BY ts ASC
            LIMIT {limit}
        """
        async with self.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [dict(r) for r in records]

    async def get_latest_ohlcv_ts(
        self,
        exchange: str,
        symbol: str,
        interval: str,
    ) -> datetime | None:
        """Return the most recent candle timestamp, or None if empty."""
        sql = """
            SELECT MAX(ts) FROM ohlcv
            WHERE exchange = $1 AND symbol = $2 AND interval = $3
        """
        async with self.acquire() as conn:
            row = await conn.fetchval(sql, exchange, symbol, interval)
        return row


def _split_sql(sql: str) -> list[str]:
    """Split SQL on semicolons, keeping each statement intact."""
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current))
            current = []
    if current:
        statements.append("\n".join(current))
    return statements
