"""
Polymarket data ingestion — market discovery + orderbook snapshots.

Provides:
    - Gamma API polling for crypto binary market metadata
    - CLOB API orderbook snapshot fetching
    - Storage into pm_markets and pm_orderbook_snapshots tables
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from data_pipeline.db import Database

log = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Keywords to identify crypto short-term markets
CRYPTO_KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol"]
SHORT_TERM_KEYWORDS = ["5 min", "5-min", "15 min", "15-min", "5m", "15m", "minute"]


def _classify_underlying(question: str) -> str | None:
    """Extract underlying asset from market question text."""
    q = question.lower()
    if any(kw in q for kw in ["bitcoin", "btc"]):
        return "BTC"
    if any(kw in q for kw in ["ethereum", "eth"]):
        return "ETH"
    if any(kw in q for kw in ["solana", "sol"]):
        return "SOL"
    return None


def _is_crypto_short_term(market: dict[str, Any]) -> bool:
    """Check if a market is a crypto short-term binary."""
    question = (market.get("question", "") + " " + market.get("description", "")).lower()
    has_crypto = any(kw in question for kw in CRYPTO_KEYWORDS)
    has_short_term = any(kw in question for kw in SHORT_TERM_KEYWORDS)
    return has_crypto and has_short_term


async def fetch_polymarket_markets(
    session: aiohttp.ClientSession,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch markets from Polymarket Gamma API."""
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if active_only:
        params["active"] = "true"
        params["closed"] = "false"

    url = f"{GAMMA_API_URL}/markets"
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def discover_crypto_markets(
    db: Database,
    max_pages: int = 5,
) -> list[dict[str, Any]]:
    """
    Poll Gamma API for active crypto short-term markets.
    Upserts metadata into pm_markets table.
    Returns list of discovered crypto markets.
    """
    discovered: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        for page in range(max_pages):
            try:
                markets = await fetch_polymarket_markets(
                    session,
                    active_only=True,
                    limit=100,
                    offset=page * 100,
                )
            except aiohttp.ClientError as exc:
                log.error("Gamma API error (page %d): %s", page, exc)
                break

            if not markets:
                break

            for m in markets:
                if not _is_crypto_short_term(m):
                    continue

                underlying = _classify_underlying(m.get("question", ""))
                if not underlying:
                    continue

                market_row = {
                    "market_id": m.get("id", m.get("conditionId", "")),
                    "condition_id": m.get("conditionId"),
                    "question": m.get("question"),
                    "underlying": underlying,
                    "expiry_ts": _parse_expiry(m),
                    "strike_price": None,
                    "strike_type": "ATM",
                    "fee_tier": 0.02,  # default; refine from actual schedule
                    "status": "active" if m.get("active") else "closed",
                    "raw_json": m,
                }
                discovered.append(market_row)

            await asyncio.sleep(0.3)  # rate limit

    # Upsert into DB
    if discovered:
        await _upsert_pm_markets(db, discovered)
        log.info("Discovered %d crypto short-term markets", len(discovered))

    return discovered


def _parse_expiry(market: dict[str, Any]) -> datetime | None:
    """Try to extract expiry timestamp from market data."""
    for field in ["endDate", "end_date_iso", "expirationDate"]:
        val = market.get(field)
        if val:
            try:
                if isinstance(val, str):
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                if isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val, tz=timezone.utc)
            except (ValueError, OSError):
                continue
    return None


async def _upsert_pm_markets(db: Database, markets: list[dict[str, Any]]) -> None:
    """Upsert market metadata. On conflict update status and raw_json."""
    sql = """
        INSERT INTO pm_markets (market_id, condition_id, question, underlying, expiry_ts,
                                strike_price, strike_type, fee_tier, status, raw_json)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (market_id) DO UPDATE SET
            status = EXCLUDED.status,
            raw_json = EXCLUDED.raw_json
    """
    async with db.acquire() as conn:
        for m in markets:
            try:
                await conn.execute(
                    sql,
                    m["market_id"],
                    m["condition_id"],
                    m["question"],
                    m["underlying"],
                    m["expiry_ts"],
                    m["strike_price"],
                    m["strike_type"],
                    m["fee_tier"],
                    m["status"],
                    json.dumps(m["raw_json"]),
                )
            except Exception as exc:
                log.warning("Failed to upsert market %s: %s", m["market_id"], exc)


# ---------------------------------------------------------------------------
# Orderbook snapshots
# ---------------------------------------------------------------------------

async def fetch_orderbook_snapshot(
    session: aiohttp.ClientSession,
    token_id: str,
) -> dict[str, Any] | None:
    """Fetch current orderbook for a Polymarket token from CLOB API."""
    url = f"{CLOB_API_URL}/book"
    params = {"token_id": token_id}
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except aiohttp.ClientError as exc:
        log.warning("CLOB book fetch failed for %s: %s", token_id, exc)
        return None


async def snapshot_active_orderbooks(
    db: Database,
    market_ids: list[str],
) -> int:
    """
    Fetch and store orderbook snapshots for a list of active markets.
    Returns number of snapshots stored.
    """
    now = datetime.now(timezone.utc)
    stored = 0

    async with aiohttp.ClientSession() as session:
        for market_id in market_ids:
            book = await fetch_orderbook_snapshot(session, market_id)
            if not book:
                continue

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            yes_best_bid = float(bids[0]["price"]) if bids else None
            yes_best_ask = float(asks[0]["price"]) if asks else None

            # Depth = sum of top 5 levels
            yes_bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
            yes_ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])

            mid = None
            spread_bps = None
            if yes_best_bid is not None and yes_best_ask is not None:
                mid = (yes_best_bid + yes_best_ask) / 2
                if mid > 0:
                    spread_bps = (yes_best_ask - yes_best_bid) / mid * 10_000

            sql = """
                INSERT INTO pm_orderbook_snapshots
                    (ts, market_id, yes_best_bid, yes_best_ask, no_best_bid, no_best_ask,
                     yes_bid_depth, yes_ask_depth, spread_bps, mid)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            async with db.acquire() as conn:
                await conn.execute(
                    sql,
                    now,
                    market_id,
                    yes_best_bid,
                    yes_best_ask,
                    1 - yes_best_ask if yes_best_ask else None,  # NO best bid ≈ 1 - YES ask
                    1 - yes_best_bid if yes_best_bid else None,  # NO best ask ≈ 1 - YES bid
                    yes_bid_depth,
                    yes_ask_depth,
                    spread_bps,
                    mid,
                )
            stored += 1
            await asyncio.sleep(0.3)  # rate limit

    log.info("Stored %d orderbook snapshots", stored)
    return stored
