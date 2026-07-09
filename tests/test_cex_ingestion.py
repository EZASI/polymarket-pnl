"""
Tests for data_pipeline.cex_ingestion — Binance kline parsing.

Unit tests that don't require a live DB or API.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.cex_ingestion import fetch_binance_klines


# Sample Binance kline response (one candle)
SAMPLE_KLINE = [
    1707782400000,   # open time (ms)
    "65000.00",      # open
    "65100.00",      # high
    "64900.00",      # low
    "65050.00",      # close
    "123.456",       # volume
    1707782459999,   # close time
    "8023456.78",    # quote volume
    1500,            # number of trades
    "60.0",          # taker buy base volume
    "3900000.0",     # taker buy quote volume
    "0",             # ignore
]


@pytest.mark.asyncio
async def test_parse_binance_kline() -> None:
    """Verify Binance kline response is parsed into our canonical format."""
    # Mock aiohttp session
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=[SAMPLE_KLINE])
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)

    rows = await fetch_binance_klines(
        mock_session,
        symbol="BTCUSDT",
        interval="1m",
        start_ms=1707782400000,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["exchange"] == "binance"
    assert row["symbol"] == "BTCUSDT"
    assert row["interval"] == "1m"
    assert row["open"] == "65000.00"
    assert row["high"] == "65100.00"
    assert row["close"] == "65050.00"
    assert row["volume"] == "123.456"
    assert row["trades_count"] == 1500
    assert isinstance(row["ts"], datetime)
    assert row["ts"].tzinfo == timezone.utc
