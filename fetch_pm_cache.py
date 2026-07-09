#!/usr/bin/env python3
"""
Fast concurrent fetcher for Polymarket 5m BTC markets.
Uses asyncio + aiohttp to fetch many slugs in parallel.
Saves to parquet/CSV cache for the main backtester.
"""
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd

CACHE_DIR = Path("/Users/eiji/polymarket-pnl/data/cache")
GAMMA_URL = "https://gamma-api.polymarket.com/events"
# Earliest known 5m market: Dec 18, 2025 18:55 UTC
EARLIEST_TS = 1766084100
CONCURRENCY = 10  # parallel requests
RATE_DELAY = 0.15  # seconds between batches


async def fetch_one(session: aiohttp.ClientSession, ts: int) -> dict | None:
    slug = f"btc-updown-5m-{ts}"
    try:
        async with session.get(
            f"{GAMMA_URL}?slug={slug}", timeout=aiohttp.ClientTimeout(total=8)
        ) as resp:
            data = await resp.json()
            if not data:
                return None
            evt = data[0]
            markets = evt.get("markets", [])
            if not markets:
                return None
            mk = markets[0]
            if not mk.get("closed", False):
                return None
            prices = mk.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            up_price = float(prices[0]) if prices else 0.5
            return {
                "round_ts": ts,
                "outcome": 1 if up_price > 0.5 else 0,
                "volume": float(mk.get("volume", 0)),
                "last_trade_price": float(mk.get("lastTradePrice", 0.5)),
                "up_resolved_price": up_price,
            }
    except Exception:
        return None


async def fetch_all(n_rounds: int = 5000) -> pd.DataFrame:
    now = int(time.time())
    # Start from 2 hours ago and go backwards
    start_ts = ((now - 7200) // 300) * 300

    # Generate all candidate timestamps
    candidates = []
    ts = start_ts
    while ts >= EARLIEST_TS and len(candidates) < n_rounds * 2:
        candidates.append(ts)
        ts -= 300

    print(f"Checking {len(candidates)} candidate timestamps for {n_rounds} markets...")
    print(f"  Range: {datetime.fromtimestamp(candidates[-1], tz=timezone.utc)} to "
          f"{datetime.fromtimestamp(candidates[0], tz=timezone.utc)}")

    results = []
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in batches of CONCURRENCY
        for batch_start in range(0, len(candidates), CONCURRENCY):
            batch = candidates[batch_start:batch_start + CONCURRENCY]
            tasks = [fetch_one(session, ts) for ts in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in batch_results:
                if isinstance(r, dict) and r is not None:
                    results.append(r)

            if len(results) >= n_rounds:
                break

            if len(results) % 500 == 0 and len(results) > 0:
                elapsed = time.time() - now
                rate = len(results) / max(elapsed, 1)
                eta = (n_rounds - len(results)) / max(rate, 0.1)
                print(f"  Fetched {len(results)}/{n_rounds} markets "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)")

            await asyncio.sleep(RATE_DELAY)

    results = results[:n_rounds]
    df = pd.DataFrame(results)
    if df.empty:
        print("ERROR: No markets found!")
        return df

    df.sort_values("round_ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\nFetched {len(df)} resolved markets")
    print(f"  Date range: "
          f"{datetime.fromtimestamp(df['round_ts'].min(), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to "
          f"{datetime.fromtimestamp(df['round_ts'].max(), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(f"  Outcomes: UP={df['outcome'].sum()} ({100*df['outcome'].mean():.1f}%) "
          f"DOWN={len(df)-df['outcome'].sum()} ({100*(1-df['outcome'].mean()):.1f}%)")
    print(f"  Avg volume: ${df['volume'].mean():.0f}/market")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.time()
    df = asyncio.run(fetch_all(args.rounds))

    if df.empty:
        return

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"pm_5m_markets_{args.rounds}.parquet"
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        cache_path = CACHE_DIR / f"pm_5m_markets_{args.rounds}.csv"
        df.to_csv(cache_path, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved {len(df)} markets → {cache_path}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
