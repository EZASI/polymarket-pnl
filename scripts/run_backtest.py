#!/usr/bin/env python3
"""
Run a backtest from the CLI and persist results.

Usage:
    python scripts/run_backtest.py                              # default: BTC mean-reversion, 30 days
    python scripts/run_backtest.py --strategy btc_mean_reversion --days 90 --symbol BTCUSDT
    python scripts/run_backtest.py --strategy btc_mean_reversion --days 7 --fee-bps 3.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.db import Database
from research.backtester import run_backtest, save_backtest_result
from research.alpha_evaluator import evaluate, format_report, to_json
from strategies.btc_mean_reversion import BtcMeanReversion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_backtest")

# Strategy registry
STRATEGIES = {
    "btc_mean_reversion": BtcMeanReversion,
}


async def main(
    strategy_name: str,
    symbol: str,
    days: int,
    fee_bps: float,
    position_size: float,
    save: bool,
    params: dict | None,
) -> None:
    # Instantiate strategy
    if strategy_name not in STRATEGIES:
        log.error("Unknown strategy: %s. Available: %s", strategy_name, list(STRATEGIES.keys()))
        return
    strategy = STRATEGIES[strategy_name](params=params)

    db = Database()
    await db.connect()

    # Run backtest
    result = await run_backtest(
        db=db,
        strategy=strategy,
        symbol=symbol,
        exchange="binance",
        interval="1m",
        days=days,
        fee_bps=fee_bps,
        position_size_usd=position_size,
    )

    # Evaluate
    report = evaluate(result)

    # Print report
    print("\n" + "=" * 70)
    print(format_report(report))
    print("=" * 70)

    # Save to DB
    if save and result.n_trades > 0:
        # Enrich params with run metadata for dashboard display
        result.params = {
            **result.params,
            "n_candles": len(result.equity_curve) + strategy.warmup_periods(),
            "fee_bps": fee_bps,
            "position_size_usd": position_size,
        }
        row_id = await save_backtest_result(db, result)
        log.info("Saved backtest result (id=%d) to TimescaleDB", row_id)

        # Also save JSON report
        report_path = Path(__file__).resolve().parent.parent / "logs" / f"backtest_{strategy_name}_{symbol}_{days}d.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(to_json(report), f, indent=2, default=str)
        log.info("Report saved to %s", report_path)

    await db.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on OHLCV data")
    parser.add_argument("--strategy", type=str, default="btc_mean_reversion",
                        help=f"Strategy name. Available: {list(STRATEGIES.keys())}")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--fee-bps", type=float, default=2.0, help="Fee per side in basis points")
    parser.add_argument("--position-size", type=float, default=10_000, help="USD per trade")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to DB")
    parser.add_argument("--params", type=str, default=None,
                        help='Strategy params as JSON, e.g. \'{"rsi_period": 10}\'')
    args = parser.parse_args()

    params = json.loads(args.params) if args.params else None

    asyncio.run(main(
        strategy_name=args.strategy,
        symbol=args.symbol,
        days=args.days,
        fee_bps=args.fee_bps,
        position_size=args.position_size,
        save=not args.no_save,
        params=params,
    ))


if __name__ == "__main__":
    cli()
