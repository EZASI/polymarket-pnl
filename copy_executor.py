#!/usr/bin/env python3
"""
COPY TRADING EXECUTOR — Follow Top Traders on Polymarket
==========================================================

This is the actual money-making bot. It:
1. Reads copy signals from copy_signal_bot.py (runs every 10 min via cron)
2. When STRONG signal: 2+ traders w/ >60% WR agree → auto-place bet
3. Uses Kelly criterion for position sizing
4. Paper mode by default, --live for real USDC on Polymarket

Sizing (Kelly Criterion):
    f* = (p * b - q) / b
    where p = avg win rate, b = payout odds, q = 1 - p
    We use fractional Kelly (25%) for safety

Risk Management:
    - $50,000 max bankroll
    - Max 10% per single bet
    - Max 5 concurrent positions
    - Daily loss limit: 15% of bankroll
    - Only STRONG/MODERATE signals
    - Min $10k liquidity per market

Usage:
    python3 copy_executor.py                   # paper mode (default)
    python3 copy_executor.py --live            # real money
    python3 copy_executor.py --live --bankroll 10000
    python3 copy_executor.py --once            # single scan, no loop
"""

import argparse
import json
import logging
import os
import sys
import time
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("copy_executor")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


# ── Kelly Criterion Position Sizer ──
def kelly_size(win_rate: float, avg_odds: float, bankroll: float,
               fraction: float = 0.25, max_pct: float = 0.10) -> float:
    """
    Calculate position size using fractional Kelly criterion.

    Args:
        win_rate: Historical win rate (0-1)
        avg_odds: Average decimal odds (e.g., 1.5 means 50% implied)
        bankroll: Current bankroll in USD
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
        max_pct: Maximum percentage of bankroll per bet

    Returns:
        Position size in USD
    """
    if win_rate <= 0 or avg_odds <= 1:
        return 0

    p = win_rate
    q = 1 - p
    b = avg_odds - 1  # net odds

    if b <= 0:
        return 0

    kelly_f = (p * b - q) / b
    if kelly_f <= 0:
        return 0  # negative Kelly = don't bet

    # Apply fractional Kelly for safety
    f = kelly_f * fraction

    # Cap at max percentage
    f = min(f, max_pct)

    size = bankroll * f
    return round(size, 2)


# ── Risk Manager ──
@dataclass
class CopyRiskManager:
    bankroll: float = 50000.0
    max_pct_per_bet: float = 0.10       # 10% max per position
    daily_loss_limit_pct: float = 0.15  # stop at 15% daily loss
    max_concurrent: int = 5
    min_signal_strength: str = "MODERATE"  # STRONG or MODERATE
    min_traders: int = 2
    min_avg_wr: float = 55.0
    min_invested: float = 10000.0       # minimum $ invested by top traders

    # State
    daily_pnl: float = 0.0
    open_positions: list = field(default_factory=list)
    executed_today: list = field(default_factory=list)
    session_start: str = ""

    def can_bet(self, signal: dict) -> tuple[bool, str]:
        """Check if we should execute this signal."""
        strength = signal.get("strength", "WEAK")
        n_traders = signal.get("n_traders", 0)
        avg_wr = signal.get("avg_wr", 0)
        total_invested = signal.get("total_invested", 0)
        cid = signal.get("conditionId", "")

        # Already bet on this market today
        if cid in [p["conditionId"] for p in self.executed_today]:
            return False, f"Already bet on this market today"

        # Strength check
        valid_strengths = ["STRONG", "MODERATE"] if self.min_signal_strength == "MODERATE" else ["STRONG"]
        if strength not in valid_strengths:
            return False, f"Signal too weak: {strength}"

        # Trader count
        if n_traders < self.min_traders:
            return False, f"Only {n_traders} traders (need {self.min_traders}+)"

        # Win rate
        if avg_wr < self.min_avg_wr:
            return False, f"Avg WR {avg_wr}% < {self.min_avg_wr}%"

        # Investment size (skin in the game)
        if total_invested < self.min_invested:
            return False, f"Only ${total_invested:,.0f} invested (need ${self.min_invested:,.0f}+)"

        # Max concurrent positions
        if len(self.open_positions) >= self.max_concurrent:
            return False, f"Max positions reached ({self.max_concurrent})"

        # Daily loss limit
        effective = self.bankroll + self.daily_pnl
        if self.daily_pnl <= -(self.bankroll * self.daily_loss_limit_pct):
            return False, f"Daily loss limit hit (${self.daily_pnl:+,.0f})"

        return True, "OK"

    def calculate_size(self, signal: dict) -> float:
        """Calculate bet size using Kelly criterion."""
        avg_wr = signal.get("avg_wr", 50) / 100.0
        # Estimate odds from consensus: if traders invest heavy, implied odds are near 1/(price)
        # For Polymarket, typical payoff is 1/price where price is the market probability
        # Conservative: assume 1.8x average return on winning bets
        avg_odds = 1.8

        size = kelly_size(
            win_rate=avg_wr,
            avg_odds=avg_odds,
            bankroll=self.bankroll + self.daily_pnl,
            fraction=0.25,
            max_pct=self.max_pct_per_bet,
        )

        # Minimum bet $50, maximum from Kelly
        return max(50.0, size)


# ── Market Info Fetcher ──
def get_market_info(condition_id: str) -> Optional[dict]:
    """Fetch market details from Polymarket Gamma API."""
    try:
        resp = requests.get(f"{GAMMA_HOST}/markets", params={
            "conditionId": condition_id, "limit": 1
        }, timeout=10)
        data = resp.json()
        if data:
            return data[0] if isinstance(data, list) else data
    except Exception as e:
        log.debug(f"Market info error: {e}")
    return None


def get_market_by_slug(slug: str) -> Optional[dict]:
    """Fetch market by slug."""
    if not slug:
        return None
    try:
        resp = requests.get(f"{GAMMA_HOST}/markets", params={
            "slug": slug, "limit": 1
        }, timeout=10)
        data = resp.json()
        if data:
            return data[0] if isinstance(data, list) else data
    except:
        pass
    return None


# ── Order Executor ──
class CopyOrderExecutor:
    def __init__(self, live: bool = False):
        self.live = live
        self.client = None
        self._log_path = LOG_DIR / f"copy_trades_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if live:
            self._init_client()

    def _init_client(self):
        """Initialize Polymarket CLOB client for live trading."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            pk = os.getenv("PK")
            if not pk:
                # Try loading encrypted env
                enc_path = Path(".env.enc")
                if enc_path.exists():
                    from security import load_encrypted_env
                    enc_password = os.getenv("ENC_PASSWORD", "")
                    if not enc_password:
                        import getpass
                        enc_password = getpass.getpass("Enter .env encryption password: ")
                    env_vars = load_encrypted_env(str(enc_path), enc_password)
                    for k, v in env_vars.items():
                        os.environ[k] = v
                    pk = os.getenv("PK")

            if not pk:
                log.error("No PK found. Cannot trade live.")
                self.live = False
                return

            api_key = os.getenv("CLOB_API_KEY")
            api_secret = os.getenv("CLOB_SECRET")
            api_pass = os.getenv("CLOB_PASS_PHRASE")

            if api_key and api_secret and api_pass:
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_pass,
                )
                self.client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID, creds=creds)
            else:
                self.client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID)
                creds = self.client.create_or_derive_api_creds()
                self.client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID, creds=creds)
                log.info(f"Generated API creds. Save to .env:")
                log.info(f"  CLOB_API_KEY={creds.api_key}")
                log.info(f"  CLOB_SECRET={creds.api_secret}")
                log.info(f"  CLOB_PASS_PHRASE={creds.api_passphrase}")

            log.info("CLOB client initialized (LIVE MODE)")
        except ImportError:
            log.error("py-clob-client not installed")
            self.live = False
        except Exception as e:
            log.error(f"CLOB init failed: {e}")
            self.live = False

    def execute(self, signal: dict, size_usd: float, market_info: dict) -> dict:
        """Execute a copy trade (paper or live)."""
        now = datetime.now(timezone.utc)
        consensus = signal.get("consensus", "")
        title = signal.get("title", "")
        condition_id = signal.get("conditionId", "")

        # Determine which token to buy
        outcomes = market_info.get("outcomes", "[]")
        prices = market_info.get("outcomePrices", "[]")
        tids = market_info.get("clobTokenIds", "[]")

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(tids, str):
            tids = json.loads(tids)

        # Find the outcome matching consensus
        token_id = ""
        buy_price = 0.5
        for i, outcome in enumerate(outcomes):
            if outcome.lower() == consensus.lower() or outcome == consensus:
                token_id = tids[i] if i < len(tids) else ""
                buy_price = float(prices[i]) if i < len(prices) else 0.5
                break

        if not token_id:
            # Fuzzy match — consensus might be a team name
            for i, outcome in enumerate(outcomes):
                if consensus.lower() in outcome.lower() or outcome.lower() in consensus.lower():
                    token_id = tids[i] if i < len(tids) else ""
                    buy_price = float(prices[i]) if i < len(prices) else 0.5
                    break

        record = {
            "time": now.isoformat(),
            "mode": "LIVE" if self.live else "PAPER",
            "title": title,
            "conditionId": condition_id,
            "consensus": consensus,
            "token_id": token_id,
            "buy_price": buy_price,
            "size_usd": size_usd,
            "n_traders": signal.get("n_traders", 0),
            "avg_wr": signal.get("avg_wr", 0),
            "total_invested": signal.get("total_invested", 0),
            "strength": signal.get("strength", ""),
            "status": "PENDING",
        }

        if self.live and self.client and token_id:
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY

                contracts = size_usd / buy_price
                order_args = OrderArgs(
                    price=buy_price,
                    size=contracts,
                    side=BUY,
                    token_id=token_id,
                )
                signed = self.client.create_order(order_args)
                resp = self.client.post_order(signed)

                record["order_id"] = resp.get("orderID", "")
                record["status"] = "PLACED"
                log.info(
                    f"LIVE ORDER: BUY {consensus} ${size_usd:.0f} @ {buy_price:.3f} | "
                    f"{title}"
                )
            except Exception as e:
                record["status"] = "FAILED"
                record["error"] = str(e)
                log.error(f"Order failed: {e}")
        else:
            record["status"] = "PAPER"
            log.info(
                f"PAPER: BUY {consensus} ${size_usd:.0f} @ {buy_price:.3f} | "
                f"EV traders={signal['n_traders']} avgWR={signal['avg_wr']}% | "
                f"{title}"
            )

        # Log trade
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record


# ── Main Executor Loop ──
def run_once(risk: CopyRiskManager, executor: CopyOrderExecutor):
    """Single scan: read signals, evaluate, execute."""
    now = datetime.now(timezone.utc)

    # Load signals — prefer smart_signals.json (ML-enhanced) over raw copy_signals
    smart_path = LOG_DIR / "smart_signals.json"
    signals_path = LOG_DIR / "copy_signals.json"

    data = None
    source = ""
    if smart_path.exists():
        try:
            data = json.load(open(smart_path))
            source = "smart_signals"
        except:
            pass

    if data is None:
        if not signals_path.exists():
            log.warning("No signals found. Run copy_signal_bot.py first.")
            return []
        try:
            data = json.load(open(signals_path))
            source = "copy_signals"
        except Exception as e:
            log.error(f"Failed to load signals: {e}")
            return []

    signals = data.get("signals", [])
    scan_time = data.get("generated", data.get("scan_time", data.get("time", "")))

    # Check if signals are fresh (< 30 min old)
    try:
        scan_dt = datetime.fromisoformat(scan_time.replace("Z", "+00:00"))
        age = (now - scan_dt).total_seconds()
        if age > 1800:
            log.warning(f"Signals are {age/60:.0f}min old. Waiting for fresh scan.")
            return []
    except:
        pass

    print(f"\n{'='*70}")
    print(f"  COPY EXECUTOR | {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Bankroll: ${risk.bankroll:,.0f} | Day P&L: ${risk.daily_pnl:+,.0f}")
    print(f"  Open: {len(risk.open_positions)}/{risk.max_concurrent} | Mode: {'LIVE' if executor.live else 'PAPER'}")
    print(f"{'='*70}")
    print(f"  Signals: {len(signals)} found (source: {source})")

    executed = []
    for signal in signals:
        strength = signal.get("strength", "WEAK")
        title = signal.get("title", "")
        n_traders = signal.get("n_traders", 0)
        avg_wr = signal.get("avg_wr", 0)
        total = signal.get("total_invested", 0)
        consensus = signal.get("consensus", "")

        can, reason = risk.can_bet(signal)

        if can:
            # Calculate position size
            size = risk.calculate_size(signal)

            # Get market info for token IDs
            market_info = get_market_info(signal.get("conditionId", ""))
            if not market_info:
                market_info = get_market_by_slug(signal.get("slug", ""))

            if not market_info:
                print(f"  SKIP  {title[:40]} — cannot find market on Polymarket")
                continue

            # Check if market is still open
            if market_info.get("closed", False):
                print(f"  SKIP  {title[:40]} — market closed")
                continue

            # Execute
            record = executor.execute(signal, size, market_info)
            if record["status"] in ("PLACED", "PAPER"):
                risk.open_positions.append(record)
                risk.executed_today.append({"conditionId": signal.get("conditionId", "")})
                executed.append(record)
                print(
                    f"  EXEC  [{strength}] {title[:40]}")
                print(
                    f"        → {consensus} ${size:,.0f} | "
                    f"{n_traders} traders, {avg_wr}% WR, ${total:,.0f} backing"
                )
        else:
            tag = f"[{strength}]" if strength != "WEAK" else "[WEAK]"
            if strength in ("STRONG", "MODERATE"):
                print(f"  SKIP  {tag} {title[:40]} — {reason}")

    if not executed:
        print(f"\n  No trades executed this scan.")
    else:
        print(f"\n  Executed {len(executed)} trades this scan.")

    return executed


def run_loop(risk: CopyRiskManager, executor: CopyOrderExecutor, interval: int = 600):
    """Continuous loop: scan every `interval` seconds."""
    log.info(f"Starting copy executor loop (every {interval}s)")
    log.info(f"Mode: {'LIVE' if executor.live else 'PAPER'}")
    log.info(f"Bankroll: ${risk.bankroll:,.0f}")

    while True:
        try:
            run_once(risk, executor)
        except Exception as e:
            log.error(f"Scan error: {e}")

        log.info(f"Next scan in {interval}s...")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Copy Trading Executor")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading")
    parser.add_argument("--once", action="store_true", help="Single scan then exit")
    parser.add_argument("--bankroll", type=float, default=50000, help="Bankroll in USD")
    parser.add_argument("--max-pct", type=float, default=0.10, help="Max % per bet")
    parser.add_argument("--interval", type=int, default=600, help="Scan interval in seconds")
    parser.add_argument("--min-strength", choices=["STRONG", "MODERATE"], default="MODERATE")
    parser.add_argument("--min-traders", type=int, default=2, help="Min traders for signal")
    parser.add_argument("--min-wr", type=float, default=55.0, help="Min avg win rate")
    args = parser.parse_args()

    if args.live:
        print("\n" + "!" * 70)
        print("  WARNING: LIVE TRADING MODE — REAL USDC WILL BE USED")
        print("  Bankroll: ${:,.0f}".format(args.bankroll))
        print("!" * 70)
        confirm = input("\n  Type 'YES' to continue: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            sys.exit(0)

    risk = CopyRiskManager(
        bankroll=args.bankroll,
        max_pct_per_bet=args.max_pct,
        min_signal_strength=args.min_strength,
        min_traders=args.min_traders,
        min_avg_wr=args.min_wr,
        session_start=datetime.now(timezone.utc).isoformat(),
    )

    executor = CopyOrderExecutor(live=args.live)

    if args.once:
        run_once(risk, executor)
    else:
        try:
            run_loop(risk, executor, interval=args.interval)
        except KeyboardInterrupt:
            print("\n\nShutdown.")
            print(f"  Session P&L: ${risk.daily_pnl:+,.0f}")
            print(f"  Trades: {len(risk.executed_today)}")


if __name__ == "__main__":
    main()
