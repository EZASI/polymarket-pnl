#!/usr/bin/env python3
"""
POLYMARKET LIVE TRADER
========================

Connects the 60% WR streaming predictor to Polymarket's CLOB API.
Automatically finds short-term BTC price markets and places trades.

Modes:
  PAPER (default):  Simulates trades against Binance price. No real money.
  LIVE (--live):    Places real orders on Polymarket via py-clob-client.

Safety:
  - Paper mode by default (must explicitly pass --live)
  - Daily loss limit: stops if down 20% of bankroll
  - Max 3 simultaneous positions
  - Cooldown: 60 seconds between trades
  - Pause after 5 consecutive losses

Usage:
  # Paper trade (default, safe, uses Binance price)
  python3 polymarket_trader.py

  # Live trade on Polymarket (real money!)
  python3 polymarket_trader.py --live

Setup:
  1. cp .env.example .env
  2. Edit .env with your Polymarket private key
  3. pip install py-clob-client python-dotenv
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trader")


# ============================================================
# RISK MANAGER
# ============================================================
@dataclass
class RiskManager:
    """Enforces all safety limits."""
    bankroll: float = 500.0
    max_risk_per_trade: float = 0.05      # 5% of bankroll
    daily_loss_limit: float = 0.20         # stop at 20% daily loss
    cooldown_seconds: float = 60.0
    max_open_positions: int = 3
    max_consecutive_losses: int = 5
    pause_after_losses_seconds: float = 600.0  # 10 min pause

    # State
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    open_positions: int = 0
    last_trade_time: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    paused_until: float = 0.0
    session_start: float = field(default_factory=time.time)

    @property
    def trade_size(self) -> float:
        """Base position size."""
        return self.bankroll * self.max_risk_per_trade

    def sized_trade(self, confidence: float) -> float:
        """
        Dynamic position sizing based on signal confidence.
        Higher confidence = larger position (up to 2x base).
        Lower confidence = smaller position (down to 0.5x base).
        This is a simplified half-Kelly approach.
        """
        base = self.trade_size
        if confidence >= 0.70:
            return min(base * 2.0, self.bankroll * 0.10)  # 2x but max 10% of bankroll
        elif confidence >= 0.60:
            return base * 1.5
        elif confidence >= 0.50:
            return base * 1.0
        else:
            return base * 0.5

    @property
    def win_rate(self) -> float:
        return self.total_wins / self.total_trades if self.total_trades > 0 else 0

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed. Multiple safety gates."""
        now = time.time()

        # Gate 1: Pause after consecutive losses
        if now < self.paused_until:
            remaining = int(self.paused_until - now)
            return False, f"PAUSED ({remaining}s remaining after {self.max_consecutive_losses} consecutive losses)"

        # Gate 2: Daily loss limit
        if self.daily_pnl <= -(self.bankroll * self.daily_loss_limit):
            return False, f"DAILY LOSS LIMIT HIT (${self.daily_pnl:+.2f}, limit ${-self.bankroll * self.daily_loss_limit:.2f})"

        # Gate 3: Max simultaneous positions
        if self.open_positions >= self.max_open_positions:
            return False, f"MAX POSITIONS ({self.open_positions}/{self.max_open_positions})"

        # Gate 4: Cooldown between trades
        elapsed = now - self.last_trade_time
        if elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            return False, f"COOLDOWN ({remaining}s)"

        # Gate 5: Minimum bankroll (stop if balance too low)
        effective_balance = self.bankroll + self.daily_pnl
        if effective_balance < 50:
            return False, f"BALANCE TOO LOW (${effective_balance:.2f} < $50 minimum)"

        # Gate 6: Single trade size sanity check (hard cap at 10% of bankroll)
        if self.trade_size > self.bankroll * 0.10:
            return False, f"TRADE SIZE EXCEEDS 10% HARD CAP"

        return True, "OK"

    def record_trade(self, pnl: float):
        """Record a completed trade."""
        self.total_trades += 1
        self.daily_pnl += pnl

        if pnl > 0:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.paused_until = time.time() + self.pause_after_losses_seconds
                log.warning(
                    f"SAFETY: {self.consecutive_losses} consecutive losses. "
                    f"Pausing for {self.pause_after_losses_seconds/60:.0f} minutes."
                )

    def record_open(self):
        self.open_positions += 1
        self.last_trade_time = time.time()

    def record_close(self):
        self.open_positions = max(0, self.open_positions - 1)

    def status(self) -> str:
        return (
            f"Bankroll: ${self.bankroll:.0f} | "
            f"Day P&L: ${self.daily_pnl:+.2f} | "
            f"Trades: {self.total_trades} | "
            f"WR: {self.win_rate:.0%} | "
            f"Open: {self.open_positions}/{self.max_open_positions}"
        )


# ============================================================
# MARKET FINDER
# ============================================================
@dataclass
class TradableMarket:
    """A Polymarket market suitable for our strategy."""
    question: str
    condition_id: str
    token_id_yes: str
    token_id_no: str
    yes_price: float
    no_price: float
    strike_price: float      # the BTC price level in the question
    end_date: str
    volume: float


class MarketFinder:
    """Discovers active BTC price markets on Polymarket."""

    def __init__(self):
        self._cache: list[TradableMarket] = []
        self._last_scan: float = 0
        self._scan_interval = 300  # 5 minutes

    def scan(self) -> list[TradableMarket]:
        """Scan Polymarket for tradable BTC price markets."""
        now = time.time()
        if now - self._last_scan < self._scan_interval and self._cache:
            return self._cache

        self._last_scan = now
        markets = []

        try:
            # ── 1. Search for 15-MINUTE "Bitcoin Up or Down" markets ──
            # These use slug pattern: btc-updown-15m-{timestamp}
            # They're the BEST fit for our streaming predictor
            try:
                resp = requests.get(f"{GAMMA_HOST}/markets", params={
                    "limit": 50,
                    "active": "true",
                    "closed": "false",
                    "order": "createdAt",
                    "ascending": "false",
                }, timeout=10)
                recent = resp.json()

                for m in recent:
                    slug = m.get("slug", "")
                    if not slug.startswith("btc-updown-15m"):
                        continue
                    if not m.get("acceptingOrders"):
                        continue

                    outcomes = m.get("outcomes", "[]")
                    prices = m.get("outcomePrices", "[]")
                    tids = m.get("clobTokenIds", "[]")

                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    if isinstance(tids, str):
                        tids = json.loads(tids)

                    if len(outcomes) < 2 or len(prices) < 2:
                        continue

                    up_idx = 0 if outcomes[0].lower() == "up" else 1
                    down_idx = 1 - up_idx

                    market = TradableMarket(
                        question=m.get("question", ""),
                        condition_id=m.get("conditionId", ""),
                        token_id_yes=tids[up_idx] if len(tids) > up_idx else "",
                        token_id_no=tids[down_idx] if len(tids) > down_idx else "",
                        yes_price=float(prices[up_idx]),
                        no_price=float(prices[down_idx]),
                        strike_price=0,
                        end_date=m.get("endDateIso", m.get("endDate", "")),
                        volume=float(m.get("volumeClob", 0) or m.get("volume", 0) or 0),
                    )
                    markets.append(market)
                    log.info(
                        f"15-MIN MARKET: {market.question} | "
                        f"UP={market.yes_price:.3f} DOWN={market.no_price:.3f} | "
                        f"Bid={m.get('bestBid','')} Ask={m.get('bestAsk','')} Spread={m.get('spread','')}"
                    )
            except Exception as e:
                log.debug(f"15-min search error: {e}")

            # ── 2. Also search for daily "Bitcoin Up or Down" markets ──
            today = datetime.now(timezone.utc)
            for day_offset in range(0, 5):
                target = today + timedelta(days=day_offset)
                month = target.strftime("%B").lower()
                day_num = target.day
                slug = f"bitcoin-up-or-down-on-{month}-{day_num}"

                try:
                    resp = requests.get(f"{GAMMA_HOST}/events", params={
                        "slug": slug,
                    }, timeout=10)
                    events = resp.json()

                    for e in events:
                        for m in e.get("markets", []):
                            if m.get("closed"):
                                continue

                            outcomes_raw = m.get("outcomes", "[]")
                            prices_raw = m.get("outcomePrices", "[]")
                            tids_raw = m.get("clobTokenIds", "[]")

                            outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                            tids = json.loads(tids_raw) if isinstance(tids_raw, str) else tids_raw

                            if len(outcomes) < 2 or len(prices) < 2:
                                continue

                            up_idx = 0 if outcomes[0].lower() == "up" else 1
                            down_idx = 1 - up_idx

                            market = TradableMarket(
                                question=m.get("question", ""),
                                condition_id=m.get("conditionId", ""),
                                token_id_yes=tids[up_idx] if len(tids) > up_idx else "",
                                token_id_no=tids[down_idx] if len(tids) > down_idx else "",
                                yes_price=float(prices[up_idx]),
                                no_price=float(prices[down_idx]),
                                strike_price=0,
                                end_date=m.get("endDate", ""),
                                volume=float(m.get("volume", 0) or 0),
                            )
                            markets.append(market)
                            log.info(
                                f"DAILY MARKET: {market.question} | "
                                f"UP={market.yes_price:.3f} DOWN={market.no_price:.3f}"
                            )
                except Exception:
                    pass

            # ── 2. Also search for "Bitcoin above $X" strike markets ──
            for day_offset in range(0, 5):
                target = today + timedelta(days=day_offset)
                month = target.strftime("%B").lower()
                day_num = target.day
                slug = f"bitcoin-above-on-{month}-{day_num}"

                try:
                    resp = requests.get(f"{GAMMA_HOST}/events", params={
                        "slug": slug,
                    }, timeout=10)
                    events = resp.json()

                    for e in events:
                        for m in e.get("markets", []):
                            if m.get("closed"):
                                continue

                            prices_raw = m.get("outcomePrices", "[]")
                            tids_raw = m.get("clobTokenIds", "[]")
                            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                            tids = json.loads(tids_raw) if isinstance(tids_raw, str) else tids_raw

                            if len(prices) < 2:
                                continue

                            # Extract strike price from question
                            import re
                            q = m.get("question", "")
                            match = re.search(r'\$?([\d,]+)', q)
                            strike = float(match.group(1).replace(",", "")) if match else 0

                            market = TradableMarket(
                                question=q,
                                condition_id=m.get("conditionId", ""),
                                token_id_yes=tids[0] if isinstance(tids, list) and tids else "",
                                token_id_no=tids[1] if isinstance(tids, list) and len(tids) > 1 else "",
                                yes_price=float(prices[0]),
                                no_price=float(prices[1]),
                                strike_price=strike,
                                end_date=m.get("endDate", ""),
                                volume=float(m.get("volume", 0) or 0),
                            )
                            markets.append(market)
                except Exception:
                    pass

        except Exception as e:
            log.error(f"Market scan error: {e}")

        self._cache = markets
        return markets

    def find_best_market(self, btc_price: float) -> Optional[TradableMarket]:
        """
        Find the best market for trading.
        Priority:
          1. "Up or Down" markets (direct match for our signal)
          2. "Above $X" markets where strike is near current price
        """
        markets = self.scan()
        if not markets:
            return None

        # Prefer Up/Down markets (strike_price == 0 means it's an Up/Down market)
        updown = [m for m in markets if m.strike_price == 0 and m.token_id_yes]
        if updown:
            # Pick the one with most volume and closest end date
            updown.sort(key=lambda m: m.volume, reverse=True)
            return updown[0]

        # Fallback: "Above $X" markets, pick closest to current price
        strike_markets = [m for m in markets if m.strike_price > 0]
        if strike_markets:
            scored = []
            for m in strike_markets:
                distance_pct = abs(btc_price - m.strike_price) / m.strike_price * 100
                if distance_pct < 5:
                    scored.append((distance_pct, m))
            if scored:
                scored.sort(key=lambda x: x[0])
                return scored[0][1]

        return None


# ============================================================
# EDGE EVALUATOR
# ============================================================
class EdgeEvaluator:
    """
    Compares streaming predictor signal against market price.
    Only trades when expected edge > cost.
    """

    MIN_EDGE = 0.03  # minimum 3% edge to trade

    def evaluate(self, signal_direction: str, signal_confidence: float,
                 market: TradableMarket, btc_price: float) -> Optional[dict]:
        """
        Evaluate if there's a tradeable edge.

        For "Up or Down" markets:
          - token_id_yes = UP token, token_id_no = DOWN token
          - If we predict UP: buy UP token at yes_price
          - If we predict DOWN: buy DOWN token at no_price

        Returns trade dict or None if no edge.
        """
        if signal_direction not in ("UP", "DOWN"):
            return None

        if signal_confidence < 0.55:
            return None

        is_updown = market.strike_price == 0  # Up/Down market

        if is_updown:
            # Direct mapping: our signal -> which token to buy
            if signal_direction == "UP":
                buy_price = market.yes_price
                token_id = market.token_id_yes
                side = "UP"
                # EV = P(correct) * (1 - price) - P(wrong) * price
                ev = signal_confidence * (1 - buy_price) - (1 - signal_confidence) * buy_price
            else:
                buy_price = market.no_price
                token_id = market.token_id_no
                side = "DOWN"
                ev = signal_confidence * (1 - buy_price) - (1 - signal_confidence) * buy_price

            if ev > self.MIN_EDGE and token_id:
                return {
                    "side": side,
                    "token_id": token_id,
                    "price": buy_price,
                    "ev": ev,
                    "direction": signal_direction,
                    "confidence": signal_confidence,
                    "market": market.question,
                    "condition_id": market.condition_id,
                }
        else:
            # "Above $X" market logic
            our_up_prob = signal_confidence if signal_direction == "UP" else (1 - signal_confidence)
            our_down_prob = 1 - our_up_prob

            ev_yes = our_up_prob * (1 - market.yes_price) - our_down_prob * market.yes_price
            ev_no = our_down_prob * (1 - market.no_price) - our_up_prob * market.no_price

            if ev_yes > ev_no and ev_yes > self.MIN_EDGE:
                return {
                    "side": "YES",
                    "token_id": market.token_id_yes,
                    "price": market.yes_price,
                    "ev": ev_yes,
                    "direction": signal_direction,
                    "confidence": signal_confidence,
                    "market": market.question,
                    "condition_id": market.condition_id,
                }
            elif ev_no > self.MIN_EDGE:
                return {
                    "side": "NO",
                    "token_id": market.token_id_no,
                    "price": market.no_price,
                    "ev": ev_no,
                    "direction": signal_direction,
                    "confidence": signal_confidence,
                    "market": market.question,
                    "condition_id": market.condition_id,
                }

        return None


# ============================================================
# ORDER MANAGER
# ============================================================
class OrderManager:
    """Places and tracks orders on Polymarket."""

    def __init__(self, live: bool = False):
        self.live = live
        self.client = None
        self._positions: list[dict] = []
        self._trade_log_path = LOG_DIR / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if live:
            self._init_client()

    def _init_client(self):
        """Initialize Polymarket CLOB client."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            pk = os.getenv("PK")
            if not pk:
                log.error("No PK found in .env. Cannot trade live.")
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
                # Generate API creds
                self.client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID)
                creds = self.client.create_or_derive_api_creds()
                self.client = ClobClient(CLOB_HOST, key=pk, chain_id=CHAIN_ID, creds=creds)
                log.info("Generated API credentials. Save these to .env:")
                log.info(f"  CLOB_API_KEY={creds.api_key}")
                log.info(f"  CLOB_SECRET={creds.api_secret}")
                log.info(f"  CLOB_PASS_PHRASE={creds.api_passphrase}")

            log.info("Polymarket CLOB client initialized (LIVE MODE)")

        except ImportError:
            log.error("py-clob-client not installed. Install: pip install py-clob-client")
            self.live = False
        except Exception as e:
            log.error(f"CLOB client init failed: {e}")
            self.live = False

    def place_order(self, trade: dict, size_usd: float) -> dict:
        """Place an order (paper or live)."""
        order_record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "mode": "LIVE" if self.live else "PAPER",
            "side": trade["side"],
            "price": trade["price"],
            "size_usd": size_usd,
            "ev": trade["ev"],
            "direction": trade["direction"],
            "confidence": trade["confidence"],
            "market": trade["market"],
            "condition_id": trade["condition_id"],
            "status": "OPEN",
        }

        if self.live and self.client:
            try:
                from py_clob_client.clob_types import OrderArgs
                from py_clob_client.order_builder.constants import BUY

                # Calculate number of contracts
                contracts = size_usd / trade["price"]

                order_args = OrderArgs(
                    price=trade["price"],
                    size=contracts,
                    side=BUY,
                    token_id=trade["token_id"],
                )
                signed_order = self.client.create_order(order_args)
                resp = self.client.post_order(signed_order)

                order_record["order_id"] = resp.get("orderID", "")
                order_record["status"] = "PLACED"
                log.info(f"LIVE ORDER PLACED: {trade['side']} ${size_usd:.2f} @ {trade['price']:.3f}")

            except Exception as e:
                order_record["status"] = "FAILED"
                order_record["error"] = str(e)
                log.error(f"Order failed: {e}")
        else:
            order_record["status"] = "PAPER"
            log.info(
                f"PAPER TRADE: {trade['side']} ${size_usd:.2f} @ {trade['price']:.3f} | "
                f"EV={trade['ev']:+.3f} | {trade['direction']} {trade['confidence']:.0%}"
            )

        # Log to file
        with open(self._trade_log_path, "a") as f:
            f.write(json.dumps(order_record) + "\n")

        self._positions.append(order_record)
        return order_record


# ============================================================
# POSITION REDEEMER (auto-redeem winning positions)
# ============================================================
# Polygon contract addresses
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYGON_RPC = "https://polygon-rpc.com"

# Minimal CTF ABI for redeemPositions
CTF_REDEEM_ABI = json.loads("""[{
    "inputs": [
        {"name": "collateralToken", "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId", "type": "bytes32"},
        {"name": "indexSets", "type": "uint256[]"}
    ],
    "name": "redeemPositions",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
}]""")


class PositionRedeemer:
    """
    Auto-redeems winning positions after 15-minute markets resolve.

    Flow:
      1. Tracks all live-placed positions with condition_id + token_id
      2. Polls Polymarket API every 60s to check if markets resolved
      3. If resolved and we hold winning tokens, calls CTF.redeemPositions
      4. Logs redemption with tx hash
    """

    def __init__(self, private_key: str = None):
        self._positions: list[dict] = []
        self._redeemed: list[dict] = []
        self._pk = private_key or os.getenv("PK", "")
        self._w3 = None
        self._account = None
        self._ctf = None
        self._log_path = LOG_DIR / f"redemptions_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if self._pk:
            self._init_web3()

    def _init_web3(self):
        """Initialize Web3 connection to Polygon."""
        try:
            from web3 import Web3

            self._w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            if not self._w3.is_connected():
                log.warning("Cannot connect to Polygon RPC")
                return

            # Derive account from private key
            pk = self._pk if self._pk.startswith("0x") else f"0x{self._pk}"
            self._account = self._w3.eth.account.from_key(pk)
            self._ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
                abi=CTF_REDEEM_ABI,
            )
            log.info(f"Web3 connected to Polygon. Wallet: {self._account.address}")
        except ImportError:
            log.warning("web3 not installed. Auto-redeem disabled.")
        except Exception as e:
            log.warning(f"Web3 init failed: {e}")

    def track_position(self, order: dict):
        """Track a new live position for future redemption."""
        if order.get("status") in ("PLACED", "FILLED"):
            self._positions.append({
                "condition_id": order.get("condition_id", ""),
                "token_id": order.get("token_id", ""),
                "side": order.get("side", ""),
                "size_usd": order.get("size_usd", 0),
                "market": order.get("market", ""),
                "placed_at": order.get("time", ""),
                "redeemed": False,
            })
            log.info(f"Tracking position for redemption: {order.get('market', '')}")

    def check_and_redeem(self):
        """Check all tracked positions and redeem resolved ones."""
        if not self._positions:
            return

        for pos in list(self._positions):
            if pos["redeemed"]:
                continue

            cid = pos["condition_id"]
            if not cid:
                continue

            # Check if market is resolved
            try:
                resp = requests.get(f"{GAMMA_HOST}/markets", params={
                    "conditionId": cid,
                    "limit": 1,
                }, timeout=10)
                data = resp.json()

                if not data:
                    continue

                market = data[0] if isinstance(data, list) else data
                is_closed = market.get("closed", False)

                if not is_closed:
                    continue  # not resolved yet

                # Market resolved -- attempt redemption
                log.info(f"Market RESOLVED: {pos['market']}")

                if self._w3 and self._ctf and self._account:
                    self._redeem_on_chain(pos)
                else:
                    log.info(f"  Auto-redeem skipped (no Web3). Redeem manually on polymarket.com")

                pos["redeemed"] = True

            except Exception as e:
                log.debug(f"Resolution check failed for {cid[:20]}: {e}")

    def _redeem_on_chain(self, pos: dict):
        """Call CTF.redeemPositions on Polygon."""
        try:
            from web3 import Web3

            condition_id_bytes = bytes.fromhex(pos["condition_id"].replace("0x", ""))
            parent_collection_id = b"\x00" * 32
            index_sets = [1, 2]  # both outcomes

            tx = self._ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent_collection_id,
                condition_id_bytes,
                index_sets,
            ).build_transaction({
                "from": self._account.address,
                "nonce": self._w3.eth.get_transaction_count(self._account.address),
                "gas": 300000,
                "gasPrice": self._w3.eth.gas_price,
                "chainId": CHAIN_ID,
            })

            signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            success = receipt["status"] == 1
            tx_hash_hex = tx_hash.hex()

            record = {
                "time": datetime.now(timezone.utc).isoformat(),
                "market": pos["market"],
                "condition_id": pos["condition_id"],
                "side": pos["side"],
                "size_usd": pos["size_usd"],
                "tx_hash": tx_hash_hex,
                "success": success,
                "gas_used": receipt.get("gasUsed", 0),
            }

            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if success:
                log.info(f"  REDEEMED: tx={tx_hash_hex[:20]}... | {pos['market']}")
                self._redeemed.append(record)
            else:
                log.error(f"  Redeem TX FAILED: tx={tx_hash_hex[:20]}...")

        except Exception as e:
            log.error(f"  Redeem error: {e}")

    def get_stats(self) -> dict:
        return {
            "tracked": len(self._positions),
            "redeemed": len(self._redeemed),
            "pending": sum(1 for p in self._positions if not p["redeemed"]),
        }


# ============================================================
# PAPER TRADE ENGINE (trades against Binance price)
# ============================================================
class PaperTradeEngine:
    """
    Simulates Polymarket trades using Binance price.
    Used when no suitable Polymarket markets are available.
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk = risk_manager
        self._pending: list[dict] = []  # trades waiting for resolution
        self._log_path = LOG_DIR / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.btc_price = 0.0

    def on_signal(self, direction: str, confidence: float, price: float):
        """Process a signal from the streaming predictor."""
        can, reason = self.risk.can_trade()
        if not can:
            return

        if direction not in ("UP", "DOWN") or confidence < 0.50:
            return

        # Dynamic position sizing based on confidence
        size = self.risk.sized_trade(confidence)

        # Simulate buying a binary option at implied price
        buy_price = 0.50
        slippage = 0.02
        spread = 0.01
        effective_cost = buy_price * (1 + slippage) + spread

        trade = {
            "time": time.time(),
            "direction": direction,
            "confidence": confidence,
            "entry_price": price,
            "buy_cost": effective_cost,
            "size": size,
            "check_after": 60,  # check result after 60 seconds
        }

        self._pending.append(trade)
        self.risk.record_open()

        log.info(
            f"PAPER: {direction} ${self.risk.trade_size:.0f} @ ${price:,.2f} | "
            f"conf={confidence:.0%}"
        )

    def check_results(self, current_price: float):
        """Check if any pending trades have resolved."""
        self.btc_price = current_price
        now = time.time()
        resolved = []

        for trade in self._pending:
            age = now - trade["time"]
            if age >= trade["check_after"]:
                # Determine result
                if trade["direction"] == "UP":
                    correct = current_price > trade["entry_price"]
                else:
                    correct = current_price < trade["entry_price"]

                if correct:
                    pnl = (1.0 - trade["buy_cost"]) * trade["size"]
                else:
                    pnl = -trade["buy_cost"] * trade["size"]

                self.risk.record_trade(pnl)
                self.risk.record_close()

                result = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "direction": trade["direction"],
                    "confidence": trade["confidence"],
                    "entry_price": trade["entry_price"],
                    "exit_price": current_price,
                    "correct": correct,
                    "pnl": pnl,
                    "size": trade["size"],
                    "running_pnl": self.risk.daily_pnl,
                    "win_rate": self.risk.win_rate,
                }

                with open(self._log_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                icon = "+" if correct else "-"
                log.info(
                    f"RESULT: {icon} ${pnl:+.2f} | {trade['direction']} "
                    f"{'CORRECT' if correct else 'WRONG'} | "
                    f"Entry ${trade['entry_price']:,.2f} → ${current_price:,.2f} | "
                    f"Day: ${self.risk.daily_pnl:+.2f} | WR: {self.risk.win_rate:.0%} "
                    f"({self.risk.total_wins}/{self.risk.total_trades})"
                )

                resolved.append(trade)

        for t in resolved:
            self._pending.remove(t)


# ============================================================
# MAIN LOOP
# ============================================================
async def run_trader(live: bool = False):
    """Main trading loop: streaming predictor + Polymarket/paper trading."""

    # Import streaming predictor components
    from streaming_predictor import (
        OrderbookStream, TradeStream, MempoolStream,
        StreamingFeatureEngine, StreamingPredictor,
        SYMBOL, ETH_SYMBOL, PREDICTION_INTERVAL,
    )

    # ── Initialize components ──
    bankroll = float(os.getenv("BANKROLL", "500"))
    risk = RiskManager(
        bankroll=bankroll,
        max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.05")),
        daily_loss_limit=float(os.getenv("DAILY_LOSS_LIMIT", "0.20")),
        cooldown_seconds=float(os.getenv("COOLDOWN_SECONDS", "60")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "3")),
    )

    # ── Load encrypted credentials if .env.enc exists ──
    enc_path = Path(".env.enc")
    if enc_path.exists() and live:
        try:
            from security import load_encrypted_env
            enc_password = os.getenv("ENC_PASSWORD")
            if not enc_password:
                import getpass
                enc_password = getpass.getpass("Enter .env encryption password: ")
            env_vars = load_encrypted_env(str(enc_path), enc_password)
            for k, v in env_vars.items():
                os.environ[k] = v
            log.info("Loaded encrypted credentials from .env.enc")
        except Exception as e:
            log.error(f"Failed to decrypt .env.enc: {e}")
            if live:
                log.error("Cannot start live mode without credentials. Exiting.")
                return

    market_finder = MarketFinder()
    edge_eval = EdgeEvaluator()
    order_mgr = OrderManager(live=live)
    paper_engine = PaperTradeEngine(risk)

    # ── Position Redeemer (auto-redeem winning positions) ──
    redeemer = PositionRedeemer(private_key=os.getenv("PK", ""))
    redeem_tick = 0

    # ── Initialize streaming predictor ──
    orderbook = OrderbookStream()
    btc_trades = TradeStream(SYMBOL, is_eth=False)
    eth_trades = TradeStream(ETH_SYMBOL, is_eth=True)
    mempool = MempoolStream()

    engine = StreamingFeatureEngine(orderbook, btc_trades, eth_trades, mempool)
    predictor = StreamingPredictor(engine)

    # Self-train
    log.info("Training ML model on recent data...")
    try:
        from ultra_predictor import BinanceDataCollector, engineer_all_features, create_targets, get_feature_cols
        collector = BinanceDataCollector("BTCUSDT")
        df = collector.get_klines_history(days=7)
        if not df.empty:
            df = engineer_all_features(df)
            df = create_targets(df)
            feat_cols = get_feature_cols(df)
            for c in feat_cols:
                df[c] = df[c].fillna(df[c].median())

            from sklearn.preprocessing import StandardScaler
            target = "target_1"
            clean = df.dropna(subset=feat_cols + [target])
            X = clean[feat_cols].values
            y = clean[target].values

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            import lightgbm as lgb
            m = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                verbose=-1, n_jobs=-1, random_state=42)
            m.fit(X_s, y)
            predictor.models = [("lgb", m)]
            predictor.scaler = scaler
            predictor.feature_cols = feat_cols
            predictor.model_loaded = True
            log.info(f"Model trained on {len(clean):,} rows")
    except Exception as e:
        log.warning(f"Model training failed: {e}")

    # ── Start streams ──
    tasks = [
        asyncio.create_task(orderbook.run()),
        asyncio.create_task(btc_trades.run()),
        asyncio.create_task(eth_trades.run()),
        asyncio.create_task(mempool.run()),
    ]

    log.info("Starting data streams...")
    await asyncio.sleep(3)

    mode_str = "LIVE" if live else "PAPER"

    print(f"\n{'=' * 72}")
    print(f"  POLYMARKET TRADER [{mode_str}]")
    print(f"{'=' * 72}")
    print(f"  Bankroll:  ${bankroll:,.0f}")
    print(f"  Per trade: ${risk.trade_size:,.0f} ({risk.max_risk_per_trade:.0%})")
    print(f"  Loss limit: ${bankroll * risk.daily_loss_limit:,.0f}/day")
    print(f"  Cooldown:   {risk.cooldown_seconds:.0f}s")
    print(f"  Mode:       {mode_str}")
    if not live:
        print(f"  (Paper trading against Binance price. Use --live for real money)")
    print(f"{'=' * 72}\n")

    # ── Scan for Polymarket markets ──
    log.info("Scanning Polymarket for BTC price markets...")
    poly_markets = market_finder.scan()
    if poly_markets:
        log.info(f"Found {len(poly_markets)} tradable BTC markets:")
        for m in poly_markets[:5]:
            log.info(f"  {m.question} | YES=${m.yes_price:.3f}")
    else:
        log.info("No short-term BTC price markets currently active on Polymarket.")
        log.info("Running in paper-trade mode against Binance BTC price.")
        log.info("Will re-scan every 5 minutes for new markets.")

    tick = 0
    try:
        while True:
            await asyncio.sleep(PREDICTION_INTERVAL)
            tick += 1

            # Get streaming features + prediction
            features = engine.compute_features()
            pred = predictor.predict(features)
            predictor.check_accuracy()

            btc_price = features.get("btc_price", 0)
            paper_engine.check_results(btc_price)

            # Only act on actionable signals
            if pred["signal"] in ("STRONG", "MODERATE"):
                can, reason = risk.can_trade()

                if can:
                    # Try Polymarket first
                    best_market = market_finder.find_best_market(btc_price) if tick % 300 == 1 or poly_markets else None

                    if best_market and live:
                        trade = edge_eval.evaluate(
                            pred["direction"], pred["confidence"],
                            best_market, btc_price,
                        )
                        if trade:
                            order_record = order_mgr.place_order(trade, risk.trade_size)
                            risk.record_open()
                            # Track for auto-redemption
                            order_record["token_id"] = trade.get("token_id", "")
                            redeemer.track_position(order_record)
                    else:
                        # Paper trade against Binance price
                        paper_engine.on_signal(pred["direction"], pred["confidence"], btc_price)

            # Periodic status (every 30 seconds)
            if tick % 30 == 0:
                signal_stats = predictor.get_stats()
                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"BTC ${btc_price:>10,.2f} | "
                    f"{risk.status()} | "
                    f"Stream WR: {signal_stats['win_rate']:.0%} "
                    f"({signal_stats['correct']}/{signal_stats['checked']})"
                )

            # Re-scan Polymarket every 5 minutes
            if tick % 300 == 0:
                poly_markets = market_finder.scan()
                if poly_markets:
                    log.info(f"Polymarket: {len(poly_markets)} BTC markets available")

            # Check for resolved positions to redeem (every 60 seconds)
            if tick % 60 == 0 and live:
                redeemer.check_and_redeem()
                redeem_stats = redeemer.get_stats()
                if redeem_stats["pending"] > 0:
                    log.info(
                        f"Redemptions: {redeem_stats['redeemed']} done, "
                        f"{redeem_stats['pending']} pending"
                    )

    except asyncio.CancelledError:
        pass
    finally:
        for t in tasks:
            t.cancel()

        print(f"\n{'=' * 72}")
        print(f"  SESSION SUMMARY")
        print(f"{'=' * 72}")
        print(f"  Mode:       {mode_str}")
        print(f"  Trades:     {risk.total_trades}")
        print(f"  Wins:       {risk.total_wins}")
        print(f"  Win Rate:   {risk.win_rate:.1%}" if risk.total_trades > 0 else "  Win Rate: N/A")
        print(f"  Day P&L:    ${risk.daily_pnl:+,.2f}")
        print(f"  Bankroll:   ${risk.bankroll + risk.daily_pnl:,.2f}")
        signal_stats = predictor.get_stats()
        print(f"  Signals:    {signal_stats['signals']} ({signal_stats['correct']}/{signal_stats['checked']} checked)")
        print(f"  Log:        {paper_engine._log_path}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Live Trader")
    parser.add_argument("--live", action="store_true",
                        help="Enable LIVE trading with real money (default: paper)")
    args = parser.parse_args()

    if args.live:
        pk = os.getenv("PK")
        if not pk:
            print("ERROR: --live requires PK in .env file")
            print("  1. cp .env.example .env")
            print("  2. Add your Polymarket private key to .env")
            sys.exit(1)

        print("\n" + "!" * 72)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be used. You can lose your entire bankroll.")
        print("!" * 72)

        confirm = input("\n  Type 'YES I UNDERSTAND' to continue: ")
        if confirm.strip() != "YES I UNDERSTAND":
            print("  Aborted.")
            sys.exit(0)

    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        print("\n\nShutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(run_trader(live=args.live))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
