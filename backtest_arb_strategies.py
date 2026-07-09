#!/usr/bin/env python3
"""
backtest_arb_strategies.py — Full backtest of Polymarket arbitrage strategies
with realistic order book simulation.

Strategies tested:
  1. Core: Buy YES at discount, exit at 90-95¢ near expiry
  2. Complement Arb: YES(p) + NO(1-p) < $1 → risk-free profit
  3. Statistical Model Arb: p_theo vs market mispricing → scale in
  4. Cross-Interval Arb: 5m vs 15m inconsistency
  5. Oracle Snipe: Chainlink settlement lag (1-2s) vs Binance

Data: 5000 cached real Polymarket 5m BTC markets + Binance 1m klines
Order book: Simulated from real market parameters (spread, depth, fees)
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("arb_backtest")

CACHE_DIR = Path("/Users/eiji/polymarket-pnl/data/cache")

# ============================================================================
# Fee Engine (from config/fees.yaml)
# ============================================================================

TAKER_FEE_SCHEDULE = [
    (0.45, 0.55, 0.03),   # near 50¢ → 3%
    (0.30, 0.45, 0.02),   # 30-45¢ → 2%
    (0.55, 0.70, 0.02),   # 55-70¢ → 2%
    (0.00, 0.30, 0.01),   # deep OTM → 1%
    (0.70, 1.00, 0.01),   # deep ITM → 1%
]
MAKER_REBATE = 0.005  # 0.5%


def taker_fee(price: float) -> float:
    for lo, hi, fee in TAKER_FEE_SCHEDULE:
        if lo <= price <= hi:
            return fee
    return 0.02


def maker_rebate() -> float:
    return MAKER_REBATE


# ============================================================================
# Order Book Simulator
# ============================================================================

@dataclass
class BookLevel:
    price: float
    size: float  # in USD notional


@dataclass
class OrderBook:
    """Simulated L2 order book for one side of a Polymarket binary market."""
    yes_bids: list[BookLevel] = field(default_factory=list)
    yes_asks: list[BookLevel] = field(default_factory=list)
    no_bids: list[BookLevel] = field(default_factory=list)
    no_asks: list[BookLevel] = field(default_factory=list)
    spread_cents: float = 1.0  # typical 1-2 cent spread

    @property
    def yes_mid(self) -> float:
        if self.yes_bids and self.yes_asks:
            return (self.yes_bids[0].price + self.yes_asks[0].price) / 2
        return 0.50

    @property
    def yes_best_bid(self) -> float:
        return self.yes_bids[0].price if self.yes_bids else 0.0

    @property
    def yes_best_ask(self) -> float:
        return self.yes_asks[0].price if self.yes_asks else 1.0

    @property
    def no_best_bid(self) -> float:
        return self.no_bids[0].price if self.no_bids else 0.0

    @property
    def no_best_ask(self) -> float:
        return self.no_asks[0].price if self.no_asks else 1.0

    @property
    def complement_cost(self) -> float:
        """Cost to buy YES + NO = guaranteed $1 payout."""
        return self.yes_best_ask + self.no_best_ask

    @property
    def total_bid_depth(self) -> float:
        return sum(b.size for b in self.yes_bids[:5])

    @property
    def total_ask_depth(self) -> float:
        return sum(a.size for a in self.yes_asks[:5])


def simulate_order_book(
    fair_value: float,
    volume: float,
    btc_vol: float,
    time_to_expiry_frac: float,  # 0=just opened, 1=about to resolve
    rng: np.random.Generator,
) -> OrderBook:
    """
    Simulate a realistic Polymarket order book for a 5m BTC market.

    Based on observed parameters:
      - Spread: 1-3 cents (tighter near expiry)
      - Depth: $200-$5000 per level (proportional to volume)
      - 5 levels each side
      - Skew: book is slightly thinner on the side away from fair value
    """
    # Spread narrows as expiry approaches and widens in high vol
    base_spread = 0.01 + 0.01 * btc_vol * 100  # 1-3¢ typical
    spread = base_spread * (1.0 - 0.5 * time_to_expiry_frac)  # tighter near expiry
    spread = max(0.005, min(0.05, spread))

    half_spread = spread / 2

    # Depth per level: proportional to market volume
    # Real markets show $500-$5000 at top of book for $10k+ volume markets
    base_depth = max(100, volume * 0.05)
    depth_decay = 0.7  # each level has 70% of the prior level's depth

    book = OrderBook(spread_cents=spread * 100)

    for i in range(5):
        level_depth = base_depth * (depth_decay ** i) * (0.8 + 0.4 * rng.random())

        bid_price = round(fair_value - half_spread - i * 0.01, 3)
        ask_price = round(fair_value + half_spread + i * 0.01, 3)

        if bid_price > 0.001:
            book.yes_bids.append(BookLevel(max(0.001, bid_price), level_depth))
        if ask_price < 0.999:
            book.yes_asks.append(BookLevel(min(0.999, ask_price), level_depth))

        # NO side mirrors YES: NO bid = 1 - YES ask, NO ask = 1 - YES bid
        no_bid = round(1.0 - ask_price, 3)
        no_ask = round(1.0 - bid_price, 3)
        if no_bid > 0.001:
            book.no_bids.append(BookLevel(max(0.001, no_bid), level_depth * 0.9))
        if no_ask < 0.999:
            book.no_asks.append(BookLevel(min(0.999, no_ask), level_depth * 0.9))

    return book


def simulate_exit_book(
    btc_move_direction: int,  # 1 if UP, 0 if DOWN
    time_remaining_frac: float,  # how close to settlement
    volume: float,
    rng: np.random.Generator,
) -> OrderBook:
    """Simulate book near exit/settlement time."""
    if time_remaining_frac < 0.1:
        # Near settlement — price converges to 0 or 1
        if btc_move_direction == 1:
            fair = 0.90 + 0.09 * (1 - time_remaining_frac)
        else:
            fair = 0.10 - 0.09 * (1 - time_remaining_frac)
    else:
        # Mid-life — still uncertain
        if btc_move_direction == 1:
            fair = 0.50 + 0.40 * (1 - time_remaining_frac)
        else:
            fair = 0.50 - 0.40 * (1 - time_remaining_frac)

    fair = max(0.05, min(0.95, fair))
    return simulate_order_book(fair, volume, 0.005, 1 - time_remaining_frac, rng)


def execute_taker_buy(book_side: list[BookLevel], amount_usd: float, price_limit: float = 1.0):
    """
    Simulate taker buy walking the book. Returns (avg_fill_price, filled_amount, shares).
    """
    remaining = amount_usd
    total_shares = 0
    total_cost = 0

    for level in book_side:
        if level.price > price_limit:
            break
        fillable = min(remaining, level.size)
        shares = fillable / level.price
        total_shares += shares
        total_cost += fillable
        remaining -= fillable
        if remaining <= 0:
            break

    if total_shares == 0:
        return 0, 0, 0

    avg_price = total_cost / total_shares
    return avg_price, total_cost, total_shares


# ============================================================================
# Lead-Lag Signal (from state.py)
# ============================================================================

def compute_p_theo(klines: pd.DataFrame, round_ts: int, interval_min: int = 5) -> float:
    """Compute theoretical probability using drift+vol model."""
    mask = klines["ts"] < round_ts
    available = klines[mask]
    if len(available) < 60:
        return 0.5

    window = available.iloc[-60:]
    closes = window["close"].values
    taker_buy = window["taker_buy_base"].values
    volumes = window["volume"].values

    if len(closes) >= 2:
        ret_30s = np.log(closes[-1] / closes[-2])
    else:
        ret_30s = 0.0

    log_rets = np.diff(np.log(closes))
    vol_1m = np.std(log_rets) if len(log_rets) >= 2 else 0.001

    total_vol = volumes[-5:].sum()
    ofi = (taker_buy[-5:].sum() / total_vol - 0.5) * 2.0 if total_vol > 0 else 0.0
    ofi_norm = max(-1.0, min(1.0, ofi))

    ofi_weight = 0.3
    drift = ret_30s + ofi_weight * ofi_norm * vol_1m * 0.01
    drift_z = drift / (max(vol_1m, 1e-8) * np.sqrt(max(interval_min, 0.01)))

    p_theo = 0.5 * (1.0 + math.erf(drift_z / np.sqrt(2.0)))
    return max(0.01, min(0.99, p_theo))


def compute_btc_volatility(klines: pd.DataFrame, round_ts: int, window: int = 30) -> float:
    """Compute recent BTC realized volatility (1m)."""
    mask = klines["ts"] < round_ts
    available = klines[mask]
    if len(available) < window:
        return 0.005
    closes = available.iloc[-window:]["close"].values
    log_rets = np.diff(np.log(closes))
    return np.std(log_rets) if len(log_rets) > 1 else 0.005


def oracle_snipe_signal(klines: pd.DataFrame, round_ts: int, expiry_ts: int) -> dict | None:
    """
    Oracle Snipe: Chainlink settlement feed lags Binance by 1-2 seconds.
    When BTC moves sharply in the last 30 seconds before settlement,
    there's an opportunity to buy the side that Chainlink will report.

    We check the LAST 1-minute candle overlapping with settlement.
    If it shows strong directional movement, we snipe.
    """
    # Look at the candle that contains the settlement time
    settle_candle_ts = (expiry_ts // 60) * 60
    mask = (klines["ts"] >= settle_candle_ts - 60) & (klines["ts"] <= settle_candle_ts)
    candles = klines[mask]

    if candles.empty:
        return None

    last = candles.iloc[-1]
    move = (last["close"] - last["open"]) / last["open"]

    # Only snipe on strong moves (>0.02% in 1 minute = significant for BTC)
    if abs(move) < 0.0002:
        return None

    # Direction BTC is heading at settlement
    direction = "YES" if move > 0 else "NO"
    confidence = min(abs(move) * 5000, 0.95)  # scale move to confidence

    return {
        "direction": direction,
        "confidence": confidence,
        "btc_move_pct": move * 100,
        "edge_source": "oracle_lag",
    }


# ============================================================================
# Strategy Implementations
# ============================================================================

@dataclass
class Trade:
    ts: int
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    shares: float
    amount_usd: float
    pnl: float
    fee_paid: float
    is_maker: bool
    edge_bps: float
    won: bool
    details: dict = field(default_factory=dict)


def strategy_core_buy_exit(
    book_entry: OrderBook,
    book_exit: OrderBook,
    outcome: int,
    volume: float,
    bankroll: float,
    max_bet: float,
    rng: np.random.Generator,
) -> Trade | None:
    """
    Core Strategy: Buy YES when price drops below 50¢ (discount),
    hold until near expiry when price converges to 90-95¢.

    Entry: Buy YES at ~40-60¢ (taker or maker)
    Exit: Sell at 90-95¢ near settlement, or hold to settlement
    """
    yes_ask = book_entry.yes_best_ask
    no_ask = book_entry.no_best_ask

    # Look for entry below 50¢ (discount)
    # Also check NO side if cheaper
    buy_side = None
    buy_price = 0

    if yes_ask <= 0.60:
        buy_side = "YES"
        buy_price = yes_ask
    elif no_ask <= 0.60:
        buy_side = "NO"
        buy_price = no_ask

    if buy_side is None:
        return None

    # Position size: up to max_bet, scaled by how cheap the entry is
    discount = 0.50 - buy_price  # positive if below fair
    size_frac = min(0.05 + discount * 2, 0.10)  # 5-10% of bankroll
    amount = min(size_frac * bankroll, max_bet)
    if amount < 5:
        return None

    shares = amount / buy_price
    entry_fee = taker_fee(buy_price) * amount

    # Exit simulation: near expiry, price converges
    if buy_side == "YES":
        if outcome == 1:
            # Won — exit at 90-95¢ or hold to $1
            exit_price = 0.90 + rng.random() * 0.09  # 90-99¢
            pnl = shares * (exit_price - buy_price)
            won = True
        else:
            # Lost — price drops toward 0
            exit_price = 0.02 + rng.random() * 0.08  # 2-10¢
            pnl = shares * (exit_price - buy_price)
            won = False
    else:
        if outcome == 0:
            exit_price = 0.90 + rng.random() * 0.09
            pnl = shares * (exit_price - buy_price)
            won = True
        else:
            exit_price = 0.02 + rng.random() * 0.08
            pnl = shares * (exit_price - buy_price)
            won = False

    # Exit fee (taker to get out fast)
    exit_fee = taker_fee(exit_price) * shares * exit_price
    total_fee = entry_fee + exit_fee
    pnl -= total_fee

    return Trade(
        ts=0, strategy="Core Buy-Exit", side=buy_side,
        entry_price=buy_price, exit_price=exit_price,
        shares=shares, amount_usd=amount, pnl=pnl,
        fee_paid=total_fee, is_maker=False,
        edge_bps=round((exit_price - buy_price) / buy_price * 10000, 1) if won else 0,
        won=won,
        details={"discount": round(0.50 - buy_price, 3)},
    )


def strategy_complement_arb(
    book: OrderBook,
    volume: float,
    max_bet: float,
    rng: np.random.Generator,
) -> Trade | None:
    """
    Complement Arb: Buy YES(p) + NO(1-p) when total cost < $1.00.
    Guaranteed $1.00 payout regardless of outcome.

    With one maker leg: YES maker + NO taker (or vice versa).
    Profit = $1.00 - cost - fees + rebate.
    """
    yes_ask = book.yes_best_ask
    no_ask = book.no_best_ask
    total_cost = yes_ask + no_ask

    if total_cost >= 1.00:
        return None

    # Gross profit per share before fees
    gross_per_share = 1.00 - total_cost

    # Fee model: one leg maker (rebate), one leg taker
    # Put maker order on the more liquid side
    maker_leg_price = max(yes_ask, no_ask)
    taker_leg_price = min(yes_ask, no_ask)

    maker_rebate_amt = maker_rebate() * maker_leg_price  # income
    taker_fee_amt = taker_fee(taker_leg_price) * taker_leg_price  # cost

    net_per_share = gross_per_share + maker_rebate_amt - taker_fee_amt

    if net_per_share <= 0:
        return None

    # Size: limited by book depth on thinnest side
    depth = min(book.total_ask_depth, sum(a.size for a in book.no_asks[:5]))
    amount = min(depth * 0.5, max_bet)  # don't take more than 50% of book
    if amount < 5:
        return None

    shares = amount / total_cost
    total_pnl = shares * net_per_share

    return Trade(
        ts=0, strategy="Complement Arb", side="YES+NO",
        entry_price=total_cost, exit_price=1.00,
        shares=shares, amount_usd=amount, pnl=total_pnl,
        fee_paid=taker_fee_amt * shares - maker_rebate_amt * shares,
        is_maker=True,
        edge_bps=round(net_per_share / total_cost * 10000, 1),
        won=True,  # guaranteed
        details={
            "yes_ask": yes_ask, "no_ask": no_ask,
            "gross_per_share": round(gross_per_share, 4),
            "net_per_share": round(net_per_share, 4),
        },
    )


def strategy_stat_model_arb(
    p_theo: float,
    book: OrderBook,
    outcome: int,
    bankroll: float,
    max_bet: float,
    kelly_alpha: float = 0.25,
) -> Trade | None:
    """
    Statistical Model Arb: Compare p_theo from CEX lead-lag model
    against the order book price. If mispricing > threshold, scale in.

    Uses Kelly sizing with fractional scaling.
    Entry via postOnly maker order for rebate.
    """
    yes_mid = book.yes_mid
    edge_yes = p_theo - yes_mid
    edge_no = (1 - p_theo) - (1 - yes_mid)

    min_edge = 0.03  # 3% minimum edge

    side = None
    edge = 0
    entry_price = 0

    if edge_yes > min_edge:
        side = "YES"
        edge = edge_yes
        # Maker: place limit order at best bid + 1 tick
        entry_price = book.yes_best_bid + 0.01
    elif edge_no > min_edge:
        side = "NO"
        edge = edge_no
        entry_price = book.no_best_bid + 0.01

    if side is None:
        return None

    # Kelly sizing: f* = (p - q) / (1 - q) with alpha scaling
    if side == "YES":
        p = p_theo
        q = entry_price
    else:
        p = 1 - p_theo
        q = entry_price

    if q <= 0 or q >= 1 or p <= q:
        return None

    f_star = (p - q) / (1 - q)
    f = kelly_alpha * f_star
    amount = min(f * bankroll, max_bet)
    if amount < 5:
        return None

    shares = amount / entry_price

    # Maker order: earn rebate
    rebate = maker_rebate() * amount

    # Settlement
    if side == "YES":
        won = outcome == 1
    else:
        won = outcome == 0

    if won:
        pnl = shares * (1.0 - entry_price) + rebate
    else:
        pnl = -shares * entry_price + rebate

    return Trade(
        ts=0, strategy="Stat Model Arb", side=side,
        entry_price=entry_price, exit_price=1.0 if won else 0.0,
        shares=shares, amount_usd=amount, pnl=pnl,
        fee_paid=-rebate,  # negative = income
        is_maker=True,
        edge_bps=round(edge * 10000, 1),
        won=won,
        details={
            "p_theo": round(p_theo, 4),
            "market_mid": round(yes_mid, 4),
            "kelly_f": round(f, 4),
        },
    )


def strategy_cross_interval_arb(
    p_theo_5m: float,
    p_theo_15m: float,
    book_5m: OrderBook,
    outcome_5m: int,
    bankroll: float,
    max_bet: float,
) -> Trade | None:
    """
    Cross-Interval Arb: 5m and 15m markets on the same asset can
    become inconsistent. If 5m says 60% UP but 15m says 40% UP
    (implied from its structure), the 5m market may be overpriced.

    Model: 15m encompasses 3x 5m periods.  If the 5m move direction
    is consistent with 15m trend, there's a carry.  If inconsistent,
    there's a reversion opportunity.
    """
    # Divergence between 5m and 15m implied probabilities
    divergence = abs(p_theo_5m - p_theo_15m)

    if divergence < 0.05:  # need at least 5% divergence
        return None

    # Trade the 5m market in the direction the 15m model suggests
    # (assuming 15m model is more informed due to larger sample)
    if p_theo_15m > 0.55 and p_theo_5m < p_theo_15m - 0.05:
        side = "YES"
        edge = p_theo_15m - book_5m.yes_best_ask
    elif p_theo_15m < 0.45 and p_theo_5m > p_theo_15m + 0.05:
        side = "NO"
        edge = (1 - p_theo_15m) - book_5m.no_best_ask
    else:
        return None

    if edge < 0.02:
        return None

    entry_price = book_5m.yes_best_ask if side == "YES" else book_5m.no_best_ask
    amount = min(0.03 * bankroll, max_bet)
    if amount < 5:
        return None

    shares = amount / entry_price
    rebate = maker_rebate() * amount

    won = (side == "YES" and outcome_5m == 1) or (side == "NO" and outcome_5m == 0)

    if won:
        pnl = shares * (1.0 - entry_price) + rebate
    else:
        pnl = -shares * entry_price + rebate

    return Trade(
        ts=0, strategy="Cross-Interval Arb", side=side,
        entry_price=entry_price, exit_price=1.0 if won else 0.0,
        shares=shares, amount_usd=amount, pnl=pnl,
        fee_paid=-rebate, is_maker=True,
        edge_bps=round(edge * 10000, 1),
        won=won,
        details={
            "p_theo_5m": round(p_theo_5m, 4),
            "p_theo_15m": round(p_theo_15m, 4),
            "divergence": round(divergence, 4),
        },
    )


def strategy_oracle_snipe(
    signal: dict,
    book: OrderBook,
    outcome: int,
    bankroll: float,
    max_bet: float,
    rng: np.random.Generator,
) -> Trade | None:
    """
    Oracle Snipe: When Binance shows BTC moving strongly in the final
    seconds before settlement but Chainlink hasn't updated yet, the
    market hasn't caught up.  We buy the correct side before settlement.

    The 1-2 second lag means we need to be fast (assumed: co-located).
    """
    if signal is None:
        return None

    direction = signal["direction"]
    confidence = signal["confidence"]

    if confidence < 0.60:
        return None

    # Entry: aggressive taker to get filled before oracle updates
    if direction == "YES":
        entry_price = book.yes_best_ask
        won = outcome == 1
    else:
        entry_price = book.no_best_ask
        won = outcome == 0

    # Size proportional to confidence
    amount = min(confidence * 0.05 * bankroll, max_bet)
    if amount < 5:
        return None

    shares = amount / entry_price
    fee = taker_fee(entry_price) * amount

    if won:
        pnl = shares * (1.0 - entry_price) - fee
    else:
        pnl = -shares * entry_price - fee

    return Trade(
        ts=0, strategy="Oracle Snipe", side=direction,
        entry_price=entry_price, exit_price=1.0 if won else 0.0,
        shares=shares, amount_usd=amount, pnl=pnl,
        fee_paid=fee, is_maker=False,
        edge_bps=round(signal["btc_move_pct"] * 100, 1),
        won=won,
        details={
            "confidence": round(confidence, 3),
            "btc_move_pct": round(signal["btc_move_pct"], 4),
            "lag_exploited": True,
        },
    )


# ============================================================================
# Main Backtest Engine
# ============================================================================

def load_data():
    """Load cached Polymarket + Binance data."""
    pm_path = CACHE_DIR / "pm_5m_markets_5000.parquet"
    if not pm_path.exists():
        pm_path = CACHE_DIR / "pm_5m_markets_5000.csv"
    if not pm_path.exists():
        log.error("No cached data found. Run fetch_pm_cache.py first.")
        sys.exit(1)

    if pm_path.suffix == ".parquet":
        pm = pd.read_parquet(pm_path)
    else:
        pm = pd.read_csv(pm_path)

    bn_path = CACHE_DIR / "binance_1m_klines_35d.csv"
    if not bn_path.exists():
        bn_path = CACHE_DIR / "binance_1m_klines_35d.parquet"
    if not bn_path.exists():
        log.error("No Binance cache. Run backtest_quant_strategy.py --cache first.")
        sys.exit(1)

    if bn_path.suffix == ".parquet":
        klines = pd.read_parquet(bn_path)
    else:
        klines = pd.read_csv(bn_path)

    log.info("Loaded %d PM markets, %d Binance candles", len(pm), len(klines))
    return pm, klines


def run_backtest(
    bankroll: float = 10_000,
    max_bet: float = 500,
    kelly_alpha: float = 0.25,
    train_pct: float = 0.60,
):
    pm, klines = load_data()
    rng = np.random.default_rng(42)

    # Sort chronologically
    pm.sort_values("round_ts", inplace=True)
    pm.reset_index(drop=True, inplace=True)

    # Use 60% for "warmup" (model calibration), 40% for testing
    split = int(len(pm) * train_pct)
    test_markets = pm.iloc[split:].reset_index(drop=True)

    log.info("=" * 80)
    log.info("POLYMARKET ARBITRAGE STRATEGY BACKTEST")
    log.info("  Markets: %d total, %d test", len(pm), len(test_markets))
    log.info("  Bankroll: $%.0f | Max bet: $%.0f | Kelly: %.2f",
             bankroll, max_bet, kelly_alpha)
    log.info("  Date range: %s to %s",
             datetime.fromtimestamp(test_markets["round_ts"].min(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
             datetime.fromtimestamp(test_markets["round_ts"].max(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 80)

    # Track all trades by strategy
    all_trades: list[Trade] = []
    current_bankroll = bankroll
    peak_bankroll = bankroll

    for idx in range(len(test_markets)):
        row = test_markets.iloc[idx]
        round_ts = int(row["round_ts"])
        outcome = int(row["outcome"])
        volume = float(row["volume"])
        expiry_ts = round_ts + 300  # 5 min later

        # Compute signals
        p_theo = compute_p_theo(klines, round_ts, 5)
        btc_vol = compute_btc_volatility(klines, round_ts)

        # Simulate p_theo for a hypothetical 15m interval
        p_theo_15m = compute_p_theo(klines, round_ts, 15)

        # Generate order book at entry (market just opened, ~30s in)
        # Fair value at open: ~0.50 with slight adjustment from early CEX signal
        mask_early = (klines["ts"] >= round_ts) & (klines["ts"] < round_ts + 1)
        early = klines[mask_early]
        if len(early) > 0:
            first_ret = (early.iloc[0]["close"] - early.iloc[0]["open"]) / early.iloc[0]["open"]
            fair_at_entry = 0.50 + 0.30 * np.clip(first_ret * 100, -0.4, 0.4)
        else:
            fair_at_entry = 0.50

        fair_at_entry = max(0.10, min(0.90, fair_at_entry))

        book_entry = simulate_order_book(
            fair_value=fair_at_entry,
            volume=volume,
            btc_vol=btc_vol,
            time_to_expiry_frac=0.0,  # just opened
            rng=rng,
        )

        # Simulate exit book (near settlement)
        book_exit = simulate_exit_book(outcome, 0.05, volume, rng)

        # Oracle snipe signal (last-second CEX move)
        oracle_signal = oracle_snipe_signal(klines, round_ts, expiry_ts)

        # Drawdown circuit breaker
        dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd > 0.20:
            continue

        # ---- Run all strategies ----

        # 1. Core Buy-Exit
        t = strategy_core_buy_exit(book_entry, book_exit, outcome, volume,
                                    current_bankroll, max_bet, rng)
        if t:
            t.ts = round_ts
            all_trades.append(t)
            current_bankroll += t.pnl

        # 2. Complement Arb
        t = strategy_complement_arb(book_entry, volume, max_bet, rng)
        if t:
            t.ts = round_ts
            all_trades.append(t)
            current_bankroll += t.pnl

        # 3. Statistical Model Arb
        t = strategy_stat_model_arb(p_theo, book_entry, outcome,
                                     current_bankroll, max_bet, kelly_alpha)
        if t:
            t.ts = round_ts
            all_trades.append(t)
            current_bankroll += t.pnl

        # 4. Cross-Interval Arb
        t = strategy_cross_interval_arb(p_theo, p_theo_15m, book_entry,
                                          outcome, current_bankroll, max_bet)
        if t:
            t.ts = round_ts
            all_trades.append(t)
            current_bankroll += t.pnl

        # 5. Oracle Snipe
        t = strategy_oracle_snipe(oracle_signal, book_entry, outcome,
                                   current_bankroll, max_bet, rng)
        if t:
            t.ts = round_ts
            all_trades.append(t)
            current_bankroll += t.pnl

        peak_bankroll = max(peak_bankroll, current_bankroll)

        if (idx + 1) % 500 == 0:
            log.info("  Round %d/%d | Bankroll: $%.0f | Trades: %d",
                     idx + 1, len(test_markets), current_bankroll, len(all_trades))

    # ================================================================
    # RESULTS BY STRATEGY
    # ================================================================

    log.info("\n" + "=" * 80)
    log.info("RESULTS BY STRATEGY")
    log.info("=" * 80)

    strategies = {}
    for t in all_trades:
        if t.strategy not in strategies:
            strategies[t.strategy] = []
        strategies[t.strategy].append(t)

    summary = {}

    for name, trades in sorted(strategies.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.won)
        total_pnl = sum(t.pnl for t in trades)
        win_rate = wins / n if n > 0 else 0
        pnl_list = [t.pnl for t in trades]
        avg_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else 1
        total_fees = sum(t.fee_paid for t in trades)

        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_edge = np.mean([t.edge_bps for t in trades])
        avg_win = np.mean([t.pnl for t in trades if t.won]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl for t in trades if not t.won]) if (n - wins) > 0 else 0
        max_win = max(pnl_list)
        max_loss = min(pnl_list)

        # Sharpe (annualized assuming 288 rounds/day)
        sharpe = (avg_pnl / std_pnl) * np.sqrt(288 * 365) if std_pnl > 0 else 0

        maker_count = sum(1 for t in trades if t.is_maker)

        log.info("\n  ━━━ %s ━━━", name)
        log.info("  Trades:         %d (maker: %d, taker: %d)", n, maker_count, n - maker_count)
        log.info("  Win Rate:       %.1f%% (%d/%d)", win_rate * 100, wins, n)
        log.info("  Total PnL:      $%+.2f (%.1f%%)", total_pnl, total_pnl / bankroll * 100)
        log.info("  Avg PnL/Trade:  $%+.2f", avg_pnl)
        log.info("  Avg Win:        $%+.2f | Avg Loss: $%+.2f", avg_win, avg_loss)
        log.info("  Max Win:        $%+.2f | Max Loss: $%+.2f", max_win, max_loss)
        log.info("  Profit Factor:  %.2f", pf)
        log.info("  Sharpe:         %.1f", sharpe)
        log.info("  Total Fees:     $%.2f (net of rebates)", total_fees)
        log.info("  Avg Edge:       %.0f bps", avg_edge)

        summary[name] = {
            "trades": n, "wins": wins, "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2), "profit_factor": round(pf, 4),
            "sharpe": round(sharpe, 1), "avg_edge_bps": round(avg_edge, 1),
            "total_fees": round(total_fees, 2),
        }

    # ================================================================
    # AGGREGATE
    # ================================================================

    log.info("\n" + "=" * 80)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 80)

    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t.won)
    total_pnl = current_bankroll - bankroll
    max_dd = (peak_bankroll - min(
        bankroll + sum(t.pnl for t in all_trades[:i+1])
        for i in range(len(all_trades))
    )) / peak_bankroll if all_trades else 0

    log.info("  Total Trades:     %d", total_trades)
    log.info("  Total Win Rate:   %.1f%%", total_wins / total_trades * 100 if total_trades else 0)
    log.info("  Total PnL:        $%+.2f (%.1f%%)", total_pnl, total_pnl / bankroll * 100)
    log.info("  Final Bankroll:   $%.2f", current_bankroll)
    log.info("  Peak Bankroll:    $%.2f", peak_bankroll)
    log.info("  Max Drawdown:     %.1f%%", max_dd * 100)
    log.info("  Total Fees (net): $%.2f", sum(t.fee_paid for t in all_trades))

    # PnL by day
    daily_pnl = {}
    for t in all_trades:
        day = datetime.fromtimestamp(t.ts, tz=timezone.utc).strftime("%Y-%m-%d")
        daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl

    log.info("\n  --- Daily PnL ---")
    for day, dpnl in sorted(daily_pnl.items()):
        log.info("  %s: $%+.2f", day, dpnl)

    # Strategy comparison table
    log.info("\n  --- Strategy Comparison ---")
    log.info("  %-22s %6s %7s %10s %8s %7s", "Strategy", "Trades", "WinRate", "PnL", "PF", "Sharpe")
    log.info("  " + "-" * 66)
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["total_pnl"]):
        log.info("  %-22s %6d %6.1f%% %+10.2f %8.2f %7.1f",
                 name, s["trades"], s["win_rate"] * 100,
                 s["total_pnl"], s["profit_factor"], s["sharpe"])

    # Save results
    output = {
        "config": {
            "bankroll": bankroll, "max_bet": max_bet,
            "kelly_alpha": kelly_alpha, "n_test_markets": len(test_markets),
        },
        "strategies": summary,
        "aggregate": {
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "final_bankroll": round(current_bankroll, 2),
            "max_drawdown": round(max_dd, 4),
        },
        "daily_pnl": daily_pnl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path("/Users/eiji/polymarket-pnl/backtest_results_arb.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("\nResults saved to %s", out_path)

    log.info("\n" + "=" * 80)
    log.info("BACKTEST COMPLETE")
    log.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arb Strategy Backtest")
    parser.add_argument("--bankroll", type=float, default=10_000)
    parser.add_argument("--max-bet", type=float, default=500)
    parser.add_argument("--kelly-alpha", type=float, default=0.25)
    args = parser.parse_args()

    run_backtest(
        bankroll=args.bankroll,
        max_bet=args.max_bet,
        kelly_alpha=args.kelly_alpha,
    )


if __name__ == "__main__":
    main()
