"""
Market Model — fee engine, market metadata helpers, net-edge computation.
==========================================================================

Implements the Polymarket fee schedule from config/fees.yaml and provides
the core ``compute_net_edge`` function that every trading decision passes
through before quoting.

All prices/probabilities are on [0, 1].  Fees are expressed as fractions
(0.03 = 3 %).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_FEES_PATH = Path(__file__).resolve().parent.parent / "config" / "fees.yaml"


# ---------------------------------------------------------------------------
# Fee schedule (loaded once, cached)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeBucket:
    lo: float
    hi: float
    fee_pct: float


_fee_schedule: list[FeeBucket] | None = None
_maker_rebate: float = 0.0


def _load_fee_schedule() -> None:
    global _fee_schedule, _maker_rebate
    with open(_FEES_PATH) as f:
        cfg = yaml.safe_load(f)
    pm = cfg.get("polymarket", {})
    _fee_schedule = [
        FeeBucket(lo=b["price_range"][0], hi=b["price_range"][1], fee_pct=b["fee_pct"])
        for b in pm.get("taker_fee_schedule", [])
    ]
    _maker_rebate = float(pm.get("maker_rebate_pct", 0.0))
    log.info(
        "Fee schedule loaded: %d buckets, maker rebate=%.3f%%",
        len(_fee_schedule), _maker_rebate * 100,
    )


def get_taker_fee(price: float) -> float:
    """Return the taker fee fraction for a given YES price on [0, 1]."""
    if _fee_schedule is None:
        _load_fee_schedule()
    for b in _fee_schedule:  # type: ignore[union-attr]
        if b.lo <= price <= b.hi:
            return b.fee_pct
    return 0.02  # default fallback


def get_maker_rebate() -> float:
    """Return the maker rebate fraction (positive = you get paid)."""
    if _fee_schedule is None:
        _load_fee_schedule()
    return _maker_rebate


# ---------------------------------------------------------------------------
# Market metadata (lightweight in-memory cache)
# ---------------------------------------------------------------------------

@dataclass
class MarketMeta:
    """Metadata for one active Polymarket crypto binary market."""
    market_id: str
    condition_id: str
    question: str
    underlying: str          # BTC, ETH, SOL
    expiry_ts: datetime | None
    strike_price: float | None  # None = ATM
    strike_type: str         # ATM or custom
    fee_tier: float          # effective taker fee for this market
    maker_rebate: float
    status: str              # active / closed / settled

    @property
    def time_to_expiry_s(self) -> float:
        """Seconds until settlement. Negative if already expired."""
        if self.expiry_ts is None:
            return float("inf")
        return (self.expiry_ts - datetime.now(timezone.utc)).total_seconds()


# ---------------------------------------------------------------------------
# Net-edge computation
# ---------------------------------------------------------------------------

def compute_net_edge(
    p_theo: float,
    market_price: float,
    side: str,
    is_maker: bool = True,
) -> float:
    """
    Compute the expected value per dollar **after fees/rebates**.

    Parameters
    ----------
    p_theo : float
        Our theoretical probability that YES settles True.
    market_price : float
        Current YES price on the book (the level we'd trade at).
    side : str
        "YES" or "NO" — which side we're buying/providing.
    is_maker : bool
        If True, we receive the maker rebate; otherwise pay taker fee.

    Returns
    -------
    float
        Net edge as a fraction.  Positive → favourable trade.
        E.g. 0.02 means we expect +2 % return per dollar risked.
    """
    if side.upper() == "YES":
        # Buying YES at `market_price`.
        # If YES settles: payout = 1.  If NO settles: payout = 0.
        raw_ev = p_theo * 1.0 + (1 - p_theo) * 0.0 - market_price
    else:
        # Buying NO is equivalent to selling YES (payout = 1 - market_price).
        # Probability NO settles = 1 - p_theo.
        no_price = 1.0 - market_price
        raw_ev = (1 - p_theo) * 1.0 + p_theo * 0.0 - no_price

    # Fee / rebate adjustment (applied on entry cost)
    if is_maker:
        fee_adjustment = -get_maker_rebate()  # rebate is income → negative cost
    else:
        fee_adjustment = get_taker_fee(market_price)

    net_edge = raw_ev - fee_adjustment * market_price
    return net_edge


def net_edge_yes(p_theo: float, yes_ask: float, is_maker: bool = True) -> float:
    """Shortcut: net edge of buying YES at the current ask."""
    return compute_net_edge(p_theo, yes_ask, "YES", is_maker)


def net_edge_no(p_theo: float, yes_bid: float, is_maker: bool = True) -> float:
    """Shortcut: net edge of selling YES (= buying NO) at the current bid."""
    return compute_net_edge(p_theo, yes_bid, "NO", is_maker)
