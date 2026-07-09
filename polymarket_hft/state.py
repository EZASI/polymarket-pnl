"""
Market State — real-time state vector per active Polymarket market.
===================================================================

Holds CEX + Polymarket data for a single market and computes the
theoretical probability ``p_theo`` from CEX price action.

The p_theo model is *drift + vol*:
    p_theo = Φ(drift_z)
where
    drift_z = (cex_return_30s + ofi_adj) / (vol_1m * sqrt(T))
    T       = time-to-expiry in minutes

Φ is the standard-normal CDF.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from polymarket_hft.market_model import (
    MarketMeta,
    net_edge_yes,
    net_edge_no,
)


def _norm_cdf(x: float) -> float:
    """Fast standard-normal CDF (Abramowitz & Stegun approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class MarketState:
    """Real-time state vector for one active Polymarket market."""

    # ---- Identity ----
    market: MarketMeta

    # ---- CEX side (Binance Futures) ----
    cex_mid: float = 0.0           # latest midprice
    cex_return_30s: float = 0.0    # rolling 30-second log return
    cex_ofi: float = 0.0           # order-flow imbalance (rolling 30s, net buy volume)
    cex_realized_vol_1m: float = 0.0  # 1-minute realized vol (annualised σ)
    cex_last_trade_ts: float = 0.0 # unix timestamp of latest aggTrade

    # ---- Polymarket side ----
    pm_yes_bid: float = 0.0
    pm_yes_ask: float = 1.0
    pm_no_bid: float = 0.0
    pm_no_ask: float = 1.0
    pm_mid: float = 0.5
    pm_depth_bid: float = 0.0      # total top-5 bid depth ($)
    pm_depth_ask: float = 0.0      # total top-5 ask depth ($)
    pm_last_trade_price: float = 0.5
    pm_last_trade_ts: float = 0.0

    # ---- Derived (computed) ----
    p_theo: float = 0.5
    net_ev_yes: float = 0.0
    net_ev_no: float = 0.0
    lead_lag_ms: float = 500.0     # current estimated lag
    updated_at: float = field(default_factory=time.time)

    # ---- Configuration ----
    ofi_weight: float = 0.3        # how strongly OFI adjusts the drift signal

    # ------------------------------------------------------------------
    # p_theo computation
    # ------------------------------------------------------------------

    def recompute(self) -> None:
        """
        Recompute p_theo and net EVs from current CEX + PM state.

        Model:
            drift_z = (cex_return_30s + ofi_weight * normalised_ofi) / (vol * sqrt(T))
            p_theo  = Φ(drift_z)

        For an ATM BTC-5m market, the question is "will BTC be higher than
        current price at expiry?"  If drift_z > 0, we estimate YES is more
        likely; if drift_z < 0, NO is more likely.
        """
        tte_s = self.market.time_to_expiry_s
        if tte_s <= 0:
            # Already expired — no edge
            self.p_theo = self.pm_mid
            self.net_ev_yes = 0.0
            self.net_ev_no = 0.0
            return

        tte_min = max(tte_s / 60.0, 0.01)  # avoid division by zero

        vol = max(self.cex_realized_vol_1m, 1e-8)

        # Normalise OFI: sign only (direction), clamp to [-1, 1]
        ofi_norm = max(-1.0, min(1.0, self.cex_ofi / max(abs(self.cex_ofi), 1e-8)))

        drift = self.cex_return_30s + self.ofi_weight * ofi_norm * vol * 0.01
        drift_z = drift / (vol * math.sqrt(tte_min))

        self.p_theo = _norm_cdf(drift_z)

        # Clamp to [0.01, 0.99] to avoid extreme quotes
        self.p_theo = max(0.01, min(0.99, self.p_theo))

        # Net EV after fees (maker)
        if self.pm_yes_ask > 0:
            self.net_ev_yes = net_edge_yes(self.p_theo, self.pm_yes_ask, is_maker=True)
        if self.pm_yes_bid > 0:
            self.net_ev_no = net_edge_no(self.p_theo, self.pm_yes_bid, is_maker=True)

        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def time_to_expiry_s(self) -> float:
        return self.market.time_to_expiry_s

    @property
    def implied_prob(self) -> float:
        """Market-implied probability (YES mid)."""
        return self.pm_mid

    @property
    def edge_yes_bps(self) -> float:
        return self.net_ev_yes * 10_000

    @property
    def edge_no_bps(self) -> float:
        return self.net_ev_no * 10_000

    @property
    def best_side(self) -> str:
        """Which side has positive edge, if any."""
        if self.net_ev_yes > self.net_ev_no and self.net_ev_yes > 0:
            return "YES"
        if self.net_ev_no > self.net_ev_yes and self.net_ev_no > 0:
            return "NO"
        return "NONE"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API / Redis."""
        return {
            "market_id": self.market.market_id,
            "underlying": self.market.underlying,
            "question": self.market.question,
            "expiry": self.market.expiry_ts.isoformat() if self.market.expiry_ts else None,
            "tte_s": round(self.time_to_expiry_s, 1),
            "cex_mid": self.cex_mid,
            "cex_return_30s": round(self.cex_return_30s, 6),
            "cex_ofi": round(self.cex_ofi, 4),
            "cex_vol_1m": round(self.cex_realized_vol_1m, 6),
            "pm_yes_bid": self.pm_yes_bid,
            "pm_yes_ask": self.pm_yes_ask,
            "pm_mid": round(self.pm_mid, 4),
            "pm_depth_bid": round(self.pm_depth_bid, 2),
            "pm_depth_ask": round(self.pm_depth_ask, 2),
            "implied_prob": round(self.implied_prob, 4),
            "p_theo": round(self.p_theo, 4),
            "net_ev_yes": round(self.net_ev_yes, 6),
            "net_ev_no": round(self.net_ev_no, 6),
            "edge_yes_bps": round(self.edge_yes_bps, 1),
            "edge_no_bps": round(self.edge_no_bps, 1),
            "best_side": self.best_side,
            "lead_lag_ms": round(self.lead_lag_ms, 1),
            "status": self.market.status,
            "updated_at": self.updated_at,
        }
