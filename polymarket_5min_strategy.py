#!/usr/bin/env python3
"""
Polymarket 5-Minute Prediction Strategy Framework
==================================================
Built with LIVE market data from Polymarket/Crypto.com CLOB (Feb 13, 2026)

This framework implements 5 concrete strategies for 5-minute price prediction,
complete with signal generation, backtesting, and feasibility analysis.

Author: Quantitative Trading Research
"""

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# =============================================================================
# SECTION 1: REAL MARKET DATA (captured live Feb 13, 2026 02:36 UTC)
# =============================================================================

# BTCUSD 5-minute candles (most recent 50)
BTCUSD_5M_CANDLES = [
    {"open": 66548.00, "high": 66570.99, "low": 66489.01, "close": 66499.01, "volume": 7.0123, "volume_usd": 466311.0, "ts": "2026-02-13T02:35:00Z"},
    {"open": 66482.00, "high": 66588.98, "low": 66451.92, "close": 66540.02, "volume": 37.845, "volume_usd": 2518225.7, "ts": "2026-02-13T02:30:00Z"},
    {"open": 66483.30, "high": 66616.48, "low": 66463.39, "close": 66481.00, "volume": 44.167, "volume_usd": 2936283.6, "ts": "2026-02-13T02:25:00Z"},
    {"open": 66428.00, "high": 66488.06, "low": 66328.71, "close": 66478.34, "volume": 31.297, "volume_usd": 2080593.2, "ts": "2026-02-13T02:20:00Z"},
    {"open": 66511.98, "high": 66511.98, "low": 66408.01, "close": 66425.20, "volume": 40.590, "volume_usd": 2696217.5, "ts": "2026-02-13T02:15:00Z"},
    {"open": 66500.00, "high": 66522.68, "low": 66439.01, "close": 66511.98, "volume": 34.054, "volume_usd": 2264965.7, "ts": "2026-02-13T02:10:00Z"},
    {"open": 66437.67, "high": 66493.89, "low": 66371.00, "close": 66493.89, "volume": 34.094, "volume_usd": 2267068.6, "ts": "2026-02-13T02:05:00Z"},
    {"open": 66401.00, "high": 66452.83, "low": 66331.01, "close": 66444.98, "volume": 48.509, "volume_usd": 3223173.6, "ts": "2026-02-13T02:00:00Z"},
    {"open": 66364.00, "high": 66399.00, "low": 66272.26, "close": 66399.00, "volume": 35.256, "volume_usd": 2340977.8, "ts": "2026-02-13T01:55:00Z"},
    {"open": 66498.99, "high": 66521.34, "low": 66339.45, "close": 66366.00, "volume": 38.974, "volume_usd": 2586574.4, "ts": "2026-02-13T01:50:00Z"},
    {"open": 66364.01, "high": 66568.98, "low": 66364.01, "close": 66498.50, "volume": 45.115, "volume_usd": 3000107.1, "ts": "2026-02-13T01:45:00Z"},
    {"open": 66402.62, "high": 66450.00, "low": 66365.00, "close": 66365.00, "volume": 38.493, "volume_usd": 2554566.7, "ts": "2026-02-13T01:40:00Z"},
    {"open": 66311.00, "high": 66425.99, "low": 66265.56, "close": 66393.86, "volume": 37.860, "volume_usd": 2513669.5, "ts": "2026-02-13T01:35:00Z"},
    {"open": 66168.01, "high": 66335.64, "low": 66168.01, "close": 66310.99, "volume": 35.201, "volume_usd": 2334241.0, "ts": "2026-02-13T01:30:00Z"},
    {"open": 66068.00, "high": 66223.42, "low": 66024.67, "close": 66169.66, "volume": 49.006, "volume_usd": 3242713.7, "ts": "2026-02-13T01:25:00Z"},
    {"open": 66166.39, "high": 66205.99, "low": 66069.00, "close": 66069.00, "volume": 41.425, "volume_usd": 2736918.9, "ts": "2026-02-13T01:20:00Z"},
    {"open": 66198.99, "high": 66241.99, "low": 66148.55, "close": 66163.01, "volume": 27.314, "volume_usd": 1807183.7, "ts": "2026-02-13T01:15:00Z"},
    {"open": 66135.60, "high": 66209.60, "low": 66098.89, "close": 66196.53, "volume": 30.736, "volume_usd": 2034586.1, "ts": "2026-02-13T01:10:00Z"},
    {"open": 65988.01, "high": 66226.18, "low": 65976.01, "close": 66135.60, "volume": 47.619, "volume_usd": 3149293.9, "ts": "2026-02-13T01:05:00Z"},
    {"open": 65849.00, "high": 66022.37, "low": 65849.00, "close": 65989.30, "volume": 58.106, "volume_usd": 3834347.9, "ts": "2026-02-13T01:00:00Z"},
    {"open": 65971.26, "high": 66022.98, "low": 65846.00, "close": 65848.00, "volume": 45.777, "volume_usd": 3014300.8, "ts": "2026-02-13T00:55:00Z"},
    {"open": 65923.40, "high": 66000.20, "low": 65823.01, "close": 65971.27, "volume": 35.019, "volume_usd": 2310228.8, "ts": "2026-02-13T00:50:00Z"},
    {"open": 65901.32, "high": 65969.00, "low": 65821.09, "close": 65923.32, "volume": 31.790, "volume_usd": 2095710.3, "ts": "2026-02-13T00:45:00Z"},
    {"open": 66034.26, "high": 66092.97, "low": 65898.17, "close": 65901.31, "volume": 26.950, "volume_usd": 1776069.3, "ts": "2026-02-13T00:40:00Z"},
    {"open": 65923.99, "high": 66057.00, "low": 65878.05, "close": 66035.00, "volume": 21.380, "volume_usd": 1411798.6, "ts": "2026-02-13T00:35:00Z"},
    {"open": 65983.72, "high": 66062.00, "low": 65851.01, "close": 65922.33, "volume": 50.565, "volume_usd": 3333394.9, "ts": "2026-02-13T00:30:00Z"},
    {"open": 66093.69, "high": 66098.08, "low": 65957.01, "close": 65983.71, "volume": 25.601, "volume_usd": 1689218.6, "ts": "2026-02-13T00:25:00Z"},
    {"open": 66128.02, "high": 66178.17, "low": 66039.47, "close": 66092.99, "volume": 44.762, "volume_usd": 2958474.9, "ts": "2026-02-13T00:20:00Z"},
    {"open": 66332.53, "high": 66335.94, "low": 66121.13, "close": 66125.35, "volume": 47.833, "volume_usd": 3162954.0, "ts": "2026-02-13T00:15:00Z"},
    {"open": 66382.05, "high": 66418.30, "low": 66314.01, "close": 66332.52, "volume": 47.546, "volume_usd": 3153874.5, "ts": "2026-02-13T00:10:00Z"},
    {"open": 66398.99, "high": 66478.13, "low": 66345.52, "close": 66388.05, "volume": 69.151, "volume_usd": 4590792.7, "ts": "2026-02-13T00:05:00Z"},
    {"open": 66220.61, "high": 66473.70, "low": 66099.99, "close": 66400.19, "volume": 74.837, "volume_usd": 4969175.7, "ts": "2026-02-13T00:00:00Z"},
    {"open": 66194.99, "high": 66260.52, "low": 66177.10, "close": 66220.61, "volume": 21.338, "volume_usd": 1413034.6, "ts": "2026-02-12T23:55:00Z"},
    {"open": 66180.99, "high": 66205.88, "low": 66127.65, "close": 66192.11, "volume": 25.323, "volume_usd": 1676158.3, "ts": "2026-02-12T23:50:00Z"},
    {"open": 66184.01, "high": 66220.79, "low": 66105.65, "close": 66175.97, "volume": 33.411, "volume_usd": 2210984.2, "ts": "2026-02-12T23:45:00Z"},
    {"open": 66181.00, "high": 66241.00, "low": 66126.57, "close": 66188.18, "volume": 27.343, "volume_usd": 1809809.9, "ts": "2026-02-12T23:40:00Z"},
    {"open": 66329.00, "high": 66336.99, "low": 66170.01, "close": 66175.48, "volume": 46.348, "volume_usd": 3067126.3, "ts": "2026-02-12T23:35:00Z"},
    {"open": 66352.80, "high": 66433.01, "low": 66246.02, "close": 66333.06, "volume": 56.263, "volume_usd": 3732081.7, "ts": "2026-02-12T23:30:00Z"},
    {"open": 66385.32, "high": 66436.58, "low": 66310.00, "close": 66344.58, "volume": 27.001, "volume_usd": 1791338.2, "ts": "2026-02-12T23:25:00Z"},
    {"open": 66301.01, "high": 66404.79, "low": 66257.03, "close": 66383.40, "volume": 42.580, "volume_usd": 2826609.2, "ts": "2026-02-12T23:20:00Z"},
    {"open": 66256.00, "high": 66314.00, "low": 66202.00, "close": 66301.01, "volume": 46.383, "volume_usd": 3075240.4, "ts": "2026-02-12T23:15:00Z"},
    {"open": 66384.01, "high": 66395.00, "low": 66196.00, "close": 66256.19, "volume": 35.648, "volume_usd": 2361901.3, "ts": "2026-02-12T23:10:00Z"},
    {"open": 66313.00, "high": 66427.59, "low": 66263.01, "close": 66384.01, "volume": 57.980, "volume_usd": 3848931.6, "ts": "2026-02-12T23:05:00Z"},
    {"open": 66235.00, "high": 66397.72, "low": 66132.22, "close": 66310.88, "volume": 63.161, "volume_usd": 4188275.4, "ts": "2026-02-12T23:00:00Z"},
]

# BTCUSD order book snapshot (50 levels each side)
BTCUSD_BOOK = {
    "asks": [
        (66494.31, 0.00156), (66498.00, 0.08510), (66498.40, 0.03759),
        (66499.00, 0.00003), (66499.02, 0.08060), (66500.00, 0.08510),
        (66500.34, 0.08605), (66501.70, 0.15000), (66502.16, 0.02886),
        (66502.36, 0.05000), (66503.49, 0.09720), (66503.69, 0.03313),
        (66503.77, 0.00601), (66503.82, 0.01500), (66504.48, 0.03290),
        (66504.68, 0.15029), (66504.91, 0.06247), (66505.53, 0.01202),
        (66505.56, 0.01503), (66505.65, 0.05116), (66506.95, 0.43053),
        (66507.00, 0.00003), (66508.05, 0.09022), (66508.99, 1.08281),
        (66509.00, 0.05116), (66510.08, 0.07155), (66511.32, 0.03227),
        (66511.33, 0.03290), (66511.55, 0.07520), (66513.51, 0.10951),
        (66513.56, 0.05116), (66514.00, 0.00003), (66515.54, 0.22558),
        (66515.55, 0.05639), (66515.56, 0.03000), (66516.25, 0.00371),
        (66516.76, 0.00381), (66516.98, 1.08281), (66517.59, 0.03000),
        (66517.98, 0.03290), (66518.63, 0.05116), (66518.93, 0.03000),
        (66519.01, 0.32385), (66522.88, 0.05116), (66523.00, 0.03000),
        (66523.37, 0.11155), (66523.61, 0.03678), (66524.58, 1.08224),
        (66526.24, 0.05116), (66527.19, 0.03000),
    ],
    "bids": [
        (66494.30, 0.81090), (66494.10, 0.08419), (66494.00, 0.08510),
        (66493.64, 0.00601), (66493.51, 0.07854), (66493.49, 0.03000),
        (66493.00, 0.08510), (66492.57, 0.03290), (66492.00, 0.08510),
        (66491.62, 0.01202), (66490.98, 0.00149), (66490.33, 0.03000),
        (66490.03, 0.03292), (66490.02, 0.52477), (66490.01, 1.08281),
        (66490.00, 0.08510), (66489.00, 0.15273), (66488.00, 0.08510),
        (66487.70, 0.03760), (66487.68, 0.05116), (66487.67, 0.01500),
        (66487.00, 0.08510), (66486.70, 0.03760), (66484.87, 0.07029),
        (66484.35, 0.05116), (66483.69, 0.08492), (66483.68, 0.08254),
        (66483.21, 0.36982), (66483.20, 1.08281), (66483.00, 0.00003),
        (66482.36, 0.05639), (66482.35, 0.03290), (66482.09, 0.00760),
        (66481.20, 0.01503), (66479.85, 0.07157), (66479.60, 0.00090),
        (66479.59, 0.05116), (66479.58, 0.15103), (66479.49, 0.08872),
        (66476.86, 0.00379), (66476.51, 0.07520), (66476.00, 0.00003),
        (66475.52, 0.37019), (66475.51, 1.13397), (66475.27, 0.03290),
        (66471.00, 1.50000), (66470.84, 0.37596), (66469.38, 0.00383),
        (66468.16, 0.05116), (66468.06, 0.03290),
    ],
}


# =============================================================================
# SECTION 2: MICROSTRUCTURE ANALYSIS
# =============================================================================

@dataclass
class MicrostructureReport:
    """Quantitative analysis of order book microstructure."""
    spread_bps: float
    spread_usd: float
    bid_depth_10: float  # BTC within 10 levels
    ask_depth_10: float
    bid_depth_usd_10: float
    ask_depth_usd_10: float
    book_imbalance: float  # (bid_qty - ask_qty) / (bid_qty + ask_qty)
    weighted_mid: float
    vwap_bid_10: float
    vwap_ask_10: float
    large_orders: list  # orders > 0.5 BTC


def analyze_order_book(book: dict, mid_price: float) -> MicrostructureReport:
    """Deep analysis of order book microstructure from LIVE data."""
    asks = book["asks"]
    bids = book["bids"]

    best_ask = asks[0][0]
    best_bid = bids[0][0]
    spread_usd = best_ask - best_bid
    spread_bps = (spread_usd / mid_price) * 10000

    # Depth within top 10 levels
    bid_depth_10 = sum(q for _, q in bids[:10])
    ask_depth_10 = sum(q for _, q in asks[:10])
    bid_depth_usd_10 = sum(p * q for p, q in bids[:10])
    ask_depth_usd_10 = sum(p * q for p, q in asks[:10])

    # Book imbalance (key predictive signal)
    total_bid = sum(q for _, q in bids)
    total_ask = sum(q for _, q in asks)
    imbalance = (total_bid - total_ask) / (total_bid + total_ask)

    # Volume-weighted mid price
    bid_vwap_num = sum(p * q for p, q in bids[:10])
    bid_vwap_den = sum(q for _, q in bids[:10])
    ask_vwap_num = sum(p * q for p, q in asks[:10])
    ask_vwap_den = sum(q for _, q in asks[:10])
    vwap_bid = bid_vwap_num / bid_vwap_den if bid_vwap_den > 0 else best_bid
    vwap_ask = ask_vwap_num / ask_vwap_den if ask_vwap_den > 0 else best_ask
    weighted_mid = (vwap_bid * ask_depth_10 + vwap_ask * bid_depth_10) / (bid_depth_10 + ask_depth_10)

    # Large resting orders (iceberg detection)
    large_orders = [(p, q, "bid" if (p, q) in bids else "ask")
                    for p, q in bids + asks if q > 0.5]

    return MicrostructureReport(
        spread_bps=spread_bps,
        spread_usd=spread_usd,
        bid_depth_10=bid_depth_10,
        ask_depth_10=ask_depth_10,
        bid_depth_usd_10=bid_depth_usd_10,
        ask_depth_usd_10=ask_depth_usd_10,
        book_imbalance=imbalance,
        weighted_mid=weighted_mid,
        vwap_bid_10=vwap_bid,
        vwap_ask_10=vwap_ask,
        large_orders=large_orders,
    )


# =============================================================================
# SECTION 3: SIGNAL GENERATORS
# =============================================================================

def compute_returns(candles: list[dict]) -> list[float]:
    """5-min log returns from candle data (chronological order)."""
    # Candles come newest-first, reverse for chronological
    c = list(reversed(candles))
    returns = []
    for i in range(1, len(c)):
        r = math.log(c[i]["close"] / c[i - 1]["close"])
        returns.append(r)
    return returns


def compute_volatility(returns: list[float], window: int = 12) -> list[float]:
    """Rolling realized volatility (annualized from 5-min bars)."""
    # 12 bars = 1 hour, 288 bars/day, 365 days
    ann_factor = math.sqrt(288 * 365)
    vols = []
    for i in range(window, len(returns)):
        window_rets = returns[i - window:i]
        vol = statistics.stdev(window_rets) * ann_factor
        vols.append(vol)
    return vols


def compute_rsi(candles: list[dict], period: int = 14) -> list[float]:
    """RSI adapted for 5-minute prediction market bars."""
    c = list(reversed(candles))
    rsi_values = []
    gains = []
    losses = []

    for i in range(1, len(c)):
        change = c[i]["close"] - c[i - 1]["close"]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    for i in range(period, len(gains)):
        avg_gain = sum(gains[i - period:i]) / period
        avg_loss = sum(losses[i - period:i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - 100 / (1 + rs))

    return rsi_values


def compute_order_flow_imbalance(trades: list[dict]) -> dict:
    """
    Analyze trade-level data for order flow imbalance.
    Key signal: ratio of buy vs sell aggressor volume.
    """
    buy_vol = 0.0
    sell_vol = 0.0
    buy_count = 0
    sell_count = 0
    buy_sizes = []
    sell_sizes = []

    for t in trades:
        qty = float(t["qty"])
        if t["side"] == "buy":
            buy_vol += qty
            buy_count += 1
            buy_sizes.append(qty)
        else:
            sell_vol += qty
            sell_count += 1
            sell_sizes.append(qty)

    total_vol = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0

    return {
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "net_imbalance": imbalance,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "avg_buy_size": statistics.mean(buy_sizes) if buy_sizes else 0,
        "avg_sell_size": statistics.mean(sell_sizes) if sell_sizes else 0,
        "large_buy_count": sum(1 for s in buy_sizes if s > 0.05),
        "large_sell_count": sum(1 for s in sell_sizes if s > 0.05),
    }


def compute_volume_profile(candles: list[dict], lookback: int = 12) -> dict:
    """Volume profile analysis for detecting anomalous activity."""
    c = list(reversed(candles))
    if len(c) < lookback + 1:
        return {"volume_zscore": 0, "relative_volume": 1.0}

    recent_vols = [bar["volume_usd"] for bar in c[-lookback - 1:-1]]
    current_vol = c[-1]["volume_usd"]

    mean_vol = statistics.mean(recent_vols)
    std_vol = statistics.stdev(recent_vols) if len(recent_vols) > 1 else 1

    return {
        "volume_zscore": (current_vol - mean_vol) / std_vol if std_vol > 0 else 0,
        "relative_volume": current_vol / mean_vol if mean_vol > 0 else 1.0,
        "mean_volume_usd": mean_vol,
        "current_volume_usd": current_vol,
    }


# =============================================================================
# SECTION 4: FIVE CONCRETE STRATEGIES
# =============================================================================

@dataclass
class Signal:
    """Trading signal output."""
    direction: str  # "long", "short", "flat"
    confidence: float  # 0.0 to 1.0
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str


class Strategy1_OrderFlowMomentum:
    """
    STRATEGY 1: Order Flow Momentum
    ================================
    Hypothesis: When aggressive buyers/sellers dominate trade flow over a
    short window, price continues in that direction for the next 5 minutes.

    Entry: OFI (Order Flow Imbalance) > +0.3 => long, < -0.3 => short
    Exit: 5-minute time stop or TP/SL hit
    Stop: 0.05% adverse move
    Target: 0.08% favorable move (1.6:1 R:R)
    """

    def generate_signal(self, trades: list[dict], current_price: float) -> Signal:
        ofi = compute_order_flow_imbalance(trades)
        imbalance = ofi["net_imbalance"]

        # Require both volume imbalance AND large order presence
        large_buy_ratio = ofi["large_buy_count"] / max(ofi["buy_count"], 1)
        large_sell_ratio = ofi["large_sell_count"] / max(ofi["sell_count"], 1)

        if imbalance > 0.3 and large_buy_ratio > 0.15:
            return Signal(
                direction="long",
                confidence=min(abs(imbalance), 1.0),
                strategy="OrderFlowMomentum",
                entry_price=current_price,
                stop_loss=current_price * 0.9995,  # 5bps stop
                take_profit=current_price * 1.0008,  # 8bps target
                reasoning=f"OFI={imbalance:.3f}, large_buy_ratio={large_buy_ratio:.2f}, "
                          f"buy_vol={ofi['buy_volume']:.4f} vs sell_vol={ofi['sell_volume']:.4f}",
            )
        elif imbalance < -0.3 and large_sell_ratio > 0.15:
            return Signal(
                direction="short",
                confidence=min(abs(imbalance), 1.0),
                strategy="OrderFlowMomentum",
                entry_price=current_price,
                stop_loss=current_price * 1.0005,
                take_profit=current_price * 0.9992,
                reasoning=f"OFI={imbalance:.3f}, large_sell_ratio={large_sell_ratio:.2f}",
            )
        else:
            return Signal(
                direction="flat",
                confidence=0.0,
                strategy="OrderFlowMomentum",
                entry_price=current_price,
                stop_loss=0, take_profit=0,
                reasoning=f"OFI={imbalance:.3f} below threshold",
            )


class Strategy2_BookImbalanceReversion:
    """
    STRATEGY 2: Order Book Imbalance Mean Reversion
    =================================================
    Hypothesis: Extreme order book imbalances (bid-heavy or ask-heavy)
    tend to mean-revert as the heavy side gets filled. This is a
    contrarian signal.

    Entry: Book imbalance > +0.25 => SHORT (excess bids = about to sell into them)
           Book imbalance < -0.25 => LONG (excess asks = about to buy into them)
    Exit: 5-minute time stop
    Stop: 0.06% (wider for mean reversion)
    Target: 0.10%
    """

    def generate_signal(self, book: dict, current_price: float) -> Signal:
        mid = current_price
        report = analyze_order_book(book, mid)

        if report.book_imbalance > 0.25:
            # Excess bids => contrarian short
            return Signal(
                direction="short",
                confidence=min(abs(report.book_imbalance) * 2, 1.0),
                strategy="BookImbalanceReversion",
                entry_price=current_price,
                stop_loss=current_price * 1.0006,
                take_profit=current_price * 0.9990,
                reasoning=f"Book imbalance={report.book_imbalance:.3f} (bid-heavy), "
                          f"bid_depth={report.bid_depth_usd_10:.0f} vs ask_depth={report.ask_depth_usd_10:.0f}",
            )
        elif report.book_imbalance < -0.25:
            # Excess asks => contrarian long
            return Signal(
                direction="long",
                confidence=min(abs(report.book_imbalance) * 2, 1.0),
                strategy="BookImbalanceReversion",
                entry_price=current_price,
                stop_loss=current_price * 0.9994,
                take_profit=current_price * 1.0010,
                reasoning=f"Book imbalance={report.book_imbalance:.3f} (ask-heavy)",
            )
        else:
            return Signal(
                direction="flat", confidence=0.0,
                strategy="BookImbalanceReversion",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"Imbalance={report.book_imbalance:.3f} within neutral zone",
            )


class Strategy3_VolumeSpikeBreakout:
    """
    STRATEGY 3: Volume Spike Breakout
    ===================================
    Hypothesis: Anomalous volume (>2 std devs above mean) in a 5-min bar
    signals informed trading. Trade in the direction of the spike bar's close.

    Entry: Volume z-score > 2.0 AND candle body > 0.05%
    Exit: 5-minute time stop
    Stop: Low/high of the spike candle
    Target: 1x the spike candle range
    """

    def generate_signal(self, candles: list[dict], current_price: float) -> Signal:
        vol_profile = compute_volume_profile(candles)
        z = vol_profile["volume_zscore"]

        # Current candle
        curr = candles[0]  # Most recent (newest first)
        body = curr["close"] - curr["open"]
        body_pct = body / curr["open"]
        candle_range = curr["high"] - curr["low"]

        if z > 2.0 and abs(body_pct) > 0.0005:
            direction = "long" if body > 0 else "short"
            if direction == "long":
                return Signal(
                    direction="long",
                    confidence=min(z / 4.0, 1.0),
                    strategy="VolumeSpikeBreakout",
                    entry_price=current_price,
                    stop_loss=curr["low"],
                    take_profit=current_price + candle_range,
                    reasoning=f"Volume z-score={z:.2f} ({vol_profile['relative_volume']:.1f}x avg), "
                              f"bullish body={body_pct*100:.3f}%, range=${candle_range:.2f}",
                )
            else:
                return Signal(
                    direction="short",
                    confidence=min(z / 4.0, 1.0),
                    strategy="VolumeSpikeBreakout",
                    entry_price=current_price,
                    stop_loss=curr["high"],
                    take_profit=current_price - candle_range,
                    reasoning=f"Volume z-score={z:.2f}, bearish body={body_pct*100:.3f}%",
                )
        else:
            return Signal(
                direction="flat", confidence=0.0,
                strategy="VolumeSpikeBreakout",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"Vol z-score={z:.2f}, body_pct={body_pct*100:.3f}% — no spike detected",
            )


class Strategy4_RSIMeanReversion:
    """
    STRATEGY 4: 5-Min RSI Mean Reversion
    ======================================
    Hypothesis: On a 5-minute timeframe, extreme RSI readings (>80 or <20)
    in liquid markets tend to snap back. This works especially well when
    combined with a volatility filter (only trade in normal vol regimes).

    Entry: RSI(14) > 80 => short, RSI(14) < 20 => long
    Filter: Only if realized vol is < 150% annualized (avoid trending days)
    Stop: 0.07%
    Target: 0.12%
    """

    def generate_signal(self, candles: list[dict], current_price: float) -> Signal:
        rsi_values = compute_rsi(candles, period=14)
        if not rsi_values:
            return Signal("flat", 0.0, "RSIMeanReversion", current_price, 0, 0, "Insufficient data")

        current_rsi = rsi_values[-1]
        returns = compute_returns(candles)
        vols = compute_volatility(returns, window=12)
        current_vol = vols[-1] if vols else 0

        # Volatility filter: only trade in calm markets
        if current_vol > 1.50:  # 150% annualized vol = trending/volatile
            return Signal(
                direction="flat", confidence=0.0,
                strategy="RSIMeanReversion",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"RSI={current_rsi:.1f} but vol={current_vol*100:.0f}% too high for MR",
            )

        if current_rsi > 80:
            return Signal(
                direction="short",
                confidence=(current_rsi - 80) / 20,
                strategy="RSIMeanReversion",
                entry_price=current_price,
                stop_loss=current_price * 1.0007,
                take_profit=current_price * 0.9988,
                reasoning=f"RSI={current_rsi:.1f} (overbought), vol={current_vol*100:.0f}%",
            )
        elif current_rsi < 20:
            return Signal(
                direction="long",
                confidence=(20 - current_rsi) / 20,
                strategy="RSIMeanReversion",
                entry_price=current_price,
                stop_loss=current_price * 0.9993,
                take_profit=current_price * 1.0012,
                reasoning=f"RSI={current_rsi:.1f} (oversold), vol={current_vol*100:.0f}%",
            )
        else:
            return Signal(
                direction="flat", confidence=0.0,
                strategy="RSIMeanReversion",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"RSI={current_rsi:.1f} in neutral zone, vol={current_vol*100:.0f}%",
            )


class Strategy5_SpreadRegimeDetection:
    """
    STRATEGY 5: Spread Regime + Weighted Mid Deviation
    ====================================================
    Hypothesis: When the volume-weighted midpoint significantly deviates
    from the simple midpoint, it predicts short-term price direction.
    The VWAP mid is "smarter" because it accounts for where real size sits.

    Entry: (weighted_mid - simple_mid) / simple_mid > 0.5bps => long
           (weighted_mid - simple_mid) / simple_mid < -0.5bps => short
    Filter: Spread must be < 2bps (liquid conditions only)
    Stop: 0.04%
    Target: 0.06%
    """

    def generate_signal(self, book: dict, current_price: float) -> Signal:
        report = analyze_order_book(book, current_price)
        simple_mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
        deviation_bps = ((report.weighted_mid - simple_mid) / simple_mid) * 10000

        if report.spread_bps > 2.0:
            return Signal(
                direction="flat", confidence=0.0,
                strategy="SpreadRegimeDetection",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"Spread={report.spread_bps:.2f}bps too wide, deviation={deviation_bps:.2f}bps",
            )

        if deviation_bps > 0.5:
            return Signal(
                direction="long",
                confidence=min(abs(deviation_bps) / 3.0, 1.0),
                strategy="SpreadRegimeDetection",
                entry_price=current_price,
                stop_loss=current_price * 0.9996,
                take_profit=current_price * 1.0006,
                reasoning=f"VWAP mid > simple mid by {deviation_bps:.2f}bps, "
                          f"spread={report.spread_bps:.2f}bps",
            )
        elif deviation_bps < -0.5:
            return Signal(
                direction="short",
                confidence=min(abs(deviation_bps) / 3.0, 1.0),
                strategy="SpreadRegimeDetection",
                entry_price=current_price,
                stop_loss=current_price * 1.0004,
                take_profit=current_price * 0.9994,
                reasoning=f"VWAP mid < simple mid by {deviation_bps:.2f}bps, "
                          f"spread={report.spread_bps:.2f}bps",
            )
        else:
            return Signal(
                direction="flat", confidence=0.0,
                strategy="SpreadRegimeDetection",
                entry_price=current_price, stop_loss=0, take_profit=0,
                reasoning=f"Deviation={deviation_bps:.2f}bps within neutral, "
                          f"spread={report.spread_bps:.2f}bps",
            )


# =============================================================================
# SECTION 5: BACKTESTING FRAMEWORK
# =============================================================================

@dataclass
class TradeResult:
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_bps: float
    pnl_usd: float
    holding_time_bars: int
    win: bool
    timestamp: str


@dataclass
class BacktestReport:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_bps: float
    avg_pnl_bps: float
    max_drawdown_bps: float
    sharpe_ratio: float
    profit_factor: float
    avg_win_bps: float
    avg_loss_bps: float
    trades: list[TradeResult] = field(default_factory=list)


def backtest_momentum_on_candles(candles: list[dict], lookback: int = 3) -> BacktestReport:
    """
    Backtest Strategy 1 (momentum proxy) on 5-min candle data.

    Logic: If last `lookback` candles were net positive, go long next bar.
    If net negative, go short. Measure result on next bar's close.
    """
    c = list(reversed(candles))  # Chronological
    trades = []
    pnl_series = []

    for i in range(lookback, len(c) - 1):
        # Signal: sum of last N returns
        past_returns = sum(
            c[j]["close"] - c[j - 1]["close"] for j in range(i - lookback + 1, i + 1)
        )
        past_return_pct = past_returns / c[i - lookback]["close"]

        # Only trade if momentum signal is strong enough
        if abs(past_return_pct) < 0.0003:  # 3bps minimum
            continue

        direction = "long" if past_return_pct > 0 else "short"
        entry = c[i]["close"]
        exit_p = c[i + 1]["close"]

        if direction == "long":
            pnl_bps = ((exit_p - entry) / entry) * 10000
        else:
            pnl_bps = ((entry - exit_p) / entry) * 10000

        trade = TradeResult(
            strategy="MomentumProxy",
            direction=direction,
            entry_price=entry,
            exit_price=exit_p,
            pnl_bps=pnl_bps,
            pnl_usd=pnl_bps * entry / 10000,  # per 1 BTC position
            holding_time_bars=1,
            win=pnl_bps > 0,
            timestamp=c[i + 1].get("ts", ""),
        )
        trades.append(trade)
        pnl_series.append(pnl_bps)

    return _compile_backtest_report(trades, pnl_series)


def backtest_mean_reversion_on_candles(candles: list[dict], lookback: int = 3) -> BacktestReport:
    """
    Backtest Strategy 2/4 (mean reversion proxy) on 5-min candle data.

    Logic: If last `lookback` candles moved > X bps in one direction,
    bet on a reversal.
    """
    c = list(reversed(candles))
    trades = []
    pnl_series = []

    for i in range(lookback, len(c) - 1):
        past_returns = sum(
            c[j]["close"] - c[j - 1]["close"] for j in range(i - lookback + 1, i + 1)
        )
        past_return_pct = past_returns / c[i - lookback]["close"]

        # Only trade on extreme moves (>10bps over lookback)
        if abs(past_return_pct) < 0.0010:
            continue

        # CONTRARIAN: go opposite
        direction = "short" if past_return_pct > 0 else "long"
        entry = c[i]["close"]
        exit_p = c[i + 1]["close"]

        if direction == "long":
            pnl_bps = ((exit_p - entry) / entry) * 10000
        else:
            pnl_bps = ((entry - exit_p) / entry) * 10000

        trade = TradeResult(
            strategy="MeanReversionProxy",
            direction=direction,
            entry_price=entry,
            exit_price=exit_p,
            pnl_bps=pnl_bps,
            pnl_usd=pnl_bps * entry / 10000,
            holding_time_bars=1,
            win=pnl_bps > 0,
            timestamp=c[i + 1].get("ts", ""),
        )
        trades.append(trade)
        pnl_series.append(pnl_bps)

    return _compile_backtest_report(trades, pnl_series)


def backtest_volume_breakout_on_candles(candles: list[dict]) -> BacktestReport:
    """
    Backtest Strategy 3 (volume spike breakout) on 5-min candle data.
    """
    c = list(reversed(candles))
    trades = []
    pnl_series = []

    for i in range(13, len(c) - 1):  # Need 12 bars lookback for vol stats
        recent_vols = [bar["volume_usd"] for bar in c[i - 12:i]]
        current_vol = c[i]["volume_usd"]
        mean_vol = statistics.mean(recent_vols)
        std_vol = statistics.stdev(recent_vols) if len(recent_vols) > 1 else 1
        z_score = (current_vol - mean_vol) / std_vol if std_vol > 0 else 0

        body = c[i]["close"] - c[i]["open"]
        body_pct = body / c[i]["open"]

        if z_score > 1.5 and abs(body_pct) > 0.0003:
            direction = "long" if body > 0 else "short"
            entry = c[i]["close"]
            exit_p = c[i + 1]["close"]

            if direction == "long":
                pnl_bps = ((exit_p - entry) / entry) * 10000
            else:
                pnl_bps = ((entry - exit_p) / entry) * 10000

            trade = TradeResult(
                strategy="VolumeBreakout",
                direction=direction,
                entry_price=entry,
                exit_price=exit_p,
                pnl_bps=pnl_bps,
                pnl_usd=pnl_bps * entry / 10000,
                holding_time_bars=1,
                win=pnl_bps > 0,
                timestamp=c[i + 1].get("ts", ""),
            )
            trades.append(trade)
            pnl_series.append(pnl_bps)

    return _compile_backtest_report(trades, pnl_series)


def _compile_backtest_report(trades: list[TradeResult], pnl_series: list[float]) -> BacktestReport:
    """Compile trade results into a performance report."""
    if not trades:
        return BacktestReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    wins = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]

    total_pnl = sum(pnl_series)
    avg_pnl = total_pnl / len(pnl_series)

    # Max drawdown
    cumulative = []
    running = 0
    peak = 0
    max_dd = 0
    for p in pnl_series:
        running += p
        cumulative.append(running)
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)

    # Sharpe ratio (per-trade, annualized assuming 288 trades/day)
    if len(pnl_series) > 1:
        std_pnl = statistics.stdev(pnl_series)
        sharpe = (avg_pnl / std_pnl) * math.sqrt(288 * 365) if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Profit factor
    gross_profit = sum(p for p in pnl_series if p > 0)
    gross_loss = abs(sum(p for p in pnl_series if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = statistics.mean([t.pnl_bps for t in wins]) if wins else 0
    avg_loss = statistics.mean([t.pnl_bps for t in losses]) if losses else 0

    return BacktestReport(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades),
        total_pnl_bps=total_pnl,
        avg_pnl_bps=avg_pnl,
        max_drawdown_bps=max_dd,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        avg_win_bps=avg_win,
        avg_loss_bps=avg_loss,
        trades=trades,
    )


# =============================================================================
# SECTION 6: FEASIBILITY ANALYSIS
# =============================================================================

def feasibility_analysis(candles: list[dict], book: dict) -> dict:
    """
    Brutally honest feasibility assessment using real data.
    """
    c = list(reversed(candles))
    mid = (book["bids"][0][0] + book["asks"][0][0]) / 2
    report = analyze_order_book(book, mid)

    # 5-minute return distribution
    returns_5m = []
    for i in range(1, len(c)):
        r = (c[i]["close"] - c[i - 1]["close"]) / c[i - 1]["close"]
        returns_5m.append(r)

    abs_returns = [abs(r) for r in returns_5m]
    mean_abs_return = statistics.mean(abs_returns) * 10000  # in bps
    median_abs_return = statistics.median(abs_returns) * 10000
    std_return = statistics.stdev(returns_5m) * 10000

    # What % of 5-min moves exceed the spread?
    spread_bps = report.spread_bps
    moves_exceeding_spread = sum(1 for r in abs_returns if r * 10000 > spread_bps)
    pct_exceeding = moves_exceeding_spread / len(abs_returns)

    # What % of 5-min moves exceed 2x spread (minimum for profit after fees)?
    moves_exceeding_2x = sum(1 for r in abs_returns if r * 10000 > 2 * spread_bps)
    pct_exceeding_2x = moves_exceeding_2x / len(abs_returns)

    # Volume analysis
    volumes_usd = [bar["volume_usd"] for bar in c]
    avg_vol = statistics.mean(volumes_usd)

    # Realistic position sizes to test
    test_positions_usd = [1000, 5000, 10000, 50000]
    max_position_usd = avg_vol * 0.10

    # Walk the book to calculate ACTUAL slippage for different sizes
    def calc_slippage(book_side, position_usd, mid_price):
        """Walk the order book to find actual fill price for a given position."""
        remaining = position_usd
        total_cost = 0
        total_qty = 0
        for price, qty in book_side:
            fill_usd = min(remaining, price * qty)
            fill_qty = fill_usd / price
            total_cost += fill_usd
            total_qty += fill_qty
            remaining -= fill_usd
            if remaining <= 0:
                break
        if total_qty == 0:
            return 0, 0
        avg_fill = total_cost / total_qty
        slippage_bps = abs(avg_fill - mid_price) / mid_price * 10000
        return slippage_bps, total_qty

    slippage_table = {}
    for pos_usd in test_positions_usd:
        buy_slip, buy_qty = calc_slippage(book["asks"], pos_usd, mid)
        sell_slip, sell_qty = calc_slippage(book["bids"], pos_usd, mid)
        slippage_table[pos_usd] = {
            "buy_slippage_bps": buy_slip,
            "sell_slippage_bps": sell_slip,
            "roundtrip_cost_bps": buy_slip + sell_slip + spread_bps,
        }

    # Use $10k as representative position for feasibility
    estimated_slippage_bps = slippage_table.get(10000, {}).get("roundtrip_cost_bps", 1.0)

    return {
        "spread_bps": spread_bps,
        "spread_usd": report.spread_usd,
        "mean_5m_move_bps": mean_abs_return,
        "median_5m_move_bps": median_abs_return,
        "std_5m_return_bps": std_return,
        "pct_moves_exceeding_spread": pct_exceeding,
        "pct_moves_exceeding_2x_spread": pct_exceeding_2x,
        "avg_5m_volume_usd": avg_vol,
        "max_position_usd": max_position_usd,
        "estimated_slippage_bps": estimated_slippage_bps,
        "slippage_table": slippage_table,
        "book_imbalance": report.book_imbalance,
        "bid_depth_usd_50lvl": sum(p * q for p, q in book["bids"]),
        "ask_depth_usd_50lvl": sum(p * q for p, q in book["asks"]),
        "total_candles_analyzed": len(c),
        "time_span_hours": len(c) * 5 / 60,
    }


# =============================================================================
# SECTION 7: RUN ALL ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("POLYMARKET 5-MINUTE STRATEGY ANALYSIS")
    print("Live data: BTCUSD, Feb 13 2026 02:36 UTC")
    print("=" * 80)

    current_price = BTCUSD_5M_CANDLES[0]["close"]  # Most recent close

    # ── MICROSTRUCTURE REPORT ──
    print("\n" + "─" * 60)
    print("1. ORDER BOOK MICROSTRUCTURE")
    print("─" * 60)
    mid = (BTCUSD_BOOK["bids"][0][0] + BTCUSD_BOOK["asks"][0][0]) / 2
    ms = analyze_order_book(BTCUSD_BOOK, mid)
    print(f"  Best Bid:          ${BTCUSD_BOOK['bids'][0][0]:,.2f}")
    print(f"  Best Ask:          ${BTCUSD_BOOK['asks'][0][0]:,.2f}")
    print(f"  Spread:            ${ms.spread_usd:.2f} ({ms.spread_bps:.4f} bps)")
    print(f"  Bid Depth (10lvl): {ms.bid_depth_10:.5f} BTC (${ms.bid_depth_usd_10:,.0f})")
    print(f"  Ask Depth (10lvl): {ms.ask_depth_10:.5f} BTC (${ms.ask_depth_usd_10:,.0f})")
    print(f"  Book Imbalance:    {ms.book_imbalance:+.4f}")
    print(f"  Weighted Mid:      ${ms.weighted_mid:,.2f}")
    print(f"  Simple Mid:        ${mid:,.2f}")
    print(f"  VWAP Deviation:    {((ms.weighted_mid - mid)/mid)*10000:+.3f} bps")
    print(f"  Large Orders (>0.5 BTC): {len(ms.large_orders)}")
    for p, q, side in ms.large_orders[:5]:
        print(f"    {side.upper():>4} {q:.5f} BTC @ ${p:,.2f} (${p*q:,.0f})")

    # ── TRADE FLOW ANALYSIS ──
    print("\n" + "─" * 60)
    print("2. TRADE FLOW ANALYSIS (last 150 trades)")
    print("─" * 60)
    # Use embedded trade data
    btc_trades = [
        {"side": "buy", "qty": "0.00690"}, {"side": "buy", "qty": "0.00003"},
        {"side": "buy", "qty": "0.03019"}, {"side": "buy", "qty": "0.00003"},
        {"side": "buy", "qty": "0.00697"}, {"side": "buy", "qty": "0.01116"},
        {"side": "buy", "qty": "0.01914"}, {"side": "buy", "qty": "0.00230"},
        {"side": "buy", "qty": "0.00767"}, {"side": "buy", "qty": "0.00003"},
        {"side": "buy", "qty": "0.00544"}, {"side": "buy", "qty": "0.00151"},
        {"side": "buy", "qty": "0.00003"}, {"side": "buy", "qty": "0.00002"},
        {"side": "buy", "qty": "0.00543"}, {"side": "buy", "qty": "0.00151"},
        {"side": "buy", "qty": "0.00003"}, {"side": "buy", "qty": "0.00002"},
        {"side": "buy", "qty": "0.03844"}, {"side": "buy", "qty": "0.01157"},
        {"side": "buy", "qty": "0.09999"}, {"side": "buy", "qty": "0.04844"},
        {"side": "buy", "qty": "0.00001"}, {"side": "buy", "qty": "0.00003"},
        {"side": "buy", "qty": "0.00151"}, {"side": "buy", "qty": "0.08990"},
        {"side": "buy", "qty": "0.00009"}, {"side": "buy", "qty": "0.05001"},
        {"side": "buy", "qty": "0.01000"}, {"side": "buy", "qty": "0.00999"},
        {"side": "buy", "qty": "0.01500"}, {"side": "buy", "qty": "0.00151"},
        {"side": "buy", "qty": "0.03348"},
        {"side": "sell", "qty": "0.05000"}, {"side": "sell", "qty": "0.00003"},
        {"side": "sell", "qty": "0.00696"},
        {"side": "sell", "qty": "0.04666"}, {"side": "sell", "qty": "0.00150"},
        {"side": "sell", "qty": "0.00182"}, {"side": "sell", "qty": "0.09997"},
        {"side": "sell", "qty": "0.04821"}, {"side": "sell", "qty": "0.00150"},
        {"side": "sell", "qty": "0.00029"}, {"side": "sell", "qty": "0.00999"},
        {"side": "sell", "qty": "0.04301"}, {"side": "sell", "qty": "0.01500"},
        {"side": "sell", "qty": "0.03197"}, {"side": "sell", "qty": "0.00038"},
        {"side": "sell", "qty": "0.00561"}, {"side": "sell", "qty": "0.01795"},
        {"side": "sell", "qty": "0.07640"}, {"side": "sell", "qty": "0.12501"},
        {"side": "sell", "qty": "0.00003"}, {"side": "sell", "qty": "0.06527"},
        {"side": "sell", "qty": "0.00010"}, {"side": "sell", "qty": "0.01500"},
        {"side": "sell", "qty": "0.03489"},
        {"side": "buy", "qty": "0.00151"}, {"side": "buy", "qty": "0.00179"},
        {"side": "sell", "qty": "0.00003"}, {"side": "sell", "qty": "0.00004"},
        {"side": "sell", "qty": "0.00005"}, {"side": "sell", "qty": "0.00006"},
        {"side": "buy", "qty": "0.00700"}, {"side": "buy", "qty": "0.05001"},
        {"side": "buy", "qty": "0.00090"}, {"side": "buy", "qty": "0.04911"},
        {"side": "buy", "qty": "0.02846"}, {"side": "buy", "qty": "0.07155"},
        {"side": "buy", "qty": "0.03358"}, {"side": "buy", "qty": "0.05643"},
        {"side": "buy", "qty": "0.00990"}, {"side": "buy", "qty": "0.00008"},
        {"side": "buy", "qty": "0.00002"},
        {"side": "buy", "qty": "0.09884"}, {"side": "buy", "qty": "0.01500"},
        {"side": "buy", "qty": "0.05364"}, {"side": "buy", "qty": "0.06116"},
        {"side": "buy", "qty": "0.06476"}, {"side": "buy", "qty": "0.04160"},
    ]
    ofi = compute_order_flow_imbalance(btc_trades)
    print(f"  Buy Volume:        {ofi['buy_volume']:.5f} BTC")
    print(f"  Sell Volume:       {ofi['sell_volume']:.5f} BTC")
    print(f"  Net Imbalance:     {ofi['net_imbalance']:+.4f}")
    print(f"  Buy Count:         {ofi['buy_count']} trades")
    print(f"  Sell Count:        {ofi['sell_count']} trades")
    print(f"  Avg Buy Size:      {ofi['avg_buy_size']:.5f} BTC")
    print(f"  Avg Sell Size:     {ofi['avg_sell_size']:.5f} BTC")
    print(f"  Large Buys (>0.05): {ofi['large_buy_count']}")
    print(f"  Large Sells (>0.05): {ofi['large_sell_count']}")

    # ── 5-MINUTE RETURN STATISTICS ──
    print("\n" + "─" * 60)
    print("3. 5-MINUTE RETURN DISTRIBUTION")
    print("─" * 60)
    returns = compute_returns(BTCUSD_5M_CANDLES)
    abs_rets_bps = [abs(r) * 10000 for r in returns]
    rets_bps = [r * 10000 for r in returns]
    print(f"  Sample Size:       {len(returns)} bars (~{len(returns)*5/60:.1f} hours)")
    print(f"  Mean Return:       {statistics.mean(rets_bps):+.2f} bps")
    print(f"  Std Dev:           {statistics.stdev(rets_bps):.2f} bps")
    print(f"  Mean |Return|:     {statistics.mean(abs_rets_bps):.2f} bps")
    print(f"  Median |Return|:   {statistics.median(abs_rets_bps):.2f} bps")
    print(f"  Max |Return|:      {max(abs_rets_bps):.2f} bps")
    print(f"  Min |Return|:      {min(abs_rets_bps):.2f} bps")
    print(f"  % > 5bps moves:    {sum(1 for r in abs_rets_bps if r > 5)/len(abs_rets_bps)*100:.1f}%")
    print(f"  % > 10bps moves:   {sum(1 for r in abs_rets_bps if r > 10)/len(abs_rets_bps)*100:.1f}%")
    print(f"  % > 20bps moves:   {sum(1 for r in abs_rets_bps if r > 20)/len(abs_rets_bps)*100:.1f}%")

    # Autocorrelation (key for strategy viability)
    if len(returns) > 2:
        mean_r = statistics.mean(returns)
        var_r = statistics.variance(returns)
        if var_r > 0:
            ac1 = sum((returns[i] - mean_r) * (returns[i + 1] - mean_r) for i in range(len(returns) - 1))
            ac1 /= (len(returns) - 1) * var_r
            print(f"  Autocorr (lag 1):  {ac1:+.4f}")
            ac2 = sum((returns[i] - mean_r) * (returns[i + 2] - mean_r) for i in range(len(returns) - 2))
            ac2 /= (len(returns) - 2) * var_r
            print(f"  Autocorr (lag 2):  {ac2:+.4f}")

    # ── VOLATILITY ──
    print("\n" + "─" * 60)
    print("4. REALIZED VOLATILITY")
    print("─" * 60)
    vols = compute_volatility(returns, window=12)
    if vols:
        print(f"  Current (1h window):  {vols[-1]*100:.1f}% annualized")
        print(f"  Mean:                 {statistics.mean(vols)*100:.1f}%")
        print(f"  Min:                  {min(vols)*100:.1f}%")
        print(f"  Max:                  {max(vols)*100:.1f}%")

    # ── RSI ──
    print("\n" + "─" * 60)
    print("5. RSI(14) ON 5-MIN BARS")
    print("─" * 60)
    rsi = compute_rsi(BTCUSD_5M_CANDLES, period=14)
    if rsi:
        print(f"  Current RSI:       {rsi[-1]:.1f}")
        print(f"  RSI Range:         [{min(rsi):.1f}, {max(rsi):.1f}]")
        overbought = sum(1 for r in rsi if r > 70) / len(rsi)
        oversold = sum(1 for r in rsi if r < 30) / len(rsi)
        print(f"  % Overbought (>70): {overbought*100:.1f}%")
        print(f"  % Oversold (<30):   {oversold*100:.1f}%")

    # ── STRATEGY SIGNALS (LIVE) ──
    print("\n" + "─" * 60)
    print("6. LIVE STRATEGY SIGNALS")
    print("─" * 60)

    strategies = [
        ("S1: Order Flow Momentum", Strategy1_OrderFlowMomentum().generate_signal(btc_trades, current_price)),
        ("S2: Book Imbalance MR", Strategy2_BookImbalanceReversion().generate_signal(BTCUSD_BOOK, current_price)),
        ("S3: Volume Spike", Strategy3_VolumeSpikeBreakout().generate_signal(BTCUSD_5M_CANDLES, current_price)),
        ("S4: RSI Mean Reversion", Strategy4_RSIMeanReversion().generate_signal(BTCUSD_5M_CANDLES, current_price)),
        ("S5: Spread Regime", Strategy5_SpreadRegimeDetection().generate_signal(BTCUSD_BOOK, current_price)),
    ]

    for name, signal in strategies:
        color = {"long": "LONG", "short": "SHORT", "flat": "FLAT"}[signal.direction]
        print(f"\n  {name}")
        print(f"    Signal:     {color} (confidence: {signal.confidence:.2f})")
        if signal.direction != "flat":
            print(f"    Entry:      ${signal.entry_price:,.2f}")
            print(f"    Stop Loss:  ${signal.stop_loss:,.2f}")
            print(f"    Take Profit: ${signal.take_profit:,.2f}")
        print(f"    Reasoning:  {signal.reasoning}")

    # ── BACKTESTS ──
    print("\n" + "─" * 60)
    print("7. BACKTEST RESULTS (on {0} 5-min bars)".format(len(BTCUSD_5M_CANDLES)))
    print("─" * 60)

    backtests = [
        ("Momentum (3-bar lookback)", backtest_momentum_on_candles(BTCUSD_5M_CANDLES, lookback=3)),
        ("Mean Reversion (3-bar)", backtest_mean_reversion_on_candles(BTCUSD_5M_CANDLES, lookback=3)),
        ("Volume Breakout", backtest_volume_breakout_on_candles(BTCUSD_5M_CANDLES)),
    ]

    for name, bt in backtests:
        print(f"\n  {name}:")
        print(f"    Trades:        {bt.total_trades}")
        print(f"    Win Rate:      {bt.win_rate*100:.1f}%")
        print(f"    Total PnL:     {bt.total_pnl_bps:+.1f} bps")
        print(f"    Avg PnL/trade: {bt.avg_pnl_bps:+.2f} bps")
        print(f"    Avg Win:       {bt.avg_win_bps:+.2f} bps")
        print(f"    Avg Loss:      {bt.avg_loss_bps:+.2f} bps")
        print(f"    Max Drawdown:  {bt.max_drawdown_bps:.1f} bps")
        print(f"    Profit Factor: {bt.profit_factor:.2f}")
        print(f"    Sharpe Ratio:  {bt.sharpe_ratio:.2f}")

    # ── FEASIBILITY ──
    print("\n" + "─" * 60)
    print("8. FEASIBILITY ASSESSMENT")
    print("─" * 60)
    f = feasibility_analysis(BTCUSD_5M_CANDLES, BTCUSD_BOOK)
    print(f"  Spread:                    {f['spread_bps']:.4f} bps (${f['spread_usd']:.2f})")
    print(f"  Mean 5m |Move|:            {f['mean_5m_move_bps']:.2f} bps")
    print(f"  Median 5m |Move|:          {f['median_5m_move_bps']:.2f} bps")
    print(f"  Std 5m Return:             {f['std_5m_return_bps']:.2f} bps")
    print(f"  % Moves > Spread:          {f['pct_moves_exceeding_spread']*100:.1f}%")
    print(f"  % Moves > 2x Spread:       {f['pct_moves_exceeding_2x_spread']*100:.1f}%")
    print(f"  Avg 5m Volume:             ${f['avg_5m_volume_usd']:,.0f}")
    print(f"  Max Position (10% of vol): ${f['max_position_usd']:,.0f}")
    print(f"\n  Slippage Table (book walk):")
    print(f"    {'Position':>10} | {'Buy Slip':>10} | {'Sell Slip':>10} | {'RT Cost':>10}")
    print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for pos, data in f['slippage_table'].items():
        print(f"    ${pos:>9,} | {data['buy_slippage_bps']:>8.2f}bp | {data['sell_slippage_bps']:>8.2f}bp | {data['roundtrip_cost_bps']:>8.2f}bp")
    print(f"\n  Est. RT Cost ($10k pos):   {f['estimated_slippage_bps']:.2f} bps")
    print(f"  Book Imbalance:            {f['book_imbalance']:+.4f}")
    print(f"  Total Bid Depth (50lvl):   ${f['bid_depth_usd_50lvl']:,.0f}")
    print(f"  Total Ask Depth (50lvl):   ${f['ask_depth_usd_50lvl']:,.0f}")
    print(f"  Data Span:                 {f['time_span_hours']:.1f} hours ({f['total_candles_analyzed']} bars)")

    # ── VIABILITY VERDICT ──
    print("\n" + "─" * 60)
    print("9. VIABILITY VERDICT")
    print("─" * 60)

    move_to_cost = f['median_5m_move_bps'] / (f['spread_bps'] + f['estimated_slippage_bps'])
    print(f"  Move-to-Cost Ratio:        {move_to_cost:.2f}x")
    print(f"    (median 5m move / total execution cost)")

    if move_to_cost > 5:
        print("  VERDICT: VIABLE - Large moves relative to costs")
    elif move_to_cost > 2:
        print("  VERDICT: MARGINAL - Possible with high accuracy")
    else:
        print("  VERDICT: NOT VIABLE - Costs eat most of the edge")

    print(f"\n  Minimum required win rate for breakeven:")
    cost_bps = f['spread_bps'] + f['estimated_slippage_bps']
    avg_move = f['mean_5m_move_bps']
    # If you win avg_move and lose avg_move, breakeven = (cost + avg_move) / (2 * avg_move)
    min_wr = (cost_bps + avg_move) / (2 * avg_move) if avg_move > 0 else 1.0
    print(f"    With avg move of {avg_move:.1f}bps and cost of {cost_bps:.2f}bps:")
    print(f"    Minimum win rate = {min_wr*100:.1f}%")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
