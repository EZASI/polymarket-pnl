"""
War Room (smart version)

Goal:
- Make the War Room outputs *better* (more rigorous, less hand-wavy),
  even when we can't run CrewAI due to local dependency constraints.

What "smarter" means here:
- Clear assumptions + constraints
- A scoring rubric (EV after fees, capacity, tail risk, latency needs)
- Multi-round debate (proposal → attack → patch → final spec)
- Outputs are bot-ready: entry/exit rules + sizing + kill-switches

If you want *actual* CrewAI agent execution (not simulated printouts),
you must run it under Python 3.11/3.12 (Python 3.14 breaks `tiktoken`
wheels, so it tries to compile Rust).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class StrategyScore:
    """Higher is better. All values should be justified."""

    ev_after_fees: float  # expected value per $1 notional
    capacity_usd_per_day: float  # deployable size without self-slippage
    tail_risk: float  # 0..1, lower is safer
    latency_requirement_ms: int  # lower is easier
    operational_complexity: float  # 0..1, lower is easier

    def total(self) -> float:
        # Weighted blend: we care about EV and capacity, penalize risk/complexity/latency.
        return (
            10.0 * self.ev_after_fees
            + 0.00001 * self.capacity_usd_per_day
            - 5.0 * self.tail_risk
            - 0.001 * self.latency_requirement_ms
            - 2.0 * self.operational_complexity
        )


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def simulate_smart_war_room(rounds: int = 2) -> None:
    """
    Simulated debate, but *structured* and *actionable*.

    NOTE: This does NOT claim guaranteed profitability. It yields a plan
    that can be objectively backtested / paper-traded.
    """

    _print_header("WAR ROOM (SMART) — Round-based strategy design")

    print("Constraints we must respect:")
    print("- Target: $8,000/day *net* is only plausible with BIG capacity or real structural arb.")
    print("- 15m crypto markets: fees + slippage dominate small edges.")
    print("- Safety: avoid 'one-leg fill' and avoid strategies requiring illegal info.")
    print()

    # Round 1: Propose 3 candidates
    _print_header("Round 1 — Proposal (3 candidates)")

    candidates: List[Tuple[str, str, StrategyScore]] = []

    # Candidate A: Mutually-exclusive basket arb (negative-risk style)
    candidates.append(
        (
            "Basket Arbitrage (Mutually Exclusive Group)",
            (
                "Monitor outcome sets where exactly 1 outcome can win.\n"
                "Compute Σ(best_ask_yes_i). If Σ < (1 - fees - slippage_buffer), buy YES on all.\n"
                "Compute Σ(best_bid_yes_i). If Σ > (1 + fees + slippage_buffer), sell YES on all (or equivalently buy NO).\n"
                "This is structural (math), not prediction."
            ),
            StrategyScore(
                ev_after_fees=0.005,  # 0.5%/turn once costs included (conservative)
                capacity_usd_per_day=2_000_000,  # can scale if many groups exist + liquidity
                tail_risk=0.15,  # still has execution/partial fill risk
                latency_requirement_ms=150,  # needs fast-ish but not HFT-level
                operational_complexity=0.55,
            ),
        )
    )

    # Candidate B: Fee-aware market making around fair value (carry the spread)
    candidates.append(
        (
            "Maker-Rebate Market Making (Edge = Spread + Rebate, Hedge with Futures)",
            (
                "Provide quotes on YES/NO around computed fair value; earn spread + rebates.\n"
                "Hedge underlying exposure via Binance/Bybit perps.\n"
                "Primary profit = microstructure, not being 'right'."
            ),
            StrategyScore(
                ev_after_fees=0.0015,
                capacity_usd_per_day=5_000_000,
                tail_risk=0.35,  # inventory risk if hedge breaks / gaps
                latency_requirement_ms=50,  # quote management needs speed
                operational_complexity=0.80,  # high (hedging + infra)
            ),
        )
    )

    # Candidate C: Strike-ladder attention decay (not generic lead-lag)
    candidates.append(
        (
            "Strike-Ladder Consistency Arb (Same asset, multiple strikes/expiries)",
            (
                "For BTC/ETH 15m products with multiple strikes (K1<K2<K3):\n"
                "Probability must be monotone: P(S>K1) ≥ P(S>K2) ≥ P(S>K3).\n"
                "And differences must align with realized vol + time-to-expiry.\n"
                "Trade violations: buy underpriced, sell overpriced legs while delta-hedging.\n"
                "This is *structure + no-arb constraints*, not simple latency."
            ),
            StrategyScore(
                ev_after_fees=0.003,
                capacity_usd_per_day=1_000_000,
                tail_risk=0.25,
                latency_requirement_ms=120,
                operational_complexity=0.65,
            ),
        )
    )

    for name, desc, score in candidates:
        print(f"- {name}\n  {desc}\n  Score(total)≈ {score.total():.3f}\n")

    # Round 2: Attack + patch best candidate
    best = max(candidates, key=lambda x: x[2].total())
    best_name, best_desc, best_score = best

    _print_header("Round 2 — Attack the best candidate (Risk Officer) + Patch (Strategist)")
    print(f"Selected to stress-test: {best_name}\n")

    print("Risk Officer: failure modes")
    print("- Partial fills: you buy 7/10 outcomes and price moves, leaving you exposed.")
    print("- Hidden correlation: 'mutually exclusive' may not be strictly exclusive (bad market definition).")
    print("- Fees/slippage: edge disappears if you’re crossing the spread on many legs.")
    print("- API limits: fetching 10 books x 50 groups can rate-limit you.")
    print()

    print("Strategist: patches / constraints")
    print("- Only trade when edge > max(3*fees, 2*spread_sum, fixed $ buffer).")
    print("- Use IOC/FOK style execution where possible; otherwise execute via staged basket:")
    print("  - Leg into the *most liquid* outcomes first, then complete the tail.")
    print("- Enforce exclusivity via whitelist of known group types.")
    print("- Cap exposure: if not 100% filled within 300ms, cancel remainder and unwind filled legs.")
    print()

    _print_header("Final Output — Bot-ready spec (the thing you actually implement)")
    print("Strategy: Basket Arbitrage (Mutually Exclusive Group)\n")
    print("Inputs:")
    print("- Group definition: list of token_ids in a mutually exclusive set")
    print("- For each token_id: best_ask_yes, best_bid_yes, available size at those levels")
    print("- Fee model: maker/taker + expected slippage buffer per leg\n")

    print("Entry conditions (BUY basket):")
    print("- Compute cost = Σ(ask_yes_i) using *only* sizes you can fill (depth-aware).")
    print("- Require cost <= 1.0 - (fee_buffer + slippage_buffer + safety_margin).")
    print("- safety_margin default: 0.5% of notional.\n")

    print("Entry conditions (SELL basket):")
    print("- Compute proceeds = Σ(bid_yes_i) depth-aware.")
    print("- Require proceeds >= 1.0 + (fee_buffer + slippage_buffer + safety_margin).\n")

    print("Sizing:")
    print("- Let fillable_notional = min_i(depth_at_best_i).")
    print("- Place size so every leg can fill at quoted levels.")
    print("- Per-trade cap: min(10% bankroll, $25k) until proven in paper-trade.\n")

    print("Kill-switches:")
    print("- If any leg not filled in 300ms, cancel all remaining.")
    print("- If filled legs exist, immediately unwind at market if residual edge < 0.")
    print("- Daily loss limit: 1% bankroll.\n")

    print("Path to $8k/day (realistic math):")
    print("- If net edge ≈ 0.5% per completed basket, need ~$1.6M/day notional *turnover*.")
    print("- That’s doable only if there are enough liquid groups OR you have maker rebates.\n")


if __name__ == "__main__":
    simulate_smart_war_room(rounds=2)
