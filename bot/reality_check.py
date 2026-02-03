"""
Reality Check: Are these results actually achievable?
"""

import random


def show_the_math():
    """Show why the simulation results may be misleading."""

    print("=" * 70)
    print("REALITY CHECK: IS $20K/DAY ACTUALLY ACHIEVABLE?")
    print("=" * 70)

    print("""
THE CRITICAL ASSUMPTION
-----------------------
The VALUE strategy simulation assumes:

  "Cheap outcomes (priced at 20%) actually win ~20% of the time"

This means the simulation assumes markets are WELL-CALIBRATED.

BUT WAIT - if markets are well-calibrated, let's do the math:
""")

    # Well-calibrated market math
    entry = 0.20  # Buy at 20 cents
    win_rate = 0.20  # Wins 20% (well-calibrated)
    win_payout = 0.80  # Win $0.80 profit
    lose_cost = 0.20  # Lose $0.20
    fee = 0.01  # 1% fee on position

    ev_per_dollar = (win_rate * win_payout) - ((1 - win_rate) * lose_cost) - fee
    print(f"Entry price: ${entry:.2f}")
    print(f"Win rate: {win_rate:.0%}")
    print(f"Win payout: ${win_payout:.2f}")
    print(f"Lose cost: ${lose_cost:.2f}")
    print(f"Fee: {fee:.1%}")
    print()
    print(f"EV per $1 bet = ({win_rate} × ${win_payout}) - ({1-win_rate} × ${lose_cost}) - ${fee}")
    print(f"EV per $1 bet = ${win_rate * win_payout:.3f} - ${(1-win_rate) * lose_cost:.3f} - ${fee:.3f}")
    print(f"EV per $1 bet = ${ev_per_dollar:.3f}")

    if ev_per_dollar < 0:
        print(f"\n❌ NEGATIVE EV! You lose ${abs(ev_per_dollar):.3f} per dollar bet")
    else:
        print(f"\n✓ Positive EV: ${ev_per_dollar:.3f} per dollar bet")

    print("""
THE SIMULATION'S HIDDEN ASSUMPTION
----------------------------------
The code has this line:

    accuracy = 0.20 + random.gauss(0, 0.05)

This adds random noise centered at 20%, meaning sometimes
accuracy is 25%, sometimes 15%, averaging to 20%.

When accuracy > entry_price, the trade is +EV.
When accuracy < entry_price, the trade is -EV.

The simulation assumes SYMMETRIC mispricing - markets are
equally likely to be overpriced as underpriced.
""")

    print("\nWHAT WOULD MAKE THIS PROFITABLE IN REALITY:")
    print("-" * 50)

    print("""
1. SYSTEMATIC MISPRICING
   - Cheap outcomes must ACTUALLY win more often than their price suggests
   - Example: 15-cent outcomes actually win 18-20% of the time
   - This requires markets to be consistently wrong about longshots

2. EVIDENCE FOR MISPRICING (Favorite-Longshot Bias):
   - Academic research shows prediction markets often UNDERPRICE longshots
   - Bettors overbet favorites, creating value in cheap outcomes
   - This bias is documented in sports betting and prediction markets

3. EVIDENCE AGAINST:
   - Polymarket is populated by sophisticated traders
   - Arbitrageurs correct mispricing quickly
   - High-volume markets are more efficient
""")

    print("\n" + "=" * 70)
    print("REALISTIC SCENARIO ANALYSIS")
    print("=" * 70)

    scenarios = [
        ("Well-calibrated (no edge)", 0.20, 0.20),
        ("Slight mispricing (2% edge)", 0.20, 0.22),
        ("Moderate mispricing (5% edge)", 0.20, 0.25),
        ("Strong mispricing (10% edge)", 0.20, 0.30),
    ]

    position = 2000  # $2000 per trade
    trades_per_day = 168
    fee_rate = 0.01

    print(f"\nAssumptions: ${position} position, {trades_per_day} trades/day, {fee_rate:.1%} fees")
    print()
    print(f"{'Scenario':<35} | {'Entry':<6} | {'Actual WR':<10} | {'EV/Trade':<10} | {'Daily P&L':<12}")
    print("-" * 85)

    for name, entry, actual_wr in scenarios:
        win_payout = (1 - entry) * position
        lose_cost = entry * position
        fees = position * fee_rate

        ev = (actual_wr * win_payout) - ((1 - actual_wr) * lose_cost) - fees
        daily = ev * trades_per_day

        print(f"{name:<35} | ${entry:<5.2f} | {actual_wr:<10.0%} | ${ev:<+10.2f} | ${daily:<+12,.2f}")

    print("""
CONCLUSION
----------
The simulation shows $20k/day because it assumes ~20% accuracy for
20-cent outcomes (well-calibrated). But this is essentially BREAK-EVEN
before we consider the noise that sometimes pushes accuracy higher.

FOR $20K/DAY TO BE REAL, YOU NEED:
1. Markets to systematically misprice cheap outcomes by 5-10%
2. OR access to information others don't have
3. OR very low fees (<0.5%)

WHAT'S MORE REALISTIC:
- If favorite-longshot bias exists: $1,000-5,000/day possible
- If markets are efficient: $0 or small losses
- If you have information edge: potentially higher

The simulation is ILLUSTRATIVE, not PREDICTIVE.
""")


def test_calibration_sensitivity():
    """Test how sensitive results are to calibration assumptions."""

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    print("\nIf cheap outcomes (20¢) actually win X% of the time:")
    print()

    position = 2000
    trades = 168
    fee_rate = 0.01
    entry = 0.20

    for actual_wr in [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        win_payout = (1 - entry) * position
        lose_cost = entry * position
        fees = position * fee_rate

        ev = (actual_wr * win_payout) - ((1 - actual_wr) * lose_cost) - fees
        daily = ev * trades

        status = "PROFIT" if daily > 0 else "LOSS"
        print(f"  {actual_wr:>5.0%} win rate → ${daily:>+12,.2f}/day ({status})")

    print("""
KEY INSIGHT:
- At 20% (well-calibrated): roughly break-even
- Need 22%+ actual win rate for meaningful profit
- The question is: ARE markets mispriced by 2-10%?
""")


if __name__ == "__main__":
    show_the_math()
    test_calibration_sensitivity()
