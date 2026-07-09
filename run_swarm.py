#!/usr/bin/env python3
"""
SOVEREIGN SWARM — Race Controller
====================================
Orchestrates 22 trading agents + 1 Pit Boss (Agent 00).
Every 10 minutes:
  1. Run all 22 strategy agents (isolated error handling)
  2. Run Agent 00 Pit Boss (evaluates everyone, generates dashboard)
  3. Check 15% drawdown purge threshold
  4. Update leaderboard + live_race_stats.json
  5. Print race standings

$20,000 per agent = $440,000 portfolio.
Target: $3,000/day per agent = $66,000/day collective.
"""
import importlib
import traceback
from datetime import datetime, timezone
from swarm_base import update_leaderboard, check_and_purge, DAILY_TARGET, N_TRADING_AGENTS

# 22 trading agents across 5 categories
TRADING_AGENTS = [
    # --- Cross-Exchange Statistical Arbitrage (01-05) ---
    ("agent_01", "Agent 01: Pairs Trading"),
    ("agent_02", "Agent 02: Cross-Category Spread"),
    ("agent_03", "Agent 03: Volatility Arbitrage"),
    ("agent_04", "Agent 04: Market Maker Spread"),
    ("agent_05", "Agent 05: Relative Value"),
    # --- Sentiment-Based Event Prediction (06-10) ---
    ("agent_06", "Agent 06: News Sentiment NLP"),
    ("agent_07", "Agent 07: Social Buzz Tracker"),
    ("agent_08", "Agent 08: Contrarian Sentiment"),
    ("agent_09", "Agent 09: Political Momentum"),
    ("agent_10", "Agent 10: Event Catalyst Hunter"),
    # --- Order Book Imbalance & Liquidity Sniping (11-15) ---
    ("agent_11", "Agent 11: Liquidity Vacuum"),
    ("agent_12", "Agent 12: Depth Imbalance"),
    ("agent_13", "Agent 13: Whale Detector"),
    ("agent_14", "Agent 14: Thin Market Sniper"),
    ("agent_15", "Agent 15: Spread Compression"),
    # --- Recursive Kelly Criterion (16-20) ---
    ("agent_16", "Agent 16: Full Kelly Aggressor"),
    ("agent_17", "Agent 17: Half Kelly Conservative"),
    ("agent_18", "Agent 18: Dynamic Kelly Scaler"),
    ("agent_19", "Agent 19: Multi-Leg Kelly"),
    ("agent_20", "Agent 20: Kelly Momentum Hybrid"),
    # --- Chaos Theory (21-22) ---
    ("agent_21", "Agent 21: Entropy Edge"),
    ("agent_22", "Agent 22: Fractal Pattern Scanner"),
    # --- Weather Latency Arbitrage (23) ---
    ("agent_23", "Agent 23: Weather Model Arb"),
]

# Pit Boss — evaluates everyone, generates dashboard
PIT_BOSS = ("agent_00", "Agent 00: Pit Boss")

CATEGORY_MAP = {
    "stat_arb": "Cross-Exchange Stat Arb",
    "sentiment": "Sentiment Event Prediction",
    "orderbook": "Order Book & Liquidity",
    "kelly": "Recursive Kelly Criterion",
    "chaos": "Chaos Theory",
    "weather": "Weather Latency Arb",
}


def main():
    start = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"  SOVEREIGN SWARM — {start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  {N_TRADING_AGENTS} Trading Agents | $460,000 Portfolio")
    print(f"  Daily Target: ${DAILY_TARGET:,.0f}")
    print(f"{'='*70}\n")

    results = []

    # Phase 1: Run all 22 trading agents
    for module_name, label in TRADING_AGENTS:
        try:
            print(f"\n--- {label} ---")
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            mod.run()
            results.append((label, "OK"))
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append((label, f"ERROR: {e}"))

    # Phase 2: Pit Boss evaluates all agents + generates dashboard
    try:
        print(f"\n{'─'*70}")
        print(f"--- {PIT_BOSS[1]} ---")
        mod = importlib.import_module(PIT_BOSS[0])
        importlib.reload(mod)
        mod.run()
        results.append((PIT_BOSS[1], "OK"))
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        results.append((PIT_BOSS[1], f"ERROR: {e}"))

    # Phase 3: Purge check (15% drawdown = FIRED)
    print(f"\n--- PURGE CHECK (15% drawdown threshold) ---")
    purge_events = check_and_purge()
    if purge_events:
        for pe in purge_events:
            print(f"  FIRED: {pe['agent_id']} (gen {pe['generation']}) - {pe['reason']}")
            print(f"     Respawned as {pe.get('respawned_as', 'next gen')}")
    else:
        print(f"  All agents within limits")

    # Phase 4: Final leaderboard update
    lb = update_leaderboard()

    # Phase 5: Print race standings
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    agents = lb.get("agents", [])
    total_pnl = lb.get("total_pnl", 0)
    progress = lb.get("daily_progress_pct", 0)

    print(f"\n{'='*70}")
    print(f"  SOVEREIGN SWARM — RACE STANDINGS")
    print(f"{'='*70}")
    print(f"  Portfolio PnL: ${total_pnl:+,.2f} / ${DAILY_TARGET:,.0f} target")

    bar_len = 40
    filled = int(min(max(progress, 0), 100) / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  Progress: [{bar}] {progress:+.1f}%")
    print(f"  Avg Sharpe: {lb.get('avg_sharpe', 0):.2f}")
    print(f"  Capital Deployed: ${lb.get('total_capital', 0):,.0f}")
    print(f"  Total Trades: {lb.get('total_trades', 0)}")
    print(f"  Open Positions: {lb.get('total_open_positions', 0)}")

    print(f"\n{'─'*70}")
    print(f"  {'#':<4} {'Agent':<12} {'Category':<18} {'Gen':<4} {'PnL':>10} {'WR':>5} {'Sharpe':>7} {'DD%':>5} {'Trades':>6} {'Status':<7}")
    print(f"{'─'*70}")
    for a in agents:
        cat = CATEGORY_MAP.get(a.get("type", ""), a.get("type", ""))[:17]
        dd = a.get("drawdown_pct", 0)
        status = "FIRED" if a.get("purged") else ("DANGER" if dd >= 12 else "ACTIVE")
        icon = "🟢" if a["pnl"] >= 0 else "🔴"
        print(f"  {a['rank']:<4} {icon}{a['agent_id']:<11} {cat:<18} {a.get('generation',1):<4} "
              f"${a['pnl']:>+8,.0f} {a['win_rate']:>4.0f}% {a.get('sharpe',0):>6.1f} "
              f"{dd:>4.1f}% {a['n_trades']:>5} {status}")

    print(f"{'='*70}")

    # Run summary
    ok = sum(1 for _, s in results if s == "OK")
    err = len(results) - ok
    active = sum(1 for a in agents if not a.get("purged", False))
    danger = sum(1 for a in agents if a.get("drawdown_pct", 0) >= 12)

    print(f"  Agents: {ok}/{len(results)} OK | Active: {active} | Danger: {danger} | Fired: {len(purge_events)}")
    print(f"  Errors: {err} | Runtime: {elapsed:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
