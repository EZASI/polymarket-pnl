#!/usr/bin/env python3
"""
RISK MANAGER — Agent 10 (The Overseer)
========================================
Monitors all 9 strategy agents every 15 minutes.
- Kill signal for agents losing >2% of capital ($400)
- Tracks Sharpe, drawdown, correlation across agents
- Portfolio-wide kill-all at 5% total drawdown ($10,000)
- Generates risk_report.json and leaderboard.md
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from swarm_base import SwarmAgent, update_leaderboard, LOG_DIR, STARTING_CAPITAL, DAILY_TARGET

agent = SwarmAgent("risk_manager", "risk")

# Risk parameters
MAX_LOSS_PER_AGENT = STARTING_CAPITAL * 0.02   # 2% = $400
MAX_PORTFOLIO_LOSS = STARTING_CAPITAL * 10 * 0.05  # 5% of total $200k = $10,000
MAX_OPEN_POSITIONS_PER_AGENT = 15
MAX_CONCENTRATION_PCT = 0.20  # No single market > 20% of agent capital
CORRELATION_LIMIT = 3  # Max 3 agents on same market

ALL_AGENTS = [
    "quant_arb", "quant_ta", "quant_copy",
    "news_politics", "news_entertainment", "news_world",
    "scalp_momentum", "scalp_spread", "scalp_volume",
]

def load_agent_state(agent_id):
    path = LOG_DIR / f"pnl_{agent_id}.json"
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except:
        return None

def issue_kill(agent_id, reason):
    """Send kill signal to an agent."""
    kill_path = LOG_DIR / "kill_signals.json"
    kills = {"kills": [], "updated": ""}
    if kill_path.exists():
        try:
            kills = json.load(open(kill_path))
        except:
            kills = {"kills": [], "updated": ""}

    for k in kills.get("kills", []):
        if k.get("agent_id") == agent_id and k.get("active", True):
            return False

    kills["kills"].append({
        "agent_id": agent_id,
        "reason": reason,
        "killed_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    })
    kills["updated"] = datetime.now(timezone.utc).isoformat()

    with open(kill_path, "w") as f:
        json.dump(kills, f, indent=2)
    print(f"  ⛔ KILL SIGNAL: {agent_id} — {reason}")
    return True

def run():
    print(f"[{agent.agent_id}] === RISK MANAGEMENT SWEEP ===")

    risk_events = []
    total_pnl = 0
    total_open = 0
    market_exposure = {}
    agent_summaries = []

    for aid in ALL_AGENTS:
        state = load_agent_state(aid)
        if not state:
            continue

        pnl = state.get("pnl", 0)
        capital = state.get("capital", STARTING_CAPITAL)
        positions = state.get("positions", [])
        sharpe = state.get("sharpe", 0)
        dd_pct = state.get("drawdown_pct", 0)
        gen = state.get("generation", 1)
        total_pnl += pnl
        total_open += len(positions)

        agent_summaries.append({
            "agent_id": aid, "pnl": pnl, "sharpe": sharpe,
            "drawdown_pct": dd_pct, "generation": gen,
            "positions": len(positions),
        })

        # === RULE 1: Max loss per agent ===
        if pnl < -MAX_LOSS_PER_AGENT:
            issue_kill(aid, f"Loss ${pnl:+.0f} > -${MAX_LOSS_PER_AGENT:.0f}")
            risk_events.append({
                "type": "KILL_LOSS", "agent": aid, "pnl": pnl,
                "severity": "high"
            })

        # === RULE 2: Too many positions ===
        if len(positions) > MAX_OPEN_POSITIONS_PER_AGENT:
            risk_events.append({
                "type": "WARN_POSITIONS", "agent": aid,
                "count": len(positions), "max": MAX_OPEN_POSITIONS_PER_AGENT,
                "severity": "medium"
            })

        # === RULE 3: Concentration ===
        for pos in positions:
            size = pos.get("size", 0)
            if size / max(capital, 1) > MAX_CONCENTRATION_PCT:
                risk_events.append({
                    "type": "WARN_CONCENTRATION", "agent": aid,
                    "market": pos.get("market", "")[:50],
                    "size_pct": round(size / max(capital, 1) * 100, 1),
                    "severity": "medium"
                })
            cid = pos.get("condition_id", "")
            if cid:
                if cid not in market_exposure:
                    market_exposure[cid] = []
                market_exposure[cid].append(aid)

        # === RULE 4: Win rate deterioration ===
        wins = state.get("wins", 0)
        losses = state.get("losses", 0)
        total_trades = wins + losses
        if total_trades >= 10:
            wr = wins / total_trades
            if wr < 0.35:
                risk_events.append({
                    "type": "WARN_WINRATE", "agent": aid,
                    "win_rate": round(wr * 100, 1), "trades": total_trades,
                    "severity": "medium"
                })
                if wr < 0.25:
                    issue_kill(aid, f"Win rate {wr:.0%} over {total_trades} trades")

        # === RULE 5: Drawdown approaching purge threshold ===
        if dd_pct >= 4.0 and dd_pct < 5.0:
            risk_events.append({
                "type": "WARN_DRAWDOWN", "agent": aid,
                "drawdown_pct": dd_pct,
                "severity": "high"
            })

    # === RULE 6: Portfolio-wide loss limit ===
    if total_pnl < -MAX_PORTFOLIO_LOSS:
        risk_events.append({
            "type": "CRITICAL_PORTFOLIO", "total_pnl": total_pnl,
            "threshold": -MAX_PORTFOLIO_LOSS,
            "severity": "critical"
        })
        for aid in ALL_AGENTS:
            issue_kill(aid, f"Portfolio drawdown: ${total_pnl:+.0f}")

    # === RULE 7: Correlation limit ===
    for cid, agents_list in market_exposure.items():
        if len(agents_list) > CORRELATION_LIMIT:
            risk_events.append({
                "type": "WARN_CORRELATION", "condition_id": cid[:20],
                "agents": agents_list, "count": len(agents_list),
                "severity": "low"
            })

    # Update leaderboard + live_race_stats.json
    leaderboard = update_leaderboard()

    # Risk report
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "portfolio_pnl": round(total_pnl, 2),
        "daily_target": DAILY_TARGET,
        "daily_progress_pct": round(total_pnl / DAILY_TARGET * 100, 1) if DAILY_TARGET else 0,
        "total_open_positions": total_open,
        "unique_markets": len(market_exposure),
        "risk_events": risk_events,
        "n_kills": sum(1 for e in risk_events if "KILL" in e.get("type", "")),
        "n_warnings": sum(1 for e in risk_events if "WARN" in e.get("type", "")),
        "n_critical": sum(1 for e in risk_events if e.get("severity") == "critical"),
        "agent_summaries": agent_summaries,
        "leaderboard_summary": {
            "best": leaderboard.get("best_agent"),
            "worst": leaderboard.get("worst_agent"),
            "total_agents": leaderboard.get("total_agents", 0),
            "avg_sharpe": leaderboard.get("avg_sharpe", 0),
        }
    }

    with open(LOG_DIR / "risk_report.json", "w") as f:
        json.dump(report, f, indent=2)

    generate_leaderboard_md(leaderboard, report)

    print(f"\n{'='*60}")
    print(f"  Portfolio PnL: ${total_pnl:+,.2f} / ${DAILY_TARGET:,.0f} target")
    print(f"  Open positions: {total_open} across {len(market_exposure)} markets")
    print(f"  Risk events: {len(risk_events)} ({report['n_kills']} kills, {report['n_warnings']} warnings)")
    print(f"  Best: {leaderboard.get('best_agent', 'N/A')} | Worst: {leaderboard.get('worst_agent', 'N/A')}")
    print(f"{'='*60}\n")

def generate_leaderboard_md(leaderboard, risk_report):
    """Generate human-readable leaderboard markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    agents = leaderboard.get("agents", [])
    total_pnl = leaderboard.get("total_pnl", 0)
    progress = leaderboard.get("daily_progress_pct", 0)

    lines = [
        f"# Trading Swarm Race — Leaderboard",
        f"_Updated: {now}_\n",
        f"**Portfolio: ${total_pnl:+,.2f}** | "
        f"**Target: ${DAILY_TARGET:,.0f}/day** | "
        f"**Progress: {progress:+.1f}%** | "
        f"**Agents: {leaderboard.get('total_agents', 0)}** | "
        f"**Avg Sharpe: {leaderboard.get('avg_sharpe', 0):.2f}**\n",
        "| Rank | Agent | Gen | PnL | WR | Sharpe | DD% | Trades | Status |",
        "|------|-------|-----|-----|-----|--------|-----|--------|--------|",
    ]

    for a in agents:
        pnl_icon = "+" if a["pnl"] >= 0 else ""
        status = "PURGED" if a.get("purged") else ("DANGER" if a.get("drawdown_pct", 0) >= 4 else "ACTIVE")
        lines.append(
            f"| {a['rank']} | {a['agent_id']} | {a.get('generation', 1)} | "
            f"${pnl_icon}{a['pnl']:,.0f} | {a['win_rate']:.0f}% | "
            f"{a.get('sharpe', 0):.1f} | {a.get('drawdown_pct', 0):.1f}% | "
            f"{a['n_trades']} | {status} |"
        )

    events = risk_report.get("risk_events", [])
    if events:
        lines.append(f"\n## Risk Events ({len(events)})")
        for e in events:
            sev = e.get("severity", "low").upper()
            if "KILL" in e.get("type", ""):
                lines.append(f"- [{sev}] KILL {e.get('agent', '')} — PnL: ${e.get('pnl', 0):+.0f}")
            elif "CORRELATION" in e.get("type", ""):
                lines.append(f"- [{sev}] {len(e.get('agents', []))} agents overlapping on same market")
            elif "DRAWDOWN" in e.get("type", ""):
                lines.append(f"- [{sev}] {e.get('agent', '')} drawdown at {e.get('drawdown_pct', 0):.1f}% (purge at 5%)")
            elif "WINRATE" in e.get("type", ""):
                lines.append(f"- [{sev}] {e.get('agent', '')} win rate {e.get('win_rate', 0):.0f}%")
            else:
                lines.append(f"- [{sev}] {e.get('type', '')} — {e.get('agent', '')}")
    else:
        lines.append("\n## All Clear — No Risk Events")

    lines.append(f"\n---\n_Risk Manager — 15-min evaluation cycle — 5% drawdown purge active_")

    with open(LOG_DIR / "leaderboard.md", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    run()
