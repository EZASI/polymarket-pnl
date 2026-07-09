#!/usr/bin/env python3
"""
AGENT 00 — THE PIT BOSS
=========================
Lead Monitor for the Sovereign Swarm.
- Queries all 22 trading agents every 10 minutes
- Generates dashboard.html (visual race dashboard)
- Generates leaderboard.md
- Fires agents with >15% drawdown, logs the termination
- Tracks daily P&L progress toward $66k target
"""
import json
import glob
from datetime import datetime, timezone
from pathlib import Path
from swarm_base import (
    update_leaderboard, check_and_purge, LOG_DIR,
    STARTING_CAPITAL, DAILY_TARGET, N_TRADING_AGENTS,
    DRAWDOWN_PURGE_PCT, DAILY_TARGET_PER_AGENT
)

AGENT_IDS = [f"agent_{i:02d}" for i in range(1, 23)]

CATEGORY_MAP = {
    "stat_arb": "Cross-Exchange Stat Arb",
    "sentiment": "Sentiment Event Prediction",
    "orderbook": "Order Book & Liquidity",
    "kelly": "Recursive Kelly Criterion",
    "chaos": "Chaos Theory",
    "weather": "Weather Latency Arb",
}

def run():
    print("[agent_00] === PIT BOSS EVALUATION ===")

    # Phase 1: Evaluate all agents
    agent_states = []
    for aid in AGENT_IDS:
        path = LOG_DIR / f"pnl_{aid}.json"
        if path.exists():
            try:
                state = json.load(open(path))
                agent_states.append(state)
            except:
                pass

    # Phase 2: Purge check (15% drawdown)
    purge_events = check_and_purge()
    for pe in purge_events:
        print(f"  🔥 FIRED: {pe['agent_id']} (gen {pe['generation']}) — {pe['reason']}")
        # Log to individual agent log
        log_path = LOG_DIR / f"{pe['agent_id']}.log"
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] FIRED: {pe['reason']}\n")
            f.write(f"  Respawned as generation {pe.get('respawned_as', '?')}\n")

    if not purge_events:
        print("  ✅ No agents fired this cycle")

    # Phase 3: Update leaderboard + live_race_stats.json
    lb = update_leaderboard()
    agents = lb.get("agents", [])
    total_pnl = lb.get("total_pnl", 0)

    # Phase 4: Generate dashboard.html
    generate_dashboard_html(agents, lb, purge_events)

    # Phase 5: Generate leaderboard.md
    generate_leaderboard_md(agents, lb, purge_events)

    # Phase 6: Write agent logs
    for state in agent_states:
        aid = state.get("agent_id", "")
        log_path = LOG_DIR / f"{aid}.log"
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"PnL=${state.get('pnl', 0):+.2f} "
                    f"WR={state.get('win_rate', 0):.0f}% "
                    f"Sharpe={state.get('sharpe', 0):.2f} "
                    f"DD={state.get('drawdown_pct', 0):.1f}% "
                    f"Trades={state.get('n_trades', 0)} "
                    f"Open={len(state.get('positions', []))}\n")

    # Summary
    active = sum(1 for a in agents if not a.get("purged", False))
    danger = sum(1 for a in agents if a.get("drawdown_pct", 0) >= 12)
    progress = lb.get("daily_progress_pct", 0)
    print(f"\n  Portfolio: ${total_pnl:+,.2f} / ${DAILY_TARGET:,.0f} target ({progress:+.1f}%)")
    print(f"  Agents: {active} active, {danger} in danger zone, {len(purge_events)} fired")
    print(f"  Avg Sharpe: {lb.get('avg_sharpe', 0):.2f}")


def generate_dashboard_html(agents, lb, purge_events):
    """Generate a self-contained HTML dashboard file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_pnl = lb.get("total_pnl", 0)
    total_capital = lb.get("total_capital", 0)
    total_trades = lb.get("total_trades", 0)
    total_open = lb.get("total_open_positions", 0)
    progress = lb.get("daily_progress_pct", 0)
    avg_sharpe = lb.get("avg_sharpe", 0)

    pnl_color = "#00d4aa" if total_pnl >= 0 else "#ff4055"
    progress_bar_pct = min(max(progress, 0), 100)

    # Build agent rows
    rows = ""
    for a in agents:
        pnl_cls = "g" if a["pnl"] >= 0 else "r"
        wr_cls = "g" if a.get("win_rate", 0) >= 55 else ("r" if a.get("win_rate", 0) < 40 else "")
        dd = a.get("drawdown_pct", 0)
        dd_cls = "r" if dd >= 12 else ("y" if dd >= 8 else "")
        status = "FIRED" if a.get("purged") else ("DANGER" if dd >= 12 else "ACTIVE")
        status_cls = "r" if status == "FIRED" else ("y" if status == "DANGER" else "g")
        cat = CATEGORY_MAP.get(a.get("type", ""), a.get("type", ""))
        rows += f"""<tr>
<td class="rank">{a['rank']}</td>
<td><b>{a.get('agent_id','')}</b><br><span class="dim">{cat}</span></td>
<td>{a.get('generation',1)}</td>
<td class="{pnl_cls}" style="font-weight:700">${a['pnl']:+,.0f}</td>
<td class="{wr_cls}">{a.get('win_rate',0):.0f}%</td>
<td>{a.get('sharpe',0):.2f}</td>
<td class="{dd_cls}">{dd:.1f}%</td>
<td>{a.get('n_trades',0)}</td>
<td>{a.get('open_positions',0)}</td>
<td class="{status_cls}" style="font-weight:700">{status}</td>
</tr>"""

    # Purge log
    purge_html = ""
    purge_log_path = LOG_DIR / "purge_log.json"
    if purge_log_path.exists():
        try:
            purge_log = json.load(open(purge_log_path))
            for pe in purge_log[-10:]:
                purge_html += f'<div class="purge-entry">🔥 <b>{pe.get("agent_id","")}</b> gen {pe.get("generation",1)} — {pe.get("reason","")} — {pe.get("purged_at","")[:19]}</div>'
        except:
            pass

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Sovereign Swarm — Race Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0b0f;color:#e4e5ea;min-height:100vh;padding:20px}}
.header{{text-align:center;margin-bottom:24px}}
.header h1{{font-size:24px;color:#00d4aa;letter-spacing:4px;margin-bottom:4px}}
.header .sub{{color:#5c6478;font-size:12px;letter-spacing:2px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px;max-width:1200px;margin-left:auto;margin-right:auto}}
.stat{{background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:16px;text-align:center}}
.stat-label{{font-size:10px;color:#5c6478;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}}
.stat-val{{font-size:22px;font-weight:700}}
.progress-bar{{max-width:1200px;margin:0 auto 24px;background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:16px}}
.progress-label{{font-size:11px;color:#5c6478;margin-bottom:8px;display:flex;justify-content:space-between}}
.bar-bg{{height:20px;background:#1a1d2a;border-radius:10px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:10px;background:linear-gradient(90deg,#00d4aa,#00eebb);transition:width 1s}}
table{{width:100%;max-width:1200px;margin:0 auto;border-collapse:collapse;background:#12131a;border:1px solid #1f2133;border-radius:12px;overflow:hidden}}
th{{padding:10px 8px;text-align:left;font-size:10px;color:#5c6478;letter-spacing:1px;text-transform:uppercase;background:#0f1017;border-bottom:1px solid #1f2133}}
td{{padding:8px;font-size:12px;border-bottom:1px solid #141520}}
tr:hover{{background:#14151e}}
.rank{{font-weight:800;color:#ffd43b;font-size:14px}}
.g{{color:#00d4aa}}.r{{color:#ff4055}}.y{{color:#ffd43b}}.dim{{color:#5c6478;font-size:10px}}
.purge-section{{max-width:1200px;margin:24px auto;background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:16px}}
.purge-section h3{{font-size:13px;color:#ff4055;margin-bottom:12px;letter-spacing:1px}}
.purge-entry{{font-size:11px;padding:6px 0;border-bottom:1px solid #141520;color:#b0b5c3}}
.footer{{text-align:center;padding:24px;font-size:10px;color:#2a2d3a;letter-spacing:1px;margin-top:24px}}
</style></head><body>
<div class="header">
<h1>SOVEREIGN SWARM</h1>
<div class="sub">22 AGENTS — $440,000 PORTFOLIO — UPDATED {now}</div>
</div>
<div class="stats">
<div class="stat"><div class="stat-label">Portfolio P&L</div><div class="stat-val" style="color:{pnl_color}">${total_pnl:+,.0f}</div></div>
<div class="stat"><div class="stat-label">Capital Deployed</div><div class="stat-val" style="color:#4da6ff">${total_capital:,.0f}</div></div>
<div class="stat"><div class="stat-label">Daily Target</div><div class="stat-val" style="color:#ffd43b">${DAILY_TARGET:,.0f}</div></div>
<div class="stat"><div class="stat-label">Total Trades</div><div class="stat-val" style="color:#b0b5c3">{total_trades}</div></div>
<div class="stat"><div class="stat-label">Open Positions</div><div class="stat-val" style="color:#a78bfa">{total_open}</div></div>
<div class="stat"><div class="stat-label">Avg Sharpe</div><div class="stat-val" style="color:#00c8e0">{avg_sharpe:.2f}</div></div>
</div>
<div class="progress-bar">
<div class="progress-label"><span>Daily Progress</span><span>${total_pnl:+,.0f} / ${DAILY_TARGET:,.0f} ({progress:+.1f}%)</span></div>
<div class="bar-bg"><div class="bar-fill" style="width:{progress_bar_pct:.1f}%"></div></div>
</div>
<table>
<thead><tr><th>#</th><th>Agent</th><th>Gen</th><th>P&L</th><th>WR</th><th>Sharpe</th><th>DD%</th><th>Trades</th><th>Open</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<div class="purge-section">
<h3>🔥 Termination Log</h3>
{purge_html if purge_html else '<div class="purge-entry dim">No agents terminated yet</div>'}
</div>
<div class="footer">Sovereign Swarm — Pit Boss Agent 00 — Auto-refreshes every 60 seconds</div>
</body></html>"""

    with open(LOG_DIR / "dashboard.html", "w") as f:
        f.write(html)


def generate_leaderboard_md(agents, lb, purge_events):
    """Generate leaderboard.md for the race."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_pnl = lb.get("total_pnl", 0)
    progress = lb.get("daily_progress_pct", 0)

    lines = [
        "# Sovereign Swarm — Race Leaderboard",
        f"_Updated: {now}_\n",
        f"**Portfolio: ${total_pnl:+,.2f}** | "
        f"**Target: ${DAILY_TARGET:,.0f}/day** | "
        f"**Progress: {progress:+.1f}%** | "
        f"**Agents: {lb.get('total_agents', 0)}/{N_TRADING_AGENTS}** | "
        f"**Avg Sharpe: {lb.get('avg_sharpe', 0):.2f}**\n",
        "| # | Agent | Category | Gen | PnL | WR | Sharpe | DD% | Trades | Status |",
        "|---|-------|----------|-----|-----|-----|--------|-----|--------|--------|",
    ]

    for a in agents:
        cat = CATEGORY_MAP.get(a.get("type", ""), a.get("type", ""))
        dd = a.get("drawdown_pct", 0)
        status = "FIRED" if a.get("purged") else ("DANGER" if dd >= 12 else "ACTIVE")
        pnl_sign = "+" if a["pnl"] >= 0 else ""
        lines.append(
            f"| {a['rank']} | {a['agent_id']} | {cat} | {a.get('generation',1)} | "
            f"${pnl_sign}{a['pnl']:,.0f} | {a['win_rate']:.0f}% | "
            f"{a.get('sharpe',0):.1f} | {dd:.1f}% | "
            f"{a['n_trades']} | {status} |"
        )

    # Purge log
    purge_log_path = LOG_DIR / "purge_log.json"
    if purge_log_path.exists():
        try:
            purge_log = json.load(open(purge_log_path))
            if purge_log:
                lines.append(f"\n## Termination Log ({len(purge_log)} total)")
                for pe in purge_log[-10:]:
                    lines.append(f"- {pe.get('agent_id','')} gen {pe.get('generation',1)}: {pe.get('reason','')} @ {pe.get('purged_at','')[:19]}")
        except:
            pass

    lines.append(f"\n---\n_Pit Boss Agent 00 — 10-min eval cycle — 15% drawdown purge active_")

    with open(LOG_DIR / "leaderboard.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
