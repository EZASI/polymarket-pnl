#!/usr/bin/env python3
"""
COMMAND CENTER — Cyberpunk Quant Lab Dashboard
================================================
Provides:
  - get_command_data(): Rich API endpoint for the Command Center
  - COMMAND_CENTER_HTML: Full self-contained HTML/CSS/JS dashboard
"""
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path("logs")

CATEGORY_META = {
    "stat_arb":   {"label": "Stat Arb",      "color": "#06b6d4", "icon": "📊"},
    "sentiment":  {"label": "Sentiment",      "color": "#a78bfa", "icon": "🧠"},
    "orderbook":  {"label": "Order Book",     "color": "#f59e0b", "icon": "📖"},
    "kelly":      {"label": "Kelly",          "color": "#ec4899", "icon": "🎰"},
    "chaos":      {"label": "Chaos",          "color": "#ef4444", "icon": "🌀"},
    "weather":    {"label": "Weather",        "color": "#38bdf8", "icon": "🌦️"},
}

AGENT_STRATEGIES = {
    "agent_01": "Pairs Trading",
    "agent_02": "Cross-Category Spread",
    "agent_03": "Volatility Arb",
    "agent_04": "Market Maker Spread",
    "agent_05": "Relative Value",
    "agent_06": "News Sentiment NLP",
    "agent_07": "Social Buzz Tracker",
    "agent_08": "Contrarian Sentiment",
    "agent_09": "Political Momentum",
    "agent_10": "Event Catalyst Hunter",
    "agent_11": "Liquidity Vacuum",
    "agent_12": "Depth Imbalance",
    "agent_13": "Whale Detector",
    "agent_14": "Thin Market Sniper",
    "agent_15": "Spread Compression",
    "agent_16": "Full Kelly Aggressor",
    "agent_17": "Half Kelly Conservative",
    "agent_18": "Dynamic Kelly Scaler",
    "agent_19": "Multi-Leg Kelly",
    "agent_20": "Kelly Momentum Hybrid",
    "agent_21": "Entropy Edge",
    "agent_22": "Fractal Pattern Scanner",
    "agent_23": "Weather Model Arb",
}


def get_command_data():
    """Build enriched data payload for the Command Center."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "portfolio": {},
        "agents": [],
        "cortex_feed": [],
        "purge_log": [],
        "category_stats": {},
    }

    # Load leaderboard
    lb_path = LOG_DIR / "leaderboard.json"
    lb = {}
    if lb_path.exists():
        try:
            lb = json.load(open(lb_path))
        except:
            pass

    agents_list = lb.get("agents", []) if isinstance(lb, dict) else []

    # Portfolio stats
    total_pnl = lb.get("total_pnl", 0)
    total_capital = lb.get("total_capital", 0)
    total_trades = lb.get("total_trades", 0)
    total_open = lb.get("total_open_positions", 0)
    avg_sharpe = lb.get("avg_sharpe", 0)
    progress = lb.get("daily_progress_pct", 0)

    data["portfolio"] = {
        "total_pnl": round(total_pnl, 2),
        "total_capital": round(total_capital, 2),
        "starting_capital": 440000,
        "total_trades": total_trades,
        "open_positions": total_open,
        "avg_sharpe": round(avg_sharpe, 2),
        "daily_target": 66000,
        "progress_pct": round(progress, 1),
        "n_agents": len(agents_list),
        "n_active": sum(1 for a in agents_list if not a.get("purged", False)),
        "n_danger": sum(1 for a in agents_list if a.get("drawdown_pct", 0) >= 12),
        "best_agent": lb.get("best_agent", ""),
        "worst_agent": lb.get("worst_agent", ""),
    }

    # Category aggregation
    cat_stats = {}
    for a in agents_list:
        atype = a.get("type", "")
        if atype not in cat_stats:
            meta = CATEGORY_META.get(atype, {"label": atype, "color": "#666", "icon": "?"})
            cat_stats[atype] = {
                "label": meta["label"],
                "color": meta["color"],
                "icon": meta["icon"],
                "total_pnl": 0,
                "n_agents": 0,
                "avg_sharpe": 0,
                "total_trades": 0,
                "sharpes": [],
            }
        cs = cat_stats[atype]
        cs["total_pnl"] += a.get("pnl", 0)
        cs["n_agents"] += 1
        cs["total_trades"] += a.get("n_trades", 0)
        cs["sharpes"].append(a.get("sharpe", 0))

    for k, cs in cat_stats.items():
        if cs["sharpes"]:
            cs["avg_sharpe"] = round(sum(cs["sharpes"]) / len(cs["sharpes"]), 2)
        cs["total_pnl"] = round(cs["total_pnl"], 2)
        del cs["sharpes"]
    data["category_stats"] = cat_stats

    # Build enriched agent list with strategy names
    for a in agents_list:
        aid = a.get("agent_id", "")
        atype = a.get("type", "")
        meta = CATEGORY_META.get(atype, {"label": atype, "color": "#666", "icon": "?"})
        dd = a.get("drawdown_pct", 0)
        status = "FIRED" if a.get("purged") else ("DANGER" if dd >= 12 else "ACTIVE")

        agent_entry = {
            "id": aid,
            "rank": a.get("rank", 0),
            "strategy": AGENT_STRATEGIES.get(aid, "Unknown"),
            "category": meta["label"],
            "category_color": meta["color"],
            "category_icon": meta["icon"],
            "type": atype,
            "generation": a.get("generation", 1),
            "pnl": round(a.get("pnl", 0), 2),
            "pnl_pct": round(a.get("pnl_pct", 0), 1),
            "capital": round(a.get("capital", 20000), 2),
            "win_rate": round(a.get("win_rate", 0), 1),
            "wins": a.get("wins", 0),
            "losses": a.get("losses", 0),
            "sharpe": round(a.get("sharpe", 0) if isinstance(a.get("sharpe", 0), (int, float)) else 0, 2),
            "drawdown_pct": round(dd, 1),
            "max_drawdown": round(a.get("max_drawdown", 0), 2),
            "n_trades": a.get("n_trades", 0),
            "open_positions": a.get("open_positions", 0),
            "status": status,
        }
        data["agents"].append(agent_entry)

    # Build Cortex Feed from signals files
    cortex = []
    for f in sorted(glob.glob(str(LOG_DIR / "signals_agent_*.json"))):
        try:
            sig = json.load(open(f))
            aid = Path(f).stem.replace("signals_", "")
            strategy = sig.get("strategy", AGENT_STRATEGIES.get(aid, ""))
            signals = sig.get("signals", [])
            summary = sig.get("summary", {})

            # Generate "thought" messages from signals
            for s in signals[:2]:  # Top 2 signals per agent
                market = s.get("market", "")[:60]
                direction = s.get("direction", "?")
                price = s.get("price", 0)

                # Build reason text based on available data
                reason_parts = []
                if s.get("edge"):
                    reason_parts.append(f"edge={s['edge']:.1%}")
                if s.get("spread_pct"):
                    reason_parts.append(f"spread={s['spread_pct']:.1%}")
                if s.get("imbalance"):
                    reason_parts.append(f"imb={s['imbalance']:+.2f}")
                if s.get("score"):
                    reason_parts.append(f"score={s['score']:.2f}")
                if s.get("momentum"):
                    reason_parts.append(f"mom={s['momentum']:.3f}")
                if s.get("relative_value"):
                    reason_parts.append(f"rv={s['relative_value']:+.2f}")
                if s.get("zscore"):
                    reason_parts.append(f"z={s['zscore']:.1f}")
                if s.get("entropy"):
                    reason_parts.append(f"H={s['entropy']:.2f}")

                reason = ", ".join(reason_parts) if reason_parts else f"@{price:.3f}"

                cortex.append({
                    "agent": aid,
                    "strategy": strategy,
                    "time": sig.get("time", ""),
                    "action": f"{direction} {market}",
                    "reason": reason,
                    "type": "signal",
                })

            # Summary entry
            if summary:
                pnl = summary.get("pnl", 0)
                op = summary.get("open_positions", 0)
                cortex.append({
                    "agent": aid,
                    "strategy": strategy,
                    "time": sig.get("time", ""),
                    "action": f"Scan complete: {len(signals)} signals, {op} positions open",
                    "reason": f"PnL=${pnl:+,.0f}",
                    "type": "status",
                })
        except:
            continue

    # Sort by time, most recent first
    cortex.sort(key=lambda x: x.get("time", ""), reverse=True)
    data["cortex_feed"] = cortex[:50]

    # Load purge log
    purge_path = LOG_DIR / "purge_log.json"
    if purge_path.exists():
        try:
            purge_log = json.load(open(purge_path))
            data["purge_log"] = purge_log[-20:] if isinstance(purge_log, list) else []
        except:
            pass

    return data


# ═══════════════════════════════════════════════════════════════════
# COMMAND CENTER HTML — Cyberpunk Quant Lab
# ═══════════════════════════════════════════════════════════════════

COMMAND_CENTER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Sovereign Swarm — Command Center</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* ── RESET & BASE ── */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg-deep:#0f172a;--bg-panel:#1e293b;--bg-card:#0f172a;
  --border:#334155;--border-glow:#06b6d433;
  --cyan:#06b6d4;--cyan-dim:#0e7490;--cyan-glow:#06b6d422;
  --amber:#f59e0b;--amber-dim:#d97706;
  --magenta:#ec4899;--magenta-dim:#db2777;
  --green:#10b981;--green-dim:#059669;
  --red:#ef4444;--red-dim:#dc2626;
  --purple:#a78bfa;--purple-dim:#7c3aed;
  --text:#e2e8f0;--text-dim:#94a3b8;--text-muted:#64748b;
  --font-mono:'JetBrains Mono',monospace;
  --font-sans:'Inter',system-ui,sans-serif;
}
html{font-size:14px}
body{font-family:var(--font-sans);background:var(--bg-deep);color:var(--text);min-height:100vh;overflow-x:hidden}

/* ── SCANLINES OVERLAY ── */
body::before{
  content:'';position:fixed;top:0;left:0;width:100%;height:100%;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);
  pointer-events:none;z-index:9999;
}

/* ── TOP HUD BAR ── */
.hud{
  background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
  border-bottom:1px solid var(--border);
  padding:12px 24px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;backdrop-filter:blur(16px);
}
.hud-title{
  font-family:var(--font-mono);font-size:1.1rem;font-weight:700;
  letter-spacing:3px;color:var(--cyan);text-transform:uppercase;
}
.hud-title span{color:var(--text-dim);font-weight:300;font-size:0.75rem;letter-spacing:1px;margin-left:12px;}
.hud-stats{display:flex;gap:24px;align-items:center;}
.hud-stat{text-align:center;min-width:90px;}
.hud-stat-label{font-size:0.6rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1.5px;font-family:var(--font-mono);}
.hud-stat-val{font-size:1.2rem;font-weight:800;font-family:var(--font-mono);}
.hud-controls{display:flex;gap:8px;}
.hud-btn{
  font-family:var(--font-mono);font-size:0.65rem;font-weight:600;
  padding:6px 12px;border:1px solid var(--border);border-radius:4px;
  background:transparent;color:var(--text-dim);cursor:pointer;
  text-transform:uppercase;letter-spacing:1px;transition:all 0.2s;
}
.hud-btn:hover{border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 12px var(--cyan-glow);}
.hud-btn.danger{border-color:var(--red-dim);color:var(--red-dim);}
.hud-btn.danger:hover{border-color:var(--red);color:var(--red);box-shadow:0 0 12px rgba(239,68,68,0.3);}
.hud-btn.amber{border-color:var(--amber-dim);color:var(--amber-dim);}

/* ── PROGRESS BAR ── */
.progress-strip{
  height:3px;background:var(--bg-panel);position:relative;overflow:hidden;
}
.progress-fill{
  height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));
  transition:width 1s ease;position:relative;
}
.progress-fill::after{
  content:'';position:absolute;right:0;top:-1px;bottom:-1px;width:20px;
  background:linear-gradient(90deg,transparent,rgba(6,182,212,0.6));
  animation:pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow{0%,100%{opacity:0.3}50%{opacity:1}}

/* ── MAIN GRID ── */
.god-view{
  display:grid;
  grid-template-columns:1fr 340px;
  grid-template-rows:auto 1fr;
  gap:0;
  height:calc(100vh - 50px);
  min-height:600px;
}

/* ── PANELS ── */
.panel{
  background:var(--bg-panel);border:1px solid var(--border);
  overflow:hidden;display:flex;flex-direction:column;
}
.panel-header{
  padding:10px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(15,23,42,0.6);min-height:36px;
}
.panel-title{
  font-family:var(--font-mono);font-size:0.7rem;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;letter-spacing:2px;
}
.panel-badge{
  font-family:var(--font-mono);font-size:0.6rem;font-weight:500;
  padding:2px 8px;border-radius:3px;border:1px solid;
}
.panel-body{flex:1;overflow-y:auto;padding:0;}

/* ── PETRI DISH (Scatter) ── */
.petri-dish{grid-column:1;grid-row:1;border-right:1px solid var(--border);}
.petri-body{padding:12px;height:100%;position:relative;}
#petriCanvas{width:100%!important;height:100%!important;}

/* ── CORTEX FEED ── */
.cortex{grid-column:2;grid-row:1/3;}
.cortex-body{padding:0;font-family:var(--font-mono);font-size:0.72rem;}
.cortex-entry{
  padding:8px 12px;border-bottom:1px solid rgba(51,65,85,0.5);
  transition:background 0.15s;
}
.cortex-entry:hover{background:rgba(6,182,212,0.05);}
.cortex-agent{
  font-weight:700;color:var(--cyan);display:inline-block;min-width:70px;
}
.cortex-action{color:var(--text);line-height:1.4;}
.cortex-reason{color:var(--text-muted);font-size:0.65rem;margin-top:2px;}
.cortex-time{color:var(--text-muted);font-size:0.6rem;float:right;}
.cortex-type-signal .cortex-agent{color:var(--magenta);}
.cortex-type-status .cortex-agent{color:var(--green);}
.cortex-type-kill .cortex-agent{color:var(--red);}

/* ── LEADERBOARD ── */
.leaderboard{grid-column:1;grid-row:2;border-right:1px solid var(--border);border-top:1px solid var(--border);}
.lb-table{width:100%;border-collapse:collapse;font-size:0.75rem;}
.lb-table thead{position:sticky;top:0;z-index:2;}
.lb-table th{
  padding:6px 8px;text-align:left;font-family:var(--font-mono);font-size:0.6rem;
  color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;
  background:var(--bg-deep);border-bottom:1px solid var(--border);font-weight:600;
}
.lb-table td{
  padding:5px 8px;border-bottom:1px solid rgba(51,65,85,0.3);
  font-family:var(--font-mono);font-size:0.72rem;white-space:nowrap;
}
.lb-table tr{transition:background 0.15s;}
.lb-table tr:hover{background:rgba(6,182,212,0.04);}
.lb-table tr.fired{opacity:0.4;text-decoration:line-through;}
.lb-table tr.danger{background:rgba(245,158,11,0.06);}

.rank-num{font-weight:800;color:var(--amber);font-size:0.85rem;width:24px;display:inline-block;text-align:center;}
.rank-1{color:#fbbf24;text-shadow:0 0 6px rgba(251,191,36,0.4);}
.rank-2{color:#d1d5db;}
.rank-3{color:#cd7f32;}
.agent-name{font-weight:600;color:var(--text);}
.agent-strategy{color:var(--text-muted);font-size:0.62rem;font-weight:400;}
.cat-badge{
  font-size:0.55rem;padding:1px 6px;border-radius:2px;
  border:1px solid;font-weight:500;letter-spacing:0.5px;
}
.pnl-pos{color:var(--green);font-weight:700;}
.pnl-neg{color:var(--red);font-weight:700;}
.pnl-zero{color:var(--text-dim);}
.wr-good{color:var(--green);}
.wr-bad{color:var(--red);}
.dd-warn{color:var(--amber);font-weight:600;}
.dd-danger{color:var(--red);font-weight:700;animation:blink 1s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.5}}
.status-active{color:var(--green);font-weight:600;}
.status-danger{color:var(--amber);font-weight:700;}
.status-fired{color:var(--red);font-weight:700;text-decoration:line-through;}
.gen-badge{
  font-size:0.6rem;padding:0 4px;border-radius:2px;
  background:rgba(6,182,212,0.15);color:var(--cyan);border:1px solid var(--cyan-dim);
}

/* ── CATEGORY SUMMARY STRIP ── */
.cat-strip{
  display:flex;gap:0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);
  background:var(--bg-deep);
}
.cat-chip{
  flex:1;padding:6px 10px;text-align:center;
  border-right:1px solid var(--border);font-family:var(--font-mono);
  font-size:0.65rem;
}
.cat-chip:last-child{border-right:none;}
.cat-chip-label{font-weight:600;letter-spacing:0.5px;}
.cat-chip-pnl{font-weight:700;font-size:0.8rem;margin-top:2px;}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg-deep);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--text-muted);}

/* ── RESPONSIVE ── */
@media(max-width:1200px){
  .god-view{grid-template-columns:1fr 280px;}
}
@media(max-width:900px){
  .god-view{grid-template-columns:1fr;grid-template-rows:300px auto auto;}
  .cortex{grid-column:1;grid-row:3;max-height:300px;}
}

/* ── GLOW EFFECTS ── */
.glow-cyan{text-shadow:0 0 8px var(--cyan-glow);}
.glow-green{text-shadow:0 0 8px rgba(16,185,129,0.3);}
.glow-red{text-shadow:0 0 8px rgba(239,68,68,0.3);}
</style>
</head>
<body>

<!-- ═══ TOP HUD BAR ═══ -->
<div class="hud">
  <div>
    <div class="hud-title">SOVEREIGN SWARM <span>Command Center</span></div>
  </div>
  <div class="hud-stats" id="hudStats">
    <div class="hud-stat">
      <div class="hud-stat-label">Portfolio P&L</div>
      <div class="hud-stat-val" id="hudPnl" style="color:var(--cyan)">$0</div>
    </div>
    <div class="hud-stat">
      <div class="hud-stat-label">Capital</div>
      <div class="hud-stat-val" id="hudCapital" style="color:var(--purple)">$440k</div>
    </div>
    <div class="hud-stat">
      <div class="hud-stat-label">Daily Target</div>
      <div class="hud-stat-val" id="hudTarget" style="color:var(--amber)">$66k</div>
    </div>
    <div class="hud-stat">
      <div class="hud-stat-label">Avg Sharpe</div>
      <div class="hud-stat-val" id="hudSharpe" style="color:var(--cyan)">0.00</div>
    </div>
    <div class="hud-stat">
      <div class="hud-stat-label">Agents</div>
      <div class="hud-stat-val" id="hudAgents" style="color:var(--green)">22</div>
    </div>
    <div class="hud-stat">
      <div class="hud-stat-label">Trades</div>
      <div class="hud-stat-val" id="hudTrades" style="color:var(--text-dim)">0</div>
    </div>
  </div>
  <div class="hud-controls">
    <a href="/swarm" class="hud-btn">Classic View</a>
    <a href="/" class="hud-btn">Home</a>
  </div>
</div>

<!-- ═══ PROGRESS STRIP ═══ -->
<div class="progress-strip">
  <div class="progress-fill" id="progressFill" style="width:0%"></div>
</div>

<!-- ═══ CATEGORY SUMMARY STRIP ═══ -->
<div class="cat-strip" id="catStrip"></div>

<!-- ═══ GOD VIEW GRID ═══ -->
<div class="god-view">

  <!-- ── PETRI DISH (Scatter Plot) ── -->
  <div class="panel petri-dish">
    <div class="panel-header">
      <div class="panel-title">The Petri Dish — Agent Survival Map</div>
      <div class="panel-badge" style="border-color:var(--cyan);color:var(--cyan);" id="petriCount">22 agents</div>
    </div>
    <div class="panel-body petri-body">
      <canvas id="petriCanvas"></canvas>
    </div>
  </div>

  <!-- ── CORTEX FEED ── -->
  <div class="panel cortex">
    <div class="panel-header">
      <div class="panel-title">Live Cortex Feed</div>
      <div class="panel-badge" style="border-color:var(--magenta);color:var(--magenta);" id="cortexCount">0 thoughts</div>
    </div>
    <div class="panel-body cortex-body" id="cortexFeed"></div>
  </div>

  <!-- ── LEADERBOARD ── -->
  <div class="panel leaderboard">
    <div class="panel-header">
      <div class="panel-title">Genealogy Leaderboard — Survival of the Smartest</div>
      <div class="panel-badge" style="border-color:var(--amber);color:var(--amber);" id="lbPurged">0 fired</div>
    </div>
    <div class="panel-body" style="padding:0;">
      <table class="lb-table">
        <thead><tr>
          <th>#</th><th>Agent</th><th>Category</th><th>Gen</th>
          <th>P&L</th><th>WR</th><th>Sharpe</th><th>DD%</th>
          <th>Trades</th><th>Open</th><th>Status</th>
        </tr></thead>
        <tbody id="lbBody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
// ═══════════════════════════════════════════════════
// CHART.JS SCATTER PLOT — THE PETRI DISH
// ═══════════════════════════════════════════════════
const catColors={
  'stat_arb':'#06b6d4','sentiment':'#a78bfa','orderbook':'#f59e0b',
  'kelly':'#ec4899','chaos':'#ef4444','weather':'#38bdf8'
};
const catLabels={
  'stat_arb':'Stat Arb','sentiment':'Sentiment','orderbook':'Order Book',
  'kelly':'Kelly','chaos':'Chaos','weather':'Weather'
};

let petriChart=null;

function initPetri(){
  const ctx=document.getElementById('petriCanvas').getContext('2d');
  petriChart=new Chart(ctx,{
    type:'bubble',
    data:{datasets:[]},
    options:{
      responsive:true,
      maintainAspectRatio:false,
      animation:{duration:800,easing:'easeOutQuart'},
      plugins:{
        legend:{display:true,position:'top',labels:{
          color:'#94a3b8',font:{family:"'JetBrains Mono'",size:10},
          usePointStyle:true,pointStyle:'circle',padding:12
        }},
        tooltip:{
          backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,
          titleFont:{family:"'JetBrains Mono'",size:11,weight:'bold'},
          bodyFont:{family:"'JetBrains Mono'",size:10},
          titleColor:'#06b6d4',bodyColor:'#e2e8f0',
          padding:10,cornerRadius:4,
          callbacks:{
            title:function(items){return items[0]?.raw?.label||'';},
            label:function(item){
              const r=item.raw;
              return[
                `Strategy: ${r.strategy||''}`,
                `P&L: $${r.pnl>=0?'+':''}${r.pnl.toLocaleString()}`,
                `Drawdown: ${r.dd.toFixed(1)}%`,
                `Capital: $${r.capital.toLocaleString()}`,
                `Win Rate: ${r.wr}%`,
                `Sharpe: ${r.sharpe}`,
                `Gen: ${r.gen}`,
              ];
            }
          }
        }
      },
      scales:{
        x:{
          title:{display:true,text:'DRAWDOWN %',color:'#64748b',
            font:{family:"'JetBrains Mono'",size:10,weight:'bold'}},
          grid:{color:'rgba(51,65,85,0.3)'},
          ticks:{color:'#64748b',font:{family:"'JetBrains Mono'",size:9},
            callback:v=>v.toFixed(0)+'%'},
          min:0,suggestedMax:20,
        },
        y:{
          title:{display:true,text:'P&L ($)',color:'#64748b',
            font:{family:"'JetBrains Mono'",size:10,weight:'bold'}},
          grid:{color:'rgba(51,65,85,0.3)'},
          ticks:{color:'#64748b',font:{family:"'JetBrains Mono'",size:9},
            callback:v=>'$'+(v>=0?'+':'')+v.toLocaleString()},
        }
      }
    }
  });
}

function updatePetri(agents){
  if(!petriChart) return;
  const groups={};
  agents.forEach(a=>{
    const t=a.type||'unknown';
    if(!groups[t]) groups[t]=[];
    const radius=Math.max(4,Math.min(25,Math.sqrt(a.capital/500)));
    groups[t].push({
      x:a.drawdown_pct,
      y:a.pnl,
      r:radius,
      label:a.id,
      strategy:a.strategy,
      pnl:a.pnl,
      dd:a.drawdown_pct,
      capital:a.capital,
      wr:a.win_rate,
      sharpe:a.sharpe,
      gen:a.generation,
      status:a.status,
    });
  });

  const datasets=Object.entries(groups).map(([type,data])=>{
    const color=catColors[type]||'#666';
    return{
      label:catLabels[type]||type,
      data:data,
      backgroundColor:data.map(d=>
        d.status==='FIRED'?'rgba(239,68,68,0.6)':
        d.status==='DANGER'?'rgba(245,158,11,0.6)':
        color+'99'
      ),
      borderColor:data.map(d=>
        d.status==='FIRED'?'#ef4444':
        d.status==='DANGER'?'#f59e0b':
        color
      ),
      borderWidth:2,
      hoverBorderWidth:3,
    };
  });

  petriChart.data.datasets=datasets;
  petriChart.update('none');
}

// ═══════════════════════════════════════════════════
// CORTEX FEED
// ═══════════════════════════════════════════════════
function updateCortex(feed){
  const el=document.getElementById('cortexFeed');
  if(!feed||!feed.length){
    el.innerHTML='<div class="cortex-entry" style="color:var(--text-muted);padding:20px;text-align:center;">Waiting for agent signals...</div>';
    return;
  }
  let html='';
  feed.forEach(f=>{
    const t=f.time?f.time.substring(11,19):'';
    const typeClass='cortex-type-'+(f.type||'signal');
    html+=`<div class="cortex-entry ${typeClass}">
      <span class="cortex-time">${t}</span>
      <span class="cortex-agent">${f.agent}</span>
      <div class="cortex-action">${escHtml(f.action)}</div>
      <div class="cortex-reason">${escHtml(f.reason)}</div>
    </div>`;
  });
  el.innerHTML=html;
  document.getElementById('cortexCount').textContent=feed.length+' thoughts';
}

// ═══════════════════════════════════════════════════
// LEADERBOARD
// ═══════════════════════════════════════════════════
function updateLeaderboard(agents){
  const tbody=document.getElementById('lbBody');
  let html='';
  let fired=0;
  agents.forEach((a,i)=>{
    const rankCls=i===0?'rank-1':i===1?'rank-2':i===2?'rank-3':'';
    const pnlCls=a.pnl>0?'pnl-pos':a.pnl<0?'pnl-neg':'pnl-zero';
    const wrCls=a.win_rate>=55?'wr-good':a.win_rate<40&&a.n_trades>=5?'wr-bad':'';
    const ddCls=a.drawdown_pct>=12?'dd-danger':a.drawdown_pct>=8?'dd-warn':'';
    const statusCls='status-'+a.status.toLowerCase();
    const rowCls=a.status==='FIRED'?'fired':a.status==='DANGER'?'danger':'';
    if(a.status==='FIRED') fired++;

    html+=`<tr class="${rowCls}">
      <td><span class="rank-num ${rankCls}">${a.rank}</span></td>
      <td>
        <span class="agent-name">${a.id}</span><br>
        <span class="agent-strategy">${a.strategy}</span>
      </td>
      <td><span class="cat-badge" style="border-color:${a.category_color};color:${a.category_color};">${a.category_icon} ${a.category}</span></td>
      <td><span class="gen-badge">G${a.generation}</span></td>
      <td class="${pnlCls}">$${a.pnl>=0?'+':''}${a.pnl.toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:0})}</td>
      <td class="${wrCls}">${a.win_rate.toFixed(0)}%</td>
      <td>${a.sharpe.toFixed(1)}</td>
      <td class="${ddCls}">${a.drawdown_pct.toFixed(1)}%</td>
      <td>${a.n_trades}</td>
      <td>${a.open_positions}</td>
      <td class="${statusCls}">${a.status}</td>
    </tr>`;
  });
  tbody.innerHTML=html;
  document.getElementById('lbPurged').textContent=fired+' fired';
}

// ═══════════════════════════════════════════════════
// CATEGORY STRIP
// ═══════════════════════════════════════════════════
function updateCatStrip(cats){
  const el=document.getElementById('catStrip');
  let html='';
  const order=['stat_arb','sentiment','orderbook','kelly','chaos','weather'];
  order.forEach(k=>{
    const c=cats[k];
    if(!c) return;
    const pnlColor=c.total_pnl>=0?'var(--green)':'var(--red)';
    html+=`<div class="cat-chip">
      <div class="cat-chip-label" style="color:${catColors[k]||'#666'}">${c.icon} ${c.label}</div>
      <div class="cat-chip-pnl" style="color:${pnlColor}">$${c.total_pnl>=0?'+':''}${c.total_pnl.toLocaleString()}</div>
    </div>`;
  });
  el.innerHTML=html;
}

// ═══════════════════════════════════════════════════
// HUD STATS
// ═══════════════════════════════════════════════════
function updateHUD(p){
  const pnlEl=document.getElementById('hudPnl');
  pnlEl.textContent='$'+(p.total_pnl>=0?'+':'')+p.total_pnl.toLocaleString();
  pnlEl.style.color=p.total_pnl>=0?'var(--green)':'var(--red)';

  document.getElementById('hudCapital').textContent='$'+(p.total_capital/1000).toFixed(0)+'k';
  document.getElementById('hudTarget').textContent='$'+(p.daily_target/1000).toFixed(0)+'k';
  document.getElementById('hudSharpe').textContent=p.avg_sharpe.toFixed(2);
  document.getElementById('hudAgents').textContent=p.n_active+'/'+p.n_agents;
  document.getElementById('hudTrades').textContent=p.total_trades;

  const pct=Math.max(0,Math.min(100,p.progress_pct));
  document.getElementById('progressFill').style.width=pct+'%';

  document.getElementById('petriCount').textContent=p.n_agents+' agents';
}

// ═══════════════════════════════════════════════════
// FETCH & UPDATE LOOP
// ═══════════════════════════════════════════════════
function escHtml(s){
  if(!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function fetchData(){
  try{
    const r=await fetch('/api/command');
    if(!r.ok) throw new Error('API error');
    const d=await r.json();

    if(d.portfolio) updateHUD(d.portfolio);
    if(d.agents) updatePetri(d.agents);
    if(d.agents) updateLeaderboard(d.agents);
    if(d.cortex_feed) updateCortex(d.cortex_feed);
    if(d.category_stats) updateCatStrip(d.category_stats);
  }catch(e){
    console.error('Fetch error:',e);
  }
}

// Init
initPetri();
fetchData();
setInterval(fetchData,15000);
</script>
</body>
</html>"""
