#!/usr/bin/env python3
"""
MTY-HFT Dashboard API
=====================
FastAPI backend serving OHLCV data, market state, and system health.
Serves a built-in HTML dashboard at /.

Usage:
    python dashboard/hft_api.py --port 3001 --password YourPassword
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("hft_dashboard")

DB_DSN = os.environ.get(
    "TIMESCALEDB_DSN",
    "postgresql://postgres:postgres@localhost:5432/mty_hft",
)
PASSWORD = os.environ.get("HFT_DASHBOARD_PASSWORD", "TradingBot2026")

pool: asyncpg.Pool | None = None


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

def check_auth(request: web.Request) -> bool:
    """Check cookie or query param auth."""
    if request.cookies.get("hft_auth") == PASSWORD:
        return True
    if request.query.get("password") == PASSWORD:
        return True
    return False


# ---------------------------------------------------------------------------
# HTML Dashboard (inline — no build step needed)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MTY-HFT Dashboard</title>
<style>
  :root { --bg: #0a0e17; --card: #111827; --border: #1f2937; --text: #e5e7eb; --accent: #3b82f6; --green: #10b981; --red: #ef4444; --yellow: #f59e0b; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: var(--bg); color: var(--text); font-size: 13px; }
  .header { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 16px; font-weight: 600; color: var(--accent); }
  .header .status { display: flex; gap: 16px; font-size: 12px; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }
  .status-dot.green { background: var(--green); }
  .status-dot.red { background: var(--red); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; padding: 16px 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 13px; color: #9ca3af; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); }
  .metric:last-child { border-bottom: none; }
  .metric .label { color: #9ca3af; }
  .metric .value { font-weight: 600; }
  .metric .value.green { color: var(--green); }
  .metric .value.red { color: var(--red); }
  .metric .value.yellow { color: var(--yellow); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: #6b7280; padding: 8px 6px; border-bottom: 1px solid var(--border); font-weight: 500; }
  td { padding: 6px; border-bottom: 1px solid var(--border); }
  .price-up { color: var(--green); }
  .price-down { color: var(--red); }
  .chart-container { width: 100%; height: 200px; position: relative; }
  canvas { width: 100% !important; height: 100% !important; }
  .login-page { display: flex; justify-content: center; align-items: center; height: 100vh; }
  .login-box { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 32px; width: 320px; }
  .login-box h2 { color: var(--accent); margin-bottom: 16px; text-align: center; }
  .login-box input { width: 100%; padding: 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); margin-bottom: 12px; font-family: inherit; }
  .login-box button { width: 100%; padding: 10px; background: var(--accent); border: none; border-radius: 4px; color: white; cursor: pointer; font-weight: 600; }
  .full-width { grid-column: 1 / -1; }
  .tabs { display: flex; gap: 8px; padding: 0 24px; margin-top: 16px; }
  .tab { padding: 8px 16px; background: var(--card); border: 1px solid var(--border); border-radius: 6px 6px 0 0; cursor: pointer; color: #9ca3af; }
  .tab.active { color: var(--accent); border-bottom-color: var(--bg); }
  #app { display: none; }
</style>
</head>
<body>

<!-- Login -->
<div id="login" class="login-page">
  <div class="login-box">
    <h2>MTY-HFT</h2>
    <input type="password" id="pw" placeholder="Password" onkeydown="if(event.key==='Enter')doLogin()">
    <button onclick="doLogin()">Enter</button>
  </div>
</div>

<!-- Dashboard -->
<div id="app">
  <div class="header">
    <h1>MTY-HFT &mdash; Polymarket Crypto HFT</h1>
    <div class="status">
      <span><span class="status-dot" id="db-status"></span>TimescaleDB</span>
      <span><span class="status-dot" id="redis-status"></span>Redis</span>
      <span id="last-update" style="color:#6b7280"></span>
    </div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="showTab('overview')">Overview</div>
    <div class="tab" onclick="showTab('hft')">HFT Live</div>
    <div class="tab" onclick="showTab('ohlcv')">OHLCV Data</div>
    <div class="tab" onclick="showTab('research')">Research</div>
    <div class="tab" onclick="showTab('markets')">Polymarket</div>
    <div class="tab" onclick="showTab('system')">System</div>
  </div>

  <!-- OVERVIEW TAB -->
  <div id="tab-overview" class="tab-content">
    <div class="grid">
      <div class="card">
        <h2>Database Stats</h2>
        <div id="db-stats"></div>
      </div>
      <div class="card">
        <h2>Infrastructure</h2>
        <div id="infra-stats"></div>
      </div>
      <div class="card">
        <h2>Data Coverage</h2>
        <div id="coverage-stats"></div>
      </div>
    </div>
  </div>

  <!-- HFT LIVE TAB -->
  <div id="tab-hft" class="tab-content" style="display:none">
    <div class="grid">
      <div class="card">
        <h2>Bot Status</h2>
        <div id="hft-bot-stats"></div>
      </div>
      <div class="card">
        <h2>Exposure Summary</h2>
        <div id="hft-exposure"></div>
      </div>
      <div class="card full-width">
        <h2>Active Markets &mdash; Live State</h2>
        <div style="max-height:500px;overflow-y:auto">
          <table id="hft-states-table">
            <thead><tr>
              <th>Underlying</th><th>TTE</th><th>CEX Mid</th>
              <th>Implied</th><th>p_theo</th>
              <th>YES Bid/Ask</th><th>Edge YES</th><th>Edge NO</th>
              <th>Best Side</th><th>OFI</th><th>Vol 1m</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- OHLCV TAB -->
  <div id="tab-ohlcv" class="tab-content" style="display:none">
    <div class="grid">
      <div class="card full-width">
        <h2>BTC/USDT 1m Candles (last 100)</h2>
        <div class="chart-container"><canvas id="btc-chart"></canvas></div>
      </div>
      <div class="card full-width">
        <h2>Recent Candles</h2>
        <div style="max-height:400px;overflow-y:auto">
          <table id="ohlcv-table">
            <thead><tr><th>Time (UTC)</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>Change</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- RESEARCH TAB -->
  <div id="tab-research" class="tab-content" style="display:none">
    <div class="grid">
      <div class="card full-width">
        <h2>Backtest Results</h2>
        <div style="max-height:500px;overflow-y:auto">
          <table id="backtest-table">
            <thead><tr><th>Run</th><th>Strategy</th><th>Symbol</th><th>Period</th><th>Trades</th><th>Win Rate</th><th>Sharpe</th><th>Max DD</th><th>PnL</th><th>PnL/Trade</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <h2>Strategy Comparison</h2>
        <div id="strategy-comparison"></div>
      </div>
      <div class="card">
        <h2>Latest Report</h2>
        <div id="latest-report" style="font-size:12px;max-height:300px;overflow-y:auto"></div>
      </div>
    </div>
  </div>

  <!-- MARKETS TAB -->
  <div id="tab-markets" class="tab-content" style="display:none">
    <div class="grid">
      <div class="card full-width">
        <h2>Active Polymarket Crypto Markets</h2>
        <div id="pm-markets">
          <p style="color:#6b7280">Polymarket market data will appear here once markets are discovered.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- SYSTEM TAB -->
  <div id="tab-system" class="tab-content" style="display:none">
    <div class="grid">
      <div class="card">
        <h2>Processes</h2>
        <div id="processes"></div>
      </div>
      <div class="card">
        <h2>Docker Containers</h2>
        <div id="containers"></div>
      </div>
    </div>
  </div>
</div>

<script>
const API = '';
let auth = '';

function doLogin() {
  const pw = document.getElementById('pw').value;
  fetch(API + '/api/health?password=' + encodeURIComponent(pw))
    .then(r => { if(!r.ok) throw 'bad'; return r.json(); })
    .then(d => {
      auth = pw;
      document.cookie = 'hft_auth=' + pw + ';path=/;max-age=86400';
      document.getElementById('login').style.display = 'none';
      document.getElementById('app').style.display = 'block';
      refresh();
      setInterval(refresh, 10000);
    })
    .catch(() => alert('Wrong password'));
}

// Check if already authed
(function() {
  const c = document.cookie.match(/hft_auth=([^;]+)/);
  if (c) {
    auth = c[1];
    fetch(API + '/api/health?password=' + encodeURIComponent(auth))
      .then(r => { if(r.ok) { document.getElementById('login').style.display='none'; document.getElementById('app').style.display='block'; refresh(); setInterval(refresh,10000); } });
  }
})();

function q(p) { return API + p + (p.includes('?') ? '&' : '?') + 'password=' + encodeURIComponent(auth); }

function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(e => e.style.display = 'none');
  document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = 'block';
  event.target.classList.add('active');
  refresh();
}

async function refresh() {
  try {
    // Health
    const h = await fetch(q('/api/health')).then(r=>r.json());
    document.getElementById('db-status').className = 'status-dot ' + (h.timescaledb ? 'green' : 'red');
    document.getElementById('redis-status').className = 'status-dot ' + (h.redis ? 'green' : 'red');
    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();

    // Stats
    const s = await fetch(q('/api/stats')).then(r=>r.json());
    document.getElementById('db-stats').innerHTML = Object.entries(s.tables||{}).map(([k,v]) =>
      `<div class="metric"><span class="label">${k}</span><span class="value">${Number(v).toLocaleString()} rows</span></div>`
    ).join('');

    document.getElementById('infra-stats').innerHTML = [
      ['TimescaleDB', h.timescaledb ? 'Connected' : 'Down', h.timescaledb ? 'green' : 'red'],
      ['Redis', h.redis ? 'Connected' : 'Down', h.redis ? 'green' : 'red'],
      ['Backfill', s.backfill_running ? 'Running' : 'Idle', s.backfill_running ? 'yellow' : 'green'],
    ].map(([l,v,c]) => `<div class="metric"><span class="label">${l}</span><span class="value ${c}">${v}</span></div>`).join('');

    document.getElementById('coverage-stats').innerHTML = (s.coverage||[]).map(c =>
      `<div class="metric"><span class="label">${c.symbol}</span><span class="value">${c.count.toLocaleString()} candles<br><span style="font-size:11px;color:#6b7280">${c.earliest||'?'} &rarr; ${c.latest||'?'}</span></span></div>`
    ).join('');

    // OHLCV
    const ohlcv = await fetch(q('/api/ohlcv?symbol=BTCUSDT&limit=100')).then(r=>r.json());
    if (ohlcv.length > 0) {
      const tbody = document.querySelector('#ohlcv-table tbody');
      tbody.innerHTML = ohlcv.slice().reverse().slice(0, 50).map(r => {
        const chg = ((r.close - r.open)/r.open*100).toFixed(3);
        const cls = chg >= 0 ? 'price-up' : 'price-down';
        return `<tr><td>${r.ts}</td><td>$${Number(r.open).toLocaleString()}</td><td>$${Number(r.high).toLocaleString()}</td><td>$${Number(r.low).toLocaleString()}</td><td>$${Number(r.close).toLocaleString()}</td><td>${Number(r.volume).toFixed(2)}</td><td class="${cls}">${chg}%</td></tr>`;
      }).join('');
      drawChart(ohlcv);
    }
    // HFT Live
    try {
      const hft = await fetch(q('/api/hft/states')).then(r=>r.json());
      const hftStats = await fetch(q('/api/hft/stats')).then(r=>r.json());

      // Bot status
      document.getElementById('hft-bot-stats').innerHTML = [
        ['Running', hftStats.running ? 'Active' : 'Stopped', hftStats.running ? 'green' : 'red'],
        ['Markets Tracked', hftStats.markets_tracked || 0, ''],
        ['Ticks Processed', (hftStats.ticks_processed||0).toLocaleString(), ''],
        ['Quotes Generated', hftStats.quotes_generated || 0, 'yellow'],
        ['Errors', hftStats.errors || 0, hftStats.errors > 0 ? 'red' : 'green'],
        ['Started', hftStats.started_at ? hftStats.started_at.slice(0,19) : 'N/A', ''],
      ].map(([l,v,c]) => `<div class="metric"><span class="label">${l}</span><span class="value ${c}">${v}</span></div>`).join('');

      // Exposure summary
      if (Array.isArray(hft) && hft.length > 0) {
        const byUnderlying = {};
        hft.forEach(s => {
          if (!byUnderlying[s.underlying]) byUnderlying[s.underlying] = {cnt: 0, bestEdge: 0};
          byUnderlying[s.underlying].cnt++;
          const maxE = Math.max(s.edge_yes_bps||0, s.edge_no_bps||0);
          if (maxE > byUnderlying[s.underlying].bestEdge) byUnderlying[s.underlying].bestEdge = maxE;
        });
        document.getElementById('hft-exposure').innerHTML = Object.entries(byUnderlying).map(([u, d]) =>
          `<div class="metric"><span class="label">${u}</span><span class="value">${d.cnt} mkts, best edge ${d.bestEdge.toFixed(0)} bps</span></div>`
        ).join('');

        // States table
        const tbody = document.querySelector('#hft-states-table tbody');
        tbody.innerHTML = hft.map(s => {
          const edgeYesCls = s.edge_yes_bps > 50 ? 'price-up' : s.edge_yes_bps > 0 ? '' : 'price-down';
          const edgeNoCls = s.edge_no_bps > 50 ? 'price-up' : s.edge_no_bps > 0 ? '' : 'price-down';
          const bestCls = s.best_side !== 'NONE' ? 'price-up' : '';
          const tteMin = (s.tte_s / 60).toFixed(1);
          return `<tr>
            <td><strong>${s.underlying}</strong></td>
            <td>${tteMin}m</td>
            <td>$${s.cex_mid ? s.cex_mid.toLocaleString() : '—'}</td>
            <td>${(s.implied_prob*100).toFixed(1)}%</td>
            <td><strong>${(s.p_theo*100).toFixed(1)}%</strong></td>
            <td>${s.pm_yes_bid.toFixed(3)} / ${s.pm_yes_ask.toFixed(3)}</td>
            <td class="${edgeYesCls}">${s.edge_yes_bps.toFixed(1)}</td>
            <td class="${edgeNoCls}">${s.edge_no_bps.toFixed(1)}</td>
            <td class="${bestCls}"><strong>${s.best_side}</strong></td>
            <td>${s.cex_ofi.toFixed(2)}</td>
            <td>${(s.cex_vol_1m*100).toFixed(2)}%</td>
          </tr>`;
        }).join('');
      } else {
        document.getElementById('hft-exposure').innerHTML = '<div class="metric"><span class="label">No active markets</span><span class="value">Start the bot to see live data</span></div>';
        document.querySelector('#hft-states-table tbody').innerHTML = '';
      }
    } catch(hftErr) { console.warn('HFT data unavailable:', hftErr); }

    // Backtests
    const bt = await fetch(q('/api/backtests')).then(r=>r.json());
    if (Array.isArray(bt) && bt.length > 0) {
      const btBody = document.querySelector('#backtest-table tbody');
      btBody.innerHTML = bt.map((r, i) => {
        const pnlCls = r.total_pnl >= 0 ? 'price-up' : 'price-down';
        const wrCls = r.win_rate >= 0.5 ? 'price-up' : 'price-down';
        const shCls = r.sharpe >= 1.0 ? 'price-up' : r.sharpe >= 0 ? '' : 'price-down';
        return `<tr>
          <td>#${r.id}</td>
          <td>${r.strategy_name}</td>
          <td>${r.symbol}</td>
          <td>${r.period_days}d</td>
          <td>${r.n_trades}</td>
          <td class="${wrCls}">${(r.win_rate*100).toFixed(1)}%</td>
          <td class="${shCls}">${r.sharpe.toFixed(2)}</td>
          <td class="price-down">${(r.max_drawdown_pct*100).toFixed(1)}%</td>
          <td class="${pnlCls}">$${r.total_pnl.toFixed(2)}</td>
          <td>$${r.pnl_per_trade.toFixed(2)}</td>
        </tr>`;
      }).join('');

      // Strategy comparison (best by Sharpe)
      const best = bt.reduce((a, b) => a.sharpe > b.sharpe ? a : b);
      document.getElementById('strategy-comparison').innerHTML = [
        ['Best Sharpe', `${best.strategy_name} (${best.sharpe.toFixed(2)})`, best.sharpe >= 1 ? 'green' : ''],
        ['Total Runs', bt.length, ''],
        ['Avg Win Rate', (bt.reduce((s,r)=>s+r.win_rate,0)/bt.length*100).toFixed(1) + '%', ''],
        ['Avg PnL', '$' + (bt.reduce((s,r)=>s+r.total_pnl,0)/bt.length).toFixed(2), ''],
      ].map(([l,v,c]) => `<div class="metric"><span class="label">${l}</span><span class="value ${c}">${v}</span></div>`).join('');

      // Latest report summary
      const latest = bt[0];
      document.getElementById('latest-report').innerHTML = `
        <pre style="color:var(--text);white-space:pre-wrap">
Strategy:       ${latest.strategy_name}
Symbol:         ${latest.symbol}
Period:         ${latest.period_days}d (${latest.n_candles.toLocaleString()} candles)
Trades:         ${latest.n_trades}
Win Rate:       ${(latest.win_rate*100).toFixed(1)}%
Sharpe Ratio:   ${latest.sharpe.toFixed(3)}
Max Drawdown:   ${(latest.max_drawdown_pct*100).toFixed(2)}%
Total PnL:      $${latest.total_pnl.toFixed(2)}
PnL/Trade:      $${latest.pnl_per_trade.toFixed(2)}
Fee (bps):      ${latest.fee_bps}
Position Size:  $${latest.position_size_usd.toLocaleString()}
Run Date:       ${latest.created_at.slice(0,19)}
        </pre>`;
    }

  } catch(e) { console.error('Refresh error:', e); }
}

function drawChart(data) {
  const canvas = document.getElementById('btc-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  ctx.scale(2, 2);
  const W = rect.width, H = rect.height;

  const closes = data.map(d => Number(d.close));
  const mn = Math.min(...closes), mx = Math.max(...closes);
  const pad = (mx - mn) * 0.1;

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 5; i++) {
    const y = H * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    const price = mx + pad - (mx - mn + 2*pad) * i / 4;
    ctx.fillStyle = '#6b7280'; ctx.font = '10px monospace';
    ctx.fillText('$' + price.toFixed(0), 4, y + 12);
  }

  // Line
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  closes.forEach((c, i) => {
    const x = (i / (closes.length - 1)) * W;
    const y = H - ((c - mn + pad) / (mx - mn + 2*pad)) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill
  const last = closes.length - 1;
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(59,130,246,0.15)'); grad.addColorStop(1, 'rgba(59,130,246,0)');
  ctx.fillStyle = grad; ctx.fill();

  // Current price label
  const lastPrice = closes[last];
  ctx.fillStyle = '#3b82f6'; ctx.font = 'bold 12px monospace';
  ctx.fillText('$' + lastPrice.toLocaleString(), W - 90, 16);
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

routes = web.RouteTableDef()


@routes.get("/")
async def index(request: web.Request) -> web.Response:
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


@routes.get("/api/health")
async def health(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    db_ok = False
    redis_ok = False

    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        pass

    try:
        import redis as rlib
        r = rlib.Redis(host="localhost", port=6379, socket_timeout=2)
        redis_ok = r.ping()
        r.close()
    except Exception:
        pass

    return web.json_response({
        "status": "ok",
        "timescaledb": db_ok,
        "redis": redis_ok,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


@routes.get("/api/stats")
async def stats(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    tables = {}
    coverage = []
    backfill_running = False

    try:
        async with pool.acquire() as conn:
            # Row counts
            for table in ["ohlcv", "trades_raw", "pm_markets", "pm_orderbook_snapshots", "pm_fills", "backtest_results", "live_metrics"]:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    tables[table] = count
                except Exception:
                    tables[table] = 0

            # Coverage per symbol
            rows = await conn.fetch("""
                SELECT symbol, COUNT(*) as cnt,
                       MIN(ts)::text as earliest,
                       MAX(ts)::text as latest
                FROM ohlcv
                WHERE exchange = 'binance' AND interval = '1m'
                GROUP BY symbol
                ORDER BY symbol
            """)
            for r in rows:
                coverage.append({
                    "symbol": r["symbol"],
                    "count": r["cnt"],
                    "earliest": r["earliest"][:19] if r["earliest"] else None,
                    "latest": r["latest"][:19] if r["latest"] else None,
                })

            # Check if backfill is running
            active = await conn.fetchval("""
                SELECT COUNT(*) FROM pg_stat_activity
                WHERE query LIKE '%INSERT INTO ohlcv%' AND state = 'active'
            """)
            backfill_running = active > 0

    except Exception as exc:
        log.error("Stats error: %s", exc)

    return web.json_response({
        "tables": tables,
        "coverage": coverage,
        "backfill_running": backfill_running,
    })


@routes.get("/api/ohlcv")
async def ohlcv(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    symbol = request.query.get("symbol", "BTCUSDT")
    limit = min(int(request.query.get("limit", "100")), 500)
    exchange = request.query.get("exchange", "binance")
    interval = request.query.get("interval", "1m")

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT ts::text, open::float, high::float, low::float, close::float,
                       volume::float, quote_volume::float, trades_count
                FROM ohlcv
                WHERE exchange = $1 AND symbol = $2 AND interval = $3
                ORDER BY ts DESC
                LIMIT $4
            """, exchange, symbol, interval, limit)

        data = [dict(r) for r in reversed(rows)]
        return web.json_response(data)

    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@routes.get("/api/backtests")
async def backtests(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    limit = min(int(request.query.get("limit", "20")), 100)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, strategy_name, underlying,
                       period_start::text as period_start,
                       period_end::text as period_end,
                       n_trades, hit_rate::float, sharpe::float,
                       max_drawdown::float, pnl_total::float,
                       pnl_per_trade::float,
                       run_ts::text as run_ts,
                       params_json
                FROM backtest_results
                ORDER BY run_ts DESC
                LIMIT $1
            """, limit)

        data = []
        for r in rows:
            n_trades = r["n_trades"] or 0
            pnl_total = float(r["pnl_total"] or 0)
            win_rate = float(r["hit_rate"] or 0)
            sharpe = float(r["sharpe"] or 0)
            max_dd = float(r["max_drawdown"] or 0)
            pnl_per_trade = float(r["pnl_per_trade"] or 0)

            # Compute period days
            try:
                from datetime import datetime as dt
                s = dt.fromisoformat(r["period_start"][:19])
                e = dt.fromisoformat(r["period_end"][:19])
                period_days = (e - s).days
            except Exception:
                period_days = 0

            # Extract n_candles from params if available
            raw_params = r["params_json"]
            if isinstance(raw_params, str):
                import json as _json
                try:
                    params = _json.loads(raw_params)
                except Exception:
                    params = {}
            elif isinstance(raw_params, dict):
                params = raw_params
            else:
                params = {}
            n_candles = params.get("n_candles", 0)
            fee_bps = params.get("fee_bps", 2.0)
            position_size_usd = params.get("position_size_usd", 10000)

            data.append({
                "id": r["id"],
                "strategy_name": r["strategy_name"],
                "symbol": r["underlying"] or "BTCUSDT",
                "period_days": period_days,
                "n_candles": n_candles,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "sharpe": sharpe,
                "max_drawdown_pct": max_dd,
                "total_pnl": pnl_total,
                "pnl_per_trade": pnl_per_trade,
                "fee_bps": fee_bps,
                "position_size_usd": position_size_usd,
                "created_at": r["run_ts"] or "",
            })

        return web.json_response(data)

    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# HFT Live endpoints — reads from lead_lag_bot's in-memory state
# ---------------------------------------------------------------------------

# Import bot state (will be empty dict if bot isn't running in same process)
try:
    from polymarket_hft.lead_lag_bot import active_states as _hft_states, bot_stats as _bot_stats
except ImportError:
    _hft_states: dict = {}
    _bot_stats: dict = {"running": False, "ticks_processed": 0, "markets_tracked": 0, "quotes_generated": 0, "errors": 0, "started_at": None}


@routes.get("/api/hft/states")
async def hft_states(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    states = [s.to_dict() for s in _hft_states.values()]
    # Sort by underlying then TTE
    states.sort(key=lambda s: (s["underlying"], s["tte_s"]))
    return web.json_response(states)


@routes.get("/api/hft/stats")
async def hft_stats(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return web.json_response(dict(_bot_stats))


@routes.get("/api/pm/markets")
async def pm_markets(request: web.Request) -> web.Response:
    if not check_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT market_id, question, underlying, expiry_ts::text, status, fee_tier::float
                FROM pm_markets
                WHERE status = 'active'
                ORDER BY discovered_at DESC
                LIMIT 50
            """)
        return web.json_response([dict(r) for r in rows])
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    global pool
    log.info("Connecting to TimescaleDB...")
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=5)
    log.info("DB pool ready")


async def on_cleanup(app: web.Application) -> None:
    if pool:
        await pool.close()


def create_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def _start_bot(app: web.Application) -> None:
    """Optionally start the lead-lag bot as a background task."""
    underlyings = app.get("bot_underlyings")
    if not underlyings:
        return
    from polymarket_hft.lead_lag_bot import run_bot
    app["bot_task"] = asyncio.create_task(run_bot(underlyings))
    log.info("Lead-lag bot started in-process for %s", underlyings)


async def _stop_bot(app: web.Application) -> None:
    task = app.get("bot_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="MTY-HFT Dashboard")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument(
        "--with-bot", nargs="*", default=None, metavar="UNDERLYING",
        help="Start lead-lag bot in-process. E.g. --with-bot BTC ETH",
    )
    args = parser.parse_args()

    if args.password:
        global PASSWORD
        PASSWORD = args.password

    app = create_app()

    if args.with_bot is not None:
        underlyings = [u.upper() for u in args.with_bot] if args.with_bot else ["BTC"]
        app["bot_underlyings"] = underlyings
        app.on_startup.append(_start_bot)
        app.on_cleanup.append(_stop_bot)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
