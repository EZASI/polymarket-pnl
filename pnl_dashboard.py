#!/usr/bin/env python3
"""
MTY22 Polymarket PnL Dashboard
Auto-refreshes every 2 minutes. Shows real-time profit tracking.
Usage: python3 pnl_dashboard.py
Then open http://localhost:8888
"""

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from flask import Flask, jsonify

app = Flask(__name__)

USER = "0x1831d1abf62dfadabcabb3d1c2bf4476146a6c99"
BASE = "https://data-api.polymarket.com"

# Cache to avoid hammering API
cache = {"data": None, "ts": 0}
CACHE_TTL = 120  # 2 min cache


def fetch_activity(hours_back=168):
    """Fetch activity using offset-based pagination."""
    now = int(datetime.now(timezone.utc).timestamp())
    cutoff = now - hours_back * 3600

    all_activity = []
    offset = 0

    while offset < 20000:
        try:
            r = requests.get(f"{BASE}/activity", params={
                "user": USER, "limit": 500, "offset": offset
            }, timeout=30)
            if r.status_code != 200:
                print(f"  Activity API status {r.status_code}")
                break
            batch = r.json()
            if not isinstance(batch, list) or len(batch) == 0:
                break
            all_activity.extend(batch)

            # Check if we've gone far enough back
            oldest = min(int(a["timestamp"]) for a in batch)
            if oldest < cutoff:
                break
            if len(batch) < 500:
                break
            offset += 500
        except Exception as e:
            print(f"Activity API error: {e}")
            break

    print(f"  Fetched {len(all_activity)} activities across {offset // 500 + 1} pages")
    return all_activity


def fetch_positions():
    """Fetch all positions from Polymarket Data API.
    Returns dict of conditionId -> {redeemable, size, curPrice} for quick lookup.
    """
    positions = {}
    try:
        r = requests.get(f"{BASE}/positions", params={
            "user": USER, "limit": 500, "sizeThreshold": 0
        }, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                for p in data:
                    cid = p.get("conditionId", "")
                    if cid:
                        positions[cid] = {
                            "redeemable": p.get("redeemable", False),
                            "size": float(p.get("size", 0)),
                            "curPrice": float(p.get("curPrice", 0)),
                        }
    except Exception as e:
        print(f"Positions API error: {e}")
    return positions


def compute_pnl(activity, window_seconds, positions):
    """Compute PnL for a given time window.

    Classification rules (from reconcile.py):
      A. OPEN — market not ended
      B. CLOSED_WIN_REDEEMED — ended, redeemed > 0
      C. CLOSED_WIN_UNREDEEMED — ended, no redeem, positions says redeemable
      D. CLOSED_LOSS — ended, not redeemable (zero-redeem or no position)

    Two PnL views:
      - realized_pnl: only actual cash flows (redeems + sells - buys)
      - economic_pnl: realized + expected value of unredeemed winning positions
    """
    now = int(datetime.now(timezone.utc).timestamp())
    cutoff = now - window_seconds

    recent = [a for a in activity if int(a["timestamp"]) >= cutoff]

    trades = [a for a in recent if a.get("type") == "TRADE"]
    redeems = [a for a in recent if a.get("type") == "REDEEM"]

    # Group by market
    markets = defaultdict(lambda: {
        "buys": [], "sells": [], "redeems": [],
        "title": "", "slug": "", "asset": "", "has_buy_in_window": False
    })

    for t in trades:
        cid = t.get("conditionId", "")
        side = t.get("side", "")
        usdc = float(t.get("usdcSize", 0))
        size = float(t.get("size", 0))
        slug = t.get("slug", "")
        asset = slug.split("-")[0].upper() if slug else "?"

        if side == "BUY":
            markets[cid]["buys"].append({"size": size, "usdc": usdc})
            markets[cid]["has_buy_in_window"] = True
        elif side == "SELL":
            markets[cid]["sells"].append({"size": size, "usdc": usdc})

        markets[cid]["title"] = t.get("title", "")
        markets[cid]["slug"] = slug
        markets[cid]["asset"] = asset

    for r in redeems:
        cid = r.get("conditionId", "")
        usdc = float(r.get("usdcSize", 0))
        size = float(r.get("size", 0))
        slug = r.get("slug", "")
        asset = slug.split("-")[0].upper() if slug else "?"
        markets[cid]["redeems"].append({"size": size, "usdc": usdc})
        if not markets[cid]["slug"] and slug:
            markets[cid]["slug"] = slug
            markets[cid]["asset"] = asset

    # Compute per-market results
    realized_total = 0
    economic_total = 0
    total_bought = 0
    wins = 0
    losses = 0
    open_count = 0
    unredeemed_wins = 0
    unrealized_redeemable = 0
    asset_stats = defaultdict(lambda: {
        "wins": 0, "losses": 0, "open": 0, "realized_pnl": 0,
        "economic_pnl": 0, "volume": 0, "trades": 0, "unredeemed_wins": 0
    })

    for cid, data in markets.items():
        if not data["has_buy_in_window"]:
            continue

        gross_buy = sum(b["usdc"] for b in data["buys"])
        gross_sell = sum(s["usdc"] for s in data["sells"])
        redeemed = sum(r["usdc"] for r in data["redeems"])
        net_shares = sum(b["size"] for b in data["buys"]) - sum(s["size"] for s in data["sells"])

        asset = data["asset"]
        asset_stats[asset]["volume"] += gross_buy
        asset_stats[asset]["trades"] += 1
        total_bought += gross_buy

        if redeemed > 0.01:
            # Rule B: CLOSED_WIN_REDEEMED
            rpnl = redeemed + gross_sell - gross_buy
            wins += 1
            realized_total += rpnl
            economic_total += rpnl
            asset_stats[asset]["wins"] += 1
            asset_stats[asset]["realized_pnl"] += rpnl
            asset_stats[asset]["economic_pnl"] += rpnl
        elif net_shares > 0.01:
            slug = data["slug"]
            is_open = True
            try:
                parts = slug.split("-")
                mkt_start = int(parts[-1])
                dur = 900 if "15m" in slug else 300
                mkt_end = mkt_start + dur
                if mkt_end < now:
                    is_open = False
            except:
                pass

            if is_open:
                # Rule A: OPEN
                open_count += 1
                asset_stats[asset]["open"] += 1
            else:
                pos = positions.get(cid)
                if pos and pos["redeemable"]:
                    # Rule C: CLOSED_WIN_UNREDEEMED
                    expected_redeem = pos["size"]
                    epnl = expected_redeem + gross_sell - gross_buy
                    wins += 1
                    unredeemed_wins += 1
                    unrealized_redeemable += expected_redeem
                    # realized_pnl stays 0 for these
                    economic_total += epnl
                    asset_stats[asset]["wins"] += 1
                    asset_stats[asset]["unredeemed_wins"] += 1
                    asset_stats[asset]["economic_pnl"] += epnl
                else:
                    # Rule D: CLOSED_LOSS
                    rpnl = gross_sell - gross_buy
                    losses += 1
                    realized_total += rpnl
                    economic_total += rpnl
                    asset_stats[asset]["losses"] += 1
                    asset_stats[asset]["realized_pnl"] += rpnl
                    asset_stats[asset]["economic_pnl"] += rpnl
        else:
            # Sold out or negligible
            rpnl = gross_sell - gross_buy
            if rpnl < -0.01:
                losses += 1
                asset_stats[asset]["losses"] += 1
            else:
                wins += 1
                asset_stats[asset]["wins"] += 1
            realized_total += rpnl
            economic_total += rpnl
            asset_stats[asset]["realized_pnl"] += rpnl
            asset_stats[asset]["economic_pnl"] += rpnl

    return {
        "markets": len([c for c, d in markets.items() if d["has_buy_in_window"]]),
        "wins": wins,
        "losses": losses,
        "open": open_count,
        "unredeemed_wins": unredeemed_wins,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "total_bought": round(total_bought, 2),
        "realized_pnl": round(realized_total, 2),
        "unrealized_redeemable": round(unrealized_redeemable, 2),
        "economic_pnl": round(economic_total, 2),
        "asset_stats": {k: dict(v) for k, v in asset_stats.items()},
    }


def get_dashboard_data():
    """Get full dashboard data with caching."""
    now = time.time()
    if cache["data"] and now - cache["ts"] < CACHE_TTL:
        return cache["data"]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching fresh data...")
    activity = fetch_activity(hours_back=168)  # 7 days
    positions = fetch_positions()
    print(f"  {len(positions)} positions "
          f"({sum(1 for p in positions.values() if p['redeemable'])} redeemable)")

    windows = {
        "15m": 900,
        "1h": 3600,
        "3h": 3 * 3600,
        "6h": 6 * 3600,
        "24h": 24 * 3600,
        "7d": 7 * 24 * 3600,
    }

    results = {}
    for label, seconds in windows.items():
        results[label] = compute_pnl(activity, seconds, positions)

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wallet": USER,
        "windows": results,
    }

    cache["data"] = data
    cache["ts"] = now
    return data


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/pnl")
def api_pnl():
    data = get_dashboard_data()
    return jsonify(data)


@app.route("/api/flush")
def api_flush():
    """Force cache flush."""
    cache["data"] = None
    cache["ts"] = 0
    return jsonify({"status": "cache flushed"})


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTY22 PnL Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 20px 30px;
    border-bottom: 1px solid #2a2a4a;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .header h1 { font-size: 22px; color: #fff; }
  .header .subtitle { color: #888; font-size: 13px; margin-top: 4px; }
  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #888;
  }
  .status .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00c853;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .container { padding: 24px 30px; max-width: 1400px; margin: 0 auto; }

  /* Main PnL Cards */
  .pnl-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 30px;
  }
  .pnl-card {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
  }
  .pnl-card:hover {
    transform: translateY(-2px);
    border-color: #4a4a6a;
  }
  .pnl-card .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #666;
    margin-bottom: 8px;
  }
  .pnl-card .value {
    font-size: 32px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .pnl-card .value.positive { color: #00e676; }
  .pnl-card .value.negative { color: #ff5252; }
  .pnl-card .value.zero { color: #888; }
  .pnl-card .realized {
    font-size: 13px;
    color: #999;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }
  .pnl-card .meta {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
  }
  .pnl-card .meta .wr { color: #4fc3f7; }

  /* Detail Sections */
  .section {
    background: #12121f;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
  }
  .section h2 {
    font-size: 16px;
    color: #aaa;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Asset Table */
  .asset-table {
    width: 100%;
    border-collapse: collapse;
  }
  .asset-table th {
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    padding: 8px 12px;
    border-bottom: 1px solid #2a2a4a;
  }
  .asset-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #1a1a2a;
    font-variant-numeric: tabular-nums;
    font-size: 14px;
  }
  .asset-table tr:hover td { background: #1a1a2e; }
  .asset-icon {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
  }
  .asset-icon .coin {
    width: 24px; height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 700;
    color: #fff;
  }
  .coin.btc { background: #f7931a; }
  .coin.sol { background: #9945ff; }
  .coin.xrp { background: #23292f; }
  .coin.other { background: #555; }

  .positive { color: #00e676; }
  .negative { color: #ff5252; }
  .pending-color { color: #ffa726; }

  /* Countdown */
  .countdown {
    font-size: 12px;
    color: #555;
    text-align: center;
    margin-top: 20px;
  }
  .countdown span { color: #4fc3f7; font-weight: 600; }

  /* Loading */
  .loading {
    text-align: center;
    padding: 60px;
    color: #555;
    font-size: 16px;
  }
  .loading .spinner {
    width: 40px; height: 40px;
    border: 3px solid #2a2a4a;
    border-top-color: #4fc3f7;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Daily projection */
  .projection {
    display: flex;
    gap: 24px;
    margin-top: 16px;
    flex-wrap: wrap;
  }
  .proj-item {
    flex: 1;
    min-width: 150px;
    background: #1a1a2e;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .proj-item .proj-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  .proj-item .proj-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>MTY22 PnL Dashboard</h1>
    <div class="subtitle">Polymarket Up/Down Sniper &mdash; Live Tracking</div>
  </div>
  <div class="status">
    <div class="dot"></div>
    <span id="update-time">Loading...</span>
  </div>
</div>

<div class="container">
  <div id="content">
    <div class="loading">
      <div class="spinner"></div>
      Fetching data from Polymarket...
    </div>
  </div>
  <div class="countdown">Next refresh in <span id="countdown">120</span>s</div>
</div>

<script>
const REFRESH_INTERVAL = 120; // 2 minutes
let countdown = REFRESH_INTERVAL;

function fmt(n) {
  if (n === undefined || n === null) return '$0';
  const abs = Math.abs(n);
  const sign = n >= 0 ? '+' : '-';
  if (abs >= 1000) return sign + '$' + abs.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
  return sign + '$' + abs.toFixed(2);
}

function cls(n) {
  if (n > 0.01) return 'positive';
  if (n < -0.01) return 'negative';
  return 'zero';
}

function coinClass(asset) {
  const a = asset.toLowerCase();
  if (a === 'btc') return 'btc';
  if (a === 'sol') return 'sol';
  if (a === 'xrp') return 'xrp';
  return 'other';
}

function render(data) {
  const w = data.windows;
  const windows = ['15m', '1h', '3h', '6h', '24h', '7d'];

  // Get 24h data for projection
  const d24 = w['24h'] || {};
  const hourlyRate = (d24.economic_pnl || 0) / 24;
  const dailyProj = hourlyRate * 24;
  const monthlyProj = dailyProj * 30;

  let html = '';

  // PnL Cards — show economic PnL as main, realized as subtitle
  html += '<div class="pnl-grid">';
  for (const wk of windows) {
    const d = w[wk] || {};
    const epnl = d.economic_pnl || 0;
    const rpnl = d.realized_pnl || 0;
    const unredeemed = d.unrealized_redeemable || 0;
    const urWins = d.unredeemed_wins || 0;
    html += `
      <div class="pnl-card">
        <div class="label">${wk}</div>
        <div class="value ${cls(epnl)}">${fmt(epnl)}</div>
        <div class="realized">realized: <span class="${cls(rpnl)}">${fmt(rpnl)}</span>${unredeemed > 0 ? ` · <span class="pending-color">$${unredeemed.toLocaleString('en-US',{maximumFractionDigits:0})} pending</span>` : ''}</div>
        <div class="meta">
          ${d.wins || 0}W / ${d.losses || 0}L${d.open ? ` / ${d.open} open` : ''}${urWins ? ` <span class="pending-color">(${urWins} unredeemed)</span>` : ''}
          ${(d.wins + d.losses) > 0 ? ` &middot; <span class="wr">${d.win_rate}%</span>` : ''}
        </div>
      </div>`;
  }
  html += '</div>';

  // Projection
  html += `
    <div class="section">
      <h2>Projection (based on 24h rate)</h2>
      <div class="projection">
        <div class="proj-item">
          <div class="proj-label">Hourly Rate</div>
          <div class="proj-value ${cls(hourlyRate)}">${fmt(hourlyRate)}</div>
        </div>
        <div class="proj-item">
          <div class="proj-label">Daily Projection</div>
          <div class="proj-value ${cls(dailyProj)}">${fmt(dailyProj)}</div>
        </div>
        <div class="proj-item">
          <div class="proj-label">Monthly Projection</div>
          <div class="proj-value ${cls(monthlyProj)}">${fmt(monthlyProj)}</div>
        </div>
        <div class="proj-item">
          <div class="proj-label">24h Volume</div>
          <div class="proj-value" style="color:#4fc3f7">$${(d24.total_bought||0).toLocaleString('en-US',{maximumFractionDigits:0})}</div>
        </div>
      </div>
    </div>`;

  // Asset Breakdown (24h)
  const assets = d24.asset_stats || {};
  const assetKeys = Object.keys(assets).sort();
  if (assetKeys.length > 0) {
    html += `
      <div class="section">
        <h2>Asset Breakdown (24h)</h2>
        <table class="asset-table">
          <tr>
            <th>Asset</th>
            <th>Markets</th>
            <th>Wins</th>
            <th>Losses</th>
            <th>Open</th>
            <th>Win Rate</th>
            <th>Volume</th>
            <th>Realized PnL</th>
            <th>Economic PnL</th>
          </tr>`;
    for (const asset of assetKeys) {
      const s = assets[asset];
      const wr = (s.wins + s.losses) > 0 ? (s.wins / (s.wins + s.losses) * 100).toFixed(1) : '-';
      const rpnl = s.realized_pnl || 0;
      const epnl = s.economic_pnl || 0;
      html += `
          <tr>
            <td><span class="asset-icon"><span class="coin ${coinClass(asset)}">${asset.slice(0,1)}</span> ${asset}</span></td>
            <td>${s.trades}</td>
            <td class="positive">${s.wins}${s.unredeemed_wins ? ` <span class="pending-color" style="font-size:11px">(${s.unredeemed_wins})</span>` : ''}</td>
            <td class="${s.losses > 0 ? 'negative' : ''}">${s.losses}</td>
            <td>${s.open || 0}</td>
            <td><span class="wr">${wr}%</span></td>
            <td>$${s.volume.toLocaleString('en-US',{maximumFractionDigits:0})}</td>
            <td class="${cls(rpnl)}">${fmt(rpnl)}</td>
            <td class="${cls(epnl)}">${fmt(epnl)}</td>
          </tr>`;
    }
    html += `</table></div>`;
  }

  document.getElementById('content').innerHTML = html;
  document.getElementById('update-time').textContent =
    'Updated ' + new Date(data.updated_at).toLocaleTimeString();
}

async function fetchData() {
  try {
    const res = await fetch('/api/pnl');
    const data = await res.json();
    render(data);
    countdown = REFRESH_INTERVAL;
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

setInterval(() => {
  countdown--;
  document.getElementById('countdown').textContent = countdown;
  if (countdown <= 0) {
    fetchData();
  }
}, 1000);

fetchData();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MTY22 PnL Dashboard")
    print("  Open http://localhost:8888")
    print("  Auto-refreshes every 2 minutes")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=8888, debug=False)
