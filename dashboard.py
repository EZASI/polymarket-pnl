#!/usr/bin/env python3
"""
MTY QUANT TERMINAL — Unified Trading Dashboard
=================================================
Sections:
  1. HOME: Multi-category prediction engine
  2. SWARM: 10-agent trading swarm leaderboard
  3. PERFORMANCE: AI prediction accuracy tracker
  4. TRADERS: Copy signal tracker
  5. PICKS: ML + Copy signal ranked picks
  6. CRYPTO: 30-bot tournament on Polymarket 15-min BTC markets
  7. SPORTS: Pinnacle vs Polymarket arbitrage scanner
"""

import argparse, glob, hashlib, json, os, secrets, subprocess, time, threading
import numpy as np
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from command_center import get_command_data, COMMAND_CENTER_HTML
from weather_station import get_weather_station_data, WEATHER_STATION_HTML
from thermodynamics import get_thermodynamic_data
from zombie_hunter import discover_zombie_markets
from trader_screener import get_trader_screening_data, get_trader_detail, simulate_portfolio

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DASH_PASSWORD_HASH = ""
AUTH_TOKENS = {}
TOKEN_TTL = 86400

# ─── Background Cache for Weather APIs ────────────────────────────
# These 3 endpoints make live API calls (Open-Meteo, METAR, Polymarket)
# that take 10-70s each. Cache them in background, serve instantly.
# On restart, stale disk cache is served immediately while fresh data loads.
CACHE_FILE = Path(__file__).parent / ".weather_cache.json"
WEATHER_CACHE = {
    "weather_station": None,
    "thermodynamic": None,
    "zombie_hunter": None,
    "last_refresh": 0,
    "refreshing": False,
    "lock": threading.Lock(),
}
CACHE_TTL = 90  # seconds between background refreshes

def _save_cache_to_disk():
    """Persist current cache to disk for instant restart recovery."""
    try:
        snapshot = {
            "weather_station": WEATHER_CACHE["weather_station"],
            "thermodynamic": WEATHER_CACHE["thermodynamic"],
            "zombie_hunter": WEATHER_CACHE["zombie_hunter"],
            "last_refresh": WEATHER_CACHE["last_refresh"],
        }
        tmp = str(CACHE_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp, str(CACHE_FILE))
    except Exception as e:
        print(f"[CACHE] Disk save error: {e}")

def _load_cache_from_disk():
    """Load last cached data from disk so first page load is instant."""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                snapshot = json.load(f)
            age = time.time() - snapshot.get("last_refresh", 0)
            # Accept cache up to 10 minutes old (serves stale while refreshing)
            if age < 600:
                WEATHER_CACHE["weather_station"] = snapshot.get("weather_station")
                WEATHER_CACHE["thermodynamic"] = snapshot.get("thermodynamic")
                WEATHER_CACHE["zombie_hunter"] = snapshot.get("zombie_hunter")
                WEATHER_CACHE["last_refresh"] = snapshot.get("last_refresh", 0)
                print(f"[CACHE] Loaded from disk (age={age:.0f}s)")
                return True
            else:
                print(f"[CACHE] Disk cache too old ({age:.0f}s), fetching fresh")
        else:
            print("[CACHE] No disk cache found, fetching fresh")
    except Exception as e:
        print(f"[CACHE] Disk load error: {e}")
    return False

def _refresh_weather_cache():
    """Refresh all weather data in background. Called by timer thread."""
    with WEATHER_CACHE["lock"]:
        if WEATHER_CACHE["refreshing"]:
            return
        WEATHER_CACHE["refreshing"] = True
    try:
        t0 = time.time()
        ws = get_weather_station_data()
        td = get_thermodynamic_data()
        zh = discover_zombie_markets()
        elapsed = time.time() - t0
        now = time.time()
        for d in (ws, td, zh):
            if isinstance(d, dict):
                d["_cached_at"] = now
                d["_fetch_ms"] = int(elapsed * 1000)
        WEATHER_CACHE["weather_station"] = ws
        WEATHER_CACHE["thermodynamic"] = td
        WEATHER_CACHE["zombie_hunter"] = zh
        WEATHER_CACHE["last_refresh"] = now
        _save_cache_to_disk()
        print(f"[CACHE] Weather data refreshed in {elapsed:.1f}s")
    except Exception as e:
        print(f"[CACHE] Refresh error: {e}")
    finally:
        WEATHER_CACHE["refreshing"] = False

def _cache_loop():
    """Background loop that refreshes weather cache every CACHE_TTL seconds."""
    while True:
        try:
            _refresh_weather_cache()
        except Exception as e:
            print(f"[CACHE] Loop error: {e}")
        time.sleep(CACHE_TTL)

def _get_cached(key):
    """Return cached data instantly. Stale data served while background refreshes."""
    data = WEATHER_CACHE.get(key)
    if data is not None:
        age = time.time() - WEATHER_CACHE.get("last_refresh", 0)
        # Return a shallow copy so we don't mutate the cache
        if isinstance(data, dict):
            out = dict(data)
            out["_cache_age_s"] = round(age, 1)
            return out
        return data
    return {"_loading": True, "_message": "Data loading... refresh in a few seconds"}

def hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()
def check_auth(h):
    if not DASH_PASSWORD_HASH: return True
    cookie = h.headers.get("Cookie","")
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("dash_token="):
            token = part.split("=",1)[1]
            if token in AUTH_TOKENS and AUTH_TOKENS[token] > time.time(): return True
    return False
def create_auth_token():
    token = secrets.token_hex(32)
    AUTH_TOKENS[token] = time.time() + TOKEN_TTL
    return token


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/login": self._send_page(LOGIN_HTML); return
        if not check_auth(self):
            self.send_response(302); self.send_header("Location","/login"); self.end_headers(); return
        if p in ("/", "/index.html"): self._send_page(HOME_HTML)
        elif p == "/command-center": self._send_page(COMMAND_CENTER_HTML)
        elif p == "/api/command": self._send_json(get_command_data())
        elif p == "/swarm": self._send_page(SWARM_HTML)
        elif p == "/performance": self._send_page(PERF_HTML)
        elif p == "/crypto": self._send_page(MAIN_HTML)
        elif p == "/sports": self._send_page(SPORTS_HTML)
        elif p == "/traders": self._send_page(TRADERS_HTML)
        elif p == "/picks": self._send_page(PICKS_HTML)
        elif p == "/weather": self._send_page(WEATHER_STATION_HTML)
        elif p == "/api/weather-station": self._send_json(_get_cached("weather_station"))
        elif p == "/api/thermodynamic": self._send_json(_get_cached("thermodynamic"))
        elif p == "/api/zombie-hunter": self._send_json(_get_cached("zombie_hunter"))
        elif p == "/api/swarm": self._send_json(get_swarm_data())
        elif p == "/api/performance": self._send_json(get_performance_data())
        elif p == "/api/predictions": self._send_json(get_all_predictions())
        elif p == "/api/tournament": self._send_json(get_tournament_data())
        elif p == "/api/sports": self._send_json(get_sports_data())
        elif p == "/api/crossplatform": self._send_json(get_crossplatform_data())
        elif p == "/api/traders": self._send_json(get_tracker_data())
        elif p == "/api/copysignals": self._send_json(get_copy_signals_data())
        elif p == "/api/executions": self._send_json(get_execution_data())
        elif p == "/api/smartpicks": self._send_json(get_smart_picks_data())
        elif p == "/api/system": self._send_json(get_system_data())
        elif p == "/api/trader-screening": self._send_json(get_trader_screening_data())
        elif p.startswith("/api/trader-detail"):
            qs = parse_qs(urlparse(self.path).query)
            wallet = qs.get("wallet", [""])[0]
            self._send_json(get_trader_detail(wallet) if wallet else {"error": "wallet required"})
        else: self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path).path
        cl = int(self.headers.get("Content-Length",0))
        body = self.rfile.read(cl).decode() if cl else ""
        if p == "/login": self._handle_login(body)
        elif p == "/api/emergency-stop":
            if not check_auth(self): self._send_json({"error":"unauthorized"}); return
            subprocess.run(["sudo","systemctl","stop","polymarket-tournament"],capture_output=True,timeout=10)
            subprocess.run(["sudo","systemctl","stop","polymarket-bot"],capture_output=True,timeout=10)
            self._send_json({"status":"ALL BOTS STOPPED"})
        elif p == "/api/scan-sports":
            if not check_auth(self): self._send_json({"error":"unauthorized"}); return
            try:
                result = subprocess.run(
                    ["python3","sports_arb_scanner.py","--api-key",os.getenv("ODDS_API_KEY",""),"--min-edge","3"],
                    capture_output=True, text=True, timeout=30, cwd=str(Path(__file__).parent))
                self._send_json({"status":"ok","output":result.stdout[-500:]})
            except Exception as e:
                self._send_json({"status":"error","msg":str(e)})
        elif p == "/api/trader-simulate":
            if not check_auth(self): self._send_json({"error":"unauthorized"}); return
            try:
                wallets = json.loads(body).get("wallets", [])
                self._send_json(simulate_portfolio(wallets))
            except Exception as e:
                self._send_json({"error": str(e)})
        else: self.send_error(404)

    def _handle_login(self, body):
        params = dict(p.split("=",1) for p in body.split("&") if "=" in p)
        pw = params.get("password","").replace("+"," ").replace("%21","!")
        if hash_password(pw) == DASH_PASSWORD_HASH:
            token = create_auth_token()
            self.send_response(302)
            self.send_header("Set-Cookie",f"dash_token={token}; Path=/; Max-Age={TOKEN_TTL}; HttpOnly")
            self.send_header("Location","/"); self.end_headers()
        else:
            self._send_page(LOGIN_HTML.replace("{{ERROR}}",'<p style="color:#ff4055;margin-top:10px">Wrong password</p>'))

    def _send_json(self, data):
        b = json.dumps(data).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",len(b)); self.end_headers(); self.wfile.write(b)

    def _send_page(self, html):
        b = html.encode()
        self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",len(b)); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a): pass


# ── DATA FUNCTIONS ──
def get_tournament_data():
    results = []
    for f in sorted(glob.glob(str(LOG_DIR/"bot_*.jsonl"))):
        trades = []
        try:
            for line in open(f):
                l = line.strip()
                if l: trades.append(json.loads(l))
        except: continue
        if not trades: continue
        name = trades[0].get("bot",Path(f).stem.replace("bot_",""))
        t=len(trades); w=sum(1 for x in trades if x.get("correct")); pnl=sum(x.get("pnl",0) for x in trades)
        pnls=[x.get("pnl",0) for x in trades]
        sh=float(np.mean(pnls)/np.std(pnls)*np.sqrt(96)) if len(pnls)>=2 and np.std(pnls)>0 else 0
        uw=max(len(set(x.get("window_start",i) for i,x in enumerate(trades))),1)
        daily=(pnl/uw)*96 if t>0 else 0
        results.append({"name":name,"trades":t,"wins":w,"losses":t-w,
            "wr":round(w/t*100,1) if t>0 else 0,"pnl":round(pnl,2),
            "avg_pnl":round(pnl/t,2) if t>0 else 0,"sharpe":round(sh,2),
            "daily_est":round(daily,0),"recent":trades[-5:]})
    results.sort(key=lambda x:x["pnl"],reverse=True)
    return results

def get_sports_data():
    # Load latest arb scan
    arb = []
    files = sorted(glob.glob(str(LOG_DIR/"arb_scan_*.json")))
    if files:
        try: arb = json.load(open(files[-1]))
        except: pass

    # Also load live sports tournament signals
    live_files = sorted(glob.glob(str(LOG_DIR/"sports_live_*.jsonl")))
    live_signals = []
    if live_files:
        try:
            for line in open(live_files[-1]):
                l = line.strip()
                if l: live_signals.append(json.loads(l))
        except: pass

    # Combine: arb opportunities + live strategy signals
    combined = arb.copy()

    # Group live signals by game and strategy
    if live_signals:
        games = {}
        for s in live_signals:
            game = s.get("game", "")
            if game not in games:
                games[game] = {
                    "game": game, "home": s.get("home", ""), "away": s.get("away", ""),
                    "ml_prob": s.get("ml_prob", 0), "elo_expected": s.get("elo_expected", 0),
                    "poly_price": s.get("poly_price", 0.5),
                    "strategies": [], "condition_id": s.get("condition_id", ""),
                }
            games[game]["strategies"].append({
                "name": s.get("strategy", ""),
                "side": s.get("side", ""),
                "edge": s.get("edge", 0),
                "confidence": s.get("confidence", 0),
            })

        for g in games.values():
            combined.append({
                "game": g["game"],
                "question": g["game"],
                "pinnacle_prob": g["elo_expected"],
                "polymarket_prob": g["poly_price"],
                "edge": max(s["edge"] for s in g["strategies"]) if g["strategies"] else 0,
                "side": g["strategies"][0]["side"] if g["strategies"] else "",
                "team": g["home"] if g["strategies"][0]["side"] == "HOME" else g["away"],
                "strategies": g["strategies"],
                "n_strategies": len(g["strategies"]),
            })

    combined.sort(key=lambda x: x.get("edge", 0) or 0, reverse=True)
    return combined

def get_tracker_data():
    """Get top trader tracking data."""
    path = LOG_DIR / "trader_tracker.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    return {"traders": [], "time": ""}

def get_copy_signals_data():
    """Get latest copy trading signals."""
    path = LOG_DIR / "copy_signals.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    return {"signals": [], "time": "", "traders_analyzed": 0}

def get_execution_data():
    """Get today's copy executor trades."""
    today = datetime.now().strftime("%Y%m%d")
    path = LOG_DIR / f"copy_trades_{today}.jsonl"
    trades = []
    if path.exists():
        try:
            for line in open(path):
                l = line.strip()
                if l:
                    trades.append(json.loads(l))
        except: pass

    mode = trades[-1].get("mode", "PAPER") if trades else "PAPER"
    return {"trades": trades, "mode": mode, "count": len(trades)}

def get_performance_data():
    """Get AI prediction performance tracking data."""
    path = LOG_DIR / "performance.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    return {"stats": {}, "resolved": [], "daily_pnl": [], "last_check": None}

def get_all_predictions():
    """Get all category predictions from multi_predictor."""
    path = LOG_DIR / "all_predictions.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    return {"predictions": [], "total_predictions": 0, "categories": {}}

def get_smart_picks_data():
    """Get latest smart picks."""
    today = datetime.now().strftime("%Y%m%d")
    path = LOG_DIR / f"smart_picks_{today}.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    # Try yesterday if today's not generated yet
    from datetime import timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    path = LOG_DIR / f"smart_picks_{yesterday}.json"
    if path.exists():
        try: return json.load(open(path))
        except: pass
    return {"picks": [], "n_picks": 0, "strong_picks": 0, "moderate_picks": 0}

def get_crossplatform_data():
    """Get latest cross-platform arbitrage scan."""
    files = sorted(glob.glob(str(LOG_DIR/"crossplatform_*.json")))
    if files:
        try:
            data = json.load(open(files[-1]))
            return data
        except: pass
    return {"opportunities": [], "poly_count": 0, "sx_count": 0, "pin_count": 0}

def get_system_data():
    services = {}
    for svc in ["polymarket-tournament","polymarket-dashboard","polymarket-bot"]:
        try:
            r = subprocess.run(["systemctl","is-active",svc],capture_output=True,text=True,timeout=5)
            services[svc] = r.stdout.strip()
        except: services[svc] = "unknown"
    return {"services":services,"time":datetime.now(timezone.utc).isoformat()}

def get_swarm_data():
    """Get swarm agent data: leaderboard, risk report, individual signals."""
    data = {"leaderboard": [], "risk_events": [], "agents": [], "portfolio": {}}

    # Load leaderboard
    lb_path = LOG_DIR / "leaderboard.json"
    if lb_path.exists():
        try: data["leaderboard"] = json.load(open(lb_path))
        except: pass

    # Load risk report
    rr_path = LOG_DIR / "risk_report.json"
    if rr_path.exists():
        try: data["risk_events"] = json.load(open(rr_path))
        except: pass

    # Load individual agent signals
    agents = []
    for f in sorted(glob.glob(str(LOG_DIR / "signals_*.json"))):
        try:
            agent_data = json.load(open(f))
            agent_name = Path(f).stem.replace("signals_", "")
            agents.append({"name": agent_name, "signals": agent_data if isinstance(agent_data, list) else [agent_data]})
        except:
            continue
    data["agents"] = agents

    # Compute portfolio-wide stats from leaderboard
    lb = data["leaderboard"]
    # leaderboard.json is a dict with "agents" list inside
    agents_list = lb.get("agents", []) if isinstance(lb, dict) else (lb if isinstance(lb, list) else [])
    if agents_list:
        total_pnl = sum(a.get("pnl", 0) for a in agents_list)
        total_capital = sum(a.get("capital", 0) for a in agents_list)
        open_positions = sum(a.get("open_positions", 0) for a in agents_list)
        data["total_agents"] = len(agents_list)
        data["total_pnl"] = round(total_pnl, 2)
        data["total_capital"] = round(total_capital, 2)
        data["total_open_positions"] = open_positions
        data["portfolio"] = {
            "total_pnl": round(total_pnl, 2),
            "total_capital": round(total_capital, 2),
            "open_positions": open_positions,
            "n_agents": len(agents_list),
        }

    return data


# ════════════════════════════════════════════════════════════════════
# UNIFIED CSS — Shared design tokens used by all pages
# ════════════════════════════════════════════════════════════════════

UNIFIED_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0f;color:#e4e5ea;min-height:100vh}
a{color:inherit;text-decoration:none}

/* ── Header / Nav ── */
.top-bar{background:linear-gradient(180deg,#13141c 0%,#0f1017 100%);border-bottom:1px solid #1f2133;padding:14px 24px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.brand{display:flex;align-items:center;gap:10px}
.brand-logo{font-size:18px;font-weight:800;letter-spacing:4px;color:#00d4aa}
.brand-sep{width:1px;height:20px;background:#1f2133}
.brand-sub{font-size:11px;color:#5c6478;letter-spacing:2px}
.nav{display:flex;gap:4px;flex-wrap:wrap}
.nav a{font-size:11px;padding:7px 16px;border-radius:8px;color:#5c6478;transition:.2s;font-weight:500}
.nav a:hover{color:#e4e5ea;background:rgba(255,255,255,.04)}
.nav a.on{background:rgba(0,212,170,.12);color:#00d4aa;font-weight:600}

/* ── Layout ── */
.wrap{max-width:1200px;margin:0 auto;padding:20px 24px}

/* ── Stat Cards Row ── */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}
.stat{background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:18px;text-align:center;transition:border-color .2s}
.stat:hover{border-color:#282b3a}
.stat-label{font-size:10px;color:#5c6478;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;font-weight:500}
.stat-val{font-size:24px;font-weight:700}
.stat-sub{font-size:9px;color:#3a3f4c;margin-top:4px}

/* ── Sections / Cards ── */
.section{background:#12131a;border:1px solid #1f2133;border-radius:12px;margin-bottom:16px;overflow:hidden}
.sec-hdr{padding:14px 18px;border-bottom:1px solid #1f2133;font-size:13px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
.sec-hdr span.sec-meta{font-size:10px;color:#5c6478;font-weight:400}
.sec-body{padding:16px 18px}

/* ── Tables ── */
table{width:100%;border-collapse:collapse}
th{text-align:left;padding:8px 10px;font-size:10px;color:#5c6478;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1f2133}
td{padding:8px 10px;font-size:12px;border-bottom:1px solid #111219}
tr:hover{background:#14151e}

/* ── Utility Colors ── */
.g{color:#00d4aa}.r{color:#ff4055}.y{color:#ffd43b}.b{color:#4da6ff}.d{color:#5c6478}.p{color:#a78bfa}.o{color:#ff8c42}.c{color:#66d9e8}

/* ── Tags / Badges ── */
.tag{font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600}
.tag-win{background:rgba(0,212,170,.12);color:#00d4aa}
.tag-loss{background:rgba(255,64,85,.1);color:#ff4055}
.tag-strong{background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3)}
.tag-moderate{background:rgba(255,212,59,.1);color:#ffd43b;border:1px solid rgba(255,212,59,.2)}
.tag-weak{background:rgba(92,100,120,.08);color:#5c6478;border:1px solid #1f2133}

/* ── Empty State ── */
.empty{text-align:center;padding:40px 20px;color:#5c6478;font-size:12px}

/* ── Footer ── */
.foot{text-align:center;padding:24px;font-size:10px;color:#2a2d3a;letter-spacing:1px;border-top:1px solid #111320;margin-top:32px}
.foot a{color:#5c6478}

/* ── Bottom Bar ── */
.bottom-bar{position:fixed;bottom:0;left:0;right:0;background:#12131a;border-top:1px solid #1f2133;padding:10px 24px;display:flex;justify-content:space-between;align-items:center;z-index:100}
.bottom-bar .bar-info{font-size:10px;color:#5c6478}
.btn{padding:7px 18px;border-radius:8px;border:1px solid #1f2133;background:#12131a;color:#e4e5ea;font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:11px;cursor:pointer;font-weight:500;transition:.2s}
.btn:hover{border-color:#00d4aa;color:#00d4aa}
.btn-green{background:#00d4aa;color:#0a0b0f;border:none;font-weight:700}
.btn-green:hover{background:#00eebb}
.btn-red{background:transparent;border:1px solid #ff4055;color:#ff4055}
.btn-red:hover{background:#ff4055;color:white}

/* ── Live Pulse ── */
.live-dot{width:6px;height:6px;border-radius:50%;background:#00d4aa;display:inline-block;margin-right:6px;animation:livepulse 2s infinite}
@keyframes livepulse{0%,100%{opacity:1;box-shadow:0 0 4px rgba(0,212,170,.5)}50%{opacity:.3;box-shadow:none}}

/* ── Responsive ── */
@media(max-width:768px){
.top-bar{padding:12px 16px;flex-direction:column;gap:10px}
.stats{grid-template-columns:repeat(2,1fr)}
.wrap{padding:14px 12px}
}
"""

def _nav_html(active):
    """Generate consistent nav bar. active is the page key."""
    links = [
        ("Dashboard", "/", "home"),
        ("Command Center", "/command-center", "command"),
        ("Swarm", "/swarm", "swarm"),
        ("Weather", "/weather", "weather"),
        ("Performance", "/performance", "perf"),
        ("Traders", "/traders", "traders"),
        ("Picks", "/picks", "picks"),
    ]
    parts = []
    for label, href, key in links:
        cls = ' class="on"' if key == active else ''
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    return "".join(parts)


# ════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════════════════════════

LOGIN_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0b0f;color:#e4e5ea;display:flex;align-items:center;justify-content:center;min-height:100vh}
.box{background:#12131a;border:1px solid #1f2133;border-radius:16px;padding:48px 40px;width:360px;text-align:center}
h2{font-size:1.1rem;letter-spacing:4px;color:#00d4aa;margin-bottom:6px;font-weight:800}
.sub{font-size:.65rem;color:#5c6478;letter-spacing:2px;margin-bottom:32px}
input{width:100%;padding:14px;border:1px solid #1f2133;border-radius:10px;background:#0a0b0f;color:#e4e5ea;font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:.9rem;margin-bottom:16px;text-align:center}
input:focus{outline:none;border-color:#00d4aa}
button{width:100%;padding:14px;border:none;border-radius:10px;background:#00d4aa;color:#0a0b0f;font-weight:700;font-size:.85rem;letter-spacing:2px;cursor:pointer;font-family:-apple-system,BlinkMacSystemFont,sans-serif;transition:.2s}
button:hover{background:#00eebb}
</style></head><body>
<div class="box"><h2>MTY</h2><div class="sub">QUANTITATIVE TRADING TERMINAL</div>
<form method="POST" action="/login">
<input type="password" name="password" placeholder="ACCESS CODE" autofocus>
<button type="submit">AUTHENTICATE</button></form>{{ERROR}}</div></body></html>""".replace("{{ERROR}}","")


# ════════════════════════════════════════════════════════════════════
# HOME PAGE — Multi-Category Prediction Engine
# ════════════════════════════════════════════════════════════════════

HOME_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Predictions</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Home-specific ── */
.hero{background:linear-gradient(135deg,#0e1420 0%,#151830 50%,#0e1420 100%);border:1px solid #1f2133;border-radius:16px;padding:24px;margin-bottom:24px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-50%;right:-20%;width:300px;height:300px;background:radial-gradient(circle,rgba(0,212,170,.06) 0%,transparent 70%);pointer-events:none}
.hero-title{font-size:13px;color:#5c6478;letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;font-weight:600}
.hero-title span{color:#00d4aa}
.best-bets{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.best-bet{background:rgba(0,0,0,.3);border:1px solid #1f2133;border-radius:12px;padding:16px;transition:all .25s}
.best-bet:hover{border-color:#00d4aa44;transform:translateY(-1px)}
.best-bet.hot{border-color:rgba(0,212,170,.3);background:rgba(0,212,170,.03)}
.bb-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.bb-cat{font-size:9px;padding:3px 10px;border-radius:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.bb-edge{font-size:14px;font-weight:800;color:#00d4aa}
.bb-title{font-size:13px;font-weight:600;line-height:1.5;margin-bottom:8px;color:#e4e5ea}
.bb-pick{font-size:15px;font-weight:800;color:#00d4aa;margin-bottom:6px}
.bb-meta{display:flex;gap:12px;font-size:10px;color:#5c6478;flex-wrap:wrap}
.bb-meta b{color:#b0b5c3}
.cats{display:flex;gap:6px;margin-bottom:16px;overflow-x:auto;padding-bottom:6px;-webkit-overflow-scrolling:touch}
.cats::-webkit-scrollbar{height:2px}
.cats::-webkit-scrollbar-thumb{background:#1f2133;border-radius:2px}
.cat-btn{font-size:11px;padding:8px 18px;border-radius:24px;border:1px solid #1f2133;background:transparent;color:#5c6478;cursor:pointer;white-space:nowrap;transition:all .2s;font-family:inherit;font-weight:500;display:flex;align-items:center;gap:6px}
.cat-btn:hover{border-color:#333;color:#e4e5ea;background:rgba(255,255,255,.02)}
.cat-btn.active{background:rgba(0,212,170,.12);color:#00d4aa;font-weight:600;border-color:rgba(0,212,170,.3)}
.cat-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.cat-count{font-size:9px;opacity:.6;font-weight:400}
.view-toggle{display:flex;gap:2px;background:#12131a;border:1px solid #1f2133;border-radius:8px;padding:2px}
.view-btn{padding:6px 12px;border:none;background:transparent;color:#5c6478;cursor:pointer;font-size:10px;font-family:inherit;border-radius:6px;transition:.2s;font-weight:500}
.view-btn.active{background:rgba(0,212,170,.12);color:#00d4aa}
.cards{display:grid;gap:8px}
.card{background:#12131a;border:1px solid #1a1d2a;border-radius:12px;padding:16px 18px;transition:all .2s;cursor:default}
.card:hover{border-color:#252840;background:#14151e}
.card.has-edge{border-left:3px solid #00d4aa}
.card.has-weak-edge{border-left:3px solid #ffd43b}
.card-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;gap:12px}
.card-title{font-size:13px;font-weight:600;line-height:1.45;flex:1;color:#d0d4de}
.card-badge{font-size:10px;padding:4px 12px;border-radius:8px;font-weight:700;white-space:nowrap;flex-shrink:0}
.badge-hot{background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3)}
.badge-warm{background:rgba(255,212,59,.1);color:#ffd43b;border:1px solid rgba(255,212,59,.2)}
.badge-neutral{background:rgba(92,100,120,.08);color:#4a4f5c;border:1px solid #1a1d2a}
.card-body{display:flex;align-items:center;gap:14px;margin-bottom:8px}
.card-pick{font-size:15px;font-weight:800;color:#00d4aa}
.card-price{display:flex;align-items:center;gap:6px}
.price-bar{width:80px;height:6px;background:#1a1d2a;border-radius:3px;overflow:hidden;position:relative}
.price-fill{height:100%;border-radius:3px;background:#00d4aa;transition:width .5s}
.price-val{font-size:11px;font-weight:600;color:#b0b5c3}
.card-row{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:#5c6478}
.card-row b{color:#b0b5c3;font-weight:600}
.card-cat{font-size:9px;padding:3px 10px;border-radius:6px;text-transform:uppercase;letter-spacing:1px;font-weight:600}
.signal-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle}
.signal-strong{background:#00d4aa;box-shadow:0 0 6px rgba(0,212,170,.4)}
.signal-moderate{background:#ffd43b;box-shadow:0 0 6px rgba(255,212,59,.3)}
.signal-none{background:#2a2d3a}
.warn{font-size:10px;color:#ff8c42;margin-top:6px;font-weight:500;padding:4px 8px;background:rgba(255,140,66,.06);border-radius:4px}
.section-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding:0 2px}
.section-title{font-size:14px;font-weight:700;color:#e4e5ea}
.section-title span{color:#5c6478;font-weight:400;font-size:12px;margin-left:8px}

@media(max-width:768px){
.best-bets{grid-template-columns:1fr}
.hero{padding:16px}
.card-body{flex-direction:column;align-items:flex-start;gap:8px}
}
</style></head><body>
<div class="top-bar">
<div class="brand">
<div class="brand-logo">MTY</div>
<div class="brand-sep"></div>
<div class="brand-sub">PREDICTION ENGINE</div>
</div>
<div class="nav">
""" + _nav_html("home") + r"""
</div>
</div>
<div class="wrap">

<!-- Stats Overview -->
<div class="stats">
<div class="stat"><div class="stat-label"><span class="live-dot"></span>Markets Live</div><div class="stat-val" id="s-markets" style="color:#4da6ff">--</div><div class="stat-sub">Across Polymarket</div></div>
<div class="stat"><div class="stat-label">Tracked</div><div class="stat-val" id="s-preds" style="color:#b0b5c3">--</div><div class="stat-sub">All categories</div></div>
<div class="stat"><div class="stat-label">With Signal</div><div class="stat-val" id="s-signals" style="color:#00d4aa">--</div><div class="stat-sub">Copy trader edge</div></div>
<div class="stat"><div class="stat-label">Best Edge</div><div class="stat-val" id="s-edge" style="color:#ffd43b">--</div><div class="stat-sub">Top opportunity</div></div>
<div class="stat"><div class="stat-label">Last Scan</div><div class="stat-val" id="s-time" style="color:#5c6478;font-size:13px">--</div><div class="stat-sub">Updates every 30m</div></div>
</div>

<!-- Best Bets Section -->
<div class="hero" id="hero-section" style="display:none">
<div class="hero-title"><span>Best Bets</span> — Highest edge opportunities right now</div>
<div class="best-bets" id="best-bets"></div>
</div>

<!-- Category Tabs + View Toggle -->
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px">
<div class="section-title">All Markets <span id="showing-count"></span></div>
<div class="view-toggle">
<button class="view-btn active" onclick="setView('edge')" id="vb-edge">Edge First</button>
<button class="view-btn" onclick="setView('recent')" id="vb-recent">Recent</button>
<button class="view-btn" onclick="setView('volume')" id="vb-volume">Volume</button>
</div>
</div>

<div class="cats" id="cat-tabs"></div>

<div class="cards" id="card-list">
<div class="empty"><h3>Loading predictions...</h3><p>Scanning Polymarket markets across all categories</p></div>
</div>

</div>
<div class="foot">MTY Prediction Engine — ML + Copy Signals + Multi-API Data — Auto-refreshes every 2 minutes</div>

<script>
const CAT_COLORS={weather:'#4da6ff',sports:'#00d4aa',politics:'#a78bfa',crypto:'#ffd43b',
entertainment:'#ff8c42',economics:'#66d9e8',science_tech:'#ff6b6b',olympics:'#ffd43b',
world:'#4dabf7',esports:'#b197fc',pop_culture:'#ff6b9d',other:'#5c6478'};
const CAT_LABELS={weather:'Weather',sports:'Sports',politics:'Politics',crypto:'Crypto',
entertainment:'Entertainment',economics:'Economics',science_tech:'Sci & Tech',olympics:'Olympics',
world:'World',esports:'Esports',pop_culture:'Pop Culture',other:'Other'};

let allPreds=[];
let activeCat='all';
let sortMode='edge';

function load(){
fetch('/api/predictions').then(r=>r.json()).then(d=>{
document.getElementById('s-markets').textContent=(d.total_markets||0).toLocaleString();
document.getElementById('s-preds').textContent=(d.total_predictions||0).toLocaleString();
let signalCount=(d.with_signals||0);
document.getElementById('s-signals').textContent=signalCount;
let edgePreds=(d.predictions||[]).filter(p=>Math.abs(p.edge||0)>0.03);
let bestEdge=edgePreds.length?Math.max(...edgePreds.map(p=>Math.abs(p.edge||0))):0;
document.getElementById('s-edge').textContent=bestEdge?(bestEdge*100).toFixed(1)+'%':'--';
if(d.generated){let dt=new Date(d.generated);document.getElementById('s-time').textContent=dt.toLocaleTimeString();}

allPreds=d.predictions||[];
let cats=d.categories||{};
renderTabs(cats);
renderBestBets(edgePreds);
renderCards();
}).catch(e=>{document.getElementById('card-list').innerHTML='<div class="empty"><h3>No predictions yet</h3><p>The multi-category predictor runs every 30 minutes.<br>It scans 1000+ Polymarket markets for edge opportunities.</p></div>';});
}

function renderBestBets(edgePreds){
let hero=document.getElementById('hero-section');
let box=document.getElementById('best-bets');
if(!edgePreds.length){hero.style.display='none';return;}
hero.style.display='block';
let top=edgePreds.sort((a,b)=>Math.abs(b.edge||0)-Math.abs(a.edge||0)).slice(0,4);
box.innerHTML=top.map((p,i)=>{
let color=CAT_COLORS[p.category]||'#5c6478';
let edge=p.edge||0;
let hot=Math.abs(edge)>0.1;
return '<div class="best-bet'+(hot?' hot':'')+'">'+
'<div class="bb-top">'+
'<span class="bb-cat" style="background:'+color+'18;color:'+color+';border:1px solid '+color+'33">'+(CAT_LABELS[p.category]||p.category)+'</span>'+
'<span class="bb-edge">'+(edge>0?'+':'')+((edge*100).toFixed(1))+'% edge</span>'+
'</div>'+
'<div class="bb-title">'+p.title+'</div>'+
'<div class="bb-pick">'+p.pick+'</div>'+
'<div class="bb-meta">'+
'<span>Our prob: <b>'+(p.our_prob*100).toFixed(0)+'%</b></span>'+
'<span>Market: <b>'+(p.market_price*100).toFixed(0)+'c</b></span>'+
(p.source?'<span><b>'+p.source.substring(0,30)+'</b></span>':'')+
'</div></div>';
}).join('');
}

function renderTabs(cats){
let predCounts={};
allPreds.forEach(p=>{predCounts[p.category]=(predCounts[p.category]||0)+1;});
let html='<button class="cat-btn active" onclick="filterCat(\'all\')"><span class="cat-dot" style="background:#e4e5ea"></span>All<span class="cat-count">'+allPreds.length+'</span></button>';
let sorted=Object.entries(cats).sort((a,b)=>b[1]-a[1]);
for(let[cat,count]of sorted){
let label=CAT_LABELS[cat]||cat;
let color=CAT_COLORS[cat]||'#5c6478';
let pCount=predCounts[cat]||0;
html+='<button class="cat-btn" data-cat="'+cat+'" onclick="filterCat(\''+cat+'\')"><span class="cat-dot" style="background:'+color+'"></span>'+label+'<span class="cat-count">'+pCount+'</span></button>';
}
document.getElementById('cat-tabs').innerHTML=html;
}

function filterCat(cat){
activeCat=cat;
document.querySelectorAll('.cat-btn').forEach(b=>{
b.classList.toggle('active',cat==='all'?!b.dataset.cat:b.dataset.cat===cat);
});
renderCards();
}

function setView(mode){
sortMode=mode;
document.querySelectorAll('.view-btn').forEach(b=>b.classList.remove('active'));
document.getElementById('vb-'+mode).classList.add('active');
renderCards();
}

function renderCards(){
let filtered=activeCat==='all'?allPreds:allPreds.filter(p=>p.category===activeCat);

if(sortMode==='edge'){
filtered.sort((a,b)=>Math.abs(b.edge||0)-Math.abs(a.edge||0));
}else if(sortMode==='volume'){
filtered.sort((a,b)=>(parseFloat((b.details||{}).volume||0))-(parseFloat((a.details||{}).volume||0)));
}

document.getElementById('showing-count').textContent='('+filtered.length+' markets)';

if(!filtered.length){
document.getElementById('card-list').innerHTML='<div class="empty"><h3>No markets in this category</h3><p>Try selecting a different category</p></div>';
return;
}

// Limit display to 50 for performance
let display=filtered.slice(0,50);
let html='';

display.forEach((p,i)=>{
let edge=p.edge||0;
let absEdge=Math.abs(edge);
let conf=p.confidence||0;
let color=CAT_COLORS[p.category]||'#5c6478';
let hasEdge=absEdge>0.03;
let hasWeakEdge=absEdge>0.01&&absEdge<=0.03;

// Badge
let badgeClass=absEdge>0.1?'badge-hot':absEdge>0.03?'badge-warm':'badge-neutral';
let edgePct=(edge*100).toFixed(1);
let edgeSign=edge>0?'+':'';
let edgeLabel=absEdge>0.03?edgeSign+edgePct+'%':'--';

// Signal indicator
let signalClass='signal-none';
let signalLabel='No signal';
if(absEdge>0.1){signalClass='signal-strong';signalLabel='Strong edge';}
else if(absEdge>0.03){signalClass='signal-moderate';signalLabel='Edge detected';}

// Price as percentage
let pricePct=(p.market_price*100).toFixed(0);

let cardClass='card';
if(hasEdge)cardClass+=' has-edge';
else if(hasWeakEdge)cardClass+=' has-weak-edge';

html+='<div class="'+cardClass+'">';
html+='<div class="card-top">';
html+='<div class="card-title">'+p.title+'</div>';
if(hasEdge)html+='<span class="card-badge '+badgeClass+'">'+edgeLabel+'</span>';
html+='</div>';

html+='<div class="card-body">';
html+='<div class="card-pick">'+p.pick+'</div>';
html+='<div class="card-price">';
html+='<div class="price-bar"><div class="price-fill" style="width:'+(p.market_price*100)+'%;background:'+color+'"></div></div>';
html+='<span class="price-val">'+pricePct+'c</span>';
html+='</div>';
if(hasEdge){
html+='<div style="margin-left:auto;display:flex;align-items:center;gap:4px"><span class="signal-dot '+signalClass+'"></span><span style="font-size:10px;color:#5c6478">'+signalLabel+'</span></div>';
}
html+='</div>';

html+='<div class="card-row">';
html+='<span class="card-cat" style="background:'+color+'12;color:'+color+';border:1px solid '+color+'22">'+(CAT_LABELS[p.category]||p.category)+'</span>';
if(hasEdge){
html+='<span>Our: <b>'+(p.our_prob*100).toFixed(0)+'%</b></span>';
html+='<span>Conf: <b>'+(conf*100).toFixed(0)+'%</b></span>';
}
if(p.source&&p.source!=='Market consensus')html+='<span>'+p.source.substring(0,35)+'</span>';
let det=p.details||{};
if(det.volume&&parseFloat(det.volume)>0)html+='<span>Vol: <b>$'+(parseFloat(det.volume)/1000).toFixed(0)+'k</b></span>';
if(det.n_traders)html+='<span>'+det.n_traders+' traders, '+(det.avg_wr||0).toFixed(0)+'% WR</span>';
html+='</div>';

if(det.ml_caution)html+='<div class="warn">'+det.ml_caution+'</div>';
html+='</div>';
});

if(filtered.length>50){
html+='<div class="empty"><p>Showing top 50 of '+filtered.length+' markets. Use category filters to narrow down.</p></div>';
}

document.getElementById('card-list').innerHTML=html;
}

load();
setInterval(load,120000);
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# SWARM PAGE — 10-Agent Trading Swarm
# ════════════════════════════════════════════════════════════════════

SWARM_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Swarm</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Swarm-specific ── */
.agent-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin-top:16px}
.agent-card{background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:16px;transition:border-color .2s}
.agent-card:hover{border-color:#282b3a}
.agent-name{font-size:13px;font-weight:700;color:#ffd43b;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.agent-pnl{font-size:16px;font-weight:800}
.agent-signals{margin-top:10px;border-top:1px solid #1f2133;padding-top:10px}
.signal-row{display:flex;justify-content:space-between;font-size:11px;padding:4px 0;border-bottom:1px solid #111219}
.signal-row:last-child{border-bottom:none}
.risk-item{padding:12px 18px;border-bottom:1px solid #111219;display:flex;justify-content:space-between;align-items:center;font-size:12px}
.risk-item:last-child{border-bottom:none}
.risk-level{font-size:9px;padding:3px 10px;border-radius:6px;font-weight:700;text-transform:uppercase}
.risk-high{background:rgba(255,64,85,.12);color:#ff4055}
.risk-medium{background:rgba(255,212,59,.1);color:#ffd43b}
.risk-low{background:rgba(0,212,170,.1);color:#00d4aa}
</style></head><body>
<div class="top-bar">
<div class="brand">
<div class="brand-logo">MTY</div>
<div class="brand-sep"></div>
<div class="brand-sub">TRADING SWARM</div>
</div>
<div class="nav">
""" + _nav_html("swarm") + r"""
</div>
</div>
<div class="wrap">

<!-- Portfolio Stats -->
<div class="stats">
<div class="stat"><div class="stat-label"><span class="live-dot"></span>Agents</div><div class="stat-val b" id="s-agents">--</div><div class="stat-sub">Active agents</div></div>
<div class="stat"><div class="stat-label">Total P&L</div><div class="stat-val" id="s-pnl" style="color:#00d4aa">--</div><div class="stat-sub">All agents combined</div></div>
<div class="stat"><div class="stat-label">Capital Deployed</div><div class="stat-val y" id="s-capital">--</div><div class="stat-sub">Across all agents</div></div>
<div class="stat"><div class="stat-label">Open Positions</div><div class="stat-val" id="s-positions" style="color:#a78bfa">--</div><div class="stat-sub">Live trades</div></div>
<div class="stat"><div class="stat-label">Risk Events</div><div class="stat-val" id="s-risks" style="color:#ff4055">--</div><div class="stat-sub">Warnings active</div></div>
</div>

<!-- Leaderboard -->
<div class="section">
<div class="sec-hdr">Agent Leaderboard <span class="sec-meta" id="lb-meta">Ranked by P&L</span></div>
<div class="sec-body" style="padding:0">
<table>
<thead><tr><th>#</th><th>Agent</th><th>Type</th><th>Trades</th><th>Win Rate</th><th>P&L</th><th>Capital</th><th>Open</th></tr></thead>
<tbody id="lb-table"><tr><td colspan="8" class="d" style="text-align:center;padding:30px">Loading swarm data...</td></tr></tbody>
</table>
</div>
</div>

<!-- Risk Events -->
<div class="section">
<div class="sec-hdr">Risk Events <span class="sec-meta" id="risk-meta">--</span></div>
<div id="risk-list"><div class="empty">No risk events</div></div>
</div>

<!-- Agent Cards -->
<div class="section">
<div class="sec-hdr">Agent Signals <span class="sec-meta">Recent signals from each agent</span></div>
</div>
<div class="agent-grid" id="agent-grid">
<div class="empty">Loading agent data...</div>
</div>

</div>
<div class="foot">MTY Trading Swarm — 10 Agents — Auto-refreshes every 15 seconds</div>

<script>
async function refresh(){
try{
const r=await fetch('/api/swarm');const d=await r.json();
const lbRaw=d.leaderboard||{};
const lb=lbRaw.agents||lbRaw||[];
const risks=(d.risk_events||{}).risk_events||d.risk_events||[];
const agents=d.agents||[];
const port=d.portfolio||{};

// Portfolio stats
document.getElementById('s-agents').textContent=port.n_agents||lb.length||'--';
let tpnl=port.total_pnl||0;
document.getElementById('s-pnl').textContent=tpnl?'$'+(tpnl>=0?'+':'')+tpnl.toLocaleString(undefined,{maximumFractionDigits:0}):'$0';
document.getElementById('s-pnl').style.color=tpnl>=0?'#00d4aa':'#ff4055';
document.getElementById('s-capital').textContent=port.total_capital?'$'+port.total_capital.toLocaleString(undefined,{maximumFractionDigits:0}):'--';
document.getElementById('s-positions').textContent=port.open_positions||0;
document.getElementById('s-risks').textContent=risks.length||0;
document.getElementById('s-risks').style.color=risks.length>0?'#ff4055':'#00d4aa';

// Leaderboard table
const tb=document.getElementById('lb-table');
if(!lb.length){
tb.innerHTML='<tr><td colspan="8" class="d" style="text-align:center;padding:30px">No agents reporting yet. Swarm will populate when agents start trading.</td></tr>';
} else {
let sorted=[...lb].sort((a,b)=>(b.pnl||0)-(a.pnl||0));
document.getElementById('lb-meta').textContent=sorted.length+' agents ranked by P&L';
tb.innerHTML=sorted.map((a,i)=>{
let pc=(a.pnl||0)>=0?'g':'r';
let wr=a.win_rate||0;
let wc=wr>=55?'g':(wr<45?'r':'');
return '<tr>'+
'<td style="font-weight:800;color:#ffd43b">'+(i+1)+'</td>'+
'<td style="font-weight:600">'+( a.name||a.agent||'Agent '+(i+1))+'</td>'+
'<td class="d">'+(a.type||a.strategy||'--')+'</td>'+
'<td>'+(a.trades||0)+'</td>'+
'<td class="'+wc+'" style="font-weight:600">'+wr.toFixed(1)+'%</td>'+
'<td class="'+pc+'" style="font-weight:700">$'+(a.pnl>=0?'+':'')+( a.pnl||0).toFixed(0)+'</td>'+
'<td>$'+(a.capital||0).toLocaleString(undefined,{maximumFractionDigits:0})+'</td>'+
'<td class="p">'+(a.open_positions||0)+'</td>'+
'</tr>';
}).join('');
}

// Risk events
const rbox=document.getElementById('risk-list');
document.getElementById('risk-meta').textContent=risks.length+' events';
if(!risks.length){
rbox.innerHTML='<div class="empty">No risk events. All systems nominal.</div>';
} else {
rbox.innerHTML=risks.map(r=>{
let level=r.level||r.severity||'low';
let lvlCls='risk-'+level.toLowerCase();
return '<div class="risk-item">'+
'<div><span style="font-weight:600">'+(r.agent||r.source||'System')+'</span> <span class="d" style="margin-left:8px">'+(r.message||r.event||'')+'</span></div>'+
'<div><span class="risk-level '+lvlCls+'">'+level.toUpperCase()+'</span> <span class="d" style="margin-left:8px;font-size:10px">'+(r.time||r.timestamp||'')+'</span></div>'+
'</div>';
}).join('');
}

// Agent cards
const agrid=document.getElementById('agent-grid');
if(!agents.length){
agrid.innerHTML='<div class="empty">No agent signal files found. Agents write to logs/signals_*.json</div>';
} else {
agrid.innerHTML=agents.map(a=>{
let sigs=a.signals||[];
let html='<div class="agent-card">';
html+='<div class="agent-name"><span>'+a.name+'</span></div>';
if(sigs.length){
html+='<div class="agent-signals">';
sigs.slice(-5).reverse().forEach(s=>{
let side=s.side||s.direction||'--';
let sideCls=side.toUpperCase()==='BUY'||side.toUpperCase()==='YES'?'g':'r';
html+='<div class="signal-row">';
html+='<span style="flex:1">'+(s.market||s.title||'--')+'</span>';
html+='<span class="'+sideCls+'" style="font-weight:700;width:50px;text-align:center">'+side+'</span>';
html+='<span class="d" style="width:60px;text-align:right">'+(s.confidence?(s.confidence*100).toFixed(0)+'%':(s.edge?(s.edge*100).toFixed(1)+'%':'--'))+'</span>';
html+='</div>';
});
html+='</div>';
} else {
html+='<div class="d" style="font-size:11px;padding:8px 0">No signals yet</div>';
}
html+='</div>';
return html;
}).join('');
}

}catch(e){console.log('swarm error',e)}
}
refresh();setInterval(refresh,15000);
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# PERFORMANCE PAGE — AI Prediction Tracker
# ════════════════════════════════════════════════════════════════════

PERF_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Performance</title>
<style>
""" + UNIFIED_CSS + r"""
.pnl-chart{display:flex;align-items:end;gap:3px;height:80px;padding:8px 0}
.pnl-bar{flex:1;min-width:8px;max-width:40px;border-radius:3px 3px 0 0;transition:height .3s}
</style></head><body>
<div class="top-bar">
<div class="brand"><div class="brand-logo">MTY</div><div class="brand-sep"></div><div class="brand-sub">PERFORMANCE</div></div>
<div class="nav">
""" + _nav_html("perf") + r"""
</div>
</div>
<div class="wrap">

<div class="stats">
<div class="stat"><div class="stat-label">Predictions Resolved</div><div class="stat-val b" id="s-total">--</div></div>
<div class="stat"><div class="stat-label">Accuracy</div><div class="stat-val" id="s-acc" style="color:#00d4aa">--</div></div>
<div class="stat"><div class="stat-label">Simulated P&L</div><div class="stat-val" id="s-pnl">--</div><div class="stat-sub">$50k bankroll, Kelly sizing</div></div>
<div class="stat"><div class="stat-label">ROI</div><div class="stat-val" id="s-roi">--</div></div>
<div class="stat"><div class="stat-label">Win Streak</div><div class="stat-val y" id="s-streak">--</div></div>
<div class="stat"><div class="stat-label">Last Check</div><div class="stat-val d" id="s-time" style="font-size:13px">--</div></div>
</div>

<div class="section">
<div class="sec-hdr">Category Accuracy <span class="sec-meta">Which categories perform best</span></div>
<div class="sec-body">
<table><thead><tr><th>Category</th><th>Accuracy</th><th>W/L</th><th>P&L</th><th>Avg P&L</th></tr></thead>
<tbody id="cat-table"><tr><td colspan="5" class="d" style="text-align:center">Loading...</td></tr></tbody>
</table></div></div>

<div class="section">
<div class="sec-hdr">Edge Bucket Analysis <span class="sec-meta">Higher edge = better accuracy?</span></div>
<div class="sec-body">
<table><thead><tr><th>Edge Range</th><th>Accuracy</th><th>Trades</th><th>P&L</th></tr></thead>
<tbody id="edge-table"><tr><td colspan="4" class="d" style="text-align:center">Loading...</td></tr></tbody>
</table></div></div>

<div class="section">
<div class="sec-hdr">Recent Resolved Predictions <span class="sec-meta" id="res-count">--</span></div>
<div class="sec-body">
<table><thead><tr><th>Market</th><th>Category</th><th>Our Pick</th><th>Result</th><th>Edge</th><th>P&L</th></tr></thead>
<tbody id="res-table"><tr><td colspan="6" class="d" style="text-align:center">Loading...</td></tr></tbody>
</table></div></div>

<div class="section">
<div class="sec-hdr">Automation Stack <span class="sec-meta">Tools powering our predictions</span></div>
<div class="sec-body" style="font-size:12px;line-height:1.8">
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">
<div>
<div style="font-weight:700;color:#00d4aa;margin-bottom:6px">Our System (Running Now)</div>
<div><span class="d">Signal Scanner:</span> copy_signal_bot.py <span class="d">(every 10m)</span></div>
<div><span class="d">Multi-Predictor:</span> multi_predictor.py <span class="d">(every 30m)</span></div>
<div><span class="d">Smart Picks:</span> ML + Copy signals <span class="d">(every 30m)</span></div>
<div><span class="d">Executor:</span> copy_executor.py <span class="d">(systemd, paper mode)</span></div>
<div><span class="d">Tracker:</span> performance_tracker.py <span class="d">(hourly)</span></div>
</div>
<div>
<div style="font-weight:700;color:#4da6ff;margin-bottom:6px">Official Polymarket Tools</div>
<div><a href="https://docs.polymarket.com" style="color:#4da6ff">docs.polymarket.com</a> <span class="d">CLOB API docs</span></div>
<div><a href="https://github.com/Polymarket/py-clob-client" style="color:#4da6ff">py-clob-client</a> <span class="d">Official Python SDK</span></div>
<div><a href="https://github.com/Polymarket/agents" style="color:#4da6ff">Polymarket Agents</a> <span class="d">AI agent framework</span></div>
</div>
<div>
<div style="font-weight:700;color:#ffd43b;margin-bottom:6px">Paper Trading & Backtesting</div>
<div><a href="https://polysimulator.com" style="color:#ffd43b">PolySimulator</a> <span class="d">Paper trade with fake $1k</span></div>
<div><a href="https://polybacktest.com" style="color:#ffd43b">PolyBackTest</a> <span class="d">Historical order book data</span></div>
<div><a href="https://polymarketanalytics.com" style="color:#ffd43b">PolyMarket Analytics</a> <span class="d">Leaderboard & analytics</span></div>
</div>
<div>
<div style="font-weight:700;color:#a78bfa;margin-bottom:6px">Automation Platforms</div>
<div><a href="https://nautilustrader.io/docs/latest/integrations/polymarket/" style="color:#a78bfa">NautilusTrader</a> <span class="d">Institutional algo platform</span></div>
<div><a href="https://github.com/caiovicentino/polymarket-mcp-server" style="color:#a78bfa">MCP Server</a> <span class="d">Claude AI trading integration</span></div>
<div><a href="https://github.com/gnosis/prediction-market-agent-tooling" style="color:#a78bfa">Gnosis Agent Tooling</a> <span class="d">Multi-platform AI agents</span></div>
</div>
</div>
</div></div>

</div>
<div class="foot">MTY Performance Tracker — Updates every hour — Simulated P&L based on Kelly Criterion sizing</div>

<script>
function load(){
fetch('/api/performance').then(r=>r.json()).then(d=>{
let s=d.stats||{};
document.getElementById('s-total').textContent=s.total_predictions||0;
let acc=s.accuracy||0;
document.getElementById('s-acc').textContent=acc?acc.toFixed(1)+'%':'--';
document.getElementById('s-acc').style.color=acc>=60?'#00d4aa':acc>=50?'#ffd43b':'#ff4055';
let pnl=s.total_pnl||0;
document.getElementById('s-pnl').textContent=pnl?'$'+(pnl>=0?'+':'')+pnl.toLocaleString(undefined,{maximumFractionDigits:0}):'$0';
document.getElementById('s-pnl').style.color=pnl>=0?'#00d4aa':'#ff4055';
let roi=s.roi||0;
document.getElementById('s-roi').textContent=roi?(roi>=0?'+':'')+roi.toFixed(1)+'%':'--';
document.getElementById('s-roi').style.color=roi>=0?'#00d4aa':'#ff4055';
document.getElementById('s-streak').textContent=s.max_win_streak||'--';
if(d.last_check){let dt=new Date(d.last_check);document.getElementById('s-time').textContent=dt.toLocaleTimeString();}

// Category table
let catStats=s.category_stats||{};
let catHtml='';
let catEntries=Object.entries(catStats).sort((a,b)=>b[1].total-a[1].total);
if(!catEntries.length){catHtml='<tr><td colspan="5" class="d" style="text-align:center">No resolved predictions yet. Performance tracking starts when markets resolve.</td></tr>';}
catEntries.forEach(([cat,cs])=>{
let accCls=cs.accuracy>=60?'g':cs.accuracy>=50?'y':'r';
let pnlCls=cs.pnl>=0?'g':'r';
catHtml+='<tr><td style="font-weight:600;text-transform:capitalize">'+cat+'</td>';
catHtml+='<td class="'+accCls+'" style="font-weight:700">'+cs.accuracy+'%</td>';
catHtml+='<td>'+cs.wins+'W / '+(cs.total-cs.wins)+'L</td>';
catHtml+='<td class="'+pnlCls+'" style="font-weight:600">$'+(cs.pnl>=0?'+':'')+cs.pnl.toLocaleString(undefined,{maximumFractionDigits:0})+'</td>';
catHtml+='<td class="d">$'+cs.avg_pnl.toFixed(0)+'/trade</td>';
catHtml+='</tr>';
});
document.getElementById('cat-table').innerHTML=catHtml;

// Edge bucket table
let edgeBuckets=s.edge_buckets||{};
let edgeHtml='';
let bucketOrder=['50%+','30-50%','10-30%','3-10%','<3%'];
let hasBuckets=false;
bucketOrder.forEach(bucket=>{
let eb=edgeBuckets[bucket];
if(!eb)return;
hasBuckets=true;
let accCls=eb.accuracy>=60?'g':eb.accuracy>=50?'y':'r';
let pnlCls=eb.pnl>=0?'g':'r';
edgeHtml+='<tr><td style="font-weight:600">'+bucket+'</td>';
edgeHtml+='<td class="'+accCls+'" style="font-weight:700">'+eb.accuracy+'%</td>';
edgeHtml+='<td>'+eb.total+' trades ('+eb.wins+'W)</td>';
edgeHtml+='<td class="'+pnlCls+'" style="font-weight:600">$'+(eb.pnl>=0?'+':'')+eb.pnl.toLocaleString(undefined,{maximumFractionDigits:0})+'</td>';
edgeHtml+='</tr>';
});
if(!hasBuckets)edgeHtml='<tr><td colspan="4" class="d" style="text-align:center">No data yet</td></tr>';
document.getElementById('edge-table').innerHTML=edgeHtml;

// Resolved predictions
let resolved=(d.resolved||[]).slice().reverse().slice(0,30);
document.getElementById('res-count').textContent=(d.resolved||[]).length+' total';
let resHtml='';
if(!resolved.length){resHtml='<tr><td colspan="6" class="d" style="text-align:center">No markets have resolved yet. Check back as markets settle.</td></tr>';}
resolved.forEach(r=>{
let correct=r.correct;
let tagCls=correct===true?'tag-win':'tag-loss';
let tagText=correct===true?'WIN':'LOSS';
let pnlCls=r.pnl>=0?'g':'r';
resHtml+='<tr>';
resHtml+='<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+r.title+'</td>';
resHtml+='<td style="text-transform:capitalize">'+r.category+'</td>';
resHtml+='<td style="font-weight:600">'+r.our_pick+'</td>';
resHtml+='<td><span class="tag '+tagCls+'">'+tagText+'</span> <span class="d">'+(r.winner||'')+'</span></td>';
resHtml+='<td>'+(r.edge*100).toFixed(1)+'%</td>';
resHtml+='<td class="'+pnlCls+'" style="font-weight:600">$'+(r.pnl>=0?'+':'')+r.pnl.toFixed(0)+'</td>';
resHtml+='</tr>';
});
document.getElementById('res-table').innerHTML=resHtml;
}).catch(e=>{console.log('perf error',e);});
}
load();setInterval(load,300000);
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# TRADERS PAGE — Copy-Trade Screener (4-tab)
# ════════════════════════════════════════════════════════════════════

TRADERS_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Trader Screener</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Sub-tabs ── */
.sub-tabs{display:flex;gap:4px;margin-bottom:20px;border-bottom:1px solid #1f2133;padding-bottom:10px}
.sub-tab{font-size:11px;padding:8px 20px;border-radius:8px 8px 0 0;color:#5c6478;cursor:pointer;font-weight:600;letter-spacing:1px;transition:.2s;border:1px solid transparent;border-bottom:none}
.sub-tab:hover{color:#e4e5ea;background:rgba(255,255,255,.03)}
.sub-tab.on{background:rgba(0,212,170,.1);color:#00d4aa;border-color:#1f2133}
.tab-content{display:none}.tab-content.active{display:block}

/* ── Grade Badges ── */
.grade{font-size:11px;font-weight:900;padding:4px 10px;border-radius:6px;letter-spacing:1px;display:inline-block;min-width:28px;text-align:center}
.grade-S{background:rgba(167,139,250,.2);color:#a78bfa;border:1px solid rgba(167,139,250,.4)}
.grade-A{background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3)}
.grade-B{background:rgba(255,212,59,.1);color:#ffd43b;border:1px solid rgba(255,212,59,.2)}
.grade-C{background:rgba(255,140,66,.1);color:#ff8c42;border:1px solid rgba(255,140,66,.2)}
.grade-D{background:rgba(255,64,85,.1);color:#ff4055;border:1px solid rgba(255,64,85,.2)}

/* ── Score Bar ── */
.score-bar{width:80px;height:6px;background:#1a1d2a;border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle;margin-left:8px}
.score-fill{height:100%;border-radius:3px;transition:width .3s}

/* ── Rankings Table ── */
.rank-row{cursor:pointer;transition:background .15s}
.rank-row:hover{background:#181a24!important}
.rank-row td{vertical-align:middle}
.sparkline-cell{width:80px;height:30px}
.sparkline-cell canvas{width:80px;height:30px}

/* ── Filters Bar ── */
.filter-bar{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
.filter-bar select,.filter-bar input{background:#12131a;border:1px solid #1f2133;color:#e4e5ea;padding:6px 12px;border-radius:8px;font-size:11px;font-family:inherit}
.filter-bar label{font-size:10px;color:#5c6478;letter-spacing:1px}

/* ── Deep Dive ── */
.dd-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.dd-grid{grid-template-columns:1fr}}
.dd-chart{background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:16px;position:relative}
.dd-chart-title{font-size:10px;color:#5c6478;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;font-weight:600}
.dd-chart canvas{width:100%;height:200px}
.dd-assess{background:#12131a;border:1px solid #1f2133;border-radius:12px;padding:18px;margin-top:16px}
.dd-assess-title{font-size:11px;color:#00d4aa;font-weight:700;margin-bottom:10px;letter-spacing:1px}
.dd-assess-row{display:flex;justify-content:space-between;padding:6px 0;font-size:12px;border-bottom:1px solid #111219}
.heatmap-grid{display:grid;grid-template-columns:repeat(24,1fr);gap:2px}
.heatmap-cell{aspect-ratio:1;border-radius:3px;position:relative}
.heatmap-cell:hover::after{content:attr(data-tip);position:absolute;bottom:110%;left:50%;transform:translateX(-50%);background:#1f2133;color:#e4e5ea;padding:4px 8px;border-radius:4px;font-size:9px;white-space:nowrap;z-index:10}

/* ── Simulator ── */
.sim-chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.sim-chip{padding:6px 14px;border-radius:20px;font-size:11px;cursor:pointer;border:1px solid #1f2133;transition:.2s;font-weight:600}
.sim-chip:hover{border-color:#5c6478}
.sim-chip.selected{background:rgba(0,212,170,.15);border-color:#00d4aa;color:#00d4aa}
.sim-inputs{display:flex;gap:16px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.sim-inputs input{background:#12131a;border:1px solid #1f2133;color:#e4e5ea;padding:8px 14px;border-radius:8px;font-size:13px;font-family:inherit;width:120px}
.corr-matrix{display:grid;gap:2px;font-size:10px;text-align:center}
.corr-cell{padding:8px 4px;border-radius:4px;font-weight:700}

/* ── Live Monitor (existing styles) ── */
.signal-card{border-bottom:1px solid #1f2133;padding:16px 18px;transition:background .15s}
.signal-card:hover{background:#14151e}
.signal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.signal-title{font-size:13px;font-weight:700;flex:1}
.signal-tag{font-size:9px;padding:4px 10px;font-weight:700;letter-spacing:1px;border-radius:6px}
.signal-consensus{font-size:15px;font-weight:800;color:#00d4aa;margin:6px 0}
.signal-stats{display:flex;gap:20px;font-size:11px;margin:8px 0;flex-wrap:wrap}
.signal-stat{display:flex;flex-direction:column}
.signal-stat-label{color:#5c6478;font-size:9px;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px}
.signal-stat-val{font-weight:700;font-size:13px}
.signal-traders{margin-top:8px;padding-top:8px;border-top:1px solid #1a1d2a}
.trader-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:11px}
.trader-name-sm{font-weight:700;color:#ffd43b}
.trader-wr{font-weight:700}
.trader-size{color:#00d4aa;font-weight:700}
.kelly-box{background:#0e0f16;border:1px solid #1f2133;border-radius:8px;padding:10px 14px;margin-top:8px;font-size:11px;display:flex;justify-content:space-between;align-items:center}
.kelly-label{color:#5c6478;font-size:9px;letter-spacing:1px}
.kelly-val{font-weight:800;font-size:16px;color:#00d4aa}
.exec-badge{font-size:9px;padding:3px 8px;border-radius:6px;background:rgba(0,212,170,.12);color:#00d4aa;border:1px solid rgba(0,212,170,.25);margin-left:6px}
</style></head><body>
<div class="top-bar">
<div class="brand"><div class="brand-logo">MTY</div><div class="brand-sep"></div><div class="brand-sub">TRADER SCREENER</div></div>
<div class="nav">
""" + _nav_html("traders") + r"""
</div>
</div>
<div class="wrap">

<!-- Header Stats -->
<div class="stats">
<div class="stat"><div class="stat-label">Screened</div><div class="stat-val b" id="h-screened">--</div></div>
<div class="stat"><div class="stat-label">Passed Filters</div><div class="stat-val g" id="h-passed">--</div></div>
<div class="stat"><div class="stat-label">Avg Score</div><div class="stat-val y" id="h-avgscore">--</div></div>
<div class="stat"><div class="stat-label">S-Tier</div><div class="stat-val p" id="h-stier">--</div></div>
<div class="stat"><div class="stat-label">Avg Win Rate</div><div class="stat-val g" id="h-avgwr">--</div></div>
<div class="stat"><div class="stat-label">Data Age</div><div class="stat-val d" id="h-age">--</div></div>
</div>

<!-- Sub-tabs -->
<div class="sub-tabs">
<div class="sub-tab on" data-tab="rankings" onclick="switchTab('rankings')">RANKINGS</div>
<div class="sub-tab" data-tab="deepdive" onclick="switchTab('deepdive')">DEEP DIVE</div>
<div class="sub-tab" data-tab="simulator" onclick="switchTab('simulator')">SIMULATOR</div>
<div class="sub-tab" data-tab="live" onclick="switchTab('live')">LIVE MONITOR</div>
</div>

<!-- TAB 1: RANKINGS -->
<div class="tab-content active" id="tab-rankings">
<div class="filter-bar">
<div><label>MIN GRADE</label><br><select id="f-grade" onchange="applyFilters()"><option value="">All</option><option value="S">S</option><option value="A">A+</option><option value="B">B+</option></select></div>
<div><label>MARKET TYPE</label><br><select id="f-market" onchange="applyFilters()"><option value="">All</option><option value="politics">Politics</option><option value="weather">Weather</option><option value="crypto">Crypto</option><option value="sports">Sports</option><option value="economics">Economics</option></select></div>
<div><label>MIN TRADES/30D</label><br><input type="number" id="f-mintrades" value="" placeholder="20" onchange="applyFilters()"></div>
<div style="margin-left:auto"><button class="btn btn-green" onclick="freshScan()">FRESH SCAN</button></div>
</div>
<div class="section">
<div class="sec-hdr"><span><span class="live-dot"></span>Trader Rankings — Sorted by Copyability Score</span><span class="sec-meta" id="rank-badge">Loading...</span></div>
<div style="overflow-x:auto">
<table id="rankings-table">
<thead><tr>
<th>#</th><th>Trader</th><th>Grade</th><th>Score</th><th>WR 30d</th><th>P&L 30d</th><th>Trades</th><th>Avg Pos</th><th>Markets</th><th>Sharpe</th><th>DD%</th><th>Curve</th>
</tr></thead>
<tbody id="rankings-body"><tr><td colspan="12" class="empty">Loading screening data...</td></tr></tbody>
</table>
</div>
</div>
</div>

<!-- TAB 2: DEEP DIVE -->
<div class="tab-content" id="tab-deepdive">
<div id="dd-empty" class="empty" style="padding:60px 20px">Click a trader row in Rankings to load deep dive analysis</div>
<div id="dd-content" style="display:none">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
<div><span style="font-size:18px;font-weight:800;color:#ffd43b" id="dd-name">--</span><span class="grade" style="margin-left:12px" id="dd-grade">--</span><span style="margin-left:12px;font-size:12px;color:#5c6478" id="dd-wallet">--</span></div>
<button class="btn" onclick="switchTab('rankings')">BACK TO RANKINGS</button>
</div>
<div class="stats" style="margin-bottom:20px">
<div class="stat"><div class="stat-label">Score</div><div class="stat-val b" id="dd-score">--</div></div>
<div class="stat"><div class="stat-label">Win Rate</div><div class="stat-val g" id="dd-wr">--</div></div>
<div class="stat"><div class="stat-label">P&L All</div><div class="stat-val g" id="dd-pnl">--</div></div>
<div class="stat"><div class="stat-label">Max DD</div><div class="stat-val r" id="dd-dd">--</div></div>
<div class="stat"><div class="stat-label">Copy Ratio</div><div class="stat-val p" id="dd-ratio">--</div></div>
</div>
<div class="dd-grid">
<div class="dd-chart"><div class="dd-chart-title">EQUITY CURVE</div><canvas id="dd-equity" height="200"></canvas></div>
<div class="dd-chart"><div class="dd-chart-title">DRAWDOWN</div><canvas id="dd-drawdown" height="200"></canvas></div>
<div class="dd-chart"><div class="dd-chart-title">MARKET BREAKDOWN</div><canvas id="dd-markets" height="200"></canvas></div>
<div class="dd-chart"><div class="dd-chart-title">ACTIVITY HEATMAP (DAY x HOUR)</div><div id="dd-heatmap" class="heatmap-grid"></div></div>
</div>
<div class="dd-assess">
<div class="dd-assess-title">COPYABILITY ASSESSMENT</div>
<div id="dd-assess-rows"></div>
</div>
<div class="section" style="margin-top:16px">
<div class="sec-hdr">Recent Trade History <span class="sec-meta" id="dd-trades-badge">--</span></div>
<div style="overflow-x:auto">
<table><thead><tr><th>Market</th><th>Side</th><th>Outcome</th><th>Invested</th><th>P&L</th><th>Status</th></tr></thead>
<tbody id="dd-trades-body"></tbody>
</table>
</div>
</div>
</div>
</div>

<!-- TAB 3: SIMULATOR -->
<div class="tab-content" id="tab-simulator">
<div class="section">
<div class="sec-hdr">Portfolio Simulator — Select 2-3 Traders</div>
<div class="sec-body">
<div class="sim-chips" id="sim-chips"><span class="d" style="font-size:11px">Loading traders...</span></div>
<div class="sim-inputs">
<div><label style="font-size:10px;color:#5c6478;letter-spacing:1px">CAPITAL ($)</label><br><input type="number" id="sim-capital" value="50000"></div>
<button class="btn btn-green" onclick="runSimulation()" id="sim-btn">RUN BACKTEST</button>
<span class="d" style="font-size:11px" id="sim-status"></span>
</div>
</div>
</div>
<div id="sim-results" style="display:none">
<div class="stats" style="margin-top:16px">
<div class="stat"><div class="stat-label">Combined Sharpe</div><div class="stat-val b" id="sim-sharpe">--</div></div>
<div class="stat"><div class="stat-label">Total Return</div><div class="stat-val g" id="sim-return">--</div></div>
<div class="stat"><div class="stat-label">Max Drawdown</div><div class="stat-val r" id="sim-dd">--</div></div>
<div class="stat"><div class="stat-label">Combined WR</div><div class="stat-val y" id="sim-wr">--</div></div>
</div>
<div class="dd-grid">
<div class="dd-chart"><div class="dd-chart-title">COMBINED EQUITY CURVE</div><canvas id="sim-equity" height="200"></canvas></div>
<div class="dd-chart"><div class="dd-chart-title">CORRELATION MATRIX</div><div id="sim-corr"></div></div>
</div>
</div>
</div>

<!-- TAB 4: LIVE MONITOR -->
<div class="tab-content" id="tab-live">
<div class="section">
<div class="sec-hdr"><span><span class="live-dot"></span>Copy Signals — What Top Traders Bet On Now</span><span class="sec-meta" id="sig-badge">Loading...</span></div>
<div id="signals-list"><div class="empty">Loading copy signals...</div></div>
</div>
<div class="section">
<div class="sec-hdr">Top Trader Profiles <span class="sec-meta" id="t-badge">--</span></div>
<div id="traders-list"><div class="empty">Loading traders...</div></div>
</div>
<div class="section">
<div class="sec-hdr">Copy Executor — Today's Trades <span class="sec-meta" id="exec-badge">--</span></div>
<div id="exec-list"><div class="empty">No trades today.</div></div>
</div>
</div>

</div><!-- /wrap -->

<div class="bottom-bar">
<span class="bar-info" id="updated">--</span>
<button class="btn" onclick="refreshAll()">REFRESH</button>
</div>
<div style="height:50px"></div>

<script>
// ── State ──
let allTraders = [];
let filteredTraders = [];
let selectedTrader = null;
let simSelected = new Set();

// ── Tab switching ──
function switchTab(tab) {
    document.querySelectorAll('.sub-tab').forEach(t => t.classList.toggle('on', t.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + tab));
    if (tab === 'live') refreshLive();
}

// ── Canvas Chart Helpers ──
function drawLineChart(canvas, data, opts = {}) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth - 32;
    const h = opts.height || 200;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    if (!data || data.length < 2) { ctx.fillStyle = '#5c6478'; ctx.fillText('No data', w/2-20, h/2); return; }
    const vals = data.map(d => typeof d === 'number' ? d : d.v);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    const pad = 10;
    ctx.clearRect(0, 0, w, h);
    // grid lines
    ctx.strokeStyle = '#1a1d2a'; ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
        const y = pad + (h - 2*pad) * i / 4;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w-pad, y); ctx.stroke();
    }
    // line
    ctx.beginPath();
    const color = opts.color || '#00d4aa';
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    data.forEach((d, i) => {
        const x = pad + (w - 2*pad) * i / (data.length - 1);
        const v = typeof d === 'number' ? d : d.v;
        const y = pad + (h - 2*pad) * (1 - (v - mn) / range);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // fill
    if (opts.fill) {
        const lastX = pad + (w - 2*pad);
        ctx.lineTo(lastX, h - pad); ctx.lineTo(pad, h - pad); ctx.closePath();
        ctx.fillStyle = color.replace(')', ',.1)').replace('rgb', 'rgba');
        ctx.fill();
    }
    // labels
    ctx.fillStyle = '#5c6478'; ctx.font = '10px -apple-system,sans-serif';
    ctx.fillText(mx >= 1000 ? '$' + (mx/1000).toFixed(1) + 'k' : mx.toFixed(1), pad, pad - 2);
    ctx.fillText(mn >= 1000 ? '$' + (mn/1000).toFixed(1) + 'k' : mn.toFixed(1), pad, h - 2);
}

function drawSparkline(canvas, data, color) {
    if (!data || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const w = 80, h = 30;
    canvas.width = w * 2; canvas.height = h * 2;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(2, 2);
    const vals = data.map(d => typeof d === 'number' ? d : (d.pnl || d.v || 0));
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    vals.forEach((v, i) => {
        const x = 2 + (w-4) * i / (vals.length-1);
        const y = 2 + (h-4) * (1 - (v-mn)/range);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
}

function drawPieChart(canvas, segments) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const size = Math.min(canvas.parentElement.clientWidth - 32, 200);
    canvas.width = size * dpr; canvas.height = size * dpr;
    canvas.style.width = size + 'px'; canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);
    if (!segments || !segments.length) { ctx.fillStyle = '#5c6478'; ctx.fillText('No data', size/2-20, size/2); return; }
    const total = segments.reduce((s, sg) => s + sg.v, 0);
    if (!total) return;
    const colors = ['#00d4aa','#4da6ff','#ffd43b','#a78bfa','#ff8c42','#ff4055','#66d9e8','#e4e5ea'];
    let angle = -Math.PI/2;
    const cx = size/2, cy = size/2, r = size/2 - 30;
    segments.forEach((sg, i) => {
        const slice = (sg.v / total) * Math.PI * 2;
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, angle, angle + slice);
        ctx.fillStyle = colors[i % colors.length]; ctx.fill();
        // label
        if (sg.v / total > 0.05) {
            const mid = angle + slice/2;
            const lx = cx + (r + 16) * Math.cos(mid);
            const ly = cy + (r + 16) * Math.sin(mid);
            ctx.fillStyle = '#e4e5ea'; ctx.font = '10px -apple-system,sans-serif';
            ctx.textAlign = 'center'; ctx.fillText(sg.label, lx, ly);
        }
        angle += slice;
    });
}

// ── Rankings ──
async function loadRankings() {
    try {
        const r = await fetch('/api/trader-screening');
        const d = await r.json();
        allTraders = d.traders || [];
        const stats = d.stats || {};
        document.getElementById('h-screened').textContent = stats.total_screened || allTraders.length;
        document.getElementById('h-passed').textContent = d.n_passed || allTraders.length;
        document.getElementById('h-avgscore').textContent = (stats.avg_score || 0).toFixed(1);
        document.getElementById('h-stier').textContent = (stats.grade_dist || {}).S || 0;
        document.getElementById('h-avgwr').textContent = (stats.avg_wr || 0).toFixed(1) + '%';
        if (d.timestamp) {
            try {
                const age = Math.floor((Date.now() - new Date(d.timestamp).getTime()) / 60000);
                document.getElementById('h-age').textContent = age + 'm';
                document.getElementById('h-age').style.color = age > 30 ? '#ff4055' : '#00d4aa';
            } catch(e) {}
        }
        document.getElementById('rank-badge').textContent = allTraders.length + ' traders';
        applyFilters();
        renderSimChips();
    } catch(e) { console.log('screening error', e); }
}

function applyFilters() {
    const minGrade = document.getElementById('f-grade').value;
    const marketType = document.getElementById('f-market').value;
    const minTrades = parseInt(document.getElementById('f-mintrades').value) || 0;
    const gradeOrder = ['S','A','B','C','D'];
    filteredTraders = allTraders.filter(t => {
        if (minGrade && gradeOrder.indexOf(t.grade) > gradeOrder.indexOf(minGrade)) return false;
        if (marketType && !(t.main_markets || '').toLowerCase().includes(marketType)) return false;
        if (minTrades && (t.trades_30d || 0) < minTrades) return false;
        return true;
    });
    renderRankings();
}

function renderRankings() {
    const body = document.getElementById('rankings-body');
    if (!filteredTraders.length) {
        body.innerHTML = '<tr><td colspan="12" class="empty">No traders match filters. Try relaxing criteria or run a fresh scan.</td></tr>';
        return;
    }
    body.innerHTML = filteredTraders.map((t, i) => {
        const scoreColor = t.score >= 80 ? '#a78bfa' : t.score >= 65 ? '#00d4aa' : t.score >= 50 ? '#ffd43b' : t.score >= 35 ? '#ff8c42' : '#ff4055';
        const wrColor = (t.win_rate_30d||0) >= 60 ? 'g' : (t.win_rate_30d||0) >= 50 ? 'y' : 'r';
        const pnl30d = t.pnl_30d || 0;
        const pnlStr = pnl30d >= 0 ? '+$' + pnl30d.toLocaleString() : '-$' + Math.abs(pnl30d).toLocaleString();
        const ddPct = t.max_drawdown_pct || 0;
        const ddColor = ddPct < 20 ? 'g' : ddPct < 35 ? 'y' : 'r';
        return `<tr class="rank-row" onclick="loadDeepDive('${t.wallet}')" style="background:${i%2===0?'transparent':'rgba(255,255,255,.01)'}">
            <td style="font-weight:800;color:#5c6478">${i+1}</td>
            <td style="font-weight:700">${t.name || t.wallet.substring(0,10)}</td>
            <td><span class="grade grade-${t.grade}">${t.grade}</span></td>
            <td><span style="font-weight:800;color:${scoreColor}">${(t.score||0).toFixed(0)}</span><div class="score-bar"><div class="score-fill" style="width:${t.score||0}%;background:${scoreColor}"></div></div></td>
            <td class="${wrColor}" style="font-weight:700">${(t.win_rate_30d||0).toFixed(1)}%</td>
            <td class="${pnl30d>=0?'g':'r'}" style="font-weight:700">${pnlStr}</td>
            <td>${t.trades_30d||0}</td>
            <td>$${(t.avg_position||0).toLocaleString()}</td>
            <td style="font-size:10px">${(t.main_markets||'--').substring(0,20)}</td>
            <td style="font-weight:700">${(t.consistency_sharpe||0).toFixed(2)}</td>
            <td class="${ddColor}">${ddPct.toFixed(1)}%</td>
            <td class="sparkline-cell"><canvas id="spark-${i}" width="160" height="60"></canvas></td>
        </tr>`;
    }).join('');
    // Draw sparklines
    filteredTraders.forEach((t, i) => {
        const c = document.getElementById('spark-' + i);
        if (c && t.equity_curve) drawSparkline(c, t.equity_curve, t.score >= 65 ? '#00d4aa' : '#ffd43b');
    });
}

async function freshScan() {
    document.getElementById('rank-badge').textContent = 'Scanning...';
    try {
        const r = await fetch('/api/trader-screening?fresh=1');
        const d = await r.json();
        allTraders = d.traders || [];
        applyFilters();
        document.getElementById('rank-badge').textContent = allTraders.length + ' traders (fresh)';
    } catch(e) { document.getElementById('rank-badge').textContent = 'Scan failed'; }
}

// ── Deep Dive ──
async function loadDeepDive(wallet) {
    switchTab('deepdive');
    document.getElementById('dd-empty').style.display = 'none';
    document.getElementById('dd-content').style.display = 'block';
    document.getElementById('dd-name').textContent = 'Loading...';
    try {
        const r = await fetch('/api/trader-detail?wallet=' + encodeURIComponent(wallet));
        const d = await r.json();
        if (d.error) { document.getElementById('dd-name').textContent = 'Error: ' + d.error; return; }
        selectedTrader = d;
        renderDeepDive(d);
    } catch(e) { document.getElementById('dd-name').textContent = 'Failed to load'; }
}

function renderDeepDive(d) {
    document.getElementById('dd-name').textContent = d.name || 'Unknown';
    const gEl = document.getElementById('dd-grade');
    gEl.textContent = d.grade || '--';
    gEl.className = 'grade grade-' + (d.grade || 'D');
    document.getElementById('dd-wallet').textContent = d.wallet || '';
    document.getElementById('dd-score').textContent = (d.score || 0).toFixed(1);
    document.getElementById('dd-wr').textContent = (d.win_rate_all || d.win_rate_30d || 0).toFixed(1) + '%';
    document.getElementById('dd-pnl').textContent = '$' + (d.pnl_all || 0).toLocaleString();
    document.getElementById('dd-dd').textContent = (d.max_drawdown_pct || 0).toFixed(1) + '%';
    document.getElementById('dd-ratio').textContent = d.copy_ratio ? d.copy_ratio.toFixed(0) + '%' : '--';

    // Equity curve
    if (d.equity_curve && d.equity_curve.length) {
        drawLineChart(document.getElementById('dd-equity'), d.equity_curve.map(p => ({v: p.pnl || p})), {color:'#00d4aa', fill:true});
    }
    // Drawdown
    if (d.drawdown_series && d.drawdown_series.length) {
        drawLineChart(document.getElementById('dd-drawdown'), d.drawdown_series.map(p => ({v: -(p.dd_pct || p || 0)})), {color:'#ff4055', fill:true});
    }
    // Market breakdown pie
    if (d.market_breakdown) {
        const segs = Object.entries(d.market_breakdown).filter(([k,v]) => v > 0).map(([k,v]) => ({label:k, v:v})).sort((a,b)=>b.v-a.v);
        drawPieChart(document.getElementById('dd-markets'), segs);
    }
    // Timing heatmap
    if (d.timing_heatmap) {
        renderHeatmap(d.timing_heatmap);
    }
    // Assessment
    const assessEl = document.getElementById('dd-assess-rows');
    const items = [
        ['Win Rate (30d)', (d.win_rate_30d||0).toFixed(1) + '%', (d.win_rate_30d||0) >= 55 ? 'g' : 'y'],
        ['Consistency (Sharpe)', (d.consistency_sharpe||0).toFixed(2), (d.consistency_sharpe||0) >= 1 ? 'g' : 'y'],
        ['Max Drawdown', (d.max_drawdown_pct||0).toFixed(1) + '%', (d.max_drawdown_pct||0) < 25 ? 'g' : 'r'],
        ['Trade Frequency (30d)', (d.trades_30d||0) + ' trades', (d.trades_30d||0) >= 20 ? 'g' : 'y'],
        ['Avg Position Size', '$' + (d.avg_position||0).toLocaleString(), 'b'],
        ['ROI Efficiency', (d.roi_pct||0).toFixed(2) + '%', (d.roi_pct||0) > 1 ? 'g' : 'y'],
        ['Recommended Copy Ratio', (d.copy_ratio||0).toFixed(0) + '% of capital', 'p'],
    ];
    assessEl.innerHTML = items.map(([lbl, val, cls]) =>
        `<div class="dd-assess-row"><span>${lbl}</span><span class="${cls}" style="font-weight:700">${val}</span></div>`
    ).join('');
    // Trades table
    const trades = d.recent_trades || d.all_markets || [];
    document.getElementById('dd-trades-badge').textContent = trades.length + ' markets';
    const tbody = document.getElementById('dd-trades-body');
    tbody.innerHTML = trades.slice(0, 50).map(t => {
        const pnlCls = (t.pnl||0) >= 0 ? 'g' : 'r';
        return `<tr>
            <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${t.title||'--'}</td>
            <td class="${t.last_side==='BUY'?'g':'r'}" style="font-weight:700">${t.last_side||'--'}</td>
            <td>${t.outcome||'--'}</td>
            <td>$${(t.buys||0).toLocaleString()}</td>
            <td class="${pnlCls}" style="font-weight:700">$${(t.pnl||0).toLocaleString()}</td>
            <td>${t.resolved ? '<span class="tag tag-win">RESOLVED</span>' : '<span class="tag tag-moderate">OPEN</span>'}</td>
        </tr>`;
    }).join('');
}

function renderHeatmap(heatmap) {
    const el = document.getElementById('dd-heatmap');
    // heatmap is 7x24 array (day x hour)
    if (!heatmap || !heatmap.length) { el.innerHTML = '<span class="d" style="font-size:11px">No timing data</span>'; return; }
    const flat = heatmap.flat();
    const maxVal = Math.max(...flat, 1);
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    let html = '<div style="display:grid;grid-template-columns:40px repeat(24,1fr);gap:2px;font-size:9px">';
    // Header row
    html += '<div></div>';
    for (let h = 0; h < 24; h++) html += `<div style="text-align:center;color:#5c6478">${h}</div>`;
    // Data rows
    for (let d = 0; d < 7 && d < heatmap.length; d++) {
        html += `<div style="color:#5c6478;display:flex;align-items:center">${days[d]}</div>`;
        for (let h = 0; h < 24; h++) {
            const v = (heatmap[d] && heatmap[d][h]) || 0;
            const intensity = v / maxVal;
            const bg = intensity > 0 ? `rgba(0,212,170,${0.1 + intensity * 0.8})` : '#111219';
            html += `<div class="heatmap-cell" style="background:${bg}" data-tip="${days[d]} ${h}:00 - ${v} trades"></div>`;
        }
    }
    html += '</div>';
    el.innerHTML = html;
}

// ── Simulator ──
function renderSimChips() {
    const el = document.getElementById('sim-chips');
    if (!allTraders.length) { el.innerHTML = '<span class="d" style="font-size:11px">Load rankings first</span>'; return; }
    el.innerHTML = allTraders.filter(t => t.grade === 'S' || t.grade === 'A' || t.grade === 'B').slice(0, 20).map(t => {
        const sel = simSelected.has(t.wallet) ? ' selected' : '';
        return `<div class="sim-chip${sel}" onclick="toggleSim('${t.wallet}')" data-wallet="${t.wallet}"><span class="grade grade-${t.grade}" style="padding:2px 6px;font-size:9px;margin-right:4px">${t.grade}</span>${t.name || t.wallet.substring(0,8)}</div>`;
    }).join('');
}

function toggleSim(wallet) {
    if (simSelected.has(wallet)) simSelected.delete(wallet);
    else if (simSelected.size < 3) simSelected.add(wallet);
    renderSimChips();
}

async function runSimulation() {
    if (simSelected.size < 2) { document.getElementById('sim-status').textContent = 'Select 2-3 traders'; return; }
    document.getElementById('sim-btn').textContent = 'RUNNING...';
    document.getElementById('sim-status').textContent = '';
    try {
        const r = await fetch('/api/trader-simulate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({wallets: [...simSelected]})
        });
        const d = await r.json();
        if (d.error) { document.getElementById('sim-status').textContent = d.error; return; }
        document.getElementById('sim-results').style.display = 'block';
        document.getElementById('sim-sharpe').textContent = (d.combined_sharpe || 0).toFixed(2);
        document.getElementById('sim-return').textContent = '$' + (d.total_return || 0).toLocaleString();
        document.getElementById('sim-dd').textContent = (d.max_drawdown_pct || 0).toFixed(1) + '%';
        document.getElementById('sim-wr').textContent = (d.combined_win_rate || 0).toFixed(1) + '%';
        // Equity chart
        if (d.combined_equity && d.combined_equity.length) {
            drawLineChart(document.getElementById('sim-equity'), d.combined_equity.map(p => ({v: p.pnl || p})), {color:'#4da6ff', fill:true});
        }
        // Correlation matrix
        if (d.correlations) {
            renderCorrelationMatrix(d.correlations, d.trader_names || [...simSelected].map(w => w.substring(0,8)));
        }
    } catch(e) { document.getElementById('sim-status').textContent = 'Simulation failed'; }
    document.getElementById('sim-btn').textContent = 'RUN BACKTEST';
}

function renderCorrelationMatrix(corr, names) {
    const el = document.getElementById('sim-corr');
    const n = names.length;
    let html = `<div class="corr-matrix" style="grid-template-columns:80px repeat(${n},1fr)">`;
    html += '<div></div>';
    names.forEach(nm => html += `<div style="font-weight:700;color:#5c6478;font-size:10px">${nm.substring(0,8)}</div>`);
    for (let i = 0; i < n; i++) {
        html += `<div style="font-weight:700;color:#5c6478;font-size:10px;display:flex;align-items:center">${names[i].substring(0,8)}</div>`;
        for (let j = 0; j < n; j++) {
            const v = (corr[i] && corr[i][j]) || (i === j ? 1 : 0);
            const abs = Math.abs(v);
            const bg = v > 0.5 ? 'rgba(255,64,85,.2)' : v > 0.2 ? 'rgba(255,212,59,.15)' : 'rgba(0,212,170,.1)';
            const color = v > 0.5 ? '#ff4055' : v > 0.2 ? '#ffd43b' : '#00d4aa';
            html += `<div class="corr-cell" style="background:${bg};color:${color}">${v.toFixed(2)}</div>`;
        }
    }
    html += '</div>';
    el.innerHTML = html;
}

// ── Live Monitor ──
function kellySize(wr, odds, bankroll) {
    const p = wr / 100; const q = 1 - p; const b = odds - 1;
    if (b <= 0 || p <= 0) return 0;
    let f = (p * b - q) / b;
    if (f <= 0) return 0;
    f = f * 0.25; f = Math.min(f, 0.10);
    return Math.max(50, bankroll * f);
}

async function refreshLive() {
    // Copy signals
    try {
        const r = await fetch('/api/copysignals'); const d = await r.json();
        const signals = d.signals || [];
        document.getElementById('sig-badge').textContent = signals.length + ' signals | ' + (d.time ? d.time.substring(11,19) : '--') + ' UTC';
        const box = document.getElementById('signals-list');
        if (!signals.length) {
            box.innerHTML = '<div class="empty">No signals yet. Waiting for copy_signal_bot.py scan.</div>';
        } else {
            box.innerHTML = signals.slice(0,15).map(s => {
                const tagCls = s.strength === 'STRONG' ? 'tag-strong' : (s.strength === 'MODERATE' ? 'tag-moderate' : 'tag-weak');
                const kelly = kellySize(s.avg_wr, 1.8, 50000);
                let html = `<div class="signal-card">
                    <div class="signal-header"><div class="signal-title">${s.title}</div><div class="signal-tag ${tagCls}">${s.strength}</div></div>
                    <div class="signal-consensus">BET: ${s.consensus} (${s.consensus_pct}% agreement)</div>
                    <div class="signal-stats">
                        <div class="signal-stat"><div class="signal-stat-label">Traders</div><div class="signal-stat-val c">${s.n_traders}</div></div>
                        <div class="signal-stat"><div class="signal-stat-label">Avg WR</div><div class="signal-stat-val ${s.avg_wr>=60?'g':'y'}">${s.avg_wr}%</div></div>
                        <div class="signal-stat"><div class="signal-stat-label">$ Invested</div><div class="signal-stat-val g">$${s.total_invested.toLocaleString()}</div></div>
                        <div class="signal-stat"><div class="signal-stat-label">Score</div><div class="signal-stat-val b">${s.score.toFixed(1)}</div></div>
                    </div>`;
                if (s.traders && s.traders.length) {
                    html += '<div class="signal-traders">';
                    s.traders.forEach(t => {
                        const wrCls = t.win_rate >= 60 ? 'g' : (t.win_rate >= 50 ? 'y' : 'r');
                        html += `<div class="trader-row"><span class="trader-name-sm">${t.name}</span><span>WR: <span class="trader-wr ${wrCls}">${t.win_rate}%</span></span><span>P&L: <span class="g">$${(t.pnl||0).toLocaleString()}</span></span><span>Bet: <span class="trader-size">$${(t.size||0).toLocaleString()}</span> &rarr; ${t.outcome}</span></div>`;
                    });
                    html += '</div>';
                }
                if (s.strength !== 'WEAK') {
                    html += `<div class="kelly-box"><div><div class="kelly-label">KELLY SUGGESTED SIZE (25% fractional)</div></div><div class="kelly-val">$${kelly.toLocaleString(undefined,{maximumFractionDigits:0})}</div></div>`;
                }
                html += '</div>';
                return html;
            }).join('');
        }
    } catch(e) {}

    // Trader profiles
    try {
        const r = await fetch('/api/traders'); const d = await r.json();
        const traders = d.traders || [];
        document.getElementById('t-badge').textContent = traders.length + ' traders | ' + (d.time ? d.time.substring(11,19) : '--');
        const box = document.getElementById('traders-list');
        if (!traders.length) { box.innerHTML = '<div class="empty">No trader data.</div>'; }
        else {
            box.innerHTML = traders.map((t, i) => {
                let html = `<div class="signal-card"><div style="font-weight:700;color:#ffd43b;margin-bottom:6px">#${i+1} ${t.name}</div><div class="signal-stats">
                    <div class="signal-stat"><div class="signal-stat-label">P&L</div><div class="signal-stat-val g">$${(t.leaderboard_pnl||t.net_pnl||0).toLocaleString()}</div></div>
                    <div class="signal-stat"><div class="signal-stat-label">Win Rate</div><div class="signal-stat-val ${t.win_rate>=60?'g':'y'}">${t.win_rate}%</div></div>
                    <div class="signal-stat"><div class="signal-stat-label">Markets</div><div class="signal-stat-val b">${t.markets_traded}</div></div>
                    <div class="signal-stat"><div class="signal-stat-label">Active</div><div class="signal-stat-val c">${t.active_positions}</div></div>
                </div></div>`;
                return html;
            }).join('');
        }
    } catch(e) {}

    // Executions
    try {
        const r = await fetch('/api/executions'); const d = await r.json();
        const trades = d.trades || [];
        document.getElementById('exec-badge').textContent = trades.length + ' trades today | ' + (d.mode || 'PAPER');
        const box = document.getElementById('exec-list');
        if (!trades.length) { box.innerHTML = '<div class="empty">No trades executed today.</div>'; }
        else {
            box.innerHTML = trades.map(t => `<div class="signal-card" style="padding:10px 18px"><div style="display:flex;justify-content:space-between;align-items:center"><div><span class="exec-badge">${t.status}</span><span style="font-weight:700;margin-left:8px">${t.consensus||t.side||''}</span><span class="d" style="margin-left:8px">${(t.title||'').substring(0,40)}</span></div><div><span class="g" style="font-weight:800">$${(t.size_usd||0).toLocaleString()}</span><span class="d" style="margin-left:8px;font-size:10px">${(t.time||'').substring(11,19)}</span></div></div></div>`).join('');
        }
    } catch(e) {}
}

// ── Master refresh ──
async function refreshAll() {
    await loadRankings();
    if (document.getElementById('tab-live').classList.contains('active')) refreshLive();
    document.getElementById('updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

// ── Init ──
loadRankings();
setInterval(() => { loadRankings(); }, 30000);
document.getElementById('updated').textContent = 'Loading...';
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# PICKS PAGE — Smart Picks (ML + Copy Signal)
# ════════════════════════════════════════════════════════════════════

PICKS_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Smart Picks</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Picks-specific ── */
.pick-card{border-bottom:1px solid #1f2133;padding:16px 18px;transition:background .15s}
.pick-card:hover{background:#14151e}
.pick-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.pick-title{font-size:13px;font-weight:700;flex:1}
.grade{font-size:11px;padding:4px 10px;font-weight:800;letter-spacing:1px;border-radius:6px}
.grade-a{background:rgba(0,212,170,.15);color:#00d4aa;border:1px solid rgba(0,212,170,.3)}
.grade-b{background:rgba(255,212,59,.1);color:#ffd43b;border:1px solid rgba(255,212,59,.2)}
.grade-c{background:rgba(92,100,120,.08);color:#5c6478;border:1px solid #1f2133}
.grade-d{background:rgba(255,64,85,.08);color:#ff4055;border:1px solid rgba(255,64,85,.15)}
.pick-consensus{font-size:15px;font-weight:800;color:#00d4aa;margin:6px 0}
.pick-meta{display:flex;gap:16px;flex-wrap:wrap;color:#5c6478;font-size:11px;margin-top:6px}
.pick-meta b{color:#e4e5ea}
.bar-fill-wrap{height:4px;background:#1a1d2a;margin-top:8px;border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .5s}
.caution{color:#ff8c42;font-size:10px;margin-top:6px;font-weight:600;padding:4px 10px;background:rgba(255,140,66,.06);border-radius:4px}
.no-picks{text-align:center;padding:40px;color:#5c6478;font-size:12px}
</style></head><body>
<div class="top-bar">
<div class="brand"><div class="brand-logo">MTY</div><div class="brand-sep"></div><div class="brand-sub">SMART PICKS</div></div>
<div class="nav">
""" + _nav_html("picks") + r"""
</div>
</div>
<div class="wrap">

<div class="stats">
<div class="stat"><div class="stat-label">Total Picks</div><div class="stat-val" id="h-total" style="color:#66d9e8">--</div></div>
<div class="stat"><div class="stat-label">Strong (70+)</div><div class="stat-val g" id="h-strong">--</div></div>
<div class="stat"><div class="stat-label">Moderate (50+)</div><div class="stat-val y" id="h-moderate">--</div></div>
<div class="stat"><div class="stat-label">Generated</div><div class="stat-val d" id="h-time" style="font-size:13px">--</div></div>
</div>

<div class="section">
<div class="sec-hdr">Smart Picks — ML + Copy Signal Consensus</div>
<div id="picks-list"><div class="no-picks">Loading picks...</div></div>
</div>

</div>
<div class="foot">MTY Smart Picks — ML + Copy Signals — Auto-refreshes every minute</div>

<script>
function load(){
fetch('/api/smartpicks').then(r=>r.json()).then(d=>{
document.getElementById('h-total').textContent=d.n_picks||0;
document.getElementById('h-strong').textContent=d.strong_picks||0;
document.getElementById('h-moderate').textContent=d.moderate_picks||0;
if(d.generated){let dt=new Date(d.generated);document.getElementById('h-time').textContent=dt.toLocaleTimeString();}
let html='';
let picks=(d.picks||[]).filter(p=>p.score>=30);
if(!picks.length){html='<div class="no-picks">No picks available yet. Smart picks run every 30 minutes.</div>';}
picks.forEach((p,i)=>{
let gc='grade-d';
if(p.grade&&p.grade.startsWith('A'))gc='grade-a';
else if(p.grade&&p.grade.startsWith('B'))gc='grade-b';
else if(p.grade=='C')gc='grade-c';
let bar_pct=Math.min(p.score,100);
let bar_color=p.score>=70?'#00d4aa':p.score>=50?'#ffd43b':'#5c6478';
html+='<div class="pick-card">';
html+='<div class="pick-header"><div class="pick-title">#'+(i+1)+' '+p.title+'</div>';
html+='<span class="grade '+gc+'">'+p.grade+' ('+p.score+')</span></div>';
html+='<div class="pick-consensus">'+p.consensus+'</div>';
html+='<div class="pick-meta">';
html+='<span>Traders: <b>'+p.n_traders+'</b></span>';
html+='<span>Avg WR: <b>'+p.avg_wr.toFixed(0)+'%</b></span>';
html+='<span>Invested: <b>$'+(p.total_invested||0).toLocaleString()+'</b></span>';
if(p.market_price)html+='<span>Market: <b>'+(p.market_price*100).toFixed(0)+'c</b></span>';
if(p.ml_prob!==null&&p.ml_prob!==undefined)html+='<span>ML: <b>'+(p.ml_prob*100).toFixed(0)+'%</b></span>';
html+='</div>';
let bd=p.breakdown||{};
if(bd.ml_caution)html+='<div class="caution">CAUTION: '+bd.ml_caution+'</div>';
html+='<div class="bar-fill-wrap"><div class="bar-fill" style="width:'+bar_pct+'%;background:'+bar_color+'"></div></div>';
html+='</div>';
});
document.getElementById('picks-list').innerHTML=html;
}).catch(e=>{document.getElementById('picks-list').innerHTML='<div class="no-picks">Error loading picks</div>';});
}
load();setInterval(load,60000);
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# CRYPTO PAGE — 30-Bot Tournament
# ════════════════════════════════════════════════════════════════════

MAIN_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Crypto Bots</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Crypto-specific ── */
.rank{color:#ffd43b;font-weight:800;font-size:14px}
.tag-profit{background:rgba(0,212,170,.12);color:#00d4aa;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.tag-loss-sm{background:rgba(255,64,85,.1);color:#ff4055;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.tag-wait{background:rgba(92,100,120,.08);color:#5c6478;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.detail{padding:16px 18px;font-size:12px}
.trade-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #111219}
</style></head><body>
<div class="top-bar">
<div class="brand"><div class="brand-logo">MTY</div><div class="brand-sep"></div><div class="brand-sub">CRYPTO BOTS</div></div>
<div class="nav">
""" + _nav_html("crypto") + r"""
</div>
</div>
<div class="wrap">

<div class="stats">
<div class="stat"><div class="stat-label">Bots Active</div><div class="stat-val y" id="s-bots">--</div></div>
<div class="stat"><div class="stat-label">Windows</div><div class="stat-val b" id="s-windows">--</div></div>
<div class="stat"><div class="stat-label">Best P&L</div><div class="stat-val g" id="s-best">--</div></div>
<div class="stat"><div class="stat-label">Status</div><div class="stat-val" id="s-status"><span class="live-dot"></span>LIVE</div></div>
</div>

<div class="section">
<div class="sec-hdr">30-Bot Tournament Leaderboard <span class="sec-meta" id="t-badge">Polymarket 15-Min BTC</span></div>
<div class="sec-body" style="padding:0">
<table><thead><tr><th>#</th><th>Bot</th><th>Trades</th><th>WR</th><th>P&L</th><th>$/Day</th><th>Sharpe</th></tr></thead>
<tbody id="leaderboard"><tr><td colspan="7" class="d" style="text-align:center;padding:30px">Loading tournament data...</td></tr></tbody></table>
</div></div>

<div class="section">
<div class="sec-hdr">Selected Bot Detail</div>
<div class="sec-body detail" id="detail">Tap a bot above to see trade history</div>
</div>

</div>

<div class="bottom-bar">
<span class="bar-info" id="updated">--</span>
<div style="display:flex;gap:8px">
<button class="btn btn-red" onclick="if(confirm('STOP ALL BOTS?'))fetch('/api/emergency-stop',{method:'POST'}).then(r=>r.json()).then(d=>alert(d.status))">KILL ALL</button>
<button class="btn btn-green" onclick="refresh()">REFRESH</button>
</div>
</div>
<div style="height:50px"></div>

<script>
let D=[];
async function refresh(){
    try{const r=await fetch('/api/tournament');D=await r.json()}catch{return}
    const active=D.filter(s=>s.trades>0).length;
    const wins=Math.max(...D.map(s=>s.trades),0);
    const best=D.length?Math.max(...D.map(s=>s.pnl)):0;
    document.getElementById('s-bots').textContent=`${active}/30`;
    document.getElementById('s-windows').textContent=wins;
    const bp=document.getElementById('s-best');bp.textContent=`${best>=0?'+':''}$${best.toFixed(0)}`;bp.style.color=best>=0?'#00d4aa':'#ff4055';
    const tb=document.getElementById('leaderboard');
    if(!D.length){tb.innerHTML='<tr><td colspan="7" class="d" style="text-align:center;padding:30px">Waiting for first 15-min window to resolve...</td></tr>';return}
    tb.innerHTML=D.map((s,i)=>{
        const pc=s.pnl>=0?'g':'r';const wc=s.wr>=55?'g':(s.wr<45&&s.trades>0?'r':'');
        const d=s.daily_est||0;const dc=d>=0?'g':'r';
        const tag=s.pnl>0&&s.trades>1?'<span class="tag-profit">PROFIT</span>':(s.pnl<-100?'<span class="tag-loss-sm">LOSS</span>':(s.trades===0?'<span class="tag-wait">WAIT</span>':''));
        const wr=s.trades>0?`${s.wr}%`:'--';const pnl=s.trades>0?`${s.pnl>=0?'+':''}$${s.pnl.toFixed(0)}`:'--';
        const dy=s.trades>0?`${d>=0?'+':''}$${d.toFixed(0)}`:'--';const sh=s.trades>0?s.sharpe.toFixed(1):'--';
        return`<tr onclick="showBot(${i})" style="cursor:pointer"><td class="rank">${i+1}</td><td>${s.name}${tag}</td><td>${s.trades}</td><td class="${wc}">${wr}</td><td class="${pc}" style="font-weight:700">${pnl}</td><td class="${dc}">${dy}</td><td>${sh}</td></tr>`;
    }).join('');
    document.getElementById('updated').textContent='Updated '+new Date().toLocaleTimeString();
}
function showBot(i){
    const s=D[i];const d=document.getElementById('detail');
    let h=`<div style="margin-bottom:10px"><strong>${s.name}</strong> | ${s.trades}t | WR ${s.wr}% | P&L $${s.pnl.toFixed(0)}</div>`;
    if(s.recent&&s.recent.length){
        s.recent.slice().reverse().forEach(t=>{
            const c=t.correct?'g':'r';const ic=t.correct?'WIN':'LOSS';
            const tm=(t.time||'').substring(11,19);const bc=t.btc_change_pct?t.btc_change_pct.toFixed(3)+'%':'';
            h+=`<div class="trade-row"><div><span class="${c}" style="font-weight:700">${ic}</span> ${t.direction} <span class="d">${tm}</span></div><div class="${c}" style="font-weight:600">$${(t.pnl||0).toFixed(0)} <span class="d">${bc}</span></div></div>`;
        });
    }else h+='<div class="d">Waiting for trades...</div>';
    d.innerHTML=h;
}
refresh();setInterval(refresh,15000);
</script></body></html>"""


# ════════════════════════════════════════════════════════════════════
# SPORTS PAGE — Cross-Platform Arbitrage Scanner
# ════════════════════════════════════════════════════════════════════

SPORTS_HTML = r"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTY Sports</title>
<style>
""" + UNIFIED_CSS + r"""

/* ── Sports-specific ── */
.opp-card{border-bottom:1px solid #1f2133;padding:14px 18px;transition:background .15s}
.opp-card:hover{background:#14151e}
.opp-teams{font-size:13px;font-weight:700;margin-bottom:6px}
.opp-row{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
.opp-edge{display:inline-block;background:rgba(0,212,170,.08);border:1px solid rgba(0,212,170,.25);padding:4px 12px;font-size:13px;font-weight:800;color:#00d4aa;margin-top:6px;border-radius:6px}
.tag-arb{background:rgba(0,212,170,.12);color:#00d4aa;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.tag-edge{background:rgba(77,166,255,.12);color:#4da6ff;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.tag-soft{background:rgba(92,100,120,.08);color:#5c6478;font-size:9px;padding:3px 8px;border-radius:6px;font-weight:600;margin-left:6px}
.strat-row{display:grid;grid-template-columns:40px 1fr 60px 80px 60px;gap:4px;padding:8px 18px;border-bottom:1px solid #111219;font-size:12px;align-items:center}
.strat-row:hover{background:#14151e}
</style></head><body>
<div class="top-bar">
<div class="brand"><div class="brand-logo">MTY</div><div class="brand-sep"></div><div class="brand-sub">SPORTS TERMINAL</div></div>
<div class="nav">
""" + _nav_html("sports") + r"""
</div>
</div>
<div class="wrap">

<div class="stats">
<div class="stat"><div class="stat-label">Platforms Live</div><div class="stat-val c" id="h-plat">--</div></div>
<div class="stat"><div class="stat-label">Opportunities</div><div class="stat-val g" id="h-opps">--</div></div>
<div class="stat"><div class="stat-label">Max Edge</div><div class="stat-val y" id="h-edge">--</div></div>
<div class="stat"><div class="stat-label">Scan Cycle</div><div class="stat-val d" id="h-cycle">3 MIN</div></div>
</div>

<div class="stats">
<div class="stat"><div class="stat-label">Backtest WR</div><div class="stat-val g">70.9%</div></div>
<div class="stat"><div class="stat-label">Backtest ROI</div><div class="stat-val g">+36.4%</div></div>
<div class="stat"><div class="stat-label">ML Accuracy</div><div class="stat-val b">74.9%</div></div>
<div class="stat"><div class="stat-label">Sharpe</div><div class="stat-val y">10.22</div></div>
</div>

<div class="section">
<div class="sec-hdr">Cross-Platform Arbitrage <span class="sec-meta" id="xp-badge">SCANNING...</span></div>
<div id="xp-body"><div class="empty">Connecting to 4 platforms...</div></div>
</div>

<div class="section">
<div class="sec-hdr">Strategy Engines <span class="sec-meta">6 ACTIVE</span></div>
<div style="padding:0">
<div class="strat-row" style="font-size:10px;color:#5c6478;font-weight:600"><div>#</div><div>STRATEGY</div><div>WR</div><div>P&L</div><div>ROI</div></div>
<div class="strat-row"><div class="g" style="font-weight:800">S1</div><div>Pinnacle Arb</div><div class="g">71%</div><div class="g">+$185k</div><div>+36%</div></div>
<div class="strat-row"><div class="c" style="font-weight:800">S2</div><div>Smart Money Copy</div><div class="g">84%</div><div class="g">+$224k</div><div>+61%</div></div>
<div class="strat-row"><div class="y" style="font-weight:800">S3</div><div>Line Movement</div><div class="g">74%</div><div class="g">+$157k</div><div>+43%</div></div>
<div class="strat-row"><div class="p" style="font-weight:800">S4</div><div>Closing Line Value</div><div class="g">83%</div><div class="g">+$264k</div><div>+59%</div></div>
<div class="strat-row"><div class="b" style="font-weight:800">S5</div><div>Rest Day Edge</div><div class="g">74%</div><div class="g">+$37k</div><div>+43%</div></div>
<div class="strat-row"><div class="r" style="font-weight:800">S6</div><div>Consensus Override</div><div class="g">87%</div><div class="g">+$135k</div><div>+68%</div></div>
</div></div>

<div class="section">
<div class="sec-hdr">Platform Status</div>
<div class="sec-body" style="padding:0">
<table><tr><th>PLATFORM</th><th>TYPE</th><th>MARKETS</th><th>STATUS</th></tr>
<tr><td>Polymarket</td><td class="d">CLOB / Polygon</td><td id="p-poly">--</td><td class="g">LIVE</td></tr>
<tr><td>SX Bet</td><td class="d">Orderbook / SX Network</td><td id="p-sx">--</td><td class="g">LIVE</td></tr>
<tr><td>Pinnacle</td><td class="d">Sportsbook / Odds API</td><td id="p-pin">--</td><td class="g">LIVE</td></tr>
<tr><td>Cloudbet</td><td class="d">Crypto Sportsbook</td><td id="p-cb">--</td><td class="y">NEEDS KEY</td></tr>
</table></div></div>

</div>

<div class="bottom-bar">
<div class="bar-info"><span class="live-dot"></span>SCANNING <span id="bar-time" style="margin-left:8px">--</span></div>
<button class="btn" onclick="refresh()">REFRESH</button>
</div>
<div style="height:50px"></div>

<script>
async function refresh(){
    try{
        const r=await fetch('/api/crossplatform');const d=await r.json();
        const opps=d.opportunities||[];const c=d.counts||{};
        document.getElementById('h-plat').textContent=(c.polymarket?1:0)+(c.sxbet?1:0)+(c.pinnacle?1:0)+(c.cloudbet?1:0)+'/4';
        document.getElementById('h-opps').textContent=opps.length;
        document.getElementById('h-edge').textContent=opps.length?((opps[0].edge||0)*100).toFixed(1)+'%':'0%';
        document.getElementById('xp-badge').textContent=`P:${c.polymarket||0} SX:${c.sxbet||0} PIN:${c.pinnacle||0} CB:${c.cloudbet||0}`;
        document.getElementById('p-poly').textContent=c.polymarket||0;
        document.getElementById('p-sx').textContent=c.sxbet||0;
        document.getElementById('p-pin').textContent=c.pinnacle||0;
        document.getElementById('p-cb').textContent=c.cloudbet||0;

        const box=document.getElementById('xp-body');
        if(!opps.length){
            box.innerHTML='<div class="empty">No arbitrage detected. Markets are efficiently priced. Next scan in 3 min.</div>';
        }else{
            box.innerHTML=opps.slice(0,12).map(o=>{
                const pm=((o.polymarket||0)*100).toFixed(1);
                const other=o.sx_bet||o.pinnacle||o.cloudbet||0;
                const otherPct=(other*100).toFixed(1);
                const src=o.platforms||'';
                const typ=o.type||'SOFT_ARB';
                const tagCls=typ==='TRUE_ARB'?'tag-arb':(typ==='STRONG_EDGE'?'tag-edge':'tag-soft');
                return`<div class="opp-card">
                    <div class="opp-teams">${o.game||o.poly_question||''} <span class="${tagCls}">${typ}</span></div>
                    <div class="opp-row"><span class="d">${src}</span><span>Buy: <span class="g">${o.buy_on||''}</span></span></div>
                    <div class="opp-row"><span>Polymarket: <span class="p">${pm}%</span></span><span>vs <span class="c">${otherPct}%</span></span></div>
                    <div><span class="opp-edge">${(o.edge*100).toFixed(1)}% EDGE</span></div>
                </div>`;
            }).join('');
        }
    }catch(e){console.log(e)}

    // Also load sports tournament signals
    try{
        const r2=await fetch('/api/sports');const sd=await r2.json();
        // Already shown via crossplatform
    }catch{}

    document.getElementById('bar-time').textContent=new Date().toLocaleTimeString();
}
refresh();setInterval(refresh,30000);
</script></body></html>"""


def main():
    global DASH_PASSWORD_HASH
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--password", type=str, default=None)
    args = parser.parse_args()
    password = args.password or os.environ.get("DASH_PASSWORD", "")
    if password:
        DASH_PASSWORD_HASH = hash_password(password)
        print("Password protection: ENABLED")
    # Load cached weather data from disk for instant first load
    _load_cache_from_disk()
    # Start background weather cache thread
    cache_thread = threading.Thread(target=_cache_loop, daemon=True)
    cache_thread.start()
    print("Weather cache: background refresh every %ds" % CACHE_TTL)

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"MTY Terminal at http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()

if __name__ == "__main__":
    main()
