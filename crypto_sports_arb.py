#!/usr/bin/env python3
"""
CRYPTO SPORTS ARBITRAGE MONITOR
=================================
Polymarket vs Cloudbet vs SX Bet vs Pinnacle

Scans every 3 minutes. Shows on dashboard.
Finds true arb (guaranteed profit) and soft arb (edge bets).

Platforms:
  - Polymarket: CLOB, Polygon/USDC (LIVE)
  - SX Bet: orderbook, SX Network (LIVE)
  - Pinnacle: via The Odds API (LIVE)
  - Cloudbet: crypto sportsbook (needs API key)
"""

import json, os, time, requests, uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com"
SXBET = "https://api.sx.bet"
ODDS_API = "https://api.the-odds-api.com/v4"
CLOUDBET = "https://sports-api.cloudbet.com/pub/v2/odds"

ODDS_KEY = os.getenv("ODDS_API_KEY", "")
CLOUDBET_KEY = os.getenv("CLOUDBET_API_KEY", "")

# All 30 NBA teams
TEAMS = {
    "ATL": ["Hawks","Atlanta Hawks","Atlanta"],
    "BOS": ["Celtics","Boston Celtics","Boston"],
    "BKN": ["Nets","Brooklyn Nets","Brooklyn"],
    "CHA": ["Hornets","Charlotte Hornets","Charlotte"],
    "CHI": ["Bulls","Chicago Bulls","Chicago"],
    "CLE": ["Cavaliers","Cleveland Cavaliers","Cleveland","Cavs"],
    "DAL": ["Mavericks","Dallas Mavericks","Dallas","Mavs"],
    "DEN": ["Nuggets","Denver Nuggets","Denver"],
    "DET": ["Pistons","Detroit Pistons","Detroit"],
    "GSW": ["Warriors","Golden State Warriors","Golden State","GS Warriors"],
    "HOU": ["Rockets","Houston Rockets","Houston"],
    "IND": ["Pacers","Indiana Pacers","Indiana"],
    "LAC": ["Clippers","LA Clippers","Los Angeles Clippers"],
    "LAL": ["Lakers","LA Lakers","Los Angeles Lakers"],
    "MEM": ["Grizzlies","Memphis Grizzlies","Memphis"],
    "MIA": ["Heat","Miami Heat","Miami"],
    "MIL": ["Bucks","Milwaukee Bucks","Milwaukee"],
    "MIN": ["Timberwolves","Minnesota Timberwolves","Minnesota","T-Wolves"],
    "NOP": ["Pelicans","New Orleans Pelicans","New Orleans"],
    "NYK": ["Knicks","New York Knicks","New York"],
    "OKC": ["Thunder","Oklahoma City Thunder","Oklahoma City","OKC Thunder"],
    "ORL": ["Magic","Orlando Magic","Orlando"],
    "PHI": ["76ers","Philadelphia 76ers","Philadelphia","Sixers"],
    "PHX": ["Suns","Phoenix Suns","Phoenix"],
    "POR": ["Trail Blazers","Portland Trail Blazers","Portland","Blazers"],
    "SAC": ["Kings","Sacramento Kings","Sacramento"],
    "SAS": ["Spurs","San Antonio Spurs","San Antonio"],
    "TOR": ["Raptors","Toronto Raptors","Toronto"],
    "UTA": ["Jazz","Utah Jazz","Utah"],
    "WAS": ["Wizards","Washington Wizards","Washington"],
}

def find_team(name):
    """Find team abbreviation from any name variant."""
    name_l = name.lower().strip()
    for abbr, aliases in TEAMS.items():
        for a in aliases:
            if a.lower() in name_l or name_l in a.lower():
                return abbr
    return None

def american_to_prob(odds):
    try:
        odds = float(odds)
        if odds == 0: return 0.5
        if odds > 0: return round(100 / (odds + 100), 4)
        else: return round(abs(odds) / (abs(odds) + 100), 4)
    except: return 0.5

def decimal_to_prob(odds):
    try: return round(1.0 / float(odds), 4)
    except: return 0.5


# ════════════════════════════════════
# PLATFORM FETCHERS
# ════════════════════════════════════

def fetch_polymarket():
    """Fetch Polymarket NBA moneyline/spread/total markets."""
    markets = []
    for offset in range(0, 300, 100):
        try:
            resp = requests.get(f"{GAMMA}/markets", params={
                "limit":100,"offset":offset,"active":"true","closed":"false",
                "order":"createdAt","ascending":"false"}, timeout=15)
            for m in resp.json():
                slug = m.get("slug","")
                if "nba-" not in slug: continue
                if not m.get("acceptingOrders"): continue
                prices = m.get("outcomePrices","[]")
                outcomes = m.get("outcomes","[]")
                tids = m.get("clobTokenIds","[]")
                if isinstance(prices,str): prices = json.loads(prices)
                if isinstance(outcomes,str): outcomes = json.loads(outcomes)
                if isinstance(tids,str): tids = json.loads(tids)
                if len(prices)<2: continue

                parts = slug.split("-")
                away = parts[1].upper() if len(parts)>1 else ""
                home = parts[2].upper() if len(parts)>2 else ""

                mtype = "ML" if "moneyline" in slug or "winner" in slug else (
                    "SPREAD" if "spread" in slug else (
                    "TOTAL" if "total" in slug else "OTHER"))

                markets.append({
                    "platform":"Polymarket","home":home,"away":away,
                    "type":mtype,"slug":slug,
                    "question":m.get("question",""),
                    "prob1":float(prices[0]),"prob2":float(prices[1]),
                    "outcome1":outcomes[0] if outcomes else "",
                    "outcome2":outcomes[1] if len(outcomes)>1 else "",
                    "bid":float(m.get("bestBid",0) or 0),
                    "ask":float(m.get("bestAsk",0) or 0),
                    "token_ids":tids,
                    "condition_id":m.get("conditionId",""),
                })
        except: break
    return markets

def fetch_sxbet():
    """Fetch SX Bet NBA markets."""
    markets = []
    try:
        resp = requests.get(f"{SXBET}/fixture/active", params={"leagueId":1}, timeout=10)
        for f in resp.json().get("data",[]):
            if "2026-02" not in str(f.get("startDate","")): continue
            eid = f["eventId"]
            t1 = f["participantOneName"]; t2 = f["participantTwoName"]
            h = find_team(t1) or t1[:3].upper()
            a = find_team(t2) or t2[:3].upper()

            try:
                resp2 = requests.get(f"{SXBET}/markets/active", params={"eventId":eid}, timeout=10)
                for m in resp2.json().get("data",{}).get("markets",[]):
                    o1 = m.get("outcomeOneName","")
                    o2 = m.get("outcomeTwoName","")
                    l1 = m.get("outcomeVigOneLine","")
                    l2 = m.get("outcomeVigTwoLine","")
                    mtype_id = m.get("type",0)
                    line = m.get("line","")

                    p1 = american_to_prob(l1); p2 = american_to_prob(l2)
                    mtype = "ML" if mtype_id==226 else ("TOTAL" if mtype_id==28 else (
                            "SPREAD" if mtype_id==342 else f"T{mtype_id}"))

                    markets.append({
                        "platform":"SX Bet","home":h,"away":a,
                        "type":mtype,"line":line,
                        "game":f"{t1} vs {t2}",
                        "prob1":p1,"prob2":p2,
                        "outcome1":o1,"outcome2":o2,
                        "odds1":l1,"odds2":l2,
                        "market_hash":m.get("marketHash",""),
                    })
            except: pass
    except: pass
    return markets

def fetch_pinnacle():
    """Fetch Pinnacle odds via The Odds API."""
    if not ODDS_KEY: return []
    markets = []
    try:
        resp = requests.get(f"{ODDS_API}/sports/basketball_nba/odds", params={
            "apiKey":ODDS_KEY,"regions":"us,eu","markets":"h2h,spreads,totals",
            "bookmakers":"pinnacle","oddsFormat":"decimal"}, timeout=15)
        if not resp.ok: return []
        for event in resp.json():
            home = event.get("home_team",""); away = event.get("away_team","")
            h = find_team(home) or home[:3].upper()
            a = find_team(away) or away[:3].upper()
            for bm in event.get("bookmakers",[]):
                if bm.get("key")!="pinnacle": continue
                for mkt in bm.get("markets",[]):
                    mtype = {"h2h":"ML","spreads":"SPREAD","totals":"TOTAL"}.get(mkt["key"],"OTHER")
                    for o in mkt.get("outcomes",[]):
                        side = "home" if o.get("name")==home else "away"
                        markets.append({
                            "platform":"Pinnacle","home":h,"away":a,
                            "type":mtype,"game":f"{away} @ {home}",
                            "side":side,"outcome":o.get("name",""),
                            "prob":decimal_to_prob(o.get("price",2.0)),
                            "odds":float(o.get("price",2.0)),
                            "point":o.get("point",""),
                        })
    except: pass
    return markets

def fetch_cloudbet():
    """Fetch Cloudbet NBA odds (requires API key)."""
    if not CLOUDBET_KEY: return []
    markets = []
    try:
        resp = requests.get(f"{CLOUDBET}/competitions/basketball-usa-nba",
            headers={"X-API-Key":CLOUDBET_KEY,"accept":"application/json"}, timeout=15)
        if not resp.ok: return []
        data = resp.json()
        for event in data.get("events",[]):
            home = event.get("home",{}).get("name","")
            away = event.get("away",{}).get("name","")
            h = find_team(home) or home[:3].upper()
            a = find_team(away) or away[:3].upper()
            for mkt_key, mkt_data in event.get("markets",{}).items():
                if "moneyline" in mkt_key:
                    for sel in mkt_data.get("selections",[]):
                        side = "home" if sel.get("outcome")=="home" else "away"
                        odds = float(sel.get("price",2.0))
                        markets.append({
                            "platform":"Cloudbet","home":h,"away":a,
                            "type":"ML","game":f"{away} @ {home}",
                            "side":side,"outcome":sel.get("outcome",""),
                            "prob":decimal_to_prob(odds),"odds":odds,
                        })
    except: pass
    return markets


# ════════════════════════════════════
# ARBITRAGE ENGINE
# ════════════════════════════════════

def find_all_arb(poly, sx, pin, cloud):
    """Find arbitrage across all platform pairs."""
    opps = []

    # Build lookup: platform → {(home,away,type): market}
    def build_index(markets, key_fn):
        idx = {}
        for m in markets:
            k = key_fn(m)
            if k: idx.setdefault(k,[]).append(m)
        return idx

    poly_idx = build_index(poly, lambda m: (m["home"],m["away"],m["type"]))

    # Compare Polymarket vs each other platform
    for other_name, other_markets in [("SX Bet",sx),("Pinnacle",pin),("Cloudbet",cloud)]:
        for om in other_markets:
            h = om.get("home",""); a = om.get("away","")
            mtype = om.get("type","")

            # Find matching Polymarket market
            for key in [(h,a,mtype),(a,h,mtype)]:
                if key not in poly_idx: continue
                for pm in poly_idx[key]:
                    # Get probabilities
                    poly_p1 = pm["prob1"]
                    poly_p2 = pm["prob2"]

                    if "prob" in om:  # Pinnacle/Cloudbet format
                        other_prob = om["prob"]
                        other_side = om.get("side","home")

                        if other_side == "home":
                            diff = other_prob - poly_p1
                        else:
                            diff = other_prob - poly_p2

                        if abs(diff) > 0.03:
                            if diff > 0:
                                buy_on = "Polymarket"
                                buy_price = poly_p1 if other_side=="home" else poly_p2
                            else:
                                buy_on = other_name
                                buy_price = other_prob

                            opps.append({
                                "game": om.get("game", pm["question"]),
                                "market": mtype,
                                "polymarket": round(poly_p1 if other_side=="home" else poly_p2, 3),
                                other_name.lower().replace(" ","_"): round(other_prob, 3),
                                "edge": round(abs(diff), 3),
                                "buy_on": buy_on,
                                "buy_price": round(buy_price, 3),
                                "type": "SOFT_ARB" if abs(diff)<0.15 else "STRONG_EDGE",
                                "platforms": f"Polymarket vs {other_name}",
                            })

                    elif "prob1" in om:  # SX Bet format
                        sx_p1 = om["prob1"]; sx_p2 = om["prob2"]
                        if sx_p1 == 0.5 and sx_p2 == 0.5: continue  # no real pricing

                        diff = abs(poly_p1 - sx_p1)
                        if diff > 0.03:
                            opps.append({
                                "game": om.get("game", pm["question"]),
                                "market": mtype,
                                "polymarket": round(poly_p1, 3),
                                "sx_bet": round(sx_p1, 3),
                                "edge": round(diff, 3),
                                "buy_on": "Polymarket" if poly_p1<sx_p1 else "SX Bet",
                                "buy_price": round(min(poly_p1,sx_p1), 3),
                                "type": "SOFT_ARB",
                                "platforms": "Polymarket vs SX Bet",
                            })

                        # TRUE ARBITRAGE check: can we bet both sides for guaranteed profit?
                        # Buy YES on cheaper platform, buy NO on other
                        if poly_p1 + sx_p2 < 0.97:  # after fees
                            margin = 1.0 - poly_p1 - sx_p2
                            opps.append({
                                "game": om.get("game", pm["question"]),
                                "market": mtype,
                                "polymarket": round(poly_p1, 3),
                                "sx_bet": round(sx_p2, 3),
                                "edge": round(margin, 3),
                                "buy_on": "BOTH (true arb)",
                                "buy_price": 0,
                                "type": "TRUE_ARB",
                                "platforms": "Polymarket + SX Bet",
                            })

    opps.sort(key=lambda x: (-{"TRUE_ARB":100,"STRONG_EDGE":50,"SOFT_ARB":10}.get(x["type"],0), -x["edge"]))
    return opps


# ════════════════════════════════════
# MAIN SCAN
# ════════════════════════════════════

def run_scan():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"  CRYPTO SPORTS ARB SCAN | {now.strftime('%H:%M:%S UTC')}")
    print(f"  Polymarket vs SX Bet vs Pinnacle vs Cloudbet")
    print(f"{'='*70}")

    print("  Fetching Polymarket...", end=" "); poly = fetch_polymarket(); print(f"{len(poly)}")
    print("  Fetching SX Bet...", end=" "); sx = fetch_sxbet(); print(f"{len(sx)}")
    print("  Fetching Pinnacle...", end=" "); pin = fetch_pinnacle(); print(f"{len(pin)}")
    print("  Fetching Cloudbet...", end=" "); cloud = fetch_cloudbet(); print(f"{len(cloud)}")

    opps = find_all_arb(poly, sx, pin, cloud)

    print(f"\n  OPPORTUNITIES: {len(opps)}")
    for o in opps[:10]:
        emoji = {"TRUE_ARB":"***","STRONG_EDGE":"**","SOFT_ARB":"*"}.get(o["type"],"")
        print(f"\n  {o['type']} {emoji}")
        print(f"    {o['game'][:45]} | {o['market']}")
        print(f"    {o['platforms']} | Edge: {o['edge']:.1%}")
        print(f"    Buy on: {o['buy_on']} at {o['buy_price']:.1%}")

    # Save
    result = {
        "time": now.isoformat(),
        "opportunities": opps,
        "counts": {"polymarket":len(poly),"sxbet":len(sx),"pinnacle":len(pin),"cloudbet":len(cloud)},
    }
    log_path = LOG_DIR / f"crossplatform_{now.strftime('%Y%m%d_%H%M')}.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    if args.loop:
        while True:
            try: run_scan()
            except Exception as e: print(f"  Error: {e}")
            print(f"\n  Next scan in 3 minutes...")
            time.sleep(180)
    else:
        run_scan()

if __name__ == "__main__":
    main()
