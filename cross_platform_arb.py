#!/usr/bin/env python3
"""
CROSS-PLATFORM ARBITRAGE SCANNER
==================================
Compares odds across crypto sports platforms every 3 minutes:
  - Polymarket (CLOB, Polygon)
  - SX Bet (orderbook, SX Network)
  - Pinnacle (via The Odds API)

Finds mispricing between platforms on the same NBA/NFL games.
"""

import json, os, time, requests
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
GAMMA = "https://gamma-api.polymarket.com"
SXBET = "https://api.sx.bet"
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_KEY = os.getenv("ODDS_API_KEY", "")


def american_to_prob(odds):
    try:
        odds = float(odds)
        if odds == 0: return 0.5
        if odds > 0: return round(100 / (odds + 100), 3)
        else: return round(abs(odds) / (abs(odds) + 100), 3)
    except: return 0.5


def fetch_polymarket_nba():
    """Fetch Polymarket NBA markets."""
    markets = []
    for offset in range(0, 300, 100):
        try:
            resp = requests.get(f"{GAMMA}/markets", params={
                "limit": 100, "offset": offset,
                "active": "true", "closed": "false",
                "order": "createdAt", "ascending": "false",
            }, timeout=15)
            for m in resp.json():
                slug = m.get("slug", "")
                if "nba-" not in slug: continue
                if not m.get("acceptingOrders"): continue
                prices = m.get("outcomePrices", "[]")
                outcomes = m.get("outcomes", "[]")
                if isinstance(prices, str): prices = json.loads(prices)
                if isinstance(outcomes, str): outcomes = json.loads(outcomes)
                if len(prices) < 2: continue
                markets.append({
                    "source": "Polymarket",
                    "question": m.get("question", ""),
                    "slug": slug,
                    "outcome1": outcomes[0] if outcomes else "",
                    "outcome2": outcomes[1] if len(outcomes) > 1 else "",
                    "prob1": float(prices[0]),
                    "prob2": float(prices[1]),
                    "bid": float(m.get("bestBid", 0) or 0),
                    "ask": float(m.get("bestAsk", 0) or 0),
                })
        except: break
    return markets


def fetch_sxbet_nba():
    """Fetch SX Bet NBA markets."""
    markets = []
    try:
        resp = requests.get(f"{SXBET}/fixture/active", params={"leagueId": 1}, timeout=10)
        fixtures = resp.json().get("data", [])
        recent = [f for f in fixtures if "2026-02" in str(f.get("startDate", ""))]

        for f in recent:
            eid = f["eventId"]
            t1 = f["participantOneName"]
            t2 = f["participantTwoName"]
            date = str(f.get("startDate", ""))[:10]

            try:
                resp2 = requests.get(f"{SXBET}/markets/active", params={"eventId": eid}, timeout=10)
                mkts = resp2.json().get("data", {}).get("markets", [])
            except:
                continue

            for m in mkts:
                o1 = m.get("outcomeOneName", "")
                o2 = m.get("outcomeTwoName", "")
                l1 = m.get("outcomeVigOneLine", "") or m.get("outcomeOneLine", "")
                l2 = m.get("outcomeVigTwoLine", "") or m.get("outcomeTwoLine", "")
                line = m.get("line", "")
                mtype = m.get("type", "")

                p1 = american_to_prob(l1)
                p2 = american_to_prob(l2)

                # Determine market type name
                if mtype in (226,): type_name = "Moneyline"
                elif mtype in (28,): type_name = f"Total {line}"
                elif mtype in (342,): type_name = f"Spread {line}"
                elif mtype in (63,): type_name = "1H ML"
                elif mtype in (202, 203, 204): type_name = f"Q{mtype-201} ML"
                else: type_name = f"Type{mtype}"

                markets.append({
                    "source": "SX Bet",
                    "question": f"{t1} vs {t2}: {type_name}",
                    "game": f"{t1} vs {t2}",
                    "date": date,
                    "outcome1": o1,
                    "outcome2": o2,
                    "prob1": p1,
                    "prob2": p2,
                    "odds1": l1,
                    "odds2": l2,
                    "line": line,
                    "type": type_name,
                })
    except Exception as e:
        print(f"  SX Bet error: {e}")

    return markets


def fetch_pinnacle_nba():
    """Fetch Pinnacle NBA odds via The Odds API."""
    if not ODDS_KEY: return []
    markets = []
    try:
        resp = requests.get(f"{ODDS_API}/sports/basketball_nba/odds", params={
            "apiKey": ODDS_KEY, "regions": "us,eu",
            "markets": "h2h,spreads,totals",
            "bookmakers": "pinnacle", "oddsFormat": "decimal",
        }, timeout=15)
        if not resp.ok: return []

        for event in resp.json():
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            for bm in event.get("bookmakers", []):
                if bm.get("key") != "pinnacle": continue
                for mkt in bm.get("markets", []):
                    mtype = mkt.get("key", "")
                    for o in mkt.get("outcomes", []):
                        name = o.get("name", "")
                        odds = float(o.get("price", 2.0))
                        prob = round(1.0 / odds, 3)
                        point = o.get("point", "")
                        markets.append({
                            "source": "Pinnacle",
                            "game": f"{away} @ {home}",
                            "type": mtype,
                            "outcome": name,
                            "prob": prob,
                            "odds": odds,
                            "point": point,
                        })
    except: pass
    return markets


def find_arbitrage(poly, sx, pinnacle):
    """Compare prices across platforms and find arbitrage."""
    opportunities = []

    # Match Polymarket vs SX Bet by game + market type
    for pm in poly:
        slug = pm["slug"]
        # Extract game info from slug: nba-AWAY-HOME-DATE-TYPE
        parts = slug.split("-")
        if len(parts) < 5: continue
        poly_away = parts[1].upper()
        poly_home = parts[2].upper()

        for sx_m in sx:
            # Match by team names
            sx_game = sx_m.get("game", "").lower()
            if poly_away.lower()[:3] not in sx_game and poly_home.lower()[:3] not in sx_game:
                continue

            # Match by market type
            poly_type = ""
            if "total" in slug: poly_type = "total"
            elif "spread" in slug: poly_type = "spread"
            elif "moneyline" in slug or "winner" in slug: poly_type = "moneyline"
            else: continue

            sx_type = sx_m.get("type", "").lower()
            if poly_type not in sx_type.lower() and sx_type.lower()[:4] not in poly_type:
                continue

            # Compare probabilities
            poly_p = pm["prob1"]
            sx_p = sx_m["prob1"]

            if poly_p > 0 and sx_p > 0 and poly_p != 0.5 and sx_p != 0.5:
                diff = abs(poly_p - sx_p)
                if diff > 0.03:  # 3% minimum edge
                    if poly_p < sx_p:
                        buy_on = "Polymarket"
                        buy_price = poly_p
                        fair_price = sx_p
                    else:
                        buy_on = "SX Bet"
                        buy_price = sx_p
                        fair_price = poly_p

                    opportunities.append({
                        "game": sx_m.get("game", pm["question"]),
                        "market": poly_type.upper(),
                        "polymarket": round(poly_p, 3),
                        "sxbet": round(sx_p, 3),
                        "edge": round(diff, 3),
                        "buy_on": buy_on,
                        "buy_price": round(buy_price, 3),
                        "fair_price": round(fair_price, 3),
                        "poly_question": pm["question"],
                    })

    # Also compare with Pinnacle
    for pm in poly:
        slug = pm["slug"]
        parts = slug.split("-")
        if len(parts) < 5: continue

        for pin in pinnacle:
            pin_game = pin.get("game", "").lower()
            # Rough match
            if parts[1][:3] not in pin_game and parts[2][:3] not in pin_game:
                continue

            poly_p = pm["prob1"]
            pin_p = pin["prob"]

            if abs(poly_p - pin_p) > 0.03:
                opportunities.append({
                    "game": pin.get("game", pm["question"]),
                    "market": pin.get("type", "").upper(),
                    "polymarket": round(poly_p, 3),
                    "pinnacle": round(pin_p, 3),
                    "edge": round(abs(poly_p - pin_p), 3),
                    "buy_on": "Polymarket" if poly_p < pin_p else "Pinnacle implied",
                    "buy_price": round(min(poly_p, pin_p), 3),
                    "fair_price": round(max(poly_p, pin_p), 3),
                    "poly_question": pm["question"],
                })

    opportunities.sort(key=lambda x: x["edge"], reverse=True)
    return opportunities


def run_scan():
    """Run one scan cycle."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"  CROSS-PLATFORM ARB SCAN | {now.strftime('%H:%M:%S UTC')}")
    print(f"{'='*70}")

    # Fetch from all platforms
    print("  Fetching Polymarket...", end=" ")
    poly = fetch_polymarket_nba()
    print(f"{len(poly)} markets")

    print("  Fetching SX Bet...", end=" ")
    sx = fetch_sxbet_nba()
    print(f"{len(sx)} markets")

    print("  Fetching Pinnacle...", end=" ")
    pin = fetch_pinnacle_nba()
    print(f"{len(pin)} odds")

    # Find arbitrage
    opps = find_arbitrage(poly, sx, pin)

    # Display
    print(f"\n  OPPORTUNITIES: {len(opps)}")
    if opps:
        print(f"  {'─'*65}")
        for o in opps[:10]:
            pm = o.get("polymarket", 0)
            sx_p = o.get("sxbet", o.get("pinnacle", 0))
            source = "SXBet" if "sxbet" in o else "Pinnacle"
            print(f"  {o['game'][:40]}")
            print(f"    {o['market']} | Poly: {pm:.1%} vs {source}: {sx_p:.1%} | EDGE: {o['edge']:.1%}")
            print(f"    BUY on {o['buy_on']} at {o['buy_price']:.1%}")
    else:
        print("  No arbitrage found. Markets are efficiently priced right now.")

    # Save
    log_path = LOG_DIR / f"crossplatform_{now.strftime('%Y%m%d_%H%M')}.json"
    with open(log_path, "w") as f:
        json.dump({"time": now.isoformat(), "opportunities": opps,
                   "poly_count": len(poly), "sx_count": len(sx), "pin_count": len(pin)}, f, indent=2)

    return opps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run every 3 minutes")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    print("="*70)
    print("  CROSS-PLATFORM ARBITRAGE SCANNER")
    print("  Polymarket vs SX Bet vs Pinnacle")
    print("="*70)

    if args.loop:
        while True:
            try:
                run_scan()
                print(f"\n  Next scan in 3 minutes...")
                time.sleep(180)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(60)
    else:
        run_scan()


if __name__ == "__main__":
    main()
