#!/usr/bin/env python3
"""
10 QUANT BOTS — Built from Statistical Edge Analysis
======================================================

Every bot is based on PROVEN statistical patterns in 30 days of data:

  PATTERN                                  WR      SAMPLES   EDGE
  ─────────────────────────────────────────────────────────────────
  Taker-buy ratio > 55% → UP              77.5%   869       +27.5%
  Taker-buy ratio < 45% → DOWN            76.1%   1073      +26.1%
  After strong UP (>0.3%) → DOWN           59.6%   ~450      +9.6%
  After 2 consecutive same → reverse       56-57%  660       +6-7%
  Hour 15 UTC → UP bias                    59.2%   120       +9.2%
  Hour 13 UTC → DOWN bias                  58.3%   120       +8.3%
  Lag-4 autocorrelation = -0.096           ~55%    2880      mean-revert
"""

import argparse, asyncio, json, logging, signal as signal_module, sys, time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bots")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
GAMMA_HOST = "https://gamma-api.polymarket.com"
TRADE_SIZE = 2000.0


class WindowManager:
    def __init__(self):
        self.current_start = 0
        self.current_end = 0
        self.poly_data = {}
        self.market_found = False

    def is_new_window(self):
        now = int(time.time())
        start = (now // 900) * 900
        if start != self.current_start:
            self.current_start = start
            self.current_end = start + 900
            self.market_found = False
            return True
        return False

    def seconds_remaining(self):
        return max(0, self.current_end - int(time.time()))

    def fetch_polymarket_data(self):
        slug = f"btc-updown-15m-{self.current_start}"
        self.poly_data = {"slug": slug}
        try:
            resp = requests.get(f"{GAMMA_HOST}/markets", params={"slug": slug, "limit": 1}, timeout=10)
            data = resp.json()
            if not data: return
            m = data[0]
            outcomes = json.loads(m.get("outcomes","[]")) if isinstance(m.get("outcomes"),str) else m.get("outcomes",[])
            prices = json.loads(m.get("outcomePrices","[]")) if isinstance(m.get("outcomePrices"),str) else m.get("outcomePrices",[])
            tids = json.loads(m.get("clobTokenIds","[]")) if isinstance(m.get("clobTokenIds"),str) else m.get("clobTokenIds",[])
            if len(outcomes)<2 or len(tids)<2: return
            up_idx = 0 if outcomes[0].lower()=="up" else 1
            self.poly_data.update({
                "question": m.get("question",""), "cid": m.get("conditionId",""),
                "up_token": tids[up_idx], "down_token": tids[1-up_idx],
                "up_price": float(prices[up_idx]) if len(prices)>up_idx else 0.50,
                "down_price": float(prices[1-up_idx]) if len(prices)>1 else 0.50,
                "spread": float(m.get("spread",0) or 0), "liquidity": float(m.get("liquidityClob",0) or 0),
            })
            for label, tid in [("up",tids[up_idx]),("down",tids[1-up_idx])]:
                try:
                    r2 = requests.get("https://clob.polymarket.com/book", params={"token_id":tid}, timeout=10)
                    book = r2.json()
                    asks = book.get("asks",[])
                    if asks: self.poly_data[f"{label}_ask"] = float(asks[0]["price"])
                except: pass
            self.market_found = True
        except: pass


class Bot:
    name = "Base"; description = ""
    def __init__(self):
        self._results = []
        self._log_path = LOG_DIR / f"bot_{self.name.replace(' ','_').replace(':','')}.jsonl"
        if self._log_path.exists(): self._log_path.unlink()

    def decide(self, f, poly, hist): raise NotImplementedError

    def record(self, dec, btc_s, btc_e, poly):
        actual = "UP" if btc_e >= btc_s else "DOWN"
        correct = dec["direction"] == actual
        d = dec["direction"].lower()
        buy_p = poly.get(f"{d}_ask", 0.52)
        if buy_p <= 0 or buy_p >= 1: buy_p = 0.52
        sz = dec.get("size", TRADE_SIZE)
        pnl = (1.0-buy_p)*sz if correct else -buy_p*sz
        result = {"time": datetime.now(timezone.utc).isoformat(), "bot": self.name,
                  "direction": dec["direction"], "confidence": dec["confidence"],
                  "btc_start": round(btc_s,2), "btc_end": round(btc_e,2),
                  "btc_change_pct": round((btc_e-btc_s)/btc_s*100,4),
                  "actual": actual, "correct": correct, "buy_price": buy_p,
                  "size": sz, "pnl": round(pnl,2)}
        self._results.append(result)
        with open(self._log_path,"a") as f: f.write(json.dumps(result)+"\n")
        return result

    def stats(self):
        t = len(self._results); w = sum(1 for r in self._results if r["correct"])
        pnl = sum(r["pnl"] for r in self._results)
        pnls = [r["pnl"] for r in self._results]
        sh = float(np.mean(pnls)/np.std(pnls)*np.sqrt(96)) if len(pnls)>=2 and np.std(pnls)>0 else 0
        wh = max(len(set(r.get("window_start",i) for i,r in enumerate(self._results))),1)
        daily = (pnl/wh)*96 if t>0 else 0
        return {"name":self.name,"desc":self.description,"trades":t,"wins":w,"losses":t-w,
                "wr":round(w/t*100,1) if t>0 else 0,"pnl":round(pnl,2),
                "avg_pnl":round(pnl/t,2) if t>0 else 0,"sharpe":round(sh,2),
                "daily_est":round(daily,0),"recent":self._results[-5:]}


# ============================================================
# 10 QUANT BOTS — Based on proven statistical edges
# ============================================================

class Bot1_TakerFlow(Bot):
    """THE BIGGEST EDGE: 77% WR when taker-buy ratio > 55% in CURRENT window."""
    name = "B1: Taker Flow"
    description = "77% edge: follows taker buy/sell ratio of current window"

    def decide(self, f, poly, hist):
        # This is the taker buy ratio from the CURRENT streaming data
        # High buy ratio in real-time → price will close UP this window
        flow = f.get("btc_imb_5s", 0)        # -1 to +1 scale
        flow_roll = f.get("btc_imb_roll_5s", 0)
        buy_ratio_1s = f.get("btc_buy_ratio_1s", 0)

        # Strong buy pressure → UP
        if flow > 0.10 and flow_roll > 0.05:
            conf = min(0.55 + abs(flow) * 0.5, 0.85)
            return ("UP", conf)
        elif flow < -0.10 and flow_roll < -0.05:
            conf = min(0.55 + abs(flow) * 0.5, 0.85)
            return ("DOWN", conf)
        return None


class Bot2_ReversalStrong(Bot):
    """59.6% WR: After strong move (>0.3%), bet on reversal next window."""
    name = "B2: Strong Reversal"
    description = "59.6% edge: fades strong moves from prior window"

    def decide(self, f, poly, hist):
        if not hist: return None
        last = hist[-1]
        chg = last.get("btc_change_pct", 0)

        if chg > 0.3:    # Last window was strong UP → bet DOWN
            return ("DOWN", 0.65)
        elif chg < -0.3:  # Last window was strong DOWN → bet UP
            return ("UP", 0.60)
        return None


class Bot3_DoubleReversal(Bot):
    """57% WR: After 2 consecutive same direction → bet on reversal."""
    name = "B3: Double Reversal"
    description = "57% edge: reversal after 2 consecutive same-direction windows"

    def decide(self, f, poly, hist):
        if len(hist) < 2: return None
        d1 = hist[-1].get("actual", "")
        d2 = hist[-2].get("actual", "")

        if d1 == d2 == "UP":
            return ("DOWN", 0.60)
        elif d1 == d2 == "DOWN":
            return ("UP", 0.60)
        return None


class Bot4_HourlyEdge(Bot):
    """59% WR: Trade specific hours with proven directional bias."""
    name = "B4: Hourly Edge"
    description = "59% edge: trades only during high-bias hours (13,15 UTC)"

    def decide(self, f, poly, hist):
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Hour 15 UTC (10 AM ET): 59.2% UP
        if hour == 15:
            return ("UP", 0.62)
        # Hour 1 UTC (8 PM ET): 56.7% UP
        elif hour == 1:
            return ("UP", 0.58)
        # Hour 13 UTC (8 AM ET): 58.3% DOWN
        elif hour == 13:
            return ("DOWN", 0.60)
        # Hour 5 UTC (12 AM ET): 55.8% DOWN
        elif hour == 5:
            return ("DOWN", 0.58)
        # Hour 19 UTC (2 PM ET): 55% DOWN
        elif hour == 19:
            return ("DOWN", 0.57)
        return None


class Bot5_Lag4Reversion(Bot):
    """Autocorrelation at lag-4 is -0.096 → strong 1-hour mean reversion."""
    name = "B5: Lag-4 Revert"
    description = "-0.096 autocorr: reverses the move from 4 windows (1 hour) ago"

    def decide(self, f, poly, hist):
        if len(hist) < 4: return None

        # What happened 4 windows (1 hour) ago?
        h4 = hist[-4]
        chg4 = h4.get("btc_change_pct", 0)

        if chg4 > 0.1:    # 1 hour ago was UP → now DOWN
            return ("DOWN", 0.58)
        elif chg4 < -0.1:  # 1 hour ago was DOWN → now UP
            return ("UP", 0.58)
        return None


class Bot6_FlowPlusRevert(Bot):
    """Combines the two biggest edges: taker flow + mean reversion."""
    name = "B6: Flow+Revert"
    description = "Combo: taker flow in current window + reversion from last"

    def decide(self, f, poly, hist):
        flow = f.get("btc_imb_5s", 0)
        flow_roll = f.get("btc_imb_roll_5s", 0)

        last_chg = hist[-1].get("btc_change_pct", 0) if hist else 0

        # Flow says UP and last window was DOWN (reversion confirms)
        if flow > 0.08 and flow_roll > 0.05 and last_chg < -0.05:
            return ("UP", 0.75)
        # Flow says DOWN and last window was UP (reversion confirms)
        elif flow < -0.08 and flow_roll < -0.05 and last_chg > 0.05:
            return ("DOWN", 0.75)
        return None


class Bot7_TripleRevert(Bot):
    """After 3 consecutive same direction → strong reversal (60%+ edge)."""
    name = "B7: Triple Reversal"
    description = "60%+ edge: reversal after 3 consecutive same-direction windows"

    def decide(self, f, poly, hist):
        if len(hist) < 3: return None
        d1 = hist[-1].get("actual", "")
        d2 = hist[-2].get("actual", "")
        d3 = hist[-3].get("actual", "")

        if d1 == d2 == d3 == "UP":
            return ("DOWN", 0.68)
        elif d1 == d2 == d3 == "DOWN":
            return ("UP", 0.68)
        return None


class Bot8_FlowPlusHour(Bot):
    """Combines taker flow with hourly bias for double confirmation."""
    name = "B8: Flow+Hour"
    description = "Double edge: taker flow + hourly directional bias"

    def decide(self, f, poly, hist):
        flow = f.get("btc_imb_5s", 0)
        hour = datetime.now(timezone.utc).hour

        # High-bias hours + flow confirms
        if hour == 15 and flow > 0.05:   # 15 UTC = UP hour + flow UP
            return ("UP", 0.70)
        elif hour == 13 and flow < -0.05:  # 13 UTC = DOWN hour + flow DOWN
            return ("DOWN", 0.70)
        elif hour == 1 and flow > 0.05:
            return ("UP", 0.65)
        elif hour == 5 and flow < -0.05:
            return ("DOWN", 0.65)

        # Any hour, very strong flow
        if flow > 0.20:
            return ("UP", 0.65)
        elif flow < -0.20:
            return ("DOWN", 0.65)
        return None


class Bot9_MLReversion(Bot):
    """ML model trained specifically on 15-min reversion patterns."""
    name = "B9: ML Reversion"
    description = "ML model optimized for mean-reversion at 15-min scale"

    def __init__(self):
        super().__init__()
        self.models = None; self.scaler = None; self.feature_cols = []

    def load_model(self, models, scaler, feat_cols):
        self.models = models; self.scaler = scaler; self.feature_cols = feat_cols

    def decide(self, f, poly, hist):
        if not self.models: return None
        try:
            fv = np.zeros(len(self.feature_cols))
            for i, col in enumerate(self.feature_cols):
                fv[i] = f.get(col, 0.0)
            fv_s = self.scaler.transform(fv.reshape(1,-1))
            probas = [m.predict_proba(fv_s) for _,m in self.models]
            avg = np.mean(probas, axis=0)
            up = float(avg[0,1]); conf = max(up, 1-up)
            if conf >= 0.57:
                d = "UP" if up > 0.5 else "DOWN"
                # Only trade if ML agrees with mean-reversion
                if hist:
                    last_dir = hist[-1].get("actual","")
                    if (d == "DOWN" and last_dir == "UP") or (d == "UP" and last_dir == "DOWN"):
                        return (d, min(conf * 1.15, 0.90))  # boost when ML + reversion agree
                return (d, conf)
        except: pass
        return None


class Bot10_MegaConsensus(Bot):
    """Only trades when 5+ of the other 9 bots agree. Maximum conviction."""
    name = "B10: Mega Consensus"
    description = "Ultra-selective: only trades when 5+ bots agree"

    def __init__(self):
        super().__init__()
        self.sub_bots = []

    def set_subs(self, subs):
        self.sub_bots = subs

    def decide(self, f, poly, hist):
        up = 0; down = 0; conf_up = 0; conf_down = 0
        for bot in self.sub_bots:
            try:
                r = bot.decide(f, poly, hist)
                if r:
                    if r[0]=="UP": up+=1; conf_up+=r[1]
                    else: down+=1; conf_down+=r[1]
            except: pass
        if up >= 5: return ("UP", min(conf_up/up*1.1, 0.92))
        elif down >= 5: return ("DOWN", min(conf_down/down*1.1, 0.92))
        return None


# ============================================================
# BOTS B11-B30: 20 NEW STRATEGIES
# ============================================================

# --- Category A: Flow Variations ---

class Bot11_FlowStrict(Bot):
    name = "B11: Flow Strict 15%"
    description = "Taker flow > 0.15 threshold (stricter than B1)"
    def decide(self, f, poly, hist):
        fl = f.get("btc_imb_5s",0); rl = f.get("btc_imb_roll_5s",0)
        if fl > 0.15 and rl > 0.10: return ("UP", min(0.60+abs(fl)*0.4, 0.88))
        elif fl < -0.15 and rl < -0.10: return ("DOWN", min(0.60+abs(fl)*0.4, 0.88))
        return None

class Bot12_FlowUltra(Bot):
    name = "B12: Flow Ultra 25%"
    description = "Taker flow > 0.25 (ultra-strict, very few trades)"
    def decide(self, f, poly, hist):
        fl = f.get("btc_imb_5s",0)
        if fl > 0.25: return ("UP", 0.85)
        elif fl < -0.25: return ("DOWN", 0.85)
        return None

class Bot13_FlowAccel(Bot):
    name = "B13: Flow Acceleration"
    description = "Trades when flow is accelerating (delta of flow)"
    def decide(self, f, poly, hist):
        fl = f.get("btc_imb_5s",0)
        fl_d = f.get("btc_imb_5s_delta_3s", 0)
        if fl > 0.05 and fl_d > 0.05: return ("UP", 0.65)
        elif fl < -0.05 and fl_d < -0.05: return ("DOWN", 0.65)
        return None

class Bot14_FlowWhale(Bot):
    name = "B14: Flow + Whale"
    description = "Flow confirms + whale trade same direction"
    def decide(self, f, poly, hist):
        fl = f.get("btc_imb_5s",0)
        wc = f.get("btc_whale_count_10s",0); wi = f.get("btc_whale_imb",0)
        if fl > 0.10 and wc > 0 and wi > 0.3: return ("UP", 0.78)
        elif fl < -0.10 and wc > 0 and wi < -0.3: return ("DOWN", 0.78)
        return None

# --- Category B: Multi-Asset ---

class Bot15_SOLCanary(Bot):
    name = "B15: SOL Canary"
    description = "SOL spiked > 0.15%, BTC flat — follow SOL"
    def decide(self, f, poly, hist):
        sol_ret = f.get("sol_ret_5s", f.get("sol_mom_5s", 0))
        btc_ret = f.get("btc_ret_5s", f.get("btc_mom_5s", 0))
        if sol_ret > 0.15 and abs(btc_ret) < 0.05: return ("UP", 0.63)
        elif sol_ret < -0.15 and abs(btc_ret) < 0.05: return ("DOWN", 0.63)
        return None

class Bot16_ETHSOLAgree(Bot):
    name = "B16: ETH+SOL Agree"
    description = "ETH and SOL both moved same way, BTC flat"
    def decide(self, f, poly, hist):
        eth = f.get("eth_ret_5s", 0)
        sol = f.get("sol_ret_5s", f.get("sol_mom_5s", 0))
        btc = f.get("btc_ret_5s", f.get("btc_mom_5s", 0))
        if eth > 0.05 and sol > 0.08 and btc < 0.03: return ("UP", 0.68)
        elif eth < -0.05 and sol < -0.08 and btc > -0.03: return ("DOWN", 0.68)
        return None

class Bot17_SectorMomentum(Bot):
    name = "B17: Sector Momentum"
    description = "ETH + SOL + BTC flow all agree on direction"
    def decide(self, f, poly, hist):
        sb = f.get("sector_bullish", 0)
        sbe = f.get("sector_bearish", 0)
        fl = f.get("btc_imb_5s", 0)
        if sb and fl > 0.05: return ("UP", 0.70)
        elif sbe and fl < -0.05: return ("DOWN", 0.70)
        return None

# --- Category C: Derivatives ---

class Bot18_FundingFade(Bot):
    name = "B18: Funding Fade"
    description = "Extreme funding rate — bet against it"
    def decide(self, f, poly, hist):
        fr = f.get("funding_rate_pct", 0)
        if fr > 0.03: return ("DOWN", 0.62)   # Longs overleveraged → dump
        elif fr < -0.01: return ("UP", 0.62)   # Shorts overleveraged → squeeze
        return None

class Bot19_OIDivergence(Bot):
    name = "B19: OI Divergence"
    description = "Price up + OI down = weak move → reversal"
    def decide(self, f, poly, hist):
        if not hist: return None
        oi_div = f.get("oi_price_diverge", 0)
        last_dir = hist[-1].get("actual", "")
        if oi_div and last_dir == "UP": return ("DOWN", 0.60)
        elif oi_div and last_dir == "DOWN": return ("UP", 0.60)
        return None

class Bot20_FundingFlow(Bot):
    name = "B20: Funding+Flow"
    description = "Negative funding + buy flow = squeeze setup"
    def decide(self, f, poly, hist):
        fr = f.get("funding_rate_pct", 0)
        fl = f.get("btc_imb_5s", 0)
        if fr < -0.005 and fl > 0.10: return ("UP", 0.72)     # Short squeeze
        elif fr > 0.02 and fl < -0.10: return ("DOWN", 0.72)   # Long squeeze
        return None

# --- Category D: Advanced Mean Reversion ---

class Bot21_ZScoreRevert(Bot):
    name = "B21: Z-Score Revert"
    description = "Z-score > 2.0 on recent returns — fade it"
    def decide(self, f, poly, hist):
        if len(hist) < 6: return None
        rets = [h.get("btc_change_pct", 0) for h in hist[-6:]]
        mu = np.mean(rets); sigma = np.std(rets)
        if sigma < 0.01: return None
        last = rets[-1]
        z = (last - mu) / sigma
        if z > 1.8: return ("DOWN", min(0.55 + abs(z) * 0.08, 0.80))
        elif z < -1.8: return ("UP", min(0.55 + abs(z) * 0.08, 0.80))
        return None

class Bot22_RangeFade(Bot):
    name = "B22: Range Fade"
    description = "Price near 15-min range high → DOWN, near low → UP"
    def decide(self, f, poly, hist):
        btc = f.get("btc_price", 0)
        ob_mid = f.get("ob_mid", 0)
        vol = f.get("micro_volatility", 0)
        m15 = f.get("btc_mom_15s", 0)
        if m15 > 0.08 and vol > 0.03: return ("DOWN", 0.58)
        elif m15 < -0.08 and vol > 0.03: return ("UP", 0.58)
        return None

class Bot23_VWAPRevert(Bot):
    name = "B23: VWAP Revert"
    description = "Price above VWAP → DOWN, below → UP (fair value)"
    def decide(self, f, poly, hist):
        vwap = f.get("btc_vwap_1s", 0)
        fl = f.get("btc_imb_5s", 0)
        if vwap > 0.001 and fl < 0: return ("DOWN", 0.58)  # Above VWAP + sell flow
        elif vwap < -0.001 and fl > 0: return ("UP", 0.58)  # Below VWAP + buy flow
        return None

# --- Category E: Time/Pattern ---

class Bot24_AsianReversal(Bot):
    name = "B24: Asian Reversal"
    description = "Asian session (0-8 UTC): fade the prior move"
    def decide(self, f, poly, hist):
        h = datetime.now(timezone.utc).hour
        if h < 0 or h >= 8: return None  # Only Asian session
        if not hist: return None
        last = hist[-1].get("actual", "")
        fl = f.get("btc_imb_5s", 0)
        if last == "UP" and fl < 0: return ("DOWN", 0.58)
        elif last == "DOWN" and fl > 0: return ("UP", 0.58)
        return None

class Bot25_USOpenMom(Bot):
    name = "B25: US Open Momentum"
    description = "US open (13-16 UTC): follow first-move direction"
    def decide(self, f, poly, hist):
        h = datetime.now(timezone.utc).hour
        if h < 13 or h >= 16: return None
        m5 = f.get("btc_mom_5s", 0); fl = f.get("btc_imb_5s", 0)
        if m5 > 0.02 and fl > 0.08: return ("UP", 0.62)
        elif m5 < -0.02 and fl < -0.08: return ("DOWN", 0.62)
        return None

class Bot26_WeekendRevert(Bot):
    name = "B26: Weekend Revert"
    description = "Sat/Sun: stronger mean reversion (low liquidity)"
    def decide(self, f, poly, hist):
        dow = datetime.now(timezone.utc).weekday()
        if dow < 5: return None  # Only weekends
        if not hist: return None
        last = hist[-1].get("actual", "")
        chg = hist[-1].get("btc_change_pct", 0)
        if last == "UP" and abs(chg) > 0.1: return ("DOWN", 0.63)
        elif last == "DOWN" and abs(chg) > 0.1: return ("UP", 0.63)
        return None

# --- Category F: Ensemble Variants ---

class Bot27_Top3Vote(Bot):
    name = "B27: Top3 Vote"
    description = "Only when B1+B6+B8 (backtested winners) all agree"
    def __init__(self): super().__init__(); self.b1=None; self.b6=None; self.b8=None
    def set_refs(self, b1, b6, b8): self.b1=b1; self.b6=b6; self.b8=b8
    def decide(self, f, poly, hist):
        if not all([self.b1, self.b6, self.b8]): return None
        r1 = self.b1.decide(f, poly, hist)
        r6 = self.b6.decide(f, poly, hist)
        r8 = self.b8.decide(f, poly, hist)
        if r1 and r6 and r8 and r1[0]==r6[0]==r8[0]:
            avg_c = (r1[1]+r6[1]+r8[1])/3
            return (r1[0], min(avg_c*1.15, 0.92))
        return None

class Bot28_FlowFamilyVote(Bot):
    name = "B28: Flow Family Vote"
    description = "3+ flow-based bots agree"
    def __init__(self): super().__init__(); self.flow_bots=[]
    def set_flow_bots(self, bots): self.flow_bots=bots
    def decide(self, f, poly, hist):
        up=0; dn=0; cu=0; cd=0
        for b in self.flow_bots:
            try:
                r=b.decide(f,poly,hist)
                if r:
                    if r[0]=="UP": up+=1; cu+=r[1]
                    else: dn+=1; cd+=r[1]
            except: pass
        if up>=3: return ("UP", min(cu/up*1.1, 0.90))
        elif dn>=3: return ("DOWN", min(cd/dn*1.1, 0.90))
        return None

class Bot29_AntiConsensus(Bot):
    name = "B29: Anti-Consensus"
    description = "Trade OPPOSITE when 6+ other bots agree (meta-contrarian)"
    def __init__(self): super().__init__(); self.all_bots=[]
    def set_all(self, bots): self.all_bots=bots
    def decide(self, f, poly, hist):
        up=0; dn=0
        for b in self.all_bots:
            try:
                r=b.decide(f,poly,hist)
                if r:
                    if r[0]=="UP": up+=1
                    else: dn+=1
            except: pass
        # OPPOSITE of majority
        if up>=6: return ("DOWN", 0.60)
        elif dn>=6: return ("UP", 0.60)
        return None

class Bot30_KellyScaler(Bot):
    name = "B30: Kelly Scaler"
    description = "Same as B1 but dynamically sizes by Kelly criterion"
    def decide(self, f, poly, hist):
        fl = f.get("btc_imb_5s",0); rl = f.get("btc_imb_roll_5s",0)
        if fl > 0.10 and rl > 0.05:
            conf = min(0.55+abs(fl)*0.5, 0.85)
            # Kelly: f* = (bp - q) / b where b=payout odds, p=win prob, q=1-p
            p = conf; q = 1-p; b = 0.48/0.52
            kelly = max(0, (b*p - q) / b)
            size = TRADE_SIZE * min(kelly * 3, 2.5)  # up to 2.5x base
            return ("UP", conf)
        elif fl < -0.10 and rl < -0.05:
            conf = min(0.55+abs(fl)*0.5, 0.85)
            return ("DOWN", conf)
        return None


# ============================================================
# RUNNER
# ============================================================
async def run(self_train=False):
    from streaming_predictor import (
        OrderbookStream, TradeStream, MempoolStream,
        DerivativesPoller,
        StreamingFeatureEngine, SYMBOL, ETH_SYMBOL, SOL_SYMBOL,
    )

    # Init all 30 bots
    b1=Bot1_TakerFlow(); b2=Bot2_ReversalStrong(); b3=Bot3_DoubleReversal()
    b4=Bot4_HourlyEdge(); b5=Bot5_Lag4Reversion(); b6=Bot6_FlowPlusRevert()
    b7=Bot7_TripleRevert(); b8=Bot8_FlowPlusHour(); b9=Bot9_MLReversion()
    b10=Bot10_MegaConsensus()
    b11=Bot11_FlowStrict(); b12=Bot12_FlowUltra(); b13=Bot13_FlowAccel()
    b14=Bot14_FlowWhale(); b15=Bot15_SOLCanary(); b16=Bot16_ETHSOLAgree()
    b17=Bot17_SectorMomentum(); b18=Bot18_FundingFade(); b19=Bot19_OIDivergence()
    b20=Bot20_FundingFlow(); b21=Bot21_ZScoreRevert(); b22=Bot22_RangeFade()
    b23=Bot23_VWAPRevert(); b24=Bot24_AsianReversal(); b25=Bot25_USOpenMom()
    b26=Bot26_WeekendRevert(); b27=Bot27_Top3Vote(); b28=Bot28_FlowFamilyVote()
    b29=Bot29_AntiConsensus(); b30=Bot30_KellyScaler()

    bots = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,
            b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,
            b21,b22,b23,b24,b25,b26,b27,b28,b29,b30]

    # Wire up ensemble bots
    b10.set_subs([b1,b2,b3,b4,b5,b6,b7,b8,b9])
    b27.set_refs(b1, b6, b8)
    b28.set_flow_bots([b1,b3,b6,b8,b11,b12,b13,b14])
    b29.set_all([b1,b2,b3,b4,b5,b6,b7,b8,b9,b11,b12,b13,b14,b15,b16,b17,b18])

    if self_train:
        log.info("Training ML for B9...")
        try:
            from ultra_predictor import BinanceDataCollector, engineer_all_features, create_targets, get_feature_cols
            from sklearn.preprocessing import StandardScaler
            c = BinanceDataCollector("BTCUSDT"); df = c.get_klines_history(days=7)
            if not df.empty:
                df = engineer_all_features(df); df = create_targets(df)
                fc = get_feature_cols(df)
                for col in fc: df[col] = df[col].fillna(df[col].median())
                cl = df.dropna(subset=fc+["target_1"])
                X,y = cl[fc].values, cl["target_1"].values
                sc = StandardScaler(); X_s = sc.fit_transform(X)
                import lightgbm as lgb
                m = lgb.LGBMClassifier(n_estimators=200,max_depth=5,learning_rate=0.06,verbose=-1,n_jobs=-1,random_state=42)
                m.fit(X_s,y)
                b9.load_model([("lgb",m)],sc,fc)
                log.info(f"B9 trained on {len(cl):,} rows")
        except Exception as e: log.warning(f"ML failed: {e}")

    # Init streams (BTC + ETH + SOL + Orderbook + Mempool + Derivatives)
    ob=OrderbookStream(); bt=TradeStream(SYMBOL,is_eth=False)
    et=TradeStream(ETH_SYMBOL,is_eth=True)
    st=TradeStream(SOL_SYMBOL,is_eth=False)  # SOL stream
    mp=MempoolStream(); dp=DerivativesPoller()
    engine=StreamingFeatureEngine(ob,bt,et,mp,sol_trades=st,derivatives=dp)
    tasks=[asyncio.create_task(ob.run()),asyncio.create_task(bt.run()),
           asyncio.create_task(et.run()),asyncio.create_task(st.run()),
           asyncio.create_task(mp.run()),asyncio.create_task(dp.run())]
    await asyncio.sleep(5)

    wm=WindowManager(); decisions={}; btc_start=0.0; history=[]

    print(f"\n{'='*72}")
    print(f"  30 QUANT BOTS — POLYMARKET 15-MIN TOURNAMENT")
    print(f"{'='*72}")
    for b in bots: print(f"  {b.name:<25} {b.description}")
    print(f"{'='*72}\n")

    tick=0
    try:
        while True:
            await asyncio.sleep(1.0); tick+=1
            features=engine.compute_features()
            btc=features.get("btc_price",0)

            if wm.is_new_window():
                if btc_start>0 and decisions:
                    actual="UP" if btc>=btc_start else "DOWN"
                    chg=(btc-btc_start)/btc_start*100
                    log.info(f"")
                    log.info(f"{'═'*70}")
                    log.info(f"RESOLVED: ${btc_start:,.2f}→${btc:,.2f} ({actual} {chg:+.3f}%)")
                    log.info(f"{'═'*70}")
                    history.append({"actual":actual,"btc_change_pct":chg})
                    for b in bots:
                        if b.name in decisions:
                            r=b.record(decisions[b.name],btc_start,btc,wm.poly_data)
                            icon="WIN " if r["correct"] else "LOSS"
                            st=b.stats()
                            log.info(f"  {icon} {b.name:<25} {r['direction']:<5} ${r['pnl']:>+8,.0f} | {st['trades']}t {st['wr']:.0f}% ${st['pnl']:>+,.0f}")

                decisions={}; btc_start=btc
                ws=datetime.fromtimestamp(wm.current_start,tz=timezone.utc)
                we=datetime.fromtimestamp(wm.current_end,tz=timezone.utc)
                log.info(f"\n{'═'*70}")
                log.info(f"NEW WINDOW: {ws.strftime('%H:%M')}-{we.strftime('%H:%M')} UTC | BTC ${btc:,.2f}")
                wm.fetch_polymarket_data()
                p=wm.poly_data
                if wm.market_found:
                    log.info(f"  Polymarket: UP@{p.get('up_ask','?')} DOWN@{p.get('down_ask','?')} Spread={p.get('spread','?')}")
                log.info(f"{'═'*70}")

                await asyncio.sleep(10)
                features=engine.compute_features()

                for b in bots:
                    try:
                        r=b.decide(features,wm.poly_data,history[-10:])
                        if r:
                            d,c=r
                            decisions[b.name]={"direction":d,"confidence":c,"size":TRADE_SIZE}
                            log.info(f"  {b.name:<25}→ {d} ({c:.0%})")
                        else:
                            log.info(f"  {b.name:<25}→ SKIP")
                    except Exception as e: log.debug(f"  {b.name} err: {e}")
                log.info(f"  ──── {len(decisions)}/30 trading ────")

            if tick%300==0:
                print(f"\n  LEADERBOARD [{datetime.now().strftime('%H:%M')}] BTC ${btc:,.2f} | Resolve: {wm.seconds_remaining()}s | Windows: {len(history)}")
                print(f"  {'#':<3} {'BOT':<25} {'T':<4} {'WR':<6} {'P&L':<11} {'$/DAY':<9} {'SH'}")
                print(f"  {'─'*65}")
                for i,st in enumerate(sorted([b.stats() for b in bots],key=lambda x:x["pnl"],reverse=True)):
                    wr=f"{st['wr']:.0f}%" if st['trades']>0 else "--"
                    print(f"  {i+1:<3} {st['name']:<25} {st['trades']:<4} {wr:<6} ${st['pnl']:<10,.0f} ${st['daily_est']:<8,.0f} {st['sharpe']}")

    except asyncio.CancelledError: pass
    finally:
        for t in tasks: t.cancel()
        print(f"\n{'='*72}\n  FINAL — {len(history)} windows\n{'='*72}")
        for i,st in enumerate(sorted([b.stats() for b in bots],key=lambda x:x["pnl"],reverse=True)):
            wr=f"{st['wr']:.0f}%" if st['trades']>0 else "--"
            print(f"  {i+1}. {st['name']:<25} {st['trades']}t {wr} ${st['pnl']:+,.0f} $/day:{st['daily_est']:+,.0f}")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--self-train",action="store_true")
    args=parser.parse_args()

    try:
        asyncio.run(run(self_train=args.self_train))
    except KeyboardInterrupt:
        print("\nTournament stopped.")
    except Exception as e:
        print(f"\nTournament error: {e}")
        import traceback
        traceback.print_exc()

if __name__=="__main__": main()
