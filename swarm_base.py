#!/usr/bin/env python3
"""
SWARM BASE — Sovereign Swarm Racing Infrastructure
=====================================================================

22 Trading Agents + 1 Pit Boss = 23 Total
$20,000 per trading agent. $440,000 total portfolio.
Target: $3,000/day per agent = $66,000/day collective.

Features:
  - Portfolio tracking with Sharpe ratio, max drawdown, daily P&L
  - live_race_stats.json updated every 15 minutes
  - 5% drawdown purge: underperformers get terminated and respawned
  - Shared market data pipeline with rate limiting
  - Kill signal mechanism for risk management
"""

import json
import math
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

# ── SOVEREIGN SWARM RACE CONFIG ──
STARTING_CAPITAL = 20000.0
MAX_POSITION_PCT = 0.15       # 15% of capital per trade
POLYMARKET_FEE = 0.02         # 2% fee on winnings
DAILY_TARGET_PER_AGENT = 3000.0   # $3k/day per agent
N_TRADING_AGENTS = 23
DAILY_TARGET = DAILY_TARGET_PER_AGENT * N_TRADING_AGENTS  # $69k/day collective
DRAWDOWN_PURGE_PCT = 0.15    # 15% drawdown = agent gets fired & replaced
N_AGENTS = N_TRADING_AGENTS + 1  # 23 traders + 1 pit boss = 24


class SwarmAgent:
    """Base class for all swarm trading agents."""

    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capital = STARTING_CAPITAL
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.peak_capital = STARTING_CAPITAL  # For max drawdown tracking
        self.pnl_history: List[Dict] = []     # Time-series for Sharpe
        self.generation = 1                    # Increments on respawn
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.purged = False

        self._load_state()

    def _state_path(self):
        return LOG_DIR / f"pnl_{self.agent_id}.json"

    def _load_state(self):
        path = self._state_path()
        if path.exists():
            try:
                data = json.load(open(path))
                self.capital = data.get("capital", STARTING_CAPITAL)
                self.positions = data.get("positions", [])
                self.trades = data.get("trades", [])
                self.pnl = data.get("pnl", 0.0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.peak_capital = data.get("peak_capital", STARTING_CAPITAL)
                self.pnl_history = data.get("pnl_history", [])
                self.generation = data.get("generation", 1)
                self.start_time = data.get("start_time", self.start_time)
                self.purged = data.get("purged", False)
            except:
                pass

    def save_state(self):
        # Track peak capital
        current_total = self.capital + sum(p.get("size", 0) for p in self.positions if p.get("status") == "open")
        self.peak_capital = max(self.peak_capital, current_total)

        # Record P&L snapshot for Sharpe calculation
        now = datetime.now(timezone.utc).isoformat()
        self.pnl_history.append({"time": now, "pnl": round(self.pnl, 2), "capital": round(self.capital, 2)})
        # Keep last 200 snapshots (~3 days at 15-min intervals)
        self.pnl_history = self.pnl_history[-200:]

        data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capital": round(self.capital, 2),
            "starting_capital": STARTING_CAPITAL,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl / STARTING_CAPITAL * 100, 2),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / max(self.wins + self.losses, 1) * 100, 1),
            "n_trades": self.wins + self.losses,
            "sharpe": self.calc_sharpe(),
            "max_drawdown": self.calc_max_drawdown(),
            "drawdown_pct": self.calc_drawdown_pct(),
            "peak_capital": round(self.peak_capital, 2),
            "generation": self.generation,
            "purged": self.purged,
            "positions": self.positions,
            "open_value": round(sum(p.get("size", 0) for p in self.positions if p.get("status") == "open"), 2),
            "trades": self.trades[-50:],
            "pnl_history": self.pnl_history,
            "start_time": self.start_time,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._state_path(), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def calc_sharpe(self) -> float:
        """Annualized Sharpe ratio from P&L history."""
        if len(self.pnl_history) < 3:
            return 0.0
        returns = []
        for i in range(1, len(self.pnl_history)):
            prev = self.pnl_history[i - 1]["pnl"]
            curr = self.pnl_history[i]["pnl"]
            returns.append(curr - prev)
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.001
        # Annualize: ~96 fifteen-min periods per day, ~365 days
        sharpe = (mean_r / std_r) * math.sqrt(96 * 365)
        return round(sharpe, 2)

    def calc_max_drawdown(self) -> float:
        """Maximum drawdown in dollars from peak."""
        current_total = self.capital + sum(p.get("size", 0) for p in self.positions if p.get("status") == "open")
        dd = self.peak_capital - current_total
        return round(max(dd, 0), 2)

    def calc_drawdown_pct(self) -> float:
        """Current drawdown as % of starting capital."""
        dd = self.calc_max_drawdown()
        return round(dd / STARTING_CAPITAL * 100, 2)

    def should_be_purged(self) -> bool:
        """Returns True if agent has exceeded 5% drawdown threshold."""
        return self.calc_drawdown_pct() >= DRAWDOWN_PURGE_PCT * 100

    def is_killed(self) -> bool:
        """Check if risk manager has sent a kill signal."""
        if self.purged:
            return True
        kill_path = LOG_DIR / "kill_signals.json"
        if kill_path.exists():
            try:
                signals = json.load(open(kill_path))
                for sig in signals.get("kills", []):
                    if sig.get("agent_id") == self.agent_id and sig.get("active", True):
                        return True
            except:
                pass
        return False

    def open_position(self, market_title, condition_id, outcome, size, price, reason=""):
        """Open a simulated position."""
        if self.is_killed():
            return False

        # Prevent duplicate positions on same market
        for pos in self.positions:
            if pos.get("condition_id") == condition_id and pos.get("status") == "open":
                return False

        max_size = self.capital * MAX_POSITION_PCT
        size = min(size, max_size, self.capital)
        if size < 5:
            return False

        position = {
            "market": market_title[:100],
            "condition_id": condition_id,
            "outcome": outcome,
            "size": round(size, 2),
            "entry_price": round(price, 4),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason[:120],
            "status": "open",
        }
        self.positions.append(position)
        self.capital -= size
        self.save_state()
        return True

    def close_position(self, condition_id, exit_price, resolved=False):
        """Close a position and calculate P&L."""
        for pos in self.positions:
            if pos.get("condition_id") == condition_id and pos.get("status") == "open":
                entry = pos["entry_price"]
                size = pos["size"]

                if resolved:
                    if exit_price >= 0.5:
                        payout = size / entry
                        fee = (payout - size) * POLYMARKET_FEE
                        trade_pnl = payout - size - fee
                        self.wins += 1
                    else:
                        trade_pnl = -size
                        self.losses += 1
                else:
                    trade_pnl = size * (exit_price - entry) / entry
                    if trade_pnl > 0:
                        trade_pnl *= (1 - POLYMARKET_FEE)
                        self.wins += 1
                    else:
                        self.losses += 1

                self.pnl += trade_pnl
                self.capital += size + trade_pnl

                pos["status"] = "closed"
                pos["exit_price"] = round(exit_price, 4)
                pos["trade_pnl"] = round(trade_pnl, 2)
                pos["closed_at"] = datetime.now(timezone.utc).isoformat()

                self.trades.append(pos)
                self.positions = [p for p in self.positions if p.get("status") == "open"]
                self.save_state()
                return trade_pnl

        return 0

    def get_summary(self) -> Dict:
        """Full agent performance summary."""
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type,
            "generation": self.generation,
            "capital": round(self.capital, 2),
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl / STARTING_CAPITAL * 100, 2),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / max(self.wins + self.losses, 1) * 100, 1),
            "sharpe": self.calc_sharpe(),
            "max_drawdown": self.calc_max_drawdown(),
            "drawdown_pct": self.calc_drawdown_pct(),
            "open_positions": len(self.positions),
            "total_trades": self.wins + self.losses,
            "purged": self.purged,
        }

    def respawn(self, new_generation: int = None):
        """Reset agent state for a new generation after being purged."""
        gen = new_generation or (self.generation + 1)
        self.capital = STARTING_CAPITAL
        self.positions = []
        self.trades = []
        self.pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.peak_capital = STARTING_CAPITAL
        self.pnl_history = []
        self.generation = gen
        self.purged = False
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.save_state()


# ============================================================
# LEADERBOARD + LIVE RACE STATS
# ============================================================
def update_leaderboard() -> Dict:
    """Aggregate all agent P&L files into a leaderboard + live_race_stats.json."""
    agents = []
    for f in LOG_DIR.glob("pnl_*.json"):
        if "risk_manager" in str(f):
            continue
        try:
            data = json.load(open(f))
            agents.append({
                "agent_id": data.get("agent_id", f.stem.replace("pnl_", "")),
                "type": data.get("agent_type", "unknown"),
                "generation": data.get("generation", 1),
                "capital": data.get("capital", STARTING_CAPITAL),
                "pnl": data.get("pnl", 0),
                "pnl_pct": data.get("pnl_pct", 0),
                "wins": data.get("wins", 0),
                "losses": data.get("losses", 0),
                "win_rate": data.get("win_rate", 0),
                "sharpe": data.get("sharpe", 0),
                "max_drawdown": data.get("max_drawdown", 0),
                "drawdown_pct": data.get("drawdown_pct", 0),
                "n_trades": data.get("n_trades", 0),
                "open_positions": len(data.get("positions", [])),
                "open_value": data.get("open_value", 0),
                "purged": data.get("purged", False),
                "last_update": data.get("last_update", ""),
            })
        except:
            pass

    agents.sort(key=lambda x: x["pnl"], reverse=True)
    for i, a in enumerate(agents):
        a["rank"] = i + 1

    total_pnl = sum(a["pnl"] for a in agents)
    total_capital = sum(a["capital"] for a in agents)
    total_open = sum(a["open_positions"] for a in agents)
    total_trades = sum(a["n_trades"] for a in agents)

    # Daily progress toward $3k target
    daily_progress = min(total_pnl / DAILY_TARGET * 100, 999) if DAILY_TARGET > 0 else 0

    leaderboard = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total_agents": len(agents),
        "total_pnl": round(total_pnl, 2),
        "total_capital": round(total_capital, 2),
        "total_open_positions": total_open,
        "total_trades": total_trades,
        "daily_target": DAILY_TARGET,
        "daily_progress_pct": round(daily_progress, 1),
        "best_agent": agents[0]["agent_id"] if agents else None,
        "worst_agent": agents[-1]["agent_id"] if agents else None,
        "avg_sharpe": round(sum(a["sharpe"] for a in agents) / max(len(agents), 1), 2),
        "agents": agents,
    }

    with open(LOG_DIR / "leaderboard.json", "w") as f:
        json.dump(leaderboard, f, indent=2, default=str)

    # Also write live_race_stats.json (the 15-min competition tracker)
    race_stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "race_status": "ACTIVE",
        "portfolio": {
            "total_capital": round(total_capital, 2),
            "starting_capital": STARTING_CAPITAL * N_AGENTS,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / (STARTING_CAPITAL * N_AGENTS) * 100, 2),
            "total_trades": total_trades,
            "open_positions": total_open,
        },
        "daily_target": {
            "target_usd": DAILY_TARGET,
            "current_pnl": round(total_pnl, 2),
            "progress_pct": round(daily_progress, 1),
            "on_track": total_pnl >= 0,
        },
        "rankings": [{
            "rank": a["rank"],
            "agent_id": a["agent_id"],
            "type": a["type"],
            "gen": a["generation"],
            "pnl": a["pnl"],
            "win_rate": a["win_rate"],
            "sharpe": a["sharpe"],
            "drawdown_pct": a["drawdown_pct"],
            "trades": a["n_trades"],
            "status": "PURGED" if a["purged"] else ("DANGER" if a["drawdown_pct"] >= 4 else "ACTIVE"),
        } for a in agents],
        "purge_threshold": f"{DRAWDOWN_PURGE_PCT * 100}%",
    }

    with open(LOG_DIR / "live_race_stats.json", "w") as f:
        json.dump(race_stats, f, indent=2, default=str)

    return leaderboard


def check_and_purge() -> List[Dict]:
    """Check all agents for 5% drawdown. Purge and respawn violators."""
    purge_events = []

    for f in LOG_DIR.glob("pnl_*.json"):
        if "risk_manager" in str(f):
            continue
        try:
            data = json.load(open(f))
            agent_id = data.get("agent_id", "")
            dd_pct = data.get("drawdown_pct", 0)
            pnl = data.get("pnl", 0)
            gen = data.get("generation", 1)

            if dd_pct >= DRAWDOWN_PURGE_PCT * 100 and not data.get("purged", False):
                # PURGE this agent
                data["purged"] = True
                with open(f, "w") as fh:
                    json.dump(data, fh, indent=2, default=str)

                purge_events.append({
                    "agent_id": agent_id,
                    "reason": f"Drawdown {dd_pct:.1f}% exceeded {DRAWDOWN_PURGE_PCT*100}% threshold",
                    "pnl_at_purge": pnl,
                    "generation": gen,
                    "purged_at": datetime.now(timezone.utc).isoformat(),
                })

                # Respawn: reset the state file
                data["capital"] = STARTING_CAPITAL
                data["positions"] = []
                data["trades"] = []
                data["pnl"] = 0.0
                data["pnl_pct"] = 0.0
                data["wins"] = 0
                data["losses"] = 0
                data["peak_capital"] = STARTING_CAPITAL
                data["pnl_history"] = []
                data["generation"] = gen + 1
                data["purged"] = False
                data["start_time"] = datetime.now(timezone.utc).isoformat()
                data["last_update"] = datetime.now(timezone.utc).isoformat()
                data["max_drawdown"] = 0
                data["drawdown_pct"] = 0
                data["sharpe"] = 0
                data["open_value"] = 0

                with open(f, "w") as fh:
                    json.dump(data, fh, indent=2, default=str)

                purge_events[-1]["respawned_as"] = f"gen {gen + 1}"

        except:
            continue

    # Log purge events
    if purge_events:
        purge_log_path = LOG_DIR / "purge_log.json"
        existing = []
        if purge_log_path.exists():
            try:
                existing = json.load(open(purge_log_path))
            except:
                existing = []
        existing.extend(purge_events)
        with open(purge_log_path, "w") as f:
            json.dump(existing[-100:], f, indent=2, default=str)  # Keep last 100

    return purge_events


# ============================================================
# SHARED MARKET DATA PIPELINE
# ============================================================
_last_request_time = 0
_market_cache = {}
_cache_time = 0
CACHE_TTL = 120  # 2 minutes

def rate_limited_get(url, params=None, min_interval=0.25, timeout=15):
    """Rate-limited GET request."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        return resp
    except:
        return None


def fetch_market_price(condition_id):
    """Get current price for a market by condition ID."""
    resp = rate_limited_get(
        f"{GAMMA_HOST}/markets",
        params={"condition_id": condition_id}
    )
    if resp and resp.status_code == 200:
        data = resp.json()
        if data:
            m = data[0] if isinstance(data, list) else data
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except:
                    prices = []
            return float(prices[0]) if prices else None
    return None


def fetch_active_markets(limit=500):
    """Fetch active Polymarket markets with caching."""
    global _market_cache, _cache_time

    if _market_cache and (time.time() - _cache_time) < CACHE_TTL:
        return _market_cache[:limit]

    all_markets = []
    offset = 0
    while offset < limit:
        resp = rate_limited_get(
            f"{GAMMA_HOST}/markets",
            params={"limit": 100, "active": "true", "closed": "false", "offset": offset}
        )
        if not resp or resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        all_markets.extend(data)
        offset += 100
        if len(data) < 100:
            break

    if all_markets:
        _market_cache = all_markets
        _cache_time = time.time()

    return all_markets[:limit]


def get_copy_signals():
    """Load latest copy signals for agents that need them."""
    path = LOG_DIR / "copy_signals.json"
    if path.exists():
        try:
            return json.load(open(path))
        except:
            pass
    return {"signals": [], "time": ""}


def get_trader_data():
    """Load latest trader tracker data."""
    path = LOG_DIR / "trader_tracker.json"
    if path.exists():
        try:
            return json.load(open(path))
        except:
            pass
    return {"traders": [], "time": ""}
