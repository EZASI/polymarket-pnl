# MTY Polymarket Trading System

## Project Overview
Automated prediction and trading system for Polymarket prediction markets.
Uses multi-source edge detection (sportsbook odds, crypto TA, copy signals) to find mispriced markets.

## Architecture
```
polymarket-pnl/
  dashboard.py          — Web dashboard (port 8080, password-protected)
  multi_predictor.py    — Core prediction engine v2 (runs every 30m via cron)
  copy_signal_bot.py    — Tracks top 15 Polymarket traders, finds convergence signals
  trader_tracker.py     — Detailed trader analysis and win rate tracking
  smart_picks.py        — Combines ML + copy signals for ranked picks
  performance_tracker.py— Tracks prediction accuracy and simulated P&L
  tournament.py         — 30-bot crypto trading tournament
  sports_arb_scanner.py — Cross-platform sports arbitrage scanner
  prediction_engine.py  — NBA ML model (ELO + LightGBM, 80.6% accuracy)
```

## Key APIs
- **Polymarket Gamma API**: `https://gamma-api.polymarket.com/markets` — market discovery
- **Polymarket CLOB**: `https://clob.polymarket.com` — order book data
- **Binance**: Klines for crypto TA (RSI/MACD/Bollinger)
- **The Odds API**: Sportsbook odds comparison (key in env: ODDS_API_KEY)
- **Open-Meteo**: Weather forecast ensemble

## Data Flow
1. `copy_signal_bot.py` scans top traders every 10 min → `logs/copy_signals.json`
2. `multi_predictor.py` runs every 30 min, combines all data sources → `logs/all_predictions.json`
3. `smart_picks.py` ranks picks with scoring engine → `logs/smart_signals.json`
4. `performance_tracker.py` checks market resolutions hourly → `logs/performance.json`
5. `dashboard.py` serves all data via API endpoints

## AWS Server
- EC2: `43.207.206.223` (Ubuntu, user: ubuntu)
- SSH key: `/Users/eiji/Downloads/polymarket-key.pem`
- Dashboard: port 8080, password: `TradingBot2026`
- Venv: `/home/ubuntu/polymarket-pnl/venv/`

## Agent Teams Guidelines
When working as a team member:
- **DO NOT** edit `dashboard.py` or `multi_predictor.py` unless you are the designated owner
- Each strategy agent owns its own `strategy_*.py` file
- Write results to `logs/` directory with your agent name prefix
- Use `logs/leaderboard.json` for shared scoreboard (append-only)
- Rate limit all API calls (0.3s between Polymarket, 0.2s between Binance)
- Maximum position size: $500 per market (10% of $5k allocation)
- Kill signal: check `logs/kill_signals.json` before each trade cycle

## Trading Swarm Architecture
```
Leader (orchestrator)
  ├── Agent 1-3: Quant Arbitrage (strategy_quant_*.py)
  ├── Agent 4-6: News/Sentiment Hawks (strategy_news_*.py)
  ├── Agent 7-9: Scalpers (strategy_scalp_*.py)
  └── Agent 10: Risk Manager (strategy_risk.py)
```

Each agent:
1. Reads market data from Polymarket/Binance APIs
2. Generates trade signals in `logs/signals_{agent_name}.json`
3. Simulates execution, tracks P&L in `logs/pnl_{agent_name}.json`
4. Risk Manager monitors all agent P&L files, can write kill signals
