# Claude Implementation Prompt for MTY-HFT

Use this as your system + user prompt when telling Claude 4.6 to build the platform.

---

## SYSTEM PROMPT

```
You are a Senior Quantitative Developer building a Polymarket crypto HFT platform.

You think like a senior quant: you care about net edge after fees, not gross signals. You measure everything — latency, fill rates, calibration, Brier scores. You never trust a backtest blindly. You build modular, testable systems where adding a new strategy is just adding a new file and config entry.

Your working style:
- Start with data infrastructure (you can't trade what you can't measure)
- Build incrementally: get one feed working end-to-end before adding complexity
- Every module has clear inputs, outputs, and tests
- Config-driven: parameters live in YAML, not hardcoded
- Log aggressively: every decision, every fill, every latency measurement
- Risk controls are non-negotiable: build them alongside execution, not after

You are working in the polymarket-pnl/ project directory. Read PRD.md for the full specification. Read CLAUDE.md for project conventions.

Tech stack: Python 3.12 (async), TimescaleDB, Redis 7, FastAPI, Docker Compose. Polymarket via py-clob-client. CEX via ccxt.pro or raw WebSocket.
```

---

## USER PROMPT (paste this to kick off implementation)

```
I want you to implement the Polymarket Crypto HFT platform specified in PRD.md. This is a maker-only HFT system that exploits the lead–lag between Binance/Bybit futures and Polymarket's short-term (5m/15m) crypto binary markets.

Read the full PRD.md first, then implement in this order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1: DATA INFRASTRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build these modules first — nothing else works without clean data:

1. `data/binance_feed.py` — Async WebSocket connector for Binance Futures
   - Subscribe to: btcusdt@trade, btcusdt@depth20@100ms, ethusdt@trade, ethusdt@depth20@100ms, solusdt@trade, solusdt@depth20@100ms
   - Parse into unified internal format (timestamp, mid, bid/ask, trade price/size/side)
   - Auto-reconnect with exponential backoff (1s→2s→4s→8s→max 30s)
   - Write to Redis (real-time state) and queue for TimescaleDB persistence

2. `data/bybit_feed.py` — Same pattern for Bybit Futures
   - Subscribe to orderbook.50 and publicTrade topics
   - Same unified internal format

3. `data/polymarket_feed.py` — Polymarket CLOB WebSocket + Gamma API
   - WebSocket: subscribe to order book updates and trades for active crypto markets
   - Gamma API: poll every 30s for new/expiring markets, parse metadata
   - Store market lifecycle events (created, active, settling, settled)

4. `data/recorder.py` — Tick-level data recorder
   - Writes all raw ticks to TimescaleDB hypertables (for backtest replay)
   - Schema per PRD Section 11

5. `data/db.py` — TimescaleDB + Redis helpers
   - Connection pools, hypertable creation, Redis pub/sub wrappers
   - All SQL schemas from PRD (market_metadata, positions, alpha_evaluations)

6. `core/state.py` — MarketState builder
   - Consumes feeds, builds MarketState dataclass per PRD Section 4.2
   - Publishes state updates via Redis pub/sub

7. `core/fee_engine.py` — Fee & rebate calculations
   - Load fee schedule from config/fees.yaml
   - compute_taker_fee(), compute_maker_rebate(), compute_net_edge()

8. `core/lead_lag.py` — Lead–lag estimator
   - Rolling cross-correlation between Binance mid changes and Polymarket mid changes
   - Window: 500–2000ms configurable
   - Publish estimates to Redis

Create config/trading.yaml and config/fees.yaml with defaults from the PRD.

Test: Run all three feeds simultaneously, verify state updates are flowing, verify tick recording works. Print a live dashboard showing: Binance BTC mid, Polymarket BTC 5m YES/NO mid, lead–lag estimate, latency.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2: CORE TRADING ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build the pricing and strategy framework:

1. `core/theo_engine.py` — p_theo computation
   - Method 1 (drift-based): short-term return + OFI → directional probability via normal CDF
   - Method 2 (vol-scaled): realized vol → P(move > X bps)
   - Method 3 (ensemble): adaptive weighted combination
   - Output: p_theo for each active Polymarket market

2. `core/scheduler.py` — Multi-market scheduler
   - Rank active markets by: net_ev × capacity × time_decay
   - Allocate capital respecting all constraints from PRD Section 5.3
   - Output: {market_id: allocation_usd}

3. `strategies/base.py` — BaseStrategy ABC
   - compute_signal(state) → Signal(direction, confidence, sizing_hint)
   - backtest(historical_states) → BacktestResult

4. `strategies/lead_lag.py` — Strategy Family 1: Lead–Lag Binary Directional
   - This is the primary strategy. Get this working end-to-end first.
   - Map 30s CEX return + OFI to directional probability
   - Compare p_theo vs pm_mid, trade if net EV > fee drag + min edge
   - Kelly fraction sizing

5. `strategies/registry.py` — Strategy auto-discovery from strategies.yaml

6. `execution/polymarket_client.py` — Polymarket CLOB wrapper
   - Thin wrapper around py-clob-client
   - create_limit_order(), cancel_order(), get_order_status()
   - Enforce maker-only: reject if order would take

7. `execution/order_manager.py` — Order lifecycle
   - Signal → risk check → order submit → track → fill → position update → PnL

8. `execution/fill_tracker.py` — Fill tracking, position & PnL recording
   - Write to positions table per PRD Section 10.3

Test: Run the lead–lag strategy against live data (paper trading mode — log what you WOULD trade, don't execute). Verify signal generation, p_theo computation, and fee-adjusted edge calculations make sense.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3: RISK & HARDENING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. `risk/inventory.py` — Per-underlying inventory limits, time-to-expiry de-risking, spread widening
2. `risk/drawdown.py` — Strategy/underlying/global loss limits, kill-switches
3. `risk/hedge.py` — Interface stub for future CEX hedging
4. `core/latency.py` — Latency monitoring, Prometheus metrics, circuit breaker
5. `cli.py` — CLI for pause/resume/status/latency

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4: RESEARCH & EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. `research/backtester.py` — Replay recorded ticks through strategies
2. `research/alpha_evaluator.py` — All metrics from PRD Section 8.1
3. `strategies/calibration.py` — Strategy Family 2: Implied vs Realized
4. `strategies/cross_maturity.py` — Strategy Family 3
5. `strategies/basis.py` — Strategy Family 4
6. `research/whatif.py` — What-if simulator (re-run with different params)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 5: META-RESEARCH & DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. `research/meta_agent.py` — Claude 4.6 daily review agent (PRD Section 9)
2. `dashboard/api.py` — FastAPI endpoints for all dashboard panels
3. `dashboard/frontend/` — Next.js dashboard (or Grafana dashboards config)
4. Telegram alerting integration
5. `main.py` — Entry point that orchestrates all components

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 6: DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Dockerfile + docker-compose.yml
2. Prometheus + Grafana monitoring config
3. requirements.txt / pyproject.toml
4. Updated CLAUDE.md with new architecture

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL CONSTRAINTS (apply to ALL phases):
- All decision logic uses net edge AFTER fees + rebates. Never raw edge.
- Maker-only on Polymarket. Never take liquidity.
- Every module is independently testable.
- Config in YAML, not hardcoded constants.
- Log every trade decision (signal, p_theo, market_price, net_ev, fees, outcome).
- Latency measured end-to-end, stored in Prometheus.
- Kill-switches are first-class, not afterthoughts.
- Adding a new strategy = new .py file + entry in strategies.yaml. No other changes.

Start with Phase 1 now. Build each module, test it, then move to the next. Show me the code.
```

---

## ALTERNATIVE: RESUME PROMPT (for continuing work across sessions)

```
Continue implementing the MTY-HFT platform from PRD.md.

Check the current state of the codebase:
1. What modules exist under core/, strategies/, execution/, risk/, data/, research/?
2. Which phase from PRD.md Section 15 are we on?
3. What's the next unbuilt module?

Then pick up where we left off. Build the next module, test it, and proceed.
```

---

## TIPS FOR USING THESE PROMPTS

1. **First session:** Paste the system prompt as a CLAUDE.md update or system instruction. Then paste the full user prompt. Claude will start with Phase 1.

2. **Subsequent sessions:** Use the resume prompt. Claude will inspect what exists and continue.

3. **If Claude tries to build too much at once:** Tell it: "Focus on just `data/binance_feed.py` right now. Get it working end-to-end before moving on."

4. **If Claude skips risk controls:** Remind it: "Build `risk/drawdown.py` alongside `execution/order_manager.py`, not after. The PRD says risk controls are non-negotiable."

5. **To trigger meta-research manually:** After you have some backtest results, paste this:
   ```
   Run the meta-research review now. Load the latest alpha_evaluations from the DB,
   format the summary, and give me your senior quant analysis:
   - Which strategies show genuine edge vs overfit?
   - What should we scale up/down?
   - What 2-3 new ideas should we test next?
   ```

6. **To add a new strategy idea:**
   ```
   Add a new strategy to the platform:
   Name: [your name]
   Idea: [describe the signal]

   Implement it as a BaseStrategy subclass in strategies/[name].py,
   add it to config/strategies.yaml (disabled by default),
   run a backtest, and show me the alpha_evaluator results.
   ```
