# Polymarket Crypto HFT Platform — Product Requirements Document

**Version:** 1.0 | **Date:** February 2026 | **Status:** Development
**Codename:** MTY-HFT | **Author:** Eiji / Quantitative Trading Research

---

## 1. Executive Summary

MTY-HFT is a high-frequency trading platform purpose-built for Polymarket's short-term crypto binary markets (5-minute and 15-minute BTC, ETH, SOL). The system exploits cross-venue lead–lag between centralized exchanges (Binance, Bybit) and Polymarket's CLOB to compute theoretical probabilities, identify mispricings, and execute maker-only orders on Polymarket.

The platform operates as a senior quant would: it maintains a research lab for generating and backtesting strategy ideas, a live execution engine with institutional-grade risk controls, and a meta-research agent (Claude 4.6) that reviews performance and proposes next experiments.

**Core Value Proposition:** Systematic, fee-aware, latency-measured exploitation of the information lag between crypto CEX price discovery and Polymarket binary outcome markets, with continuous AI-driven strategy improvement.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    EXCHANGE DATA FEEDS                                │
│   Binance Futures WS · Bybit Futures WS · Polymarket CLOB WS        │
└──────────┬───────────────────────────────────────────────────────────┘
           │  WebSocket (sub-second)
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   CROSS-VENUE MICROSTRUCTURE LAYER                    │
│  Lead–Lag Estimator · OFI Calculator · Realized Vol · Fee Engine     │
│  State Space Builder · Depth Aggregator · Latency Monitor            │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    THEO PRICING ENGINE                                │
│  p_theo Calculator · Net EV Engine (after fees/rebates)              │
│  Multi-Market Scheduler · Capital Allocator                          │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                                     │
│  Lead–Lag Directional · Implied vs Realized · Cross-Maturity Arb     │
│  Cross-Exchange Basis · (Pluggable: add new strategy = new module)    │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────┐     ┌─────────────────────────────────────────────┐
│  RISK ENGINE     │◄───►│  EXECUTION ENGINE (Maker-Only)              │
│  Inventory Mgr   │     │  Polymarket CLOB API · Order Manager        │
│  Kill Switch     │     │  Fill Tracker · Reconnect Logic             │
│  Drawdown Guard  │     └─────────────────────────────────────────────┘
└──────────┬───────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    RESEARCH & META LAYER                              │
│  Backtest Engine · Alpha Evaluator · Strategy Registry               │
│  Meta-Research Agent (Claude 4.6) · What-If Simulator                │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY & TOOLS                              │
│  Dashboard (Next.js or Grafana) · TimescaleDB/Prometheus             │
│  CLI Controls (pause/resume/kill) · Telegram Alerts                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Runtime | Python 3.12 | async throughout (asyncio) |
| CEX Connectivity | `ccxt.pro` or raw WebSocket | Binance Futures, Bybit Futures |
| Polymarket Connectivity | `py-clob-client` + raw WS | CLOB API + WebSocket subscriptions |
| Time-Series DB | TimescaleDB (PostgreSQL 16) | Tick data, candles, fills, PnL |
| Cache / State | Redis 7 | Real-time state, lead–lag estimates, kill signals |
| Dashboard | Next.js 15 or Grafana | PnL, positions, latency panels |
| Monitoring | Prometheus + Grafana | Latency histograms, fill rates |
| Deployment | Docker Compose on AWS EC2 | Same infra pattern as FundingEdge |
| AI Agent | Claude 4.6 API | Meta-research, strategy review |

### 2.3 Deployment Topology

```
EC2 Instance (existing: 43.207.206.223 or new dedicated)
├── docker-compose.yml
│   ├── timescaledb     (127.0.0.1:5432)
│   ├── redis           (127.0.0.1:6379)
│   ├── hft-engine      (internal)       — Core trading loop
│   ├── research-lab    (internal)       — Backtesting & alpha eval
│   ├── api             (127.0.0.1:8000) — FastAPI for dashboard
│   └── dashboard       (0.0.0.0:3001)  — Web UI
└── grafana             (127.0.0.1:3000)
```

---

## 3. Polymarket Market Model

### 3.1 Supported Markets

| Market Type | Underlying | Expiry Window | Priority |
|------------|-----------|---------------|----------|
| 5-minute price prediction | BTC, ETH, SOL | Rolling every 5 min | P0 |
| 15-minute price prediction | BTC, ETH, SOL | Rolling every 15 min | P0 |
| Custom strike binary | BTC, ETH | Variable | P1 |

### 3.2 Market Metadata Store

Schema for `market_metadata` table:

```sql
CREATE TABLE market_metadata (
    market_id        TEXT PRIMARY KEY,
    condition_id     TEXT NOT NULL,
    underlying       TEXT NOT NULL,        -- 'BTC', 'ETH', 'SOL'
    expiry_ts        TIMESTAMPTZ NOT NULL,
    strike_price     NUMERIC,             -- NULL for ATM
    strike_type      TEXT DEFAULT 'ATM',  -- 'ATM', 'custom'
    fee_tier         NUMERIC NOT NULL,    -- e.g. 0.03 for 3%
    maker_rebate_pct NUMERIC DEFAULT 0,   -- maker rebate if eligible
    status           TEXT DEFAULT 'active',
    discovered_at    TIMESTAMPTZ DEFAULT NOW(),
    settled_outcome  BOOLEAN,             -- NULL until settled
    settled_at       TIMESTAMPTZ
);
```

### 3.3 Fee Engine

Polymarket charges taker fees up to ~3% on short-term markets (highest near 50% implied probability). Maker rebates may apply per their program.

**Requirements:**

1. Implement `fee_engine.py` module:
   - `compute_taker_fee(price: float, market_type: str) -> float`
   - `compute_maker_rebate(price: float, volume: float) -> float`
   - `compute_net_edge(p_theo: float, market_price: float, side: str, is_maker: bool) -> float`
2. **All** trading decisions use `net_edge_after_fees`, never raw edge.
3. Fee schedule must be configurable (YAML/JSON) so it can be updated when Polymarket changes fees.
4. Log fee impact per trade for post-hoc analysis.

```python
# Example fee config (config/fees.yaml)
polymarket:
  taker_fee_schedule:
    - price_range: [0.45, 0.55]  # near 50%
      fee_pct: 0.03
    - price_range: [0.30, 0.45]
      fee_pct: 0.02
    - price_range: [0.55, 0.70]
      fee_pct: 0.02
    - price_range: [0.0, 0.30]
      fee_pct: 0.01
    - price_range: [0.70, 1.0]
      fee_pct: 0.01
  maker_rebate_pct: 0.005  # 0.5% rebate for maker orders
```

---

## 4. Cross-Venue Microstructure Layer

### 4.1 Lead–Lag Measurement

A dedicated `lead_lag.py` module continuously measures the time delay between:
- **Binance/Bybit** midprice changes (futures)
- **Polymarket** midprice changes (mid of YES/NO best bid/offer)

**Requirements:**

1. Maintain rolling lead–lag estimates using a sliding window of 500–2000 ms.
2. Use cross-correlation or Hayashi–Yoshida estimator for asynchronous tick streams.
3. Store lead–lag estimates in Redis (key: `lead_lag:{underlying}`, updated every tick batch).
4. Log lead–lag time series to TimescaleDB for research.

### 4.2 State Space

For each active Polymarket market, maintain a real-time state vector:

```python
@dataclass
class MarketState:
    # CEX side (Binance/Bybit futures)
    cex_mid: float              # Binance futures midprice
    cex_ofi: float              # Order Flow Imbalance (rolling 30s)
    cex_realized_vol: float     # Short-horizon realized vol (1m, 5m)
    cex_funding_rate: float     # Current funding rate
    cex_basis: float            # Binance vs Bybit basis

    # Polymarket side
    pm_yes_bid: float           # Best bid for YES
    pm_yes_ask: float           # Best ask for YES
    pm_no_bid: float            # Best bid for NO
    pm_no_ask: float            # Best ask for NO
    pm_mid: float               # Mid of YES best bid/ask
    pm_depth_bid: float         # Total depth on bid side (top 5 levels)
    pm_depth_ask: float         # Total depth on ask side (top 5 levels)
    pm_last_trade_price: float
    pm_last_trade_ts: float

    # Derived
    time_to_expiry_s: float     # Seconds until settlement
    p_theo: float               # Theoretical probability from CEX data
    net_ev_yes: float           # EV of buying YES, after fees
    net_ev_no: float            # EV of buying NO, after fees
    lead_lag_ms: float          # Current lead–lag estimate
```

### 4.3 p_theo Computation

```
p_theo = P(underlying > strike at expiry | CEX state)
```

Methods (strategy-dependent):
1. **Drift-based:** Use short-term return momentum + OFI to estimate directional probability via normal CDF.
2. **Vol-scaled:** Use realized vol to compute P(move > X bps) over remaining time.
3. **Ensemble:** Combine multiple estimators with adaptive weights.

### 4.4 Latency Monitoring

Measure and persist end-to-end latency:

```
[Binance tick received] → [state update] → [p_theo computed] → [quote decision] → [order submitted] → [order confirmed]
```

- Store latency percentiles (p50, p95, p99) in Prometheus.
- **Circuit breaker:** If p99 latency exceeds configured threshold (e.g., 2000ms), disable quoting or switch to wide passive-only mode.

---

## 5. Multi-Market Scheduler

### 5.1 Purpose

At any moment, multiple 5m and 15m markets may be active across BTC, ETH, SOL. The scheduler ranks and allocates capital.

### 5.2 Ranking Function

For each active market `m`:

```
score(m) = net_ev(m) × capacity(m) × time_decay(m)
```

Where:
- `net_ev(m)`: Expected value per contract after fees/rebates
- `capacity(m)`: min(available depth at target price, max per-contract size)
- `time_decay(m)`: Decay factor that reduces appetite as settlement approaches (e.g., exponential decay below 60s to expiry)

### 5.3 Capital Allocation Constraints

| Constraint | Limit | Rationale |
|-----------|-------|-----------|
| Per-contract max | $50,000 | Polymarket position limit |
| Per-underlying max | 2× equity | Prevent BTC-only concentration |
| Per-direction max | 1.5× equity long or short | Limit net directional bias |
| Simultaneous active markets | 6 | Operational complexity cap |
| Min net EV threshold | 0.5% after fees | Don't trade noise |

### 5.4 Allocation Algorithm

```python
def allocate_capital(markets: List[MarketState], equity: float, config: AllocationConfig) -> Dict[str, float]:
    """
    1. Filter markets by min_net_ev threshold
    2. Score and rank remaining markets
    3. Allocate greedily, respecting per-contract, per-underlying, per-direction caps
    4. Return {market_id: allocation_usd}
    """
```

---

## 6. Strategy Layer

### 6.1 Design Principle

**Adding a new strategy = adding a new module + config entry, not rewriting the bot.**

All strategies implement:

```python
class BaseStrategy(ABC):
    name: str
    config: dict

    @abstractmethod
    def compute_signal(self, state: MarketState) -> Signal:
        """Return Signal(direction, confidence, sizing_hint)"""

    @abstractmethod
    def backtest(self, historical_states: List[MarketState]) -> BacktestResult:
        """Return BacktestResult with PnL, Sharpe, hit rate, Brier score"""
```

New strategies are registered in `config/strategies.yaml` and auto-discovered at startup.

### 6.2 Strategy Family 1: Lead–Lag Binary Directional

**Core idea:** Map Binance short-term return + OFI signals to Polymarket outcome probabilities. Trade when net EV is positive after fees.

**Inputs:** `cex_mid` changes, `cex_ofi`, `lead_lag_ms`, `time_to_expiry_s`

**Logic:**
1. Compute rolling 30s CEX return and OFI.
2. Estimate directional probability using logistic regression (or simple threshold model).
3. Compare `p_theo` vs Polymarket `pm_mid`.
4. If `|p_theo - pm_mid| > fee_drag + min_edge`, generate signal.

**Parameters:** OFI threshold, return lookback, min edge after fees, Kelly fraction.

### 6.3 Strategy Family 2: Implied Probability vs Realized Outcome

**Core idea:** Backtest how often Polymarket implied probabilities are miscalibrated at different times to expiry. Trade mispricing around settlement windows.

**Inputs:** Historical `pm_mid` at T-5min, T-2min, T-1min vs actual settlement outcome.

**Logic:**
1. Build calibration curve: at each `pm_mid` bucket (e.g., [0.30, 0.35]), what fraction actually settle YES?
2. If live `pm_mid` deviates from calibration by > threshold, trade the deviation.
3. Most profitable near settlement (T-60s to T-10s) where miscalibration is highest.

**Parameters:** Calibration window (days), bucket size, min deviation, time-to-expiry filter.

### 6.4 Strategy Family 3: Cross-Maturity Consistency

**Core idea:** If 5-minute and 15-minute markets on the same underlying have inconsistent implied probabilities, exploit the term structure.

**Inputs:** Active 5m and 15m markets on same underlying, their `pm_mid` values.

**Logic:**
1. 15m market embeds the 5m market outcome as a conditional.
2. If P(15m YES) < P(5m YES) and they share the same directional bias, there's an inconsistency.
3. Trade the mispriced leg (or both legs if capacity allows).

**Parameters:** Consistency threshold, min depth on both legs, max spread cost.

### 6.5 Strategy Family 4: Cross-Exchange Basis

**Core idea:** Use Binance vs Bybit futures basis and funding rate differential to adjust short-term directional priors.

**Inputs:** `cex_basis`, `cex_funding_rate` from both exchanges.

**Logic:**
1. Positive basis divergence (Binance premium over Bybit) → bullish short-term prior for BTC.
2. Combine with OFI and vol to compute adjusted `p_theo`.
3. This is a refinement signal layered onto Strategy 1.

**Parameters:** Basis threshold, funding rate weight, combination weights.

### 6.6 Strategy Registry

```yaml
# config/strategies.yaml
strategies:
  - name: lead_lag_directional
    module: strategies.lead_lag
    class: LeadLagDirectional
    enabled: true
    kelly_fraction: 0.15
    min_edge_bps: 50
    max_position_pct: 0.20

  - name: implied_vs_realized
    module: strategies.calibration
    class: ImpliedVsRealized
    enabled: true
    kelly_fraction: 0.10
    min_deviation: 0.05
    time_to_expiry_filter_s: 120

  - name: cross_maturity
    module: strategies.cross_maturity
    class: CrossMaturityArb
    enabled: false  # requires more data
    kelly_fraction: 0.10

  - name: cross_exchange_basis
    module: strategies.basis
    class: CrossExchangeBasis
    enabled: true
    basis_threshold_bps: 10
```

---

## 7. Risk Engine

### 7.1 Inventory Risk

Maker-only execution on Polymarket means positions may be held until settlement.

| Control | Value | Action |
|---------|-------|--------|
| Per-underlying max net EV exposure | Configurable (e.g., $100k BTC) | Reject new orders exceeding limit |
| Time-to-expiry de-risking | Below 60s: reduce size 50%. Below 30s: no new orders | Automatic |
| Spread widening | As inventory grows, widen passive quotes by `inventory_skew_bps` | Automatic |

### 7.2 Hedge Option (Future Enhancement)

When net inventory on a single underlying exceeds a configurable threshold:
- Optionally open a delta hedge on Binance/Bybit perpetual futures.
- Hedge ratio: configurable (0.5–1.0 of inventory delta).
- This is P1 priority — build the interface now, implement later.

### 7.3 Drawdown & Kill-Switch

| Level | Trigger | Action |
|-------|---------|--------|
| Strategy-level | 1% loss of equity for the day | Disable that strategy until next UTC day |
| Underlying-level | 2% loss on BTC/ETH/SOL combined | Disable all strategies on that underlying |
| Global | 3% daily equity loss | Disable ALL Polymarket bots for the day |
| Manual override | CLI command or web button | Pause/resume any individual strategy or all |

**Kill signal storage:** `Redis key: kill_signals:{scope}` with TTL = end of UTC day.

### 7.4 Latency & Failure Controls

1. **Latency spike:** If end-to-end latency p95 > 2000ms for 30s, switch to wide passive-only mode.
2. **WebSocket reconnect:** Exponential backoff (1s, 2s, 4s, 8s, max 30s) for both Binance and Polymarket WS feeds.
3. **Stale data guard:** If no Binance tick for > 5s, mark state as stale and halt quoting.
4. **API rate limiting:** Track Polymarket CLOB rate limits per their docs, queue orders if approaching limits.

---

## 8. Alpha Evaluation Framework

### 8.1 `alpha_evaluator` Module

Computes per-strategy, per-market, per-underlying:

| Metric | Description |
|--------|-------------|
| Hit rate vs implied | % of trades where outcome matched prediction, vs what implied prob predicted |
| Brier score | Mean squared error of probability forecasts vs outcomes |
| Calibration curve | Binned: for each predicted bucket, actual outcome frequency |
| PnL per trade | Net of fees, bucketed by time-to-expiry |
| PnL by underlying | BTC vs ETH vs SOL breakdown |
| PnL by hour of day | Time-of-day alpha patterns |
| Sharpe ratio | Annualized, per-strategy and aggregate |
| Max drawdown | Rolling max drawdown per strategy |

### 8.2 Persistence

```sql
CREATE TABLE alpha_evaluations (
    eval_id          SERIAL PRIMARY KEY,
    eval_type        TEXT NOT NULL,        -- 'backtest' or 'live'
    strategy_name    TEXT NOT NULL,
    underlying       TEXT,
    period_start     TIMESTAMPTZ,
    period_end       TIMESTAMPTZ,
    n_trades         INTEGER,
    hit_rate         NUMERIC,
    brier_score      NUMERIC,
    pnl_total        NUMERIC,
    pnl_per_trade    NUMERIC,
    sharpe           NUMERIC,
    max_drawdown     NUMERIC,
    calibration_json JSONB,               -- bucketed calibration data
    metadata_json    JSONB,               -- params, config snapshot
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
```

### 8.3 Backtest Engine

**Requirements:**
1. Record all Binance + Polymarket tick data to TimescaleDB (replay-ready format).
2. Backtester replays recorded data through any strategy module.
3. Fee model applied identically in backtest and live.
4. Output: `BacktestResult` object with all metrics from 8.1.
5. **What-if mode:** Re-run backtests with modified parameters (different Kelly, different fees, different time filters).

---

## 9. Meta-Research Agent

### 9.1 Purpose

The Meta-Research Agent is the "senior quant brain" of the system. It reviews all performance data and proposes next actions — this is what makes the system think, not just execute.

### 9.2 Daily Review Job

A scheduled job (cron, daily at 00:30 UTC) that:

1. **Collects:** All backtest results from Research Lab + all live performance metrics from the last 24h.
2. **Formats:** Summary JSON + markdown tables dumped to `data/daily_review_{date}.json`.
3. **Calls Claude 4.6** with structured prompt:

```
You are a Senior Quantitative Researcher reviewing a Polymarket crypto HFT system.

Here are all current strategies, their configurations, and recent performance:
{strategy_summary_json}

Live metrics (last 24h):
{live_metrics_json}

Backtest metrics (latest runs):
{backtest_metrics_json}

Your tasks:
1. Diagnose which strategies appear overfit (backtest >> live performance).
2. Identify which underlyings/time-of-day/time-to-expiry buckets show genuine edge.
3. Propose 2–3 specific modifications or new ideas to test next.
4. Recommend which bots to scale up, scale down, or turn off.
5. Flag any risk concerns (correlation spikes, unusual drawdown patterns, fee drag).

Output as structured JSON with keys: diagnosis, edge_sources, proposals, scaling_recs, risk_flags.
```

4. **Parses Claude's output** into:
   - New config files to try (YAML diffs).
   - New strategy prompts for the Research Lab.
   - Scaling recommendations applied (with human approval gate).

### 9.3 Research Lab Integration

The Research Lab is a separate process that:
1. Receives strategy ideas (from Meta-Research Agent or manual input).
2. Implements them as `BaseStrategy` subclasses.
3. Runs backtests against recorded data.
4. Stores results in `alpha_evaluations` table.
5. Strategies that pass evaluation thresholds are promoted to live (with human approval).

### 9.4 Promotion Criteria

A strategy is eligible for live deployment when:
- Backtest Sharpe > 1.5 (after fees)
- Backtest hit rate > 52% (for directional strategies)
- Brier score < 0.24 (better than naive 50/50)
- Min 200 simulated trades
- No single-day drawdown > 2% in backtest
- Human approval via CLI or dashboard

---

## 10. Execution Engine

### 10.1 Maker-Only Execution

All Polymarket orders are **maker-only** (limit orders that provide liquidity).

**Requirements:**
1. Use `py-clob-client` for Polymarket CLOB API interaction.
2. Orders placed as limit orders at computed price (p_theo adjusted for edge + fee).
3. Reject any fill that would be taker (pay taker fee).
4. Track fill rate and adjust quote aggression based on fill rate targets.

### 10.2 Order Lifecycle

```
[Signal generated] → [Risk check passed] → [Order submitted (limit)]
    → [Order on book] → [Fill (partial/full)] → [Position updated]
    → [Settlement check] → [PnL recorded]
```

### 10.3 Position Tracking

```sql
CREATE TABLE positions (
    position_id    SERIAL PRIMARY KEY,
    market_id      TEXT NOT NULL,
    strategy_name  TEXT NOT NULL,
    side           TEXT NOT NULL,         -- 'YES' or 'NO'
    size           NUMERIC NOT NULL,
    entry_price    NUMERIC NOT NULL,
    entry_ts       TIMESTAMPTZ NOT NULL,
    exit_price     NUMERIC,
    exit_ts        TIMESTAMPTZ,
    pnl_gross      NUMERIC,
    fee_paid       NUMERIC,
    rebate_earned  NUMERIC,
    pnl_net        NUMERIC,
    settled        BOOLEAN DEFAULT FALSE,
    settlement_outcome BOOLEAN
);
```

---

## 11. Data Collection & Storage

### 11.1 Real-Time Feeds

| Source | Data | Frequency | Storage |
|--------|------|-----------|---------|
| Binance Futures WS | BTC/ETH/SOL trades, book (L2 top 20) | Tick | Redis (live) + TimescaleDB (persist) |
| Bybit Futures WS | BTC/ETH/SOL trades, book (L2 top 10) | Tick | Redis (live) + TimescaleDB (persist) |
| Polymarket CLOB WS | YES/NO book, trades, market lifecycle | Tick | Redis (live) + TimescaleDB (persist) |
| Polymarket Gamma API | Market discovery, metadata | 30s poll | TimescaleDB |

### 11.2 Derived Data

| Data | Computation | Storage |
|------|-------------|---------|
| 1m/5m OHLCV (per CEX) | Aggregated from ticks | TimescaleDB hypertable |
| OFI (30s rolling) | From trade ticks | Redis |
| Realized vol (1m, 5m) | From returns | Redis + TimescaleDB |
| Lead–lag estimates | Cross-correlation | Redis + TimescaleDB |
| Calibration curves | From settlement history | TimescaleDB |

---

## 12. Observability & Dashboard

### 12.1 Dashboard Panels

| Panel | Content |
|-------|---------|
| Strategy PnL | Per-strategy PnL curve, Sharpe, drawdown (backtest vs live overlaid) |
| Positions | Current open positions by underlying and market, time to expiry |
| Exposure | Net exposure by underlying, direction, and total |
| Fee Breakdown | Fill rate, maker/taker ratio, fee drag, rebate earned |
| Latency | Histogram: Binance tick → quote decision → order submission |
| Alpha Metrics | Hit rate, Brier score, calibration plots per strategy |
| Risk Status | Kill-switch status, drawdown gauges, inventory levels |
| Meta-Research | Latest Claude review summary, proposals, scaling recs |

### 12.2 Alerting (Telegram)

| Alert | Trigger |
|-------|---------|
| Kill-switch activated | Any level kill triggered |
| Large fill | Fill > $10k |
| Latency spike | p95 > 2000ms |
| Strategy disabled | Auto-disabled due to loss limit |
| Daily summary | EOD PnL, top/bottom strategies, exposure |
| Meta-research output | New Claude review available |

### 12.3 CLI Tools

```bash
# Pause/resume
python cli.py pause --strategy lead_lag_directional
python cli.py resume --all

# Status
python cli.py status           # Active strategies, positions, PnL
python cli.py latency          # Current latency percentiles

# Research
python cli.py backtest --strategy lead_lag --days 7
python cli.py whatif --strategy lead_lag --kelly 0.20 --min-edge 30

# Meta-research
python cli.py review --trigger-now   # Force immediate Claude review
python cli.py proposals              # Show latest Claude proposals
```

---

## 13. Configuration Reference

### 13.1 Environment Variables

```bash
# Polymarket
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_FUNDER=                    # Wallet address

# Binance
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Bybit
BYBIT_API_KEY=
BYBIT_API_SECRET=

# Claude (Meta-Research)
ANTHROPIC_API_KEY=

# Infrastructure
TIMESCALEDB_URL=postgresql://...
REDIS_URL=redis://...
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### 13.2 Trading Config (`config/trading.yaml`)

```yaml
global:
  equity_usd: 250_000
  max_daily_loss_pct: 3.0
  max_strategy_loss_pct: 1.0
  max_underlying_loss_pct: 2.0

markets:
  underlyings: [BTC, ETH, SOL]
  types: [5m, 15m]
  per_contract_max_usd: 50_000
  per_underlying_max_multiplier: 2.0   # 2x equity
  per_direction_max_multiplier: 1.5
  max_simultaneous_markets: 6
  min_net_ev_bps: 50

execution:
  mode: maker_only
  max_slippage_bps: 2.0
  fill_rate_target: 0.40
  stale_data_timeout_s: 5
  latency_circuit_breaker_ms: 2000

risk:
  inventory_skew_bps_per_pct: 5       # widen 5bps per 1% inventory
  time_derisking_threshold_s: 60      # reduce size below 60s to expiry
  time_derisking_factor: 0.50         # halve size
  no_new_orders_below_s: 30           # no new orders below 30s

meta_research:
  schedule_cron: "30 0 * * *"         # daily at 00:30 UTC
  model: claude-opus-4-6
  auto_apply_recs: false              # require human approval
  promotion_sharpe_min: 1.5
  promotion_hit_rate_min: 0.52
  promotion_brier_max: 0.24
  promotion_min_trades: 200
```

---

## 14. Project Structure

```
polymarket-pnl/
├── config/
│   ├── trading.yaml          # Main trading config
│   ├── fees.yaml             # Polymarket fee schedule
│   └── strategies.yaml       # Strategy registry
├── core/
│   ├── state.py              # MarketState dataclass & state builder
│   ├── fee_engine.py         # Fee & rebate calculations
│   ├── lead_lag.py           # Lead–lag estimator
│   ├── theo_engine.py        # p_theo computation
│   ├── scheduler.py          # Multi-market scheduler & capital allocator
│   └── latency.py            # Latency monitoring & circuit breaker
├── strategies/
│   ├── base.py               # BaseStrategy ABC
│   ├── lead_lag.py           # Strategy Family 1
│   ├── calibration.py        # Strategy Family 2
│   ├── cross_maturity.py     # Strategy Family 3
│   ├── basis.py              # Strategy Family 4
│   └── registry.py           # Auto-discovery from strategies.yaml
├── execution/
│   ├── polymarket_client.py  # Polymarket CLOB wrapper
│   ├── order_manager.py      # Order lifecycle management
│   └── fill_tracker.py       # Fill tracking & PnL
├── risk/
│   ├── inventory.py          # Inventory limits & de-risking
│   ├── drawdown.py           # Drawdown guards & kill-switches
│   └── hedge.py              # Future: CEX hedge interface
├── data/
│   ├── binance_feed.py       # Binance WS connector
│   ├── bybit_feed.py         # Bybit WS connector
│   ├── polymarket_feed.py    # Polymarket WS + Gamma connector
│   ├── recorder.py           # Tick data recording for backtests
│   └── db.py                 # TimescaleDB + Redis helpers
├── research/
│   ├── backtester.py         # Backtest engine (replay recorded data)
│   ├── alpha_evaluator.py    # Metrics: Brier, calibration, PnL bucketing
│   ├── whatif.py             # What-if simulator
│   └── meta_agent.py         # Claude 4.6 meta-research agent
├── dashboard/
│   ├── api.py                # FastAPI endpoints
│   └── frontend/             # Next.js dashboard (or Grafana config)
├── cli.py                    # CLI for pause/resume/status/backtest
├── main.py                   # Entry point: starts all components
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── CLAUDE.md                 # Agent guidelines (updated)
└── PRD.md                    # This document
```

---

## 15. Implementation Phases

### Phase 1: Data Infrastructure (Week 1)
- [ ] Binance + Bybit WebSocket feeds with tick recording
- [ ] Polymarket CLOB WebSocket + Gamma API polling
- [ ] TimescaleDB schema + Redis state store
- [ ] MarketState builder with real-time updates
- [ ] Lead–lag estimator (basic cross-correlation)
- [ ] Fee engine with configurable schedule

### Phase 2: Core Trading Engine (Week 2)
- [ ] p_theo computation engine
- [ ] Multi-market scheduler with capital allocation
- [ ] BaseStrategy ABC + Strategy Family 1 (Lead–Lag)
- [ ] Strategy registry with auto-discovery
- [ ] Execution engine: maker-only Polymarket CLOB orders
- [ ] Position tracking + PnL recording

### Phase 3: Risk & Execution Hardening (Week 3)
- [ ] Inventory risk controls
- [ ] Drawdown guards + kill-switches
- [ ] Latency monitoring + circuit breakers
- [ ] WebSocket reconnect with exponential backoff
- [ ] Stale data guards
- [ ] CLI tools (pause/resume/status)

### Phase 4: Research & Evaluation (Week 4)
- [ ] Backtest engine using recorded tick data
- [ ] Alpha evaluator (Brier, calibration, bucketed PnL)
- [ ] Strategy Family 2 (Implied vs Realized)
- [ ] Strategy Family 3 (Cross-Maturity)
- [ ] Strategy Family 4 (Cross-Exchange Basis)
- [ ] What-if simulator

### Phase 5: Meta-Research & Observability (Week 5)
- [ ] Meta-Research Agent (Claude 4.6 daily review)
- [ ] Dashboard (Next.js or Grafana)
- [ ] Telegram alerting
- [ ] Strategy promotion pipeline (backtest → approval → live)
- [ ] Daily report generation

### Phase 6: Production Hardening (Week 6)
- [ ] Docker Compose deployment
- [ ] Monitoring (Prometheus + Grafana)
- [ ] Load testing & latency optimization
- [ ] Hedge interface (stub for P1)
- [ ] Documentation & runbook

---

## 16. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency | p50 < 500ms, p99 < 2000ms | Prometheus histogram |
| Fill rate (maker) | > 35% | Per-strategy from fill_tracker |
| Net Sharpe (after fees) | > 1.5 across all strategies | Alpha evaluator |
| Brier score | < 0.24 | Alpha evaluator |
| Daily PnL stability | < 3% max drawdown | Risk engine |
| Uptime | > 99.5% | Monitoring |
| Strategy addition time | < 1 day from idea to backtest | Process metric |

---

## Appendix A: Polymarket CLOB API Reference

- REST: `https://clob.polymarket.com`
- WebSocket: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Gamma API (market discovery): `https://gamma-api.polymarket.com/markets`
- Authentication: API key + secret (HMAC signing)
- Rate limits: Per their current docs (check and implement accordingly)
- `py-clob-client` Python SDK for order placement

## Appendix B: Binance Futures API Reference

- WebSocket: `wss://fstream.binance.com/ws`
- Streams: `btcusdt@trade`, `btcusdt@depth20@100ms`, `btcusdt@kline_1m`
- REST: `https://fapi.binance.com`
- Rate limits: 2400 request weight/min, 300 orders/min

## Appendix C: Bybit Futures API Reference

- WebSocket: `wss://stream.bybit.com/v5/public/linear`
- Topics: `orderbook.50.BTCUSDT`, `publicTrade.BTCUSDT`
- REST: `https://api.bybit.com`
