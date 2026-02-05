# Top 3 Unbiased Strategies for Polymarket 15-Min Crypto

## Executive Summary

After correcting all backtesting biases, these are the only strategies with positive expected value:

| Rank | Strategy | Test Return | Win Rate | Requirements |
|------|----------|-------------|----------|--------------|
| 🥇 | **CEX Lead-Lag** | +6.0% | 75% | Binance API |
| 🥈 | **Contrarian at Extremes** | +2.0% | 33% | None |
| 🥉 | Early Momentum | -5.6% | 42% | None |

**Key Finding**: The only consistently profitable strategy requires **external data** (Binance price). Order book signals alone are not predictive.

---

## Strategy 1: CEX Lead-Lag (Best)

### Concept
Binance/Coinbase price movements **lead** Polymarket prediction market prices by 10-60 seconds. Trade Polymarket in the direction Binance is moving.

### Mechanism
```
1. Fetch BTC price from Binance every 5 seconds
2. Calculate 30-second momentum: (price_now - price_30s_ago) / price_30s_ago
3. If momentum > +0.1%: BUY Polymarket "UP" side
4. If momentum < -0.1%: BUY Polymarket "DOWN" side (or short "UP")
5. Exit at progress = 0.75 (before resolution uncertainty)
```

### Why It Works
- Academic research confirms CEX leads other venues by 7-17 seconds
- Polymarket is even slower (lower liquidity, fewer arbitrageurs)
- You're essentially front-running the prediction market

### Implementation
```python
from binance.client import Client

def get_signal():
    client = Client()
    
    # Current price
    ticker = client.get_ticker(symbol='BTCUSDT')
    current = float(ticker['lastPrice'])
    
    # Price 30s ago (you'd need to cache this)
    old_price = price_cache.get(time.time() - 30)
    
    momentum = (current - old_price) / old_price
    
    if momentum > 0.001:  # 0.1%
        return "BUY_UP"
    elif momentum < -0.001:
        return "BUY_DOWN"
    return "HOLD"
```

### Expected Performance
| Metric | Conservative | Optimistic |
|--------|-------------|------------|
| Win Rate | 60% | 75% |
| Per-Trade Edge | 3% | 8% |
| Sharpe Ratio | 1.5 | 2.5 |

---

## Strategy 2: Contrarian at Extremes

### Concept
When price hits extreme levels (< 0.15 or > 0.85), it often reverts before resolution. Fade the extreme.

### Mechanism
```
1. Monitor Polymarket price during 15-min window
2. If price drops below $0.15: BUY (oversold)
3. If price rises above $0.85: SELL/SHORT (overbought)
4. Exit when price reverts to $0.30-0.70 range
```

### Why It Works
- Extreme prices reflect panic/euphoria
- Market makers step in at extremes
- Resolution is still uncertain at early extremes

### Caution
- Only works if extreme happens early in window (progress < 0.5)
- If extreme happens late, it's likely the final outcome

### Expected Performance
| Metric | Value |
|--------|-------|
| Win Rate | 35-45% |
| Avg Win | 15-25% |
| Avg Loss | 8-12% |
| Sharpe | 0.8-1.5 |

---

## Strategy 3: Early Momentum (Marginal)

### Concept
The direction established in the first 20% of the window often continues.

### Mechanism
```
1. Wait for progress = 0.20
2. Compare current price to starting price
3. If up > 2%: Go LONG
4. If down > 2%: Go SHORT
5. Exit at progress = 0.70
```

### Why It's Marginal
- Works in trending markets
- Fails in choppy/reversing markets
- Net result: Slightly negative after costs

### When to Use
Only use this as a **confirmation signal** for Strategy 1, not standalone.

---

## Cost Structure (Critical)

All strategies must overcome these costs:

| Cost | Amount | Impact |
|------|--------|--------|
| Spread | 2.35% | Pay on entry AND exit |
| Slippage | 0.30% | Each direction |
| Taker Fee | 0.50% | Each direction |
| **Total Round-Trip** | **~4%** | Must profit more than this |

### Break-Even Requirements
- With 50% win rate: **Impossible** (lose 4% per trade)
- With 60% win rate: Need 10%+ avg wins
- With 70% win rate: Need 6%+ avg wins

---

## Implementation Checklist

### Phase 1: Data Collection (Week 1)
- [ ] Set up Binance WebSocket feed
- [ ] Cache price every 1 second
- [ ] Calculate rolling 30s momentum
- [ ] Log signals to file

### Phase 2: Paper Trading (Weeks 2-3)
- [ ] Connect to Polymarket API
- [ ] Execute simulated trades based on signals
- [ ] Track hypothetical P&L
- [ ] Measure actual win rate

### Phase 3: Live Trading (Week 4+)
- [ ] Start with $100 max position
- [ ] Set 10% stop loss per trade
- [ ] Scale up only after 100+ profitable trades
- [ ] Monitor for alpha decay

---

## Files Reference

| File | Purpose |
|------|---------|
| `top3_strategies.py` | Backtest of top 3 strategies |
| `production_strategy.py` | Live trading skeleton with Binance |
| `proper_backtest.py` | Unbiased per-market backtester |
| `BACKTEST_FINDINGS.md` | Documentation of bias corrections |

---

## Reality Check

### What You Should Expect
```
Daily profit target: $50-200 (not $8,000)
Win rate: 55-65%
Sharpe: 1.5-2.5
Drawdowns: 10-20%
```

### What You Need
1. **Fast API access** to Binance (< 100ms latency)
2. **Real-time monitoring** (not manual trading)
3. **Disciplined risk management** (stop losses, position sizing)
4. **Large sample size** (100+ trades to validate)

### The $8,000/day Question
To make $8,000/day with a 5% edge per trade:
- Need $160,000 in daily turnover
- At $100/trade: 1,600 trades/day
- With 96 markets/day: 17 trades per market

This is **not realistic** for a single trader. Professional quant funds with:
- Co-located servers
- Multiple markets
- Automated execution
- Large capital base

...might achieve this, but individual traders should target $50-500/day.
