# Polymarket Backtesting: Bias Audit & Corrected Results

## Summary

The original backtest showing **Sharpe 20.95** was severely biased. After corrections:

| Metric | Original (Biased) | Corrected |
|--------|------------------|-----------|
| Sharpe Ratio | 20.95 | **-2 to +1** |
| Win Rate | 77.3% | **2-18%** |
| Daily Return | Several % | **Negative** |

## Biases Identified & Fixed

### 1. Trading Through Resolution Events
**Problem**: The dataset contains 34 separate 15-minute markets that RESOLVE (price → 0 or 1). The original backtest treated this as continuous price action.

**Impact**: +1000% inflated returns from "trading" price jumps that aren't tradeable.

**Fix**: Segmented by market, forced close at progress < 0.85.

### 2. Fill at Mid Price
**Problem**: Original backtest assumed fills at mid-price.

**Reality**: 
- Buys execute at ASK + slippage
- Sells execute at BID - slippage
- Average spread: **2.35%**

**Fix**: Execute at bid/ask with realistic slippage.

### 3. No Fee Modeling
**Problem**: Zero transaction costs.

**Reality**:
- Maker fee: 0.3%
- Taker fee: 0.5%
- Slippage: ~0.3%
- **Round-trip cost: 3.95%**

### 4. Random Signal Test Failed
**Problem**: Random signals produced Sharpe 51+ (should be ~0).

**Root Cause**: Price movements from resolution events contaminated all signals.

## Corrected Economics

### Cost Structure
```
Average spread:     2.35%
Slippage (both):    0.60%
Fees (both):        1.00%
─────────────────────────
Round-trip cost:    3.95%
```

### Break-Even Requirements
- To profit with 5% average trade capture: **Need 89.5% win rate**
- To profit with 10% average trade capture: **Need 69.8% win rate**
- With 50% random accuracy: **Lose 3.95% per trade guaranteed**

### Price Movement Reality
```
Within tradeable window (progress 0.1-0.85):
- Mean movement: 126.55%  (looks great!)
- But: direction is unpredictable
- Perfect entry/exit needed to capture
```

## Strategies Tested (After Corrections)

| Strategy | Test PnL | Win Rate | Verdict |
|----------|----------|----------|---------|
| Spread Capture | -$275 | 17.5% | ❌ Loses |
| Mean Reversion | -$3,177 | 2.1% | ❌ Loses badly |
| Random | -$3,782 | 7.4% | ❌ Loses (baseline) |

**All strategies lose** because:
1. Direction is not predictable from order book signals
2. Costs eat 4% per round-trip
3. Short 15-minute windows limit recovery

## Realistic Expectations

### What a Good Strategy Would Look Like
```
Win Rate:      55-60%
Avg Win:       8-10%
Avg Loss:      5-7%
Round-trips:   4-8 per day
Daily Return:  0.2-0.5%
Sharpe:        1.5-2.5 (annualized)
```

### What Would Actually Work
1. **Market Making**: Capture spread by posting limits on both sides
   - Requires: Inventory management, adverse selection avoidance
   - Risk: Getting picked off by informed traders

2. **Resolution Prediction**: Predict final outcome (UP or DOWN)
   - Requires: External signal (CEX price movement, news)
   - Edge: If you know outcome 1 minute before resolution

3. **Arbitrage**: YES + NO < $1.00 opportunities
   - Rare: Typically 0-3 per day
   - Size-limited: Usually <$50 profit each

## Files Created

| File | Purpose |
|------|---------|
| `realistic_backtest.py` | First fix attempt |
| `realistic_backtest_v2.py` | Filter resolution events |
| `proper_backtest.py` | Per-market segmentation |
| `BACKTEST_FINDINGS.md` | This document |

## Conclusion

The original Sharpe 20.95 was a backtesting artifact, not real alpha. After proper corrections:

- **No strategy tested is profitable**
- **4% round-trip costs require high prediction accuracy**
- **Need external signal (CEX lead-lag) for edge**

A Sharpe 2.0 strategy you can trust beats a Sharpe 20.0 that's a bug.
