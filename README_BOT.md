# Polymarket Crypto Prediction Bot

A demo trading bot for Polymarket crypto prediction markets, based on **academically validated signals** with proper risk management and backtesting capabilities.

## Overview

This bot implements a multi-signal strategy combining:

| Signal | Academic Validation | Source |
|--------|---------------------|--------|
| Order Book Imbalance (OBI) | **Strong** - 72% accuracy | Wang (2025), Cont et al. (2014) |
| Lead-Lag (Lévy Area) | **Strong** - >20% annual returns | Cartea, Cucuringu, Jin (2023) |
| Funding Rate | **Moderate** - Contrarian only | He et al., Presto Research |
| Liquidation Cascades | **Moderate** - Contrarian indicator | Cheng et al. (2021), Ali (2025) |
| Social Sentiment | **Weak** - Small effect size | Renault et al. |
| Whale Tracking | **Disabled** - Only valid at hours | Chi, Hou, Trinh (2024) |

## Quick Start

```bash
# Install dependencies
pip install numpy

# Run interactive demo
python run_demo.py

# Or run backtest directly
python -m bot.main backtest --start 2024-01-01 --end 2024-12-31
```

## Project Structure

```
bot/
├── config.py           # Configuration with academic citations
├── strategy.py         # Main strategy engine
├── main.py            # Bot orchestrator
├── data/
│   └── fetchers.py    # Data fetchers (simulated for demo)
├── signals/
│   └── generators.py  # Signal generators
├── models/
│   └── ml_models.py   # XGBoost + LSTM ensemble
└── backtest/
    └── engine.py      # Backtesting framework
```

## Usage Modes

### 1. Backtesting

```bash
# Quick backtest
python -m bot.main backtest

# Custom date range
python -m bot.main backtest --start 2024-06-01 --end 2024-12-31

# Conservative mode (strongly validated signals only)
python -m bot.main backtest --conservative

# Export results
python -m bot.main backtest --output results.json

# Custom capital
python -m bot.main backtest --capital 50000
```

### 2. Signal Monitoring

```bash
# Monitor signals for 60 minutes
python -m bot.main monitor --duration 60
```

### 3. Paper Trading

```bash
# Paper trade for 4 hours
python -m bot.main paper --hours 4
```

## Signal Details

### Order Book Imbalance (OBI)

**Academic Basis:**
- Cont, Kukanov, Stoikov (2014): R² ~65% for price changes
- Wang (2025): 72% binary classification accuracy at 500ms
- Brazilian BMF&Bovespa study: ±0.6 threshold validated

**Implementation:**
```python
OBI = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
Signal when |OBI| > 0.6
```

### Lead-Lag (Lévy Area)

**Academic Basis:**
- Cartea, Cucuringu, Jin (2023): >20% annualized returns
- Schei (2019): 70% directional accuracy, up to 15s lead
- Bitwise SEC filing: 6.5-17s lead from futures to spot

**Implementation:**
Uses rough path theory to detect nonlinear lead-lag relationships between exchanges.

### Funding Rate

**Academic Basis:**
- He et al.: Sharpe ratios 1.8-3.5 for deviation strategies
- Presto Research: R² ~0 for single-asset prediction

**IMPORTANT:** Funding rates have **near-zero predictive power** for future price movements. Used only as a contrarian regime indicator for extreme values (>100% annualized).

### ML Ensemble

**XGBoost (Direction):**
- Springer (2025): 97% accuracy for Bitcoin
- MDPI (2025): R² 0.9694-0.9827

**LSTM (Magnitude):**
- Tripathy et al. (2024): Bi-LSTM MAE 0.633

## Risk Management

The bot implements several risk controls:

1. **Death Zone Avoidance**: No trades when probability is 45-55% (transaction costs eliminate edge)

2. **Position Sizing**: Kelly-inspired sizing based on edge and confidence

3. **Daily Loss Limits**: Stops trading after configurable daily loss

4. **Maximum Positions**: Limits concurrent open positions

5. **Minimum Edge Requirement**: Only trades when estimated edge exceeds costs

## Critical Limitations

### Timeframe Mismatch (from Academic Analysis)

> "The most critical validation failure is the **timeframe mismatch**: academic evidence supports on-chain and whale signals at 1-24 hour windows, not the minutes-level prediction required for 15-minute Polymarket resolution."

### Alpha Decay

> "McLean & Pontiff (2016): ~50% of anomaly alpha disappears post-publication. Alpha decay expectations of 12 months suggest any competitive advantage will be temporary."

### Missing from Literature

- No peer-reviewed study directly measures lead-lag between Polymarket and spot exchanges
- TikTok sentiment claims trace to arXiv preprint only
- Whale tracking only validated at hour+ timeframes

## Configuration

### Default Configuration

Uses all signals with academic-based weights:

```python
OBI: 25% weight
Lead-Lag: 25% weight
Funding Rate: 15% weight
Sentiment: 10% weight
Liquidation: 10% weight
```

### Conservative Configuration

Only strongly validated signals:

```python
OBI: 40% weight
Lead-Lag: 40% weight
Liquidation: 20% weight
```

Enable with `--conservative` flag.

## Extending the Bot

### Adding Real Data Sources

Replace simulated fetchers in `bot/data/fetchers.py` with real API calls:

```python
class RealOrderBookFetcher(DataFetcher):
    async def fetch(self) -> OrderBook:
        # Implement real exchange API call
        response = await self.client.get_orderbook("BTC-USD")
        return OrderBook(...)
```

### Adding New Signals

1. Create generator in `bot/signals/generators.py`
2. Add configuration in `bot/config.py`
3. Register in `PolymarketStrategy.generate_signals()`

### Production ML Models

Replace simplified models with production libraries:

```python
# In bot/models/ml_models.py
import xgboost as xgb
import torch

class ProductionXGBoost(MLModel):
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)
```

## Disclaimer

This is a **demonstration bot** for educational purposes. Key warnings:

1. **Simulated Data**: Uses synthetic data for backtesting. Real performance will differ.

2. **No Live Trading**: Dry run mode only. Live trading requires additional implementation.

3. **Alpha Decay**: Signals described here are publicly documented and likely already arbitraged.

4. **No Guarantees**: Past backtest performance does not guarantee future results.

5. **Risk of Loss**: Prediction market trading involves substantial risk of loss.

## Academic References

1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*.

2. Cartea, A., Cucuringu, M., & Jin, H. (2023). Lévy area analysis and lead-lag detection. *Oxford-Man Institute*.

3. Wang (2025). Order book imbalance for crypto price prediction. *Bybit Research*.

4. Tripathy et al. (2024). Bi-LSTM for cryptocurrency forecasting. *Frontiers in Blockchain*.

5. He, Manela, Ross, & von Wachter. Funding rate deviations in crypto perpetuals. *arXiv:2212.06888*.

6. McLean, R.D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*.
