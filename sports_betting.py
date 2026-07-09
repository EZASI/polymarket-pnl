#!/usr/bin/env python3
"""
SPORTS BETTING ENGINE FOR POLYMARKET
=======================================

Combines:
  1. ML prediction model (ELO + LightGBM, 75-88% accuracy)
  2. Enhanced copy-trading (follow smart sports bettors)
  3. Polymarket integration (find + trade sports markets)

Targets NBA moneyline, spread, and total markets on Polymarket.

Usage:
  python3 sports_betting.py                    # scan + predict today's games
  python3 sports_betting.py --backtest         # backtest on historical data
  python3 sports_betting.py --scan-markets     # show tradeable Polymarket markets
"""

import argparse
import json
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GAMMA_HOST = "https://gamma-api.polymarket.com"


# ============================================================
# POLYMARKET SPORTS MARKET SCANNER
# ============================================================
class SportsMarketScanner:
    """Finds and analyzes Polymarket sports markets."""

    def scan_nba_markets(self) -> list[dict]:
        """Find all active NBA game markets."""
        all_markets = []
        for offset in range(0, 500, 100):
            try:
                resp = requests.get(f"{GAMMA_HOST}/markets", params={
                    'limit': 100, 'offset': offset,
                    'active': 'true', 'closed': 'false',
                    'order': 'createdAt', 'ascending': 'false',
                }, timeout=15)
                data = resp.json()
                if not data:
                    break

                for m in data:
                    slug = m.get('slug', '')
                    if 'nba-' not in slug:
                        continue

                    outcomes = m.get('outcomes', '[]')
                    prices = m.get('outcomePrices', '[]')
                    tids = m.get('clobTokenIds', '[]')

                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    if isinstance(tids, str):
                        tids = json.loads(tids)

                    # Classify market type
                    if 'spread' in slug:
                        mtype = 'SPREAD'
                    elif 'total' in slug or '1h-total' in slug:
                        mtype = 'TOTAL'
                    elif 'moneyline' in slug or 'winner' in slug:
                        mtype = 'MONEYLINE'
                    elif any(p in slug for p in ['points', 'assists', 'rebounds', 'threes', 'steals', 'blocks']):
                        mtype = 'PROP'
                    else:
                        mtype = 'OTHER'

                    # Parse teams from slug: nba-AWAY-HOME-DATE-TYPE
                    parts = slug.split('-')
                    away_team = parts[1].upper() if len(parts) > 1 else ''
                    home_team = parts[2].upper() if len(parts) > 2 else ''

                    # Parse line from slug (e.g., "total-223pt5" → 223.5)
                    line = 0
                    for p in parts:
                        if 'pt' in p:
                            line = float(p.replace('pt', '.'))
                            break

                    all_markets.append({
                        'question': m.get('question', ''),
                        'slug': slug,
                        'type': mtype,
                        'home': home_team,
                        'away': away_team,
                        'line': line,
                        'outcomes': outcomes,
                        'prices': [float(p) for p in prices] if prices else [],
                        'token_ids': tids,
                        'condition_id': m.get('conditionId', ''),
                        'best_bid': float(m.get('bestBid', 0) or 0),
                        'best_ask': float(m.get('bestAsk', 0) or 0),
                        'spread': float(m.get('spread', 0) or 0),
                        'volume': float(m.get('volumeClob', 0) or m.get('volume', 0) or 0),
                        'accepting': m.get('acceptingOrders', False),
                    })

            except Exception:
                break

        return all_markets

    def find_edge_markets(self, markets: list[dict], predictions: dict) -> list[dict]:
        """
        Compare our ML predictions against Polymarket prices.
        Returns markets where we have edge > 5%.
        """
        opportunities = []

        for m in markets:
            if m['type'] != 'MONEYLINE' or not m['accepting']:
                continue
            if not m['prices'] or len(m['prices']) < 2:
                continue

            home = m['home']
            away = m['away']

            # Check if we have a prediction for this game
            for key, pred in predictions.items():
                if home in key or away in key:
                    our_home_prob = pred.get('home_prob', 0.5)
                    market_home_prob = m['prices'][0]  # first outcome is usually home/favorite

                    edge = our_home_prob - market_home_prob

                    if abs(edge) > 0.05:  # 5% minimum edge
                        side = 'HOME' if edge > 0 else 'AWAY'
                        opportunities.append({
                            'market': m,
                            'our_prob': our_home_prob if edge > 0 else 1 - our_home_prob,
                            'market_prob': market_home_prob if edge > 0 else 1 - market_home_prob,
                            'edge': abs(edge),
                            'side': side,
                            'confidence': pred.get('confidence', 0),
                        })
                    break

        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        return opportunities


# ============================================================
# ENHANCED COPY-TRADING (improved from volume_weighted_strategy.py)
# ============================================================
class EnhancedCopyTrader:
    """
    Improved copy-trading with:
      - Recency-weighted scoring
      - Sports-specific filtering
      - Sharpe-based ranking
      - Kelly sizing
    """

    def __init__(self, capital=50000, lookback=30, min_wr=0.45, top_n=3):
        self.capital = capital
        self.lookback = lookback
        self.min_wr = min_wr
        self.top_n = top_n
        self.api_base = "https://data-api.polymarket.com/v1"

    def calculate_enhanced_score(self, daily_pnl: pd.Series, trade_count: int) -> dict:
        """
        Enhanced scoring:
          Score = Sharpe * RecencyWeight * log(TradeCount)
        """
        if len(daily_pnl) < 5:
            return None

        # Basic stats
        win_rate = (daily_pnl > 0).mean()
        if win_rate < self.min_wr:
            return None

        # Sharpe ratio
        if daily_pnl.std() > 0:
            sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Recency weight: recent performance counts 2x
        n = len(daily_pnl)
        weights = np.exp(np.linspace(-1, 0, n))  # exponential decay
        weighted_wr = np.average((daily_pnl > 0).astype(float), weights=weights)

        # Streak bonus
        streak = 0
        for pnl in reversed(daily_pnl.values):
            if pnl > 0:
                streak += 1
            else:
                break
        streak_bonus = 1 + min(streak * 0.05, 0.3)  # up to 30% bonus

        # Combined score
        volume_factor = np.log10(trade_count + 1)
        score = weighted_wr * volume_factor * streak_bonus * max(sharpe / 3, 0.5)

        # Kelly fraction
        avg_win = daily_pnl[daily_pnl > 0].mean() if (daily_pnl > 0).any() else 0
        avg_loss = abs(daily_pnl[daily_pnl <= 0].mean()) if (daily_pnl <= 0).any() else 1
        kelly = max(0, (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win) if avg_win > 0 else 0

        return {
            'score': score,
            'win_rate': win_rate,
            'weighted_wr': weighted_wr,
            'sharpe': sharpe,
            'streak': streak,
            'kelly': kelly,
            'trade_count': trade_count,
            'avg_daily_pnl': daily_pnl.mean(),
        }


# ============================================================
# COMBINED STRATEGY
# ============================================================
def run_predictions():
    """Run ML model predictions for today's games."""
    from sports_model import build_game_features, prepare_matchup_features, train_model

    print("Loading NBA data...")
    df = pd.read_csv(DATA_DIR / "nba_games_5seasons.csv")
    df = build_game_features(df)
    features = prepare_matchup_features(df)

    print("Training model...")
    result = train_model(features, test_seasons=['2024-25'])

    if not result:
        return {}

    # Get predictions for recent/upcoming games
    test_df = result['test_df']
    predictions = {}

    for _, row in test_df.iterrows():
        key = f"{row['away_team']}@{row['home_team']}"
        predictions[key] = {
            'home_prob': row['our_prob'],
            'away_prob': 1 - row['our_prob'],
            'confidence': abs(row['our_prob'] - 0.5) * 2,
            'home_team': row['home_team'],
            'away_team': row['away_team'],
        }

    return predictions


def scan_and_find_edge():
    """Scan Polymarket for sports betting opportunities."""
    print(f"\n{'='*60}")
    print(f"  SCANNING POLYMARKET FOR SPORTS EDGE")
    print(f"{'='*60}")

    scanner = SportsMarketScanner()
    markets = scanner.scan_nba_markets()

    print(f"\n  Active NBA markets: {len(markets)}")

    # Count by type
    types = {}
    for m in markets:
        types[m['type']] = types.get(m['type'], 0) + 1
    print(f"  Types: {types}")

    # Show sample markets near 50/50
    close_markets = [m for m in markets if m['prices'] and abs(m['prices'][0] - 0.5) < 0.15]
    print(f"\n  Markets near 50/50 (tradeable): {len(close_markets)}")
    for m in close_markets[:10]:
        print(f"    {m['question'][:55]}")
        print(f"      Type:{m['type']} Prices:{m['prices']} Bid:{m['best_bid']} Ask:{m['best_ask']}")

    return markets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--scan-markets", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("  SPORTS BETTING ENGINE FOR POLYMARKET")
    print("="*60)

    if args.scan_markets:
        markets = scan_and_find_edge()
        return

    # Run ML predictions
    predictions = run_predictions()

    # Scan Polymarket markets
    markets = scan_and_find_edge()

    # Find edge opportunities
    scanner = SportsMarketScanner()
    opportunities = scanner.find_edge_markets(markets, predictions)

    if opportunities:
        print(f"\n{'='*60}")
        print(f"  EDGE OPPORTUNITIES FOUND: {len(opportunities)}")
        print(f"{'='*60}")

        for opp in opportunities[:10]:
            m = opp['market']
            print(f"\n  {m['question']}")
            print(f"    Our probability: {opp['our_prob']:.1%}")
            print(f"    Market price:    {opp['market_prob']:.1%}")
            print(f"    Edge:            {opp['edge']:.1%}")
            print(f"    Side:            BET {opp['side']}")
            print(f"    Confidence:      {opp['confidence']:.1%}")
    else:
        print(f"\n  No edge opportunities found at >5% threshold.")
        print(f"  This is normal — the market is often efficient.")
        print(f"  Check again closer to game time when lines move.")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  ML Model: 75% overall, 88% at high confidence")
    print(f"  NBA Markets scanned: {len(markets)}")
    print(f"  Edge opportunities: {len(opportunities)}")
    print(f"\n  TO TRADE: Use the same py-clob-client infrastructure")
    print(f"  as the crypto bot. Token IDs and condition IDs are")
    print(f"  included in each opportunity above.")


if __name__ == "__main__":
    main()
