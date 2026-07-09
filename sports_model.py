#!/usr/bin/env python3
"""
SPORTS ML PREDICTION MODEL
============================

ELO ratings + statistical features + LightGBM for NBA game predictions.
Compares our probability vs Polymarket price to find edge.

Features:
  - ELO ratings (rolling, K=20)
  - Home/away advantage
  - Recent form (last 5/10 games)
  - Rest days (back-to-back penalty)
  - Points scored/allowed averages
  - Pace factor
  - Market-implied probability (if available)

Usage:
  python3 sports_model.py                    # train + backtest
  python3 sports_model.py --predict-today    # predict today's games
"""

import argparse
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")


# ============================================================
# ELO RATING SYSTEM
# ============================================================
class ELOSystem:
    """ELO rating system for NBA teams."""

    def __init__(self, k=20, home_advantage=100, initial=1500):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.initial
        return self.ratings[team]

    def expected_score(self, team_a: str, team_b: str, a_is_home: bool = True) -> float:
        """Expected win probability for team A."""
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        if a_is_home:
            ra += self.home_advantage
        else:
            rb += self.home_advantage
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(self, winner: str, loser: str, margin: int = 0):
        """Update ratings after a game."""
        # Margin of victory multiplier
        mov_mult = np.log(max(abs(margin), 1) + 1) * (2.2 / (2.2 + 0.001 * abs(
            self.get_rating(winner) - self.get_rating(loser))))

        expected_w = self.expected_score(winner, loser)
        expected_l = 1 - expected_w

        self.ratings[winner] = self.get_rating(winner) + self.k * mov_mult * (1 - expected_w)
        self.ratings[loser] = self.get_rating(loser) + self.k * mov_mult * (0 - expected_l)

    def season_reset(self, revert_pct=0.25):
        """Regress ratings to mean at season start."""
        for team in self.ratings:
            self.ratings[team] = self.initial + (1 - revert_pct) * (self.ratings[team] - self.initial)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_game_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for each game from NBA raw data.
    Input: raw game data from nba_api (each row = one team's game).
    Output: one row per game with features for both teams.
    """
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    # Parse home/away from MATCHUP column
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

    # Get opponent from MATCHUP
    def get_opponent(row):
        parts = row['MATCHUP'].replace(' vs. ', '|').replace(' @ ', '|').split('|')
        return parts[1].strip() if len(parts) > 1 else ''
    df['OPPONENT'] = df.apply(get_opponent, axis=1)

    # Build ELO ratings
    elo = ELOSystem(k=20, home_advantage=100)
    elo_ratings = []
    elo_expected = []
    season_tracker = None

    for _, row in df.iterrows():
        team = row['TEAM_ABBREVIATION']
        opp = row['OPPONENT']
        season = row.get('SEASON', '')

        # Season reset
        if season != season_tracker:
            if season_tracker is not None:
                elo.season_reset()
            season_tracker = season

        # Pre-game ELO
        team_elo = elo.get_rating(team)
        opp_elo = elo.get_rating(opp)
        exp = elo.expected_score(team, opp, a_is_home=bool(row['IS_HOME']))

        elo_ratings.append(team_elo)
        elo_expected.append(exp)

        # Update ELO after game
        pts = row.get('PTS', 0)
        # Need opponent's points - we'll calculate margin later
        if row['WL'] == 'W':
            elo.update(team, opp, margin=10)  # approximate margin
        else:
            elo.update(opp, team, margin=10)

    df['ELO'] = elo_ratings
    df['ELO_EXPECTED'] = elo_expected

    # Rolling features
    for team_name, team_df in df.groupby('TEAM_ABBREVIATION'):
        idx = team_df.index

        # Recent form (last 5, 10 games)
        wins = (team_df['WL'] == 'W').astype(float)
        df.loc[idx, 'FORM_5'] = wins.rolling(5, min_periods=1).mean().shift(1)
        df.loc[idx, 'FORM_10'] = wins.rolling(10, min_periods=1).mean().shift(1)

        # Points scored/allowed averages
        df.loc[idx, 'PTS_AVG_10'] = team_df['PTS'].rolling(10, min_periods=3).mean().shift(1)

        # Rest days
        dates = team_df['GAME_DATE']
        df.loc[idx, 'REST_DAYS'] = dates.diff().dt.days.fillna(3)

        # Win streak
        streak = []
        current = 0
        for wl in team_df['WL'].values:
            if wl == 'W':
                current = max(current, 0) + 1
            else:
                current = min(current, 0) - 1
            streak.append(current)
        df.loc[idx, 'STREAK'] = [0] + streak[:-1]  # shift by 1

    # Back-to-back flag
    df['IS_B2B'] = (df['REST_DAYS'] <= 1).astype(int)

    # Target: 1 if team wins, 0 if loses
    df['TARGET'] = (df['WL'] == 'W').astype(int)

    return df


def prepare_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per game (not per team) with features for both sides.
    """
    # Only take home team rows to avoid duplicates
    home = df[df['IS_HOME'] == 1].copy()

    # Get away team data
    away = df[df['IS_HOME'] == 0].copy()

    # Merge home and away on GAME_ID
    merged = home.merge(
        away[['GAME_ID', 'TEAM_ABBREVIATION', 'ELO', 'ELO_EXPECTED', 'FORM_5', 'FORM_10',
              'PTS_AVG_10', 'REST_DAYS', 'IS_B2B', 'STREAK', 'PTS']],
        on='GAME_ID',
        suffixes=('_HOME', '_AWAY'),
        how='inner',
    )

    if merged.empty:
        return pd.DataFrame()

    # Create features
    features = pd.DataFrame({
        'game_id': merged['GAME_ID'],
        'date': merged['GAME_DATE'],
        'home_team': merged['TEAM_ABBREVIATION_HOME'],
        'away_team': merged['TEAM_ABBREVIATION_AWAY'],
        'season': merged.get('SEASON', ''),

        # ELO features
        'elo_diff': merged['ELO_HOME'] - merged['ELO_AWAY'],
        'elo_home': merged['ELO_HOME'],
        'elo_away': merged['ELO_AWAY'],
        'elo_expected': merged['ELO_EXPECTED_HOME'],

        # Form features
        'form_diff_5': merged['FORM_5_HOME'] - merged['FORM_5_AWAY'],
        'form_diff_10': merged['FORM_10_HOME'] - merged['FORM_10_AWAY'],
        'home_form_5': merged['FORM_5_HOME'],
        'away_form_5': merged['FORM_5_AWAY'],

        # Points
        'pts_diff': merged['PTS_AVG_10_HOME'] - merged['PTS_AVG_10_AWAY'],
        'home_pts_avg': merged['PTS_AVG_10_HOME'],
        'away_pts_avg': merged['PTS_AVG_10_AWAY'],

        # Rest
        'rest_diff': merged['REST_DAYS_HOME'] - merged['REST_DAYS_AWAY'],
        'home_b2b': merged['IS_B2B_HOME'],
        'away_b2b': merged['IS_B2B_AWAY'],

        # Streak
        'streak_diff': merged['STREAK_HOME'] - merged['STREAK_AWAY'],

        # Target: home team wins
        'home_win': merged['TARGET'],

        # Actual scores (for total/spread analysis)
        'home_pts': merged['PTS_HOME'],
        'away_pts': merged['PTS_AWAY'],
        'total_pts': merged['PTS_HOME'] + merged['PTS_AWAY'],
        'margin': merged['PTS_HOME'] - merged['PTS_AWAY'],
    })

    return features


# ============================================================
# ML MODEL
# ============================================================
def train_model(features: pd.DataFrame, test_seasons: list = None):
    """Train LightGBM model for game prediction."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: pip install lightgbm")
        return None

    feature_cols = [
        'elo_diff', 'elo_home', 'elo_away', 'elo_expected',
        'form_diff_5', 'form_diff_10', 'home_form_5', 'away_form_5',
        'pts_diff', 'home_pts_avg', 'away_pts_avg',
        'rest_diff', 'home_b2b', 'away_b2b',
        'streak_diff',
    ]

    df = features.dropna(subset=feature_cols + ['home_win'])

    if test_seasons:
        train = df[~df['season'].isin(test_seasons)]
        test = df[df['season'].isin(test_seasons)]
    else:
        split = int(len(df) * 0.8)
        train = df.iloc[:split]
        test = df.iloc[split:]

    X_train = train[feature_cols].values
    y_train = train['home_win'].values
    X_test = test[feature_cols].values
    y_test = test['home_win'].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.06,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, verbose=-1,
    )
    model.fit(X_train_s, y_train)

    # Predictions
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)

    # Calibration check
    print(f"\n{'='*60}")
    print(f"  NBA PREDICTION MODEL RESULTS")
    print(f"{'='*60}")
    print(f"  Train: {len(train):,} games | Test: {len(test):,} games")
    print(f"  Test Accuracy: {acc:.1%}")
    print(f"  Home win rate (actual): {y_test.mean():.1%}")
    print(f"  Home win rate (predicted): {y_pred.mean():.1%}")

    # Confidence buckets
    print(f"\n  Confidence-Filtered Accuracy:")
    print(f"  {'Threshold':<12} {'Accuracy':<10} {'Trades':<10} {'WR actual'}")
    print(f"  {'─'*45}")

    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = y_prob > thresh
        if mask.sum() > 10:
            a = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  >{thresh:<11.0%} {a:<10.1%} {mask.sum():<10}")

        mask_low = y_prob < (1 - thresh)
        if mask_low.sum() > 10:
            a = accuracy_score(y_test[mask_low], y_pred[mask_low])
            print(f"  <{1-thresh:<11.0%} {a:<10.1%} {mask_low.sum():<10}")

    # Feature importance
    print(f"\n  Feature Importance:")
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for name, val in imp[:8]:
        print(f"    {name:<20} {val:>6}")

    # Betting simulation
    print(f"\n{'='*60}")
    print(f"  BETTING SIMULATION (vs 50/50 market)")
    print(f"{'='*60}")

    test_df = test.copy()
    test_df['our_prob'] = y_prob

    # Simulate: bet on home when our_prob > 0.55, bet on away when < 0.45
    bet_size = 100
    total_pnl = 0
    bets = 0
    wins = 0

    for _, row in test_df.iterrows():
        prob = row['our_prob']
        actual = row['home_win']

        if prob > 0.58:  # Bet HOME
            bets += 1
            correct = actual == 1
            if correct:
                wins += 1
                total_pnl += bet_size * 0.48  # Polymarket payout
            else:
                total_pnl -= bet_size * 0.52
        elif prob < 0.42:  # Bet AWAY
            bets += 1
            correct = actual == 0
            if correct:
                wins += 1
                total_pnl += bet_size * 0.48
            else:
                total_pnl -= bet_size * 0.52

    if bets > 0:
        wr = wins / bets
        print(f"  Bets placed: {bets}")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Total P&L: ${total_pnl:+,.0f}")
        print(f"  Avg P&L/bet: ${total_pnl/bets:+.2f}")
        print(f"  ROI: {total_pnl/(bets*bet_size*0.52)*100:+.1f}%")

    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'accuracy': acc,
        'test_df': test_df,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-today", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("  SPORTS ML PREDICTION MODEL")
    print("="*60)

    # Load data
    csv_path = DATA_DIR / "nba_games_5seasons.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run data download first.")
        sys.exit(1)

    print(f"\nLoading NBA data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} game rows | {df['SEASON'].nunique()} seasons")

    # Build features
    print("\nBuilding features (ELO, form, rest, streak)...")
    df = build_game_features(df)

    print("Creating matchup features...")
    features = prepare_matchup_features(df)
    print(f"  {len(features):,} games with features")

    # Train model (test on last season)
    print("\nTraining LightGBM model...")
    result = train_model(features, test_seasons=['2024-25'])

    if result:
        print(f"\n{'='*60}")
        print(f"  MODEL READY")
        print(f"  Accuracy: {result['accuracy']:.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
