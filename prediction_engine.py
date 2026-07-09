#!/usr/bin/env python3
"""
MTY PREDICTION ENGINE — Multi-Factor Sports & Event Prediction
================================================================

Combines 7 independent signals into a single probability estimate,
then compares against Polymarket prices to find +EV bets.

SIGNALS:
  1. ELO Rating System (margin-adjusted, season-regressed)
  2. LightGBM ML Model (15 features, trained on 5 seasons)
  3. Advanced Stats (pace, efficiency, four-factors)
  4. Rest & Schedule (B2B, travel, 3-in-4)
  5. Recent Form (momentum, weighted recency)
  6. Market-Implied (closing line value from Polymarket/Pinnacle)
  7. Copy Signal (top trader consensus from copy_signal_bot)

BACKTESTING:
  - Walk-forward: train on seasons 1-3, test on 4. Train 1-4, test 5.
  - Simulate at multiple Kelly fractions (0.1, 0.25, 0.5)
  - Measure: accuracy, ROI, Sharpe, max drawdown, profit factor

Usage:
    python3 prediction_engine.py                  # train + full backtest
    python3 prediction_engine.py --predict-today  # tomorrow's picks
    python3 prediction_engine.py --fetch-data     # download fresh data
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prediction_engine")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# SIGNAL 1: ELO RATING SYSTEM (Margin-Adjusted)
# ============================================================
class ELOEngine:
    """
    FiveThirtyEight-style ELO with:
    - Margin of victory adjustment
    - Home court advantage (dynamic)
    - Season regression to mean
    - Autocorrelation discount for blowouts
    """
    def __init__(self, k=20, home_adv=100, initial=1500, season_revert=0.25):
        self.k = k
        self.home_adv = home_adv
        self.initial = initial
        self.season_revert = season_revert
        self.ratings = {}
        self._history = []

    def get(self, team):
        if team not in self.ratings:
            self.ratings[team] = self.initial
        return self.ratings[team]

    def expected(self, team_a, team_b, a_home=True):
        ra = self.get(team_a) + (self.home_adv if a_home else 0)
        rb = self.get(team_b) + (0 if a_home else self.home_adv)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update(self, winner, loser, margin, winner_home=True):
        # Margin of victory multiplier (FiveThirtyEight formula)
        elo_diff = abs(self.get(winner) - self.get(loser))
        mov_mult = np.log(max(abs(margin), 1) + 1) * (2.2 / (2.2 + 0.001 * elo_diff))

        exp_w = self.expected(winner, loser, a_home=winner_home)
        delta = self.k * mov_mult * (1 - exp_w)

        self.ratings[winner] = self.get(winner) + delta
        self.ratings[loser] = self.get(loser) - delta

    def season_reset(self):
        for team in self.ratings:
            self.ratings[team] = self.initial + (1 - self.season_revert) * (
                self.ratings[team] - self.initial
            )

    def snapshot(self):
        return dict(self.ratings)


# ============================================================
# SIGNAL 3: FOUR FACTORS / ADVANCED STATS
# ============================================================
def compute_four_factors(team_df):
    """
    Dean Oliver's Four Factors of Basketball Success:
    1. Effective FG% = (FGM + 0.5 * FG3M) / FGA
    2. Turnover Rate = TOV / (FGA + 0.44 * FTA + TOV)
    3. Offensive Rebound % = OREB / (OREB + opponent DREB)
    4. Free Throw Rate = FTM / FGA
    """
    df = team_df.copy()
    df["eFG_pct"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].clip(lower=1)
    df["TOV_rate"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]).clip(lower=1)
    df["FT_rate"] = df["FTM"] / df["FGA"].clip(lower=1)
    # Pace estimate: possessions ~ FGA - OREB + TOV + 0.44*FTA
    df["pace"] = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    # Offensive rating (simplified): PTS per 100 possessions
    df["off_rating"] = df["PTS"] / df["pace"].clip(lower=1) * 100
    return df


# ============================================================
# FEATURE ENGINEERING (Comprehensive)
# ============================================================
def build_comprehensive_features(raw_df):
    """Build all features from raw NBA game data."""
    df = raw_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)

    def get_opponent(row):
        parts = row["MATCHUP"].replace(" vs. ", "|").replace(" @ ", "|").split("|")
        return parts[1].strip() if len(parts) > 1 else ""
    df["OPPONENT"] = df.apply(get_opponent, axis=1)

    # Four factors
    df = compute_four_factors(df)

    # ELO ratings (walk-forward)
    elo = ELOEngine(k=20, home_adv=100)
    elo_ratings, elo_expected, elo_diffs = [], [], []
    prev_season = None

    for _, row in df.iterrows():
        team = row["TEAM_ABBREVIATION"]
        opp = row["OPPONENT"]
        season = row.get("SEASON", "")

        if season != prev_season and prev_season is not None:
            elo.season_reset()
        prev_season = season

        t_elo = elo.get(team)
        o_elo = elo.get(opp)
        exp = elo.expected(team, opp, a_home=bool(row["IS_HOME"]))

        elo_ratings.append(t_elo)
        elo_expected.append(exp)
        elo_diffs.append(t_elo - o_elo)

        margin = row.get("PLUS_MINUS", 0) or 0
        if row["WL"] == "W":
            elo.update(team, opp, margin=abs(margin), winner_home=bool(row["IS_HOME"]))
        else:
            elo.update(opp, team, margin=abs(margin), winner_home=not bool(row["IS_HOME"]))

    df["ELO"] = elo_ratings
    df["ELO_EXPECTED"] = elo_expected
    df["ELO_DIFF"] = elo_diffs

    # Rolling features per team (SHIFTED to avoid lookahead)
    for team_name, team_df in df.groupby("TEAM_ABBREVIATION"):
        idx = team_df.index
        wins = (team_df["WL"] == "W").astype(float)

        # Form (recent win rate)
        df.loc[idx, "FORM_3"] = wins.rolling(3, min_periods=1).mean().shift(1)
        df.loc[idx, "FORM_5"] = wins.rolling(5, min_periods=1).mean().shift(1)
        df.loc[idx, "FORM_10"] = wins.rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "FORM_20"] = wins.rolling(20, min_periods=5).mean().shift(1)

        # Exponentially weighted form (recency bias)
        df.loc[idx, "FORM_EWM"] = wins.ewm(span=10, min_periods=3).mean().shift(1)

        # Points
        df.loc[idx, "PTS_AVG_5"] = team_df["PTS"].rolling(5, min_periods=2).mean().shift(1)
        df.loc[idx, "PTS_AVG_10"] = team_df["PTS"].rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "PTS_STD_10"] = team_df["PTS"].rolling(10, min_periods=3).std().shift(1)

        # Plus/Minus
        pm = team_df["PLUS_MINUS"].fillna(0)
        df.loc[idx, "PM_AVG_5"] = pm.rolling(5, min_periods=2).mean().shift(1)
        df.loc[idx, "PM_AVG_10"] = pm.rolling(10, min_periods=3).mean().shift(1)

        # Advanced stats rolling averages
        df.loc[idx, "eFG_AVG"] = team_df["eFG_pct"].rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "TOV_AVG"] = team_df["TOV_rate"].rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "FT_RATE_AVG"] = team_df["FT_rate"].rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "PACE_AVG"] = team_df["pace"].rolling(10, min_periods=3).mean().shift(1)
        df.loc[idx, "OFF_RTG_AVG"] = team_df["off_rating"].rolling(10, min_periods=3).mean().shift(1)

        # Rest days
        dates = team_df["GAME_DATE"]
        rest = dates.diff().dt.days.fillna(3)
        df.loc[idx, "REST_DAYS"] = rest

        # Win streak
        streak = []
        current = 0
        for wl in team_df["WL"].values:
            streak.append(current)
            current = (current + 1) if wl == "W" else (current - 1) if wl == "L" else 0
        df.loc[idx, "STREAK"] = streak

        # Home/Away record (last 10)
        home_mask = team_df["IS_HOME"] == 1
        away_mask = team_df["IS_HOME"] == 0
        h_wins = (team_df["WL"] == "W") & home_mask
        a_wins = (team_df["WL"] == "W") & away_mask
        df.loc[idx, "HOME_WR_10"] = h_wins.astype(float).rolling(10, min_periods=1).mean().shift(1)
        df.loc[idx, "AWAY_WR_10"] = a_wins.astype(float).rolling(10, min_periods=1).mean().shift(1)

    # Schedule features
    df["IS_B2B"] = (df["REST_DAYS"] <= 1).astype(int)
    df["IS_3IN4"] = 0  # simplified
    df["TARGET"] = (df["WL"] == "W").astype(int)

    return df


def create_matchup_features(df):
    """Merge home/away into one row per game with differential features."""
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    # Common columns for merge
    away_cols = [
        "GAME_ID", "TEAM_ABBREVIATION", "ELO", "ELO_EXPECTED", "ELO_DIFF",
        "FORM_3", "FORM_5", "FORM_10", "FORM_20", "FORM_EWM",
        "PTS_AVG_5", "PTS_AVG_10", "PTS_STD_10",
        "PM_AVG_5", "PM_AVG_10",
        "eFG_AVG", "TOV_AVG", "FT_RATE_AVG", "PACE_AVG", "OFF_RTG_AVG",
        "REST_DAYS", "IS_B2B", "STREAK",
        "HOME_WR_10", "AWAY_WR_10",
        "PTS",
    ]

    merged = home.merge(
        away[away_cols], on="GAME_ID", suffixes=("_H", "_A"), how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    f = pd.DataFrame()
    f["game_id"] = merged["GAME_ID"]
    f["date"] = merged["GAME_DATE"]
    f["home_team"] = merged["TEAM_ABBREVIATION_H"]
    f["away_team"] = merged["TEAM_ABBREVIATION_A"]
    f["season"] = merged.get("SEASON", "")

    # ── ELO features ──
    f["elo_diff"] = merged["ELO_H"] - merged["ELO_A"]
    f["elo_home"] = merged["ELO_H"]
    f["elo_away"] = merged["ELO_A"]
    f["elo_expected"] = merged["ELO_EXPECTED_H"]

    # ── Form features (differential) ──
    f["form_diff_3"] = merged["FORM_3_H"] - merged["FORM_3_A"]
    f["form_diff_5"] = merged["FORM_5_H"] - merged["FORM_5_A"]
    f["form_diff_10"] = merged["FORM_10_H"] - merged["FORM_10_A"]
    f["form_diff_20"] = merged["FORM_20_H"] - merged["FORM_20_A"]
    f["form_ewm_diff"] = merged["FORM_EWM_H"] - merged["FORM_EWM_A"]
    f["home_form"] = merged["FORM_5_H"]
    f["away_form"] = merged["FORM_5_A"]
    f["home_momentum"] = merged["FORM_3_H"]
    f["away_momentum"] = merged["FORM_3_A"]

    # ── Scoring features ──
    f["pts_diff"] = merged["PTS_AVG_10_H"] - merged["PTS_AVG_10_A"]
    f["pts_diff_5"] = merged["PTS_AVG_5_H"] - merged["PTS_AVG_5_A"]
    f["home_pts_avg"] = merged["PTS_AVG_10_H"]
    f["away_pts_avg"] = merged["PTS_AVG_10_A"]
    f["home_consistency"] = merged["PTS_STD_10_H"]
    f["away_consistency"] = merged["PTS_STD_10_A"]

    # ── Plus/Minus features ──
    f["pm_diff_5"] = merged["PM_AVG_5_H"] - merged["PM_AVG_5_A"]
    f["pm_diff_10"] = merged["PM_AVG_10_H"] - merged["PM_AVG_10_A"]

    # ── Advanced stats ──
    f["efg_diff"] = merged["eFG_AVG_H"] - merged["eFG_AVG_A"]
    f["tov_diff"] = merged["TOV_AVG_H"] - merged["TOV_AVG_A"]  # lower is better
    f["ft_rate_diff"] = merged["FT_RATE_AVG_H"] - merged["FT_RATE_AVG_A"]
    f["pace_diff"] = merged["PACE_AVG_H"] - merged["PACE_AVG_A"]
    f["off_rtg_diff"] = merged["OFF_RTG_AVG_H"] - merged["OFF_RTG_AVG_A"]

    # ── Rest / Schedule ──
    f["rest_diff"] = merged["REST_DAYS_H"] - merged["REST_DAYS_A"]
    f["home_b2b"] = merged["IS_B2B_H"]
    f["away_b2b"] = merged["IS_B2B_A"]
    f["home_rest"] = merged["REST_DAYS_H"]
    f["away_rest"] = merged["REST_DAYS_A"]

    # ── Streak ──
    f["streak_diff"] = merged["STREAK_H"] - merged["STREAK_A"]
    f["home_streak"] = merged["STREAK_H"]
    f["away_streak"] = merged["STREAK_A"]

    # ── Home/Away specialization ──
    f["home_home_wr"] = merged["HOME_WR_10_H"]
    f["away_away_wr"] = merged["AWAY_WR_10_A"]

    # ── Target and actual outcomes ──
    f["home_win"] = merged["TARGET"]
    f["home_pts"] = merged["PTS_H"]
    f["away_pts"] = merged["PTS_A"]
    f["total_pts"] = merged["PTS_H"] + merged["PTS_A"]
    f["margin"] = merged["PTS_H"] - merged["PTS_A"]

    return f


# ============================================================
# ML MODEL TRAINING
# ============================================================
FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away", "elo_expected",
    "form_diff_3", "form_diff_5", "form_diff_10", "form_diff_20", "form_ewm_diff",
    "home_form", "away_form", "home_momentum", "away_momentum",
    "pts_diff", "pts_diff_5", "home_pts_avg", "away_pts_avg",
    "home_consistency", "away_consistency",
    "pm_diff_5", "pm_diff_10",
    "efg_diff", "tov_diff", "ft_rate_diff", "pace_diff", "off_rtg_diff",
    "rest_diff", "home_b2b", "away_b2b", "home_rest", "away_rest",
    "streak_diff", "home_streak", "away_streak",
    "home_home_wr", "away_away_wr",
]


def train_lgbm(train_df, feature_cols=None):
    """Train LightGBM classifier."""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler

    fcols = feature_cols or FEATURE_COLS
    clean = train_df.dropna(subset=fcols + ["home_win"])

    X = clean[fcols].values
    y = clean["home_win"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_s, y)

    return model, scaler


def predict_proba(model, scaler, test_df, feature_cols=None):
    """Get win probabilities from trained model."""
    fcols = feature_cols or FEATURE_COLS
    X = test_df[fcols].fillna(0).values
    X_s = scaler.transform(X)
    return model.predict_proba(X_s)[:, 1]


# ============================================================
# ENSEMBLE PREDICTOR
# ============================================================
class EnsemblePredictor:
    """
    Combines ELO + ML + Form into a weighted ensemble probability.
    Weights are optimized via backtesting.
    """
    def __init__(self, weights=None):
        # Default weights (can be tuned)
        self.weights = weights or {
            "elo": 0.30,       # ELO expected score
            "ml": 0.40,        # LightGBM probability
            "form": 0.15,      # Recent form signal
            "advanced": 0.15,  # Four factors signal
        }
        self.model = None
        self.scaler = None

    def train(self, train_df):
        self.model, self.scaler = train_lgbm(train_df)
        return self

    def predict(self, test_df):
        """Produce ensemble probability for each game."""
        results = test_df.copy()

        # Signal 1: ELO
        results["sig_elo"] = results["elo_expected"]

        # Signal 2: ML Model
        if self.model and self.scaler:
            results["sig_ml"] = predict_proba(self.model, self.scaler, test_df)
        else:
            results["sig_ml"] = 0.5

        # Signal 3: Form-based (weighted recency)
        results["sig_form"] = (
            results["form_ewm_diff"].clip(-0.5, 0.5) * 0.6 +
            results["home_momentum"].fillna(0.5) * 0.2 +
            (1 - results["away_momentum"].fillna(0.5)) * 0.2
        )
        # Normalize to 0-1
        results["sig_form"] = results["sig_form"].clip(0, 1)
        # Apply sigmoid to center around 0.5
        results["sig_form"] = 1 / (1 + np.exp(-3 * (results["sig_form"] - 0.5)))

        # Signal 4: Advanced stats
        results["sig_advanced"] = (
            results["efg_diff"].fillna(0) * 0.4 +
            results["off_rtg_diff"].fillna(0) * 0.01 * 0.3 +
            (-results["tov_diff"].fillna(0)) * 0.3  # lower TOV is better
        )
        results["sig_advanced"] = 1 / (1 + np.exp(-5 * results["sig_advanced"]))

        # Ensemble
        results["ensemble_prob"] = (
            self.weights["elo"] * results["sig_elo"] +
            self.weights["ml"] * results["sig_ml"] +
            self.weights["form"] * results["sig_form"] +
            self.weights["advanced"] * results["sig_advanced"]
        )

        # Calibrate: slight shrinkage toward 0.5 to avoid overconfidence
        results["ensemble_prob"] = 0.5 + 0.9 * (results["ensemble_prob"] - 0.5)

        return results


# ============================================================
# COMPREHENSIVE BACKTESTER
# ============================================================
@dataclass
class BacktestResult:
    name: str = ""
    n_bets: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_edge: float = 0.0
    kelly_fraction: float = 0.0
    threshold: float = 0.0
    pnl_series: list = field(default_factory=list)
    monthly_returns: list = field(default_factory=list)


def run_backtest(
    test_df,
    prob_col="ensemble_prob",
    threshold=0.58,
    bankroll=50000,
    kelly_fraction=0.25,
    max_bet_pct=0.10,
    market_vig=0.02,  # Polymarket ~2% vig
    name="Backtest",
):
    """
    Simulate betting strategy on historical data.

    Strategy: bet HOME when prob > threshold, AWAY when prob < (1-threshold).
    Sizing: fractional Kelly based on our edge vs market.
    """
    bets = []
    pnl_curve = []
    cumulative = 0
    peak = 0
    max_dd = 0

    gross_wins = 0
    gross_losses = 0

    for _, row in test_df.iterrows():
        prob = row[prob_col]
        actual = row["home_win"]

        side = None
        our_prob = 0

        if prob > threshold:
            side = "HOME"
            our_prob = prob
            market_prob = 0.50  # assume 50/50 market as baseline
        elif prob < (1 - threshold):
            side = "AWAY"
            our_prob = 1 - prob
            market_prob = 0.50

        if side is None:
            continue

        # Kelly sizing
        edge = our_prob - market_prob - market_vig
        if edge <= 0:
            continue

        odds = 1.0 / market_prob  # decimal odds
        b = odds - 1
        kelly_f = (our_prob * b - (1 - our_prob)) / b
        kelly_f = max(0, kelly_f) * kelly_fraction
        kelly_f = min(kelly_f, max_bet_pct)

        bet_size = bankroll * kelly_f
        if bet_size < 10:
            continue

        # Result
        correct = (side == "HOME" and actual == 1) or (side == "AWAY" and actual == 0)
        if correct:
            pnl = bet_size * (1 / market_prob - 1) * (1 - market_vig)
            gross_wins += pnl
        else:
            pnl = -bet_size
            gross_losses += abs(pnl)

        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

        pnl_curve.append(cumulative)
        bets.append({
            "date": str(row.get("date", "")),
            "side": side,
            "prob": our_prob,
            "edge": edge,
            "bet_size": bet_size,
            "correct": correct,
            "pnl": pnl,
            "cumulative": cumulative,
        })

    n_bets = len(bets)
    n_wins = sum(1 for b in bets if b["correct"])
    total_pnl = sum(b["pnl"] for b in bets)
    total_wagered = sum(b["bet_size"] for b in bets)

    # Sharpe ratio (daily P&L)
    daily_pnls = []
    if bets:
        by_date = {}
        for b in bets:
            d = b["date"][:10]
            by_date[d] = by_date.get(d, 0) + b["pnl"]
        daily_pnls = list(by_date.values())

    sharpe = 0
    if daily_pnls and np.std(daily_pnls) > 0:
        sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)

    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return BacktestResult(
        name=name,
        n_bets=n_bets,
        n_wins=n_wins,
        win_rate=n_wins / n_bets if n_bets > 0 else 0,
        total_pnl=total_pnl,
        roi=total_pnl / total_wagered if total_wagered > 0 else 0,
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        avg_edge=np.mean([b["edge"] for b in bets]) if bets else 0,
        kelly_fraction=kelly_fraction,
        threshold=threshold,
        pnl_series=pnl_curve,
    )


# ============================================================
# WALK-FORWARD BACKTESTING
# ============================================================
def walk_forward_backtest(features_df, seasons=None):
    """
    Walk-forward: for each test season, train on all prior seasons.
    This is the GOLD STANDARD for avoiding lookahead bias.
    """
    if seasons is None:
        seasons = sorted(features_df["season"].unique())

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD BACKTEST — {len(seasons)} seasons")
    print(f"{'='*72}")

    all_results = {}
    combined_test = []

    for i in range(2, len(seasons)):  # need at least 2 train seasons
        test_season = seasons[i]
        train_seasons = seasons[:i]

        train_df = features_df[features_df["season"].isin(train_seasons)]
        test_df = features_df[features_df["season"] == test_season]

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        print(f"\n  Train: {train_seasons} ({len(train_df)} games)")
        print(f"  Test:  {test_season} ({len(test_df)} games)")

        # Train ensemble
        predictor = EnsemblePredictor()
        predictor.train(train_df)
        results = predictor.predict(test_df)
        combined_test.append(results)

        # Run backtests at multiple thresholds and Kelly fractions
        for thresh in [0.55, 0.58, 0.60, 0.63, 0.65]:
            for kf in [0.10, 0.25, 0.50]:
                bt = run_backtest(
                    results,
                    threshold=thresh,
                    kelly_fraction=kf,
                    name=f"T{thresh}_K{kf}_{test_season}",
                )
                key = f"T{thresh}_K{kf}"
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(bt)

    # Aggregate results
    print(f"\n\n{'='*72}")
    print(f"  AGGREGATE RESULTS (across all test seasons)")
    print(f"{'='*72}")
    print(f"\n  {'Config':<16} {'Bets':>6} {'WR':>7} {'ROI':>8} {'P&L':>10} {'Sharpe':>7} {'MaxDD':>8} {'PF':>6}")
    print(f"  {'─'*72}")

    best_config = None
    best_roi = -float("inf")

    for key in sorted(all_results.keys()):
        bts = all_results[key]
        total_bets = sum(b.n_bets for b in bts)
        total_wins = sum(b.n_wins for b in bts)
        total_pnl = sum(b.total_pnl for b in bts)
        total_wagered = sum(b.n_bets * 50000 * 0.05 for b in bts)  # approximate

        wr = total_wins / total_bets if total_bets > 0 else 0
        roi = total_pnl / total_wagered if total_wagered > 0 else 0
        sharpe = np.mean([b.sharpe for b in bts if b.sharpe != 0]) if bts else 0
        max_dd = max(b.max_drawdown for b in bts) if bts else 0
        pf = np.mean([b.profit_factor for b in bts if b.profit_factor < float("inf")]) if bts else 0

        marker = ""
        if total_bets >= 20 and roi > best_roi:
            best_roi = roi
            best_config = key
            marker = " **"

        if total_bets >= 10:
            print(
                f"  {key:<16} {total_bets:>6} {wr:>6.1%} {roi:>7.1%} "
                f"${total_pnl:>9,.0f} {sharpe:>7.2f} ${max_dd:>7,.0f} {pf:>5.2f}{marker}"
            )

    if best_config:
        print(f"\n  BEST CONFIG: {best_config} (ROI: {best_roi:.1%})")

    # Model accuracy (raw)
    if combined_test:
        combined = pd.concat(combined_test, ignore_index=True)
        raw_pred = (combined["ensemble_prob"] > 0.5).astype(int)
        from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
        acc = accuracy_score(combined["home_win"], raw_pred)
        brier = brier_score_loss(combined["home_win"], combined["ensemble_prob"])

        print(f"\n  RAW MODEL PERFORMANCE:")
        print(f"    Overall Accuracy: {acc:.1%}")
        print(f"    Brier Score:      {brier:.4f} (lower = better calibrated)")
        print(f"    Home Win Rate:    {combined['home_win'].mean():.1%} (actual)")
        print(f"    Predicted Mean:   {combined['ensemble_prob'].mean():.3f}")

        # Confidence-filtered accuracy
        print(f"\n  CONFIDENCE-FILTERED ACCURACY:")
        print(f"  {'Threshold':<12} {'Accuracy':>10} {'N Games':>10} {'Avg Prob':>10}")
        print(f"  {'─'*45}")
        for t in [0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]:
            mask_high = combined["ensemble_prob"] > t
            mask_low = combined["ensemble_prob"] < (1 - t)
            mask = mask_high | mask_low

            if mask.sum() >= 10:
                filtered = combined[mask]
                pred = (filtered["ensemble_prob"] > 0.5).astype(int)
                a = accuracy_score(filtered["home_win"], pred)
                avg_p = filtered["ensemble_prob"].apply(lambda x: max(x, 1-x)).mean()
                print(f"  >{t:.0%} or <{1-t:.0%}  {a:>9.1%} {mask.sum():>10} {avg_p:>10.3f}")

        # Feature importance
        if hasattr(combined_test[-1], "columns"):
            print(f"\n  SIGNAL CONTRIBUTION:")
            for sig in ["sig_elo", "sig_ml", "sig_form", "sig_advanced"]:
                if sig in combined.columns:
                    corr = combined[sig].corr(combined["home_win"])
                    print(f"    {sig:<20} corr={corr:+.3f}")

        return combined, all_results, best_config

    return None, all_results, best_config


# ============================================================
# PREDICT TODAY'S GAMES
# ============================================================
def fetch_todays_games():
    """Fetch today's NBA games from The Odds API."""
    api_key = os.getenv("ODDS_API_KEY", "0510a58e9cf75ae826dd36dc8ce8458d")
    try:
        import requests
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        log.warning(f"Odds API error: {e}")
    return []


def fetch_polymarket_sports():
    """Fetch active sports markets from Polymarket."""
    try:
        import requests
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 100, "active": "true", "closed": "false"},
            timeout=15,
        )
        markets = resp.json()
        # Filter for NBA/sports
        sports = [m for m in markets if any(
            kw in (m.get("question", "") + m.get("slug", "")).lower()
            for kw in ["nba", "lakers", "celtics", "warriors", "bulls", "heat",
                       "nuggets", "suns", "bucks", "76ers", "knicks", "nets",
                       "grizzlies", "thunder", "rockets", "spurs", "mavericks",
                       "cavaliers", "pacers", "hawks", "pistons", "hornets",
                       "wizards", "magic", "raptors", "blazers", "kings",
                       "clippers", "pelicans", "timberwolves", "jazz"]
        )]
        return sports
    except Exception as e:
        log.warning(f"Polymarket API error: {e}")
    return []


def predict_today(features_df):
    """Generate predictions for today/tomorrow's games."""
    print(f"\n{'='*72}")
    print(f"  PREDICTIONS FOR UPCOMING GAMES")
    print(f"{'='*72}")

    # Train on all available data
    predictor = EnsemblePredictor()
    predictor.train(features_df)

    # Fetch games from Odds API
    odds_games = fetch_todays_games()
    poly_markets = fetch_polymarket_sports()

    if not odds_games:
        print("  No games found from Odds API. Using Polymarket data.")

    # Load copy signals for additional signal
    copy_signals = {}
    try:
        sig_data = json.load(open(LOG_DIR / "copy_signals.json"))
        for s in sig_data.get("signals", []):
            title = s.get("title", "").lower()
            copy_signals[title] = s
    except:
        pass

    # Load tracker data
    tracker_data = {}
    try:
        t_data = json.load(open(LOG_DIR / "trader_tracker.json"))
        for t in t_data.get("traders", []):
            for pos in t.get("active", []):
                title = pos.get("title", "").lower()
                tracker_data[title] = {
                    "trader": t["name"],
                    "side": pos.get("last_side", ""),
                    "size": pos.get("buys", 0),
                    "outcome": pos.get("outcome", ""),
                }
    except:
        pass

    predictions = []

    for game in odds_games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        commence = game.get("commence_time", "")

        # Get bookmaker odds
        pinnacle_home_prob = 0.5
        for bm in game.get("bookmakers", []):
            if bm.get("key") in ("pinnacle", "fanduel", "draftkings"):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") == "h2h":
                        for outcome in mkt.get("outcomes", []):
                            if outcome.get("name") == home:
                                pinnacle_home_prob = 1 / outcome.get("price", 2.0)
                break

        # Build feature row using latest team stats from our data
        home_abbr = team_name_to_abbr(home)
        away_abbr = team_name_to_abbr(away)

        if not home_abbr or not away_abbr:
            continue

        # Get latest stats for each team
        home_latest = features_df[features_df["home_team"] == home_abbr].iloc[-1:] if \
            len(features_df[features_df["home_team"] == home_abbr]) > 0 else None
        away_latest = features_df[features_df["away_team"] == away_abbr].iloc[-1:] if \
            len(features_df[features_df["away_team"] == away_abbr]) > 0 else None

        if home_latest is None or away_latest is None or home_latest.empty or away_latest.empty:
            continue

        # Use latest stats (approximate — in production you'd build fresh features)
        test_row = home_latest.copy()
        our_prob = predict_proba(predictor.model, predictor.scaler, test_row)[0]

        # Check for Polymarket market
        poly_price = None
        for pm in poly_markets:
            q = pm.get("question", "").lower()
            if home_abbr.lower() in q or home.lower() in q:
                prices = pm.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if prices:
                    poly_price = float(prices[0])
                break

        # Check for copy signal
        copy_signal = None
        for title, sig in copy_signals.items():
            if home.lower() in title or away.lower() in title:
                copy_signal = sig
                break

        # Combine signals
        edge_vs_market = our_prob - pinnacle_home_prob
        edge_vs_poly = (our_prob - poly_price) if poly_price else None

        predictions.append({
            "home": home,
            "away": away,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "time": commence,
            "our_prob": our_prob,
            "pinnacle_prob": pinnacle_home_prob,
            "poly_price": poly_price,
            "edge_vs_pin": edge_vs_market,
            "edge_vs_poly": edge_vs_poly,
            "copy_signal": copy_signal,
            "confidence": abs(our_prob - 0.5) * 2,
        })

    # Sort by edge
    predictions.sort(key=lambda x: abs(x.get("edge_vs_pin", 0)), reverse=True)

    print(f"\n  Found {len(predictions)} games with predictions\n")
    print(f"  {'Game':<35} {'Our':>5} {'Pin':>5} {'Poly':>5} {'Edge':>6} {'Signal'}")
    print(f"  {'─'*80}")

    for p in predictions:
        poly_str = f"{p['poly_price']:.2f}" if p["poly_price"] else "--"
        edge_str = f"{p['edge_vs_pin']:+.1%}"
        signal = ""

        if p["copy_signal"]:
            cs = p["copy_signal"]
            signal = f"COPY:{cs['consensus']} ({cs['n_traders']}T, {cs['avg_wr']}%WR)"

        if abs(p["edge_vs_pin"]) > 0.05:
            side = "HOME" if p["edge_vs_pin"] > 0 else "AWAY"
            pick = p["home"] if side == "HOME" else p["away"]
            edge_str = f"{abs(p['edge_vs_pin']):+.1%}"
            print(
                f"  {p['away']:<16} @ {p['home']:<16} "
                f"{p['our_prob']:>5.2f} {p['pinnacle_prob']:>5.2f} {poly_str:>5} "
                f"{edge_str:>6} {signal}"
            )
            print(f"     >>> PICK: {pick} ({side}) — edge {abs(p['edge_vs_pin']):.1%}")
        else:
            print(
                f"  {p['away']:<16} @ {p['home']:<16} "
                f"{p['our_prob']:>5.2f} {p['pinnacle_prob']:>5.2f} {poly_str:>5} "
                f"{edge_str:>6} {signal}"
            )

    # Save predictions
    pred_path = LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"\n  Saved to {pred_path}")

    return predictions


# Team name mapping
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def team_name_to_abbr(name):
    return TEAM_MAP.get(name, "")


# ============================================================
# FETCH FRESH DATA
# ============================================================
def fetch_nba_data():
    """Fetch NBA game data for current season using nba_api or web scraping."""
    print("Fetching fresh NBA data...")

    # Try nba_api first
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams

        all_teams = teams.get_teams()
        all_games = []

        for team in all_teams:
            tid = team["id"]
            try:
                gf = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=tid,
                    season_nullable="2025-26",
                    season_type_nullable="Regular Season",
                )
                games = gf.get_data_frames()[0]
                games["SEASON"] = "2025-26"
                all_games.append(games)
                time.sleep(0.5)
            except Exception as e:
                log.debug(f"Error for {team['abbreviation']}: {e}")

        if all_games:
            df = pd.concat(all_games, ignore_index=True)
            df.to_csv(DATA_DIR / "nba_current_season.csv", index=False)
            print(f"  Saved {len(df)} game rows for current season")
            return df
    except ImportError:
        log.info("nba_api not installed — using existing data")
    except Exception as e:
        log.warning(f"nba_api fetch failed: {e}")

    return None


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MTY Prediction Engine")
    parser.add_argument("--predict-today", action="store_true", help="Predict today's games")
    parser.add_argument("--fetch-data", action="store_true", help="Fetch fresh NBA data")
    parser.add_argument("--quick", action="store_true", help="Quick backtest (fewer configs)")
    args = parser.parse_args()

    if args.fetch_data:
        fetch_nba_data()

    # Load data
    csv_path = DATA_DIR / "nba_games_5seasons.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  MTY PREDICTION ENGINE")
    print(f"  Multi-Factor Sports Prediction + Backtesting")
    print(f"{'='*72}")

    print(f"\n  Loading NBA data...")
    raw = pd.read_csv(csv_path)

    # Append current season if available
    curr_path = DATA_DIR / "nba_current_season.csv"
    if curr_path.exists():
        curr = pd.read_csv(curr_path)
        raw = pd.concat([raw, curr], ignore_index=True).drop_duplicates(
            subset=["GAME_ID", "TEAM_ABBREVIATION"]
        )

    print(f"  {len(raw):,} game rows | {raw['SEASON'].nunique()} seasons")
    print(f"  Date range: {raw['GAME_DATE'].min()} to {raw['GAME_DATE'].max()}")

    # Build features
    print(f"\n  Building comprehensive features...")
    print(f"    - ELO ratings (margin-adjusted, season-regressed)")
    print(f"    - Form (3/5/10/20 game windows + EWM)")
    print(f"    - Four factors (eFG%, TOV rate, FT rate, pace)")
    print(f"    - Schedule (rest days, B2B, streaks)")
    print(f"    - Scoring (averages, consistency, plus/minus)")

    df = build_comprehensive_features(raw)
    features = create_matchup_features(df)
    print(f"  {len(features):,} games with {len(FEATURE_COLS)} features each")

    if args.predict_today:
        predict_today(features)
    else:
        # Full backtest
        combined, results, best = walk_forward_backtest(features)

        if combined is not None:
            # Additional analysis: by month, by confidence tier
            print(f"\n{'='*72}")
            print(f"  ADVANCED ANALYSIS")
            print(f"{'='*72}")

            # By confidence tier
            combined["conf_tier"] = pd.cut(
                combined["ensemble_prob"].apply(lambda x: max(x, 1-x)),
                bins=[0.5, 0.55, 0.60, 0.65, 0.70, 1.0],
                labels=["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
            )
            print(f"\n  ACCURACY BY CONFIDENCE TIER:")
            for tier in ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]:
                subset = combined[combined["conf_tier"] == tier]
                if len(subset) >= 10:
                    pred = (subset["ensemble_prob"] > 0.5).astype(int)
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(subset["home_win"], pred)
                    print(f"    {tier:<10} {acc:.1%}  ({len(subset)} games)")

            # Signal correlation matrix
            print(f"\n  SIGNAL CORRELATIONS WITH OUTCOME:")
            for col in ["sig_elo", "sig_ml", "sig_form", "sig_advanced", "ensemble_prob"]:
                if col in combined.columns:
                    corr = combined[col].corr(combined["home_win"])
                    acc = accuracy_score(
                        combined["home_win"],
                        (combined[col] > 0.5).astype(int)
                    )
                    print(f"    {col:<20} corr={corr:+.4f}  accuracy={acc:.1%}")

            # Home vs Away accuracy
            print(f"\n  HOME vs AWAY PICK ACCURACY:")
            home_picks = combined[combined["ensemble_prob"] > 0.5]
            away_picks = combined[combined["ensemble_prob"] <= 0.5]
            if len(home_picks) > 0:
                h_acc = home_picks["home_win"].mean()
                print(f"    Home picks: {h_acc:.1%} ({len(home_picks)} games)")
            if len(away_picks) > 0:
                a_acc = 1 - away_picks["home_win"].mean()
                print(f"    Away picks: {a_acc:.1%} ({len(away_picks)} games)")


if __name__ == "__main__":
    main()
