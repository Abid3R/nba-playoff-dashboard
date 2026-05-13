"""Model utilities for the NBA Playoff Predictor Streamlit app.

This file wraps the matchup model structure from the user's notebook:
- Feature differences between Team A and Team B
- Binary prediction: whether Team A wins the series
- Probability output for both teams

You can replace the training data or load your own saved model later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

FEATURE_COLS: List[str] = [
    "win_pct_diff",
    "seed_diff",
    "ppg_diff",
    "net_rating_diff",
    "has_home_court",
    "higher_seed_win_pct",
    "lower_seed_win_pct",
    "combined_win_pct",
    "team_a_win_pct",
    "team_b_win_pct",
]


@dataclass
class ModelBundle:
    model: Any
    feature_cols: List[str]
    training_accuracy: float
    cv_mean: float
    cv_std: float
    algorithm_name: str


def create_training_data_from_series() -> pd.DataFrame:
    """Create compact historical/synthetic matchup training data.

    This mirrors the notebook structure. Expand this list with more verified
    historical playoff series when you want a stronger production model.
    """

    historical_series = [
        # Format:
        # (season, team_a, team_b, team_a_won, win_pct_a, win_pct_b,
        #  seed_a, seed_b, ppg_a, ppg_b, net_rtg_a, net_rtg_b)
        ("2025-26", "OKC", "PHX", 1, 0.780, 0.549, 1, 8, 118, 110, 8.5, 1.2),
        ("2025-26", "SAS", "POR", 1, 0.756, 0.512, 2, 7, 115, 108, 7.8, 0.5),
        ("2025-26", "LAL", "HOU", 1, 0.646, 0.634, 4, 5, 112, 111, 3.2, 3.0),
        ("2025-26", "MIN", "DEN", 1, 0.598, 0.659, 6, 3, 110, 113, 2.1, 5.5),
        ("2025-26", "DET", "ORL", 1, 0.732, 0.549, 1, 8, 114, 106, 6.8, 1.5),
        ("2025-26", "NYK", "ATL", 1, 0.646, 0.561, 3, 6, 116, 112, 5.2, 2.0),
        ("2025-26", "CLE", "TOR", 1, 0.634, 0.561, 4, 5, 112, 110, 4.5, 1.8),
        ("2025-26", "PHI", "BOS", 1, 0.549, 0.683, 7, 2, 108, 115, 1.0, 6.5),
        ("2025-26", "OKC", "LAL", 1, 0.780, 0.646, 1, 4, 118, 112, 8.5, 3.2),
        ("2025-26", "NYK", "PHI", 1, 0.646, 0.549, 3, 7, 116, 108, 5.2, 1.0),
        ("2024-25", "OKC", "DEN", 1, 0.720, 0.598, 1, 8, 120, 113, 9.1, 2.5),
        ("2024-25", "MIN", "LAC", 1, 0.659, 0.622, 3, 6, 111, 109, 5.2, 3.1),
        ("2024-25", "HOU", "GSW", 1, 0.600, 0.561, 4, 5, 108, 110, 3.0, 1.5),
        ("2024-25", "LAL", "SAS", 1, 0.610, 0.573, 2, 7, 114, 111, 4.5, 2.0),
        ("2024-25", "CLE", "ORL", 1, 0.683, 0.549, 1, 8, 113, 105, 7.2, 1.0),
        ("2024-25", "BOS", "ATL", 1, 0.671, 0.524, 2, 7, 118, 112, 8.5, 0.5),
        ("2024-25", "NYK", "MIL", 1, 0.646, 0.573, 3, 6, 115, 116, 4.8, 2.5),
        ("2024-25", "IND", "DET", 1, 0.573, 0.561, 5, 4, 120, 108, 2.0, 1.5),
        ("2023-24", "BOS", "MIA", 1, 0.780, 0.549, 1, 8, 120, 108, 11.2, 1.0),
        ("2023-24", "NYK", "PHI", 1, 0.610, 0.573, 2, 7, 112, 107, 5.5, 2.8),
        ("2023-24", "CLE", "ORL", 1, 0.598, 0.573, 4, 5, 109, 104, 4.2, 3.0),
        ("2023-24", "IND", "MIL", 1, 0.573, 0.600, 6, 3, 123, 118, 3.5, 1.2),
        ("2023-24", "MIN", "PHX", 1, 0.683, 0.600, 3, 6, 110, 108, 5.8, 2.5),
        ("2023-24", "DAL", "LAC", 1, 0.610, 0.622, 5, 4, 117, 111, 3.2, 4.0),
        ("2023-24", "OKC", "NOP", 1, 0.695, 0.600, 1, 8, 118, 110, 7.5, 3.0),
        ("2023-24", "DEN", "LAL", 1, 0.695, 0.573, 2, 7, 112, 109, 4.1, 1.5),
    ]

    samples = []
    for season, ta, tb, a_won, wp_a, wp_b, seed_a, seed_b, ppg_a, ppg_b, nr_a, nr_b in historical_series:
        samples.append(
            {
                "season": season,
                "team_a": ta,
                "team_b": tb,
                "win_pct_diff": wp_a - wp_b,
                "seed_diff": seed_b - seed_a,
                "ppg_diff": ppg_a - ppg_b,
                "net_rating_diff": nr_a - nr_b,
                "has_home_court": 1 if seed_a < seed_b else 0,
                "higher_seed_win_pct": max(wp_a, wp_b),
                "lower_seed_win_pct": min(wp_a, wp_b),
                "combined_win_pct": wp_a + wp_b,
                "team_a_win_pct": wp_a,
                "team_b_win_pct": wp_b,
                "label": a_won,
            }
        )
        samples.append(
            {
                "season": season,
                "team_a": tb,
                "team_b": ta,
                "win_pct_diff": wp_b - wp_a,
                "seed_diff": seed_a - seed_b,
                "ppg_diff": ppg_b - ppg_a,
                "net_rating_diff": nr_b - nr_a,
                "has_home_court": 1 if seed_b < seed_a else 0,
                "higher_seed_win_pct": max(wp_a, wp_b),
                "lower_seed_win_pct": min(wp_a, wp_b),
                "combined_win_pct": wp_a + wp_b,
                "team_a_win_pct": wp_b,
                "team_b_win_pct": wp_a,
                "label": 1 - a_won,
            }
        )

    return pd.DataFrame(samples)


def train_model() -> ModelBundle:
    """Train the matchup model and return a cached model bundle."""
    df = create_training_data_from_series()
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Deployment-safe version of your matchup model.
    # To use your exact XGBoost pickle later, replace this training block
    # with joblib.load("models/nba_playoff_model.pkl").
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    algorithm_name = "Random Forest Classifier"

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    try:
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        cv_mean = float(np.mean(cv))
        cv_std = float(np.std(cv))
    except Exception:
        cv_mean = float("nan")
        cv_std = float("nan")

    return ModelBundle(
        model=model,
        feature_cols=FEATURE_COLS,
        training_accuracy=accuracy,
        cv_mean=cv_mean,
        cv_std=cv_std,
        algorithm_name=algorithm_name,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_prediction_features(team_a_stats: Dict[str, Any], team_b_stats: Dict[str, Any]) -> Dict[str, float]:
    """Build the exact feature row expected by the model."""
    a_wp = _safe_float(team_a_stats.get("W_PCT"), 0.50)
    b_wp = _safe_float(team_b_stats.get("W_PCT"), 0.50)
    a_seed = int(_safe_float(team_a_stats.get("SEED"), 8))
    b_seed = int(_safe_float(team_b_stats.get("SEED"), 8))
    a_ppg = _safe_float(team_a_stats.get("PTS"), 0.0)
    b_ppg = _safe_float(team_b_stats.get("PTS"), 0.0)
    a_net = _safe_float(team_a_stats.get("NET_RATING"), 0.0)
    b_net = _safe_float(team_b_stats.get("NET_RATING"), 0.0)

    return {
        "win_pct_diff": a_wp - b_wp,
        "seed_diff": b_seed - a_seed,
        "ppg_diff": a_ppg - b_ppg,
        "net_rating_diff": a_net - b_net,
        "has_home_court": 1 if a_seed < b_seed else 0,
        "higher_seed_win_pct": max(a_wp, b_wp),
        "lower_seed_win_pct": min(a_wp, b_wp),
        "combined_win_pct": a_wp + b_wp,
        "team_a_win_pct": a_wp,
        "team_b_win_pct": b_wp,
    }


def predict_series(model_bundle: ModelBundle, team_a_name: str, team_b_name: str,
                   team_a_stats: Dict[str, Any], team_b_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Predict series winner and return probabilities."""
    features = build_prediction_features(team_a_stats, team_b_stats)
    X = pd.DataFrame([features])[model_bundle.feature_cols]
    prob = model_bundle.model.predict_proba(X.values)[0]
    team_a_prob = float(prob[1])
    team_b_prob = float(prob[0])

    return {
        "team_a": team_a_name,
        "team_b": team_b_name,
        "team_a_win_prob": round(team_a_prob * 100, 1),
        "team_b_win_prob": round(team_b_prob * 100, 1),
        "predicted_winner": team_a_name if team_a_prob >= team_b_prob else team_b_name,
        "confidence": round(max(team_a_prob, team_b_prob) * 100, 1),
        "features": features,
    }


def get_feature_importance(model_bundle: ModelBundle) -> pd.DataFrame:
    """Return feature importance if supported by the estimator."""
    model = model_bundle.model
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.ones(len(model_bundle.feature_cols)) / len(model_bundle.feature_cols)

    df = pd.DataFrame({"feature": model_bundle.feature_cols, "importance": importance})
    df["importance"] = df["importance"] / df["importance"].sum()
    return df.sort_values("importance", ascending=False)
