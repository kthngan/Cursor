#!/usr/bin/env python3
"""Calibrate an additive adjustment on top of de-vig odds probability.

Model:
    adjustment = standardized_factors · beta   # probability space, uncapped
    adjusted_prob_p1 = devig_prob_p1 + adjustment
    adjusted_prob_p2 = devig_prob_p2 - adjustment

Calibration minimizes combined Brier loss on both P1 and P2 targets.
Training uses match-equal row weights; primary evaluation uses one row
per match (earliest odds snapshot) to avoid match-end label repetition bias.

Compare:
  - Baseline: de-vig Pinnacle implied probability (no adjustment)
  - Adjusted P1/P2: devig ± factor adjustment
  - Factors-only: logistic on factors alone (reference)
"""

from __future__ import annotations

import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import (  # noqa: E402
    DATA_DIR,
    METADATA_PATH,
    METRIC_COLUMNS,
    REPORTS_DIR,
    SCORE_FEATURES,
    add_score_features,
    auc_score,
    evaluate_predictions,
    fmt,
    fmt_pct,
    html_table,
    logit,
    sigmoid,
    split_events,
)
from train_xgboost_factor_models import (  # noqa: E402
    add_factor_features,
    fill_matrix,
    shap_bar_chart,
    shap_rows,
    shap_summary,
    train_xgboost,
)

OUTPUT_PATH = REPORTS_DIR / "adjusted_probability_report.html"
TARGET_COL = "target_p1_win"
WITH_ODDS_DIR = DATA_DIR / "with_pinnacle_odds"
LOGIT_ADJUSTMENT_CAP = 0.06  # legacy logit-offset helpers only
MIN_TRADE_GAP_SECONDS = 60  # minimum seconds between trades in the same match
BACKTEST_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
PRIMARY_BACKTEST_THRESHOLD = 0.15  # pre-specified decision threshold
SEGMENT_BACKTEST_THRESHOLD = PRIMARY_BACKTEST_THRESHOLD
PROB_EPS = 1e-6
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "match_winner", "target_", "winner_side", "game_winner", "future",
)


# ---------------------------------------------------------------------------
# Probability-space adjustment model (dual-side calibration)
# ---------------------------------------------------------------------------

@dataclass
class ProbabilityAdjustmentModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray


def _standardize_features(x_values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x_values), mean, x_values)
    return (filled - mean) / std


def compute_adjustment(
    model: ProbabilityAdjustmentModel,
    x_values: np.ndarray,
) -> np.ndarray:
    z = _standardize_features(x_values, model.mean, model.std)
    return z @ model.weights


def predict_adjusted_probs(
    model: ProbabilityAdjustmentModel,
    x_values: np.ndarray,
    devig_p1: np.ndarray,
    devig_p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjustment = compute_adjustment(model, x_values)
    adjusted_p1 = np.clip(devig_p1 + adjustment, PROB_EPS, 1.0 - PROB_EPS)
    adjusted_p2 = np.clip(devig_p2 - adjustment, PROB_EPS, 1.0 - PROB_EPS)
    return adjusted_p1, adjusted_p2, adjustment


def fit_probability_adjustment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    devig_p1: np.ndarray,
    devig_p2: np.ndarray,
    *,
    l2: float = 0.02,
    learning_rate: float = 0.06,
    epochs: int = 500,
    sample_weight: np.ndarray | None = None,
) -> ProbabilityAdjustmentModel:
    """Fit adjustment using combined Brier loss on P1 and P2 probabilities."""
    mean_values = np.nanmean(x_train, axis=0)
    mean_values = np.where(np.isfinite(mean_values), mean_values, 0.0)
    filled = np.where(np.isnan(x_train), mean_values, x_train)
    std_values = filled.std(axis=0)
    std_values = np.where(std_values > 1e-8, std_values, 1.0)
    z_features = (filled - mean_values) / std_values

    row_weight = np.ones(len(y_train), dtype=float) if sample_weight is None else sample_weight.astype(float)
    row_weight = row_weight / max(row_weight.sum(), 1e-12) * len(row_weight)

    weights = np.zeros(z_features.shape[1])
    y_p2 = 1.0 - y_train
    for _ in range(epochs):
        adjustment = z_features @ weights
        pred_p1 = np.clip(devig_p1 + adjustment, PROB_EPS, 1.0 - PROB_EPS)
        pred_p2 = np.clip(devig_p2 - adjustment, PROB_EPS, 1.0 - PROB_EPS)
        residual = (pred_p1 - y_train) - (pred_p2 - y_p2)
        grad = 2.0 * (z_features.T @ (row_weight * residual)) / row_weight.sum()
        grad += l2 * weights
        weights -= learning_rate * grad

    return ProbabilityAdjustmentModel(mean_values, std_values, weights)


def probability_adjustment_stats(
    model: ProbabilityAdjustmentModel,
    train: pd.DataFrame,
    devig_p1_col: str,
    devig_p2_col: str,
    target_col: str,
    features: list[str],
) -> list[dict[str, Any]]:
    """Compute t-stats for the dual-side probability adjustment model."""
    x_raw = fill_matrix(train, train, features)
    z = _standardize_features(x_raw, model.mean, model.std)
    devig_p1 = train[devig_p1_col].to_numpy(dtype=float)
    devig_p2 = train[devig_p2_col].to_numpy(dtype=float)
    adjustment = compute_adjustment(model, x_raw)
    pred_p1 = np.clip(devig_p1 + adjustment, PROB_EPS, 1.0 - PROB_EPS)
    hessian = 2.0 * (z.T @ z)
    try:
        cov = np.linalg.pinv(hessian + 1e-6 * np.eye(z.shape[1]))
        std_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        std_errors = np.full(z.shape[1], np.nan)

    rows = []
    for idx, feature in enumerate(features):
        coef = float(model.weights[idx])
        se = float(std_errors[idx]) if math.isfinite(float(std_errors[idx])) else None
        z_stat = coef / se if se and se > 0 else None
        p_value = float(2.0 * norm.sf(abs(z_stat))) if z_stat is not None else None
        rows.append({
            "feature": feature,
            "coefficient": coef,
            "std_error": se,
            "z_stat": z_stat,
            "p_value": p_value,
        })
    _ = pred_p1  # computed for potential future diagnostics
    return rows


def evaluate_dual_predictions(
    y_p1: np.ndarray,
    adjusted_p1: np.ndarray,
    adjusted_p2: np.ndarray,
) -> dict[str, float | None]:
    y_p2 = 1.0 - y_p1
    p1_metrics = evaluate_predictions(y_p1, adjusted_p1)
    p2_metrics = evaluate_predictions(y_p2, adjusted_p2)
    combined_brier = float(np.mean((adjusted_p1 - y_p1) ** 2 + (adjusted_p2 - y_p2) ** 2))
    return {
        "rows": float(len(y_p1)),
        "auc": p1_metrics["auc"],
        "accuracy": p1_metrics["accuracy"],
        "brier": combined_brier,
        "brier_p1": p1_metrics["brier"],
        "brier_p2": p2_metrics["brier"],
        "rmse": p1_metrics["rmse"],
        "log_loss": p1_metrics["log_loss"],
        "auc_p2": p2_metrics["auc"],
    }


def split_train_validation(
    train: pd.DataFrame,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest training events for XGBoost early stopping."""
    events = (
        train[["event_id", "event_start_date"]]
        .drop_duplicates("event_id")
        .sort_values(["event_start_date", "event_id"], na_position="last")
    )
    split_at = max(1, int(len(events) * (1.0 - val_frac)))
    val_events = set(events.iloc[split_at:]["event_id"])
    val = train.loc[train["event_id"].isin(val_events)].copy()
    fit = train.loc[~train["event_id"].isin(val_events)].copy()
    return fit, val


def chronological_sort_cols(df: pd.DataFrame) -> list[str]:
    cols = ["event_start_date", "event_id"]
    if "ut" in df.columns:
        cols.append("ut")
    if "seq" in df.columns:
        cols.append("seq")
    return cols


def primary_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One evaluation row per match: earliest chronological snapshot with odds."""
    sorted_df = df.sort_values(chronological_sort_cols(df), na_position="last")
    return sorted_df.groupby("event_id", as_index=False).head(1)


def match_equal_row_weights(df: pd.DataFrame) -> np.ndarray:
    """Weight rows so each match contributes equally to training loss."""
    counts = df.groupby("event_id")["event_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


MIN_VAL_BETS_FOR_THRESHOLD = 30


def dual_adjusted_probs_from_p1(
    p1_pred: np.ndarray,
    devig_p1: np.ndarray,
    devig_p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a P1 win probability into dual-side adjusted probabilities."""
    adjusted_p1 = np.clip(p1_pred, PROB_EPS, 1.0 - PROB_EPS)
    adjustment = adjusted_p1 - devig_p1
    adjusted_p2 = np.clip(devig_p2 - adjustment, PROB_EPS, 1.0 - PROB_EPS)
    return adjusted_p1, adjusted_p2, adjustment


def train_xgboost_adjustment(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[Any, np.ndarray]:
    """Train XGBoost on selected factors plus de-vig odds features."""
    booster, p1_pred = train_xgboost(train, validation, test, TARGET_COL, features)
    return booster, p1_pred


def assert_no_forward_odds(df: pd.DataFrame) -> None:
    """Reject rows where odds snapshots are timestamped after the row."""
    if "odds_age_seconds" not in df.columns:
        return
    forward = int((pd.to_numeric(df["odds_age_seconds"], errors="coerce") < 0).sum())
    if forward:
        raise ValueError(f"Found {forward:,} rows with forward-looking odds (age < 0)")


def assert_no_feature_leakage(features: list[str]) -> None:
    leaked = [
        feature for feature in features
        if any(token in feature.lower() for token in FORBIDDEN_FEATURE_SUBSTRINGS)
    ]
    if leaked:
        raise ValueError(f"Forbidden feature names in model set: {leaked}")


def max_drawdown(cumulative: np.ndarray) -> float:
    if len(cumulative) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative)
    return float(np.min(cumulative - peak))


def profit_factor(pnl: np.ndarray) -> float | None:
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl < 0].sum()))
    if losses <= 0:
        return None
    return wins / losses


def calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int | str]]:
    """Reliability bins for probability calibration checks."""
    pred = np.clip(y_pred, PROB_EPS, 1.0 - PROB_EPS)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int | str]] = []
    for idx in range(n_bins):
        low, high = edges[idx], edges[idx + 1]
        if idx == n_bins - 1:
            mask = (pred >= low) & (pred <= high)
        else:
            mask = (pred >= low) & (pred < high)
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(y_true[mask].mean())
        expected = float(pred[mask].mean())
        rows.append({
            "label": f"{low:.1f}-{high:.1f}",
            "count": count,
            "expected": expected,
            "observed": observed,
            "gap": observed - expected,
        })
    return rows


# ---------------------------------------------------------------------------
# Legacy logit-offset helpers (factors-only reference)
# ---------------------------------------------------------------------------

@dataclass
class OffsetLogisticModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray  # factor coefficients only (no intercept)


def fit_logistic_with_offset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    offset: np.ndarray,
    *,
    l2: float = 0.02,
    learning_rate: float = 0.06,
    epochs: int = 500,
    cap: float = LOGIT_ADJUSTMENT_CAP,
) -> OffsetLogisticModel:
    """Fit logistic regression where z = offset + clip(X @ beta, -cap, cap).

    The adjustment (X @ beta) is hard-capped at ±cap in logit space during
    training. Samples hitting the cap have zero gradient (no learning beyond cap).
    """
    mean_values = np.nanmean(x_train, axis=0)
    mean_values = np.where(np.isfinite(mean_values), mean_values, 0.0)
    filled = np.where(np.isnan(x_train), mean_values, x_train)
    std_values = filled.std(axis=0)
    std_values = np.where(std_values > 1e-8, std_values, 1.0)
    z_features = (filled - mean_values) / std_values

    weights = np.zeros(z_features.shape[1])
    for _ in range(epochs):
        adjustment = z_features @ weights
        clamped = np.clip(adjustment, -cap, cap)
        at_cap = np.abs(adjustment) >= cap
        z = offset + clamped
        pred = sigmoid(z)
        residual = pred - y_train
        residual[at_cap] = 0.0  # no gradient for samples at the cap
        grad = (z_features.T @ residual) / len(y_train)
        grad += l2 * weights
        weights -= learning_rate * grad

    return OffsetLogisticModel(mean_values, std_values, weights)


def predict_with_offset(
    model: OffsetLogisticModel,
    x_values: np.ndarray,
    offset: np.ndarray,
    cap: float = LOGIT_ADJUSTMENT_CAP,
) -> np.ndarray:
    filled = np.where(np.isnan(x_values), model.mean, x_values)
    z = (filled - model.mean) / model.std
    adjustment = np.clip(z @ model.weights, -cap, cap)
    return sigmoid(offset + adjustment)


def offset_logistic_stats(
    model: OffsetLogisticModel,
    train: pd.DataFrame,
    offset_col: str,
    target_col: str,
    features: list[str],
) -> list[dict[str, Any]]:
    """Compute t-stats and p-values for the offset logistic regression."""
    x_raw = fill_matrix(train, train, features)
    z = (x_raw - model.mean) / model.std
    offset = train[offset_col].to_numpy(dtype=float)
    adjustment = np.clip(z @ model.weights, -LOGIT_ADJUSTMENT_CAP, LOGIT_ADJUSTMENT_CAP)
    pred = sigmoid(offset + adjustment)
    weight = np.clip(pred * (1.0 - pred), 1e-8, None)
    hessian = (z.T * weight) @ z
    try:
        cov = np.linalg.pinv(hessian + 1e-6 * np.eye(z.shape[1]))
        std_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        std_errors = np.full(z.shape[1], np.nan)

    rows = []
    for idx, feature in enumerate(features):
        coef = float(model.weights[idx])
        se = float(std_errors[idx]) if math.isfinite(float(std_errors[idx])) else None
        z_stat = coef / se if se and se > 0 else None
        p_value = float(2.0 * norm.sf(abs(z_stat))) if z_stat is not None else None
        rows.append({"feature": feature, "coefficient": coef, "std_error": se,
                      "z_stat": z_stat, "p_value": p_value})
    return rows


def select_top_factors_per_group(
    stats: list[dict[str, Any]],
    factor_groups: dict[str, list[str]],
    max_per_group: int = 2,
) -> list[str]:
    """Select top features by |z-stat| from each factor group."""
    z_by_feature = {s["feature"]: abs(s["z_stat"] or 0) for s in stats}
    selected = []
    for group, features in factor_groups.items():
        ranked = sorted(features, key=lambda f: z_by_feature.get(f, 0), reverse=True)
        selected.extend(ranked[:max_per_group])
    return list(dict.fromkeys(selected))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_enriched_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    if df.empty or "rolling_live_form_ratio" not in df.columns:
        return None
    winner = df["match_winner_side"].replace("", np.nan).dropna()
    if winner.empty or winner.iloc[-1] not in {"P1", "P2"}:
        return None
    df["target_p1_win"] = 1.0 if winner.iloc[-1] == "P1" else 0.0
    df["event_id"] = str(int(float(df["event_id"].iloc[0])))
    df["source_file"] = path.name
    for column in METRIC_COLUMNS:
        if column == "live_form_delta_5":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["live_form_delta_5"] = df["rolling_live_form_ratio"] - df["rolling_live_form_ratio"].shift(5)
    df = add_score_features(df)

    for col in ["odds_no_vig_p1", "odds_no_vig_p2", "odds_implied_p1", "odds_implied_p2",
                "odds_p1_price", "odds_p2_price", "odds_bookmaker_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ut" in df.columns and "odds_snapshot_time" in df.columns:
        df["row_dt"] = pd.to_datetime(df["ut"], unit="s", utc=True, errors="coerce")
        df["snap_dt"] = pd.to_datetime(df["odds_snapshot_time"], utc=True, errors="coerce")
        df["odds_age_seconds"] = (df["row_dt"] - df["snap_dt"]).dt.total_seconds()
    else:
        df["odds_age_seconds"] = np.nan

    df["has_odds"] = df["odds_no_vig_p1"].notna().astype(float)

    usable = df[METRIC_COLUMNS + SCORE_FEATURES].notna().any(axis=1)
    return df.loc[usable].copy()


def load_enriched_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(WITH_ODDS_DIR.glob("*.csv")):
        if path.name.startswith("_"):
            continue
        frame = read_enriched_csv(path)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No usable CSVs found in {WITH_ODDS_DIR}")
    df = pd.concat(frames, ignore_index=True)

    if METADATA_PATH.exists():
        meta = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
        keep_cols = [
            "event_id", "competition_name", "season_name", "stage_name",
            "group_name", "event_start_date", "event_event_stats_lvl_live",
            "event_round_name",
        ]
        available = [c for c in keep_cols if c in meta.columns]
        df = df.merge(meta[available].drop_duplicates("event_id"), on="event_id", how="left")

    df["league"] = df.get("competition_name", "Unknown").fillna("Unknown").astype(str)
    df["tier"] = df.get("event_event_stats_lvl_live", "Unknown").fillna("Unknown").astype(str)
    df["event_start_date"] = pd.to_datetime(df.get("event_start_date", ""), errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_frames(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_events, validation_events, test_events = split_events(df)
    work = df.loc[df[target_col].notna()].copy()
    train = work.loc[work["event_id"].isin(train_events)].copy()
    validation = work.loc[work["event_id"].isin(validation_events)].copy()
    test = work.loc[work["event_id"].isin(test_events)].copy()
    return train, validation, test


# ---------------------------------------------------------------------------
# Staleness bins for segment analysis
# ---------------------------------------------------------------------------

STALENESS_BINS = [0, 60, 150, 300, 450, 600]  # seconds: 0-1, 1-2.5, 2.5-5, 5-7.5, 7.5-10 min
STALENESS_LABELS = ["0-1m", "1-2.5m", "2.5-5m", "5-7.5m", "7.5-10m"]


def _append_trade_stats(row: dict[str, Any], bets: pd.DataFrame) -> dict[str, Any]:
    if bets.empty:
        row["pnl"] = 0.0
        row["turnover"] = 0.0
        row["rot"] = None
        row["n_bets"] = 0
    else:
        turnover = float(bets["bet_cost"].sum())
        pnl = float(bets["pnl"].sum())
        row["turnover"] = turnover
        row["pnl"] = pnl
        row["rot"] = pnl / turnover if turnover > 0 else None
        row["n_bets"] = int(len(bets))
    return row


def segment_analysis(
    df: pd.DataFrame,
    factors_only_pred: np.ndarray | None = None,
    *,
    min_rows: int = 100,
) -> list[dict[str, Any]]:
    work = df.copy()
    work["pred_baseline"] = pd.to_numeric(work["devig_p1"], errors="coerce")
    work["pred_adjusted"] = pd.to_numeric(work["adjusted_prob_p1"], errors="coerce")
    if factors_only_pred is not None:
        work["pred_factors_only"] = factors_only_pred

    results = []
    bets_all = work.loc[work["side"] != ""] if "side" in work.columns else work.iloc[0:0]

    # By staleness (minutes: 0-1, 1-2.5, 2.5-5, 5-7.5, 7.5-10)
    stale = work.loc[work["odds_age_seconds"].notna() & (work["odds_age_seconds"] >= 0)].copy()
    stale["staleness_bin"] = pd.cut(
        stale["odds_age_seconds"],
        bins=STALENESS_BINS,
        labels=STALENESS_LABELS,
        right=False,
    )
    for label, group in stale.groupby("staleness_bin", observed=True):
        y = group[TARGET_COL].to_numpy(dtype=float)
        if len(y) < min_rows:
            continue
        row = {"segment": f"Staleness: {label}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            if col not in group.columns:
                continue
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
            row[f"{name}_log_loss"] = m["log_loss"]
        seg_bets = bets_all.loc[bets_all.index.isin(group.index)]
        _append_trade_stats(row, seg_bets)
        results.append(row)

    # By match progression
    for label, mask in [("Early (sets_total<=1)", work["sets_total"] <= 1),
                        ("Mid (sets_total==2)", work["sets_total"] == 2),
                        ("Late (sets_total>=3)", work["sets_total"] >= 3)]:
        group = work.loc[mask]
        if len(group) < min_rows:
            continue
        y = group[TARGET_COL].to_numpy(dtype=float)
        row = {"segment": f"Progression: {label}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            if col not in group.columns:
                continue
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
            row[f"{name}_log_loss"] = m["log_loss"]
        seg_bets = bets_all.loc[bets_all.index.isin(group.index)]
        _append_trade_stats(row, seg_bets)
        results.append(row)

    # By league
    for league, group in work.groupby("league", dropna=False):
        if len(group) < 500:
            continue
        y = group[TARGET_COL].to_numpy(dtype=float)
        row = {"segment": f"League: {league}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            if col not in group.columns:
                continue
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
        seg_bets = bets_all.loc[bets_all.index.isin(group.index)]
        _append_trade_stats(row, seg_bets)
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _add_next_odds_execution_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Map each row to execution prices from the next odds update in the same match."""
    out = df.copy()
    n = len(out)
    next_raw_p1 = np.full(n, np.nan)
    next_raw_p2 = np.full(n, np.nan)
    next_pinnacle_p1 = np.full(n, np.nan)
    next_pinnacle_p2 = np.full(n, np.nan)
    has_next = np.zeros(n, dtype=bool)

    p1 = pd.to_numeric(out.get("odds_p1_price"), errors="coerce").to_numpy(dtype=float)
    p2 = pd.to_numeric(out.get("odds_p2_price"), errors="coerce").to_numpy(dtype=float)
    snap = out.get("odds_snapshot_time", pd.Series([""] * n)).astype(str).to_numpy()

    for _, idx in out.groupby("event_id", sort=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            continue

        local_snap = snap[idx]
        local_p1 = p1[idx]
        local_p2 = p2[idx]
        change_points = [0]
        for pos in range(1, len(idx)):
            if (
                local_snap[pos] != local_snap[pos - 1]
                or local_p1[pos] != local_p1[pos - 1]
                or local_p2[pos] != local_p2[pos - 1]
            ):
                change_points.append(pos)
        change_points.append(len(idx))

        for seg in range(len(change_points) - 1):
            seg_start = change_points[seg]
            seg_end = change_points[seg + 1]
            if seg + 1 >= len(change_points) - 1:
                continue
            exec_pos = change_points[seg + 1]
            exec_global = idx[exec_pos]
            exec_raw_p1 = 1.0 / local_p1[exec_pos] if local_p1[exec_pos] > 0 else np.nan
            exec_raw_p2 = 1.0 / local_p2[exec_pos] if local_p2[exec_pos] > 0 else np.nan
            for pos in range(seg_start, seg_end):
                global_i = idx[pos]
                next_raw_p1[global_i] = exec_raw_p1
                next_raw_p2[global_i] = exec_raw_p2
                next_pinnacle_p1[global_i] = local_p1[exec_pos]
                next_pinnacle_p2[global_i] = local_p2[exec_pos]
                has_next[global_i] = np.isfinite(exec_raw_p1) and np.isfinite(exec_raw_p2)

    out["signal_raw_implied_p1"] = out["raw_implied_p1"] if "raw_implied_p1" in out.columns else np.nan
    out["signal_raw_implied_p2"] = out["raw_implied_p2"] if "raw_implied_p2" in out.columns else np.nan
    out["next_raw_implied_p1"] = next_raw_p1
    out["next_raw_implied_p2"] = next_raw_p2
    out["next_pinnacle_p1"] = next_pinnacle_p1
    out["next_pinnacle_p2"] = next_pinnacle_p2
    out["has_next_odds_update"] = has_next
    out["trade_price_p1"] = next_raw_p1
    out["trade_price_p2"] = next_raw_p2
    return out


def _build_match_cumulative_pnl(bets: pd.DataFrame) -> np.ndarray:
    """Aggregate bet PnL by match (chronological), then cumulative sum."""
    if bets.empty:
        return np.array([])
    event_order = (
        bets.groupby("event_id", as_index=False)
        .agg(event_start_date=("event_start_date", "first"))
        .sort_values(["event_start_date", "event_id"], na_position="last")
    )
    pnl_by_event = bets.groupby("event_id")["pnl"].sum()
    ordered_pnl = event_order["event_id"].map(pnl_by_event).fillna(0.0).to_numpy(dtype=float)
    return np.cumsum(ordered_pnl)


def run_backtest(
    test: pd.DataFrame,
    adjusted_p1: np.ndarray,
    adjusted_p2: np.ndarray,
    threshold: float = 0.01,
    min_trade_gap_seconds: int = MIN_TRADE_GAP_SECONDS,
) -> dict[str, Any]:
    """Backtest symmetric two-sided strategy with next-odds-update execution.

    - Signals use current-row adjusted vs raw implied probabilities
    - Execution price uses raw implied from the next odds update in the same match
    - Buy P1 when adjusted_prob_p1 > raw_implied_p1 + threshold
    - Buy P2 when adjusted_prob_p2 > raw_implied_p2 + threshold
    - If both fire, take the side with the larger edge
    - Skip trades with no later odds update in the match
    - At most one trade per match every min_trade_gap_seconds (default 1 minute)
    - Each bet is 1 unit (cost = trade price, payout = 1 if win)
    """
    df = test.copy()
    if "odds_no_vig_p1" in df.columns:
        df["devig_p1"] = pd.to_numeric(df["odds_no_vig_p1"], errors="coerce")
    else:
        df["devig_p1"] = np.nan
    if "odds_no_vig_p2" in df.columns:
        df["devig_p2"] = pd.to_numeric(df["odds_no_vig_p2"], errors="coerce").fillna(1.0 - df["devig_p1"])
    else:
        df["devig_p2"] = 1.0 - df["devig_p1"]
    df["adjusted_prob_p1"] = adjusted_p1
    df["adjusted_prob_p2"] = adjusted_p2

    raw_p1 = pd.to_numeric(df.get("odds_implied_p1"), errors="coerce")
    raw_p2 = pd.to_numeric(df.get("odds_implied_p2"), errors="coerce")
    if raw_p1.isna().all() or raw_p2.isna().all():
        raw_p1 = 1.0 / pd.to_numeric(df["odds_p1_price"], errors="coerce")
        raw_p2 = 1.0 / pd.to_numeric(df["odds_p2_price"], errors="coerce")
    df["raw_implied_p1"] = raw_p1
    df["raw_implied_p2"] = raw_p2

    sort_cols = ["event_start_date", "event_id"]
    if "ut" in df.columns:
        sort_cols.append("ut")
    if "seq" in df.columns:
        sort_cols.append("seq")
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    df = _add_next_odds_execution_prices(df)

    p1_edge = df["adjusted_prob_p1"] - df["raw_implied_p1"] - threshold
    p2_edge = df["adjusted_prob_p2"] - df["raw_implied_p2"] - threshold
    p1_signal = p1_edge > 0
    p2_signal = p2_edge > 0
    side_candidate = np.where(
        p1_signal & p2_signal,
        np.where(p1_edge >= p2_edge, "P1", "P2"),
        np.where(p1_signal, "P1", np.where(p2_signal, "P2", "")),
    )

    side: list[str] = []
    last_trade_ut: dict[str, float] = {}
    for i in range(len(df)):
        candidate = side_candidate[i]
        if candidate == "" or not bool(df.at[i, "has_next_odds_update"]):
            side.append("")
            continue
        if "ut" not in df.columns:
            side.append(candidate)
            continue
        event_id = str(df.at[i, "event_id"])
        row_ut = float(df.at[i, "ut"])
        prev_ut = last_trade_ut.get(event_id)
        if prev_ut is not None and (row_ut - prev_ut) < min_trade_gap_seconds:
            side.append("")
            continue
        side.append(candidate)
        last_trade_ut[event_id] = row_ut

    df["side"] = side
    p1_trade = df["side"] == "P1"
    p2_trade = df["side"] == "P2"
    df["bet_cost"] = np.where(
        p1_trade,
        df["trade_price_p1"],
        np.where(p2_trade, df["trade_price_p2"], 0.0),
    )
    p2_wins = 1.0 - df[TARGET_COL]
    df["payout"] = np.where(p1_trade, df[TARGET_COL], np.where(p2_trade, p2_wins, 0.0))
    df["pnl"] = df["payout"] - df["bet_cost"]

    bets = df.loc[df["side"] != ""].copy()
    p1_bets = df.loc[p1_trade]
    p2_bets = df.loc[p2_trade]
    total_wagered = float(bets["bet_cost"].sum()) if len(bets) else 0.0
    total_pnl = float(bets["pnl"].sum()) if len(bets) else 0.0
    if len(bets) > 0:
        win_rate = float((bets["payout"] > 0).mean())
    else:
        win_rate = 0.0
    roi = total_pnl / total_wagered if total_wagered > 0 else 0.0

    edges: list[float] = []
    if len(p1_bets) > 0:
        edges.extend((p1_bets["adjusted_prob_p1"] - p1_bets["trade_price_p1"]).tolist())
    if len(p2_bets) > 0:
        edges.extend((p2_bets["adjusted_prob_p2"] - p2_bets["trade_price_p2"]).tolist())
    avg_edge = float(np.mean(edges)) if edges else 0.0

    match_cum_pnl = _build_match_cumulative_pnl(bets)
    bet_pnl = bets["pnl"].to_numpy(dtype=float) if len(bets) else np.array([])

    return {
        "df": df,
        "n_bets": int(len(bets)),
        "n_p1_bets": int(len(p1_bets)),
        "n_p2_bets": int(len(p2_bets)),
        "n_rows": len(df),
        "n_matches": int(bets["event_id"].nunique()) if len(bets) else 0,
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "roi": roi,
        "avg_edge": avg_edge,
        "max_drawdown": max_drawdown(match_cum_pnl),
        "profit_factor": profit_factor(bet_pnl),
        "cum_pnl_series": match_cum_pnl,
        "cum_wagered_series": np.array([]),
    }


def select_validation_threshold(
    val_df: pd.DataFrame,
    adjusted_p1: np.ndarray,
    adjusted_p2: np.ndarray,
    thresholds: list[float],
    *,
    fallback: float = PRIMARY_BACKTEST_THRESHOLD,
    min_bets: int = MIN_VAL_BETS_FOR_THRESHOLD,
) -> tuple[float, dict[float, dict[str, Any]]]:
    """Pick the threshold with best validation ROI subject to a minimum bet count."""
    best_thresh = fallback
    best_roi = -np.inf
    val_results: dict[float, dict[str, Any]] = {}
    for thresh in thresholds:
        bt = run_backtest(val_df, adjusted_p1, adjusted_p2, threshold=thresh)
        val_results[thresh] = bt
        if bt["n_bets"] >= min_bets and bt["roi"] > best_roi:
            best_roi = float(bt["roi"])
            best_thresh = thresh
    return best_thresh, val_results


def pnl_svg_chart(backtest: dict[str, Any]) -> str:
    """Generate an inline SVG cumulative PnL chart."""
    cum_pnl = backtest["cum_pnl_series"]
    n = len(cum_pnl)
    if n == 0:
        return "<p>No data to plot.</p>"

    width = 900
    height = 360
    margin_l = 70
    margin_r = 30
    margin_t = 30
    margin_b = 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    y_min = float(min(cum_pnl.min(), 0))
    y_max = float(max(cum_pnl.max(), 0))
    if y_max == y_min:
        y_max = y_min + 1
    y_range = y_max - y_min
    y_pad = y_range * 0.1
    y_min -= y_pad
    y_max += y_pad

    def x_pos(i):
        return margin_l + (i / max(n - 1, 1)) * plot_w

    def y_pos(v):
        return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Build path
    points = []
    for i in range(0, n, max(1, n // 500)):
        points.append(f"{x_pos(i):.1f},{y_pos(cum_pnl[i]):.1f}")
    if (n - 1) % max(1, n // 500) != 0:
        points.append(f"{x_pos(n-1):.1f},{y_pos(cum_pnl[-1]):.1f}")

    path_d = "M " + " L ".join(points)

    # Zero line
    zero_y = y_pos(0)

    # Y-axis grid lines and labels
    n_ticks = 6
    grid_lines = ""
    y_tick_labels = ""
    for t in range(n_ticks + 1):
        val = y_min + t / n_ticks * (y_max - y_min)
        gy = y_pos(val)
        grid_lines += f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + plot_w}" y2="{gy:.1f}" stroke="#e1e4e8" stroke-width="1"/>'
        y_tick_labels += f'<text x="{margin_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" fill="#57606a">{val:+.2f}</text>'

    # X-axis labels
    x_labels = ""
    for t in range(5):
        idx = int(t / 4 * (n - 1))
        xp = x_pos(idx)
        x_labels += f'<text x="{xp:.1f}" y="{margin_t + plot_h + 20}" text-anchor="middle" font-size="11" fill="#57606a">{idx}</text>'

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="border:1px solid #d0d7de;border-radius:8px">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  {grid_lines}
  <line x1="{margin_l}" y1="{zero_y:.1f}" x2="{margin_l + plot_w}" y2="{zero_y:.1f}" stroke="#cf222e" stroke-width="1.5" stroke-dasharray="5,3"/>
  <path d="{path_d}" fill="none" stroke="#0969da" stroke-width="2"/>
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#d0d7de" stroke-width="1.5"/>
  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#d0d7de" stroke-width="1.5"/>
  {y_tick_labels}
  {x_labels}
  <text x="{margin_l + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#57606a">Match index (chronological)</text>
  <text x="20" y="{margin_t + plot_h / 2:.0f}" text-anchor="middle" font-size="12" fill="#57606a" transform="rotate(-90 20 {margin_t + plot_h / 2:.0f})">Cumulative PnL</text>
  <text x="{margin_l + plot_w - 5}" y="{y_pos(cum_pnl[-1]) - 8:.1f}" text-anchor="end" font-size="11" fill="#0969da" font-weight="bold">Final: {cum_pnl[-1]:+.2f}</text>
</svg>"""


def multi_threshold_pnl_svg(backtests: dict[float, dict[str, Any]]) -> str:
    """Generate an inline SVG chart with cumulative PnL curves for multiple thresholds."""
    width = 900
    height = 420
    margin_l = 70
    margin_r = 140
    margin_t = 30
    margin_b = 50
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    colors = ["#0969da", "#1a7f37", "#bf3989", "#d1242f", "#8250df",
              "#fb8f44", "#1f6feb", "#0550ae", "#2da44e", "#6f42c1"]

    # Find global y range
    all_vals = []
    for bt in backtests.values():
        all_vals.extend(bt["cum_pnl_series"].tolist())
    if not all_vals:
        return "<p>No data to plot.</p>"
    y_min = min(min(all_vals), 0)
    y_max = max(max(all_vals), 0)
    if y_max == y_min:
        y_max = y_min + 1
    y_pad = (y_max - y_min) * 0.1
    y_min -= y_pad
    y_max += y_pad

    # Find max length
    max_n = max(len(bt["cum_pnl_series"]) for bt in backtests.values())

    def x_pos(i):
        return margin_l + (i / max(max_n - 1, 1)) * plot_w

    def y_pos(v):
        return margin_t + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Grid lines
    n_ticks = 6
    grid_lines = ""
    y_tick_labels = ""
    for t in range(n_ticks + 1):
        val = y_min + t / n_ticks * (y_max - y_min)
        gy = y_pos(val)
        grid_lines += f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + plot_w}" y2="{gy:.1f}" stroke="#e1e4e8" stroke-width="1"/>'
        y_tick_labels += f'<text x="{margin_l - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" fill="#57606a">{val:+.1f}</text>'

    zero_y = y_pos(0)

    # X-axis labels
    x_labels = ""
    for t in range(5):
        idx = int(t / 4 * (max_n - 1))
        xp = x_pos(idx)
        x_labels += f'<text x="{xp:.1f}" y="{margin_t + plot_h + 20}" text-anchor="middle" font-size="11" fill="#57606a">{idx}</text>'

    # Build paths
    paths = ""
    legend = ""
    for idx, (thresh, bt) in enumerate(sorted(backtests.items())):
        cum_pnl = bt["cum_pnl_series"]
        n = len(cum_pnl)
        color = colors[idx % len(colors)]
        step = max(1, n // 400)
        points = []
        for i in range(0, n, step):
            points.append(f"{x_pos(i):.1f},{y_pos(cum_pnl[i]):.1f}")
        if (n - 1) % step != 0:
            points.append(f"{x_pos(n-1):.1f},{y_pos(cum_pnl[-1]):.1f}")
        path_d = "M " + " L ".join(points)
        paths += f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.85"/>'
        ly = margin_t + idx * 22
        roi = bt["roi"]
        legend += (f'<rect x="{margin_l + plot_w + 15}" y="{ly}" width="12" height="12" fill="{color}"/>'
                   f'<text x="{margin_l + plot_w + 32}" y="{ly + 11}" font-size="11" fill="#1f2328">'
                   f'{thresh:.3f} (ROI {roi:+.1%}, PnL {bt["total_pnl"]:+.0f})</text>')

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="border:1px solid #d0d7de;border-radius:8px">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  {grid_lines}
  <line x1="{margin_l}" y1="{zero_y:.1f}" x2="{margin_l + plot_w}" y2="{zero_y:.1f}" stroke="#cf222e" stroke-width="1.5" stroke-dasharray="5,3"/>
  {paths}
  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#d0d7de" stroke-width="1.5"/>
  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#d0d7de" stroke-width="1.5"/>
  {y_tick_labels}
  {x_labels}
  <text x="{margin_l + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#57606a">Match index (chronological)</text>
  <text x="20" y="{margin_t + plot_h / 2:.0f}" text-anchor="middle" font-size="12" fill="#57606a" transform="rotate(-90 20 {margin_t + plot_h / 2:.0f})">Cumulative PnL</text>
  {legend}
</svg>"""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    df: pd.DataFrame,
    factor_features: list[str],
    factor_groups: dict[str, list[str]],
    overall: dict[str, dict[str, float | None]],
    stats: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    fresh_subset: dict[str, dict[str, float | None]],
    diff_stats: dict[str, float],
    diff_bins: list[dict[str, Any]],
    backtest: dict[str, Any],
    split_info: dict[str, Any],
    backtests: dict[float, dict[str, Any]] | None = None,
    adjustment_comparison: dict[str, Any] | None = None,
    xgb_shap: list[dict[str, float | str]] | None = None,
    xgb_features: list[str] | None = None,
    model_card: dict[str, Any] | None = None,
    calibration: dict[str, list[dict[str, float | int | str]]] | None = None,
    overall_row_level: dict[str, dict[str, float | None]] | None = None,
) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    h3 { margin-top: 20px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; position: sticky; top: 0; }
    .note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .warn { background: #fff3e0; border: 1px solid #ffcc02; border-radius: 8px; padding: 12px; }
    .delta-pos { color: #1a7f37; font-weight: 600; }
    .delta-neg { color: #cf222e; font-weight: 600; }
    .formula { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; font-family: monospace; font-size: 14px; margin: 12px 0; }
    .tab-bar { display: flex; gap: 4px; border-bottom: 1px solid #d0d7de; margin: 24px 0 0; }
    .tab-btn { padding: 10px 20px; border: 1px solid #d0d7de; border-bottom: none; background: #f6f8fa; cursor: pointer; border-radius: 8px 8px 0 0; font-size: 14px; }
    .tab-btn.active { background: #fff; font-weight: 700; margin-bottom: -1px; }
    .tab-panel { display: none; padding-top: 16px; }
    .tab-panel.active { display: block; }
    .bar-row { display: grid; grid-template-columns: 280px 1fr 70px; gap: 10px; align-items: center; margin: 6px 0; font-size: 13px; }
    .bar-track { height: 12px; background: #f6f8fa; border: 1px solid #d0d7de; }
    .bar-fill { height: 100%; background: #0969da; }
    .bar-label { overflow-wrap: anywhere; }
    """

    # Overall comparison table
    model_names = ["baseline", "adjusted", "factors_only"]
    model_labels = {
        "baseline": "De-vig odds (no adjustment)",
        "adjusted": "Adjusted (odds + factor adjustment)",
        "factors_only": "Factors only (no odds prior)",
    }

    overall_rows = []
    for name in model_names:
        m = overall[name]
        overall_rows.append([
            model_labels[name],
            f"{int(m.get('rows', 0)):,}",
            fmt(m.get("auc")),
            fmt_pct(m.get("accuracy")),
            fmt(m.get("brier")),
            fmt(m.get("rmse")),
            fmt(m.get("log_loss")),
        ])

    # Delta table
    delta_rows = []
    for metric in ["auc", "brier", "log_loss"]:
        b = overall["baseline"].get(metric)
        a = overall["adjusted"].get(metric)
        f = overall["factors_only"].get(metric)
        if b is not None and a is not None:
            delta = a - b
            if metric == "auc":
                better = delta > 0
            else:
                better = delta < 0
            cls = "delta-pos" if better else "delta-neg"
            delta_str = f"<span class='{cls}'>{delta:+.4f}</span>"
        else:
            delta_str = ""
        delta_rows.append([
            metric.upper(),
            fmt(b), fmt(f), fmt(a), delta_str,
        ])

    # Fresh subset (0-1 min) comparison
    fresh_rows = []
    for name in model_names:
        m = fresh_subset.get(name, {})
        if m:
            fresh_rows.append([
                model_labels[name],
                f"{int(m.get('rows', 0)):,}",
                fmt(m.get("auc")),
                fmt(m.get("brier")),
                fmt(m.get("log_loss")),
            ])

    # Significant factors (p < 0.05)
    sig_stats = [s for s in stats if s["p_value"] is not None and s["p_value"] < 0.05]
    sig_stats.sort(key=lambda x: abs(x["z_stat"] or 0), reverse=True)

    # Segment table
    seg_rows = []
    for r in segments:
        row = [r["segment"], f"{r['rows']:,}"]
        for name in model_names:
            row.append(fmt(r.get(f"{name}_auc")))
        for name in ["baseline", "adjusted"]:
            row.append(fmt(r.get(f"{name}_brier")))
        row.extend([
            f"{int(r.get('n_bets', 0)):,}",
            fmt(r.get("pnl"), 2) if r.get("pnl") is not None else "",
            fmt(r.get("turnover"), 2) if r.get("turnover") is not None else "",
            fmt_pct(r.get("rot")) if r.get("rot") is not None else "",
        ])
        seg_rows.append(row)

    # Factor group rows
    factor_group_rows = []
    for group, vals in factor_groups.items():
        included = [f for f in vals if f in factor_features]
        factor_group_rows.append([group, str(len(included)), ", ".join(included)])

    # Precompute all HTML tables
    delta_html = html_table(
        ["Metric", "Baseline (odds)", "Factors only", "Adjusted (odds+factors)", "Delta (Adjusted - Baseline)"],
        delta_rows,
    )
    overall_html = html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss"], overall_rows)

    row_level_html = ""
    if overall_row_level:
        row_rows = []
        for name in model_names:
            m = overall_row_level[name]
            row_rows.append([
                model_labels[name],
                f"{int(m.get('rows', 0)):,}",
                fmt(m.get("auc")),
                fmt_pct(m.get("accuracy")),
                fmt(m.get("brier")),
                fmt(m.get("rmse")),
                fmt(m.get("log_loss")),
            ])
        row_level_html = html_table(
            ["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss"],
            row_rows,
        )

    fresh_html = html_table(["Model", "Rows", "AUC", "Brier", "Log loss"], fresh_rows)
    seg_html = html_table(
        ["Segment", "Rows", "Base AUC", "Adj AUC", "Fact AUC", "Base Brier", "Adj Brier",
         "Bets", "PnL", "Turnover", "ROT"],
        seg_rows,
    )

    xgb_fit_html = ""
    xgb_delta_html = ""
    xgb_thresh_html = ""
    xgb_thresh_chart = ""
    xgb_shap_html = ""
    xgb_shap_table_html = ""
    if adjustment_comparison:
        baseline_m = adjustment_comparison.get("baseline", {})
        xgb_m = adjustment_comparison.get("xgboost", {})
        xgb_fit_rows = []
        for label, m in [
            ("Baseline (de-vig)", baseline_m),
            ("XGBoost adjustment", xgb_m),
        ]:
            xgb_fit_rows.append([
                label,
                f"{int(m.get('rows', 0)):,}",
                fmt(m.get("auc")),
                fmt(m.get("brier")),
                fmt(m.get("brier_p1")),
                fmt(m.get("brier_p2")),
                fmt(m.get("log_loss")),
            ])
        xgb_fit_html = html_table(
            ["Model", "Rows", "AUC", "Combined Brier", "Brier P1", "Brier P2", "Log loss"],
            xgb_fit_rows,
        )

        xgb_delta_rows = []
        for metric in ["auc", "brier", "log_loss"]:
            b = baseline_m.get(metric)
            a = xgb_m.get(metric)
            if b is not None and a is not None:
                delta = a - b
                better = delta > 0 if metric == "auc" else delta < 0
                cls = "delta-pos" if better else "delta-neg"
                delta_str = f"<span class='{cls}'>{delta:+.4f}</span>"
            else:
                delta_str = ""
            xgb_delta_rows.append([metric.upper(), fmt(b), fmt(a), delta_str])
        xgb_delta_html = html_table(
            ["Metric", "Baseline (odds)", "XGBoost adjusted", "Delta (XGB - Baseline)"],
            xgb_delta_rows,
        )

        xgb_bts = adjustment_comparison.get("backtests", {}).get("xgboost", {})
        if xgb_bts:
            xgb_thresh_rows = []
            for thresh in sorted(xgb_bts.keys()):
                bt = xgb_bts[thresh]
                roi_cls = "delta-pos" if bt["roi"] > 0 else "delta-neg"
                xgb_thresh_rows.append([
                    f"{thresh:g}",
                    f"{bt['n_bets']:,}",
                    f"{bt.get('n_p1_bets', 0):,}",
                    f"{bt.get('n_p2_bets', 0):,}",
                    f"{bt['total_wagered']:.1f}",
                    f"{bt['total_pnl']:+.2f}",
                    f"{bt['win_rate']:.1%}",
                    f"<span class='{roi_cls}'>{bt['roi']:+.2%}</span>",
                    f"{bt['avg_edge']:.4f}",
                ])
            xgb_thresh_html = html_table(
                ["Threshold", "Bets", "P1 bets", "P2 bets", "Wagered", "Total PnL", "Win rate", "ROI", "Avg edge"],
                xgb_thresh_rows,
            )
            xgb_thresh_chart = multi_threshold_pnl_svg(xgb_bts)

    if xgb_shap:
        xgb_shap_html = shap_bar_chart(xgb_shap, "XGBoost SHAP Summary (test sample)")
        xgb_shap_table_html = html_table(
            ["Feature", "Mean |SHAP|", "Mean SHAP", "Feature/SHAP corr"],
            shap_rows(xgb_shap),
        )

    sig_stats_rows = []
    for s in sig_stats[:50]:
        sig_stats_rows.append([s["feature"], fmt(s["coefficient"], 4), fmt(s["std_error"], 4),
                               fmt(s["z_stat"], 2), f"{s['p_value']:.3g}"])
    sig_html = html_table(["Feature", "Coefficient", "Std error", "Z stat", "P-value"], sig_stats_rows)

    all_stats_rows = []
    for s in sorted(stats, key=lambda x: abs(x["z_stat"] or 0), reverse=True)[:80]:
        pv = f"{s['p_value']:.3g}" if s["p_value"] is not None else ""
        all_stats_rows.append([s["feature"], fmt(s["coefficient"], 4), fmt(s["std_error"], 4),
                               fmt(s["z_stat"], 2), pv])
    all_stats_html = html_table(["Feature", "Coefficient", "Std error", "Z stat", "P-value"], all_stats_rows)

    factor_groups_html = html_table(["Group", "Count", "Features"], factor_group_rows)

    model_card_html = ""
    limitations_html = ""
    if model_card:
        card_rows = [[key, html.escape(str(value))] for key, value in model_card.items()]
        model_card_html = html_table(["Field", "Value"], card_rows)
        limitations = model_card.get("Known failure modes", "")
        if limitations:
            limitations_html = f"<div class='warn'>{html.escape(str(limitations))}</div>"

    calibration_html = ""
    if calibration:
        cal_rows = []
        baseline_bins = {row["label"]: row for row in calibration.get("baseline", [])}
        adjusted_bins = {row["label"]: row for row in calibration.get("adjusted", [])}
        for label in sorted(set(baseline_bins) | set(adjusted_bins)):
            base = baseline_bins.get(label, {})
            adj = adjusted_bins.get(label, {})
            cal_rows.append([
                label,
                f"{int(base.get('count', 0)):,}",
                fmt(base.get("expected")),
                fmt(base.get("observed")),
                fmt(base.get("gap")),
                f"{int(adj.get('count', 0)):,}",
                fmt(adj.get("expected")),
                fmt(adj.get("observed")),
                fmt(adj.get("gap")),
            ])
        calibration_html = html_table(
            ["Bin", "Base n", "Base pred", "Base obs", "Base gap",
             "Adj n", "Adj pred", "Adj obs", "Adj gap"],
            cal_rows,
        )

    # Adjustment difference distribution
    diff_bins_rows = []
    for b in diff_bins:
        pct = b["pct"]
        bar_width = min(pct * 2, 100)
        bar = f"<div style='width:{bar_width:.1f}%;height:10px;background:#57606a;display:inline-block'></div>"
        diff_bins_rows.append([b["label"], f"{b['count']:,}", f"{pct:.1f}%", bar])
    diff_bins_html = html_table(["|Adjusted - De-vig| range", "Rows", "% of total", "Distribution"], diff_bins_rows)

    # Backtest
    pnl_chart = pnl_svg_chart(backtest)
    bt = backtest
    backtest_rows = [[
        f"{bt['n_bets']:,}",
        f"{bt['n_rows']:,}",
        f"{bt['total_wagered']:.2f}",
        f"{bt['total_pnl']:+.2f}",
        f"{bt['win_rate']:.1%}",
        f"{bt['roi']:+.2%}",
        f"{bt['avg_edge']:.4f}",
    ]]
    backtest_html = html_table(
        ["Bets placed", "Total rows", "Total wagered", "Total PnL", "Win rate", "ROI", "Avg edge"],
        backtest_rows,
    )

    # Split info
    split_rows = [
        ["Train (60%)", str(split_info["train_events"]), f"{split_info['train_rows']:,}",
         str(split_info["train_start"]), str(split_info["train_end"])],
        ["Test (40%)", str(split_info["test_events"]), f"{split_info['test_rows']:,}",
         str(split_info["test_start"]), str(split_info["test_end"])],
    ]
    split_html = html_table(["Split", "Matches", "Rows", "Date range start", "Date range end"], split_rows)

    # Multi-threshold backtest table
    if backtests:
        thresh_rows = []
        for thresh in sorted(backtests.keys()):
            bt = backtests[thresh]
            roi_cls = "delta-pos" if bt["roi"] > 0 else "delta-neg"
            thresh_rows.append([
                f"{thresh:g}{' *' if abs(thresh - PRIMARY_BACKTEST_THRESHOLD) < 1e-9 else ''}",
                f"{bt['n_bets']:,}",
                f"{bt.get('n_p1_bets', 0):,}",
                f"{bt.get('n_p2_bets', 0):,}",
                f"{bt['total_wagered']:.1f}",
                f"{bt['total_pnl']:+.2f}",
                f"{bt['win_rate']:.1%}",
                f"<span class='{roi_cls}'>{bt['roi']:+.2%}</span>",
                fmt(bt.get("max_drawdown"), 2),
                fmt(bt.get("profit_factor"), 2) if bt.get("profit_factor") is not None else "",
                f"{bt['avg_edge']:.4f}",
            ])
        thresh_html = html_table(
            ["Threshold", "Bets", "P1 bets", "P2 bets", "Wagered", "Total PnL", "Win rate", "ROI",
             "Max DD", "Profit factor", "Avg edge"],
            thresh_rows,
        )
        # Multi-threshold PnL chart
        thresh_chart = multi_threshold_pnl_svg(backtests)
    else:
        thresh_html = ""
        thresh_chart = ""

    n_rows = len(df)
    n_matches = df["event_id"].nunique()
    n_sig = len(sig_stats)
    n_features = len(factor_features)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Adjusted Probability Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Adjusted Probability: De-vig Odds + Factor Adjustment</h1>
  <p class="muted">Calibrates a probability-space adjustment on Pinnacle de-vig odds using
  dual-side Brier loss (P1 and P2). All rows with Pinnacle odds (no staleness filter). 60/40 event-level time split.
  Max 2 factors per group. Uncapped probability-space adjustment.</p>

  <div class="formula">
    adjustment = standardized_factors &middot; beta<br>
    adjusted_prob_p1 = devig_prob_p1 + adjustment<br>
    adjusted_prob_p2 = devig_prob_p2 - adjustment<br>
    <br>
    Calibration loss = mean[ (adj_p1 - y)^2 + (adj_p2 - (1-y))^2 ]<br>
    <br>
    odds source: <b>Pinnacle only</b>  &nbsp;&nbsp;|&nbsp;&nbsp; max 2 factors per group  &nbsp;&nbsp;|&nbsp;&nbsp; all odds staleness
  </div>

  <h2>How De-vig Probability Is Computed from Pinnacle Odds</h2>
  <div class="note">
    <p>The enrichment script fetches H2H (head-to-head) decimal odds from The Odds API for each match snapshot.
    This report uses <b>Pinnacle only</b> — known for sharp odds and low vig. For each snapshot, it extracts
    Pinnacle's price for each player, then removes the vig as follows:</p>
    <div class="formula">
      Step 1: pinnacle_price_p1 = Pinnacle's H2H price for P1<br>
      Step 2: pinnacle_price_p2 = Pinnacle's H2H price for P2<br>
      <br>
      Step 3: implied_p1 = 1 / pinnacle_price_p1<br>
      Step 4: implied_p2 = 1 / pinnacle_price_p2<br>
      <br>
      Step 5: vig_total = implied_p1 + implied_p2  &nbsp;&nbsp;(this sums to &gt; 1.0 due to vig)<br>
      <br>
      Step 6: no_vig_p1 = implied_p1 / vig_total  &nbsp;&nbsp;(normalize to sum to 1.0)<br>
      Step 7: no_vig_p2 = implied_p2 / vig_total<br>
      <br>
      Example: if pinnacle_price_p1 = 1.80, pinnacle_price_p2 = 2.10<br>
      &nbsp;&nbsp;implied_p1 = 0.5556, implied_p2 = 0.4762<br>
      &nbsp;&nbsp;vig_total = 1.0317 (3.17% vig)<br>
      &nbsp;&nbsp;no_vig_p1 = 0.5556 / 1.0317 = 0.5385 (53.85%)<br>
      &nbsp;&nbsp;no_vig_p2 = 0.4762 / 1.0317 = 0.4615 (46.15%)
    </div>
    <p>The baseline and adjusted probabilities use <b>no_vig_p1</b> and <b>no_vig_p2</b> from the steps above.</p>
  </div>

  <div class="grid">
    <div class="card"><div class="muted">Rows with odds</div><div class="stat">{n_rows:,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{n_matches:,}</div></div>
    <div class="card"><div class="muted">Factor features</div><div class="stat">{n_features}</div></div>
    <div class="card"><div class="muted">Significant (p&lt;0.05)</div><div class="stat">{n_sig}</div></div>
  </div>

  <div class="note">
    <b>Shared setup:</b> Pinnacle de-vig odds baseline, 60/40 event-level time split, max 2 factors per group.
    Use the tabs below to compare the <b>logistic regression</b> calibrator vs the <b>XGBoost</b> calibrator.
  </div>

  <h2>Train / Test Split</h2>
  <div class="note">
    <p>Events (matches) are sorted chronologically by <code>event_start_date</code>, then split:</p>
    <ul>
      <li><b>Train (60%)</b> &mdash; earliest matches; used to calibrate models</li>
      <li><b>Test (40%)</b> &mdash; all remaining matches; used for evaluation and backtest</li>
    </ul>
    <p>XGBoost additionally holds out the latest 15% of train events for early stopping validation.</p>
  </div>
  {split_html}

  <h2>Model Card</h2>
  {model_card_html}
  <h3>Known Limitations</h3>
  {limitations_html}

  <div class="tab-bar">
    <button type="button" class="tab-btn active" data-tab="tab-regression" onclick="showTab('tab-regression')">Logistic Regression</button>
    <button type="button" class="tab-btn" data-tab="tab-xgboost" onclick="showTab('tab-xgboost')">XGBoost</button>
  </div>

  <div id="tab-regression" class="tab-panel active">
    <div class="note">
      <b>Models compared:</b>
      <ul>
        <li><b>Baseline</b> &mdash; Pinnacle de-vig implied probability (no factors)</li>
        <li><b>Adjusted</b> &mdash; devig P1/P2 &plusmn; linear factor adjustment (dual-side Brier calibration)</li>
        <li><b>Factors only</b> &mdash; logistic regression on selected factors alone, no odds prior (for reference)</li>
      </ul>
      The adjustment operates in <b>probability space</b> with no hard cap during calibration or prediction.
      Both adjusted P1 and adjusted P2 contribute to the training loss.
    </div>

    <div class="formula">
      adjustment = standardized_factors &middot; beta<br>
      adjusted_prob_p1 = devig_prob_p1 + adjustment<br>
      adjusted_prob_p2 = devig_prob_p2 - adjustment<br>
      <br>
      Calibration loss = mean[ (adj_p1 - y)^2 + (adj_p2 - (1-y))^2 ]
    </div>

    <h2>Overall Comparison: Adjusted vs Baseline</h2>
    <p>Positive delta in AUC (or negative in Brier/LogLoss) means the factor adjustment improves on raw odds.</p>
    {delta_html}

    <h2>Probability Calibration (test set, match-level)</h2>
    <p>Reliability bins for P1 win probability on one row per match. Gaps near zero indicate good calibration.</p>
    {calibration_html}

    <h2>Detailed Results — Match-Level (primary)</h2>
    <div class="note">One row per match: earliest in-play snapshot with Pinnacle odds. Avoids inflating fit metrics with correlated within-match rows sharing the same match-end label.</div>
    {overall_html}

    <h2>Detailed Results — All In-Play Rows (secondary)</h2>
    <div class="muted">Full test set with repeated match-end label per snapshot. Useful for diagnostics only.</div>
    {row_level_html}

    <h2>Fresh Odds Subset (0-1 min staleness only)</h2>
    <div class="warn">When odds are very fresh, the bookmaker probability is most efficient.
    Does the adjustment still help?</div>
    {fresh_html}

    <h2>Adjustment Magnitude: |Adjusted Probability - De-vig Probability|</h2>
    <p>How much does the factor adjustment move the probability away from the bookmaker's de-vig estimate?
    A small mean adjustment with improved accuracy means factors fine-tune odds efficiently.</p>
    <div class="grid">
      <div class="card"><div class="muted">Mean abs diff</div><div class="stat">{fmt(diff_stats.get('mean'))}</div></div>
      <div class="card"><div class="muted">Median abs diff</div><div class="stat">{fmt(diff_stats.get('median'))}</div></div>
      <div class="card"><div class="muted">90th pctile</div><div class="stat">{fmt(diff_stats.get('p90'))}</div></div>
      <div class="card"><div class="muted">99th pctile</div><div class="stat">{fmt(diff_stats.get('p99'))}</div></div>
    </div>
    {diff_bins_html}

    <h2>Backtest: Two-Sided Strategy (P1 and P2)</h2>
    <div class="note">
      <p><b>Strategy:</b> At each row in the test set:</p>
      <ul>
        <li><b>Buy P1</b> if <code>adjusted_prob_p1 &gt; raw_implied_p1 + threshold</code></li>
        <li><b>Buy P2</b> if <code>adjusted_prob_p2 &gt; raw_implied_p2 + threshold</code></li>
        <li>Execution price = raw implied from the <b>next odds update</b> in the same match</li>
        <li>If both fire, take the side with the larger edge</li>
        <li>No bet if neither condition holds or no later odds update exists</li>
        <li>At most one trade per match every {MIN_TRADE_GAP_SECONDS} seconds</li>
      </ul>
      <ul>
        <li>PnL chart: cumulative sum of per-match aggregated PnL</li>
        <li>Each bet is 1 unit (cost = trade price, payout = 1 if win)</li>
      </ul>
      <p>Signal uses current-row odds; fill uses the next Pinnacle snapshot price.
      Pre-specified reference threshold: <b>{PRIMARY_BACKTEST_THRESHOLD:g}</b> (marked *).
      Primary backtest chart uses validation-selected threshold. Other thresholds are exploratory sensitivity checks.</p>
    </div>

    <h3>Threshold Comparison</h3>
    {thresh_html}

    <h3>Cumulative PnL by Threshold</h3>
    {thresh_chart}

    <h2>Segment Analysis</h2>
    <p>AUC by staleness, match progression, and league. Staleness buckets: 0-1, 1-2.5, 2.5-5, 5-7.5, 7.5-10 minutes.
    PnL, turnover, and return-on-turnover (ROT) use 1-unit backtest trades at threshold {SEGMENT_BACKTEST_THRESHOLD}.</p>
    {seg_html}

    <h2>Significant Factor Adjustments (p &lt; 0.05)</h2>
    <p>Factors with statistically significant coefficients in the adjustment model.
    Positive coefficient means the factor pushes probability toward P1 winning (above what odds imply).</p>
    {sig_html}

    <h2>All Factor Coefficients</h2>
    {all_stats_html}

    <h2>Factor Groups</h2>
    {factor_groups_html}
  </div>

  <div id="tab-xgboost" class="tab-panel">
    <div class="note">
      <b>XGBoost adjustment model</b> &mdash; binary classifier on selected factors plus <code>devig_p1</code> and
      <code>devig_p2</code>. Validation split carved from train events for early stopping.
      Predictions are converted to dual-side adjusted probabilities the same way as logistic:
      <code>adjustment = pred_p1 - devig_p1</code>, then <code>adj_p2 = devig_p2 - adjustment</code>.
    </div>

    <div class="formula">
      XGBoost predicts P(P1 wins) from factors + de-vig odds<br>
      adjustment = xgb_pred_p1 - devig_p1<br>
      adjusted_prob_p1 = devig_p1 + adjustment<br>
      adjusted_prob_p2 = devig_p2 - adjustment<br>
      <br>
      Features ({len(xgb_features or [])}): {", ".join(xgb_features or [])}
    </div>

    <h2>Overall Comparison: XGBoost vs Baseline</h2>
    {xgb_delta_html}

    <h2>Detailed Results (test set)</h2>
    {xgb_fit_html}

    <h2>Backtest: Two-Sided Strategy</h2>
    <div class="note">
      Same backtest rules as the logistic tab: dual-side signals, next-update fill, 1-unit bets,
      {MIN_TRADE_GAP_SECONDS}s cooldown per match.
    </div>

    <h3>Threshold Comparison</h3>
    {xgb_thresh_html}

    <h3>Cumulative PnL by Threshold</h3>
    {xgb_thresh_chart}

    <h2>SHAP Feature Importance</h2>
    <p>Mean absolute SHAP values on a random test sample ({len(xgb_shap or [])} features ranked).
    Shows which inputs drive XGBoost probability shifts beyond the de-vig baseline.</p>
    {xgb_shap_html}
    <h3>Top SHAP Features</h3>
    {xgb_shap_table_html}
  </div>

  <script>
    function showTab(id) {{
      document.querySelectorAll('.tab-panel').forEach(function(panel) {{
        panel.classList.remove('active');
      }});
      document.querySelectorAll('.tab-btn').forEach(function(btn) {{
        btn.classList.remove('active');
      }});
      document.getElementById(id).classList.add('active');
      document.querySelector('[data-tab="' + id + '"]').classList.add('active');
    }}
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Adjusted Probability Report ===")
    print()

    # Step 1: Load enriched data
    print("1. Loading enriched CSVs with odds...")
    df = load_enriched_rows()
    has_odds = (df["has_odds"] == 1).sum()
    print(f"   {len(df):,} rows, {df['event_id'].nunique()} matches, {has_odds:,} rows with odds ({has_odds/len(df):.1%})")
    print()

    # Step 2: Build factor features
    print("2. Building factor features...")
    df, factor_features, factor_groups = add_factor_features(df)
    print(f"   {len(factor_features)} factor features")
    print()

    # Step 3: Filter to rows with odds (no staleness cap)
    print("3. Filtering to rows with Pinnacle odds (no staleness filter)...")
    df = df[df["has_odds"] == 1].copy()
    df["devig_p1"] = pd.to_numeric(df["odds_no_vig_p1"], errors="coerce")
    df["devig_p2"] = pd.to_numeric(df["odds_no_vig_p2"], errors="coerce").fillna(1.0 - df["devig_p1"])
    if "odds_age_seconds" in df.columns:
        forward_rows = int((df["odds_age_seconds"] < 0).sum())
        if forward_rows:
            df = df[df["odds_age_seconds"] >= 0].copy()
            print(f"   Excluded {forward_rows:,} rows with forward-looking odds (age < 0)")
    assert_no_forward_odds(df)
    print(f"   {len(df):,} rows with odds ({df['event_id'].nunique()} matches)")
    print()

    # Step 4: Split data (60% train, 40% test — no validation)
    print("4. Splitting train/test (60/40 by event, no validation)...")
    events = (
        df[["event_id", "event_start_date"]]
        .drop_duplicates("event_id")
        .sort_values(["event_start_date", "event_id"], na_position="last")
    )
    event_ids = events["event_id"].tolist()
    n = len(event_ids)
    train_set = set(event_ids[: int(n * 0.60)])
    test_set = set(event_ids[int(n * 0.60):])
    work = df.loc[df[TARGET_COL].notna()].copy()
    train = work.loc[work["event_id"].isin(train_set)].copy()
    test = work.loc[work["event_id"].isin(test_set)].copy()
    print(f"   Train: {len(train):,} rows ({train['event_id'].nunique()} matches)")
    print(f"   Test:  {len(test):,} rows ({test['event_id'].nunique()} matches)")
    train_fit, val_fit = split_train_validation(train)
    print(f"   Train-fit (feature selection): {len(train_fit):,} rows ({train_fit['event_id'].nunique()} matches)")
    print(f"   Train-val (XGB early stop):    {len(val_fit):,} rows ({val_fit['event_id'].nunique()} matches)")
    train_weights = match_equal_row_weights(train)
    train_fit_weights = match_equal_row_weights(train_fit)
    print()

    # Step 5: Baseline prediction (de-vig odds, no adjustment)
    print("5. Computing baseline (de-vig Pinnacle odds, no adjustment)...")
    test_scored = test.copy()
    test_scored["pred_baseline_p1"] = pd.to_numeric(test["devig_p1"], errors="coerce")
    test_scored["pred_baseline_p2"] = pd.to_numeric(test["devig_p2"], errors="coerce")
    test_primary = primary_eval_rows(test_scored)
    y_test_primary = test_primary[TARGET_COL].to_numpy(dtype=float)
    baseline_pred = test_scored["pred_baseline_p1"].to_numpy(dtype=float)
    baseline_p2_pred = test_scored["pred_baseline_p2"].to_numpy(dtype=float)
    baseline_pred_primary = test_primary["pred_baseline_p1"].to_numpy(dtype=float)
    baseline_p2_primary = test_primary["pred_baseline_p2"].to_numpy(dtype=float)
    baseline_metrics = evaluate_dual_predictions(y_test_primary, baseline_pred_primary, baseline_p2_primary)
    baseline_metrics_row = evaluate_dual_predictions(
        test[TARGET_COL].to_numpy(dtype=float), baseline_pred, baseline_p2_pred,
    )
    print(f"   Baseline (match-level): AUC={fmt(baseline_metrics['auc'])} Brier={fmt(baseline_metrics['brier'])} "
          f"({int(baseline_metrics['rows'])} matches)")
    print(f"   Baseline (all rows):    AUC={fmt(baseline_metrics_row['auc'])} Brier={fmt(baseline_metrics_row['brier'])}")
    print()

    # Step 6: First pass — feature selection on train-fit fold only (nested validation)
    print("6. First pass: feature selection on train-fit fold (nested validation)...")
    x_train_fit = fill_matrix(train_fit, train_fit, factor_features)
    y_train_fit = train_fit[TARGET_COL].to_numpy(dtype=float)
    devig_p1_train_fit = train_fit["devig_p1"].to_numpy(dtype=float)
    devig_p2_train_fit = train_fit["devig_p2"].to_numpy(dtype=float)

    model_pass1 = fit_probability_adjustment(
        x_train_fit, y_train_fit, devig_p1_train_fit, devig_p2_train_fit,
        l2=0.02, learning_rate=0.06, epochs=500,
        sample_weight=train_fit_weights,
    )
    stats_pass1 = probability_adjustment_stats(
        model_pass1, train_fit, "devig_p1", "devig_p2", TARGET_COL, factor_features,
    )

    # Select top 2 factors per group by |z-stat|
    selected_features = select_top_factors_per_group(stats_pass1, factor_groups, max_per_group=2)
    assert_no_feature_leakage(selected_features)
    print(f"   Selected {len(selected_features)} factors (max 2 per group, train-fit only)")
    for grp, vals in factor_groups.items():
        selected_in_grp = [f for f in vals if f in selected_features]
        if selected_in_grp:
            print(f"     {grp}: {', '.join(selected_in_grp)}")
    print()

    # Step 7: Second pass — retrain with selected factors on full train
    print("7. Second pass: training with selected factors (dual-side Brier)...")
    x_train = fill_matrix(train, train, selected_features)
    y_train = train[TARGET_COL].to_numpy(dtype=float)
    devig_p1_train = train["devig_p1"].to_numpy(dtype=float)
    devig_p2_train = train["devig_p2"].to_numpy(dtype=float)
    x_train_sel = x_train
    model = fit_probability_adjustment(
        x_train_sel, y_train, devig_p1_train, devig_p2_train,
        l2=0.02, learning_rate=0.06, epochs=500,
        sample_weight=train_weights,
    )

    val_adj_p1, val_adj_p2, _ = predict_adjusted_probs(
        model,
        fill_matrix(train, val_fit, selected_features),
        val_fit["devig_p1"].to_numpy(dtype=float),
        val_fit["devig_p2"].to_numpy(dtype=float),
    )
    val_primary = primary_eval_rows(val_fit.assign(
        pred_adj_p1=val_adj_p1,
        pred_adj_p2=val_adj_p2,
    ))
    val_metrics = evaluate_dual_predictions(
        val_primary[TARGET_COL].to_numpy(dtype=float),
        val_primary["pred_adj_p1"].to_numpy(dtype=float),
        val_primary["pred_adj_p2"].to_numpy(dtype=float),
    )
    print(f"   Train-val check (match-level): AUC={fmt(val_metrics['auc'])} Brier={fmt(val_metrics['brier'])}")

    devig_p1_test = test["devig_p1"].to_numpy(dtype=float)
    devig_p2_test = test["devig_p2"].to_numpy(dtype=float)
    adjusted_pred, adjusted_p2_pred, adjustment_pred = predict_adjusted_probs(
        model,
        fill_matrix(train, test, selected_features),
        devig_p1_test,
        devig_p2_test,
    )
    test_scored["pred_adjusted_p1"] = adjusted_pred
    test_scored["pred_adjusted_p2"] = adjusted_p2_pred
    test_primary = primary_eval_rows(test_scored)
    adjusted_metrics = evaluate_dual_predictions(
        test_primary[TARGET_COL].to_numpy(dtype=float),
        test_primary["pred_adjusted_p1"].to_numpy(dtype=float),
        test_primary["pred_adjusted_p2"].to_numpy(dtype=float),
    )
    adjusted_metrics_row = evaluate_dual_predictions(
        test[TARGET_COL].to_numpy(dtype=float), adjusted_pred, adjusted_p2_pred,
    )
    print(f"   Logistic (match-level): AUC={fmt(adjusted_metrics['auc'])} Brier={fmt(adjusted_metrics['brier'])} "
          f"({int(adjusted_metrics['rows'])} matches)")
    print(f"   Logistic (all rows):    AUC={fmt(adjusted_metrics_row['auc'])} Brier={fmt(adjusted_metrics_row['brier'])}")
    print()

    # Step 7b: XGBoost adjustment model (same selected features + de-vig odds)
    print("7b. Training XGBoost adjustment model...")
    xgb_features = selected_features + ["devig_p1", "devig_p2"]
    assert_no_feature_leakage(xgb_features)
    print(f"   XGB train/val events: {train_fit['event_id'].nunique()} / {val_fit['event_id'].nunique()}")
    xgb_booster, xgb_p1_pred = train_xgboost_adjustment(
        train_fit, val_fit, test, xgb_features,
    )
    xgb_adj_p1, xgb_adj_p2, _ = dual_adjusted_probs_from_p1(
        xgb_p1_pred, devig_p1_test, devig_p2_test,
    )
    xgb_scored = test.copy()
    xgb_scored["pred_adj_p1"] = xgb_adj_p1
    xgb_scored["pred_adj_p2"] = xgb_adj_p2
    xgb_primary = primary_eval_rows(xgb_scored)
    xgb_metrics = evaluate_dual_predictions(
        xgb_primary[TARGET_COL].to_numpy(dtype=float),
        xgb_primary["pred_adj_p1"].to_numpy(dtype=float),
        xgb_primary["pred_adj_p2"].to_numpy(dtype=float),
    )
    xgb_metrics_row = evaluate_dual_predictions(
        test[TARGET_COL].to_numpy(dtype=float), xgb_adj_p1, xgb_adj_p2,
    )
    print(f"   XGBoost (match-level): AUC={fmt(xgb_metrics['auc'])} Brier={fmt(xgb_metrics['brier'])}")
    print(f"   XGBoost (all rows):    AUC={fmt(xgb_metrics_row['auc'])} Brier={fmt(xgb_metrics_row['brier'])}")
    print("   Computing XGBoost SHAP summary...")
    xgb_shap = shap_summary(xgb_booster, train, test, xgb_features)
    print()

    # Step 8: Factors-only model (no odds prior, for reference)
    print("8. Training factors-only model (logistic regression, no offset)...")
    from train_calibrated_probability_model import fit_logistic, predict_logistic
    factors_model = fit_logistic(
        fill_matrix(train, train, selected_features), y_train,
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    factors_pred = predict_logistic(factors_model, fill_matrix(train, test, selected_features))
    test_scored["pred_factors_p1"] = factors_pred
    factors_primary = primary_eval_rows(test_scored)
    factors_metrics = evaluate_predictions(
        factors_primary[TARGET_COL].to_numpy(dtype=float),
        factors_primary["pred_factors_p1"].to_numpy(dtype=float),
    )
    factors_metrics_row = evaluate_predictions(
        test[TARGET_COL].to_numpy(dtype=float), factors_pred,
    )
    print(f"   Factors only (match-level): AUC={fmt(factors_metrics['auc'])} Brier={fmt(factors_metrics['brier'])}")
    print(f"   Factors only (all rows):    AUC={fmt(factors_metrics_row['auc'])} Brier={fmt(factors_metrics_row['brier'])}")
    print()

    # Step 9: Compute regression stats for selected factors
    print("9. Computing factor t-stats and p-values...")
    stats = probability_adjustment_stats(
        model, train, "devig_p1", "devig_p2", TARGET_COL, selected_features,
    )
    sig_count = sum(1 for s in stats if s["p_value"] is not None and s["p_value"] < 0.05)
    print(f"   {sig_count} significant factors (p < 0.05)")
    print()

    # Step 9: Segment analysis (with backtest PnL / ROT at reference threshold)
    print(f"9. Segment analysis (backtest threshold={SEGMENT_BACKTEST_THRESHOLD})...")
    segment_bt = run_backtest(test, adjusted_pred, adjusted_p2_pred, threshold=SEGMENT_BACKTEST_THRESHOLD)
    segment_df = segment_bt["df"]
    sort_cols = ["event_start_date", "event_id"]
    if "ut" in test.columns:
        sort_cols.append("ut")
    if "seq" in test.columns:
        sort_cols.append("seq")
    sorted_test = test.sort_values(sort_cols, na_position="last")
    factors_pred_sorted = predict_logistic(
        factors_model, fill_matrix(train, sorted_test, selected_features),
    )
    segments = segment_analysis(segment_df, factors_pred_sorted)
    for r in segments:
        rot = fmt_pct(r.get("rot")) if r.get("rot") is not None else "n/a"
        print(f"   {r['segment']}: {r['rows']} rows, "
              f"base_auc={fmt(r.get('baseline_auc'))}, "
              f"adj_auc={fmt(r.get('adjusted_auc'))}, "
              f"pnl={fmt(r.get('pnl'), 2)}, rot={rot}")
    print()

    # Step 11: Fresh odds subset (0-1 min staleness)
    print("11. Evaluating on fresh odds subset (0-1 min staleness)...")
    fresh_test = test[(test["odds_age_seconds"] >= 0) & (test["odds_age_seconds"] < 60)]
    if len(fresh_test) > 0:
        fresh_y = fresh_test[TARGET_COL].to_numpy(dtype=float)
        fresh_baseline = fresh_test["devig_p1"].to_numpy(dtype=float)
        fresh_baseline_p2 = fresh_test["devig_p2"].to_numpy(dtype=float)
        fresh_adjusted_p1, fresh_adjusted_p2, _ = predict_adjusted_probs(
            model,
            fill_matrix(train, fresh_test, selected_features),
            fresh_baseline,
            fresh_baseline_p2,
        )
        fresh_factors = predict_logistic(factors_model, fill_matrix(train, fresh_test, selected_features))
        fresh_subset = {
            "baseline": evaluate_dual_predictions(fresh_y, fresh_baseline, fresh_baseline_p2),
            "adjusted": evaluate_dual_predictions(fresh_y, fresh_adjusted_p1, fresh_adjusted_p2),
            "factors_only": evaluate_predictions(fresh_y, fresh_factors),
        }
        for name, m in fresh_subset.items():
            print(f"   {name}: AUC={fmt(m['auc'])} Brier={fmt(m['brier'])} rows={m['rows']}")
    else:
        fresh_subset = {}
    print()

    # Step 11: Compute adjustment magnitude (|adjusted - baseline|)
    print("11. Computing adjustment magnitude distribution...")
    abs_diff = np.abs(adjusted_pred - baseline_pred)
    diff_stats = {
        "mean": float(np.mean(abs_diff)),
        "median": float(np.median(abs_diff)),
        "p90": float(np.percentile(abs_diff, 90)),
        "p99": float(np.percentile(abs_diff, 99)),
        "max": float(np.max(abs_diff)),
    }
    print(f"   Mean |diff|: {fmt(diff_stats['mean'])}, Median: {fmt(diff_stats['median'])}, "
          f"P90: {fmt(diff_stats['p90'])}, P99: {fmt(diff_stats['p99'])}")

    # Build histogram bins
    bin_edges = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 1.0]
    bin_labels = ["0-0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%", "20%+"]
    diff_bins = []
    total = len(abs_diff)
    for i in range(len(bin_edges) - 1):
        count = int(np.sum((abs_diff >= bin_edges[i]) & (abs_diff < bin_edges[i + 1])))
        diff_bins.append({
            "label": bin_labels[i],
            "count": count,
            "pct": count / total * 100 if total > 0 else 0,
        })
    print()

    # Step 12: Select threshold on validation, then backtest on test
    print("12. Selecting threshold on validation fold...")
    selected_threshold, val_backtests = select_validation_threshold(
        val_fit, val_adj_p1, val_adj_p2, BACKTEST_THRESHOLDS,
        fallback=PRIMARY_BACKTEST_THRESHOLD,
    )
    val_primary_bt = val_backtests.get(selected_threshold, {})
    print(f"   Validation-selected threshold: {selected_threshold:g} "
          f"(ROI={val_primary_bt.get('roi', 0):+.2%}, bets={val_primary_bt.get('n_bets', 0)})")
    print(f"   Pre-specified threshold:       {PRIMARY_BACKTEST_THRESHOLD:g}")
    print()

    print("12b. Running backtests across thresholds on test...")
    thresholds = BACKTEST_THRESHOLDS
    backtests = {}
    xgb_backtests = {}
    for thresh in thresholds:
        bt = run_backtest(test, adjusted_pred, adjusted_p2_pred, threshold=thresh)
        backtests[thresh] = bt
        xgb_bt = run_backtest(test, xgb_adj_p1, xgb_adj_p2, threshold=thresh)
        xgb_backtests[thresh] = xgb_bt
        print(f"   thresh={thresh:g}: logistic PnL={bt['total_pnl']:+.2f} ROI={bt['roi']:+.2%} | "
              f"xgb PnL={xgb_bt['total_pnl']:+.2f} ROI={xgb_bt['roi']:+.2%}")
    backtest = backtests[selected_threshold]
    prespecified_backtest = backtests.get(PRIMARY_BACKTEST_THRESHOLD, backtest)
    adjustment_comparison = {
        "baseline": baseline_metrics,
        "logistic": adjusted_metrics,
        "xgboost": xgb_metrics,
        "backtests": {
            "logistic": backtests,
            "xgboost": xgb_backtests,
        },
    }
    calibration = {
        "baseline": calibration_bins(
            test_primary[TARGET_COL].to_numpy(dtype=float),
            test_primary["pred_baseline_p1"].to_numpy(dtype=float),
        ),
        "adjusted": calibration_bins(
            test_primary[TARGET_COL].to_numpy(dtype=float),
            test_primary["pred_adjusted_p1"].to_numpy(dtype=float),
        ),
    }

    # Step 13: Collect split info
    split_info = {
        "train_events": train["event_id"].nunique(),
        "train_rows": len(train),
        "train_fit_events": train_fit["event_id"].nunique(),
        "train_val_events": val_fit["event_id"].nunique(),
        "train_start": str(train["event_start_date"].min().date()) if train["event_start_date"].notna().any() else "N/A",
        "train_end": str(train["event_start_date"].max().date()) if train["event_start_date"].notna().any() else "N/A",
        "test_events": test["event_id"].nunique(),
        "test_rows": len(test),
        "test_start": str(test["event_start_date"].min().date()) if test["event_start_date"].notna().any() else "N/A",
        "test_end": str(test["event_start_date"].max().date()) if test["event_start_date"].notna().any() else "N/A",
    }
    primary_bt = backtests[selected_threshold]
    model_card = {
        "Objective": "Improve in-play match-winner probability vs Pinnacle de-vig odds",
        "Horizon": "P(P1 wins match | state and odds at snapshot t); evaluated at earliest odds row per match",
        "Target": "target_p1_win — final match outcome used as conditional label (one eval row/match)",
        "Universe": "Pinnacle H2H tennis with enriched rolling metrics",
        "Features": ", ".join(selected_features),
        "Feature selection": "Top 2 per group by |z-stat| on train-fit fold only (nested)",
        "Training weights": "Match-equal row weights (1 / rows per match) to avoid match repetition bias",
        "Train window": f"{split_info['train_start']} to {split_info['train_end']} ({split_info['train_events']} matches)",
        "Validation method": "15% latest train events for feature selection, threshold selection, and XGB early stopping",
        "Test window": f"{split_info['test_start']} to {split_info['test_end']} ({split_info['test_events']} matches)",
        "Costs assumed": "1-unit bets; fill at next raw implied price; vig in raw price; no extra commission model",
        "Validation threshold": f"{selected_threshold:g}",
        "Pre-specified threshold": f"{PRIMARY_BACKTEST_THRESHOLD:g}",
        "Key metrics (test, match-level)": (
            f"Logistic AUC {fmt(adjusted_metrics['auc'])}, Brier {fmt(adjusted_metrics['brier'])}; "
            f"ROI @ val thresh {fmt_pct(primary_bt['roi'])}, max DD {fmt(primary_bt.get('max_drawdown'), 2)}"
        ),
        "Known failure modes": (
            "Match-end label still conditions on final outcome; match-level eval reduces but does not remove "
            "late-match leakage in backtest rows; threshold sweep on test remains exploratory; "
            "no walk-forward refit; capacity and market impact not modeled."
        ),
    }
    print(f"   Train: {split_info['train_events']} matches, {split_info['train_start']} to {split_info['train_end']}")
    print(f"   Test:  {split_info['test_events']} matches, {split_info['test_start']} to {split_info['test_end']}")
    print()

    # Step 15: Build report
    print("15. Building HTML report...")
    overall = {
        "baseline": baseline_metrics,
        "adjusted": adjusted_metrics,
        "factors_only": factors_metrics,
    }
    overall_row_level = {
        "baseline": baseline_metrics_row,
        "adjusted": adjusted_metrics_row,
        "factors_only": factors_metrics_row,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report(df, selected_features, factor_groups, overall, stats, segments,
                     fresh_subset, diff_stats, diff_bins, backtest, split_info, backtests,
                     adjustment_comparison, xgb_shap, xgb_features, model_card, calibration,
                     overall_row_level),
        encoding="utf-8",
    )
    print(f"   Report: {OUTPUT_PATH}")
    print()

    print("=== Summary (match-level fit, primary) ===")
    print(f"  Baseline (Pinnacle de-vig):  AUC={fmt(baseline_metrics['auc'])} Brier={fmt(baseline_metrics['brier'])}")
    print(f"  Logistic adjusted:          AUC={fmt(adjusted_metrics['auc'])} Brier={fmt(adjusted_metrics['brier'])}")
    print(f"  XGBoost adjusted:           AUC={fmt(xgb_metrics['auc'])} Brier={fmt(xgb_metrics['brier'])}")
    print(f"  Factors only (no odds):     AUC={fmt(factors_metrics['auc'])} Brier={fmt(factors_metrics['brier'])}")
    print(f"  Selected factors:           {len(selected_features)}")
    print(f"  Validation threshold:       {selected_threshold:g}")
    print(f"  Test ROI @ val threshold:   {primary_bt['roi']:+.2%} (pre-spec {PRIMARY_BACKTEST_THRESHOLD:g}: {prespecified_backtest['roi']:+.2%})")
    delta_auc = (adjusted_metrics['auc'] or 0) - (baseline_metrics['auc'] or 0)
    delta_brier = (adjusted_metrics['brier'] or 0) - (baseline_metrics['brier'] or 0)
    xgb_delta_auc = (xgb_metrics['auc'] or 0) - (baseline_metrics['auc'] or 0)
    xgb_delta_brier = (xgb_metrics['brier'] or 0) - (baseline_metrics['brier'] or 0)
    print(f"  Logistic delta AUC/Brier:   {delta_auc:+.4f} / {delta_brier:+.4f}")
    print(f"  XGBoost delta AUC/Brier:    {xgb_delta_auc:+.4f} / {xgb_delta_brier:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
