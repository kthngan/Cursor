#!/usr/bin/env python3
"""Compare XGBoost and logistic regression for next game/set targets.

Feature scope is intentionally limited to score state, server context, and raw
rolling metrics / direct derivatives. No HMM features and no odds features.
"""

from __future__ import annotations

import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import (  # noqa: E402
    DATA_DIR,
    METRIC_COLUMNS,
    REPORTS_DIR,
    SCORE_FEATURES,
    evaluate_predictions,
    feature_matrix,
    fit_logistic,
    fmt,
    fmt_pct,
    html_table,
    load_training_rows,
    predict_logistic,
    sigmoid,
    split_events,
)
from train_next_game_set_models import add_future_targets  # noqa: E402


OUTPUT_PATH = REPORTS_DIR / "xgboost_factor_model_report.html"
TARGETS = {
    "Next game": "target_next_game_p1",
    "Next set": "target_next_set_p1",
}


@dataclass
class RegressionStats:
    feature: str
    coefficient: float
    std_error: float | None
    z_stat: float | None
    p_value: float | None


def parse_pair(text: object) -> tuple[float, float]:
    if not isinstance(text, str) or "-" not in text:
        return 0.0, 0.0
    left, right = text.split("-", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return 0.0, 0.0


def add_factor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    df = df.copy()
    metric_base = [column for column in METRIC_COLUMNS if column != "live_form_delta_5"]

    groups: dict[str, list[str]] = {
        "Score state": list(SCORE_FEATURES),
        "Raw rolling metrics": list(metric_base),
        "Centered metric edges": [],
        "Trend deltas": [],
        "Server context": [],
        "Server metric interactions": [],
        "Pressure features": [],
        "Pressure metric interactions": [],
        "Missingness": [],
    }

    for metric in metric_base:
        edge = metric.replace("rolling_", "").replace("_ratio", "") + "_edge"
        df[edge] = df[metric] - 0.5
        groups["Centered metric edges"].append(edge)

        missing = f"has_{metric}"
        df[missing] = df[metric].notna().astype(float)
        groups["Missingness"].append(missing)

        for lag in (3, 5, 10):
            delta = f"{metric}_delta_{lag}"
            df[delta] = df.groupby("event_id", sort=False)[metric].diff(lag)
            groups["Trend deltas"].append(delta)

    df["num_available_metrics"] = df[metric_base].notna().sum(axis=1).astype(float)
    groups["Missingness"].append("num_available_metrics")

    df["server_advantage_side"] = df["is_server_p1"] - df["is_server_p2"]
    df["p1_serving_point_diff"] = df["is_server_p1"] * df["point_diff"]
    df["p2_serving_point_diff"] = df["is_server_p2"] * df["point_diff"]
    df["p1_serving_game_diff"] = df["is_server_p1"] * df["game_diff"]
    df["p2_serving_game_diff"] = df["is_server_p2"] * df["game_diff"]
    groups["Server context"].extend(
        [
            "server_advantage_side",
            "p1_serving_point_diff",
            "p2_serving_point_diff",
            "p1_serving_game_diff",
            "p2_serving_game_diff",
        ]
    )

    df["is_late_set"] = ((df["games_total"] >= 8) | (df["game_diff"].abs() >= 4)).astype(float)
    df["server_under_pressure"] = (
        ((df["is_server_p1"] == 1) & (df["point_diff"] < 0))
        | ((df["is_server_p2"] == 1) & (df["point_diff"] > 0))
    ).astype(float)
    df["receiver_pressure_opportunity"] = df["server_under_pressure"]
    df["is_game_point_like"] = (df["point_diff"].abs() >= 3).astype(float)
    df["is_break_point_like"] = (
        ((df["is_server_p1"] == 1) & (df["point_diff"] <= -3))
        | ((df["is_server_p2"] == 1) & (df["point_diff"] >= 3))
    ).astype(float)
    groups["Pressure features"].extend(
        [
            "is_late_set",
            "server_under_pressure",
            "receiver_pressure_opportunity",
            "is_game_point_like",
            "is_break_point_like",
        ]
    )

    key_edges = [
        "live_form_edge",
        "points_20_edge",
        "service_points_won_20_edge",
        "return_points_won_20_edge",
        "break_points_created_20_edge",
        "break_points_won_20_edge",
        "games_won_6_edge",
    ]
    key_edges = [feature for feature in key_edges if feature in df.columns]
    for edge in key_edges:
        for server_feature in ("is_server_p1", "is_server_p2", "server_advantage_side"):
            name = f"{server_feature}_x_{edge}"
            df[name] = df[server_feature] * df[edge]
            groups["Server metric interactions"].append(name)
        for pressure_feature in ("is_late_set", "is_break_point_like", "is_game_point_like", "server_under_pressure"):
            name = f"{pressure_feature}_x_{edge}"
            df[name] = df[pressure_feature] * df[edge]
            groups["Pressure metric interactions"].append(name)

    feature_columns: list[str] = []
    for values in groups.values():
        for feature in values:
            if feature in df.columns and feature not in feature_columns:
                feature_columns.append(feature)
    return df, feature_columns, groups


def split_frames(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_events, validation_events, test_events = split_events(df)
    work = df.loc[df[target_col].notna()].copy()
    train = work.loc[work["event_id"].isin(train_events)].copy()
    validation = work.loc[work["event_id"].isin(validation_events)].copy()
    test = work.loc[work["event_id"].isin(test_events)].copy()
    return train, validation, test


def fill_matrix(train: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    means = train[features].mean(numeric_only=True).fillna(0.0)
    return frame[features].fillna(means).to_numpy(dtype=float)


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> tuple[xgb.Booster, np.ndarray]:
    dtrain = xgb.DMatrix(fill_matrix(train, train, features), label=train[target_col].to_numpy(dtype=float), feature_names=features)
    dvalid = xgb.DMatrix(
        fill_matrix(train, validation, features),
        label=validation[target_col].to_numpy(dtype=float),
        feature_names=features,
    )
    dtest = xgb.DMatrix(fill_matrix(train, test, features), feature_names=features)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.045,
        "max_depth": 3,
        "min_child_weight": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.2,
        "tree_method": "hist",
        "seed": 42,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=450,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    return booster, booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))


def logistic_stats(
    model,
    train: pd.DataFrame,
    target_col: str,
    features: list[str],
) -> list[RegressionStats]:
    x_raw = fill_matrix(train, train, features)
    z = (x_raw - model.mean) / model.std
    x = np.column_stack([np.ones(len(z)), z])
    pred = sigmoid(x @ model.weights)
    weight = np.clip(pred * (1.0 - pred), 1e-8, None)
    hessian = (x.T * weight) @ x
    try:
        cov = np.linalg.pinv(hessian)
        std_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        std_errors = np.full(x.shape[1], np.nan)

    rows: list[RegressionStats] = []
    for idx, feature in enumerate(["intercept", *features]):
        coef = float(model.weights[idx])
        se = float(std_errors[idx]) if math.isfinite(float(std_errors[idx])) else None
        z_stat = coef / se if se and se > 0 else None
        p_value = float(2.0 * norm.sf(abs(z_stat))) if z_stat is not None else None
        rows.append(RegressionStats(feature, coef, se, z_stat, p_value))
    return rows


def shap_summary(
    booster: xgb.Booster,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    sample_size: int = 12000,
) -> list[dict[str, float | str]]:
    sample = test.sample(min(sample_size, len(test)), random_state=42)
    dmatrix = xgb.DMatrix(fill_matrix(train, sample, features), feature_names=features)
    contrib = booster.predict(dmatrix, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))
    feature_contrib = contrib[:, :-1]
    values = fill_matrix(train, sample, features)
    out: list[dict[str, float | str]] = []
    for idx, feature in enumerate(features):
        shap_values = feature_contrib[:, idx]
        raw_values = values[:, idx]
        corr = np.corrcoef(raw_values, shap_values)[0, 1] if np.std(raw_values) > 1e-12 else np.nan
        out.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                "mean_shap": float(np.mean(shap_values)),
                "value_shap_corr": float(corr) if math.isfinite(float(corr)) else 0.0,
            }
        )
    out.sort(key=lambda row: float(row["mean_abs_shap"]), reverse=True)
    return out


def model_comparison_rows(results: dict[str, dict[str, float | None]]) -> list[list[str]]:
    return [
        [
            name,
            f"{int(metrics['rows'] or 0):,}",
            fmt(metrics["auc"]),
            fmt_pct(metrics["accuracy"]),
            fmt(metrics["brier"]),
            fmt(metrics["rmse"]),
            fmt(metrics["log_loss"]),
            fmt_pct(metrics["p1_rate"]),
        ]
        for name, metrics in sorted(results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True)
    ]


def regression_rows(stats: list[RegressionStats], limit: int = 80) -> list[list[str]]:
    filtered = [row for row in stats if row.feature != "intercept"]
    filtered.sort(key=lambda row: abs(row.z_stat or 0.0), reverse=True)
    return [
        [
            row.feature,
            fmt(row.coefficient, 4),
            fmt(row.std_error, 4),
            fmt(row.z_stat, 2),
            f"{row.p_value:.3g}" if row.p_value is not None else "",
        ]
        for row in filtered[:limit]
    ]


def shap_rows(summary: list[dict[str, float | str]], limit: int = 40) -> list[list[str]]:
    return [
        [
            str(row["feature"]),
            fmt(float(row["mean_abs_shap"]), 4),
            fmt(float(row["mean_shap"]), 4),
            fmt(float(row["value_shap_corr"]), 3),
        ]
        for row in summary[:limit]
    ]


def shap_bar_chart(summary: list[dict[str, float | str]], title: str, limit: int = 20) -> str:
    top = summary[:limit]
    if not top:
        return ""
    max_value = max(float(row["mean_abs_shap"]) for row in top) or 1.0
    bars = []
    for row in top:
        feature = html.escape(str(row["feature"]))
        value = float(row["mean_abs_shap"])
        width = max(2.0, 100.0 * value / max_value)
        bars.append(
            f"<div class='bar-row'><div class='bar-label'>{feature}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
            f"<div class='bar-value'>{value:.4f}</div></div>"
        )
    return f"<h3>{html.escape(title)}</h3><div class='bar-chart'>{''.join(bars)}</div>"


def run_target(
    df: pd.DataFrame,
    target_name: str,
    target_col: str,
    features: list[str],
) -> dict[str, object]:
    train, validation, test = split_frames(df, target_col)
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    logistic = fit_logistic(feature_matrix(train, features), y_train, l2=0.02, learning_rate=0.06, epochs=500)
    logistic_pred = predict_logistic(logistic, feature_matrix(test, features))
    booster, xgb_pred = train_xgboost(train, validation, test, target_col, features)

    results = {
        "XGBoost": evaluate_predictions(y_test, xgb_pred),
        "Simple logistic regression": evaluate_predictions(y_test, logistic_pred),
    }
    stats = logistic_stats(logistic, train, target_col, features)
    shap = shap_summary(booster, train, test, features)
    print(f"{target_name}:")
    for name, metrics in sorted(results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True):
        print(
            f"  {name}: AUC={fmt(metrics['auc'])} accuracy={fmt_pct(metrics['accuracy'])} "
            f"Brier={fmt(metrics['brier'])} log_loss={fmt(metrics['log_loss'])}"
        )
    return {
        "target_name": target_name,
        "results": results,
        "stats": stats,
        "shap": shap,
        "rows": len(test),
        "matches": test["event_id"].nunique(),
    }


def build_report(
    df: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    next_game: dict[str, object],
    next_set: dict[str, object],
) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    h3 { margin-top: 20px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; position: sticky; top: 0; }
    .note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .bar-row { display: grid; grid-template-columns: 280px 1fr 70px; gap: 10px; align-items: center; margin: 6px 0; font-size: 13px; }
    .bar-track { height: 12px; background: #f6f8fa; border: 1px solid #d0d7de; }
    .bar-fill { height: 100%; background: #57606a; }
    .bar-label { overflow-wrap: anywhere; }
    """
    factor_rows = [
        [group, str(len([feature for feature in values if feature in features])), ", ".join(feature for feature in values if feature in features)]
        for group, values in groups.items()
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XGBoost Factor Model Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>XGBoost Factor Model Report</h1>
  <p class="muted">Scope: score state, server context, raw rolling metrics, direct trend/interaction derivatives. No HMM and no odds.</p>
  <div class="grid">
    <div class="card"><div class="muted">Rows loaded</div><div class="stat">{len(df):,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Features tested</div><div class="stat">{len(features)}</div></div>
  </div>
  <div class="note">
    XGBoost uses validation split for early stopping only. Logistic regression is the simple regression baseline.
    T-stats and p-values are from the logistic model coefficients on standardized features; with highly correlated engineered features,
    they should be read as diagnostic rather than causal.
  </div>

  <h2>Factor Groups Tested</h2>
  {html_table(["Group", "Feature count", "Features"], factor_rows)}

  <h2>Next Game: XGBoost vs Simple Regression</h2>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"], model_comparison_rows(next_game["results"]))}
  {shap_bar_chart(next_game["shap"], "Next Game XGBoost SHAP Summary")}
  <h3>Next Game: Top SHAP Features</h3>
  {html_table(["Feature", "Mean |SHAP|", "Mean SHAP", "Feature/SHAP corr"], shap_rows(next_game["shap"]))}
  <h3>Next Game: Logistic Regression T-Stats and P-Values</h3>
  {html_table(["Feature", "Coefficient", "Std error", "T/Z stat", "P-value"], regression_rows(next_game["stats"]))}

  <h2>Next Set: XGBoost vs Simple Regression</h2>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"], model_comparison_rows(next_set["results"]))}
  {shap_bar_chart(next_set["shap"], "Next Set XGBoost SHAP Summary")}
  <h3>Next Set: Top SHAP Features</h3>
  {html_table(["Feature", "Mean |SHAP|", "Mean SHAP", "Feature/SHAP corr"], shap_rows(next_set["shap"]))}
  <h3>Next Set: Logistic Regression T-Stats and P-Values</h3>
  {html_table(["Feature", "Coefficient", "Std error", "T/Z stat", "P-value"], regression_rows(next_set["stats"]))}
</body>
</html>
"""


def main() -> int:
    df = load_training_rows()
    df = add_future_targets(df)
    df, features, groups = add_factor_features(df)
    next_game = run_target(df, "Next game", "target_next_game_p1", features)
    next_set = run_target(df, "Next set", "target_next_set_p1", features)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(df, features, groups, next_game, next_set), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
