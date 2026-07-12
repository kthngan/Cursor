#!/usr/bin/env python3
"""Compare HMM, XGBoost, and logistic regression across tennis targets."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import (  # noqa: E402
    REPORTS_DIR,
    SCORE_FEATURES,
    add_hmm_posteriors,
    build_hmm_parameters,
    evaluate_predictions,
    feature_matrix,
    fit_logistic,
    fmt,
    fmt_pct,
    html_table,
    load_training_rows,
    predict_logistic,
    split_events,
)
from train_next_game_set_models import add_future_targets  # noqa: E402
from train_xgboost_factor_models import add_factor_features, train_xgboost  # noqa: E402


OUTPUT_PATH = REPORTS_DIR / "hmm_xgboost_logistic_comparison.html"
TARGETS = {
    "Match result": "target_p1_win",
    "Next game": "target_next_game_p1",
    "Next set": "target_next_set_p1",
}


def split_frames(df, target_col: str):
    train_events, validation_events, test_events = split_events(df)
    work = df.loc[df[target_col].notna()].copy()
    train = work.loc[work["event_id"].isin(train_events)].copy()
    validation = work.loc[work["event_id"].isin(validation_events)].copy()
    test = work.loc[work["event_id"].isin(test_events)].copy()
    return train, validation, test


def evaluate_target(df, target_name: str, target_col: str, factor_features: list[str]) -> tuple[list[list[str]], dict[str, object]]:
    train, validation, test = split_frames(df, target_col)
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    logistic_model = fit_logistic(feature_matrix(train, factor_features), y_train, l2=0.02, learning_rate=0.06, epochs=500)
    logistic_pred = predict_logistic(logistic_model, feature_matrix(test, factor_features))

    _, xgb_pred = train_xgboost(train, validation, test, target_col, factor_features)

    hmm_params = build_hmm_parameters(train)
    hmm_train = add_hmm_posteriors(train.copy(), hmm_params)
    hmm_test = add_hmm_posteriors(test.copy(), hmm_params)
    hmm_cols = [column for column in hmm_train.columns if column.startswith("hmm_state_")]
    hmm_model = fit_logistic(feature_matrix(hmm_train, hmm_cols), y_train, l2=0.05, learning_rate=0.06, epochs=500)
    hmm_pred = predict_logistic(hmm_model, feature_matrix(hmm_test, hmm_cols))

    # A simple raw HMM momentum score is useful as a sanity baseline.
    p1_cols = [column for column in hmm_test.columns if "p1_edge" in column or "p1_strong" in column]
    raw_hmm_pred = np.clip(hmm_test[p1_cols].sum(axis=1).to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    model_predictions = {
        "XGBoost": xgb_pred,
        "Simple logistic regression": logistic_pred,
        "HMM posterior + logistic calibration": hmm_pred,
        "Raw HMM P1 momentum posterior": raw_hmm_pred,
    }
    rows: list[list[str]] = []
    metrics_by_model: dict[str, object] = {}
    for model_name, pred in model_predictions.items():
        metrics = evaluate_predictions(y_test, pred)
        metrics_by_model[model_name] = metrics
        rows.append(
            [
                target_name,
                model_name,
                f"{int(metrics['rows'] or 0):,}",
                str(test["event_id"].nunique()),
                fmt(metrics["auc"]),
                fmt_pct(metrics["accuracy"]),
                fmt(metrics["brier"]),
                fmt(metrics["rmse"]),
                fmt(metrics["log_loss"]),
                fmt_pct(metrics["p1_rate"]),
            ]
        )
    rows.sort(key=lambda row: float(row[4] or -1), reverse=True)
    print(target_name)
    for row in rows:
        print(f"  {row[1]}: AUC={row[4]} accuracy={row[5]} Brier={row[6]} log_loss={row[8]}")
    return rows, metrics_by_model


def build_report(df, factor_features: list[str], all_rows: list[list[str]]) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; position: sticky; top: 0; }
    .note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    """
    headers = ["Target", "Model", "Rows", "Matches", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HMM vs XGBoost vs Logistic Comparison</title>
  <style>{css}</style>
</head>
<body>
  <h1>HMM vs XGBoost vs Logistic Comparison</h1>
  <p class="muted">Targets: match result, next game winner, and next set winner. XGBoost/logistic use score, server, and raw metric factors. HMM uses hidden-state posterior features only.</p>
  <div class="grid">
    <div class="card"><div class="muted">Rows loaded</div><div class="stat">{len(df):,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Factor features</div><div class="stat">{len(factor_features)}</div></div>
  </div>
  <div class="note">
    HMM comparison uses the same 60/20/20 match split. The HMM is trained on the 60% training matches, converted to five posterior-state probabilities,
    then calibrated to each target with a logistic layer. Raw HMM P1 momentum posterior is included as an uncalibrated sanity check.
  </div>
  <h2>Model Comparison</h2>
  {html_table(headers, all_rows)}
</body>
</html>
"""


def main() -> int:
    df = load_training_rows()
    df = add_future_targets(df)
    df, factor_features, _ = add_factor_features(df)
    all_rows: list[list[str]] = []
    for target_name, target_col in TARGETS.items():
        rows, _ = evaluate_target(df, target_name, target_col, factor_features)
        all_rows.extend(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(df, factor_features, all_rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
