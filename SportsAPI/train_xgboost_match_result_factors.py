#!/usr/bin/env python3
"""Compare XGBoost and logistic regression for final match result.

Feature scope matches the factor report: score state, server context, raw
rolling metrics, and direct trend/interaction derivatives. No HMM or odds.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import (  # noqa: E402
    REPORTS_DIR,
    evaluate_predictions,
    feature_matrix,
    fit_logistic,
    fmt,
    fmt_pct,
    html_table,
    load_training_rows,
    predict_logistic,
)
from train_xgboost_factor_models import (  # noqa: E402
    add_factor_features,
    model_comparison_rows,
    regression_rows,
    shap_bar_chart,
    shap_rows,
    shap_summary,
    split_frames,
    train_xgboost,
    logistic_stats,
)


OUTPUT_PATH = REPORTS_DIR / "xgboost_match_result_factor_model_report.html"
TARGET_COL = "target_p1_win"


def build_report(df, features, groups, results, stats, shap):
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
  <title>XGBoost Match Result Factor Model Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>XGBoost Match Result Factor Model Report</h1>
  <p class="muted">Target: final match winner from each current row. Scope: score state, server context, raw rolling metrics, and direct derivatives. No HMM and no odds.</p>
  <div class="grid">
    <div class="card"><div class="muted">Rows loaded</div><div class="stat">{len(df):,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Features tested</div><div class="stat">{len(features)}</div></div>
  </div>
  <div class="note">
    Rows are labeled with final match winner, so later score states are naturally very informative.
    T-stats and p-values are from the logistic model on standardized features and should be treated as diagnostics because engineered features are correlated.
  </div>

  <h2>Factor Groups Tested</h2>
  {html_table(["Group", "Feature count", "Features"], factor_rows)}

  <h2>Match Result: XGBoost vs Simple Regression</h2>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"], model_comparison_rows(results))}
  {shap_bar_chart(shap, "Match Result XGBoost SHAP Summary")}
  <h3>Match Result: Top SHAP Features</h3>
  {html_table(["Feature", "Mean |SHAP|", "Mean SHAP", "Feature/SHAP corr"], shap_rows(shap))}
  <h3>Match Result: Logistic Regression T-Stats and P-Values</h3>
  {html_table(["Feature", "Coefficient", "Std error", "T/Z stat", "P-value"], regression_rows(stats))}
</body>
</html>
"""


def main() -> int:
    df = load_training_rows()
    df, features, groups = add_factor_features(df)
    train, validation, test = split_frames(df, TARGET_COL)
    y_train = train[TARGET_COL].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=float)

    logistic = fit_logistic(feature_matrix(train, features), y_train, l2=0.02, learning_rate=0.06, epochs=500)
    logistic_pred = predict_logistic(logistic, feature_matrix(test, features))
    booster, xgb_pred = train_xgboost(train, validation, test, TARGET_COL, features)

    results = {
        "XGBoost": evaluate_predictions(y_test, xgb_pred),
        "Simple logistic regression": evaluate_predictions(y_test, logistic_pred),
    }
    stats = logistic_stats(logistic, train, TARGET_COL, features)
    shap = shap_summary(booster, train, test, features)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(df, features, groups, results, stats, shap), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for name, metrics in sorted(results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True):
        print(
            f"{name}: AUC={fmt(metrics['auc'])} accuracy={fmt_pct(metrics['accuracy'])} "
            f"Brier={fmt(metrics['brier'])} log_loss={fmt(metrics['log_loss'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
