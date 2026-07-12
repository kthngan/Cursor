#!/usr/bin/env python3
"""Train calibrated models for next-game and next-set winner prediction."""

from __future__ import annotations

import html
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import (  # noqa: E402
    DATA_DIR,
    HMM_OBSERVATION_COLUMNS,
    METRIC_COLUMNS,
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
    logit,
    predict_logistic,
    split_events,
)


OUTPUT_PATH = REPORTS_DIR / "next_game_set_probability_model_report.html"
SERVER_INTERACTION_FEATURES = [
    "p1_serving_live_form",
    "p2_serving_live_form",
    "p1_serving_point_diff",
    "p2_serving_point_diff",
    "server_advantage_side",
]


def side_to_target(series: pd.Series) -> pd.Series:
    return series.map({"P1": 1.0, "P2": 0.0})


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("event_id", sort=False):
        group = group.sort_values("seq").copy()
        future_game = group["game_winner_side"].where(group["game_winner_side"].isin(["P1", "P2"])).bfill().shift(-1)
        set_winner_now = group["participant_side"].where(
            group["incident_name"].astype(str).str.casefold().eq("set won")
            & group["participant_side"].isin(["P1", "P2"])
        )
        future_set = set_winner_now.bfill().shift(-1)
        group["target_next_game_p1"] = side_to_target(future_game)
        group["target_next_set_p1"] = side_to_target(future_set)
        out_frames.append(group)
    return pd.concat(out_frames, ignore_index=True)


def add_server_interactions(df: pd.DataFrame) -> pd.DataFrame:
    live_form = df["rolling_live_form_ratio"].astype(float) - 0.5
    df["p1_serving_live_form"] = df["is_server_p1"] * live_form
    df["p2_serving_live_form"] = df["is_server_p2"] * live_form
    df["p1_serving_point_diff"] = df["is_server_p1"] * df["point_diff"]
    df["p2_serving_point_diff"] = df["is_server_p2"] * df["point_diff"]
    df["server_advantage_side"] = df["is_server_p1"] - df["is_server_p2"]
    return df


def segment_results(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    segment_col: str,
    min_rows: int = 1500,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for segment, group in df.groupby(segment_col, dropna=False):
        group = group.dropna(subset=[target_col, pred_col])
        if len(group) < min_rows:
            continue
        metrics = evaluate_predictions(group[target_col].to_numpy(dtype=float), group[pred_col].to_numpy(dtype=float))
        rows.append(
            [
                str(segment),
                f"{len(group):,}",
                str(group["event_id"].nunique()),
                fmt(metrics["auc"]),
                fmt_pct(metrics["accuracy"]),
                fmt(metrics["brier"]),
                fmt(metrics["log_loss"]),
                fmt_pct(metrics["p1_rate"]),
            ]
        )
    rows.sort(key=lambda row: int(row[1].replace(",", "")), reverse=True)
    return rows


def train_target(
    df: pd.DataFrame,
    target_col: str,
    target_label: str,
    train_events: set[str],
    validation_events: set[str],
    test_events: set[str],
) -> tuple[dict[str, dict[str, float | None]], pd.DataFrame, str]:
    work = df.loc[df[target_col].notna()].copy()
    work["split"] = np.where(
        work["event_id"].isin(train_events),
        "train",
        np.where(work["event_id"].isin(validation_events), "validation", "test"),
    )

    hmm_params = build_hmm_parameters(work.loc[work["split"] == "train"])
    work = add_hmm_posteriors(work, hmm_params)
    hmm_columns = [column for column in work.columns if column.startswith("hmm_state_")]
    score_features = SCORE_FEATURES + SERVER_INTERACTION_FEATURES

    train_df = work.loc[work["split"] == "train"].copy()
    validation_df = work.loc[work["split"] == "validation"].copy()
    test_df = work.loc[work["split"] == "test"].copy()
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_validation = validation_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    feature_sets = {
        "Score + server only": score_features,
        "Raw metrics only": METRIC_COLUMNS,
        "HMM posterior only": hmm_columns,
        "Direct combined score + server + raw metrics + HMM": score_features + METRIC_COLUMNS + hmm_columns,
    }
    model_results: dict[str, dict[str, float | None]] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, columns in feature_sets.items():
        model = fit_logistic(feature_matrix(train_df, columns), y_train)
        validation_pred = predict_logistic(model, feature_matrix(validation_df, columns))
        test_pred = predict_logistic(model, feature_matrix(test_df, columns))
        validation_predictions[name] = validation_pred
        test_predictions[name] = test_pred
        test_df[f"pred_{name}"] = test_pred
        model_results[name] = evaluate_predictions(y_test, test_pred)

    stack_names = list(feature_sets)
    stack_model = fit_logistic(
        np.column_stack([logit(validation_predictions[name]) for name in stack_names]),
        y_validation,
        l2=0.10,
        learning_rate=0.05,
        epochs=500,
    )
    stacked_name = "Calibrated stacked ensemble"
    stacked_pred = predict_logistic(
        stack_model,
        np.column_stack([logit(test_predictions[name]) for name in stack_names]),
    )
    test_df[f"pred_{stacked_name}"] = stacked_pred
    model_results[stacked_name] = evaluate_predictions(y_test, stacked_pred)

    direct_name = "Direct combined score + server + raw metrics + HMM"
    direct_calibrator = fit_logistic(
        logit(validation_predictions[direct_name]).reshape(-1, 1),
        y_validation,
        l2=0.05,
        learning_rate=0.05,
        epochs=500,
    )
    calibrated_direct_name = "Calibrated direct combined"
    calibrated_direct = predict_logistic(
        direct_calibrator,
        logit(test_predictions[direct_name]).reshape(-1, 1),
    )
    test_df[f"pred_{calibrated_direct_name}"] = calibrated_direct
    model_results[calibrated_direct_name] = evaluate_predictions(y_test, calibrated_direct)

    best_name = max(model_results, key=lambda name: model_results[name]["auc"] or -math.inf)
    print(f"{target_label}:")
    for name, metrics in sorted(model_results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True):
        print(
            f"  {name}: AUC={fmt(metrics['auc'])} accuracy={fmt_pct(metrics['accuracy'])} "
            f"Brier={fmt(metrics['brier'])} log_loss={fmt(metrics['log_loss'])}"
        )
    return model_results, test_df, best_name


def model_rows(results: dict[str, dict[str, float | None]]) -> list[list[str]]:
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


def build_report(
    df: pd.DataFrame,
    next_game_results: dict[str, dict[str, float | None]],
    next_game_test: pd.DataFrame,
    next_game_best: str,
    next_set_results: dict[str, dict[str, float | None]],
    next_set_test: pd.DataFrame,
    next_set_best: str,
    split_counts: dict[str, int],
) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; position: sticky; top: 0; }
    .note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    """
    model_headers = ["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"]
    segment_headers = ["Segment", "Rows", "Matches", "AUC", "Accuracy", "Brier", "Log loss", "P1 rate"]
    next_game_pred = f"pred_{next_game_best}"
    next_set_pred = f"pred_{next_set_best}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Next Game and Next Set Probability Model Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Next Game and Next Set Probability Model Report</h1>
  <p class="muted">Targets: first future game winner and first future set winner from each current row. Source: {html.escape(str(DATA_DIR))}.</p>

  <div class="grid">
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Train matches</div><div class="stat">{split_counts['train']:,}</div></div>
    <div class="card"><div class="muted">Calibration matches</div><div class="stat">{split_counts['validation']:,}</div></div>
    <div class="card"><div class="muted">Test matches</div><div class="stat">{split_counts['test']:,}</div></div>
  </div>

  <div class="note">
    <b>Serving context:</b> all score models include <code>is_server_p1</code>, <code>is_server_p2</code>, server side,
    and interactions between serving side, point score, and live form. Base models train on 60% of matches,
    validation/calibration uses 20%, and the final 20% is held out for this report.
  </div>

  <h2>Next Game Winner: Model Comparison</h2>
  {html_table(model_headers, model_rows(next_game_results))}

  <h2>Next Set Winner: Model Comparison</h2>
  {html_table(model_headers, model_rows(next_set_results))}

  <h2>Next Game: Best Model by League</h2>
  {html_table(segment_headers, segment_results(next_game_test, "target_next_game_p1", next_game_pred, "league"))}

  <h2>Next Game: Best Model by Tier</h2>
  {html_table(segment_headers, segment_results(next_game_test, "target_next_game_p1", next_game_pred, "tier", min_rows=500))}

  <h2>Next Set: Best Model by League</h2>
  {html_table(segment_headers, segment_results(next_set_test, "target_next_set_p1", next_set_pred, "league"))}

  <h2>Next Set: Best Model by Tier</h2>
  {html_table(segment_headers, segment_results(next_set_test, "target_next_set_p1", next_set_pred, "tier", min_rows=500))}
</body>
</html>
"""


def main() -> int:
    df = load_training_rows()
    df = add_future_targets(df)
    df = add_server_interactions(df)
    train_events, validation_events, test_events = split_events(df)

    next_game_results, next_game_test, next_game_best = train_target(
        df,
        "target_next_game_p1",
        "Next game",
        train_events,
        validation_events,
        test_events,
    )
    next_set_results, next_set_test, next_set_best = train_target(
        df,
        "target_next_set_p1",
        "Next set",
        train_events,
        validation_events,
        test_events,
    )
    split_counts = {
        "train": len(train_events),
        "validation": len(validation_events),
        "test": len(test_events),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report(
            df,
            next_game_results,
            next_game_test,
            next_game_best,
            next_set_results,
            next_set_test,
            next_set_best,
            split_counts,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Best next-game model: {next_game_best}")
    print(f"Best next-set model: {next_set_best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
