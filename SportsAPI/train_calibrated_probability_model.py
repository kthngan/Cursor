#!/usr/bin/env python3
"""Train calibrated row-level match-end probability models from SportsAPI CSVs.

This excludes live odds for now. The target is the eventual match winner from
each match CSV. Models are evaluated with event-level time splits so rows from
the same match do not leak across train/test.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
REPORTS_DIR = WORKSPACE_DIR / "Reports"
OUTPUT_PATH = REPORTS_DIR / "calibrated_probability_model_report.html"
METADATA_PATH = DATA_DIR / "master_match_metadata.csv"

METRIC_COLUMNS = [
    "rolling_live_form_ratio",
    "rolling_points_ratio_20",
    "rolling_service_points_won_ratio_20",
    "rolling_return_points_won_ratio_20",
    "rolling_break_points_created_ratio_20",
    "rolling_break_points_won_ratio_20",
    "rolling_break_points_saved_ratio_20",
    "rolling_games_won_ratio_6",
    "live_form_delta_5",
]

HMM_OBSERVATION_COLUMNS = [
    "rolling_live_form_ratio",
    "rolling_points_ratio_20",
    "rolling_break_points_won_ratio_20",
    "rolling_games_won_ratio_6",
]

SCORE_FEATURES = [
    "set_diff",
    "game_diff",
    "point_diff",
    "is_server_p1",
    "is_server_p2",
    "sets_total",
    "games_total",
]

STATE_NAMES = [
    "P2 strong",
    "P2 edge",
    "Neutral",
    "P1 edge",
    "P1 strong",
]


@dataclass
class LogisticModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def parse_pair(text: object) -> tuple[float, float]:
    if not isinstance(text, str) or "-" not in text:
        return 0.0, 0.0
    left, right = text.split("-", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return 0.0, 0.0


def parse_point_score(text: object) -> tuple[float, float]:
    p1, p2 = parse_pair(text)
    # Current exporter stores point score as 0..4 internal point count.
    return p1, p2


def add_score_features(df: pd.DataFrame) -> pd.DataFrame:
    sets = df["sets_after"].map(parse_pair)
    games = df["game_score_after"].map(parse_pair)
    points = df["point_score_state"].map(parse_point_score)

    df["set_diff"] = [left - right for left, right in sets]
    df["sets_total"] = [left + right for left, right in sets]
    df["game_diff"] = [left - right for left, right in games]
    df["games_total"] = [left + right for left, right in games]
    df["point_diff"] = [left - right for left, right in points]
    df["is_server_p1"] = (df["server_side"] == "P1").astype(float)
    df["is_server_p2"] = (df["server_side"] == "P2").astype(float)
    return df


def read_match_csv(path: Path) -> pd.DataFrame | None:
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
    usable = df[METRIC_COLUMNS + SCORE_FEATURES].notna().any(axis=1)
    return df.loc[usable].copy()


def load_training_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name.startswith("_") or path.name in {"master_match_metadata.csv"}:
            continue
        frame = read_match_csv(path)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No usable match CSVs found in {DATA_DIR}")
    df = pd.concat(frames, ignore_index=True)

    if METADATA_PATH.exists():
        meta = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
        keep_cols = [
            "event_id",
            "competition_name",
            "season_name",
            "stage_name",
            "group_name",
            "event_start_date",
            "event_event_stats_lvl_live",
            "event_round_name",
        ]
        available = [column for column in keep_cols if column in meta.columns]
        df = df.merge(meta[available].drop_duplicates("event_id"), on="event_id", how="left")
    else:
        df["competition_name"] = "Unknown"
        df["event_event_stats_lvl_live"] = "Unknown"
        df["event_start_date"] = ""

    df["league"] = df.get("competition_name", "Unknown").fillna("Unknown").astype(str)
    df["tier"] = df.get("event_event_stats_lvl_live", "Unknown").fillna("Unknown").astype(str)
    if "stage_name" not in df.columns:
        df["stage_name"] = "Unknown"
    df["stage_name"] = df["stage_name"].fillna("Unknown").astype(str)
    df["event_start_date"] = pd.to_datetime(df.get("event_start_date", ""), errors="coerce")
    return df


def split_events(df: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    events = (
        df[["event_id", "event_start_date"]]
        .drop_duplicates("event_id")
        .sort_values(["event_start_date", "event_id"], na_position="last")
    )
    event_ids = events["event_id"].tolist()
    n = len(event_ids)
    train = set(event_ids[: int(n * 0.60)])
    validation = set(event_ids[int(n * 0.60) : int(n * 0.80)])
    test = set(event_ids[int(n * 0.80) :])
    return train, validation, test


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l2: float = 0.02,
    learning_rate: float = 0.08,
    epochs: int = 350,
) -> LogisticModel:
    mean_values = np.nanmean(x_train, axis=0)
    mean_values = np.where(np.isfinite(mean_values), mean_values, 0.0)
    filled = np.where(np.isnan(x_train), mean_values, x_train)
    std_values = filled.std(axis=0)
    std_values = np.where(std_values > 1e-8, std_values, 1.0)
    z = (filled - mean_values) / std_values
    x = np.column_stack([np.ones(len(z)), z])
    weights = np.zeros(x.shape[1])
    for _ in range(epochs):
        pred = sigmoid(x @ weights)
        grad = (x.T @ (pred - y_train)) / len(y_train)
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad
    return LogisticModel(mean_values, std_values, weights)


def predict_logistic(model: LogisticModel, x_values: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x_values), model.mean, x_values)
    z = (filled - model.mean) / model.std
    x = np.column_stack([np.ones(len(z)), z])
    return sigmoid(x @ model.weights)


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=float)


def build_hmm_parameters(train_df: pd.DataFrame) -> dict[str, np.ndarray]:
    work = train_df[["event_id", *HMM_OBSERVATION_COLUMNS]].copy()
    for column in HMM_OBSERVATION_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["rolling_live_form_ratio"])
    work["state"] = pd.cut(
        work["rolling_live_form_ratio"],
        bins=[-np.inf, 0.40, 0.48, 0.52, 0.60, np.inf],
        labels=False,
    ).astype(int)

    transition = np.full((5, 5), 0.5)
    for _, group in work.groupby("event_id", sort=False):
        states = group["state"].to_numpy()
        if len(states) < 2:
            continue
        for prev, cur in zip(states[:-1], states[1:]):
            transition[int(prev), int(cur)] += 1.0
    transition = transition / transition.sum(axis=1, keepdims=True)

    emissions_mean = np.zeros((5, len(HMM_OBSERVATION_COLUMNS)))
    emissions_std = np.ones((5, len(HMM_OBSERVATION_COLUMNS)))
    global_mean = work[HMM_OBSERVATION_COLUMNS].mean(numeric_only=True).fillna(0.5).to_numpy()
    global_std = work[HMM_OBSERVATION_COLUMNS].std(numeric_only=True).replace(0, np.nan).fillna(0.1).to_numpy()
    for state in range(5):
        state_rows = work.loc[work["state"] == state, HMM_OBSERVATION_COLUMNS]
        if state_rows.empty:
            emissions_mean[state] = global_mean
            emissions_std[state] = global_std
        else:
            emissions_mean[state] = state_rows.mean(numeric_only=True).fillna(pd.Series(global_mean)).to_numpy()
            emissions_std[state] = (
                state_rows.std(numeric_only=True).replace(0, np.nan).fillna(pd.Series(global_std)).to_numpy()
            )
    emissions_std = np.where(emissions_std > 1e-4, emissions_std, 0.1)
    start_probability = np.full(5, 1.0 / 5.0)
    return {
        "start": start_probability,
        "transition": transition,
        "mean": emissions_mean,
        "std": emissions_std,
    }


def emission_likelihood(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    likelihood = np.ones(mean.shape[0])
    for idx, value in enumerate(obs):
        if not np.isfinite(value):
            continue
        var = std[:, idx] ** 2
        likelihood *= np.exp(-0.5 * ((value - mean[:, idx]) ** 2) / var) / np.sqrt(2.0 * np.pi * var)
    likelihood = np.maximum(likelihood, 1e-300)
    return likelihood


def emission_likelihood_matrix(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    likelihood = np.ones((len(obs), mean.shape[0]))
    for idx in range(obs.shape[1]):
        values = obs[:, idx]
        mask = np.isfinite(values)
        if not mask.any():
            continue
        var = std[:, idx] ** 2
        diff = values[mask, None] - mean[:, idx][None, :]
        likelihood[mask] *= np.exp(-0.5 * (diff**2) / var[None, :]) / np.sqrt(2.0 * np.pi * var[None, :])
    return np.maximum(likelihood, 1e-300)


def add_hmm_posteriors(df: pd.DataFrame, params: dict[str, np.ndarray]) -> pd.DataFrame:
    posterior_columns = [f"hmm_state_{idx}_{name.lower().replace(' ', '_')}" for idx, name in enumerate(STATE_NAMES)]
    posterior = np.zeros((len(df), len(STATE_NAMES)), dtype=float)
    obs_matrix = df[HMM_OBSERVATION_COLUMNS].to_numpy(dtype=float)
    likelihood_matrix = emission_likelihood_matrix(obs_matrix, params["mean"], params["std"])
    for _, positions in df.groupby("event_id", sort=False).indices.items():
        alpha = params["start"].copy()
        for pos in positions:
            likelihood = likelihood_matrix[pos]
            alpha = (alpha @ params["transition"]) * likelihood
            total = alpha.sum()
            alpha = alpha / total if total > 0 else params["start"].copy()
            posterior[pos] = alpha
    posterior_frame = pd.DataFrame(posterior, columns=posterior_columns, index=df.index)
    return pd.concat([df, posterior_frame], axis=1)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    order = np.argsort(y_score)
    y = y_true[order]
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = np.empty(len(y), dtype=float)
    sorted_scores = y_score[order]
    rank = 1
    idx = 0
    while idx < len(y):
        end = idx + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[idx]:
            end += 1
        ranks[idx:end] = (rank + rank + end - idx - 1) / 2.0
        rank += end - idx
        idx = end
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return {
        "rows": float(len(y_true)),
        "p1_rate": float(y_true.mean()) if len(y_true) else None,
        "auc": auc_score(y_true, pred),
        "accuracy": float(((pred >= 0.5).astype(float) == y_true).mean()) if len(y_true) else None,
        "brier": float(np.mean((pred - y_true) ** 2)) if len(y_true) else None,
        "rmse": float(math.sqrt(np.mean((pred - y_true) ** 2))) if len(y_true) else None,
        "log_loss": float(-np.mean(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))) if len(y_true) else None,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def segment_results(df: pd.DataFrame, pred_col: str, segment_col: str, min_rows: int = 1500) -> list[list[str]]:
    rows: list[list[str]] = []
    for segment, group in df.groupby(segment_col, dropna=False):
        if len(group) < min_rows:
            continue
        metrics = evaluate_predictions(group["target_p1_win"].to_numpy(dtype=float), group[pred_col].to_numpy(dtype=float))
        event_count = group["event_id"].nunique()
        rows.append(
            [
                str(segment),
                f"{len(group):,}",
                str(event_count),
                fmt(metrics["auc"]),
                fmt_pct(metrics["accuracy"]),
                fmt(metrics["brier"]),
                fmt(metrics["log_loss"]),
                fmt_pct(metrics["p1_rate"]),
            ]
        )
    rows.sort(key=lambda row: int(row[1].replace(",", "")), reverse=True)
    return rows


def build_report(
    df: pd.DataFrame,
    model_results: dict[str, dict[str, float | None]],
    league_rows: list[list[str]],
    tier_rows: list[list[str]],
    stage_rows: list[list[str]],
    split_counts: dict[str, int],
) -> str:
    model_rows = [
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
        for name, metrics in sorted(model_results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True)
    ]
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
    segment_headers = ["Segment", "Rows", "Matches", "AUC", "Accuracy", "Brier", "Log loss", "P1 win rate"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Calibrated SportsAPI Probability Model Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Calibrated SportsAPI Probability Model Report</h1>
  <p class="muted">Target: eventual match winner. Live odds are excluded. Report generated from {html.escape(str(DATA_DIR))}.</p>

  <div class="grid">
    <div class="card"><div class="muted">Rows used</div><div class="stat">{len(df):,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Train matches</div><div class="stat">{split_counts['train']:,}</div></div>
    <div class="card"><div class="muted">Test matches</div><div class="stat">{split_counts['test']:,}</div></div>
  </div>

  <div class="note">
    <b>Calibration note:</b> Base models train on the first 60% of matches only. The next 20% is used only to calibrate/stack model probabilities.
    The final 20% is held out for this report. Calibration is against final match outcome, not live market odds. Metrics are row-weighted.
  </div>

  <h2>Model Comparison on Test Split</h2>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 win rate"], model_rows)}

  <h2>Combined Model by League</h2>
  {html_table(segment_headers, league_rows)}

  <h2>Combined Model by Tier</h2>
  {html_table(segment_headers, tier_rows)}

  <h2>Combined Model by Stage</h2>
  {html_table(segment_headers, stage_rows)}

  <h2>Feature Sets</h2>
  <ul>
    <li><b>Score-only:</b> set diff, game diff, point diff, server side, totals.</li>
    <li><b>Raw metrics:</b> rolling points, service, return, break-created, break-won, break-saved, games-won, live-form, live-form delta.</li>
    <li><b>HMM posterior:</b> five hidden momentum-state posterior probabilities estimated from rolling metric emissions and transitions.</li>
    <li><b>Direct combined:</b> one logistic model over score, raw metrics, and HMM posterior features.</li>
    <li><b>Calibrated stacked ensemble:</b> base model probabilities are learned on validation data, then combined by a second logistic calibration model.</li>
  </ul>
</body>
</html>
"""


def main() -> int:
    df = load_training_rows()
    train_events, validation_events, test_events = split_events(df)
    df["split"] = np.where(
        df["event_id"].isin(train_events),
        "train",
        np.where(df["event_id"].isin(validation_events), "validation", "test"),
    )

    hmm_params = build_hmm_parameters(df.loc[df["split"] == "train"])
    df = add_hmm_posteriors(df, hmm_params)
    hmm_columns = [column for column in df.columns if column.startswith("hmm_state_")]

    train_df = df.loc[df["split"] == "train"].copy()
    validation_df = df.loc[df["split"] == "validation"].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    y_train = train_df["target_p1_win"].to_numpy(dtype=float)
    y_validation = validation_df["target_p1_win"].to_numpy(dtype=float)
    y_test = test_df["target_p1_win"].to_numpy(dtype=float)

    feature_sets = {
        "Score only": SCORE_FEATURES,
        "Raw metrics only": METRIC_COLUMNS,
        "HMM posterior only": hmm_columns,
        "Direct combined score + raw metrics + HMM": SCORE_FEATURES + METRIC_COLUMNS + hmm_columns,
    }
    model_results: dict[str, dict[str, float | None]] = {}
    validation_base_predictions: dict[str, np.ndarray] = {}
    test_base_predictions: dict[str, np.ndarray] = {}
    for name, columns in feature_sets.items():
        model = fit_logistic(feature_matrix(train_df, columns), y_train)
        validation_pred = predict_logistic(model, feature_matrix(validation_df, columns))
        test_pred = predict_logistic(model, feature_matrix(test_df, columns))
        validation_base_predictions[name] = validation_pred
        test_base_predictions[name] = test_pred
        test_df[f"pred_{name}"] = test_pred
        model_results[name] = evaluate_predictions(y_test, test_pred)

    stack_model_names = [
        "Score only",
        "Raw metrics only",
        "HMM posterior only",
        "Direct combined score + raw metrics + HMM",
    ]
    stack_x_validation = np.column_stack(
        [logit(validation_base_predictions[name]) for name in stack_model_names]
    )
    stack_x_test = np.column_stack([logit(test_base_predictions[name]) for name in stack_model_names])
    stack_model = fit_logistic(stack_x_validation, y_validation, l2=0.10, learning_rate=0.05, epochs=500)
    stacked_pred = predict_logistic(stack_model, stack_x_test)
    stacked_name = "Calibrated stacked ensemble"
    test_df[f"pred_{stacked_name}"] = stacked_pred
    model_results[stacked_name] = evaluate_predictions(y_test, stacked_pred)

    direct_name = "Direct combined score + raw metrics + HMM"
    direct_calibration_x_validation = logit(validation_base_predictions[direct_name]).reshape(-1, 1)
    direct_calibration_x_test = logit(test_base_predictions[direct_name]).reshape(-1, 1)
    direct_calibrator = fit_logistic(
        direct_calibration_x_validation,
        y_validation,
        l2=0.05,
        learning_rate=0.05,
        epochs=500,
    )
    calibrated_direct_name = "Calibrated direct combined"
    calibrated_direct_pred = predict_logistic(direct_calibrator, direct_calibration_x_test)
    test_df[f"pred_{calibrated_direct_name}"] = calibrated_direct_pred
    model_results[calibrated_direct_name] = evaluate_predictions(y_test, calibrated_direct_pred)

    pred_col = f"pred_{stacked_name}"
    league_rows = segment_results(test_df, pred_col, "league")
    tier_rows = segment_results(test_df, pred_col, "tier", min_rows=500)
    stage_rows = segment_results(test_df, pred_col, "stage_name")

    split_counts = {
        "train": len(train_events),
        "validation": len(validation_events),
        "test": len(test_events),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report(test_df, model_results, league_rows, tier_rows, stage_rows, split_counts),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_PATH}")
    for name, metrics in sorted(model_results.items(), key=lambda item: (item[1]["auc"] or -1), reverse=True):
        print(
            f"{name}: AUC={fmt(metrics['auc'])} accuracy={fmt_pct(metrics['accuracy'])} "
            f"Brier={fmt(metrics['brier'])} log_loss={fmt(metrics['log_loss'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
