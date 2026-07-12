#!/usr/bin/env python3
"""Compare odds-only, factors-only, and factors+odds models for match result.

Evaluates whether in-game factor features add predictive value on top of
bookmaker implied probability. Also investigates how accuracy degrades when
the odds snapshot timestamp is far from the event row timestamp.
"""

from __future__ import annotations

import html
import math
import sys
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
    METADATA_PATH,
    METRIC_COLUMNS,
    REPORTS_DIR,
    SCORE_FEATURES,
    LogisticModel,
    add_score_features,
    auc_score,
    evaluate_predictions,
    fit_logistic,
    fmt,
    fmt_pct,
    html_table,
    logit,
    predict_logistic,
    sigmoid,
    split_events,
)
from train_xgboost_factor_models import (  # noqa: E402
    add_factor_features,
    fill_matrix,
    logistic_stats,
    shap_bar_chart,
    shap_rows,
    shap_summary,
    train_xgboost,
)

OUTPUT_PATH = REPORTS_DIR / "odds_vs_factors_match_result_report.html"
TARGET_COL = "target_p1_win"
WITH_ODDS_DIR = DATA_DIR / "with_odds"

ODDS_FEATURES = [
    "odds_no_vig_p1",
    "odds_no_vig_p2",
    "odds_implied_p1",
    "odds_implied_p2",
    "odds_bookmaker_count",
]


# ---------------------------------------------------------------------------
# Data loading from enriched CSVs
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

    # Parse odds columns
    for col in ["odds_no_vig_p1", "odds_no_vig_p2", "odds_implied_p1", "odds_implied_p2",
                "odds_p1_price", "odds_p2_price", "odds_bookmaker_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute odds snapshot age (seconds between row time and odds snapshot time)
    if "ut" in df.columns and "odds_snapshot_time" in df.columns:
        df["row_dt"] = pd.to_datetime(df["ut"], unit="s", utc=True, errors="coerce")
        df["snap_dt"] = pd.to_datetime(df["odds_snapshot_time"], utc=True, errors="coerce")
        df["odds_age_seconds"] = (df["row_dt"] - df["snap_dt"]).dt.total_seconds()
    else:
        df["odds_age_seconds"] = np.nan

    df["has_odds"] = df["odds_no_vig_p1"].notna().astype(float)
    df["odds_logit_p1"] = df["odds_no_vig_p1"].apply(
        lambda x: logit(np.array([x]))[0] if pd.notna(x) and 0 < x < 1 else np.nan
    )

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
# Split and train
# ---------------------------------------------------------------------------

def split_frames(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_events, validation_events, test_events = split_events(df)
    work = df.loc[df[target_col].notna()].copy()
    train = work.loc[work["event_id"].isin(train_events)].copy()
    validation = work.loc[work["event_id"].isin(validation_events)].copy()
    test = work.loc[work["event_id"].isin(test_events)].copy()
    return train, validation, test


def fill_features(train: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    means = train[features].mean(numeric_only=True).fillna(0.0)
    return frame[features].fillna(means).to_numpy(dtype=float)


def train_and_evaluate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    use_xgb: bool = True,
) -> tuple[dict[str, float | None], xgb.Booster | None, np.ndarray]:
    y_train = train[TARGET_COL].to_numpy(dtype=float)
    y_test = test[TARGET_COL].to_numpy(dtype=float)

    logistic = fit_logistic(
        fill_features(train, train, features), y_train,
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    logistic_pred = predict_logistic(logistic, fill_features(train, test, features))

    booster = None
    xgb_pred = None
    if use_xgb:
        booster, xgb_pred = train_xgboost(train, validation, test, TARGET_COL, features)

    results = {
        "Logistic": evaluate_predictions(y_test, logistic_pred),
    }
    if xgb_pred is not None:
        results["XGBoost"] = evaluate_predictions(y_test, xgb_pred)

    return results, booster, xgb_pred if xgb_pred is not None else logistic_pred


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

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


def build_report(
    df: pd.DataFrame,
    factor_features: list[str],
    factor_groups: dict[str, list[str]],
    odds_features: list[str],
    combined_features: list[str],
    results: dict[str, dict[str, dict]],
    shap_combined: list[dict],
    stats_combined: list,
    segment_rows: list[dict],
    models: list[str],
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
    .bar-row { display: grid; grid-template-columns: 280px 1fr 70px; gap: 10px; align-items: center; margin: 6px 0; font-size: 13px; }
    .bar-track { height: 12px; background: #f6f8fa; border: 1px solid #d0d7de; }
    .bar-fill { height: 100%; background: #57606a; }
    .bar-label { overflow-wrap: anywhere; }
    .delta-pos { color: #1a7f37; font-weight: 600; }
    .delta-neg { color: #cf222e; font-weight: 600; }
    """

    has_odds = df["has_odds"] == 1
    odds_coverage = has_odds.mean()

    # Build delta table: factors+odds vs odds-only
    delta_rows = []
    for model_type in ["Logistic", "XGBoost"]:
        odds_metrics = results["odds_only"].get(model_type, {})
        combined_metrics = results["factors_plus_odds"].get(model_type, {})
        factors_metrics = results["factors_only"].get(model_type, {})
        for metric in ["auc", "brier", "log_loss"]:
            o = odds_metrics.get(metric)
            c = combined_metrics.get(metric)
            f = factors_metrics.get(metric)
            if o is not None and c is not None:
                delta = c - o
                if metric == "auc":
                    better = delta > 0
                else:
                    better = delta < 0
                cls = "delta-pos" if better else "delta-neg"
                delta_str = f"<span class='{cls}'>{delta:+.4f}</span>"
            else:
                delta_str = ""
            delta_rows.append([
                model_type, metric.upper(),
                fmt(o), fmt(f), fmt(c), delta_str,
            ])

    factor_group_rows = [
        [group, str(len([f for f in vals if f in factor_features])),
         ", ".join(f for f in vals if f in factor_features)]
        for group, vals in factor_groups.items()
    ]

    staleness_headers = ["Staleness", "Rows"]
    for model in models:
        staleness_headers.extend([f"{model} AUC", f"{model} Brier", f"{model} LogLoss"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Odds vs Factors Match Result Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Odds vs Factors Match Result Report (0-1 min staleness only)</h1>
  <p class="muted">Target: final match winner from each row. Only rows where the odds snapshot is 0-60 seconds
  before the event row are used, ensuring odds are as fresh as possible. Compares bookmaker implied probability alone,
  factor features alone, and factors + odds combined. Uses 60/20/20 event-level time split.</p>

  <div class="grid">
    <div class="card"><div class="muted">Rows (0-1m)</div><div class="stat">{len(df):,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{df['event_id'].nunique():,}</div></div>
    <div class="card"><div class="muted">Factor features</div><div class="stat">{len(factor_features)}</div></div>
    <div class="card"><div class="muted">Combined features</div><div class="stat">{len(combined_features)}</div></div>
  </div>

  <div class="note">
    <b>Models compared (all rows have fresh odds, 0-60s staleness):</b>
    <ul>
      <li><b>Odds only</b> — uses bookmaker no-vig implied probability ({', '.join(odds_features)})</li>
      <li><b>Factors only</b> — uses score state, server context, rolling metrics and derivatives ({len(factor_features)} features, no odds)</li>
      <li><b>Factors + Odds</b> — all factor features plus odds features ({len(combined_features)} features)</li>
    </ul>
    Both logistic regression and XGBoost are trained for each model.
  </div>

  <h2>Model Comparison: Does Adding Factors to Odds Improve Prediction?</h2>
  <p>The key table. Positive delta in AUC (or negative in Brier/LogLoss) means factors add value on top of odds.</p>
  {html_table(["Model", "Metric", "Odds only", "Factors only", "Factors+Odds", "Delta (F+O - Odds)"], delta_rows)}

  <h2>Detailed Results</h2>

  <h3>Odds Only</h3>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"],
    model_comparison_rows(results["odds_only"]))}

  <h3>Factors Only</h3>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"],
    model_comparison_rows(results["factors_only"]))}

  <h3>Factors + Odds</h3>
  {html_table(["Model", "Rows", "AUC", "Accuracy", "Brier", "RMSE", "Log loss", "P1 rate"],
    model_comparison_rows(results["factors_plus_odds"]))}

  <h2>Factors + Odds: SHAP Summary (XGBoost)</h2>
  {shap_bar_chart(shap_combined, "Factors + Odds XGBoost SHAP Summary")}
  <h3>Top SHAP Features</h3>
  {html_table(["Feature", "Mean |SHAP|", "Mean SHAP", "Feature/SHAP corr"], shap_rows(shap_combined))}

  <h3>Factors + Odds: Logistic Regression T-Stats and P-Values</h3>
  {html_table(["Feature", "Coefficient", "Std error", "T/Z stat", "P-value"],
    [[r.feature, fmt(r.coefficient, 4), fmt(r.std_error, 4), fmt(r.z_stat, 2),
      f"{r.p_value:.3g}" if r.p_value is not None else ""]
     for r in sorted(stats_combined, key=lambda x: abs(x.z_stat or 0), reverse=True)[:60]])}

  <h2>Factor Groups Tested</h2>
  {html_table(["Group", "Feature count", "Features"], factor_group_rows)}

  <h2>Segment Analysis: Outperformance by League, Tier, and Match Progression</h2>
  <div class="warn">
    When odds are fresh (0-1 min), do factors still add value? This table breaks down AUC and Brier
    by segment. If combined consistently beats odds-only, factors capture information beyond what
    bookmakers have already priced in.
  </div>
  {html_table(["Segment", "Rows", "Odds AUC", "Factors AUC", "Combined AUC", "Odds Brier", "Combined Brier"],
    [[r["segment"], f"{r['rows']:,}",
      fmt(r.get("odds_only_auc")), fmt(r.get("factors_only_auc")), fmt(r.get("combined_auc")),
      fmt(r.get("odds_only_brier")), fmt(r.get("combined_brier"))]
     for r in segment_rows])}

</body>
</html>
"""


from typing import Any  # noqa: E402


def main() -> int:
    print("=== Odds vs Factors Match Result Report ===")
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

    # Step 3: Define feature sets
    odds_features = ["odds_no_vig_p1", "odds_logit_p1", "odds_bookmaker_count"]

    combined_features = factor_features + odds_features

    # Step 4: Filter to 0-1 min staleness only
    print("3. Filtering to 0-1 min staleness rows only...")
    before = len(df)
    df = df[df["odds_age_seconds"].notna() & (df["odds_age_seconds"] >= 0) & (df["odds_age_seconds"] < 60)].copy()
    after = len(df)
    print(f"   {before:,} -> {after:,} rows ({df['event_id'].nunique()} matches)")
    # Fill any remaining NaN odds (shouldn't be any after filter, but safety)
    df["odds_no_vig_p1"] = df["odds_no_vig_p1"].fillna(0.5)
    df["odds_logit_p1"] = df["odds_logit_p1"].fillna(0.0)
    df["odds_bookmaker_count"] = df["odds_bookmaker_count"].fillna(0.0)
    print()

    # Step 5: Split data
    print("4. Splitting train/validation/test (60/20/20 by event)...")
    train, validation, test = split_frames(df, TARGET_COL)
    print(f"   Train: {len(train):,} rows ({train['event_id'].nunique()} matches)")
    print(f"   Validation: {len(validation):,} rows ({validation['event_id'].nunique()} matches)")
    print(f"   Test: {len(test):,} rows ({test['event_id'].nunique()} matches)")
    print()

    # Step 6: Train and evaluate models
    print("5. Training models...")

    print("   Odds only...")
    results_odds, _, pred_odds = train_and_evaluate(train, validation, test, odds_features)
    for name, m in results_odds.items():
        print(f"     {name}: AUC={fmt(m['auc'])} Brier={fmt(m['brier'])} LogLoss={fmt(m['log_loss'])}")

    print("   Factors only...")
    results_factors, _, pred_factors = train_and_evaluate(train, validation, test, factor_features)
    for name, m in results_factors.items():
        print(f"     {name}: AUC={fmt(m['auc'])} Brier={fmt(m['brier'])} LogLoss={fmt(m['log_loss'])}")

    print("   Factors + Odds...")
    results_combined, booster_combined, pred_combined = train_and_evaluate(
        train, validation, test, combined_features,
    )
    for name, m in results_combined.items():
        print(f"     {name}: AUC={fmt(m['auc'])} Brier={fmt(m['brier'])} LogLoss={fmt(m['log_loss'])}")
    print()

    # Step 7: SHAP and stats for combined model
    print("6. Computing SHAP and regression stats for combined model...")
    shap = shap_summary(booster_combined, train, test, combined_features)

    logistic_combined = fit_logistic(
        fill_features(train, train, combined_features),
        train[TARGET_COL].to_numpy(dtype=float),
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    stats = logistic_stats(logistic_combined, train, TARGET_COL, combined_features)
    print()

    # Step 8: Segment analysis by league, tier, and match progression
    print("7. Segment analysis...")
    test_with_preds = test.copy()
    test_with_preds["pred_odds_only"] = pred_odds
    test_with_preds["pred_factors_only"] = pred_factors
    test_with_preds["pred_combined"] = pred_combined

    segment_rows = []
    # By league
    for seg_col, seg_name in [("league", "League"), ("tier", "Tier")]:
        for segment, group in test_with_preds.groupby(seg_col, dropna=False):
            if len(group) < 200:
                continue
            y = group[TARGET_COL].to_numpy(dtype=float)
            row = {"segment": f"{seg_name}: {segment}", "rows": len(group)}
            for model_name, pred_col in [("odds_only", "pred_odds_only"),
                                         ("factors_only", "pred_factors_only"),
                                         ("combined", "pred_combined")]:
                m = evaluate_predictions(y, group[pred_col].to_numpy(dtype=float))
                row[f"{model_name}_auc"] = m["auc"]
                row[f"{model_name}_brier"] = m["brier"]
            segment_rows.append(row)

    # By match progression (early vs late based on sets_total)
    for label, mask in [("Early (sets_total<=1)", test_with_preds["sets_total"] <= 1),
                        ("Mid (sets_total==2)", test_with_preds["sets_total"] == 2),
                        ("Late (sets_total>=3)", test_with_preds["sets_total"] >= 3)]:
        group = test_with_preds[mask]
        if len(group) < 200:
            continue
        y = group[TARGET_COL].to_numpy(dtype=float)
        row = {"segment": f"Progression: {label}", "rows": len(group)}
        for model_name, pred_col in [("odds_only", "pred_odds_only"),
                                     ("factors_only", "pred_factors_only"),
                                     ("combined", "pred_combined")]:
            m = evaluate_predictions(y, group[pred_col].to_numpy(dtype=float))
            row[f"{model_name}_auc"] = m["auc"]
            row[f"{model_name}_brier"] = m["brier"]
        segment_rows.append(row)

    for r in segment_rows:
        print(f"   {r['segment']}: {r['rows']} rows, "
              f"odds_auc={fmt(r.get('odds_only_auc'))}, "
              f"combined_auc={fmt(r.get('combined_auc'))}")
    print()

    # Step 9: Build report
    print("8. Building HTML report...")
    results = {
        "odds_only": results_odds,
        "factors_only": results_factors,
        "factors_plus_odds": results_combined,
    }
    models_list = ["odds_only", "factors_only", "combined"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report(
            df, factor_features, factor_groups, odds_features, combined_features,
            results, shap, stats, segment_rows, models_list,
        ),
        encoding="utf-8",
    )
    print(f"   Report: {OUTPUT_PATH}")
    print()

    # Summary
    print("=== Summary ===")
    for model_name, res in results.items():
        for mt, m in res.items():
            print(f"  {model_name} ({mt}): AUC={fmt(m['auc'])} Brier={fmt(m['brier'])} LogLoss={fmt(m['log_loss'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
