#!/usr/bin/env python3
"""Calibrate an additive adjustment on top of de-vig odds probability.

Model:
    adjusted_prob = sigmoid(logit(odds_no_vig_p1) + clip(adjustment, -0.06, 0.06))

The adjustment is calibrated from factor features using logistic regression
with the odds logit as a fixed offset (coefficient = 1, not learned). The raw
adjustment is hard-capped at ±0.06 in logit space during beta calibration.

Compare:
  - Baseline: sigmoid(odds_logit)  = raw de-vig probability (no adjustment)
  - Adjusted: sigmoid(odds_logit + factor_adjustment)
  - Factors-only: sigmoid(factor_adjustment) (no odds prior, for reference)
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
)

OUTPUT_PATH = REPORTS_DIR / "adjusted_probability_report.html"
TARGET_COL = "target_p1_win"
WITH_ODDS_DIR = DATA_DIR / "with_pinnacle_odds"
ADJUSTMENT_CAP = 0.06  # max |adjustment| in logit space during calibration and prediction


# ---------------------------------------------------------------------------
# Logistic regression with offset
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
    cap: float = ADJUSTMENT_CAP,
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
    cap: float = ADJUSTMENT_CAP,
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
    adjustment = np.clip(z @ model.weights, -ADJUSTMENT_CAP, ADJUSTMENT_CAP)
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

STALENESS_BINS = [0, 60, 300, 600, 1800, 3600, float("inf")]
STALENESS_LABELS = ["0-1m", "1-5m", "5-10m", "10-30m", "30-60m", "60m+"]


def segment_analysis(
    test: pd.DataFrame,
    baseline_pred: np.ndarray,
    adjusted_pred: np.ndarray,
    factors_only_pred: np.ndarray,
) -> list[dict[str, Any]]:
    df = test.copy()
    df["pred_baseline"] = baseline_pred
    df["pred_adjusted"] = adjusted_pred
    df["pred_factors_only"] = factors_only_pred

    results = []

    # By staleness
    df["staleness_bin"] = pd.cut(
        df["odds_age_seconds"].fillna(-1),
        bins=[-2] + STALENESS_BINS,
        labels=["no_odds"] + STALENESS_LABELS,
        right=False,
    )
    for label, group in df.groupby("staleness_bin", observed=True):
        y = group[TARGET_COL].to_numpy(dtype=float)
        if len(y) < 100:
            continue
        row = {"segment": f"Staleness: {label}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
            row[f"{name}_log_loss"] = m["log_loss"]
        results.append(row)

    # By match progression
    for label, mask in [("Early (sets_total<=1)", df["sets_total"] <= 1),
                        ("Mid (sets_total==2)", df["sets_total"] == 2),
                        ("Late (sets_total>=3)", df["sets_total"] >= 3)]:
        group = df[mask]
        if len(group) < 100:
            continue
        y = group[TARGET_COL].to_numpy(dtype=float)
        row = {"segment": f"Progression: {label}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
            row[f"{name}_log_loss"] = m["log_loss"]
        results.append(row)

    # By league
    for league, group in df.groupby("league", dropna=False):
        if len(group) < 500:
            continue
        y = group[TARGET_COL].to_numpy(dtype=float)
        row = {"segment": f"League: {league}", "rows": len(group)}
        for name, col in [("baseline", "pred_baseline"), ("adjusted", "pred_adjusted"), ("factors_only", "pred_factors_only")]:
            m = evaluate_predictions(y, group[col].to_numpy(dtype=float))
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_brier"] = m["brier"]
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    test: pd.DataFrame,
    baseline_pred: np.ndarray,
    adjusted_pred: np.ndarray,
    threshold: float = 0.01,
) -> dict[str, Any]:
    """Backtest two-sided strategy at de-vig prices.

    - Buy P1 when adjusted_prob > devig_p1 + threshold (price = devig_p1)
    - Buy P2 when adjusted_prob < devig_p1 - threshold (price = devig_p2)
    """
    df = test.copy()
    df["devig_p1"] = baseline_pred
    if "odds_no_vig_p2" in df.columns:
        df["devig_p2"] = pd.to_numeric(df["odds_no_vig_p2"], errors="coerce").fillna(1.0 - df["devig_p1"])
    else:
        df["devig_p2"] = 1.0 - df["devig_p1"]
    df["adjusted_prob"] = adjusted_pred

    df = df.sort_values(
        ["event_start_date", "seq" if "seq" in df.columns else "event_id"],
        na_position="last",
    ).reset_index(drop=True)

    p1_signal = df["adjusted_prob"] > df["devig_p1"] + threshold
    p2_signal = df["adjusted_prob"] < df["devig_p1"] - threshold

    df["side"] = np.where(p1_signal, "P1", np.where(p2_signal, "P2", ""))
    df["bet_cost"] = np.where(p1_signal, df["devig_p1"], np.where(p2_signal, df["devig_p2"], 0.0))
    p2_wins = 1.0 - df[TARGET_COL]
    df["payout"] = np.where(p1_signal, df[TARGET_COL], np.where(p2_signal, p2_wins, 0.0))
    df["pnl"] = df["payout"] - df["bet_cost"]
    df["cum_pnl"] = df["pnl"].cumsum()
    df["cum_wagered"] = df["bet_cost"].cumsum()

    bets = df.loc[df["side"] != ""]
    p1_bets = df.loc[p1_signal]
    p2_bets = df.loc[p2_signal]
    total_wagered = float(bets["bet_cost"].sum())
    total_pnl = float(bets["pnl"].sum())
    win_rate = float(bets["payout"].mean()) if len(bets) > 0 else 0.0
    roi = total_pnl / total_wagered if total_wagered > 0 else 0.0

    edges: list[float] = []
    if len(p1_bets) > 0:
        edges.extend((p1_bets["adjusted_prob"] - p1_bets["devig_p1"]).tolist())
    if len(p2_bets) > 0:
        edges.extend(((1.0 - p2_bets["adjusted_prob"]) - p2_bets["devig_p2"]).tolist())
    avg_edge = float(np.mean(edges)) if edges else 0.0

    return {
        "df": df,
        "n_bets": int(len(bets)),
        "n_p1_bets": int(len(p1_bets)),
        "n_p2_bets": int(len(p2_bets)),
        "n_rows": len(df),
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "roi": roi,
        "avg_edge": avg_edge,
        "cum_pnl_series": df["cum_pnl"].to_numpy(),
        "cum_wagered_series": df["cum_wagered"].to_numpy(),
    }


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
  <text x="{margin_l + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#57606a">Bet number (chronological)</text>
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
  <text x="{margin_l + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#57606a">Row index (chronological)</text>
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
    fresh_html = html_table(["Model", "Rows", "AUC", "Brier", "Log loss"], fresh_rows)
    seg_html = html_table(["Segment", "Rows", "Base AUC", "Adj AUC", "Fact AUC", "Base Brier", "Adj Brier"], seg_rows)

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
                f"{thresh:.3f}",
                f"{bt['n_bets']:,}",
                f"{bt.get('n_p1_bets', 0):,}",
                f"{bt.get('n_p2_bets', 0):,}",
                f"{bt['total_wagered']:.1f}",
                f"{bt['total_pnl']:+.2f}",
                f"{bt['win_rate']:.1%}",
                f"<span class='{roi_cls}'>{bt['roi']:+.2%}</span>",
                f"{bt['avg_edge']:.4f}",
            ])
        thresh_html = html_table(
            ["Threshold", "Bets", "P1 bets", "P2 bets", "Wagered", "Total PnL", "Win rate", "ROI", "Avg edge"],
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
  <p class="muted">Calibrates an additive adjustment on top of Pinnacle de-vig probability using
  logistic regression. Only rows with Pinnacle odds and staleness &le; 5 min. 60/40 event-level time split.
  Max 2 factors per group. Adjustment capped at &plusmn;{ADJUSTMENT_CAP:.2f} in logit space.</p>

  <div class="formula">
    adjusted_prob = sigmoid( logit(odds_no_vig_p1) + clip(adjustment, -{ADJUSTMENT_CAP:.2f}, {ADJUSTMENT_CAP:.2f}) )<br>
    adjustment = standardized_factors &middot; beta  &nbsp;&nbsp;(logistic regression, odds logit as fixed offset)<br>
    baseline_prob = sigmoid( logit(odds_no_vig_p1) )  &nbsp;&nbsp;(no adjustment)<br>
    <br>
    odds source: <b>Pinnacle only</b>  &nbsp;&nbsp;|&nbsp;&nbsp; max 2 factors per group  &nbsp;&nbsp;|&nbsp;&nbsp; staleness &le; 5 min
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
    <p>The <b>odds_logit_p1</b> used as the offset is then: logit(no_vig_p1) = ln(0.5385 / 0.4615) = 0.1552</p>
  </div>

  <div class="grid">
    <div class="card"><div class="muted">Rows with odds</div><div class="stat">{n_rows:,}</div></div>
    <div class="card"><div class="muted">Matches</div><div class="stat">{n_matches:,}</div></div>
    <div class="card"><div class="muted">Factor features</div><div class="stat">{n_features}</div></div>
    <div class="card"><div class="muted">Significant (p&lt;0.05)</div><div class="stat">{n_sig}</div></div>
  </div>

  <div class="note">
    <b>Models compared:</b>
    <ul>
      <li><b>Baseline</b> &mdash; raw Pinnacle de-vig implied probability (no factors)</li>
      <li><b>Adjusted</b> &mdash; Pinnacle de-vig probability + scaled factor adjustment (logistic regression with offset)</li>
      <li><b>Factors only</b> &mdash; logistic regression on selected factors alone, no odds prior (for reference)</li>
    </ul>
    The adjustment operates in <b>logit space</b>, hard-capped at &plusmn;{ADJUSTMENT_CAP:.2f} during calibration and prediction.
    The odds logit is a fixed offset (coefficient = 1, not learned); only factor coefficients are calibrated.
    Factor selection: top 2 per group by |z-stat| from first-pass full model.
  </div>

  <h2>Overall Comparison: Adjusted vs Baseline</h2>
  <p>Positive delta in AUC (or negative in Brier/LogLoss) means the factor adjustment improves on raw odds.</p>
  {delta_html}

  <h2>Train / Test Split</h2>
  <div class="note">
    <p>Events (matches) are sorted chronologically by <code>event_start_date</code>, then split:</p>
    <ul>
      <li><b>Train (60%)</b> &mdash; earliest matches; used to calibrate factor coefficients</li>
      <li><b>Test (40%)</b> &mdash; all remaining matches; used for evaluation and backtest (no validation period)</li>
    </ul>
    <p>This <b>time-based event-level split</b> prevents data leakage: no match has rows in more than one split,
    and the model is never trained on data from a later date than its test data.</p>
  </div>
  {split_html}

  <h2>Detailed Results (Pinnacle odds, staleness &le; 5 min)</h2>
  {overall_html}

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
      <li><b>Buy P1</b> if <code>adjusted_prob &gt; devig_p1 + threshold</code> at price = <code>devig_p1</code></li>
      <li><b>Buy P2</b> if <code>adjusted_prob &lt; devig_p1 - threshold</code> at price = <code>devig_p2</code></li>
      <li>No bet if neither condition holds</li>
    </ul>
    <ul>
      <li>P1 win: payout = 1, profit = 1 - devig_p1</li>
      <li>P1 lose: payout = 0, profit = -devig_p1</li>
      <li>P2 win: payout = 1, profit = 1 - devig_p2</li>
      <li>P2 lose: payout = 0, profit = -devig_p2</li>
    </ul>
    <p>Each bet is for 1 unit. Tested across thresholds from 0.005 to 0.05.</p>
  </div>

  <h3>Threshold Comparison</h3>
  {thresh_html}

  <h3>Cumulative PnL by Threshold</h3>
  {thresh_chart}

  <h2>Segment Analysis</h2>
  <p>AUC by staleness, match progression, and league. If adjusted consistently beats baseline,
  factors add value on top of odds.</p>
  {seg_html}

  <h2>Significant Factor Adjustments (p &lt; 0.05)</h2>
  <p>Factors with statistically significant coefficients in the adjustment model.
  Positive coefficient means the factor pushes probability toward P1 winning (above what odds imply).</p>
  {sig_html}

  <h2>All Factor Coefficients</h2>
  {all_stats_html}

  <h2>Factor Groups</h2>
  {factor_groups_html}

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

    # Step 3: Filter to rows with odds and <=5 min staleness
    print("3. Filtering to rows with Pinnacle odds and <=5 min staleness...")
    df = df[df["has_odds"] == 1].copy()
    df["odds_logit_p1"] = df["odds_no_vig_p1"].apply(
        lambda x: float(logit(np.array([x]))[0]) if pd.notna(x) and 0 < x < 1 else 0.0
    )
    before = len(df)
    df = df[(df["odds_age_seconds"] >= 0) & (df["odds_age_seconds"] <= 300)].copy()
    print(f"   {before:,} -> {len(df):,} rows after staleness filter ({df['event_id'].nunique()} matches)")
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
    print()

    # Step 5: Baseline prediction (raw de-vig odds)
    print("5. Computing baseline (de-vig Pinnacle odds, no adjustment)...")
    baseline_pred = sigmoid(test["odds_logit_p1"].to_numpy(dtype=float))
    baseline_metrics = evaluate_predictions(test[TARGET_COL].to_numpy(dtype=float), baseline_pred)
    print(f"   Baseline: AUC={fmt(baseline_metrics['auc'])} Brier={fmt(baseline_metrics['brier'])} LogLoss={fmt(baseline_metrics['log_loss'])}")
    print()

    # Step 6: First pass — train with all factors to get significance
    print("6. First pass: training with all factors for feature selection...")
    x_train = fill_matrix(train, train, factor_features)
    y_train = train[TARGET_COL].to_numpy(dtype=float)
    offset_train = train["odds_logit_p1"].to_numpy(dtype=float)

    model_pass1 = fit_logistic_with_offset(
        x_train, y_train, offset_train,
        l2=0.02, learning_rate=0.06, epochs=500, cap=ADJUSTMENT_CAP,
    )
    stats_pass1 = offset_logistic_stats(model_pass1, train, "odds_logit_p1", TARGET_COL, factor_features)

    # Select top 2 factors per group by |z-stat|
    selected_features = select_top_factors_per_group(stats_pass1, factor_groups, max_per_group=2)
    print(f"   Selected {len(selected_features)} factors (max 2 per group)")
    for grp, vals in factor_groups.items():
        selected_in_grp = [f for f in vals if f in selected_features]
        if selected_in_grp:
            print(f"     {grp}: {', '.join(selected_in_grp)}")
    print()

    # Step 7: Second pass — retrain with selected factors only
    print("7. Second pass: training with selected factors...")
    x_train_sel = fill_matrix(train, train, selected_features)
    model = fit_logistic_with_offset(
        x_train_sel, y_train, offset_train,
        l2=0.02, learning_rate=0.06, epochs=500, cap=ADJUSTMENT_CAP,
    )
    print(f"   Adjustment cap: ±{ADJUSTMENT_CAP:.2f} (logit space)")

    adjusted_pred = predict_with_offset(
        model,
        fill_matrix(train, test, selected_features),
        test["odds_logit_p1"].to_numpy(dtype=float),
    )
    adjusted_metrics = evaluate_predictions(test[TARGET_COL].to_numpy(dtype=float), adjusted_pred)
    print(f"   Adjusted: AUC={fmt(adjusted_metrics['auc'])} Brier={fmt(adjusted_metrics['brier'])} LogLoss={fmt(adjusted_metrics['log_loss'])}")
    print()

    # Step 8: Factors-only model (no odds prior, for reference)
    print("8. Training factors-only model (logistic regression, no offset)...")
    from train_calibrated_probability_model import fit_logistic, predict_logistic
    factors_model = fit_logistic(
        fill_matrix(train, train, selected_features), y_train,
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    factors_pred = predict_logistic(factors_model, fill_matrix(train, test, selected_features))
    factors_metrics = evaluate_predictions(test[TARGET_COL].to_numpy(dtype=float), factors_pred)
    print(f"   Factors only: AUC={fmt(factors_metrics['auc'])} Brier={fmt(factors_metrics['brier'])} LogLoss={fmt(factors_metrics['log_loss'])}")
    print()

    # Step 9: Compute regression stats for selected factors
    print("9. Computing factor t-stats and p-values...")
    stats = offset_logistic_stats(model, train, "odds_logit_p1", TARGET_COL, selected_features)
    sig_count = sum(1 for s in stats if s["p_value"] is not None and s["p_value"] < 0.05)
    print(f"   {sig_count} significant factors (p < 0.05)")
    print()

    # Step 9: Segment analysis
    print("9. Segment analysis...")
    segments = segment_analysis(test, baseline_pred, adjusted_pred, factors_pred)
    for r in segments:
        print(f"   {r['segment']}: {r['rows']} rows, "
              f"base_auc={fmt(r.get('baseline_auc'))}, "
              f"adj_auc={fmt(r.get('adjusted_auc'))}")
    print()

    # Step 11: Fresh odds subset (0-1 min staleness)
    print("11. Evaluating on fresh odds subset (0-1 min staleness)...")
    fresh_test = test[(test["odds_age_seconds"] >= 0) & (test["odds_age_seconds"] < 60)]
    if len(fresh_test) > 0:
        fresh_y = fresh_test[TARGET_COL].to_numpy(dtype=float)
        fresh_baseline = sigmoid(fresh_test["odds_logit_p1"].to_numpy(dtype=float))
        fresh_adjusted = predict_with_offset(
            model,
            fill_matrix(train, fresh_test, selected_features),
            fresh_test["odds_logit_p1"].to_numpy(dtype=float),
        )
        fresh_factors = predict_logistic(factors_model, fill_matrix(train, fresh_test, selected_features))
        fresh_subset = {
            "baseline": evaluate_predictions(fresh_y, fresh_baseline),
            "adjusted": evaluate_predictions(fresh_y, fresh_adjusted),
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

    # Step 12: Run backtests at multiple thresholds
    print("12. Running backtests across thresholds 0.005 to 0.05...")
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    backtests = {}
    for thresh in thresholds:
        bt = run_backtest(test, baseline_pred, adjusted_pred, threshold=thresh)
        backtests[thresh] = bt
        print(f"   thresh={thresh:.3f}: bets={bt['n_bets']:,} (P1={bt.get('n_p1_bets', 0):,}, P2={bt.get('n_p2_bets', 0):,}), "
              f"wagered={bt['total_wagered']:.1f}, "
              f"PnL={bt['total_pnl']:+.2f}, ROI={bt['roi']:+.2%}, win={bt['win_rate']:.1%}")
    backtest = backtests[0.01]  # default for chart
    print()

    # Step 13: Collect split info
    split_info = {
        "train_events": train["event_id"].nunique(),
        "train_rows": len(train),
        "train_start": str(train["event_start_date"].min().date()) if train["event_start_date"].notna().any() else "N/A",
        "train_end": str(train["event_start_date"].max().date()) if train["event_start_date"].notna().any() else "N/A",
        "test_events": test["event_id"].nunique(),
        "test_rows": len(test),
        "test_start": str(test["event_start_date"].min().date()) if test["event_start_date"].notna().any() else "N/A",
        "test_end": str(test["event_start_date"].max().date()) if test["event_start_date"].notna().any() else "N/A",
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_report(df, selected_features, factor_groups, overall, stats, segments,
                     fresh_subset, diff_stats, diff_bins, backtest, split_info, backtests),
        encoding="utf-8",
    )
    print(f"   Report: {OUTPUT_PATH}")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  Baseline (Pinnacle de-vig):  AUC={fmt(baseline_metrics['auc'])} Brier={fmt(baseline_metrics['brier'])}")
    print(f"  Adjusted (odds+factors):    AUC={fmt(adjusted_metrics['auc'])} Brier={fmt(adjusted_metrics['brier'])}")
    print(f"  Factors only (no odds):     AUC={fmt(factors_metrics['auc'])} Brier={fmt(factors_metrics['brier'])}")
    print(f"  Adjustment cap:           ±{ADJUSTMENT_CAP:.2f} (logit)")
    print(f"  Selected factors:           {len(selected_features)}")
    delta_auc = (adjusted_metrics['auc'] or 0) - (baseline_metrics['auc'] or 0)
    delta_brier = (adjusted_metrics['brier'] or 0) - (baseline_metrics['brier'] or 0)
    print(f"  Delta AUC:  {delta_auc:+.4f}")
    print(f"  Delta Brier: {delta_brier:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
