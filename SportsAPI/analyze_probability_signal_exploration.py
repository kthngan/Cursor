#!/usr/bin/env python3
"""Explore rolling tennis metrics against eventual match outcome.

This is an exploratory report, not a calibrated probability model. It uses the
saved per-match CSVs under Data/SportsAPI and labels every in-match row by the
eventual match winner from that file's final row.
"""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
REPORTS_DIR = WORKSPACE_DIR / "Reports"
OUTPUT_PATH = REPORTS_DIR / "rolling_probability_signal_exploration.html"

METRICS = [
    "rolling_live_form_ratio",
    "rolling_points_ratio_20",
    "rolling_service_points_won_ratio_20",
    "rolling_return_points_won_ratio_20",
    "rolling_break_points_created_ratio_20",
    "rolling_break_points_won_ratio_20",
    "rolling_break_points_saved_ratio_20",
    "rolling_games_won_ratio_6",
]

METRIC_LABELS = {
    "rolling_live_form_ratio": "Composite live form",
    "rolling_points_ratio_20": "Points won, last 20",
    "rolling_service_points_won_ratio_20": "Service points won, last 20",
    "rolling_return_points_won_ratio_20": "Return points won, last 20",
    "rolling_break_points_created_ratio_20": "Break points created, last 20",
    "rolling_break_points_won_ratio_20": "Break points won, last 20",
    "rolling_break_points_saved_ratio_20": "Break points saved, last 20",
    "rolling_games_won_ratio_6": "Games won, last 6",
}


@dataclass
class Agg:
    count: int = 0
    p1_wins: int = 0
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    deltas: list[float] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.count += 1
        self.p1_wins += int(row["p1_win"])
        for metric in METRICS:
            value = row.get(metric)
            if isinstance(value, float) and math.isfinite(value):
                self.values[metric].append(value)
        delta = row.get("live_form_delta_5")
        if isinstance(delta, float) and math.isfinite(delta):
            self.deltas.append(delta)

    @property
    def p1_win_rate(self) -> float:
        return self.p1_wins / self.count if self.count else 0.0

    def avg(self, metric: str) -> float | None:
        vals = self.values.get(metric) or []
        return mean(vals) if vals else None


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def score_state(row: dict[str, Any]) -> str:
    return (
        f"sets {row.get('sets_after') or '?'} | "
        f"games {row.get('game_score_after') or '?'} | "
        f"points {row.get('point_score_state') or '?'}"
    )


def load_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counters = {
        "files": 0,
        "files_with_winner": 0,
        "raw_rows": 0,
        "usable_rows": 0,
    }
    for path in sorted(DATA_DIR.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as file:
            file_rows = list(csv.DictReader(file))
        if not file_rows or "rolling_live_form_ratio" not in file_rows[0]:
            continue
        counters["files"] += 1
        counters["raw_rows"] += len(file_rows)

        winner = ""
        for row in reversed(file_rows):
            winner = (row.get("match_winner_side") or "").strip()
            if winner:
                break
        if winner not in {"P1", "P2"}:
            continue
        counters["files_with_winner"] += 1

        previous_live_values: list[float] = []
        for row in file_rows:
            metric_values = {metric: parse_float(row.get(metric)) for metric in METRICS}
            if not any(value is not None for value in metric_values.values()):
                continue
            live = metric_values["rolling_live_form_ratio"]
            delta_5 = None
            if live is not None and len(previous_live_values) >= 5:
                delta_5 = live - previous_live_values[-5]
            if live is not None:
                previous_live_values.append(live)
            enriched: dict[str, Any] = {
                **row,
                **metric_values,
                "source_file": path.name,
                "p1_win": 1 if winner == "P1" else 0,
                "winner_side": winner,
                "score_state": score_state(row),
                "live_form_delta_5": delta_5,
            }
            rows.append(enriched)

    counters["usable_rows"] = len(rows)
    return rows, counters


def bucket_metric(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.40:
        return "P2 strong (<40%)"
    if value < 0.48:
        return "P2 edge (40-48%)"
    if value <= 0.52:
        return "neutral (48-52%)"
    if value <= 0.60:
        return "P1 edge (52-60%)"
    return "P1 strong (>60%)"


def trend_bucket(delta: float | None) -> str:
    if delta is None:
        return "missing"
    if delta <= -0.12:
        return "sharp P2 shift (<= -12pp)"
    if delta <= -0.06:
        return "P2 shift (-12pp to -6pp)"
    if delta < 0.06:
        return "stable (-6pp to +6pp)"
    if delta < 0.12:
        return "P1 shift (+6pp to +12pp)"
    return "sharp P1 shift (>= +12pp)"


def aggregate(rows: Iterable[dict[str, Any]], key_fn) -> dict[str, Agg]:
    grouped: dict[str, Agg] = defaultdict(Agg)
    for row in rows:
        grouped[key_fn(row)].add(row)
    return grouped


def directional_accuracy(rows: Iterable[dict[str, Any]], metric: str) -> tuple[int, int, float | None, float | None]:
    correct = 0
    total = 0
    brier_values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if not isinstance(value, float):
            continue
        if value == 0.5:
            continue
        pred = 1 if value > 0.5 else 0
        actual = int(row["p1_win"])
        correct += int(pred == actual)
        total += 1
        brier_values.append((value - actual) ** 2)
    accuracy = correct / total if total else None
    brier = mean(brier_values) if brier_values else None
    return correct, total, accuracy, brier


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(rows: list[dict[str, Any]], counters: dict[str, int]) -> str:
    metric_rows: list[list[str]] = []
    for metric in METRICS:
        correct, total, accuracy, brier = directional_accuracy(rows, metric)
        grouped = aggregate(rows, lambda row, m=metric: bucket_metric(row.get(m)))
        bucket_summary = "; ".join(
            f"{key}: n={agg.count}, P1 win={fmt_pct(agg.p1_win_rate)}"
            for key, agg in sorted(grouped.items())
            if key != "missing"
        )
        metric_rows.append(
            [
                METRIC_LABELS[metric],
                str(total),
                fmt_pct(accuracy),
                fmt_num(brier, 4),
                bucket_summary,
            ]
        )

    score_groups = aggregate(rows, lambda row: row["score_state"])
    score_rows = []
    for state, agg in sorted(score_groups.items(), key=lambda item: (-item[1].count, item[0]))[:80]:
        score_rows.append(
            [
                state,
                str(agg.count),
                fmt_pct(agg.p1_win_rate),
                fmt_num(agg.avg("rolling_live_form_ratio")),
                fmt_num(agg.avg("rolling_points_ratio_20")),
                fmt_num(agg.avg("rolling_return_points_won_ratio_20")),
                fmt_num(agg.avg("rolling_service_points_won_ratio_20")),
            ]
        )

    trend_groups = aggregate(rows, lambda row: trend_bucket(row.get("live_form_delta_5")))
    trend_rows = []
    trend_order = [
        "sharp P2 shift (<= -12pp)",
        "P2 shift (-12pp to -6pp)",
        "stable (-6pp to +6pp)",
        "P1 shift (+6pp to +12pp)",
        "sharp P1 shift (>= +12pp)",
    ]
    for key in trend_order:
        agg = trend_groups.get(key)
        if not agg:
            continue
        trend_rows.append(
            [
                key,
                str(agg.count),
                fmt_pct(agg.p1_win_rate),
                fmt_num(mean(agg.deltas) if agg.deltas else None),
                fmt_num(agg.avg("rolling_live_form_ratio")),
            ]
        )

    example_rows = []
    candidates = [
        row
        for row in rows
        if isinstance(row.get("live_form_delta_5"), float)
        and abs(row["live_form_delta_5"]) >= 0.18
        and isinstance(row.get("rolling_live_form_ratio"), float)
    ]
    for row in sorted(candidates, key=lambda r: abs(r["live_form_delta_5"]), reverse=True)[:60]:
        example_rows.append(
            [
                row["source_file"],
                row.get("event_name", ""),
                row.get("event_time", ""),
                row["score_state"],
                fmt_num(row.get("rolling_live_form_ratio")),
                fmt_num(row.get("live_form_delta_5")),
                row["winner_side"],
            ]
        )

    model_text = """
    <ol>
      <li><b>Score baseline:</b> estimate P1 win probability from tennis state only:
      sets, games, point score, server, and pre-match prior when available.</li>
      <li><b>Rolling-form adjustment:</b> add centered metric edges such as
      points ratio - 0.5, service ratio - 0.5, return ratio - 0.5, break-created,
      break-won, break-saved, games-won, and composite live form.</li>
      <li><b>Trend-change adjustment:</b> add short-horizon deltas, for example
      current live form minus the value five incidents ago, and crossing indicators
      when a metric moves from P2-favored to P1-favored.</li>
      <li><b>Calibration after odds arrive:</b> train on historical live odds converted
      to no-vig implied probabilities, use time-split validation, then calibrate with
      isotonic regression or Platt scaling by sport/competition/score phase.</li>
      <li><b>Accuracy measurement:</b> use Brier score, log loss, calibration curves,
      expected calibration error, direction accuracy versus market moves, and
      profit-style backtests only after including realistic latency and overround.</li>
    </ol>
    """

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; position: sticky; top: 0; }
    td { line-height: 1.35; }
    .note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SportsAPI Rolling Metrics Probability Exploration</title>
  <style>{css}</style>
</head>
<body>
  <h1>SportsAPI Rolling Metrics Probability Exploration</h1>
  <p class="muted">Source: local CSV files in {html.escape(str(DATA_DIR))}. Generated from row-level incident states labeled by final match winner.</p>

  <div class="grid">
    <div class="card"><div class="muted">CSV files scanned</div><div class="stat">{counters['files']}</div></div>
    <div class="card"><div class="muted">Files with winner</div><div class="stat">{counters['files_with_winner']}</div></div>
    <div class="card"><div class="muted">Raw rows</div><div class="stat">{counters['raw_rows']:,}</div></div>
    <div class="card"><div class="muted">Usable metric rows</div><div class="stat">{counters['usable_rows']:,}</div></div>
  </div>

  <div class="note">
    <b>Interpretation caveat:</b> this report is row-level and exploratory. Long matches contribute more rows than short matches, and the label is eventual match winner, not next-point or market-implied probability.
  </div>

  <h2>Proposed Model</h2>
  {model_text}

  <h2>Metric Direction vs Eventual Winner</h2>
  {table(["Metric", "Rows with metric", "Direction accuracy", "Brier if used directly", "Bucket outcome summary"], metric_rows)}

  <h2>Most Common Score States</h2>
  {table(["State", "Rows", "P1 eventual win rate", "Avg live form", "Avg points ratio", "Avg return ratio", "Avg service ratio"], score_rows)}

  <h2>Trend Change Buckets</h2>
  {table(["Live-form change over 5 incidents", "Rows", "P1 eventual win rate", "Avg delta", "Avg live form"], trend_rows)}

  <h2>Largest Trend-Change Examples</h2>
  {table(["File", "Match", "Event time", "State", "Live form", "Delta over 5", "Winner"], example_rows)}
</body>
</html>
"""


def main() -> int:
    rows, counters = load_rows()
    if not rows:
        raise SystemExit(f"No usable rows found under {DATA_DIR}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(rows, counters), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Files with winner: {counters['files_with_winner']} / {counters['files']}")
    print(f"Usable metric rows: {counters['usable_rows']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
