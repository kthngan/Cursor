#!/usr/bin/env python3
"""Rank rolling SportsAPI metric columns by predictive power.

The current CSVs do not contain historical odds. This report treats each row as
an in-match state and labels it by the eventual match winner recorded in the
same CSV. Metrics are evaluated as P1-oriented signals.
"""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
REPORTS_DIR = WORKSPACE_DIR / "Reports"
OUTPUT_PATH = REPORTS_DIR / "metric_predictive_power_by_set_state.html"

METRICS = [
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

LABELS = {
    "rolling_live_form_ratio": "Composite live form",
    "rolling_points_ratio_20": "Points won, last 20",
    "rolling_service_points_won_ratio_20": "Service points won, last 20",
    "rolling_return_points_won_ratio_20": "Return points won, last 20",
    "rolling_break_points_created_ratio_20": "Break points created, last 20",
    "rolling_break_points_won_ratio_20": "Break points won, last 20",
    "rolling_break_points_saved_ratio_20": "Break points saved, last 20",
    "rolling_games_won_ratio_6": "Games won, last 6",
    "live_form_delta_5": "Composite live-form delta over 5 incidents",
}


@dataclass
class EvalResult:
    group: str
    metric: str
    rows: int
    matches: int
    auc: float | None
    abs_auc_edge: float | None
    corr: float | None
    abs_corr: float | None
    direction_accuracy: float | None
    rmse: float | None
    p1_win_rate: float
    mean_value: float


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def sets_group(sets_after: str) -> str:
    if sets_after in {"0-0", "1-0", "0-1", "1-1", "2-0", "0-2", "2-1", "1-2"}:
        return sets_after
    return sets_after or "unknown"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def roc_auc(xs: list[float], ys: list[int]) -> float | None:
    positives = sum(ys)
    negatives = len(ys) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[idx:end])
        rank += end - idx
        idx = end

    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def clamp_probability(value: float, metric: str) -> float:
    if metric == "live_form_delta_5":
        # Map roughly [-0.5, 0.5] deltas into [0, 1] for RMSE only.
        return min(1.0, max(0.0, 0.5 + value))
    return min(1.0, max(0.0, value))


def load_rows() -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counters = {"files": 0, "files_with_winner": 0, "raw_rows": 0, "usable_rows": 0}
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
            values = {metric: parse_float(row.get(metric)) for metric in METRICS if metric != "live_form_delta_5"}
            live = values.get("rolling_live_form_ratio")
            delta_5 = None
            if live is not None and len(previous_live_values) >= 5:
                delta_5 = live - previous_live_values[-5]
            if live is not None:
                previous_live_values.append(live)
            values["live_form_delta_5"] = delta_5
            if not any(value is not None for value in values.values()):
                continue

            rows.append(
                {
                    "source_file": path.name,
                    "sets_after": sets_group(row.get("sets_after", "")),
                    "p1_win": 1 if winner == "P1" else 0,
                    **values,
                }
            )
    counters["usable_rows"] = len(rows)
    return rows, counters


def evaluate(rows: Iterable[dict], group_name: str, metric: str) -> EvalResult | None:
    xs: list[float] = []
    ys: list[int] = []
    match_ids: set[str] = set()
    for row in rows:
        value = row.get(metric)
        if not isinstance(value, float):
            continue
        xs.append(value)
        ys.append(int(row["p1_win"]))
        match_ids.add(str(row["source_file"]))

    if len(xs) < 200:
        return None

    auc = roc_auc(xs, ys)
    corr = pearson(xs, [float(y) for y in ys])
    direction_total = 0
    direction_correct = 0
    squared_errors: list[float] = []
    for x, y in zip(xs, ys):
        if metric == "live_form_delta_5" and abs(x) < 1e-12:
            continue
        pred = 1 if x > (0.0 if metric == "live_form_delta_5" else 0.5) else 0
        direction_correct += int(pred == y)
        direction_total += 1
        p = clamp_probability(x, metric)
        squared_errors.append((p - y) ** 2)

    rmse = math.sqrt(mean(squared_errors)) if squared_errors else None
    return EvalResult(
        group=group_name,
        metric=metric,
        rows=len(xs),
        matches=len(match_ids),
        auc=auc,
        abs_auc_edge=abs(auc - 0.5) if auc is not None else None,
        corr=corr,
        abs_corr=abs(corr) if corr is not None else None,
        direction_accuracy=direction_correct / direction_total if direction_total else None,
        rmse=rmse,
        p1_win_rate=mean(ys),
        mean_value=mean(xs),
    )


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def result_rows(results: list[EvalResult], limit: int | None = None) -> list[list[str]]:
    selected = results[:limit] if limit else results
    return [
        [
            result.group,
            LABELS[result.metric],
            str(result.rows),
            str(result.matches),
            fmt(result.auc),
            fmt(result.corr),
            fmt_pct(result.direction_accuracy),
            fmt(result.rmse),
            fmt_pct(result.p1_win_rate),
            fmt(result.mean_value),
        ]
        for result in selected
    ]


def build_report(rows: list[dict], counters: dict[str, int], results: list[EvalResult]) -> str:
    overall = [result for result in results if result.group == "all"]
    by_set = [result for result in results if result.group != "all"]
    best_by_set: list[EvalResult] = []
    for group in sorted({result.group for result in by_set}):
        candidates = [result for result in by_set if result.group == group]
        candidates.sort(key=lambda r: (r.abs_auc_edge or -1.0, r.abs_corr or -1.0), reverse=True)
        best_by_set.extend(candidates[:5])

    all_ranked = sorted(results, key=lambda r: (r.abs_auc_edge or -1.0, r.abs_corr or -1.0), reverse=True)

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; }
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
    headers = [
        "Set state",
        "Metric",
        "Rows",
        "Matches",
        "AUC",
        "Correlation",
        "Direction accuracy",
        "RMSE",
        "P1 win rate",
        "Mean value",
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SportsAPI Metric Predictive Power by Set State</title>
  <style>{css}</style>
</head>
<body>
  <h1>SportsAPI Metric Predictive Power by Set State</h1>
  <p class="muted">Source: local CSV files in {html.escape(str(DATA_DIR))}. Labels are eventual match winner, evaluated at each in-match row.</p>

  <div class="grid">
    <div class="card"><div class="muted">CSV files scanned</div><div class="stat">{counters['files']}</div></div>
    <div class="card"><div class="muted">Files with winner</div><div class="stat">{counters['files_with_winner']}</div></div>
    <div class="card"><div class="muted">Raw rows</div><div class="stat">{counters['raw_rows']:,}</div></div>
    <div class="card"><div class="muted">Usable metric rows</div><div class="stat">{counters['usable_rows']:,}</div></div>
  </div>

  <div class="note">
    <b>How to read this:</b> AUC above 0.5 means higher metric values are associated with P1 eventually winning.
    Correlation is Pearson correlation against the final P1-win label. RMSE treats each ratio as a rough P1 probability, not a calibrated model.
    Because this is row-weighted, longer matches contribute more observations.
  </div>

  <h2>Overall Ranking</h2>
  {table(headers, result_rows(sorted(overall, key=lambda r: (r.abs_auc_edge or -1.0, r.abs_corr or -1.0), reverse=True)))}

  <h2>Best Metrics Within Each Set State</h2>
  {table(headers, result_rows(best_by_set))}

  <h2>Full Ranking</h2>
  {table(headers, result_rows(all_ranked))}
</body>
</html>
"""


def main() -> int:
    rows, counters = load_rows()
    results: list[EvalResult] = []
    for metric in METRICS:
        result = evaluate(rows, "all", metric)
        if result:
            results.append(result)

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["sets_after"]].append(row)
    for group, group_rows in sorted(grouped_rows.items()):
        for metric in METRICS:
            result = evaluate(group_rows, group, metric)
            if result:
                results.append(result)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(rows, counters, results), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for result in sorted(
        [r for r in results if r.group == "all"],
        key=lambda r: (r.abs_auc_edge or -1.0, r.abs_corr or -1.0),
        reverse=True,
    ):
        print(
            f"{LABELS[result.metric]}: AUC={fmt(result.auc)} "
            f"corr={fmt(result.corr)} accuracy={fmt_pct(result.direction_accuracy)} RMSE={fmt(result.rmse)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
