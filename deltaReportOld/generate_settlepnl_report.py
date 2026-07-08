import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def resolve_input_path(input_path: str) -> Path:
    """
    Accept a direct JSON file path or a path without the .json suffix.
    """
    raw = Path(input_path)
    if raw.is_file():
        return raw

    if raw.suffix.lower() != ".json":
        with_json = raw.with_suffix(".json")
        if with_json.is_file():
            return with_json

    raise FileNotFoundError(f"Could not find JSON file for: {input_path}")


def extract_daily_settle_pnl(payload: dict) -> List[Tuple[datetime, float]]:
    reports = payload.get("reports", [])
    daily_totals: Dict[datetime, float] = {}

    for report in reports:
        group = report.get("group", {})
        date_str = group.get("date")
        settle_pnl = (
            report.get("implied_prob", {})
            .get("timepoints", {})
            .get("settle", {})
            .get("pnl")
        )
        if date_str is None or settle_pnl is None:
            continue

        try:
            day = datetime.strptime(date_str, "%Y-%m-%d")
            pnl_value = float(settle_pnl)
            daily_totals[day] = daily_totals.get(day, 0.0) + pnl_value
        except (ValueError, TypeError):
            continue

    daily_values = sorted(daily_totals.items(), key=lambda x: x[0])
    return daily_values


def build_cumulative_series(daily_values: List[Tuple[datetime, float]]) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    cumulative: List[float] = []
    running = 0.0

    for day, pnl in daily_values:
        running += pnl
        labels.append(day.strftime("%Y-%m-%d"))
        cumulative.append(round(running, 6))

    return labels, cumulative


def parse_bucket_start_end(bucket_name: str) -> Tuple[float, float]:
    start_str, end_str = bucket_name.split("-")
    return float(start_str), float(end_str)


def extract_daily_bucket_band_settle_pnl(
    payload: dict,
) -> List[Tuple[datetime, float, float, float]]:
    reports = payload.get("reports", [])
    # date -> [low_sum, mid_sum, high_sum]
    daily_band_totals: Dict[datetime, List[float]] = {}

    for report in reports:
        date_str = report.get("group", {}).get("date")
        price_buckets = report.get("implied_prob", {}).get("price_buckets", {})
        if date_str is None or not isinstance(price_buckets, dict):
            continue

        try:
            day = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        if day not in daily_band_totals:
            daily_band_totals[day] = [0.0, 0.0, 0.0]

        for bucket_name, metrics in price_buckets.items():
            if not isinstance(metrics, dict):
                continue
            settle_pnl = metrics.get("settle_pnl")
            if settle_pnl is None:
                continue

            try:
                start, end = parse_bucket_start_end(bucket_name)
                pnl_value = float(settle_pnl)
            except (ValueError, TypeError):
                continue

            # low: [0.0, 0.4), mid: [0.4, 0.6), high: [0.6, 1.0]
            if 0.0 <= start and end <= 0.4:
                daily_band_totals[day][0] += pnl_value
            elif 0.4 <= start and end <= 0.6:
                daily_band_totals[day][1] += pnl_value
            elif 0.6 <= start and end <= 1.0:
                daily_band_totals[day][2] += pnl_value

    ordered_days = sorted(daily_band_totals.keys())
    return [
        (day, daily_band_totals[day][0], daily_band_totals[day][1], daily_band_totals[day][2])
        for day in ordered_days
    ]


def build_band_cumulative_series(
    daily_band_values: List[Tuple[datetime, float, float, float]]
) -> Tuple[List[str], List[float], List[float], List[float]]:
    labels: List[str] = []
    low_cumulative: List[float] = []
    mid_cumulative: List[float] = []
    high_cumulative: List[float] = []

    low_running = 0.0
    mid_running = 0.0
    high_running = 0.0

    for day, low, mid, high in daily_band_values:
        low_running += low
        mid_running += mid
        high_running += high
        labels.append(day.strftime("%Y-%m-%d"))
        low_cumulative.append(round(low_running, 6))
        mid_cumulative.append(round(mid_running, 6))
        high_cumulative.append(round(high_running, 6))

    return labels, low_cumulative, mid_cumulative, high_cumulative


def render_html(
    labels: List[str],
    values: List[float],
    band_labels: List[str],
    low_values: List[float],
    mid_values: List[float],
    high_values: List[float],
    source_file: Path,
) -> str:
    labels_json = json.dumps(labels)
    values_json = json.dumps(values)
    band_labels_json = json.dumps(band_labels)
    low_values_json = json.dumps(low_values)
    mid_values_json = json.dumps(mid_values)
    high_values_json = json.dumps(high_values)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cumulative settlePnL Report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f9fafb;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 16px;
      max-width: 1100px;
    }}
    h1 {{
      margin-top: 0;
      font-size: 1.4rem;
    }}
    .meta {{
      color: #6b7280;
      margin-bottom: 14px;
      font-size: 0.95rem;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Cumulative settlePnL over Dates</h1>
    <div class="meta">
      Source: {source_file.name}<br/>
      Generated: {generated_at}
    </div>
    <canvas id="pnlChart" height="110"></canvas>
  </div>
  <div class="card" style="margin-top: 16px;">
    <h1>Cumulative settlePnL by Probability Band</h1>
    <div class="meta">
      low: [0.0, 0.4), mid: [0.4, 0.6), high: [0.6, 1.0]
    </div>
    <canvas id="bandChart" height="110"></canvas>
  </div>

  <script>
    const labels = {labels_json};
    const values = {values_json};
    const bandLabels = {band_labels_json};
    const lowValues = {low_values_json};
    const midValues = {mid_values_json};
    const highValues = {high_values_json};
    const ctx = document.getElementById('pnlChart');
    const bandCtx = document.getElementById('bandChart');

    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels,
        datasets: [{{
          label: 'Cumulative settlePnL',
          data: values,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.15)',
          pointRadius: 3,
          pointHoverRadius: 5,
          borderWidth: 2,
          tension: 0.2,
          fill: true
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        scales: {{
          x: {{
            ticks: {{ maxRotation: 45, minRotation: 45 }}
          }},
          y: {{
            title: {{
              display: true,
              text: 'Cumulative settlePnL'
            }}
          }}
        }},
        plugins: {{
          legend: {{ display: true }},
          tooltip: {{
            callbacks: {{
              label: (ctx) => `Cumulative settlePnL: ${{ctx.parsed.y}}`
            }}
          }}
        }}
      }}
    }});

    new Chart(bandCtx, {{
      type: 'line',
      data: {{
        labels: bandLabels,
        datasets: [
          {{
            label: 'Low Prob Cumulative settlePnL (0.0-0.4)',
            data: lowValues,
            borderColor: '#dc2626',
            backgroundColor: 'rgba(220, 38, 38, 0.08)',
            pointRadius: 2,
            borderWidth: 2,
            tension: 0.2
          }},
          {{
            label: 'Mid Prob Cumulative settlePnL (0.4-0.6)',
            data: midValues,
            borderColor: '#d97706',
            backgroundColor: 'rgba(217, 119, 6, 0.08)',
            pointRadius: 2,
            borderWidth: 2,
            tension: 0.2
          }},
          {{
            label: 'High Prob Cumulative settlePnL (0.6-1.0)',
            data: highValues,
            borderColor: '#16a34a',
            backgroundColor: 'rgba(22, 163, 74, 0.08)',
            pointRadius: 2,
            borderWidth: 2,
            tension: 0.2
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        scales: {{
          x: {{
            ticks: {{ maxRotation: 45, minRotation: 45 }}
          }},
          y: {{
            title: {{
              display: true,
              text: 'Cumulative settlePnL'
            }}
          }}
        }},
        plugins: {{
          legend: {{ display: true }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "Data" / "deltaReportOld"
    default_input = str(data_dir / "delta_report_public.json")
    parser = argparse.ArgumentParser(
        description="Create cumulative settlePnL HTML report from delta report JSON."
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help="Path to input JSON file (or same path without .json).",
    )
    args = parser.parse_args()

    input_file = resolve_input_path(args.input)
    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    daily_values = extract_daily_settle_pnl(payload)
    if not daily_values:
        raise ValueError("No valid date + settlePnL entries found in the JSON report.")

    daily_band_values = extract_daily_bucket_band_settle_pnl(payload)
    if not daily_band_values:
        raise ValueError("No valid price bucket settle_pnl entries found in the JSON report.")

    labels, cumulative = build_cumulative_series(daily_values)
    band_labels, low_cumulative, mid_cumulative, high_cumulative = build_band_cumulative_series(
        daily_band_values
    )
    html_content = render_html(
        labels,
        cumulative,
        band_labels,
        low_cumulative,
        mid_cumulative,
        high_cumulative,
        input_file,
    )

    output_file = input_file.parent / "cumulative_settlePnL_report.html"
    output_file.write_text(html_content, encoding="utf-8")
    print(f"Report created: {output_file}")


if __name__ == "__main__":
    main()
