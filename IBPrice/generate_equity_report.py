"""
Generate an HTML report with equity curve and summary tables.

Inputs expected in output directory:
  - equity_curve.png
  - summary.txt
  - summary_by_entry_hour.csv
  - summary_by_step.csv
  - summary_by_side.csv
  - summary_by_hour_step_side.csv

Usage:
  python generate_equity_report.py
  python generate_equity_report.py --output-dir backtest_output --report-name equity_report.html
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML equity report from backtest outputs.")
    parser.add_argument("--output-dir", default="backtest_output", help="Directory containing backtest outputs.")
    parser.add_argument("--report-name", default="equity_report.html", help="Output HTML filename.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return (reader.fieldnames or []), rows


def read_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def table_html(title: str, headers: list[str], rows: list[dict[str, str]]) -> str:
    if not headers:
        return f"<h3>{html.escape(title)}</h3><p>No data.</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(row.get(h, ''))}</td>" for h in headers)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows) if body_rows else f"<tr><td colspan='{len(headers)}'>No rows</td></tr>"
    return (
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    report_path = out_dir / args.report_name

    summary = read_summary(out_dir / "summary.txt")
    equity_img = out_dir / "equity_curve.png"

    sections: list[str] = []
    for title, filename in [
        ("Summary By Entry Hour", "summary_by_entry_hour.csv"),
        ("Summary By Step", "summary_by_step.csv"),
        ("Summary By Side", "summary_by_side.csv"),
        ("Summary By Hour, Step, Side", "summary_by_hour_step_side.csv"),
    ]:
        path = out_dir / filename
        if path.exists():
            headers, rows = read_csv_rows(path)
            sections.append(table_html(title, headers, rows))
        else:
            sections.append(f"<h3>{html.escape(title)}</h3><p>File not found: {html.escape(filename)}</p>")

    summary_cards = ""
    if summary:
        cards = []
        for k, v in summary.items():
            cards.append(
                "<div class='card'>"
                f"<div class='card-key'>{html.escape(k)}</div>"
                f"<div class='card-value'>{html.escape(v)}</div>"
                "</div>"
            )
        summary_cards = "<div class='cards'>" + "".join(cards) + "</div>"

    image_block = (
        "<img class='equity' src='equity_curve.png' alt='Equity Curve' />"
        if equity_img.exists()
        else "<p>equity_curve.png not found.</p>"
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Backtest Equity Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
    h1, h2, h3 {{ margin: 0.6em 0; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; min-width: 160px; }}
    .card-key {{ font-size: 12px; color: #666; }}
    .card-value {{ font-size: 20px; font-weight: 600; margin-top: 4px; }}
    .equity {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    .table-wrap {{ overflow-x: auto; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>Backtest Equity Report</h1>
  <h2>Overview</h2>
  {summary_cards}
  <h2>Equity Curve</h2>
  {image_block}
  <h2>Round-Trip PnL Summaries</h2>
  {"".join(sections)}
</body>
</html>
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()

