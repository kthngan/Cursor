#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Data" / "SportsAPI"
SOURCE = DATA_DIR / "event_6571587_incidents_reconstructed.csv"
OUTPUT = DATA_DIR / "event_6571587_incidents_reconstructed.html"


HEADERS = [
    "seq",
    "utc_time",
    "event_status_name",
    "incident_name",
    "participant_side",
    "participant_name",
    "point_to",
    "game_score_after",
    "sets_after",
]


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with SOURCE.open(encoding="utf-8", newline="") as file:
        rows.extend(csv.DictReader(file))

    table_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(row.get(header, ''))}</td>" for header in HEADERS)
        + "</tr>"
        for row in rows
    )
    table_headers = "".join(f"<th>{html.escape(header)}</th>" for header in HEADERS)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reconstructed Tennis Incident Timeline</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2328;
      background: #ffffff;
    }}
    h1 {{
      margin-bottom: 4px;
    }}
    .meta {{
      color: #57606a;
      margin-bottom: 18px;
    }}
    .stats {{
      display: flex;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .stat {{
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 12px 14px;
      min-width: 150px;
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
      margin-bottom: 4px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d0d7de;
      padding: 6px 8px;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f6f8fa;
      z-index: 1;
    }}
    tr:nth-child(even) {{
      background: #f6f8fa;
    }}
  </style>
</head>
<body>
  <h1>Reconstructed Tennis Incident Timeline</h1>
  <div class="meta">
    Lois Boisson vs Elena Rybakina, Wimbledon (Women) 2026, event_id 6571587.
    Source: StatScore events.show events_incidents.
  </div>
  <div class="stats">
    <div class="stat"><strong>{len(rows)}</strong>Rows reconstructed</div>
    <div class="stat"><strong>1-2</strong>Final sets, Boisson-Rybakina</div>
    <div class="stat"><strong>4-6, 6-1, 3-6</strong>Final games by set</div>
  </div>
  <table>
    <thead><tr>{table_headers}</tr></thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
