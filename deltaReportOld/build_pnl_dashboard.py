"""
Build pnl_dashboard.html only from an existing JSON file.

For the full pipeline (browser download + PNG + CSV + HTML), use:

  python pnlTimeSeries.py --username ... --password ...
  python pnlTimeSeries.py --input-json delta_report_....json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pnlTimeSeries import (
    build_dashboard_payload,
    find_latest_report_json,
    iter_payload_report_rows,
    render_dashboard_html,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pnl_dashboard.html from delta report JSON.")
    ap.add_argument("--input", type=str, default="", help="Path to delta report JSON.")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output HTML path (default: outdir/pnl_dashboard.html).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Folder containing JSON; used when --input is omitted to pick newest delta_report*.json.",
    )
    args = ap.parse_args()

    folder = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    src = Path(args.input) if args.input else find_latest_report_json(folder)
    if not src.is_file():
        raise FileNotFoundError(src)

    out = Path(args.out) if args.out else folder / "pnl_dashboard.html"

    payload_raw = json.loads(src.read_text(encoding="utf-8"))
    reports = iter_payload_report_rows(payload_raw)
    if not reports:
        raise ValueError("JSON has no usable report rows (reports/groups empty).")

    payload = build_dashboard_payload(reports)
    out.write_text(render_dashboard_html(payload, src.name), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
