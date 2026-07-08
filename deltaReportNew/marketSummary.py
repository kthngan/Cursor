"""
Market-level HTML report: fixed account set (nxy, rn1, sovereign, tony + default algo users).
Algo user_ids are rolled into a single label ``Algo`` for tables and charts.
Includes daily total risk (USD) by account overall and within each sport / league / tier slice,
matching the account summary chart style.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from analytics import (
    DEFAULT_JSON_DIR,
    DEFAULT_MARKET_OUTPUT,
    DEFAULT_MARKET_USER_IDS,
    UNIFIED_LEAGUES_CSV,
    UNIFIED_MARKETS_CSV,
    add_market_account_column,
    build_market_report_html,
    load_frames,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build market PnL HTML report (fixed accounts; Algo consolidated)."
    )
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=None,
        help="Start date (ISO). Default: end minus 29 days (30-day window).",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=None,
        help="End date (ISO). Default: today.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help=f"Folder with YYYY-MM-DD.json (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_MARKET_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_MARKET_OUTPUT})",
    )
    parser.add_argument(
        "--leagues-csv",
        type=Path,
        default=UNIFIED_LEAGUES_CSV,
        help=f"unified_leagues.csv for League/Tier labels (default: {UNIFIED_LEAGUES_CSV})",
    )
    parser.add_argument(
        "--markets-csv",
        type=Path,
        default=UNIFIED_MARKETS_CSV,
        help=f"unified_markets.csv for Mkt Type (default: {UNIFIED_MARKETS_CSV})",
    )
    args = parser.parse_args()

    end = args.end or dt.date.today()
    start = args.start if args.start is not None else end - dt.timedelta(days=29)

    if start > end:
        raise SystemExit("start date must be on or before end date")

    json_dir = args.json_dir.resolve()
    user_ids = set(DEFAULT_MARKET_USER_IDS)

    df = load_frames(
        json_dir,
        start,
        end,
        user_ids,
        leagues_csv=args.leagues_csv.resolve(),
        markets_csv=args.markets_csv.resolve(),
    )
    df = add_market_account_column(df)

    report = build_market_report_html(df, start, end)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(df)} rows).")
    if df.empty:
        print(
            "Warning: no rows for the selected accounts and date range.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
