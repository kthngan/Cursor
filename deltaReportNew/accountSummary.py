"""
Build PnL / turnover HTML report from daily trade-markout JSON files.

Default: last 30 days, only ``user_id`` values for Algo0/Algo14/Algo15
(``1771``, ``574079``, ``576330``). Override via ``--accounts``.
``--exclude-accounts`` can further remove IDs from the resolved include set.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from analytics import (
    DEFAULT_EXCLUDE_ACCOUNTS,
    DEFAULT_INCLUDE_ACCOUNTS,
    DEFAULT_JSON_DIR,
    DEFAULT_OUTPUT,
    UNIFIED_LEAGUES_CSV,
    UNIFIED_MARKETS_CSV,
    build_report_html,
    collect_user_ids_in_json,
    load_frames,
    parse_account_token,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build delta PnL HTML report from JSON folder.")
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
        "--accounts",
        nargs="*",
        default=list(DEFAULT_INCLUDE_ACCOUNTS),
        help='Include only these user_id tokens (after exclusions). "0x" = hex. '
        "Default: Algo0/Algo14/Algo15 accounts (1771 574079 576330).",
    )
    parser.add_argument(
        "--exclude-accounts",
        nargs="*",
        default=list(DEFAULT_EXCLUDE_ACCOUNTS),
        help="Exclude these user_id tokens from the report (default: 1983 2082 8482 2000).",
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
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
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
    parser.add_argument(
        "--list-accounts",
        action="store_true",
        help="Print all user_id values found in JSON for the date range, then exit (no HTML).",
    )
    args = parser.parse_args()

    end = args.end or dt.date.today()
    start = args.start if args.start is not None else end - dt.timedelta(days=29)

    if start > end:
        raise SystemExit("start date must be on or before end date")

    json_dir = args.json_dir.resolve()
    all_in_json = collect_user_ids_in_json(json_dir, start, end)

    if args.list_accounts:
        for uid in all_in_json:
            print(f"{uid}\t0x{uid:x}")
        print(f"# count: {len(all_in_json)}", flush=True)
        return 0

    exclude_ids = {parse_account_token(a) for a in (args.exclude_accounts or [])}
    user_ids = {uid for uid in all_in_json if uid not in exclude_ids}
    if args.accounts:
        include_only = {parse_account_token(a) for a in args.accounts}
        user_ids &= include_only

    report_accounts = tuple(str(uid) for uid in sorted(user_ids))
    df = load_frames(
        json_dir,
        start,
        end,
        user_ids,
        leagues_csv=args.leagues_csv.resolve(),
        markets_csv=args.markets_csv.resolve(),
    )

    report = build_report_html(df, start, end, report_accounts)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path} ({len(df)} rows).")
    if df.empty:
        print(
            "Warning: no rows after filtering. Check JSON `user_id` values, --exclude-accounts, "
            "and optional --accounts. Resolved include set: "
            f"{sorted(user_ids)}.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
