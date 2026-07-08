"""
Download trade markout JSON from poly-pnl per "day".

Each day T is queried as a single window from 3pm local on T through 3pm local
on T+1, and saved as ``{T}.json`` (ISO date).

Adjusts the requested date range against existing files in ./json:
  - Skips re-fetching days before the latest file date (no overlap with
    data you already have for earlier days).
  - Starts the query at max(user_start, latest_existing_date) so the last
    existing day is included again (refresh / boundary overlap).

Requires: pip install -r requirements.txt && playwright install chromium

Use ``--verify-fetch DATE_A DATE_B`` to re-download two days in one session and
compare whether ``groups``/``trades`` match (order ignored).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect
from playwright.sync_api import sync_playwright

BASE_URL = "https://poly-pnl.it9.win/trade-markout"
USERNAME = "mm"
PASSWORD = "2047"

# Exact labels on the "Group By" toggle buttons (see trade-markout UI).
GROUP_BY_ON = frozenset(
    {
        "Wallet",
        "Sport",
        "League",
        "Mkt Type",
        "Role",
        "Stage",
        "Price Bucket",
        "Market",
    }
)

DATE_JSON_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$", re.IGNORECASE)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_existing_dates(json_dir: Path) -> set[dt.date]:
    dates: set[dt.date] = set()
    if not json_dir.is_dir():
        return dates
    for p in json_dir.iterdir():
        if not p.is_file():
            continue
        m = DATE_JSON_PATTERN.match(p.name)
        if m:
            dates.add(dt.date.fromisoformat(m.group(1)))
    return dates


def adjust_range(
    user_start: dt.date,
    user_end: dt.date,
    existing: set[dt.date],
) -> tuple[dt.date, dt.date] | None:
    """Return inclusive (start, end) to query, or None if nothing to fetch."""
    if user_start > user_end:
        print("Error: start date is after end date.", file=sys.stderr)
        return None

    if not existing:
        return user_start, user_end

    latest_existing = max(existing)
    effective_start = max(user_start, latest_existing)
    effective_end = user_end

    if effective_start > effective_end:
        print(
            f"Nothing to fetch: latest file is {latest_existing} and end {user_end} "
            "is before adjusted start.",
            file=sys.stderr,
        )
        return None

    if effective_start != user_start or effective_end != user_end:
        print(
            f"Adjusted range (existing latest={latest_existing}): "
            f"{effective_start} .. {effective_end} (requested {user_start} .. {user_end})",
        )
    return effective_start, effective_end


def daterange_inclusive(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def group_by_panel(page):
    return page.locator("span.tm-toggle-group.tm-mode-group").filter(
        has=page.locator("span.tm-mode-header", has_text="Group By")
    )


def sync_group_by_toggles(page, timeout_ms: int) -> None:
    panel = group_by_panel(page)
    panel.wait_for(state="visible", timeout=timeout_ms)
    buttons = panel.locator("button.tm-toggle")
    count = buttons.count()
    for i in range(count):
        btn = buttons.nth(i)
        name = btn.inner_text().strip()
        want_on = name in GROUP_BY_ON
        for _ in range(3):
            is_on = btn.evaluate("el => el.classList.contains('on')")
            if is_on == want_on:
                break
            btn.click(timeout=timeout_ms)
        else:
            raise RuntimeError(f'Could not set Group By "{name}" to {"on" if want_on else "off"}')


def set_datetime_range(page, day: dt.date, timeout_ms: int) -> None:
    """Set both `datetime-local` inputs: 15:00 on ``day`` through 15:00 on ``day`` + 1 (local)."""
    start_v = f"{day.isoformat()}T15:00"
    end_day = day + dt.timedelta(days=1)
    end_v = f"{end_day.isoformat()}T15:00"
    inputs = page.locator("input.tm-dt-input")
    expect(inputs).to_have_count(2, timeout=timeout_ms)
    inputs.nth(0).fill(start_v, timeout=timeout_ms)
    inputs.nth(1).fill(end_v, timeout=timeout_ms)


def download_trade_markout_raw_text(page, day: dt.date, timeout_ms: int) -> str:
    """Run Fetch + Download JSON for 3pm ``day`` → 3pm ``day``+1; return response body text."""
    day_s = day.isoformat()
    print(f"  Fetching {day_s} ...")

    set_datetime_range(page, day, timeout_ms)
    sync_group_by_toggles(page, timeout_ms)

    fetch_btn = page.locator("button.fetch-btn").filter(has_text=re.compile(r"^Fetch$", re.I))
    fetch_btn.first.click(timeout=timeout_ms)

    download_btn = page.locator("button.fetch-btn").filter(has_text=re.compile(r"Download JSON", re.I))
    expect(download_btn).to_be_enabled(timeout=timeout_ms)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = Path(tf.name)

    try:
        with page.expect_download(timeout=timeout_ms) as dl_info:
            download_btn.click(timeout=timeout_ms)
        download = dl_info.value
        download.save_as(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        tmp_path.unlink(missing_ok=True)


def canonical_list_payload(rows: list) -> str:
    """Order-insensitive fingerprint for a list of dicts (e.g. groups, trades)."""
    normalized = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    normalized.sort()
    return "\n".join(normalized)


def compare_trade_markout_payloads(a: dict, b: dict) -> tuple[bool, str]:
    """
    Return (equal, summary). Equality ignores list order within ``groups`` / ``trades``.
    """
    ga, ta = a.get("groups") or [], a.get("trades") or []
    gb, tb = b.get("groups") or [], b.get("trades") or []
    same_groups = canonical_list_payload(ga) == canonical_list_payload(gb)
    same_trades = canonical_list_payload(ta) == canonical_list_payload(tb)
    equal = same_groups and same_trades
    lines = [
        f"groups count: {len(ga)} vs {len(gb)} (content match: {same_groups})",
        f"trades count: {len(ta)} vs {len(tb)} (content match: {same_trades})",
    ]
    if ga and gb and same_groups:
        lines.append("  -> group rows are identical (same multiset) under sorted JSON keys.")
    return equal, "\n".join(lines)


def inject_query_date_into_row_dicts(data: dict, query_date: str) -> None:
    """Add ``queryDate`` alongside row fields such as ``date_bucket`` (mutates ``data``)."""
    for key in ("groups", "trades"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        data[key] = [
            {**row, "queryDate": query_date} if isinstance(row, dict) else row
            for row in rows
        ]
    reports = data.get("reports")
    if isinstance(reports, list):
        fixed: list = []
        for r in reports:
            if not isinstance(r, dict):
                fixed.append(r)
                continue
            g = r.get("group")
            if isinstance(g, dict):
                fixed.append({**r, "group": {**g, "queryDate": query_date}})
            else:
                fixed.append(r)
        data["reports"] = fixed


def run_day(
    page,
    day: dt.date,
    json_dir: Path,
    timeout_ms: int,
) -> None:
    day_s = day.isoformat()
    text = download_trade_markout_raw_text(page, day, timeout_ms)
    out_path = json_dir / f"{day_s}.json"
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            inject_query_date_into_row_dicts(data, day_s)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"    Saved {day_s}.json")
    except json.JSONDecodeError:
        out_path.write_text(text, encoding="utf-8")
        print(f"    Saved (non-JSON) {day_s}.json")


def login_trade_markout_page(context, timeout_ms: int):
    """Navigate and optional form login; return page with datetime inputs visible."""
    page = context.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=30_000)
    except PlaywrightTimeoutError:
        pass

    user_field = page.locator(
        'input[type="text"], input[name*="user" i], input#username, input[name="username"]'
    )
    pass_field = page.locator('input[type="password"]')
    if user_field.count() and pass_field.count():
        try:
            if user_field.first.is_visible(timeout=3000):
                user_field.first.fill(USERNAME, timeout=5000)
                pass_field.first.fill(PASSWORD, timeout=5000)
                submit = page.get_by_role(
                    "button", name=re.compile(r"log ?in|sign ?in|submit", re.I)
                )
                if submit.count():
                    submit.first.click()
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            pass

    page.locator("input.tm-dt-input").first.wait_for(state="visible", timeout=timeout_ms)
    return page


def verify_fetch_two_days(
    day_a: dt.date,
    day_b: dt.date,
    headed: bool,
    timeout_ms: int,
) -> int:
    """
    Re-query the server for two dates in one session and compare payloads.
    Does not write JSON files. Prints whether ``groups``/``trades`` multisets match.
    """
    print(
        f"Verify-fetch: downloading fresh files for {day_a.isoformat()} and {day_b.isoformat()} "
        "(nothing written to disk).",
        flush=True,
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(
            http_credentials={"username": USERNAME, "password": PASSWORD},
            accept_downloads=True,
        )
        try:
            page = login_trade_markout_page(context, timeout_ms)
            text_a = download_trade_markout_raw_text(page, day_a, timeout_ms)
            text_b = download_trade_markout_raw_text(page, day_b, timeout_ms)
        finally:
            context.close()
            browser.close()

    try:
        data_a = json.loads(text_a)
        data_b = json.loads(text_b)
    except json.JSONDecodeError as e:
        print(f"Error: response is not valid JSON ({e}).", file=sys.stderr)
        return 1

    equal, detail = compare_trade_markout_payloads(data_a, data_b)
    print(detail, flush=True)
    if equal:
        print(
            f"\nRESULT: Payloads for {day_a} and {day_b} are the SAME after a fresh fetch.\n"
            "The server returned identical group/trade data for both date queries.",
            flush=True,
        )
    else:
        print(
            f"\nRESULT: Payloads for {day_a} and {day_b} DIFFER - downloads are not duplicates.",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download trade-markout JSON per day (3pm T -> 3pm T+1), saved as YYYY-MM-DD.json."
    )
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 4, 1),
        help="Start date (ISO), default 2026-04-01",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 4, 7),
        help="End date (ISO), default 2026-04-07",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Directory for YYYY-MM-DD.json (default: ./json next to this script)",
    )
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--timeout", type=int, default=120_000, help="Timeout ms (default 120000)")
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Use --start/--end exactly (do not shrink range based on existing json files). "
        "Use this to backfill dates when newer days are already saved.",
    )
    parser.add_argument(
        "--refetch-existing",
        action="store_true",
        help="Ignore --start/--end. Re-download once per YYYY-MM-DD.json already in --json-dir "
        "(sorted ascending, one day per fetch in a single browser session).",
    )
    parser.add_argument(
        "--verify-fetch",
        nargs=2,
        metavar=("DATE_A", "DATE_B"),
        help=(
            "Only: fetch two ISO dates from the live UI (no files written), "
            "then print whether groups/trades payloads match. "
            "Example: --verify-fetch 2026-04-05 2026-04-06"
        ),
    )
    args = parser.parse_args()

    if args.verify_fetch:
        day_a = dt.date.fromisoformat(args.verify_fetch[0])
        day_b = dt.date.fromisoformat(args.verify_fetch[1])
        return verify_fetch_two_days(day_a, day_b, args.headed, args.timeout)

    json_dir = (args.json_dir or (script_dir() / "json")).resolve()
    json_dir.mkdir(parents=True, exist_ok=True)

    if args.refetch_existing:
        days = sorted(parse_existing_dates(json_dir))
        if not days:
            print(f"Error: no YYYY-MM-DD.json files found in {json_dir}", file=sys.stderr)
            return 1
    elif args.full_range:
        if args.start > args.end:
            print("Error: start date is after end date.", file=sys.stderr)
            return 1
        start_d, end_d = args.start, args.end
        days = daterange_inclusive(start_d, end_d)
    else:
        existing = parse_existing_dates(json_dir)
        adjusted = adjust_range(args.start, args.end, existing)
        if adjusted is None:
            return 1
        start_d, end_d = adjusted
        days = daterange_inclusive(start_d, end_d)
    print(
        f"Dates to fetch ({len(days)}): {days[0]} .. {days[-1]} "
        "(each file: 15:00 that day -> 15:00 next day, local)",
        flush=True,
    )

    timeout_ms = args.timeout

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            http_credentials={"username": USERNAME, "password": PASSWORD},
            accept_downloads=True,
        )
        try:
            page = login_trade_markout_page(context, timeout_ms)
            for d in days:
                # One request per T: window [3pm T, 3pm T+1] → {T}.json
                run_day(page, d, json_dir, timeout_ms)
        finally:
            context.close()
            browser.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
