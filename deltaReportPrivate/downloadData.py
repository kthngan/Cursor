"""
Download private-trade-analysis table rows per day and save as JSON.

Each day T is queried as a single window from 3pm local on T through 3pm local
on T+1, and saved as ``{T}.json`` (ISO date).

Adjusts the requested date range against existing files in Data/deltaReportPrivate/json:
  - Skips re-fetching days before the latest file date.
  - Starts at max(user_start, latest_existing_date) so boundary day is refreshed.

Requires: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect
from playwright.sync_api import sync_playwright

BASE_URL = "https://poly-pnl.it9.win/private-trade-analysis"
USERNAME = os.environ.get("POLY_PNL_USERNAME", "")
PASSWORD = os.environ.get("POLY_PNL_PASSWORD", "")
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "deltaReportPrivate"

# Human-readable labels for desired "Group By" toggles.
GROUP_BY_ON = frozenset(
    {
        "Sport",
        "League",
        "Date",
        "Tier",
        "Role",
        "TIF",
        "Stage",
        "Size Factor",
        "Trade Price",
        "Lat ACK - TradeMatchWS",
        "Main Book",
        "ROI CLV%",
        "Tag",
        "From Start",
        "Edge CLV",
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


def normalize_group_label(label: str) -> str:
    """
    Normalize toggle labels so punctuation/arrow variations still match.
    Example: ``Lat ACK→TradeMatchWS`` and ``Lat ACK - TradeMatchWS`` normalize equally.
    """
    s = label.strip().casefold().replace("→", "-")
    return re.sub(r"[^a-z0-9]+", "", s)


GROUP_BY_ON_NORM = frozenset(normalize_group_label(x) for x in GROUP_BY_ON)


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
        want_on = normalize_group_label(name) in GROUP_BY_ON_NORM
        for _ in range(3):
            is_on = btn.evaluate("el => el.classList.contains('on')")
            if is_on == want_on:
                break
            btn.click(timeout=timeout_ms)
        else:
            raise RuntimeError(f'Could not set Group By "{name}" to {"on" if want_on else "off"}')


def set_datetime_range(page, day: dt.date, timeout_ms: int) -> None:
    start_v = f"{day.isoformat()}T15:00"
    end_day = day + dt.timedelta(days=1)
    end_v = f"{end_day.isoformat()}T15:00"
    inputs = page.locator("input.tm-dt-input")
    expect(inputs).to_have_count(2, timeout=timeout_ms)
    inputs.nth(0).fill(start_v, timeout=timeout_ms)
    inputs.nth(1).fill(end_v, timeout=timeout_ms)


def extract_first_visible_table(page) -> dict:
    data = page.evaluate(
        """
() => {
  const textOf = (el) => (el && el.textContent ? el.textContent.trim() : "");
  const isVisible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const tables = Array.from(document.querySelectorAll("table"));
  for (const table of tables) {
    if (!isVisible(table)) continue;
    const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
    if (bodyRows.length === 0) continue;

    let headers = Array.from(table.querySelectorAll("thead th")).map(textOf).filter(Boolean);
    if (headers.length === 0) {
      const firstRowCells = Array.from(bodyRows[0].querySelectorAll("th, td"));
      headers = firstRowCells.map((_, i) => `col_${i + 1}`);
    }

    const rows = bodyRows.map((tr) => {
      const cells = Array.from(tr.querySelectorAll("th, td")).map(textOf);
      const row = {};
      headers.forEach((h, i) => {
        row[h || `col_${i + 1}`] = cells[i] ?? "";
      });
      return row;
    }).filter((row) => Object.values(row).some((v) => String(v || "").trim() !== ""));

    return {
      headers,
      rows,
      row_count: rows.length
    };
  }
  return null;
}
"""
    )
    if not data:
        raise RuntimeError("Could not find a visible results table with body rows.")
    return data


def wait_for_table_ready(page, timeout_ms: int) -> None:
    """
    Wait until the first visible results table is populated and stable.
    This avoids saving while the table is still rendering/updating.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    last_sig: tuple[int, int, int] | None = None
    stable_hits = 0
    while time.monotonic() < deadline:
        snap = page.evaluate(
            """
() => {
  const textOf = (el) => (el && el.textContent ? el.textContent.trim() : "");
  const isVisible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const tables = Array.from(document.querySelectorAll("table"));
  for (const table of tables) {
    if (!isVisible(table)) continue;
    const headers = Array.from(table.querySelectorAll("thead th")).map(textOf).filter(Boolean);
    const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
    const nonEmptyRows = bodyRows.filter((tr) => {
      const cells = Array.from(tr.querySelectorAll("th, td")).map(textOf);
      return cells.some((v) => v !== "");
    });
    const firstNonEmptyCellLen =
      nonEmptyRows.length > 0
        ? Array.from(nonEmptyRows[0].querySelectorAll("th, td"))
            .map(textOf)
            .find((v) => v !== "")?.length || 0
        : 0;
    return {
      rowCount: bodyRows.length,
      nonEmptyRowCount: nonEmptyRows.length,
      headerCount: headers.length,
      firstNonEmptyCellLen
    };
  }
  return { rowCount: 0, nonEmptyRowCount: 0, headerCount: 0, firstNonEmptyCellLen: 0 };
}
"""
        )
        row_count = int(snap.get("rowCount") or 0)
        non_empty = int(snap.get("nonEmptyRowCount") or 0)
        header_count = int(snap.get("headerCount") or 0)
        first_cell_len = int(snap.get("firstNonEmptyCellLen") or 0)
        ready = row_count > 0 and non_empty > 0 and header_count > 0 and first_cell_len > 0
        sig = (row_count, non_empty, first_cell_len)
        if ready:
            if sig == last_sig:
                stable_hits += 1
            else:
                stable_hits = 1
                last_sig = sig
            if stable_hits >= 3:
                return
        else:
            stable_hits = 0
            last_sig = None
        page.wait_for_timeout(400)
    raise RuntimeError("Timed out waiting for results table to be ready.")


def fetch_private_trade_table(page, day: dt.date, timeout_ms: int) -> dict:
    day_s = day.isoformat()
    print(f"  Fetching {day_s} ...")

    set_datetime_range(page, day, timeout_ms)
    sync_group_by_toggles(page, timeout_ms)

    fetch_btn = page.locator("button.fetch-btn").filter(has_text=re.compile(r"^Fetch$", re.I))
    fetch_btn.first.click(timeout=timeout_ms)

    wait_for_table_ready(page, timeout_ms)
    return extract_first_visible_table(page)


def fetch_private_trade_groups(page, day: dt.date, timeout_ms: int) -> dict:
    """
    Run Fetch and capture the full backend JSON payload.
    Falls back to table scraping only if response parsing fails.
    """
    day_s = day.isoformat()
    print(f"  Fetching {day_s} ...")

    set_datetime_range(page, day, timeout_ms)
    sync_group_by_toggles(page, timeout_ms)

    captured_bytes: list[bytes] = []

    def _capture_private_trade_route(route) -> None:
        request = route.request
        if (
            "/api/v1/polymarket/private-trade-analysis" not in request.url
            or request.method.upper() != "GET"
        ):
            route.continue_()
            return
        response = route.fetch(timeout=timeout_ms)
        captured_bytes.append(response.body())
        route.fulfill(response=response)

    page.route("**/api/v1/polymarket/private-trade-analysis**", _capture_private_trade_route)
    fetch_btn = page.locator("button.fetch-btn").filter(has_text=re.compile(r"^Fetch$", re.I))
    try:
        fetch_btn.first.click(timeout=timeout_ms)
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        while not captured_bytes and time.monotonic() < deadline:
            page.wait_for_timeout(100)
    finally:
        page.unroute("**/api/v1/polymarket/private-trade-analysis**", _capture_private_trade_route)

    if not captured_bytes:
        raise RuntimeError(f"Timed out waiting for private-trade-analysis API response for {day_s}")

    payload = json.loads(captured_bytes[-1].decode("utf-8"))

    wait_for_table_ready(page, timeout_ms)
    table_payload = extract_first_visible_table(page)

    if not isinstance(payload, dict):
        payload = {}
    groups = payload.get("groups")
    if not isinstance(groups, list):
        groups = []

    return {
        "api_payload": payload,
        "groups": groups,
        "table": table_payload,
    }


def login_page(context, timeout_ms: int):
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


def run_day(
    page,
    day: dt.date,
    json_dir: Path,
    timeout_ms: int,
) -> None:
    day_s = day.isoformat()
    fetched = fetch_private_trade_groups(page, day, timeout_ms)
    groups = fetched.get("groups") or []
    table_payload = fetched.get("table") or {}
    api_payload = fetched.get("api_payload") or {}
    out_path = json_dir / f"{day_s}.json"
    payload = {
        "queryDate": day_s,
        "url": BASE_URL,
        "groupBy": sorted(GROUP_BY_ON),
        "groups": groups,
        "table": table_payload,
        "apiMeta": {
            "groups_count": len(groups),
            "table_row_count": int(table_payload.get("row_count") or 0),
            "success": api_payload.get("success"),
            "summary": api_payload.get("summary"),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"    Saved {day_s}.json (groups={len(groups)}, table_rows={table_payload.get('row_count', 0)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download private-trade-analysis table per day (3pm T -> 3pm T+1), saved as YYYY-MM-DD.json."
    )
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 5, 27),
        help="Start date (ISO), default 2026-05-27",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=dt.date(2026, 5, 27),
        help="End date (ISO), default 2026-05-27",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help=f"Directory for YYYY-MM-DD.json (default: {DATA_DIR / 'json'})",
    )
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--timeout", type=int, default=300_000, help="Timeout ms (default 300000)")
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Use --start/--end exactly (do not shrink range based on existing json files).",
    )
    parser.add_argument(
        "--refetch-existing",
        action="store_true",
        help="Ignore --start/--end. Re-download once per YYYY-MM-DD.json already in --json-dir.",
    )
    args = parser.parse_args()

    json_dir = (args.json_dir or (DATA_DIR / "json")).resolve()
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
            page = login_page(context, timeout_ms)
            for d in days:
                run_day(page, d, json_dir, timeout_ms)
        finally:
            context.close()
            browser.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
