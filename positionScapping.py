"""
Scrape market results from pnl-history-v3 (algo 15), then call get_match_position
for each V6 market URL and aggregate max net positions.

Requires: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

from matchPositionScrap import get_match_position

PNL_HISTORY_URL = "https://poly-pnl.it9.win/pnl-history-v3"
USERNAME = "mm"
PASSWORD = "2047"
DEFAULT_WALLET_LABEL = "algo 15"
ALGO_15_WALLET = "0x84AD9c5C547A82EC9a08547b94bD922446e5BfB7"
V6_BASE = "https://poly-pnl.it9.win/market-pnl-v6"
RESULT_COLUMNS = [
    "Time",
    "Market",
    "PM ID",
    "V6 URL",
    "Sport",
    "P=P Max |Net|",
    "P=I Max |Net|",
    "Error",
]
DEFAULT_CSV = Path(__file__).resolve().parent / "position" / "position_scraping_results.csv"


@dataclass
class MarketResultRow:
    time: str
    market: str
    polymarket_market_id: str
    v6_url: str
    sport: str
    max_net_pregame: float | None
    max_net_inplay: float | None
    error: str


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def select_wallet_only(page, wallet_label: str) -> None:
    page.locator(".wgs.cms-wallets").click()
    page.wait_for_timeout(500)
    page.locator(".wgs.cms-wallets").locator("text=×").first.click()
    page.wait_for_timeout(500)

    target = _norm_label(wallet_label)
    for item in page.locator(".wgs-item").all():
        text = item.inner_text().strip()
        if _norm_label(text).endswith(target) or target in _norm_label(text):
            item.click()
            break
    else:
        raise RuntimeError(f"Wallet not found in dropdown: {wallet_label!r}")

    page.wait_for_timeout(300)


def apply_filters(page) -> None:
    page.locator("button.filter-btn").click()
    page.wait_for_timeout(8_000)


def date_window_for_input(date_text: str) -> tuple[str, str]:
    selected_date = dt.date.fromisoformat(date_text)
    start = dt.datetime.combine(selected_date, dt.time(hour=14))
    end = start + dt.timedelta(days=1)
    return start.strftime("%Y-%m-%dT%H:%M"), end.strftime("%Y-%m-%dT%H:%M")


def set_date_filter(page, date_text: str | None) -> None:
    if not date_text:
        return

    start, end = date_window_for_input(date_text)
    fields = page.locator('input[type="datetime-local"]')
    if fields.count() < 2:
        raise RuntimeError("Expected two datetime-local inputs for start/end filters")

    fields.nth(0).fill(start)
    fields.nth(1).fill(end)
    page.wait_for_timeout(300)


def extract_market_rows(page, wallet: str) -> list[dict]:
    return page.evaluate(
        """({ wallet, v6Base }) => {
            const rows = [];
            for (const row of document.querySelectorAll('.dyn-grid.dyn-grid-row')) {
                const v6 = [...row.querySelectorAll('button,a')]
                    .find((el) => el.textContent.trim() === 'V6');
                const pmBadge = row.querySelector('.id-badge.pm')?.textContent?.trim() || '';
                const marketId = pmBadge.replace(/^PM/i, '');
                let market = '';
                for (const span of row.querySelectorAll('.dyn-col-name span:not(.id-badge)')) {
                    const text = span.textContent.trim();
                    if (text && !/^UF\\d+/i.test(text) && !/^PM\\d+/i.test(text)) {
                        market = text;
                        break;
                    }
                }
                const cols = [...row.querySelectorAll('.dyn-col')].map((el) => el.innerText.trim());
                const time = cols[0] || '';
                const v6Url = v6 && marketId
                    ? `${v6Base}?polymarket_market_id=${marketId}&wallet=${wallet}`
                    : '';
                rows.push({ time, market, polymarket_market_id: marketId, v6_url: v6Url });
            }
            return rows;
        }""",
        {"wallet": wallet, "v6Base": V6_BASE},
    )


def scrape_market_results(
    wallet_label: str = DEFAULT_WALLET_LABEL,
    *,
    wallet_address: str = ALGO_15_WALLET,
    date: str | None = None,
    headless: bool = True,
) -> list[dict]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            http_credentials={"username": USERNAME, "password": PASSWORD}
        )
        page = context.new_page()
        page.goto(PNL_HISTORY_URL, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(5_000)
        set_date_filter(page, date)
        select_wallet_only(page, wallet_label)
        apply_filters(page)
        rows = extract_market_rows(page, wallet_address)
        browser.close()
    return rows


def parse_v6_url(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    market_id = (params.get("polymarket_market_id") or [""])[0]
    wallet = (params.get("wallet") or [""])[0]
    if not market_id or not wallet:
        raise ValueError(f"Invalid V6 URL: {url}")
    return market_id, wallet


def process_market_rows(
    market_rows: list[dict],
    *,
    headless: bool = True,
    limit: int | None = None,
) -> list[MarketResultRow]:
    total = len(market_rows)
    results: list[MarketResultRow] = []
    fetched_v6 = 0

    for idx, row in enumerate(market_rows, start=1):
        base = MarketResultRow(
            time=row.get("time", ""),
            market=row.get("market", ""),
            polymarket_market_id=row.get("polymarket_market_id", ""),
            v6_url=row.get("v6_url", ""),
            sport="",
            max_net_pregame=None,
            max_net_inplay=None,
            error="",
        )

        if not base.v6_url:
            base.error = "No V6 URL"
            results.append(base)
            print(f"Completed {idx}/{total} (no V6 URL)", file=sys.stderr)
            continue

        if limit is not None and fetched_v6 >= limit:
            base.error = "Skipped (--limit)"
            results.append(base)
            print(f"Completed {idx}/{total} (skipped, --limit reached)", file=sys.stderr)
            continue

        try:
            market_id, wallet = parse_v6_url(base.v6_url)
            match = get_match_position(
                market_id,
                wallet,
                is_private=False,
                headless=headless,
            )
            base.sport = match.sport
            base.max_net_pregame = match.max_abs_net_pregame
            base.max_net_inplay = match.max_abs_net_inplay
            fetched_v6 += 1
        except Exception as exc:
            base.error = str(exc)

        results.append(base)
        label = base.market or base.polymarket_market_id or "row"
        print(f"Completed {idx}/{total}: {label}", file=sys.stderr)

    return results


def results_to_rows(results: list[MarketResultRow]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in results:
        rows.append(
            {
                "Time": item.time,
                "Market": item.market,
                "PM ID": item.polymarket_market_id,
                "V6 URL": item.v6_url,
                "Sport": item.sport,
                "P=P Max |Net|": (
                    f"{item.max_net_pregame:.2f}"
                    if item.max_net_pregame is not None
                    else ""
                ),
                "P=I Max |Net|": (
                    f"{item.max_net_inplay:.2f}"
                    if item.max_net_inplay is not None
                    else ""
                ),
                "Error": item.error,
            }
        )
    return rows


def market_result_from_csv_row(row: dict[str, str]) -> MarketResultRow:
    def _float_or_none(value: str) -> float | None:
        value = (value or "").strip()
        if not value:
            return None
        return float(value.replace(",", ""))

    return MarketResultRow(
        time=row.get("Time", ""),
        market=row.get("Market", ""),
        polymarket_market_id=row.get("PM ID", ""),
        v6_url=row.get("V6 URL", ""),
        sport=row.get("Sport", ""),
        max_net_pregame=_float_or_none(row.get("P=P Max |Net|", "")),
        max_net_inplay=_float_or_none(row.get("P=I Max |Net|", "")),
        error=row.get("Error", ""),
    )


def fetch_match_position_for_row(
    row: MarketResultRow,
    *,
    headless: bool = True,
) -> MarketResultRow:
    if not row.v6_url:
        row.error = "No V6 URL"
        return row

    try:
        market_id, wallet = parse_v6_url(row.v6_url)
        match = get_match_position(
            market_id,
            wallet,
            is_private=False,
            headless=headless,
        )
        row.sport = match.sport
        row.max_net_pregame = match.max_abs_net_pregame
        row.max_net_inplay = match.max_abs_net_inplay
        row.error = ""
    except Exception as exc:
        row.error = str(exc)
    return row


def retry_errors_in_csv(
    csv_path: Path,
    *,
    headless: bool = True,
) -> list[MarketResultRow]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    results = [market_result_from_csv_row(row) for row in rows]
    retry_rows = [
        (idx, item)
        for idx, item in enumerate(results)
        if item.error and item.error != "No V6 URL"
    ]
    total = len(retry_rows)

    for n, (idx, item) in enumerate(retry_rows, start=1):
        label = item.market or item.polymarket_market_id or f"row {idx + 1}"
        updated = fetch_match_position_for_row(item, headless=headless)
        results[idx] = updated
        status = "ok" if not updated.error else updated.error.splitlines()[0]
        print(f"Retry {n}/{total}: {label} -> {status}", file=sys.stderr)

    save_results_csv(results, csv_path)
    return results


def save_results_csv(results: list[MarketResultRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = results_to_rows(results)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_results_table(results: list[MarketResultRow]) -> None:
    columns = RESULT_COLUMNS
    rows = results_to_rows(results)
    display_rows: list[dict[str, str]] = []
    for row in rows:
        display = dict(row)
        for col in ("P=P Max |Net|", "P=I Max |Net|"):
            if display[col]:
                display[col] = f"{float(display[col]):,.2f}"
        display_rows.append(display)

    widths = {col: len(col) for col in columns}
    for row in display_rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col])))

    print()
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in display_rows:
        print("  ".join(str(row[col]).ljust(widths[col]) for col in columns))


def run_position_scraping(
    wallet_label: str = DEFAULT_WALLET_LABEL,
    *,
    date: str | None = None,
    headless: bool = True,
    limit: int | None = None,
) -> list[MarketResultRow]:
    market_rows = scrape_market_results(wallet_label, date=date, headless=headless)
    return process_market_rows(market_rows, headless=headless, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape pnl-history-v3 market rows and aggregate match positions."
    )
    parser.add_argument("--wallet-label", default=DEFAULT_WALLET_LABEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N market-result rows (for testing).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Date to scrape as YYYY-MM-DD. Uses 2pm on that date through "
            "2pm on the following date."
        ),
    )
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV output path for the final results table.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run only rows with errors in an existing CSV and update it.",
    )
    args = parser.parse_args()

    if args.retry_errors:
        if not args.output.is_file():
            print(f"CSV not found: {args.output}", file=sys.stderr)
            return 1
        results = retry_errors_in_csv(args.output, headless=not args.no_headless)
    else:
        results = run_position_scraping(
            args.wallet_label,
            date=args.date,
            headless=not args.no_headless,
            limit=args.limit,
        )
        save_results_csv(results, args.output)

    print(f"Saved results to {args.output}", file=sys.stderr)
    print_results_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
