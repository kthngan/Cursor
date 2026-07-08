"""
Scrape bottom trade tables from market-pnl-v6, combine by time, and build net position.

Requires: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass

from playwright.sync_api import sync_playwright

BASE_URL = "https://poly-pnl.it9.win/market-pnl-v6"
USERNAME = "mm"
PASSWORD = "2047"

SPORT_ID_TO_NAME = {
    1: "Baseball",
    2: "Tennis",
    3: "Basketball",
    4: "Esports",
    5: "American Football",
    6: "Soccer",
    7: "Hockey",
    8: "MMA",
}


@dataclass
class MatchPositionResult:
    sport: str
    first_token: str
    second_token: str
    inplay_table: list[dict]
    pregame_table: list[dict]
    max_abs_net_inplay: float
    max_abs_net_pregame: float


@dataclass
class TradeRow:
    time_raw: str
    time_sort: dt.datetime
    token: str
    rspt: str
    price: str
    size: str
    liab: float


def parse_liab(value: str) -> float:
    s = str(value or "").strip().replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return 0.0
    return float(s)


def parse_trade_time(value: str, year: int | None = None) -> dt.datetime:
    year = year or dt.date.today().year
    return dt.datetime.strptime(f"{year}/{value.strip()}", "%Y/%m/%d %H:%M:%S")


def build_url(
    polymarket_market_id: str,
    wallet: str,
    is_private: bool,
) -> str:
    private = "true" if is_private else "false"
    return (
        f"{BASE_URL}?polymarket_market_id={polymarket_market_id}"
        f"&wallet={wallet}&is_private={private}"
    )


def p_token(rspt: str) -> str:
    parts = rspt.split()
    return parts[2] if len(parts) >= 3 else ""


def sport_name_from_id(sport_id: int | str | None) -> str:
    if sport_id is None or sport_id == "":
        return ""
    try:
        return SPORT_ID_TO_NAME.get(int(sport_id), str(sport_id))
    except (TypeError, ValueError):
        return str(sport_id)


def max_abs_net_position(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return max(abs(row["Net position"]) for row in rows)


def scrape_trade_tables(
    polymarket_market_id: str,
    wallet: str,
    is_private: bool = False,
    *,
    year: int | None = None,
    headless: bool = True,
) -> tuple[list[TradeRow], list[TradeRow], str, str, str]:
    url = build_url(polymarket_market_id, wallet, is_private)
    tables: list[list[TradeRow]] = []
    token_names: list[str] = []
    sport = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            http_credentials={"username": USERNAME, "password": PASSWORD}
        )
        page = context.new_page()

        def on_response(response) -> None:
            nonlocal sport
            if sport or "market-pnl" not in response.url or response.status != 200:
                return
            try:
                payload = response.json()
            except Exception:
                return
            sport = sport_name_from_id(payload.get("unified_sport_id"))

        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.locator("table.trades-table").first.wait_for(state="visible", timeout=120_000)
        page.wait_for_timeout(8_000)

        trade_tables = page.locator("table.trades-table")
        for ti in range(trade_tables.count()):
            table = trade_tables.nth(ti)
            token = page.evaluate(
                """(el) => {
                    let cur = el;
                    for (let d = 0; d < 6; d++) {
                        cur = cur.parentElement;
                        if (!cur) break;
                        const header = cur.querySelector(
                            'h1,h2,h3,h4,.token-name,.outcome-name,.section-title,[class*=title]'
                        );
                        if (header) return header.textContent.trim();
                    }
                    return '';
                }""",
                table.element_handle(),
            )
            token_names.append(token)

            rows: list[TradeRow] = []
            body_rows = table.locator("tbody tr")
            for ri in range(body_rows.count()):
                row = body_rows.nth(ri)
                cells = [c.inner_text().strip() for c in row.locator("td").all()]
                if len(cells) < 5:
                    continue

                rows.append(
                    TradeRow(
                        time_raw=cells[0],
                        time_sort=parse_trade_time(cells[0], year=year),
                        token=token,
                        rspt=cells[1],
                        price=cells[2],
                        size=cells[3],
                        liab=parse_liab(cells[4]),
                    )
                )
            tables.append(rows)

        browser.close()

    if not tables:
        raise RuntimeError("No trade tables found")

    if len(tables) == 1:
        return tables[0], [], token_names[0], "", sport

    return tables[0], tables[1], token_names[0], token_names[1], sport


def combine_tables(
    table1: list[TradeRow],
    table2: list[TradeRow],
    *,
    p_filter: str,
    second_token: str,
) -> list[dict]:
    filtered = [
        r for r in table1 + table2 if p_token(r.rspt) == p_filter
    ]
    combined = sorted(filtered, key=lambda r: (r.time_sort, r.token))
    net = 0.0
    output: list[dict] = []
    for row in combined:
        liab = (
            row.liab * -1
            if second_token and row.token == second_token
            else row.liab
        )
        net += liab
        output.append(
            {
                "Time": row.time_raw,
                "Token": row.token,
                "R S P T": row.rspt,
                "Price": row.price,
                "Size": row.size,
                "Liab": liab,
                "Net position": net,
            }
        )
    return output


def get_match_position(
    polymarket_market_id: str,
    wallet: str,
    is_private: bool = False,
    *,
    year: int | None = None,
    headless: bool = True,
) -> MatchPositionResult:
    table1, table2, first_token, second_token, sport = scrape_trade_tables(
        polymarket_market_id,
        wallet,
        is_private,
        year=year,
        headless=headless,
    )
    inplay_table = combine_tables(
        table1, table2, p_filter="I", second_token=second_token
    )
    pregame_table = combine_tables(
        table1, table2, p_filter="P", second_token=second_token
    )
    return MatchPositionResult(
        sport=sport,
        first_token=first_token,
        second_token=second_token,
        inplay_table=inplay_table,
        pregame_table=pregame_table,
        max_abs_net_inplay=max_abs_net_position(inplay_table),
        max_abs_net_pregame=max_abs_net_position(pregame_table),
    )


def print_table(rows: list[dict], title: str = "") -> None:
    if title:
        print()
        print(title)
        print("=" * len(title))
    if not rows:
        print("(no rows)")
        return

    columns = ["Time", "Token", "R S P T", "Price", "Size", "Liab", "Net position"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            if col in {"Liab", "Net position"}:
                text = f"{row[col]:,.2f}"
            else:
                text = str(row[col])
            widths[col] = max(widths[col], len(text))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        parts = []
        for col in columns:
            if col in {"Liab", "Net position"}:
                text = f"{row[col]:,.2f}"
            else:
                text = str(row[col])
            parts.append(text.ljust(widths[col]))
        print("  ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape and combine market-pnl trade tables.")
    parser.add_argument("--market-id", default="2454773")
    parser.add_argument("--wallet", default="0x84AD9c5C547A82EC9a08547b94bD922446e5BfB7")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--year", type=int, default=None, help="Year for MM/DD timestamps")
    parser.add_argument("--no-headless", action="store_true")
    args = parser.parse_args()

    result = get_match_position(
        args.market_id,
        args.wallet,
        args.private,
        year=args.year,
        headless=not args.no_headless,
    )

    print(f"Sport: {result.sport}")
    print(f"Max |Net position| (P = I): {result.max_abs_net_inplay:,.2f}")
    print(f"Max |Net position| (P = P): {result.max_abs_net_pregame:,.2f}")

    print_table(
        result.inplay_table,
        title=(
            f"P = I (Inplay) — 1st token: {result.first_token}, "
            f"2nd token: {result.second_token} (Liab × -1)"
        ),
    )
    print_table(
        result.pregame_table,
        title=(
            f"P = P (Pregame) — 1st token: {result.first_token}, "
            f"2nd token: {result.second_token} (Liab × -1)"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
