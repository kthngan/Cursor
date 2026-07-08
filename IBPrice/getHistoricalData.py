"""
Fetch historical HSI futures data from Interactive Brokers.

Behavior:
1) Resolve HSI front-month contract for the requested date (YYYY-MM-DD -> YYYYMM).
2) Try historical tick-by-tick first (TRADES + BID_ASK).
3) If tick data is unavailable/empty, fall back to 1-minute bars:
   - TRADES (last price, VWAP, total volume)
   - BID (bid price)
   - ASK (offer price)
4) Save one CSV per date under Data/IBPrice/historicalData/.

Example:
  python getHistoricalData.py --date 2026-01-01
"""

from __future__ import annotations

import argparse
import calendar
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from ib_insync import Future, IB, util

HK_TZ = ZoneInfo("Asia/Hong_Kong")
INFO_ERROR_CODES = frozenset({2104, 2106, 2119, 2158})
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "IBPrice"


@dataclass
class AppConfig:
    host: str
    port: int
    client_id: int
    date_str: str
    symbol: str
    exchange: str
    currency: str
    use_rth: bool
    out_dir: Path
    verbose: bool


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Get IB historical data for HSI front-month futures.")
    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    parser.add_argument("--port", type=int, default=7497, help="TWS: 7497 live / 7496 paper")
    parser.add_argument("--client-id", type=int, default=11, dest="client_id")
    parser.add_argument("--date", required=True, help="Historical date in YYYY-MM-DD")
    parser.add_argument("--symbol", default="HSI", help="Futures root symbol")
    parser.add_argument("--exchange", default="HKFE", help="Futures exchange")
    parser.add_argument("--currency", default="HKD", help="Contract currency")
    parser.add_argument(
        "--use-rth",
        action="store_true",
        help="Use regular trading hours only (default: full session)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATA_DIR / "historicalData"),
        help=f"Output folder for CSV files (default: {DATA_DIR / 'historicalData'})",
    )
    parser.add_argument("--verbose", action="store_true", help="Print more diagnostics")
    args = parser.parse_args()

    # Validate date format early.
    datetime.strptime(args.date, "%Y-%m-%d")

    return AppConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        date_str=args.date,
        symbol=args.symbol.upper(),
        exchange=args.exchange,
        currency=args.currency,
        use_rth=args.use_rth,
        out_dir=Path(args.out_dir),
        verbose=args.verbose,
    )


def expiry_from_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year}{dt.month:02d}"


def summarize_contracts(details: list, symbol: str, limit: int = 20) -> str:
    rows: list[str] = []
    sym = symbol.upper()
    for d in details:
        c = d.contract
        if not c or c.secType != "FUT" or c.symbol != sym:
            continue
        cm = getattr(d, "contractMonth", "") or ""
        ltd = c.lastTradeDateOrContractMonth or ""
        exp = contract_expiry_date(c, d)
        rows.append(
            f"  localSymbol={c.localSymbol!r} month={ltd!r} contractMonth={cm!r} "
            f"expiryDate={exp} conId={c.conId}"
        )
    if not rows:
        return "  (no FUT contracts returned)"
    body = "\n".join(rows[:limit])
    if len(rows) > limit:
        body += f"\n  ... and {len(rows) - limit} more"
    return body


def _parse_yyyymmdd(s: str):
    if len(s) >= 8 and s[:8].isdigit():
        return datetime.strptime(s[:8], "%Y%m%d").date()
    return None


def _parse_yyyymm_end_of_month(s: str):
    if len(s) >= 6 and s[:6].isdigit():
        y = int(s[:4])
        m = int(s[4:6])
        d = calendar.monthrange(y, m)[1]
        return datetime(y, m, d).date()
    return None


def contract_expiry_date(contract, detail):
    """
    Return contract expiry date as a date object.
    Prefer full YYYYMMDD fields; fall back to end-of-month for YYYYMM.
    """
    candidates = [
        (contract.lastTradeDateOrContractMonth or "").strip(),
        (getattr(detail, "realExpirationDate", None) or "").strip(),
        (getattr(detail, "contractMonth", None) or "").strip(),
    ]
    for raw in candidates:
        dt = _parse_yyyymmdd(raw)
        if dt:
            return dt
    for raw in candidates:
        dt = _parse_yyyymm_end_of_month(raw)
        if dt:
            return dt
    return None


def select_front_month_contract(details: list, symbol: str, target_date):
    sym = symbol.upper()
    rows = []
    for d in details:
        c = d.contract
        if not c or c.secType != "FUT" or c.symbol != sym:
            continue
        exp_date = contract_expiry_date(c, d)
        rows.append((exp_date, c))
    if not rows:
        return None, None

    # Front month by date: choose the nearest expiry strictly AFTER target date.
    # This enforces the requested behavior: if date == expiry date, roll to next month.
    future_rows = [r for r in rows if r[0] is not None and r[0] > target_date]
    if future_rows:
        future_rows.sort(key=lambda r: (r[0], r[1].lastTradeDateOrContractMonth or "", r[1].conId))
        return future_rows[0][1], future_rows[0][0]

    # Safety fallback: if all expiries are <= target date or unparseable, pick nearest available.
    rows.sort(
        key=lambda r: (
            r[0] is None,
            r[0] if r[0] is not None else datetime.max.date(),
            r[1].lastTradeDateOrContractMonth or "",
            r[1].conId,
        )
    )
    return rows[0][1], rows[0][0]


def to_hk_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).astimezone(HK_TZ)
    return value.astimezone(HK_TZ)


def ib_time_str(dt: datetime) -> str:
    # Use IB's UTC notation: YYYYMMDD-HH:MM:SS
    return dt.astimezone(timezone.utc).strftime("%Y%m%d-%H:%M:%S")


def date_window_hk(date_str: str) -> tuple[datetime, datetime]:
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=HK_TZ)
    end = start + timedelta(days=1)
    return start, end


def connect_with_fallback_ports(ib: IB, host: str, port: int, client_id: int, verbose: bool) -> int:
    """
    Try user-provided port first, then common TWS/IBG API ports.
    Returns the connected port.
    """
    candidates = [port, 7497, 7496, 4002, 4001]
    seen: set[int] = set()
    ordered_ports = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            ordered_ports.append(p)

    last_err: Exception | None = None
    for p in ordered_ports:
        print(f"Connecting to IB {host}:{p} clientId={client_id} ...", flush=True)
        try:
            # Force read-only API session: no order placement/binding behavior.
            ib.connect(host, p, clientId=client_id, timeout=15, readonly=True)
            return p
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  connection failed on port {p}: {e}", file=sys.stderr, flush=True)
    raise RuntimeError(
        f"IB connection failed on ports {ordered_ports}. "
        "Please start TWS/IB Gateway and verify API host/port permissions."
    ) from last_err


def fetch_historical_ticks(
    ib: IB,
    contract,
    start: datetime,
    end: datetime,
    what_to_show: str,
    use_rth: bool,
    verbose: bool = False,
) -> list:
    all_ticks = []
    cursor = start
    while cursor < end:
        chunk = ib.reqHistoricalTicks(
            contract=contract,
            startDateTime=ib_time_str(cursor),
            endDateTime=ib_time_str(end),
            numberOfTicks=1000,
            whatToShow=what_to_show,
            useRth=use_rth,
            ignoreSize=False,
            miscOptions=[],
        )
        if not chunk:
            break

        for tick in chunk:
            ts = to_hk_datetime(tick.time)
            if start <= ts < end:
                all_ticks.append(tick)

        last_ts = to_hk_datetime(chunk[-1].time)
        next_cursor = last_ts + timedelta(seconds=1)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if verbose:
            print(f"  fetched {len(chunk)} {what_to_show} ticks, next cursor {cursor}", flush=True)

    return all_ticks


def fetch_minute_bars(ib: IB, contract, end: datetime, use_rth: bool) -> tuple[list, list, list]:
    end_str = ib_time_str(end)
    bars_trades = ib.reqHistoricalData(
        contract,
        endDateTime=end_str,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
    )
    bars_bid = ib.reqHistoricalData(
        contract,
        endDateTime=end_str,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="BID",
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
    )
    bars_ask = ib.reqHistoricalData(
        contract,
        endDateTime=end_str,
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="ASK",
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False,
    )
    return bars_trades, bars_bid, bars_ask


def bar_datetime_hk(value) -> datetime:
    if isinstance(value, datetime):
        return to_hk_datetime(value)
    # ib_insync may return "YYYYMMDD HH:MM:SS" (or with extra spaces) strings.
    text = str(value)
    text = " ".join(text.split())
    parsed = datetime.strptime(text, "%Y%m%d %H:%M:%S")
    return parsed.replace(tzinfo=HK_TZ)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    columns = [
        "timestamp_hkt",
        "event_type",
        "last_price",
        "last_size",
        "bid_price",
        "bid_size",
        "offer_price",
        "ask_size",
        "vwap_price",
        "total_volume",
        "exchange",
        "special_conditions",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    patch = getattr(util, "patch_asyncio", None) or getattr(util, "patchAsyncio", None)
    if patch:
        patch()

    cfg = parse_args()
    start_hk, end_hk = date_window_hk(cfg.date_str)
    target_date = start_hk.date()

    out_dir = cfg.out_dir if cfg.out_dir.is_absolute() else DATA_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.date_str}.csv"

    ib = IB()
    if cfg.verbose:
        def on_error(req_id: int, code: int, msg: str, contract) -> None:
            if code in INFO_ERROR_CODES:
                return
            print(f"IB error {code} reqId={req_id}: {msg}", file=sys.stderr, flush=True)

        ib.errorEvent += on_error

    try:
        connected_port = connect_with_fallback_ports(ib, cfg.host, cfg.port, cfg.client_id, cfg.verbose)
        print(f"Connected on port {connected_port}.", flush=True)

        raw = Future(
            symbol=cfg.symbol,
            exchange=cfg.exchange,
            currency=cfg.currency,
            includeExpired=True,
        )
        details = ib.reqContractDetails(raw)
        if not details:
            raise RuntimeError(
                "No contract details returned. Check symbol/exchange/currency and IB market data permissions."
            )

        contract, chosen_expiry_date = select_front_month_contract(details, cfg.symbol, target_date)
        if not contract:
            available = summarize_contracts(details, cfg.symbol)
            raise RuntimeError(
                f"No {cfg.symbol} FUT found for date {cfg.date_str}.\nAvailable:\n{available}"
            )

        print(
            f"Resolved contract: localSymbol={contract.localSymbol!r} "
            f"expiry={contract.lastTradeDateOrContractMonth!r} "
            f"expiryDate={chosen_expiry_date} conId={contract.conId}",
            flush=True,
        )

        tick_mode = True
        trades_ticks = []
        bidask_ticks = []
        try:
            trades_ticks = fetch_historical_ticks(
                ib, contract, start_hk, end_hk, "TRADES", cfg.use_rth, cfg.verbose
            )
            bidask_ticks = fetch_historical_ticks(
                ib, contract, start_hk, end_hk, "BID_ASK", cfg.use_rth, cfg.verbose
            )
            if not trades_ticks and not bidask_ticks:
                tick_mode = False
        except Exception as e:
            if cfg.verbose:
                print(f"Tick-by-tick fetch failed, fallback to bars: {e}", file=sys.stderr, flush=True)
            tick_mode = False

        rows: list[dict] = []
        if tick_mode:
            if cfg.verbose and trades_ticks:
                t0 = to_hk_datetime(trades_ticks[0].time)
                t1 = to_hk_datetime(trades_ticks[-1].time)
                print(f"  trade ticks HK range: {t0} -> {t1}", flush=True)
            if cfg.verbose and bidask_ticks:
                q0 = to_hk_datetime(bidask_ticks[0].time)
                q1 = to_hk_datetime(bidask_ticks[-1].time)
                print(f"  bid/ask ticks HK range: {q0} -> {q1}", flush=True)
            for t in trades_ticks:
                ts = to_hk_datetime(t.time).strftime("%Y-%m-%d %H:%M:%S")
                rows.append(
                    {
                        "timestamp_hkt": ts,
                        "event_type": "TICK_TRADE",
                        "last_price": t.price,
                        "last_size": t.size,
                        "bid_price": "",
                        "bid_size": "",
                        "offer_price": "",
                        "ask_size": "",
                        "vwap_price": "",
                        "total_volume": "",
                        "exchange": getattr(t, "exchange", ""),
                        "special_conditions": getattr(t, "specialConditions", ""),
                    }
                )
            for t in bidask_ticks:
                ts = to_hk_datetime(t.time).strftime("%Y-%m-%d %H:%M:%S")
                bid_price = getattr(t, "bidPrice", getattr(t, "priceBid", ""))
                ask_price = getattr(t, "askPrice", getattr(t, "priceAsk", ""))
                bid_size = getattr(t, "bidSize", getattr(t, "sizeBid", ""))
                ask_size = getattr(t, "askSize", getattr(t, "sizeAsk", ""))
                rows.append(
                    {
                        "timestamp_hkt": ts,
                        "event_type": "TICK_BIDASK",
                        "last_price": "",
                        "last_size": "",
                        "bid_price": bid_price,
                        "bid_size": bid_size,
                        "offer_price": ask_price,
                        "ask_size": ask_size,
                        "vwap_price": "",
                        "total_volume": "",
                        "exchange": "",
                        "special_conditions": "",
                    }
                )
            rows.sort(key=lambda x: (x["timestamp_hkt"], x["event_type"]))
            print(
                f"Tick-by-tick data available: trades={len(trades_ticks)}, bidask={len(bidask_ticks)}",
                flush=True,
            )
        else:
            print("Tick-by-tick unavailable/empty, requesting 1-minute bars fallback...", flush=True)
            bars_trades, bars_bid, bars_ask = fetch_minute_bars(ib, contract, end_hk, cfg.use_rth)

            trade_map = {bar_datetime_hk(b.date): b for b in bars_trades}
            bid_map = {bar_datetime_hk(b.date): b for b in bars_bid}
            ask_map = {bar_datetime_hk(b.date): b for b in bars_ask}
            all_times = sorted(set(trade_map) | set(bid_map) | set(ask_map))
            if cfg.verbose and all_times:
                print(f"  bar HK range: {all_times[0]} -> {all_times[-1]}", flush=True)
                print(f"  requested HK range: {start_hk} -> {end_hk}", flush=True)

            for ts in all_times:
                if not (start_hk <= ts < end_hk):
                    continue
                tr = trade_map.get(ts)
                bd = bid_map.get(ts)
                ak = ask_map.get(ts)
                rows.append(
                    {
                        "timestamp_hkt": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": "BAR_1MIN",
                        "last_price": "" if tr is None else tr.close,
                        "last_size": "",
                        "bid_price": "" if bd is None else bd.close,
                        "bid_size": "",
                        "offer_price": "" if ak is None else ak.close,
                        "ask_size": "",
                        "vwap_price": "" if tr is None else tr.average,
                        "total_volume": "" if tr is None else tr.volume,
                        "exchange": "",
                        "special_conditions": "",
                    }
                )
            print(
                f"1-minute bars fetched: trades={len(bars_trades)}, bid={len(bars_bid)}, ask={len(bars_ask)}",
                flush=True,
            )

        write_csv(out_file, rows)
        print(f"Wrote {len(rows)} rows to {out_file}", flush=True)
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
