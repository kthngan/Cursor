"""
Stream IBKR tick-by-tick (AllLast + BidAsk) for futures or US-style stocks.

Requires TWS or IB Gateway running, API enabled, and market data for the instrument
(including tick-by-tick where your subscription allows).

Examples:
  ES future:  python stream_futures_ticks.py --symbol ES --exchange CME --currency USD --expiry 202606
  AAPL stock: python stream_futures_ticks.py --symbol AAPL --stock --exchange SMART --currency USD
  HSI tomorrow 09:40 HKT: python stream_futures_ticks.py --symbol HSI --exchange HKFE --currency HKD --expiry 202606 --run-tomorrow-hk 09:40

Install: pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import functools
import sys
import threading
import time
from datetime import datetime, timedelta, time as time_of_day
from zoneinfo import ZoneInfo

from ib_insync import IB, Contract, Future, Stock, TickByTickAllLast, TickByTickBidAsk, util

_HK = ZoneInfo("Asia/Hong_Kong")

# IB "informational" messages on stderr; not actionable for this script.
_INFO_ERROR_CODES = frozenset({2104, 2106, 2119, 2158})

_shutdown_lock = threading.Lock()
_shutdown_done = False


def _default_expiry_yyyymm() -> str:
    """Rough default contract month (YYYYMM); override with --expiry."""
    now = datetime.now()
    year, month = now.year, now.month
    if now.day >= 22:
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return f"{year}{month:02d}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IBKR tick-by-tick trade + quote stream for futures or stocks."
    )
    p.add_argument("--host", default="127.0.0.1", help="TWS / Gateway host")
    p.add_argument(
        "--port",
        type=int,
        default=7497,
        help="7497=TWS live, 7496=TWS paper; Gateway often 4001/4002",
    )
    p.add_argument("--client-id", type=int, default=1, dest="client_id")
    p.add_argument(
        "--symbol",
        required=True,
        metavar="SYM",
        help="Root symbol: futures (ES, HSI) or stock ticker (AAPL).",
    )
    p.add_argument(
        "--stock",
        action="store_true",
        help="Treat --symbol as an equity (STK). Do not use --expiry.",
    )
    p.add_argument(
        "--expiry",
        default=None,
        metavar="YYYYMM",
        help="Futures contract month (e.g. 202606). Ignored with --stock. Default: inferred from calendar.",
    )
    p.add_argument(
        "--exchange",
        default="SMART",
        help="Venue / routing (e.g. CME for ES, HKFE for HSI). Default: SMART",
    )
    p.add_argument(
        "--currency",
        default="USD",
        help="Contract currency (e.g. USD, HKD)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Stop after this many seconds (0 = run until Ctrl+C). Useful for smoke tests.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print IB API errors and contract-detail diagnostics.",
    )
    sched = p.add_mutually_exclusive_group()
    sched.add_argument(
        "--run-tomorrow-hk",
        metavar="HH:MM",
        default=None,
        help="Wait until tomorrow at this clock time in Hong Kong (Asia/Hong_Kong), then connect.",
    )
    sched.add_argument(
        "--wait-until-hk",
        metavar="TIME",
        default=None,
        help="Wait until this Hong Kong local time (format: YYYY-MM-DD HH:MM), then connect.",
    )
    return p.parse_args()


def _parse_hhmm(hhmm: str) -> tuple[int, int]:
    s = hhmm.strip().replace(".", ":")
    parts = s.split(":")
    if len(parts) != 2:
        raise SystemExit(f"Invalid local time {hhmm!r}; use HH:MM (e.g. 09:40 or 9:40).")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise SystemExit(f"Invalid hour/minute in {hhmm!r}.")
    return h, m


def _tomorrow_at_hk(hhmm: str) -> datetime:
    now = datetime.now(_HK)
    h, m = _parse_hhmm(hhmm)
    day = now.date() + timedelta(days=1)
    return datetime.combine(day, time_of_day(h, m, 0), tzinfo=_HK)


def _parse_wait_until_hk(s: str) -> datetime:
    text = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
            return naive.replace(tzinfo=_HK)
        except ValueError:
            continue
    raise SystemExit(
        f"Invalid --wait-until-hk {s!r}; use 'YYYY-MM-DD HH:MM' (Hong Kong wall clock)."
    )


def _schedule_startup_wait(args: argparse.Namespace) -> None:
    """Sleep until HK target; must run before ib.connect (leave TWS idle until then)."""
    if args.run_tomorrow_hk is None and args.wait_until_hk is None:
        return
    if args.run_tomorrow_hk is not None:
        target = _tomorrow_at_hk(args.run_tomorrow_hk)
    else:
        target = _parse_wait_until_hk(args.wait_until_hk)
    now = datetime.now(_HK)
    if target <= now:
        raise SystemExit(
            f"Scheduled start is not in the future (HK now {now.isoformat(timespec='minutes')}, "
            f"target {target.isoformat(timespec='minutes')})."
        )
    print(
        f"Waiting until {target.strftime('%Y-%m-%d %H:%M')} Asia/Hong_Kong "
        f"(now {now.strftime('%Y-%m-%d %H:%M')} HKT)...",
        flush=True,
    )
    last_log = time.monotonic()
    while True:
        now = datetime.now(_HK)
        secs = (target - now).total_seconds()
        if secs <= 0:
            break
        if secs > 120 and (time.monotonic() - last_log) >= 300.0:
            print(f"  ... {secs / 3600:.2f} h until start (HKT)", flush=True)
            last_log = time.monotonic()
        time.sleep(min(30.0, max(0.5, secs)))
    print(
        f"Start time reached (HK {datetime.now(_HK).strftime('%Y-%m-%d %H:%M:%S')}), connecting.",
        flush=True,
    )


def _expiry_month_prefix(expiry: str) -> str:
    """YYYYMM from YYYYMM or YYYYMMDD input."""
    s = expiry.strip()
    if len(s) < 6 or not s[:6].isdigit():
        raise SystemExit(f"Invalid --expiry {expiry!r}; use YYYYMM (e.g. 202606) or YYYYMMDD.")
    return s[:6]


def _select_future_for_month(
    details: list,
    *,
    symbol: str,
    expiry_month: str,
    verbose: bool,
) -> Contract | None:
    """
    IB returns many ContractDetails for one Future query; [0] is often NOT the
    month you asked for. Pick the FUT whose last trade / contract month matches
    the requested YYYYMM prefix.
    """
    sym = symbol.upper()
    want = expiry_month
    matches: list[Contract] = []
    for d in details:
        c = d.contract
        if not c or c.secType != "FUT" or c.symbol != sym:
            continue
        ltd = (c.lastTradeDateOrContractMonth or "").strip()
        cm = (getattr(d, "contractMonth", None) or "").strip()
        if ltd.startswith(want) or cm.startswith(want) or cm == want:
            matches.append(c)
            if verbose:
                print(
                    f"  candidate  localSymbol={c.localSymbol!r}  "
                    f"lastTradeDateOrContractMonth={ltd!r}  contractMonth={cm!r}  "
                    f"conId={c.conId}  exch={c.exchange!r}",
                    flush=True,
                )
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    # Prefer a fully specified last-trade day (YYYYMMDD) over bare YYYYMM if both exist.
    def _specificity(ct: Contract) -> tuple[int, str]:
        ltd = ct.lastTradeDateOrContractMonth or ""
        score = 2 if len(ltd) >= 8 else (1 if len(ltd) == 6 else 0)
        return (-score, ltd)

    matches.sort(key=_specificity)
    return matches[0]


def _summarize_available(details: list, symbol: str, limit: int = 25) -> str:
    sym = symbol.upper()
    rows: list[str] = []
    for d in details:
        c = d.contract
        if not c or c.secType != "FUT" or c.symbol != sym:
            continue
        ltd = c.lastTradeDateOrContractMonth or ""
        cm = getattr(d, "contractMonth", None) or ""
        rows.append(f"    {c.localSymbol!r}  month={ltd!r}  contractMonth={cm!r}  conId={c.conId}")
    if not rows:
        return "  (no FUT details returned for this symbol)"
    body = "\n".join(rows[:limit])
    extra = f"\n    ... and {len(rows) - limit} more" if len(rows) > limit else ""
    return body + extra


def _select_stock_contract(
    details: list,
    *,
    symbol: str,
    verbose: bool,
) -> Contract | None:
    sym = symbol.upper()
    matches: list[Contract] = []
    for d in details:
        c = d.contract
        if not c or c.secType != "STK" or c.symbol != sym:
            continue
        matches.append(c)
        if verbose:
            print(
                f"  candidate  localSymbol={c.localSymbol!r}  exchange={c.exchange!r}  "
                f"primaryExchange={c.primaryExchange!r}  conId={c.conId}",
                flush=True,
            )
    if not matches:
        return None
    for c in matches:
        if c.exchange == "SMART":
            return c
    return matches[0]


def _summarize_stock_details(details: list, symbol: str, limit: int = 25) -> str:
    sym = symbol.upper()
    rows: list[str] = []
    for d in details:
        c = d.contract
        if not c or c.secType != "STK" or c.symbol != sym:
            continue
        rows.append(
            f"    {c.localSymbol!r}  exch={c.exchange!r}  primary={c.primaryExchange!r}  conId={c.conId}"
        )
    if not rows:
        return "  (no STK details returned for this symbol)"
    body = "\n".join(rows[:limit])
    extra = f"\n    ... and {len(rows) - limit} more" if len(rows) > limit else ""
    return body + extra


def main() -> None:
    global _shutdown_done
    _shutdown_done = False
    patch = getattr(util, "patch_asyncio", None) or getattr(util, "patchAsyncio", None)
    if patch:
        patch()
    args = _parse_args()
    _schedule_startup_wait(args)

    ib = IB()
    if args.verbose:

        def _on_err(req_id: int, code: int, msg: str, contract: Contract | None) -> None:
            if code in _INFO_ERROR_CODES:
                return
            ctag = ""
            if contract and getattr(contract, "localSymbol", None):
                ctag = f" contract={contract.localSymbol!r}"
            print(f"IB error {code}: {msg}{ctag}", file=sys.stderr, flush=True)

        ib.errorEvent += _on_err

    ib.connect(args.host, args.port, clientId=args.client_id)

    if args.stock:
        raw = Stock(args.symbol.upper(), args.exchange, args.currency)
        details = ib.reqContractDetails(raw)
        if not details:
            print(
                "No contract matched. Check --symbol, --exchange, --currency, "
                "and Contract Search in TWS.",
                file=sys.stderr,
            )
            ib.disconnect()
            sys.exit(1)
        contract = _select_stock_contract(details, symbol=args.symbol, verbose=args.verbose)
        if not contract:
            print(
                f"No STK line for {args.symbol.upper()!r} in contract details.",
                file=sys.stderr,
            )
            print(
                "Available contracts (sample):\n" + _summarize_stock_details(details, args.symbol),
                file=sys.stderr,
            )
            ib.disconnect()
            sys.exit(1)
        if args.verbose:
            print(
                f"Selected: localSymbol={contract.localSymbol!r}  exchange={contract.exchange!r}  "
                f"conId={contract.conId}\n",
                flush=True,
            )
        print(
            f"Streaming {contract.localSymbol or contract.symbol} "
            f"conId={contract.conId} secType=STK "
            f"({args.host}:{args.port}) - Ctrl+C to stop\n",
            flush=True,
        )
    else:
        expiry = args.expiry or _default_expiry_yyyymm()
        expiry_month = _expiry_month_prefix(expiry)

        raw = Future(
            symbol=args.symbol.upper(),
            lastTradeDateOrContractMonth=expiry,
            exchange=args.exchange,
            currency=args.currency,
        )
        details = ib.reqContractDetails(raw)
        if not details:
            print(
                "No contract matched. Check --symbol, --expiry (YYYYMM), --exchange, "
                "--currency, and Contract Search in TWS.",
                file=sys.stderr,
            )
            ib.disconnect()
            sys.exit(1)

        contract = _select_future_for_month(
            details, symbol=args.symbol, expiry_month=expiry_month, verbose=args.verbose
        )
        if not contract:
            print(
                f"No FUT contract for {args.symbol.upper()} with month prefix {expiry_month!r}. "
                f"IB returned {len(details)} detail row(s); first matches are not necessarily "
                f"your month — use a listed month (ES: quarterly H/M/U/Z).",
                file=sys.stderr,
            )
            print("Available contracts (sample):\n" + _summarize_available(details, args.symbol), file=sys.stderr)
            ib.disconnect()
            sys.exit(1)

        if args.verbose:
            print(
                f"Selected: localSymbol={contract.localSymbol!r}  "
                f"lastTradeDateOrContractMonth={contract.lastTradeDateOrContractMonth!r}  "
                f"conId={contract.conId}\n",
                flush=True,
            )

        print(
            f"Streaming {contract.localSymbol or contract.symbol} "
            f"conId={contract.conId} month={contract.lastTradeDateOrContractMonth} "
            f"({args.host}:{args.port}) - Ctrl+C to stop\n",
            flush=True,
        )

    trade_ticker = ib.reqTickByTickData(contract, "AllLast", 0, False)
    quote_ticker = ib.reqTickByTickData(contract, "BidAsk", 0, False)

    trade_index = 0
    quote_index = 0

    def _drain_trades() -> None:
        nonlocal trade_index
        buf = trade_ticker.tickByTicks
        new = buf[trade_index:]
        trade_index = len(buf)
        for tick in new:
            if isinstance(tick, TickByTickAllLast):
                print(
                    f"[TRADE] {tick.time}  last={tick.price}  size={tick.size}  "
                    f"exch={tick.exchange!r}  special={tick.specialConditions!r}",
                    flush=True,
                )

    def _drain_quotes() -> None:
        nonlocal quote_index
        buf = quote_ticker.tickByTicks
        new = buf[quote_index:]
        quote_index = len(buf)
        for tick in new:
            if isinstance(tick, TickByTickBidAsk):
                print(
                    f"[QUOTE] {tick.time}  "
                    f"bid={tick.bidPrice} x {tick.bidSize}  "
                    f"ask={tick.askPrice} x {tick.askSize}",
                    flush=True,
                )

    trade_ticker.updateEvent += lambda *_: _drain_trades()
    quote_ticker.updateEvent += lambda *_: _drain_quotes()

    if args.duration > 0:

        def _timer_disconnect() -> None:
            loop = util.getLoop()
            done = functools.partial(_shutdown, ib, contract)
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(done)
                else:
                    done()
            except Exception:
                done()

        threading.Timer(args.duration, _timer_disconnect).start()

    try:
        ib.run()
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        _shutdown(ib, contract)


def _shutdown(ib: IB, contract) -> None:
    global _shutdown_done
    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True
    try:
        ib.cancelTickByTickData(contract, "AllLast")
        ib.cancelTickByTickData(contract, "BidAsk")
    except Exception:
        pass
    if ib.isConnected():
        ib.disconnect()


if __name__ == "__main__":
    main()
