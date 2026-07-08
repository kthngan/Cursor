"""
Fetch historical daily HSI index bars from Interactive Brokers and save to CSV.

Default behavior:
- Index: HSI (secType=IND)
- Exchange: HKFE
- Currency: HKD
- Start date: 2025-09-01

Example:
  python getHistoricalIndexData.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from ib_insync import IB, Index, util

INFO_ERROR_CODES = frozenset({2104, 2106, 2119, 2158})
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "IBPrice"


@dataclass
class AppConfig:
    host: str
    port: int
    client_id: int
    symbol: str
    exchange: str
    currency: str
    start_date: date
    out_file: Path
    use_rth: bool
    verbose: bool


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Get IB historical daily index OHLC data.")
    parser.add_argument("--host", default="127.0.0.1", help="TWS/Gateway host")
    parser.add_argument("--port", type=int, default=7497, help="TWS: 7497 live / 7496 paper")
    parser.add_argument("--client-id", type=int, default=12, dest="client_id")
    parser.add_argument("--symbol", default="HSI", help="Index symbol")
    parser.add_argument("--exchange", default="HKFE", help="Index exchange")
    parser.add_argument("--currency", default="HKD", help="Contract currency")
    parser.add_argument(
        "--start-date",
        default="2025-09-01",
        help="Start date in YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--out-file",
        default=str(DATA_DIR / "historicalData" / "hsi_index_daily.csv"),
        help=f"Output CSV path (default: {DATA_DIR / 'historicalData' / 'hsi_index_daily.csv'})",
    )
    parser.add_argument(
        "--use-rth",
        action="store_true",
        help="Use regular trading hours only (default: full session)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print IB diagnostics")
    args = parser.parse_args()

    parsed_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    out_file = Path(args.out_file)
    if not out_file.is_absolute():
        out_file = DATA_DIR / out_file

    return AppConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        symbol=args.symbol.upper(),
        exchange=args.exchange,
        currency=args.currency,
        start_date=parsed_start,
        out_file=out_file,
        use_rth=args.use_rth,
        verbose=args.verbose,
    )


def connect_with_fallback_ports(ib: IB, host: str, port: int, client_id: int, verbose: bool) -> int:
    candidates = [port, 7497, 7496, 4002, 4001]
    seen: set[int] = set()
    ordered_ports: list[int] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            ordered_ports.append(p)

    last_err: Exception | None = None
    for p in ordered_ports:
        print(f"Connecting to IB {host}:{p} clientId={client_id} ...", flush=True)
        try:
            ib.connect(host, p, clientId=client_id, timeout=15, readonly=True)
            return p
        except Exception as exc:
            last_err = exc
            if verbose:
                print(f"  connection failed on port {p}: {exc}", file=sys.stderr, flush=True)

    raise RuntimeError(
        f"IB connection failed on ports {ordered_ports}. "
        "Please start TWS/IB Gateway and verify API host/port permissions."
    ) from last_err


def fetch_daily_bars_since(ib: IB, contract, start_date: date, use_rth: bool) -> list:
    now_utc = datetime.now(timezone.utc)
    end_datetime = now_utc.strftime("%Y%m%d-%H:%M:%S")

    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime=end_datetime,
        durationStr="10 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )

    filtered = []
    for b in bars:
        if isinstance(b.date, datetime):
            bar_date = b.date.date()
        else:
            raw = str(b.date).strip()
            if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
                bar_date = datetime.strptime(raw[:10], "%Y-%m-%d").date()
            else:
                bar_date = datetime.strptime(raw[:8], "%Y%m%d").date()
        if bar_date >= start_date:
            filtered.append((bar_date, b.open, b.high, b.low, b.close))

    filtered.sort(key=lambda row: row[0])
    return filtered


def write_csv(path: Path, rows: list[tuple[date, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close"])
        for d, o, h, l, c in rows:
            writer.writerow([d.isoformat(), o, h, l, c])


def main() -> None:
    patch = getattr(util, "patch_asyncio", None) or getattr(util, "patchAsyncio", None)
    if patch:
        patch()

    cfg = parse_args()
    ib = IB()

    if cfg.verbose:

        def on_error(req_id: int, code: int, msg: str, contract) -> None:
            if code in INFO_ERROR_CODES:
                return
            print(f"IB error {code} reqId={req_id}: {msg}", file=sys.stderr, flush=True)

        ib.errorEvent += on_error

    try:
        connected_port = connect_with_fallback_ports(
            ib, cfg.host, cfg.port, cfg.client_id, cfg.verbose
        )
        print(f"Connected on port {connected_port}.", flush=True)

        contract = Index(
            symbol=cfg.symbol,
            exchange=cfg.exchange,
            currency=cfg.currency,
        )
        details = ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(
                "No contract details returned. Check symbol/exchange/currency and market data permissions."
            )

        qualified = details[0].contract
        print(
            f"Resolved index contract: localSymbol={qualified.localSymbol!r} conId={qualified.conId}",
            flush=True,
        )

        rows = fetch_daily_bars_since(ib, qualified, cfg.start_date, cfg.use_rth)
        if not rows:
            raise RuntimeError(
                f"No daily bars found for {cfg.symbol} on/after {cfg.start_date.isoformat()}."
            )

        write_csv(cfg.out_file, rows)
        print(f"Wrote {len(rows)} daily rows to {cfg.out_file}", flush=True)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
