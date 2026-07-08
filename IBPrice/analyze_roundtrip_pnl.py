"""
Analyze round-trip PnL from backtest trade log.

Outputs:
  - round_trips.csv
  - summary_by_entry_hour.csv
  - summary_by_step.csv
  - summary_by_side.csv
  - summary_by_hour_step_side.csv

Usage:
  python analyze_roundtrip_pnl.py
  python analyze_roundtrip_pnl.py --input ../Data/IBPrice/backtest_output/trade_log.csv --output-dir ../Data/IBPrice/backtest_output
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "IBPrice"


@dataclass
class Lot:
    qty: int
    price: float
    commission_per_contract: float
    timestamp: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-trip PnL analysis from trade_log.csv")
    parser.add_argument("--input", default=str(DATA_DIR / "backtest_output" / "trade_log.csv"), help="Input trade log CSV")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "backtest_output"), help="Directory for analysis CSV outputs")
    parser.add_argument("--contract-multiplier", type=float, default=50.0, help="Contract multiplier")
    return parser.parse_args()


def side_label(side: int) -> str:
    return "long" if side > 0 else "short"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict], key_fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        grouped[key].append(row)

    out: list[dict] = []
    for key, group_rows in grouped.items():
        n = len(group_rows)
        gross_pnl = sum(x["gross_pnl"] for x in group_rows)
        net_pnl = sum(x["net_pnl"] for x in group_rows)
        fees = sum(x["fees"] for x in group_rows)
        turnover = sum(x["turnover"] for x in group_rows)
        wins_gross = sum(1 for x in group_rows if x["gross_pnl"] > 0)
        wins_net = sum(1 for x in group_rows if x["net_pnl"] > 0)
        avg_gross = gross_pnl / n if n else 0.0
        avg_net = net_pnl / n if n else 0.0
        gross_bps = (gross_pnl / turnover) * 10000.0 if turnover else 0.0
        net_bps = (net_pnl / turnover) * 10000.0 if turnover else 0.0

        row_out = {k: v for k, v in zip(key_fields, key)}
        row_out.update(
            {
                "round_trips": n,
                "gross_pnl": round(gross_pnl, 6),
                "fees": round(fees, 6),
                "net_pnl": round(net_pnl, 6),
                "avg_gross_pnl": round(avg_gross, 6),
                "avg_net_pnl": round(avg_net, 6),
                "turnover": round(turnover, 6),
                "gross_bps_per_turnover": round(gross_bps, 6),
                "net_bps_per_turnover": round(net_bps, 6),
                "win_rate_gross": round(wins_gross / n, 6) if n else 0.0,
                "win_rate_net": round(wins_net / n, 6) if n else 0.0,
            }
        )
        out.append(row_out)

    out.sort(key=lambda r: tuple(r[k] for k in key_fields))
    return out


def analyze(input_path: Path, output_dir: Path, contract_multiplier: float) -> None:
    legs: dict[str, dict] = {}
    round_trips: list[dict] = []

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            leg_id = row["leg_id"]
            if not leg_id:
                continue

            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            order_kind = row["order_kind"]
            reason = row["reason"]
            side = int(float(row["side"]))
            qty = int(float(row["quantity"]))
            px = float(row["fill_price"])
            comm = float(row["commission"])
            threshold = float(row["entry_threshold"])

            leg = legs.setdefault(
                leg_id,
                {
                    "symbol": row["symbol"],
                    "entry_side": 0,
                    "entry_threshold": threshold,
                    "entry_lots": [],
                    "exit_lots": [],
                    "exit_reasons": set(),
                },
            )

            if order_kind == "ENTRY":
                leg["entry_side"] = side
                leg["entry_threshold"] = threshold
                cpc = comm / qty if qty else 0.0
                leg["entry_lots"].append(Lot(qty=qty, price=px, commission_per_contract=cpc, timestamp=ts))
            elif order_kind == "EXIT":
                cpc = comm / qty if qty else 0.0
                leg["exit_lots"].append(Lot(qty=qty, price=px, commission_per_contract=cpc, timestamp=ts))
                leg["exit_reasons"].add(reason)

    for leg_id, leg in legs.items():
        entry_side = leg["entry_side"]
        if entry_side == 0:
            continue
        entry_lots: list[Lot] = sorted(leg["entry_lots"], key=lambda x: x.timestamp)
        exit_lots: list[Lot] = sorted(leg["exit_lots"], key=lambda x: x.timestamp)
        if not entry_lots or not exit_lots:
            continue

        i = 0
        j = 0
        while i < len(entry_lots) and j < len(exit_lots):
            e = entry_lots[i]
            x = exit_lots[j]
            matched_qty = min(e.qty, x.qty)
            if matched_qty <= 0:
                break

            if entry_side > 0:
                gross = (x.price - e.price) * matched_qty * contract_multiplier
            else:
                gross = (e.price - x.price) * matched_qty * contract_multiplier
            fees = matched_qty * (e.commission_per_contract + x.commission_per_contract)
            net = gross - fees
            turnover = (e.price + x.price) * matched_qty * contract_multiplier
            gross_bps = (gross / turnover) * 10000.0 if turnover else 0.0
            net_bps = (net / turnover) * 10000.0 if turnover else 0.0

            round_trips.append(
                {
                    "leg_id": leg_id,
                    "symbol": leg["symbol"],
                    "side": side_label(entry_side),
                    "entry_threshold": leg["entry_threshold"],
                    "entry_timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_timestamp": x.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_hour": e.timestamp.hour,
                    "exit_reason": ",".join(sorted(leg["exit_reasons"])),
                    "quantity": matched_qty,
                    "entry_price": e.price,
                    "exit_price": x.price,
                    "gross_pnl": round(gross, 6),
                    "fees": round(fees, 6),
                    "net_pnl": round(net, 6),
                    "turnover": round(turnover, 6),
                    "gross_bps_per_turnover": round(gross_bps, 6),
                    "net_bps_per_turnover": round(net_bps, 6),
                    "win_gross": 1 if gross > 0 else 0,
                    "win_net": 1 if net > 0 else 0,
                    "holding_minutes": round((x.timestamp - e.timestamp).total_seconds() / 60.0, 6),
                }
            )

            e.qty -= matched_qty
            x.qty -= matched_qty
            if e.qty == 0:
                i += 1
            if x.qty == 0:
                j += 1

    round_trips.sort(key=lambda r: (r["entry_timestamp"], r["leg_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "round_trips.csv",
        round_trips,
        [
            "leg_id",
            "symbol",
            "side",
            "entry_threshold",
            "entry_timestamp",
            "exit_timestamp",
            "entry_hour",
            "exit_reason",
            "quantity",
            "entry_price",
            "exit_price",
            "gross_pnl",
            "fees",
            "net_pnl",
            "turnover",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "win_gross",
            "win_net",
            "holding_minutes",
        ],
    )

    by_hour = summarize(round_trips, ["entry_hour"])
    by_step = summarize(round_trips, ["entry_threshold"])
    by_side = summarize(round_trips, ["side"])
    by_hss = summarize(round_trips, ["entry_hour", "entry_threshold", "side"])

    write_csv(
        output_dir / "summary_by_entry_hour.csv",
        by_hour,
        [
            "entry_hour",
            "round_trips",
            "gross_pnl",
            "fees",
            "net_pnl",
            "avg_gross_pnl",
            "avg_net_pnl",
            "turnover",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "win_rate_gross",
            "win_rate_net",
        ],
    )
    write_csv(
        output_dir / "summary_by_step.csv",
        by_step,
        [
            "entry_threshold",
            "round_trips",
            "gross_pnl",
            "fees",
            "net_pnl",
            "avg_gross_pnl",
            "avg_net_pnl",
            "turnover",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "win_rate_gross",
            "win_rate_net",
        ],
    )
    write_csv(
        output_dir / "summary_by_side.csv",
        by_side,
        [
            "side",
            "round_trips",
            "gross_pnl",
            "fees",
            "net_pnl",
            "avg_gross_pnl",
            "avg_net_pnl",
            "turnover",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "win_rate_gross",
            "win_rate_net",
        ],
    )
    write_csv(
        output_dir / "summary_by_hour_step_side.csv",
        by_hss,
        [
            "entry_hour",
            "entry_threshold",
            "side",
            "round_trips",
            "gross_pnl",
            "fees",
            "net_pnl",
            "avg_gross_pnl",
            "avg_net_pnl",
            "turnover",
            "gross_bps_per_turnover",
            "net_bps_per_turnover",
            "win_rate_gross",
            "win_rate_net",
        ],
    )

    total_turnover = sum(x["turnover"] for x in round_trips)
    total_gross = sum(x["gross_pnl"] for x in round_trips)
    total_fees = sum(x["fees"] for x in round_trips)
    total_net = sum(x["net_pnl"] for x in round_trips)
    wins_net = sum(1 for x in round_trips if x["net_pnl"] > 0)
    total = len(round_trips)

    print("Round-trip analysis complete.")
    print(f"round_trips: {total}")
    print(f"gross_pnl: {round(total_gross, 6)}")
    print(f"fees: {round(total_fees, 6)}")
    print(f"net_pnl: {round(total_net, 6)}")
    print(
        "gross_bps_per_turnover: "
        f"{round((total_gross / total_turnover) * 10000.0, 6) if total_turnover else 0.0}"
    )
    print(
        "net_bps_per_turnover: "
        f"{round((total_net / total_turnover) * 10000.0, 6) if total_turnover else 0.0}"
    )
    print(f"win_rate_net: {round(wins_net / total, 6) if total else 0.0}")
    print(f"output_dir: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    analyze(Path(args.input), Path(args.output_dir), args.contract_multiplier)


if __name__ == "__main__":
    main()

