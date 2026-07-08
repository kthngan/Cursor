from __future__ import annotations

import csv
import subprocess
from pathlib import Path


INITIAL_CASH = 1_000_000.0


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def read_roundtrip(path: Path) -> tuple[float, float, float]:
    total_net = 0.0
    total_turn = 0.0
    wins = 0
    total = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            net = float(row["net_pnl"])
            turn = float(row["turnover"])
            total_net += net
            total_turn += turn
            wins += 1 if net > 0 else 0
            total += 1
    net_bps = (total_net / total_turn) * 10000.0 if total_turn else 0.0
    win_rate = (wins / total) if total else 0.0
    return total_net, net_bps, win_rate


def main() -> None:
    base = Path(__file__).resolve().parent
    root = base / "backtest_output"
    root.mkdir(parents=True, exist_ok=True)

    common = [
        "python",
        "-m",
        "backtest.main",
        "--data-dir",
        "historicalData",
        "--index-file",
        "historicalData/hsi_index_daily.csv",
        "--start-date",
        "2026-03-01",
        "--end-date",
        "2026-05-15",
        "--range-filter-mode",
        "two_sided",
        "--range-filter-low-pct",
        "0.2",
        "--range-filter-high-pct",
        "0.85",
    ]

    combos = [
        (0.8, 0.4),
        (1.0, 0.4),
        (1.2, 0.4),
        (1.2, 0.5),
        (1.2, 0.6),
        (1.5, 0.4),
        (1.5, 0.5),
        (1.8, 0.5),
    ]

    results: list[dict[str, str | float]] = []
    for pt_mult, sl_mult in combos:
        name = f"pt{pt_mult}_sl{sl_mult}".replace(".", "p")
        out_dir = root / f"sweep_ptsl_{name}"
        cmd = common + [
            "--output-dir",
            str(out_dir),
            "--pt-multiplier",
            str(pt_mult),
            "--sl-multiplier",
            str(sl_mult),
        ]
        run(cmd, base)
        run(
            [
                "python",
                "analyze_roundtrip_pnl.py",
                "--input",
                str(out_dir / "trade_log.csv"),
                "--output-dir",
                str(out_dir),
                "--contract-multiplier",
                "50",
            ],
            base,
        )

        s = read_summary(out_dir / "summary.txt")
        rt_net, rt_bps, win_rate = read_roundtrip(out_dir / "round_trips.csv")
        final_equity = float(s.get("final_equity", 0.0))
        results.append(
            {
                "pt_multiplier": pt_mult,
                "sl_multiplier": sl_mult,
                "pt_sl_ratio": (pt_mult / sl_mult) if sl_mult else 0.0,
                "num_trades": int(s.get("num_trades", 0.0)),
                "final_equity": final_equity,
                "net_equity_pnl": final_equity - INITIAL_CASH,
                "realized_pnl": float(s.get("realized_pnl", 0.0)),
                "max_drawdown": float(s.get("max_drawdown", 0.0)),
                "roundtrip_net_pnl": rt_net,
                "roundtrip_net_bps": rt_bps,
                "roundtrip_win_rate": win_rate,
                "output_dir": str(out_dir),
            }
        )

    out_csv = root / "pt_sl_sweep_results.csv"
    fieldnames = [
        "pt_multiplier",
        "sl_multiplier",
        "pt_sl_ratio",
        "num_trades",
        "final_equity",
        "net_equity_pnl",
        "realized_pnl",
        "max_drawdown",
        "roundtrip_net_pnl",
        "roundtrip_net_bps",
        "roundtrip_win_rate",
        "output_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)

    print(f"Saved PT/SL sweep: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

