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


def main() -> None:
    base = Path(__file__).resolve().parent
    data_root = base.parent / "Data" / "IBPrice"
    historical_dir = data_root / "historicalData"
    out_root = data_root / "backtest_output"
    out_root.mkdir(parents=True, exist_ok=True)

    range_lookbacks = [20, 30, 45]
    zscore_lookbacks = [3, 5, 8]

    results: list[dict[str, float | int | str]] = []
    for rlb in range_lookbacks:
        for zlb in zscore_lookbacks:
            name = f"rlb{rlb}_zlb{zlb}"
            out_dir = out_root / f"sweep_lookback_{name}"
            cmd = [
                "python",
                "-m",
                "backtest.main",
                "--data-dir",
                str(historical_dir),
                "--index-file",
                str(historical_dir / "hsi_index_daily.csv"),
                "--start-date",
                "2026-01-01",
                "--output-dir",
                str(out_dir),
                "--range-filter-mode",
                "upper_only",
                "--range-filter-high-pct",
                "0.8",
                "--range-filter-lookback",
                str(rlb),
                "--rolling-window",
                str(zlb),
                "--pt-multiplier",
                "1.2",
                "--sl-multiplier",
                "0.5",
            ]
            run(cmd, base)
            s = read_summary(out_dir / "summary.txt")
            final_equity = float(s.get("final_equity", 0.0))
            results.append(
                {
                    "range_lookback": rlb,
                    "zscore_lookback": zlb,
                    "num_trades": int(s.get("num_trades", 0.0)),
                    "realized_pnl": float(s.get("realized_pnl", 0.0)),
                    "final_equity": final_equity,
                    "net_equity_pnl": final_equity - INITIAL_CASH,
                    "max_drawdown": float(s.get("max_drawdown", 0.0)),
                    "output_dir": str(out_dir),
                }
            )

    out_csv = out_root / "lookback_grid_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "range_lookback",
            "zscore_lookback",
            "num_trades",
            "realized_pnl",
            "final_equity",
            "net_equity_pnl",
            "max_drawdown",
            "output_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved lookback grid results: {out_csv.resolve()}")


if __name__ == "__main__":
    main()

