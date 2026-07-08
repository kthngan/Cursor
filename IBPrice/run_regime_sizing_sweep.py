from __future__ import annotations

import csv
import subprocess
from pathlib import Path


INITIAL_CASH = 1_000_000.0


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_summary(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def read_roundtrip_metrics(path: Path) -> tuple[float, float, float]:
    net = 0.0
    turnover = 0.0
    wins = 0
    total = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = float(row["net_pnl"])
            t = float(row["turnover"])
            net += n
            turnover += t
            wins += 1 if n > 0 else 0
            total += 1
    net_bps = (net / turnover) * 10000.0 if turnover else 0.0
    win_rate = (wins / total) if total else 0.0
    return net, net_bps, win_rate


def main() -> None:
    base = Path(__file__).resolve().parent
    out_root = base / "backtest_output"
    out_root.mkdir(parents=True, exist_ok=True)

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
    ]

    configs = [
        {
            "name": "baseline_upper80_flat",
            "args": [
                "--range-filter-mode",
                "upper_only",
                "--range-filter-high-pct",
                "0.8",
            ],
            "reason": "Current baseline behavior.",
        },
        {
            "name": "upper90_flat",
            "args": [
                "--range-filter-mode",
                "upper_only",
                "--range-filter-high-pct",
                "0.9",
            ],
            "reason": "Looser skip gate includes more high-range days.",
        },
        {
            "name": "two_sided_20_85_flat",
            "args": [
                "--range-filter-mode",
                "two_sided",
                "--range-filter-low-pct",
                "0.2",
                "--range-filter-high-pct",
                "0.85",
            ],
            "reason": "Removes very-low and very-high prior-day range regimes.",
        },
        {
            "name": "two_sided_20_85_norm_flat",
            "args": [
                "--range-filter-mode",
                "two_sided",
                "--range-filter-low-pct",
                "0.2",
                "--range-filter-high-pct",
                "0.85",
                "--range-use-normalized",
            ],
            "reason": "Same two-sided gate but on normalized range.",
        },
        {
            "name": "upper80_tapered_size",
            "args": [
                "--range-filter-mode",
                "upper_only",
                "--range-filter-high-pct",
                "0.8",
                "--threshold-qty-map",
                "0.002:2,0.003:2,0.004:1,0.005:1,0.006:0",
            ],
            "reason": "Higher size at lower thresholds, disable 0.006 adds.",
        },
        {
            "name": "two_sided_norm_tapered",
            "args": [
                "--range-filter-mode",
                "two_sided",
                "--range-filter-low-pct",
                "0.2",
                "--range-filter-high-pct",
                "0.85",
                "--range-use-normalized",
                "--threshold-qty-map",
                "0.002:2,0.003:2,0.004:1,0.005:1,0.006:0",
            ],
            "reason": "Combine regime filter and threshold sizing taper.",
        },
    ]

    results: list[dict[str, str | float]] = []
    for cfg in configs:
        out_dir = out_root / f"sweep_{cfg['name']}"
        cmd = common + ["--output-dir", str(out_dir)] + cfg["args"]
        run_cmd(cmd, base)

        run_cmd(
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
        net_rt_pnl, net_bps, win_rate = read_roundtrip_metrics(out_dir / "round_trips.csv")
        final_equity = float(s.get("final_equity", 0.0))
        net_equity_pnl = final_equity - INITIAL_CASH
        results.append(
            {
                "config": cfg["name"],
                "reason": cfg["reason"],
                "num_trades": int(s.get("num_trades", 0.0)),
                "realized_pnl": float(s.get("realized_pnl", 0.0)),
                "final_equity": final_equity,
                "net_equity_pnl": net_equity_pnl,
                "max_drawdown": float(s.get("max_drawdown", 0.0)),
                "roundtrip_net_pnl": net_rt_pnl,
                "roundtrip_net_bps": net_bps,
                "roundtrip_win_rate": win_rate,
            }
        )

    out_file = out_root / "sweep_results.csv"
    fields = [
        "config",
        "reason",
        "num_trades",
        "realized_pnl",
        "final_equity",
        "net_equity_pnl",
        "max_drawdown",
        "roundtrip_net_pnl",
        "roundtrip_net_bps",
        "roundtrip_win_rate",
    ]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved sweep results: {out_file.resolve()}")


if __name__ == "__main__":
    main()

