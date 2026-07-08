"""
Test script: load delta report JSON grouped by clv_ref_bucket, merge buckets to 0.2 width,
plot cumulative Settle PnL by CLV Ref Bin over time, save figure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Data" / "deltaReportOld"
JSON_PATH = DATA_DIR / "delta_report_public.json"
OUT_DIR = DATA_DIR
OUT_FILE = OUT_DIR / "delta_report_settle_pnl_cumulative_by_clv_ref_bin.png"

BIN_WIDTH = 0.2  # merge pairs of 0.1-wide CLV Ref buckets


def parse_clv_ref_bucket(s: str) -> tuple[float, float]:
    parts = str(s).strip().split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Unrecognized clv_ref_bucket format: {s!r}")
    return float(parts[0]), float(parts[1])


def clv_ref_bin_label(bucket: str) -> str:
    """Map 0.1-spaced bucket (e.g. 0.4-0.5) to 0.2-spaced label (e.g. 0.4-0.6)."""
    lo, _hi = parse_clv_ref_bucket(bucket)
    coarse_lo = math.floor((lo + 1e-12) / BIN_WIDTH) * BIN_WIDTH
    coarse_hi = coarse_lo + BIN_WIDTH
    if abs(coarse_hi - 1.0) < 1e-9:
        return f"{coarse_lo:g}-1"
    return f"{coarse_lo:g}-{coarse_hi:g}"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data.get("reports", []):
        g = r.get("group") or {}
        bucket = g.get("clv_ref_bucket")
        if bucket is None:
            continue
        settle = (
            r.get("implied_prob", {})
            .get("timepoints", {})
            .get("settle", {})
        )
        pnl = settle.get("pnl")
        if pnl is None:
            continue
        rows.append(
            {
                "date": g.get("date"),
                "clv_ref_bucket": bucket,
                "clv_ref_bin": clv_ref_bin_label(bucket),
                "settle_pnl": float(pnl),
            }
        )
    return rows


def main() -> None:
    rows = load_rows(JSON_PATH)
    if not rows:
        raise SystemExit(f"No rows with settle PnL found in {JSON_PATH}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "clv_ref_bin"])
    # Same date + coarse bin can come from two 0.1 buckets — sum before cumulative
    df = (
        df.groupby(["clv_ref_bin", "date"], as_index=False)["settle_pnl"]
        .sum()
        .sort_values(["clv_ref_bin", "date"])
    )
    df["cumulative_settle_pnl"] = df.groupby("clv_ref_bin", sort=True)["settle_pnl"].cumsum()

    plt.figure(figsize=(12, 6))
    for bin_name, sub in df.groupby("clv_ref_bin", sort=True):
        sub = sub.sort_values("date")
        plt.plot(
            sub["date"],
            sub["cumulative_settle_pnl"],
            marker="o",
            markersize=2,
            label=bin_name,
        )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Settle PnL")
    plt.title("Cumulative Settle PnL by CLV Ref Bin (0.2 width)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
