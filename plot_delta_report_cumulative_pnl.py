"""
Test script: load delta report JSON, plot cumulative Settle PnL by Main Book, save figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

JSON_PATH = Path(r"C:\Users\user\Downloads\delta_report_2026-04-09T03-44-58.json")
OUT_DIR = Path(r"c:\Users\user\Documents\Cursor")
OUT_FILE = OUT_DIR / "delta_report_settle_pnl_cumulative_by_main_book.png"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data.get("reports", []):
        g = r.get("group") or {}
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
                "main_book": g.get("main_book"),
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
    df = df.dropna(subset=["date", "main_book"])
    df = df.sort_values(["main_book", "date"])
    df["cumulative_settle_pnl"] = df.groupby("main_book", sort=False)["settle_pnl"].cumsum()

    plt.figure(figsize=(12, 6))
    for book, sub in df.groupby("main_book", sort=True):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub["cumulative_settle_pnl"], marker="o", markersize=2, label=book)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Settle PnL")
    plt.title("Cumulative Settle PnL by Main Book")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
