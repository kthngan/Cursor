#!/usr/bin/env python3
"""Re-enrich CSVs using only Pinnacle odds from cached snapshots.

Reads the original CSVs from Data/SportsAPI and the cached snapshot files,
extracts Pinnacle-only H2H prices, computes de-vig probabilities, and saves
to Data/SportsAPI/with_pinnacle_odds/.
"""

from __future__ import annotations

import bisect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from enrich_csvs_with_odds import (  # noqa: E402
    DATA_DIR,
    METADATA_PATH,
    ODDS_COLUMNS,
    SPORT_MAP,
    SNAPSHOT_CACHE_DIR,
    find_event_in_snapshot,
    load_cached_snapshots,
    names_match,
)

OUTPUT_DIR = DATA_DIR / "with_pinnacle_odds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pinnacle_prices(event: dict[str, Any], p1: str, p2: str) -> dict[str, Any]:
    """Extract H2H price from Pinnacle only."""
    p1_price: float | None = None
    p2_price: float | None = None

    for bm in event.get("bookmakers", []):
        if bm.get("key") != "pinnacle":
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None or price <= 1:
                    continue
                if names_match(p1, name):
                    p1_price = price
                elif names_match(p2, name):
                    p2_price = price

    if p1_price is None or p2_price is None:
        return {
            "odds_p1_price": None, "odds_p2_price": None,
            "odds_implied_p1": None, "odds_implied_p2": None,
            "odds_no_vig_p1": None, "odds_no_vig_p2": None,
            "odds_bookmaker_count": 0, "odds_event_id": event.get("id", ""),
        }

    implied_p1 = 1.0 / p1_price
    implied_p2 = 1.0 / p2_price
    total = implied_p1 + implied_p2
    no_vig_p1 = implied_p1 / total
    no_vig_p2 = implied_p2 / total

    return {
        "odds_p1_price": p1_price,
        "odds_p2_price": p2_price,
        "odds_implied_p1": implied_p1,
        "odds_implied_p2": implied_p2,
        "odds_no_vig_p1": no_vig_p1,
        "odds_no_vig_p2": no_vig_p2,
        "odds_bookmaker_count": 1,
        "odds_event_id": event.get("id", ""),
    }


def enrich_csv_pinnacle(
    csv_path: Path,
    snapshots: list[dict[str, Any]],
    snapshot_times: list[datetime],
    output_path: Path,
) -> dict[str, Any]:
    """Enrich a single match CSV with Pinnacle-only odds."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"event_id": csv_path.stem, "rows": 0, "matched": 0}

    p1 = str(df["p1_name"].iloc[0])
    p2 = str(df["p2_name"].iloc[0])
    event_id = str(df["event_id"].iloc[0]) if "event_id" in df.columns else csv_path.stem

    for col in ODDS_COLUMNS:
        df[col] = None

    matched_count = 0
    snapshot_events_cache: dict[str, list[dict[str, Any]]] = {}
    event_lookup_cache: dict[str, dict[str, Any] | None] = {}
    last_odds: dict[str, Any] | None = None
    last_snap_ts = ""

    for idx, row in df.iterrows():
        ut = row.get("ut")
        if pd.isna(ut):
            continue
        row_time = datetime.fromtimestamp(int(ut), tz=timezone.utc)

        snap_idx = bisect.bisect_right(snapshot_times, row_time) - 1
        if snap_idx < 0:
            snap_idx = 0

        snap = snapshots[snap_idx]
        snap_ts = snap.get("timestamp", "")

        if snap_ts not in event_lookup_cache:
            if snap_ts not in snapshot_events_cache:
                snapshot_events_cache[snap_ts] = snap.get("data", [])
            events = snapshot_events_cache[snap_ts]
            event_lookup_cache[snap_ts] = find_event_in_snapshot(p1, p2, events)
        event = event_lookup_cache[snap_ts]

        if event is not None:
            last_odds = extract_pinnacle_prices(event, p1, p2)
            last_snap_ts = snap_ts

        if last_odds is not None and last_odds.get("odds_p1_price") is not None:
            matched_count += 1
            df.at[idx, "odds_snapshot_time"] = last_snap_ts
            for col, val in last_odds.items():
                df.at[idx, col] = val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return {
        "event_id": event_id,
        "rows": len(df),
        "matched": matched_count,
        "match_rate": matched_count / len(df) if len(df) > 0 else 0,
    }


def main() -> int:
    print("=== Pinnacle-Only Enrichment ===")
    print()

    meta = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
    meta = meta[meta["event_sport_name"].fillna("").str.lower() == "tennis"]
    meta = meta[meta["competition_name"].isin(SPORT_MAP.keys())]
    meta["sport_key"] = meta["competition_name"].map(SPORT_MAP)
    print(f"   {len(meta)} tennis matches with known competitions")

    # Group by sport and load cached snapshots
    sport_snapshots: dict[str, list[dict[str, Any]]] = {}
    sport_snapshot_times: dict[str, list[datetime]] = {}

    for sport_key in SPORT_MAP.values():
        sport_meta = meta[meta["sport_key"] == sport_key]
        if sport_meta.empty:
            continue
        snaps = load_cached_snapshots(sport_key)
        if not snaps:
            print(f"   No cached snapshots for {sport_key}")
            continue
        sport_snapshots[sport_key] = snaps
        sport_snapshot_times[sport_key] = [
            datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
            for s in snaps
        ]
        print(f"   {sport_key}: {len(snaps)} cached snapshots")

    # Process each match
    summary = []
    for _, meta_row in meta.iterrows():
        event_id = str(meta_row["event_id"])
        sport_key = meta_row["sport_key"]
        csv_path = DATA_DIR / f"{event_id}.csv"
        if not csv_path.exists():
            continue
        if sport_key not in sport_snapshots:
            continue

        output_path = OUTPUT_DIR / f"{event_id}.csv"
        result = enrich_csv_pinnacle(
            csv_path,
            sport_snapshots[sport_key],
            sport_snapshot_times[sport_key],
            output_path,
        )
        result["sport_key"] = sport_key
        result["competition"] = meta_row["competition_name"]
        summary.append(result)

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "_pinnacle_enrichment_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    total_rows = summary_df["rows"].sum()
    total_matched = summary_df["matched"].sum()
    full_matches = (summary_df["matched"] > 0).sum()
    no_matches = (summary_df["matched"] == 0).sum()

    print()
    print(f"   Processed: {len(summary)} matches")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Rows with Pinnacle odds: {total_matched:,} ({total_matched/total_rows:.1%})")
    print(f"   Matches with any Pinnacle odds: {full_matches}")
    print(f"   Matches with no Pinnacle odds: {no_matches}")
    print(f"   Output: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
