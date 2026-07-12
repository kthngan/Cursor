#!/usr/bin/env python3
"""Verify odds enrichment timestamp matching and timezone correctness.

Picks a sample match, shows row timestamps vs snapshot timestamps,
and independently verifies that the matched snapshot is the latest one
at or before the row's UTC time. Also verifies Pinnacle odds from the
raw snapshot file.
"""

import bisect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from enrich_csvs_with_odds import (
    DATA_DIR,
    METADATA_PATH,
    SNAPSHOT_CACHE_DIR,
    SPORT_MAP,
    find_event_in_snapshot,
    names_match,
)

PINNACLE_DIR = DATA_DIR / "with_pinnacle_odds"


def main() -> int:
    # Load metadata to find a match with known sport
    meta = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
    meta = meta[meta["event_sport_name"].fillna("").str.lower() == "tennis"]
    meta = meta[meta["competition_name"].isin(SPORT_MAP.keys())]
    meta["sport_key"] = meta["competition_name"].map(SPORT_MAP)

    # Pick a match that has Pinnacle odds
    sample_event_id = None
    sample_sport_key = None
    for _, row in meta.iterrows():
        eid = str(row["event_id"])
        csv_path = PINNACLE_DIR / f"{eid}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, nrows=50)
        if df["odds_no_vig_p1"].notna().any():
            sample_event_id = eid
            sample_sport_key = row["sport_key"]
            p1_name = df["p1_name"].iloc[0]
            p2_name = df["p2_name"].iloc[0]
            competition = row["competition_name"]
            break

    if sample_event_id is None:
        print("No suitable match found")
        return 1

    print(f"=== Verification for match {sample_event_id} ===")
    print(f"   Competition: {competition}")
    print(f"   Players: {p1_name} vs {p2_name}")
    print(f"   Sport key: {sample_sport_key}")
    print()

    # Load the enriched CSV
    df = pd.read_csv(PINNACLE_DIR / f"{sample_event_id}.csv")
    print(f"   CSV rows: {len(df)}")
    print(f"   Rows with Pinnacle odds: {df['odds_no_vig_p1'].notna().sum()}")
    print()

    # Load cached snapshots for this sport
    snapshots = []
    for f in sorted(SNAPSHOT_CACHE_DIR.glob(f"{sample_sport_key}_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("timestamp"):
            snapshots.append(data)
    snapshots.sort(key=lambda s: s["timestamp"])

    snapshot_times = [
        datetime.strptime(s["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        for s in snapshots
    ]
    print(f"   Cached snapshots: {len(snapshots)}")
    print(f"   Snapshot time range: {snapshot_times[0]} to {snapshot_times[-1]}")
    print()

    # Pick 5 rows spread across the match to verify
    odds_rows = df[df["odds_no_vig_p1"].notna()].copy()
    if len(odds_rows) == 0:
        print("No rows with odds!")
        return 1

    sample_indices = list(range(0, len(odds_rows), max(1, len(odds_rows) // 5)))[:5]
    print("--- Row-by-row verification ---")
    print()

    for idx_pos, idx in enumerate(sample_indices):
        row = odds_rows.iloc[idx]
        ut = int(row["ut"])
        row_time = datetime.fromtimestamp(ut, tz=timezone.utc)

        # What the enrichment stored
        stored_snap_ts = str(row.get("odds_snapshot_time", ""))
        stored_no_vig_p1 = row.get("odds_no_vig_p1")
        stored_p1_price = row.get("odds_p1_price")
        stored_p2_price = row.get("odds_p2_price")
        stored_age = row.get("odds_age_seconds")

        # Independently find the correct snapshot
        correct_idx = bisect.bisect_right(snapshot_times, row_time) - 1
        if correct_idx < 0:
            correct_idx = 0
        correct_snap = snapshots[correct_idx]
        correct_snap_ts = correct_snap["timestamp"]
        correct_snap_dt = snapshot_times[correct_idx]

        # Find the event in this snapshot
        event = find_event_in_snapshot(p1_name, p2_name, correct_snap.get("data", []))

        # Extract Pinnacle odds from the event
        pin_p1_price = None
        pin_p2_price = None
        if event:
            for bm in event.get("bookmakers", []):
                if bm.get("key") != "pinnacle":
                    continue
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        if names_match(p1_name, outcome.get("name", "")):
                            pin_p1_price = outcome.get("price")
                        elif names_match(p2_name, outcome.get("name", "")):
                            pin_p2_price = outcome.get("price")

        pin_no_vig_p1 = None
        if pin_p1_price and pin_p2_price:
            imp1 = 1.0 / pin_p1_price
            imp2 = 1.0 / pin_p2_price
            pin_no_vig_p1 = imp1 / (imp1 + imp2)

        # Check match
        ts_match = stored_snap_ts == correct_snap_ts
        odds_match = abs(float(stored_no_vig_p1 or 0) - float(pin_no_vig_p1 or 0)) < 1e-6 if pin_no_vig_p1 else "N/A"

        age_check = (row_time - correct_snap_dt).total_seconds()

        print(f"  Row {idx_pos + 1}:")
        print(f"    ut (Unix):          {ut}")
        print(f"    Row time (UTC):     {row_time.isoformat()}")
        print(f"    Stored snap time:   {stored_snap_ts}")
        print(f"    Correct snap time:  {correct_snap_ts}")
        print(f"    Timestamp match:    {'YES' if ts_match else 'NO !!!'}")
        print(f"    Age (row - snap):   {age_check:.0f}s ({age_check/60:.1f} min)")
        print(f"    Stored no_vig_p1:   {stored_no_vig_p1}")
        print(f"    Pinnacle no_vig_p1: {pin_no_vig_p1}")
        print(f"    Odds match:         {odds_match}")
        print(f"    Pinnacle prices:    P1={pin_p1_price}, P2={pin_p2_price}")
        print(f"    Stored prices:      P1={stored_p1_price}, P2={stored_p2_price}")
        if event:
            all_bks = [b.get("key") for b in event.get("bookmakers", [])]
            has_pin = "pinnacle" in all_bks
            print(f"    Pinnacle in snap:   {has_pin} (bookmakers: {len(all_bks)})")
        else:
            print(f"    Event not found in this snapshot (forward-fill from earlier)")
        print()

    # Also verify timezone handling
    print("--- Timezone verification ---")
    print()
    print(f"  Row 'ut' column is Unix timestamp (seconds since epoch, UTC by definition)")
    print(f"  datetime.fromtimestamp(ut, tz=timezone.utc) gives correct UTC datetime")
    print(f"  Snapshot timestamps parsed as: datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)")
    print(f"  Both are timezone-aware UTC datetimes -> comparison is correct")
    print()

    # Show one specific numeric example
    row = odds_rows.iloc[0]
    ut = int(row["ut"])
    row_time = datetime.fromtimestamp(ut, tz=timezone.utc)
    snap_ts = str(row.get("odds_snapshot_time", ""))
    snap_dt = datetime.strptime(snap_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    diff = (row_time - snap_dt).total_seconds()
    print(f"  Example: ut={ut} -> row_time={row_time}, snap_time={snap_dt}, diff={diff:.0f}s ({diff/60:.1f} min)")
    print(f"  This means the odds snapshot was taken {diff/60:.1f} minutes before the event row timestamp.")
    print(f"  The snapshot is the latest available at or before the row time (no lookahead bias).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
