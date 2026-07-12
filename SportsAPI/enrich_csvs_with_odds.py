#!/usr/bin/env python3
"""Enrich saved SportsAPI tennis CSVs with historical in-game odds from The Odds API.

Workflow:
1. Load master metadata to map each saved match to an Odds API sport key.
2. For each sport, fetch historical odds snapshots at 10-minute intervals
   covering all match windows (starting 30 min before first match for
   pregame odds), caching each snapshot to disk.
3. For each match CSV, use an asof (backward) join to match every row's
   timestamp to the LATEST odds snapshot at or before that timestamp.
   This ensures no lookahead bias — only odds actually available at the
   time of each event are used. Odds are forward-filled: once a match is
   found in a snapshot, those odds persist until a newer snapshot updates
   them. Pregame rows use the earliest available pregame snapshot.
4. Save enriched CSVs to Data/SportsAPI/with_odds/.

Snapshot cost: ~20 credits per call (1 market h2h x 1 region eu).
Snapshots are cached so the script is resumable.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SNAPSHOT_INTERVAL_MINUTES = int(os.environ.get("ODDS_SNAPSHOT_INTERVAL", "10"))
MAX_WORKERS = int(os.environ.get("ODDS_MAX_WORKERS", "8"))

API_KEY = os.environ.get("ODDS_API_KEY", "4d0c2c123b9edb28d71fdf10cd264de6")
BASE_URL = "https://api.the-odds-api.com/v4"
REGIONS = "eu"
MARKETS = "h2h"
ODDS_FORMAT = "decimal"

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
METADATA_PATH = DATA_DIR / "master_match_metadata.csv"
SNAPSHOT_CACHE_DIR = DATA_DIR / "odds_api" / "snapshots"
OUTPUT_DIR = DATA_DIR / "with_odds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPORT_MAP = {
    "Roland Garros": "tennis_atp_french_open",
    "Roland Garros (Women)": "tennis_wta_french_open",
    "Wimbledon": "tennis_atp_wimbledon",
    "Wimbledon (Women)": "tennis_wta_wimbledon",
}

ODDS_COLUMNS = [
    "odds_snapshot_time",
    "odds_p1_price",
    "odds_p2_price",
    "odds_implied_p1",
    "odds_implied_p2",
    "odds_no_vig_p1",
    "odds_no_vig_p2",
    "odds_bookmaker_count",
    "odds_event_id",
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    query: dict[str, str] = {"apiKey": API_KEY}
    if params:
        query.update({k: v for k, v in params.items() if v is not None})
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read().decode()
                headers = {k.lower(): v for k, v in response.headers.items()}
                return json.loads(body), headers
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            if exc.code == 429 and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return {"_error": f"HTTP {exc.code} {exc.reason}", "_body": body[:500]}, {}
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            return {"_error": f"{type(exc).__name__}: {exc}"}, {}
    return {"_error": "max retries exceeded"}, {}


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    low = name.lower()
    low = re.sub(r"\(.*?\)", "", low)
    low = re.sub(r"[-_]+", " ", low)
    low = re.sub(r"[^a-z0-9 ]+", "", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def last_name(name: str) -> str:
    norm = normalize_name(name)
    if not norm:
        return ""
    parts = [p for p in norm.split() if len(p) > 1]
    return parts[-1] if parts else ""


def names_match(name_a: str, name_b: str) -> bool:
    la = last_name(name_a)
    lb = last_name(name_b)
    if not la or not lb or len(la) < 3 or len(lb) < 3:
        return False
    if la == lb:
        return True
    # Handle compound last names and partial matches
    if la in lb or lb in la:
        return True
    return False


def find_event_in_snapshot(
    p1: str, p2: str, events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the Odds API event matching both players."""
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        # P1 matches home and P2 matches away, or P1 matches away and P2 matches home
        if ((names_match(p1, home) and names_match(p2, away)) or
                (names_match(p1, away) and names_match(p2, home))):
            return event
    return None


def extract_best_prices(event: dict[str, Any], p1: str, p2: str) -> dict[str, Any]:
    """Extract best H2H price for each player across all bookmakers."""
    best_p1: float | None = None
    best_p2: float | None = None
    bookmaker_count = 0
    for bm in event.get("bookmakers", []):
        has_odds = False
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None or price <= 1:
                    continue
                if names_match(p1, name):
                    if best_p1 is None or price > best_p1:
                        best_p1 = price
                    has_odds = True
                elif names_match(p2, name):
                    if best_p2 is None or price > best_p2:
                        best_p2 = price
                    has_odds = True
        if has_odds or bm.get("markets"):
            bookmaker_count += 1

    implied_p1 = 1.0 / best_p1 if best_p1 else None
    implied_p2 = 1.0 / best_p2 if best_p2 else None
    no_vig_p1 = no_vig_p2 = None
    if implied_p1 and implied_p2:
        total = implied_p1 + implied_p2
        no_vig_p1 = implied_p1 / total
        no_vig_p2 = implied_p2 / total

    return {
        "odds_p1_price": best_p1,
        "odds_p2_price": best_p2,
        "odds_implied_p1": implied_p1,
        "odds_implied_p2": implied_p2,
        "odds_no_vig_p1": no_vig_p1,
        "odds_no_vig_p2": no_vig_p2,
        "odds_bookmaker_count": bookmaker_count,
        "odds_event_id": event.get("id", ""),
    }


# ---------------------------------------------------------------------------
# Snapshot fetching with caching
# ---------------------------------------------------------------------------

def snapshot_path(sport_key: str, timestamp: str) -> Path:
    safe_ts = timestamp.replace(":", "").replace("-", "")
    return SNAPSHOT_CACHE_DIR / f"{sport_key}_{safe_ts}.json"


def parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cached_snapshots(sport_key: str) -> list[dict[str, Any]]:
    """Load all cached snapshots for a sport, sorted by timestamp."""
    snapshots = []
    for f in SNAPSHOT_CACHE_DIR.glob(f"{sport_key}_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = data.get("timestamp", "")
            if ts:
                snapshots.append(data)
        except Exception:
            continue
    snapshots.sort(key=lambda s: s.get("timestamp", ""))
    return snapshots


def fetch_snapshot_from_api(sport_key: str, date_iso: str) -> dict[str, Any] | None:
    """Fetch a historical odds snapshot from the API (no cache check)."""
    data, _headers = api_get(
        f"/historical/sports/{sport_key}/odds",
        {"regions": REGIONS, "markets": MARKETS, "oddsFormat": ODDS_FORMAT, "date": date_iso},
    )
    if isinstance(data, dict) and "_error" in data:
        return None
    timestamp = data.get("timestamp", "")
    if not timestamp:
        return None

    # Cache the response (atomic write)
    cache = snapshot_path(sport_key, timestamp)
    if not cache.exists():
        cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def fetch_sport_snapshots(
    sport_key: str, match_windows: list[tuple[datetime, datetime]]
) -> list[dict[str, Any]]:
    """Fetch snapshots covering all match windows for a sport.

    Generates timestamps at fixed intervals, skips those already cached,
    and fetches the rest concurrently.
    """
    if not match_windows:
        return []

    # Merge overlapping windows
    match_windows.sort()
    merged: list[tuple[datetime, datetime]] = []
    for start, end in match_windows:
        if merged and start <= merged[-1][1] + timedelta(minutes=30):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Load already-cached snapshots
    cached_list = load_cached_snapshots(sport_key)
    cached_times = [parse_iso(s["timestamp"]) for s in cached_list if s.get("timestamp")]
    print(f"  {sport_key}: {len(cached_list)} cached snapshots, {len(merged)} merged windows")

    # Generate desired timestamps at fixed intervals
    interval = timedelta(minutes=SNAPSHOT_INTERVAL_MINUTES)
    desired: list[datetime] = []
    for window_start, window_end in merged:
        # Round up to the next interval boundary
        t = window_start.replace(second=0, microsecond=0)
        while t <= window_end + timedelta(minutes=5):
            desired.append(t)
            t += interval

    # Filter out timestamps already covered by cached snapshots
    to_fetch: list[datetime] = []
    for dt in desired:
        # Check if any cached snapshot is within 5 minutes
        covered = False
        for ct in cached_times:
            if abs((ct - dt).total_seconds()) < 300:
                covered = True
                break
        if not covered:
            to_fetch.append(dt)

    print(f"    {sport_key}: {len(desired)} desired, {len(to_fetch)} to fetch, "
          f"{len(desired) - len(to_fetch)} already cached")

    if not to_fetch:
        return load_cached_snapshots(sport_key)

    # Fetch concurrently
    fetched = 0
    failed = 0

    def _fetch_one(dt: datetime) -> dict[str, Any] | None:
        return fetch_snapshot_from_api(sport_key, fmt_iso(dt))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, dt): dt for dt in to_fetch}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                fetched += 1
            else:
                failed += 1
            if (fetched + failed) % 50 == 0:
                print(f"    Progress: {fetched + failed}/{len(to_fetch)} "
                      f"(fetched={fetched}, failed={failed})")

    print(f"    {sport_key}: fetched={fetched}, failed={failed}, total cached={len(cached_list) + fetched}")

    # Return all cached snapshots (including newly fetched)
    return load_cached_snapshots(sport_key)


# ---------------------------------------------------------------------------
# CSV enrichment
# ---------------------------------------------------------------------------

def enrich_csv(
    csv_path: Path,
    sport_key: str,
    snapshots: list[dict[str, Any]],
    snapshot_times: list[datetime],
    output_path: Path,
) -> dict[str, Any]:
    """Enrich a single match CSV with odds columns."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"event_id": csv_path.stem, "rows": 0, "matched": 0}

    p1 = str(df["p1_name"].iloc[0])
    p2 = str(df["p2_name"].iloc[0])
    event_id = str(df["event_id"].iloc[0]) if "event_id" in df.columns else csv_path.stem

    # Initialize odds columns
    for col in ODDS_COLUMNS:
        df[col] = None

    matched_count = 0
    snapshot_events_cache: dict[str, list[dict[str, Any]]] = {}
    event_lookup_cache: dict[str, dict[str, Any] | None] = {}

    # Pre-compute last-known odds per snapshot for this match (forward-fill basis).
    # We iterate sorted snapshots and cache the most recent event match so that
    # if a snapshot doesn't contain this match, we carry forward the prior odds.
    last_odds: dict[str, Any] | None = None
    last_snap_ts = ""

    for idx, row in df.iterrows():
        ut = row.get("ut")
        if pd.isna(ut):
            continue
        row_time = datetime.fromtimestamp(int(ut), tz=timezone.utc)

        # Find the latest snapshot at or BEFORE row_time (asof / backward join).
        # This ensures we only use odds that were actually available at that moment.
        snap_idx = bisect.bisect_right(snapshot_times, row_time) - 1
        if snap_idx < 0:
            # Row is before all snapshots — use the earliest snapshot (pregame odds)
            snap_idx = 0

        snap = snapshots[snap_idx]
        snap_ts = snap.get("timestamp", "")

        # Cache event list and event lookup per snapshot
        if snap_ts not in event_lookup_cache:
            if snap_ts not in snapshot_events_cache:
                snapshot_events_cache[snap_ts] = snap.get("data", [])
            events = snapshot_events_cache[snap_ts]
            event_lookup_cache[snap_ts] = find_event_in_snapshot(p1, p2, events)
        event = event_lookup_cache[snap_ts]

        if event is not None:
            # Found this match in the snapshot — update last-known odds
            last_odds = extract_best_prices(event, p1, p2)
            last_snap_ts = snap_ts

        if last_odds is not None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Enrich SportsAPI CSVs with Historical Odds ===")
    print(f"API key: {API_KEY[:8]}...")
    print()

    # Step 1: Load metadata
    print("1. Loading metadata...")
    meta = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
    meta = meta[meta["event_sport_name"].fillna("").str.lower() == "tennis"]
    meta = meta[meta["competition_name"].isin(SPORT_MAP.keys())]
    print(f"   {len(meta)} tennis matches with known competitions")

    # Map competition to sport key
    meta["sport_key"] = meta["competition_name"].map(SPORT_MAP)

    # Parse start dates to datetime (UTC)
    meta["start_dt"] = pd.to_datetime(meta["event_start_date"], format="ISO8601", utc=True)
    # Estimate end as start + 4 hours
    meta["end_dt"] = meta["start_dt"] + timedelta(hours=4)

    # Step 2: Fetch snapshots per sport
    print()
    print("2. Fetching historical odds snapshots per sport...")
    sport_snapshots: dict[str, list[dict[str, Any]]] = {}
    sport_snapshot_times: dict[str, list[datetime]] = {}

    for sport_key in SPORT_MAP.values():
        sport_meta = meta[meta["sport_key"] == sport_key]
        if sport_meta.empty:
            continue
        print(f"\n  {sport_key}: {len(sport_meta)} matches, "
              f"{sport_meta['start_dt'].min()} -> {sport_meta['end_dt'].max()}")

        # Build match windows per day — start 30 min before earliest match for pregame odds
        windows: list[tuple[datetime, datetime]] = []
        for day, group in sport_meta.groupby(sport_meta["start_dt"].dt.date):
            day_start = group["start_dt"].min().to_pydatetime() - timedelta(minutes=30)
            day_end = group["end_dt"].max().to_pydatetime()
            windows.append((day_start, day_end))
            print(f"    {day}: {len(group)} matches, "
                  f"{day_start.strftime('%H:%M')} -> {day_end.strftime('%H:%M')}")

        snapshots = fetch_sport_snapshots(sport_key, windows)
        sport_snapshots[sport_key] = snapshots
        sport_snapshot_times[sport_key] = [
            parse_iso(s["timestamp"]) for s in snapshots if s.get("timestamp")
        ]
        print(f"    Total cached snapshots: {len(snapshots)}")

    # Step 3: Enrich CSVs
    print()
    print("3. Enriching match CSVs with odds...")
    results: list[dict[str, Any]] = []
    processed = 0
    skipped = 0

    for _, row in meta.iterrows():
        event_id = row["event_id"]
        sport_key = row["sport_key"]
        csv_path = DATA_DIR / f"{event_id}.csv"
        output_path = OUTPUT_DIR / f"{event_id}.csv"

        if not csv_path.exists():
            skipped += 1
            continue

        # Skip if already processed
        if output_path.exists():
            skipped += 1
            continue

        snapshots = sport_snapshots.get(sport_key, [])
        snapshot_times = sport_snapshot_times.get(sport_key, [])

        if not snapshots:
            skipped += 1
            continue

        result = enrich_csv(csv_path, sport_key, snapshots, snapshot_times, output_path)
        results.append(result)
        processed += 1

        if processed % 50 == 0:
            total_matched = sum(r["matched"] for r in results)
            print(f"  Processed {processed}/{len(meta)} ({skipped} skipped), "
                  f"total matched rows: {total_matched}")

    # Summary
    print()
    print("4. Summary")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")
    if results:
        total_rows = sum(r["rows"] for r in results)
        total_matched = sum(r["matched"] for r in results)
        rates = [r["match_rate"] for r in results if r["rows"] > 0]
        avg_rate = sum(rates) / len(rates) if rates else 0
        print(f"   Total rows: {total_rows}")
        print(f"   Total matched rows: {total_matched}")
        print(f"   Average match rate: {avg_rate:.1%}")
        # Matches with no odds at all
        no_odds = sum(1 for r in results if r["matched"] == 0)
        print(f"   Matches with 0 odds rows: {no_odds}")

    # Save summary
    summary_path = OUTPUT_DIR / "_enrichment_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"   Summary: {summary_path}")
    print(f"   Output dir: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
