#!/usr/bin/env python3
"""Export rolling tennis live-form metrics for accessible historical matches."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sportsapi_metric_helpers import compute_rolling_metric_rows


CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
BASE_URL = "https://api.statscore.com/v2"
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
OUTPUT_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
SUMMARY_PATH = OUTPUT_DIR / "_export_summary.csv"
MAX_WORKERS = int(os.environ.get("STATSCORE_EXPORT_WORKERS", "6"))


def get_json(url: str, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
        except TimeoutError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def authenticate() -> str:
    query = urllib.parse.urlencode({"client_id": CLIENT_ID, "secret_key": SECRET_KEY})
    payload = get_json(f"{BASE_URL}/oauth?{query}")
    error = payload.get("api", {}).get("error")
    if error:
        raise RuntimeError(f"Authentication failed: {error}")
    return payload["api"]["data"]["token"]


def request(path: str, token: str, **params: Any) -> dict[str, Any]:
    query = {"token": token}
    if CLIENT_ID:
        query["client_id"] = CLIENT_ID
    query.update({key: value for key, value in params.items() if value is not None})
    return get_json(f"{BASE_URL}/{path}?{urllib.parse.urlencode(query)}")


def extract_season_ids(payload: dict[str, Any]) -> list[int]:
    season_ids: set[int] = set()
    for competition in payload.get("api", {}).get("data", {}).get("competitions", []):
        for season in competition.get("seasons", []):
            season_id = season.get("id")
            if isinstance(season_id, int):
                season_ids.add(season_id)
    return sorted(season_ids)


def extract_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for competition in payload.get("api", {}).get("data", {}).get("competitions", []):
        for season in competition.get("seasons", []):
            for stage in season.get("stages", []):
                for group in stage.get("groups", []):
                    for event in group.get("events", []):
                        event = dict(event)
                        event.setdefault("competition_name", competition.get("name", ""))
                        event.setdefault("season_name", season.get("name", ""))
                        event.setdefault("stage_name", stage.get("name", ""))
                        events.append(event)
    return events


def fetch_all_events_for_season(token: str, season_id: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = request("events", token, season_id=season_id, events_details="yes", page=page)
        api = payload.get("api", {})
        error = api.get("error")
        if error:
            raise RuntimeError(f"events season={season_id} page={page} failed: {error}")
        events.extend(extract_events(payload))
        if not api.get("method", {}).get("next_page"):
            break
        page += 1
    return events


def flatten_event_show(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["api"]["data"]["competition"]["season"]["stage"]["group"]["event"]


def fetch_event_show(token: str, event_id: int) -> dict[str, Any] | None:
    try:
        payload = request(f"events/{event_id}", token)
    except urllib.error.HTTPError as exc:
        print(f"  skip event {event_id}: HTTP {exc.code}")
        return None
    error = payload.get("api", {}).get("error")
    if error:
        print(f"  skip event {event_id}: {error}")
        return None
    return flatten_event_show(payload)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_existing_summary(path: Path, event: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_row: dict[str, Any] | None = None
    row_count = 0
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "event_time" not in (reader.fieldnames or []):
            return None
        for row in reader:
            row_count += 1
            last_row = row
    if not last_row:
        return None
    return {
        "event_id": event.get("id", ""),
        "event_name": event.get("name", ""),
        "start_date": event.get("start_date", ""),
        "coverage_type": event.get("coverage_type", ""),
        "event_stats_lvl_live": event.get("event_stats_lvl_live", ""),
        "incident_rows": row_count,
        "output_file": str(path),
        "match_winner_name": last_row.get("match_winner_name", ""),
        "match_winner_side": last_row.get("match_winner_side", ""),
    }


def process_event(token: str, event_id: int, candidate_event: dict[str, Any]) -> dict[str, Any] | None:
    output_path = OUTPUT_DIR / f"{event_id}.csv"
    existing_summary = read_existing_summary(output_path, candidate_event)
    if existing_summary:
        existing_summary["status"] = "existing"
        return existing_summary

    event = fetch_event_show(token, event_id)
    if not event:
        return None
    incidents = event.get("events_incidents", [])
    if not incidents:
        return None
    rows = compute_rolling_metric_rows(event)
    if not rows:
        return None

    write_rows(output_path, rows)
    last_row = rows[-1]
    return {
        "event_id": event_id,
        "event_name": event.get("name", ""),
        "start_date": event.get("start_date", ""),
        "coverage_type": event.get("coverage_type", ""),
        "event_stats_lvl_live": event.get("event_stats_lvl_live", ""),
        "incident_rows": len(rows),
        "output_file": str(output_path),
        "match_winner_name": last_row.get("match_winner_name", ""),
        "match_winner_side": last_row.get("match_winner_side", ""),
        "status": "written",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = authenticate()

    seasons_payload = request("seasons", token)
    season_ids = extract_season_ids(seasons_payload)
    if not season_ids:
        raise RuntimeError("No accessible seasons found.")

    print(f"Accessible seasons: {len(season_ids)}")
    candidate_events: dict[int, dict[str, Any]] = {}
    for season_id in season_ids:
        events = fetch_all_events_for_season(token, season_id)
        finished_events = [
            event
            for event in events
            if event.get("id")
            and event.get("status_type") == "finished"
            and event.get("scoutsfeed") == "yes"
        ]
        for event in finished_events:
            candidate_events[int(event["id"])] = event
        print(f"  season {season_id}: {len(events)} events, {len(finished_events)} finished with scoutsfeed", flush=True)

    print(f"Unique candidate matches: {len(candidate_events)}", flush=True)
    summary_rows: list[dict[str, Any]] = []

    event_items = sorted(candidate_events.items())
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_event, token, event_id, event): event_id
            for event_id, event in event_items
        }
        for future in as_completed(futures):
            completed += 1
            event_id = futures[future]
            try:
                summary = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{completed}/{len(event_items)}] failed {event_id}: {exc}", flush=True)
                continue
            if summary:
                status = summary.pop("status", "written")
                summary_rows.append(summary)
                print(
                    f"  [{completed}/{len(event_items)}] {status} {event_id}.csv ({summary['incident_rows']} rows)",
                    flush=True,
                )
            else:
                print(f"  [{completed}/{len(event_items)}] skipped {event_id}", flush=True)

    if summary_rows:
        write_rows(SUMMARY_PATH, summary_rows)

    print(f"Matches exported: {len(summary_rows)}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)
    print(f"Summary: {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
