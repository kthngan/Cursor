#!/usr/bin/env python3
"""Create a master CSV mapping match IDs to StatScore match metadata."""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
BASE_URL = "https://api.statscore.com/v2"
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
OUTPUT_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
SUMMARY_PATH = OUTPUT_DIR / "_export_summary.csv"
MASTER_PATH = OUTPUT_DIR / "master_match_metadata.csv"
MAX_WORKERS = int(os.environ.get("STATSCORE_METADATA_WORKERS", "4"))


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def get_json(url: str, retries: int = 4) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
        except TimeoutError:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
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


def read_match_ids_from_summary(path: Path) -> list[int]:
    with path.open(encoding="utf-8", newline="") as file:
        return [int(row["event_id"]) for row in csv.DictReader(file) if row.get("event_id")]


def context_from_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    competition = payload["api"]["data"]["competition"]
    season = competition["season"]
    stage = season["stage"]
    group = stage["group"]
    event = group["event"]
    return {
        "competition": competition,
        "season": season,
        "stage": stage,
        "group": group,
        "event": event,
    }


def scalar_fields(prefix: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in source.items()
        if not isinstance(value, (dict, list))
    }


def participant_by_counter(event: dict[str, Any], counter: int) -> dict[str, Any]:
    for participant in event.get("participants", []):
        if str(participant.get("counter")) == str(counter):
            return participant
    return {}


def winner_from_event(event: dict[str, Any]) -> dict[str, Any]:
    for participant in event.get("participants", []):
        for result in participant.get("results", []):
            if result.get("name") == "Winner" and str(result.get("value")) == "1":
                return participant
    return {}


def metadata_row(token: str, event_id: int) -> dict[str, Any] | None:
    try:
        context = context_from_event_payload(request(f"events/{event_id}", token))
    except urllib.error.HTTPError as exc:
        print(f"  skip {event_id}: HTTP {exc.code}", flush=True)
        return None

    event = context["event"]
    participants = event.get("participants", [])
    p1 = participant_by_counter(event, 1)
    p2 = participant_by_counter(event, 2)
    winner = winner_from_event(event)
    incidents = event.get("events_incidents", [])
    incident_counts = Counter(incident.get("incident_name", "") for incident in incidents)
    event_without_incidents = {key: value for key, value in event.items() if key != "events_incidents"}

    row: dict[str, Any] = {
        "event_id": event.get("id", event_id),
        "event_name": event.get("name", ""),
        "competition_id": context["competition"].get("id", ""),
        "competition_name": context["competition"].get("name", ""),
        "season_id": context["season"].get("id", ""),
        "season_name": context["season"].get("name", ""),
        "stage_id": context["stage"].get("id", ""),
        "stage_name": context["stage"].get("name", ""),
        "group_id": context["group"].get("id", ""),
        "group_name": context["group"].get("name", ""),
        "p1_id": p1.get("id", ""),
        "p1_name": p1.get("name", ""),
        "p2_id": p2.get("id", ""),
        "p2_name": p2.get("name", ""),
        "winner_id": winner.get("id", ""),
        "winner_name": winner.get("name", ""),
        "winner_side": f"P{winner.get('counter')}" if winner else "",
        "participants_count": len(participants),
        "events_incidents_count": len(incidents),
        "incident_type_counts_json": json_compact(dict(incident_counts.most_common())),
        "participants_json": json_compact(participants),
        "event_metadata_json": json_compact(event_without_incidents),
    }
    row.update(scalar_fields("event", event))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SUMMARY_PATH.exists():
        raise RuntimeError(f"Missing summary file: {SUMMARY_PATH}")

    token = authenticate()
    match_ids = read_match_ids_from_summary(SUMMARY_PATH)
    print(f"Match IDs from summary: {len(match_ids)}", flush=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(metadata_row, token, event_id): event_id for event_id in match_ids}
        for index, future in enumerate(as_completed(futures), start=1):
            event_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{index}/{len(match_ids)}] failed {event_id}: {exc}", flush=True)
                continue
            if row:
                rows.append(row)
                print(f"  [{index}/{len(match_ids)}] metadata {event_id}", flush=True)

    rows.sort(key=lambda row: int(row["event_id"]))
    write_csv(MASTER_PATH, rows)
    print(f"Metadata rows written: {len(rows)}", flush=True)
    print(f"Master CSV: {MASTER_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
