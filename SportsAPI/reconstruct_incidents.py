#!/usr/bin/env python3
"""
Reconstruct a chronological incident table from a StatScore event.

Default example:
    Lois Boisson - Elena Rybakina, event_id=6571587
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
BASE_URL = "https://api.statscore.com/v2"
DEFAULT_EVENT_ID = "6571587"
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "SportsAPI"


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def authenticate() -> str:
    query = urllib.parse.urlencode({"client_id": CLIENT_ID, "secret_key": SECRET_KEY})
    payload = get_json(f"{BASE_URL}/oauth?{query}")
    error = payload.get("api", {}).get("error")
    if error:
        raise RuntimeError(f"Authentication failed: {error}")
    return payload["api"]["data"]["token"]


def fetch_event(event_id: str, token: str) -> dict[str, Any]:
    payload = get_json(f"{BASE_URL}/events/{event_id}?token={urllib.parse.quote(token)}")
    error = payload.get("api", {}).get("error")
    if error:
        raise RuntimeError(f"Event fetch failed: {error}")
    return payload["api"]["data"]["competition"]["season"]["stage"]["group"]["event"]


def unix_to_utc(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(value)


def incident_sort_key(incident: dict[str, Any]) -> tuple[int, str]:
    ut = incident.get("ut")
    try:
        ut_int = int(ut)
    except (TypeError, ValueError):
        ut_int = 0
    return ut_int, str(incident.get("id", ""))


def reconstruct_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    participants = event.get("participants", [])
    participant_side_by_id = {
        participant.get("id"): f"P{participant.get('counter')}" for participant in participants
    }
    participant_counter_by_id = {
        participant.get("id"): int(participant.get("counter") or 0) for participant in participants
    }

    games_by_status: dict[int, dict[int, int]] = {}
    sets_won = {1: 0, 2: 0}
    rows: list[dict[str, Any]] = []

    for sequence, incident in enumerate(sorted(event.get("events_incidents", []), key=incident_sort_key), start=1):
        status_id = incident.get("event_status_id")
        participant_id = incident.get("participant_id")
        counter = participant_counter_by_id.get(participant_id)
        incident_name = incident.get("incident_name") or ""

        if isinstance(status_id, int):
            games_by_status.setdefault(status_id, {1: 0, 2: 0})

        if incident_name == "Game Won" and isinstance(status_id, int) and counter in (1, 2):
            games_by_status[status_id][counter] += 1

        if incident_name == "Set won" and counter in (1, 2):
            sets_won[counter] += 1

        game_score_after = ""
        if isinstance(status_id, int) and status_id in games_by_status:
            score = games_by_status[status_id]
            game_score_after = f"{score[1]}-{score[2]}"

        point_to = incident_name if incident_name in {"0", "15", "30", "40", "A"} else ""

        rows.append(
            {
                "seq": sequence,
                "incident_record_id": incident.get("id", ""),
                "ut": incident.get("ut", ""),
                "utc_time": unix_to_utc(incident.get("ut")),
                "event_status_id": status_id if status_id is not None else "",
                "event_status_name": incident.get("event_status_name", ""),
                "event_time": incident.get("event_time", ""),
                "incident_id": incident.get("incident_id", ""),
                "incident_name": incident_name,
                "participant_side": participant_side_by_id.get(participant_id, ""),
                "participant_id": participant_id if participant_id is not None else "",
                "participant_name": incident.get("participant_name", ""),
                "point_to": point_to,
                "game_score_after": game_score_after,
                "sets_after": f"{sets_won[1]}-{sets_won[2]}",
                "confirmation": incident.get("confirmation", "") or "",
                "info": incident.get("info", "") or "",
                "for": incident.get("for", "") or "",
                "raw_attribute_ids": json.dumps(incident.get("attribute_ids", []), separators=(",", ":")),
                "raw_properties": json.dumps(incident.get("properties", []), separators=(",", ":")),
                "raw_additional_info": json.dumps(incident.get("additional_info", []), separators=(",", ":")),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]], limit: int = 40) -> None:
    headers = [
        "seq",
        "utc_time",
        "event_status_name",
        "incident_name",
        "participant_side",
        "participant_name",
        "point_to",
        "game_score_after",
        "sets_after",
    ]
    display_rows = rows[:limit]
    widths = {header: len(header) for header in headers}
    for row in display_rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in display_rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows written to CSV/JSON")


def main() -> int:
    event_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVENT_ID
    output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    token = authenticate()
    event = fetch_event(event_id, token)
    rows = reconstruct_rows(event)

    csv_path = output_dir / f"event_{event_id}_incidents_reconstructed.csv"
    json_path = output_dir / f"event_{event_id}_incidents_reconstructed.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"event": event, "rows": rows}, indent=2), encoding="utf-8")

    print(f"Event: {event.get('name')} ({event_id})")
    print(f"Incidents reconstructed: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
