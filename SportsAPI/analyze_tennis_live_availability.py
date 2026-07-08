#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
BASE_URL = "https://api.statscore.com/v2"
SPORT_ID = 4
EVENT_ID = 6571587


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def authenticate() -> str:
    query = urllib.parse.urlencode({"client_id": CLIENT_ID, "secret_key": SECRET_KEY})
    return get_json(f"{BASE_URL}/oauth?{query}")["api"]["data"]["token"]


def request(path: str, token: str, **params: Any) -> dict[str, Any]:
    query = {"token": token}
    query.update({k: v for k, v in params.items() if v is not None})
    return get_json(f"{BASE_URL}/{path}?{urllib.parse.urlencode(query)}")


def names(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("name") or item.get("short_name") or "") for item in items]


def flatten_event(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["api"]["data"]["competition"]["season"]["stage"]["group"]["event"]


def main() -> int:
    token = authenticate()
    sports_show = request(f"sports/{SPORT_ID}", token)
    event_show = request(f"events/{EVENT_ID}", token)
    event = flatten_event(event_show)

    sport = sports_show["api"]["data"]["sport"]
    stats = sport.get("stats", {})
    details = sport.get("details", [])
    results = sport.get("results", [])
    incidents = sport.get("incidents", [])
    statuses = sport.get("statuses", [])

    event_stat_names = sorted(
        {
            stat.get("name", "")
            for participant in event.get("participants", [])
            for stat in participant.get("stats", [])
        }
    )
    event_result_names = sorted(
        {
            result.get("name", "")
            for participant in event.get("participants", [])
            for result in participant.get("results", [])
        }
    )
    event_status_stat_names = sorted(
        {
            stat.get("name", "")
            for participant in event.get("participants", [])
            for stat_group in participant.get("event_status_stats", {}).values()
            for stat in stat_group
        }
    )
    incident_names = sorted({incident.get("incident_name", "") for incident in event.get("events_incidents", [])})
    incident_counts = Counter(incident.get("incident_name", "") for incident in event.get("events_incidents", []))

    searchable = "\n".join(
        names(stats.get("team", []))
        + names(stats.get("person", []))
        + names(details)
        + names(results)
        + names(incidents)
        + event_stat_names
        + event_result_names
        + incident_names
    ).lower()
    keyword_hits = {
        keyword: keyword in searchable
        for keyword in [
            "speed",
            "serve speed",
            "run",
            "distance",
            "sprint",
            "form",
            "momentum",
            "rank",
            "pressure",
            "win probability",
            "break point",
        ]
    }

    summary = {
        "sport": {
            "id": sport.get("id"),
            "name": sport.get("name"),
            "has_timer": sport.get("has_timer"),
            "stats_team_count": len(stats.get("team", [])),
            "stats_person_count": len(stats.get("person", [])),
            "details_count": len(details),
            "results_count": len(results),
            "incidents_count": len(incidents),
            "statuses_count": len(statuses),
        },
        "event": {
            "id": event.get("id"),
            "name": event.get("name"),
            "coverage_type": event.get("coverage_type"),
            "scoutsfeed": event.get("scoutsfeed"),
            "event_stats_lvl": event.get("event_stats_lvl"),
            "event_stats_lvl_live": event.get("event_stats_lvl_live"),
            "event_stats_lvl_after": event.get("event_stats_lvl_after"),
            "event_stats_available": event_stat_names,
            "event_status_stats_available": event_status_stat_names,
            "event_results_available": event_result_names,
            "incident_types_observed": incident_names,
            "incident_counts": dict(incident_counts.most_common()),
        },
        "keyword_hits": keyword_hits,
        "sports_show_person_stats": names(stats.get("person", [])),
        "sports_show_team_stats": names(stats.get("team", [])),
        "sports_show_details": names(details),
        "sports_show_results": names(results),
        "sports_show_incidents": names(incidents),
    }

    output = Path(__file__).resolve().parent / "tennis_live_availability_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["sport"], indent=2))
    print()
    print(json.dumps(summary["event"], indent=2)[:6000])
    print()
    print("Keyword hits:")
    for keyword, hit in keyword_hits.items():
        print(f"  {keyword}: {hit}")
    print()
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
