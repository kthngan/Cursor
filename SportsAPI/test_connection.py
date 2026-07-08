#!/usr/bin/env python3
"""
StatScore SportsAPI connection test and dataset explorer.

Tests REST API authentication, probes available endpoints,
and prints a summary table of accessible data for the account.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

# Credentials - provide via environment variables.
CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
USERNAME = os.environ.get("STATSCORE_USERNAME", "")

BASE_URL = "https://api.statscore.com/v2"
ALLOWED_COMPETITIONS = {
    3498: "Wimbledon",
    3500: "Roland Garros",
}
TENNIS_SPORT_ID = 4


@dataclass
class EndpointResult:
    endpoint: str
    status: str
    total_items: str | int
    notes: str


class StatScoreClient:
    def __init__(self, client_id: str, secret_key: str) -> None:
        self.client_id = client_id
        self.secret_key = secret_key
        self.token: str | None = None

    def _request(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        query = {"token": self.token, "client_id": self.client_id}
        if params:
            query.update({k: v for k, v in params.items() if v is not None})

        url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"raw": body[:500]}
            return {"api": {"error": {"message": str(exc), "status": exc.code}, "data": payload}}

    def authenticate(self) -> dict[str, Any]:
        url = (
            f"{BASE_URL}/oauth?"
            f"{urllib.parse.urlencode({'client_id': self.client_id, 'secret_key': self.secret_key})}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode())

        error = payload.get("api", {}).get("error")
        if error:
            raise RuntimeError(f"Authentication failed: {error.get('message', error)}")

        data = payload["api"]["data"]
        self.token = data["token"]
        return data

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        return self._request(path, params or None)


def count_events_in_payload(data: dict[str, Any]) -> int:
    total = 0
    for competition in data.get("competitions", []):
        for season in competition.get("seasons", []):
            for stage in season.get("stages", []):
                for group in stage.get("groups", []):
                    total += len(group.get("events", []))
    return total


def extract_event_samples(data: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for competition in data.get("competitions", []):
        for season in competition.get("seasons", []):
            for stage in season.get("stages", []):
                for group in stage.get("groups", []):
                    for event in group.get("events", []):
                        samples.append(
                            {
                                "competition": competition.get("name"),
                                "season": season.get("name"),
                                "stage": stage.get("name"),
                                "event_id": event.get("id"),
                                "name": event.get("name"),
                                "status": event.get("status_type"),
                                "start_date": event.get("start_date"),
                            }
                        )
                        if len(samples) >= limit:
                            return samples
    return samples


def probe_endpoint(client: StatScoreClient, endpoint: str, params: dict[str, Any] | None = None) -> EndpointResult:
    params = params or {}
    try:
        payload = client.get(endpoint, **params)
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(endpoint, "FAIL", "-", str(exc))

    api = payload.get("api", {})
    error = api.get("error")
    if error:
        return EndpointResult(
            endpoint,
            f"HTTP {error.get('status', '?')}",
            "-",
            error.get("message", "Unknown error"),
        )

    method = api.get("method", {})
    total = method.get("total_items", "-")
    data = api.get("data", {})

    notes: list[str] = []
    if endpoint == "sports":
        sports = data.get("sports", [])
        tennis = next((s for s in sports if s.get("id") == TENNIS_SPORT_ID), None)
        if tennis:
            notes.append(f"Tennis id={TENNIS_SPORT_ID}")
    elif endpoint == "competitions":
        names = [c.get("name") for c in data.get("competitions", [])[:5]]
        notes.append(", ".join(n for n in names if n))
    elif endpoint == "events":
        event_count = count_events_in_payload(data)
        notes.append(f"{event_count} events in tree")
    elif endpoint == "seasons":
        season_count = sum(len(c.get("seasons", [])) for c in data.get("competitions", []))
        notes.append(f"{season_count} seasons")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                notes.append(f"{key}: {len(value)}")
                break

    return EndpointResult(endpoint, "OK", total, "; ".join(notes) if notes else "accessible")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    divider = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(divider)
    for row in rows:
        print(sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def main() -> int:
    print("=" * 72)
    print("StatScore SportsAPI - Connection Test & Dataset Summary")
    print("=" * 72)
    print(f"Client ID : {CLIENT_ID}")
    print(f"Username  : {USERNAME}")
    print(f"Base URL  : {BASE_URL}")
    print()

    client = StatScoreClient(CLIENT_ID, SECRET_KEY)

    # --- Authentication ---
    print("[1] Testing REST API authentication...")
    try:
        auth = client.authenticate()
        print(f"    Status  : SUCCESS")
        print(f"    Token   : {auth['token'][:8]}...{auth['token'][-4:]}")
        print(f"    Expires : {auth.get('token_expiration', 'N/A')} (unix)")
    except Exception as exc:  # noqa: BLE001
        print(f"    Status  : FAILED - {exc}")
        return 1
    print()

    # --- Endpoint probe ---
    print("[2] Probing available API endpoints...")
    probes: list[tuple[str, dict[str, Any] | None]] = [
        ("sports", None),
        ("areas", None),
        ("competitions", None),
        ("seasons", None),
        ("events", {"events_details": "yes"}),
        ("booked_events", None),
        ("languages", None),
        ("participants", None),
        ("standings", None),
        ("incidents", None),
        ("lineups", None),
        ("stats", None),
    ]

    results: list[EndpointResult] = []
    for endpoint, params in probes:
        results.append(probe_endpoint(client, endpoint, params))

    rows = [[r.endpoint, r.status, str(r.total_items), r.notes] for r in results]
    print()
    print_table(["Endpoint", "Status", "Total Items", "Notes"], rows)
    print()

    # --- Allowed competitions detail ---
    print("[3] Allowed competitions (account scope)...")
    comp_rows: list[list[str]] = []
    for comp_id, comp_name in ALLOWED_COMPETITIONS.items():
        try:
            seasons_payload = client.get("seasons", competition_id=comp_id)
            competitions = seasons_payload["api"]["data"].get("competitions", [])
            season_names = []
            for comp in competitions:
                for season in comp.get("seasons", []):
                    season_names.append(f"{season.get('name')} (id={season.get('id')})")
            comp_rows.append([comp_name, str(comp_id), "Tennis", ", ".join(season_names) or "n/a"])
        except Exception as exc:  # noqa: BLE001
            comp_rows.append([comp_name, str(comp_id), "Tennis", f"Error: {exc}"])

    print_table(["Competition", "ID", "Sport", "Seasons"], comp_rows)
    print()

    # --- Events summary ---
    print("[4] Events dataset summary...")
    events_payload = client.get("events", events_details="yes")
    events_data = events_payload["api"]["data"]
    total_events = count_events_in_payload(events_data)
    samples = extract_event_samples(events_data, limit=5)

    print(f"    Total events accessible : {total_events}")
    if samples:
        print()
        sample_rows = [
            [
                str(s["event_id"]),
                s["competition"] or "",
                s["season"] or "",
                s["name"] or "",
                s["status"] or "",
                s["start_date"] or "",
            ]
            for s in samples
        ]
        print_table(
            ["Event ID", "Competition", "Season", "Match", "Status", "Start (UTC)"],
            sample_rows,
        )
    print()

    # --- AMQP info (not tested — requires IP whitelist) ---
    print("[5] AMQP connection details (for reference - not tested here)")
    amqp_rows = [
        ["Server", "queue.statscore.com"],
        ["Port", "5672 (SSL)"],
        ["Virtual Host", "statscore"],
        ["Queue", USERNAME],
        ["User", USERNAME],
        ["Note", "IP whitelist required before queue is active"],
    ]
    print_table(["Parameter", "Value"], amqp_rows)
    print()

    # --- Public IP for whitelist ---
    print("[6] Your public IP (send to StatScore for AMQP whitelist)...")
    try:
        with urllib.request.urlopen("https://api.ipify.org?format=json", timeout=10) as resp:
            public_ip = json.loads(resp.read().decode()).get("ip", "unknown")
        print(f"    Public IP: {public_ip}")
    except Exception as exc:  # noqa: BLE001
        print(f"    Could not detect public IP: {exc}")

    print()
    print("=" * 72)
    print("Connection test complete.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
