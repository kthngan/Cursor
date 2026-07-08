#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


CLIENT_ID = os.environ.get("STATSCORE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("STATSCORE_SECRET_KEY", "")
BASE_URL = "https://api.statscore.com/v2"
EVENT_ID = 6571587
PARTICIPANTS = [1081678, 1016468]


def get(url: str) -> tuple[int, dict[str, Any] | str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body[:500]


def token() -> str:
    query = urllib.parse.urlencode({"client_id": CLIENT_ID, "secret_key": SECRET_KEY})
    status, payload = get(f"{BASE_URL}/oauth?{query}")
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(payload)
    return payload["api"]["data"]["token"]


def summarize(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return payload
    api = payload.get("api", {})
    if "error" in api:
        return str(api["error"])
    method = api.get("method", {})
    data = api.get("data", {})
    keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
    return f"method={method.get('name')} total={method.get('total_items')} data_keys={keys}"


def main() -> int:
    auth_token = token()
    candidates = [
        f"events/{EVENT_ID}",
        f"events/{EVENT_ID}/participants",
        f"events/{EVENT_ID}/sub-participants",
        f"events/{EVENT_ID}/incidents",
        "participants/1081678",
        "participants/1016468",
        "participants/compare",
        "seasons/personal-stats",
        "seasons/participants-stats",
        "statuses",
        "incidents",
        "sports/4",
    ]
    params_by_path = {
        "participants/compare": {"participant_id": ",".join(map(str, PARTICIPANTS))},
        "seasons/personal-stats": {"season_id": 70664},
        "seasons/participants-stats": {"season_id": 70664},
        "statuses": {"sport_id": 4},
        "incidents": {"sport_id": 4},
    }
    for path in candidates:
        query = {"token": auth_token}
        query.update(params_by_path.get(path, {}))
        url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(query)}"
        status, payload = get(url)
        print(f"{path:35} HTTP {status:<3} {summarize(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
