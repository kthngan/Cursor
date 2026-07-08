"""
Fetch poly-rs unified sports / leagues / markets and write tabular data to CSV
next to this script.

Uses header: x-api-key (default ``your_secret_key``, or env POLY_RS_API_KEY).

Outputs (same directory as this file):
  unified_sports.csv
  unified_leagues.csv
  unified_markets.csv

If the response is not a list of objects (or common ``data`` / ``items`` wrappers),
writes ``unified_<name>_raw.json`` instead and skips that CSV (or writes a one-row
error CSV when the HTTP layer fails).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "https://poly-rs-data-api.it9.win/api/v1/unified"
DEFAULT_KEY = "your_secret_key"
OUT_DIR = Path(__file__).resolve().parent
# Same as accountSummary/analytics: Cloudflare 403 / "error code: 1010" with Python-urllib default UA.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch(path_suffix: str, api_key: str) -> Any:
    url = f"{BASE}/{path_suffix.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "x-api-key": api_key,
            "Accept": "application/json",
            "User-Agent": _BROWSER_UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")[:4000]
        except Exception:
            detail = ""
        msg = f"HTTP {e.code} {e.reason}"
        if detail:
            msg += f" | {detail}"
        return {"_error": msg}
    except urllib.error.URLError as e:
        r = e.reason if getattr(e, "reason", None) else e
        return {"_error": str(r)}
    except Exception as e:
        return {"_error": str(e)}
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON: {e}"}


def unwrap_rows(obj: Any) -> list[dict[str, Any]] | None:
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "items", "results", "records"):
            v = obj.get(key)
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                return v
    return None


def cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return str(v)


def column_order(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for r in rows:
        for k in r:
            sk = str(k)
            if sk not in keys:
                keys.append(sk)
    return keys


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = column_order(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: cell_str(r.get(c)) for c in cols})


def write_error_csv(path: Path, message: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error"])
        w.writerow([message])


def main() -> int:
    api_key = os.environ.get("POLY_RS_API_KEY", DEFAULT_KEY)
    specs = (
        ("sports", "unified_sports.csv"),
        ("leagues", "unified_leagues.csv"),
        ("markets", "unified_markets.csv"),
    )

    for suffix, csv_name in specs:
        csv_path = OUT_DIR / csv_name
        raw_path = OUT_DIR / csv_name.replace(".csv", "_raw.json")
        print(f"GET {BASE}/{suffix} -> {csv_path.name}", flush=True)
        payload = fetch(suffix, api_key)
        if isinstance(payload, dict) and "_error" in payload:
            write_error_csv(csv_path, str(payload["_error"]))
            print(f"  wrote error row to {csv_path.name}", flush=True)
            continue
        rows = unwrap_rows(payload)
        if rows is None:
            raw_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"  non-tabular: wrote {raw_path.name} (no CSV rows)", flush=True)
            continue
        write_rows_csv(csv_path, rows)
        print(f"  {len(rows)} rows -> {csv_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
