"""
Find the latest YYYY-MM-DD.json in a folder and print unique account IDs (groups.user_id).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

DATE_JSON = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$", re.IGNORECASE)
DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "deltaReportNew"


def latest_json_path(directory: Path) -> Path | None:
    candidates: list[tuple[dt.date, Path]] = []
    for p in directory.glob("*.json"):
        m = DATE_JSON.match(p.name)
        if m:
            candidates.append((dt.date.fromisoformat(m.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def unique_user_ids(data: dict) -> list[int]:
    seen: set[int] = set()
    for g in data.get("groups") or []:
        if not isinstance(g, dict):
            continue
        uid = g.get("user_id")
        if uid is not None:
            try:
                seen.add(int(uid))
            except (TypeError, ValueError):
                pass
    return sorted(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help=f"Folder with daily JSON files (default: {DATA_DIR / 'json'})",
    )
    args = parser.parse_args()

    json_dir = (args.json_dir.resolve() if args.json_dir else (DATA_DIR / "json"))
    if not json_dir.is_dir():
        print(f"Not a directory: {json_dir}", flush=True)
        return 1

    path = latest_json_path(json_dir)
    if path is None:
        print(f"No YYYY-MM-DD.json files in {json_dir}", flush=True)
        return 1

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print("JSON root must be an object with `groups`.", flush=True)
        return 1

    ids = unique_user_ids(data)
    print(f"Latest file: {path.name} ({path})", flush=True)
    print(f"Unique account IDs (user_id): {len(ids)}", flush=True)
    for uid in ids:
        print(f"  {uid}\t0x{uid:x}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
