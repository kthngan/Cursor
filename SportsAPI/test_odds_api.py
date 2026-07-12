#!/usr/bin/env python3
"""Test The Odds API for tennis in-game odds matching saved SportsAPI matches.

Limitations:
- The Odds API live endpoint returns only current/upcoming events.
- Tennis sports are seasonal and may be inactive outside tournament windows.
- Historical odds require the Historical Odds API (paid plan).
- Saved SportsAPI matches are historical (2025-2026), so most will not have
  live odds available now. This script tests the connection, fetches whatever
  tennis odds are currently available, and attempts to match by player name.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

API_KEY = os.environ.get("ODDS_API_KEY", "4d0c2c123b9edb28d71fdf10cd264de6")
BASE_URL = "https://api.the-odds-api.com/v4"

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "Data" / "SportsAPI"
METADATA_PATH = DATA_DIR / "master_match_metadata.csv"
OUTPUT_DIR = DATA_DIR / "odds_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPORTS_PATH = OUTPUT_DIR / "_tennis_sports.json"
LIVE_ODDS_PATH = OUTPUT_DIR / "_live_tennis_odds.json"
MATCHED_ODDS_PATH = OUTPUT_DIR / "matched_match_odds.csv"
REPORT_PATH = WORKSPACE_DIR / "Reports" / "odds_api_test_report.html"


def api_get(path: str, params: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    query = {"apiKey": API_KEY}
    if params:
        query.update({k: v for k, v in params.items() if v is not None})
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            body = response.read().decode()
            headers = {k.lower(): v for k, v in response.headers.items()}
            return json.loads(body), headers
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"_error": f"HTTP {exc.code} {exc.reason}", "_body": body[:1000]}, {}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}, {}


def list_tennis_sports() -> list[dict[str, Any]]:
    data, _ = api_get("/sports/", {"all": "true"})
    if isinstance(data, dict) and "_error" in data:
        print(f"Error listing sports: {data['_error']}")
        return []
    tennis = [s for s in data if "tennis" in (s.get("key", "") + s.get("title", "")).lower()]
    SPORTS_PATH.write_text(json.dumps(tennis, indent=2), encoding="utf-8")
    return tennis


def fetch_live_odds(sport_key: str) -> list[dict[str, Any]]:
    data, headers = api_get(f"/sports/{sport_key}/odds/", {
        "regions": "us,uk,eu,au",
        "markets": "h2h",
        "oddsFormat": "decimal",
    })
    remaining = headers.get("x-requests-remaining", "?")
    used = headers.get("x-requests-used", "?")
    if isinstance(data, dict) and "_error" in data:
        print(f"  {sport_key}: {data['_error']} (remaining={remaining})")
        return []
    if not isinstance(data, list):
        return []
    print(f"  {sport_key}: {len(data)} events (remaining={remaining})")
    return data


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    low = name.lower()
    # Remove common prefixes/suffixes
    for pattern in [
        r"\bmr\.?\b", r"\bdr\.?\b", r"\bjr\.?\b", r"\bsr\.?\b",
        r"\bthe\b", r"\b\(.*?\)\b",
    ]:
        low = re.sub(pattern, "", low)
    # Collapse whitespace and hyphens
    low = re.sub(r"[-_]+", " ", low)
    low = re.sub(r"[^a-z0-9 ]+", "", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def name_parts(name: str) -> set[str]:
    norm = normalize_name(name)
    if not norm:
        return set()
    parts = {p for p in norm.split() if len(p) > 1}
    # Also add full normalized string
    parts.add(norm)
    return parts


def last_name(name: str) -> str:
    """Extract the last significant word from a player name."""
    norm = normalize_name(name)
    if not norm:
        return ""
    parts = [p for p in norm.split() if len(p) > 1]
    return parts[-1] if parts else ""


def player_in_event(player_name: str, event_teams: list[str]) -> bool:
    player_last = last_name(player_name)
    if not player_last or len(player_last) < 3:
        return False
    for team in event_teams:
        team_last = last_name(team)
        if not team_last:
            continue
        # Strict last-name match
        if player_last == team_last:
            return True
        # Handle hyphenated or compound last names
        if player_last in team_last or team_last in player_last:
            return True
    return False


def load_saved_matches() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(METADATA_PATH, dtype={"event_id": str})
    df = df[df["event_sport_name"].fillna("").str.lower() == "tennis"].copy()
    return df


def match_events(
    saved: pd.DataFrame,
    odds_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for _, row in saved.iterrows():
        p1 = str(row.get("p1_name", ""))
        p2 = str(row.get("p2_name", ""))
        for event in odds_events:
            teams = [event.get("home_team", ""), event.get("away_team", "")]
            p1_match = player_in_event(p1, teams)
            p2_match = player_in_event(p2, teams)
            # Require BOTH saved players to map to the two event teams
            if not (p1_match and p2_match):
                continue
            # Extract best h2h prices per saved player across all bookmakers
            best_p1_price = None
            best_p2_price = None
            for bm in event.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price")
                        if price is None:
                            continue
                        if player_in_event(p1, [name]):
                            if best_p1_price is None or price > best_p1_price:
                                best_p1_price = price
                        elif player_in_event(p2, [name]):
                            if best_p2_price is None or price > best_p2_price:
                                best_p2_price = price
            matches.append({
                "saved_event_id": row.get("event_id", ""),
                "saved_event_name": row.get("event_name", ""),
                "saved_p1": p1,
                "saved_p2": p2,
                "saved_start": str(row.get("event_start_date", "")),
                "odds_event_id": event.get("id", ""),
                "odds_sport_key": event.get("sport_key", ""),
                "odds_commence_time": event.get("commence_time", ""),
                "odds_home_team": event.get("home_team", ""),
                "odds_away_team": event.get("away_team", ""),
                "best_p1_price": best_p1_price,
                "best_p2_price": best_p2_price,
                "implied_p1": 1.0 / best_p1_price if best_p1_price else None,
                "implied_p2": 1.0 / best_p2_price if best_p2_price else None,
                "no_vig_p1": (1.0 / best_p1_price) / (1.0 / best_p1_price + 1.0 / best_p2_price)
                    if best_p1_price and best_p2_price else None,
                "bookmaker_count": len(event.get("bookmakers", [])),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
    return matches


def build_html_report(
    tennis_sports: list[dict[str, Any]],
    active_tennis: list[dict[str, Any]],
    all_live_odds: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    saved_count: int,
) -> str:
    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
    .muted { color: #57606a; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat { font-size: 22px; font-weight: 700; margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; }
    th { background: #f6f8fa; text-align: left; }
    .note { background: #fff8e1; border: 1px solid #ffe082; border-radius: 8px; padding: 12px; }
    .warn { background: #fff3e0; border: 1px solid #ffcc02; border-radius: 8px; padding: 12px; }
    """

    sports_rows = [
        [s.get("key", ""), s.get("title", ""), str(s.get("active", False))]
        for s in tennis_sports
    ]
    active_rows = [
        [s.get("key", ""), s.get("title", "")]
        for s in active_tennis
    ]
    live_summary = [
        [e.get("sport_key", ""), e.get("id", ""), e.get("commence_time", ""),
         f"{e.get('home_team','')} vs {e.get('away_team','')}",
         str(len(e.get("bookmakers", [])))]
        for e in all_live_odds[:50]
    ]
    matched_rows = [
        [m["saved_event_id"], m["saved_event_name"], m["odds_commence_time"],
         str(m.get("best_p1_price", "")), str(m.get("best_p2_price", "")),
         f"{m['no_vig_p1']:.3f}" if m.get("no_vig_p1") else "",
         str(m.get("bookmaker_count", ""))]
        for m in matched
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>The Odds API Tennis Test Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>The Odds API Tennis Test Report</h1>
  <p class="muted">Source: The Odds API (v4). Saved SportsAPI matches: {saved_count} tennis matches.</p>

  <div class="grid">
    <div class="card"><div class="muted">Tennis sports available</div><div class="stat">{len(tennis_sports)}</div></div>
    <div class="card"><div class="muted">Active tennis sports now</div><div class="stat">{len(active_tennis)}</div></div>
    <div class="card"><div class="muted">Live tennis events found</div><div class="stat">{len(all_live_odds)}</div></div>
    <div class="card"><div class="muted">Matched saved events</div><div class="stat">{len(matched)}</div></div>
  </div>

  <div class="warn">
    <b>Important:</b> The Odds API live endpoint returns only current and upcoming events.
    Tennis sports are seasonal and may be inactive outside tournament windows.
    The saved SportsAPI matches are historical (2025-2026), so most will not have live odds available now.
    Historical odds require the Historical Odds API (paid plan).
  </div>

  <h2>Active Tennis Sports</h2>
  {"<p class='muted'>No tennis sports are currently active.</p>" if not active_tennis else "<table><thead><tr><th>Sport key</th><th>Title</th></tr></thead><tbody>" + "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>" for r in active_rows) + "</tbody></table>"}

  <h2>All Tennis Sports</h2>
  <table><thead><tr><th>Sport key</th><th>Title</th><th>Active</th></tr></thead><tbody>
  {"".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>" for r in sports_rows)}
  </tbody></table>

  <h2>Live Tennis Events Found</h2>
  {"<p class='muted'>No live tennis events returned by the API right now.</p>" if not all_live_odds else "<table><thead><tr><th>Sport</th><th>Event ID</th><th>Commence time</th><th>Match</th><th>Bookmakers</th></tr></thead><tbody>" + "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>" for r in live_summary) + "</tbody></table>"}

  <h2>Matched Saved Events</h2>
  {"<p class='muted'>No saved SportsAPI matches matched any live Odds API events.</p>" if not matched_rows else "<table><thead><tr><th>Saved event ID</th><th>Saved match</th><th>Odds commence</th><th>Best P1 price</th><th>Best P2 price</th><th>No-vig P1 prob</th><th>Books</th></tr></thead><tbody>" + "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>" for r in matched_rows) + "</tbody></table>"}

  <h2>Next Steps</h2>
  <ul>
    <li>Run this script during an active tennis tournament window to capture live in-game odds.</li>
    <li>For historical odds on saved 2025-2026 matches, use The Odds API Historical endpoint (requires paid plan).</li>
    <li>Live odds snapshots should be saved alongside SportsAPI incident rows with timestamps for calibration.</li>
  </ul>
</body>
</html>
"""


def main() -> int:
    print("=== The Odds API Tennis Test ===")
    print(f"API key: {API_KEY[:8]}...")
    print()

    print("1. Listing tennis sports...")
    tennis_sports = list_tennis_sports()
    active_tennis = [s for s in tennis_sports if s.get("active")]
    print(f"   Found {len(tennis_sports)} tennis sports, {len(active_tennis)} active")
    print()

    print("2. Fetching live odds for tennis sports...")
    all_live_odds: list[dict[str, Any]] = []
    # Only fetch active tennis sports to save API credits
    fetch_targets = active_tennis if active_tennis else tennis_sports[:5]
    for sport in fetch_targets:
        key = sport.get("key", "")
        events = fetch_live_odds(key)
        all_live_odds.extend(events)
        time.sleep(0.5)
    LIVE_ODDS_PATH.write_text(json.dumps(all_live_odds, indent=2), encoding="utf-8")
    print(f"   Total live tennis events: {len(all_live_odds)}")
    print()

    print("3. Loading saved SportsAPI matches...")
    saved = load_saved_matches()
    print(f"   Saved tennis matches: {len(saved)}")
    print()

    print("4. Matching saved matches against live odds events...")
    matched = match_events(saved, all_live_odds)
    if matched:
        df = pd.DataFrame(matched)
        df.to_csv(MATCHED_ODDS_PATH, index=False, encoding="utf-8")
        print(f"   Matched: {len(matched)}")
        print(f"   Saved: {MATCHED_ODDS_PATH}")
    else:
        print("   No matches found.")
    print()

    print("5. Building HTML report...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_html_report(tennis_sports, active_tennis, all_live_odds, matched, len(saved)),
        encoding="utf-8",
    )
    print(f"   Report: {REPORT_PATH}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
