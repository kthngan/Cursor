#!/usr/bin/env python3
"""Helpers for deriving tennis live-form metrics from StatScore incidents."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque


POINT_INCIDENTS = {"0", "15", "30", "40", "A"}
SERVICE_INCIDENTS = {
    "First service",
    "First service in",
    "Second service",
    "Service fault",
    "Net - first service",
    "Net - second service",
    "Service ace",
    "Double fault",
}


def unix_to_utc(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(value)


def incident_sort_key(incident: dict[str, Any]) -> tuple[int, str]:
    try:
        updated_at = int(incident.get("ut") or 0)
    except (TypeError, ValueError):
        updated_at = 0
    return updated_at, str(incident.get("id", ""))


def safe_ratio(p1_value: float, p2_value: float) -> str:
    total = p1_value + p2_value
    if total <= 0:
        return ""
    return f"{p1_value / total:.6f}"


def numeric_ratio(p1_value: float, p2_value: float) -> float | None:
    total = p1_value + p2_value
    if total <= 0:
        return None
    return p1_value / total


def side_for_counter(counter: int | None) -> str:
    if counter == 1:
        return "P1"
    if counter == 2:
        return "P2"
    return ""


def determine_winner(event: dict[str, Any]) -> dict[str, Any]:
    for participant in event.get("participants", []):
        for result in participant.get("results", []):
            if result.get("name") == "Winner" and str(result.get("value")) == "1":
                return {
                    "winner_id": participant.get("id", ""),
                    "winner_name": participant.get("name", ""),
                    "winner_side": side_for_counter(int(participant.get("counter") or 0)),
                }
    return {"winner_id": "", "winner_name": "", "winner_side": ""}


def _is_receiver_break_point(server_counter: int | None, p1_points: int, p2_points: int) -> int | None:
    if server_counter not in (1, 2):
        return None
    receiver = 2 if server_counter == 1 else 1
    receiver_points = p2_points if receiver == 2 else p1_points
    server_points = p1_points if server_counter == 1 else p2_points
    if receiver_points >= 3 and receiver_points > server_points:
        return receiver
    return None


def _advance_point_score(winner_counter: int, p1_points: int, p2_points: int) -> tuple[int, int]:
    winner_points = p1_points if winner_counter == 1 else p2_points
    loser_points = p2_points if winner_counter == 1 else p1_points

    if winner_points <= 2:
        winner_points += 1
    elif winner_points == 3 and loser_points == 4:
        winner_points = 3
        loser_points = 3
    elif winner_points == 3 and loser_points == 3:
        winner_points = 4

    if winner_counter == 1:
        return winner_points, loser_points
    return loser_points, winner_points


def _count_by_side(records: Deque[dict[str, Any]], key: str) -> tuple[int, int]:
    p1_count = sum(1 for record in records if record.get(key) == 1)
    p2_count = sum(1 for record in records if record.get(key) == 2)
    return p1_count, p2_count


def _metric_row_from_windows(
    point_window: Deque[dict[str, Any]],
    game_window: Deque[int],
) -> dict[str, str]:
    p1_points, p2_points = _count_by_side(point_window, "winner")
    p1_service, p2_service = _count_by_side(point_window, "service_point_winner")
    p1_return, p2_return = _count_by_side(point_window, "return_point_winner")
    p1_bp_created, p2_bp_created = _count_by_side(point_window, "break_point_for")
    p1_bp_won, p2_bp_won = _count_by_side(point_window, "break_point_winner")
    p1_bp_saved, p2_bp_saved = _count_by_side(point_window, "break_point_saved_by")
    p1_games = sum(1 for winner in game_window if winner == 1)
    p2_games = sum(1 for winner in game_window if winner == 2)

    component_values = [
        (numeric_ratio(p1_points, p2_points), 0.35),
        (numeric_ratio(p1_return, p2_return), 0.20),
        (numeric_ratio(p1_service, p2_service), 0.15),
        (numeric_ratio(p1_bp_created, p2_bp_created), 0.15),
        (numeric_ratio(p1_bp_saved, p2_bp_saved), 0.10),
        (numeric_ratio(p1_games, p2_games), 0.05),
    ]
    weighted_sum = sum(value * weight for value, weight in component_values if value is not None)
    weight_sum = sum(weight for value, weight in component_values if value is not None)
    live_form = "" if weight_sum == 0 else f"{weighted_sum / weight_sum:.6f}"

    return {
        "rolling_points_ratio_20": safe_ratio(p1_points, p2_points),
        "rolling_service_points_won_ratio_20": safe_ratio(p1_service, p2_service),
        "rolling_return_points_won_ratio_20": safe_ratio(p1_return, p2_return),
        "rolling_break_points_created_ratio_20": safe_ratio(p1_bp_created, p2_bp_created),
        "rolling_break_points_won_ratio_20": safe_ratio(p1_bp_won, p2_bp_won),
        "rolling_break_points_saved_ratio_20": safe_ratio(p1_bp_saved, p2_bp_saved),
        "rolling_games_won_ratio_6": safe_ratio(p1_games, p2_games),
        "rolling_live_form_ratio": live_form,
    }


def compute_rolling_metric_rows(event: dict[str, Any], point_window_size: int = 20, game_window_size: int = 6) -> list[dict[str, Any]]:
    participants = event.get("participants", [])
    counter_by_participant_id = {
        participant.get("id"): int(participant.get("counter") or 0)
        for participant in participants
        if participant.get("counter") in (1, 2, "1", "2")
    }
    name_by_counter = {
        int(participant.get("counter") or 0): participant.get("name", "")
        for participant in participants
        if participant.get("counter") in (1, 2, "1", "2")
    }

    current_server: int | None = None
    current_set_games = {1: 0, 2: 0}
    sets_won = {1: 0, 2: 0}
    point_score = {1: 0, 2: 0}
    point_window: Deque[dict[str, Any]] = deque(maxlen=point_window_size)
    game_window: Deque[int] = deque(maxlen=game_window_size)
    rows: list[dict[str, Any]] = []

    winner_info = determine_winner(event)
    incidents = sorted(event.get("events_incidents", []), key=incident_sort_key)

    for sequence, incident in enumerate(incidents, start=1):
        incident_name = incident.get("incident_name") or ""
        participant_id = incident.get("participant_id")
        participant_counter = counter_by_participant_id.get(participant_id)
        point_winner: int | None = None
        game_winner: int | None = None

        if incident_name in SERVICE_INCIDENTS and participant_counter in (1, 2):
            current_server = participant_counter

        break_point_for = _is_receiver_break_point(current_server, point_score[1], point_score[2])

        if incident_name in POINT_INCIDENTS and participant_counter in (1, 2):
            point_winner = participant_counter
            point_score[1], point_score[2] = _advance_point_score(point_winner, point_score[1], point_score[2])
        elif incident_name == "Game Won" and participant_counter in (1, 2):
            point_winner = participant_counter
            game_winner = participant_counter
            current_set_games[participant_counter] += 1
            game_window.append(participant_counter)
            point_score = {1: 0, 2: 0}
        elif incident_name == "Set won" and participant_counter in (1, 2):
            sets_won[participant_counter] += 1
        elif "set started" in incident_name:
            current_set_games = {1: 0, 2: 0}
            point_score = {1: 0, 2: 0}
            current_server = None

        if point_winner in (1, 2):
            record: dict[str, Any] = {"winner": point_winner}
            if current_server in (1, 2):
                if point_winner == current_server:
                    record["service_point_winner"] = point_winner
                else:
                    record["return_point_winner"] = point_winner
            if break_point_for in (1, 2):
                record["break_point_for"] = break_point_for
                if game_winner == break_point_for:
                    record["break_point_winner"] = break_point_for
                elif point_winner == current_server:
                    record["break_point_saved_by"] = current_server
            point_window.append(record)

        metric_values = _metric_row_from_windows(point_window, game_window)
        rows.append(
            {
                "seq": sequence,
                "event_id": event.get("id", ""),
                "event_name": event.get("name", ""),
                "p1_name": name_by_counter.get(1, ""),
                "p2_name": name_by_counter.get(2, ""),
                "ut": incident.get("ut", ""),
                "utc_time": unix_to_utc(incident.get("ut")),
                "event_status_id": incident.get("event_status_id", ""),
                "event_status_name": incident.get("event_status_name", ""),
                "event_time": incident.get("event_time", ""),
                "incident_id": incident.get("incident_id", ""),
                "incident_name": incident_name,
                "participant_side": side_for_counter(participant_counter),
                "participant_name": incident.get("participant_name", ""),
                "server_side": side_for_counter(current_server),
                "point_winner_side": side_for_counter(point_winner),
                "game_winner_side": side_for_counter(game_winner),
                "game_score_after": f"{current_set_games[1]}-{current_set_games[2]}",
                "sets_after": f"{sets_won[1]}-{sets_won[2]}",
                "point_score_state": f"{point_score[1]}-{point_score[2]}",
                **metric_values,
                "match_winner_id": "",
                "match_winner_name": "",
                "match_winner_side": "",
            }
        )

    if rows:
        rows[-1]["match_winner_id"] = winner_info["winner_id"]
        rows[-1]["match_winner_name"] = winner_info["winner_name"]
        rows[-1]["match_winner_side"] = winner_info["winner_side"]

    return rows
