from __future__ import annotations

import json
import re
from pathlib import Path

from models import ScheduleState

ROOT = Path(__file__).resolve().parent.parent
SCHEDULES_DIR = ROOT / "data" / "schedules"
WEEK_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _schedule_path(week_start: str) -> Path:
    if not WEEK_PATTERN.match(week_start):
        raise ValueError("week_start must be YYYY-MM-DD")
    return SCHEDULES_DIR / f"{week_start}.json"


def load_schedule(week_start: str) -> ScheduleState | None:
    path = _schedule_path(week_start)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScheduleState.model_validate(data)


def save_schedule(schedule: ScheduleState) -> str:
    if not schedule.week_start:
        raise ValueError("week_start is required")
    week_start = schedule.week_start
    _schedule_path(week_start)
    SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)
    path = _schedule_path(week_start)
    path.write_text(
        json.dumps(schedule.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return week_start
