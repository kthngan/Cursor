"""
Run downloadData.py once per calendar day for the *previous* day, after 17:00 local.

Rule: when the clock is at or after 5pm on day D, fetch YYYY-MM-DD for D-1
(e.g. 2026-04-10 17:00 -> 2026-04-09).

Ways to use:
  - Task Scheduler (recommended): trigger daily at 17:00; run:
      python scheduleDownload.py
    (no --watch; respects cutoff so accidental morning runs are skipped)
  - Long-running: python scheduleDownload.py --watch
  - Manual test: python scheduleDownload.py --force
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DATA = SCRIPT_DIR / "downloadData.py"


def target_previous_day(now: dt.datetime) -> dt.date:
    return now.date() - dt.timedelta(days=1)


def at_or_after_cutoff(now: dt.datetime, hour: int, minute: int) -> bool:
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now >= cutoff


def seconds_until_cutoff_today(now: dt.datetime, hour: int, minute: int) -> float:
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < cutoff:
        return (cutoff - now).total_seconds()
    return 0.0


def next_day_cutoff(now: dt.datetime, hour: int, minute: int) -> dt.datetime:
    d = now.date() + dt.timedelta(days=1)
    return dt.datetime.combine(d, dt.time(hour, minute))


def run_download(
    day: dt.date,
    *,
    json_dir: Path | None,
    headed: bool,
    timeout: int,
) -> int:
    cmd = [
        sys.executable,
        str(DOWNLOAD_DATA),
        "--start",
        day.isoformat(),
        "--end",
        day.isoformat(),
        "--full-range",
        "--timeout",
        str(timeout),
    ]
    if json_dir is not None:
        cmd.extend(["--json-dir", str(json_dir)])
    if headed:
        cmd.append("--headed")
    print(f"scheduleDownload: invoking downloadData for {day}", flush=True)
    return subprocess.run(cmd, cwd=str(SCRIPT_DIR)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="After 17:00 local, download previous day's trade-markout JSON via downloadData.py."
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=17,
        help="Local hour when the daily fetch is allowed (default 17 = 5pm)",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=0,
        help="Local minute for cutoff (default 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if current time is before today's cutoff (for testing)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running: wait until each day's cutoff, fetch previous day, repeat",
    )
    parser.add_argument("--json-dir", type=Path, default=None, help="Passed to downloadData.py")
    parser.add_argument("--headed", action="store_true", help="Passed to downloadData.py")
    parser.add_argument("--timeout", type=int, default=120_000, help="Passed to downloadData.py")
    args = parser.parse_args()

    if not DOWNLOAD_DATA.is_file():
        print(f"Error: missing {DOWNLOAD_DATA}", file=sys.stderr)
        return 1

    h, m = args.hour, args.minute

    while True:
        now = dt.datetime.now()
        if not at_or_after_cutoff(now, h, m):
            if args.force and not args.watch:
                # One-off test run before today's cutoff
                pass
            elif args.watch:
                wait_s = seconds_until_cutoff_today(now, h, m)
                if wait_s > 0:
                    print(f"Waiting {wait_s:.0f}s until {h:02d}:{m:02d} local...", flush=True)
                    time.sleep(wait_s)
                continue
            else:
                print(
                    f"Before cutoff ({h:02d}:{m:02d} local). Use --force to run now, or --watch to wait.",
                    file=sys.stderr,
                )
                return 1

        day = target_previous_day(dt.datetime.now())
        code = run_download(
            day,
            json_dir=args.json_dir,
            headed=args.headed,
            timeout=args.timeout,
        )
        if code != 0:
            return code

        if not args.watch:
            return 0

        now = dt.datetime.now()
        nxt = next_day_cutoff(now, h, m)
        sleep_s = (nxt - now).total_seconds()
        if sleep_s > 0:
            print(f"Next run at {nxt} local (in {sleep_s:.0f}s)...", flush=True)
            time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
