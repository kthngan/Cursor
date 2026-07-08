"""
End-to-end delta report pipeline: open the web UI, download JSON, then
matplotlib plot + CSV and the interactive HTML dashboard.

Requires: playwright, matplotlib
  pip install playwright matplotlib
  python -m playwright install chromium

Examples:
  python pnlTimeSeries.py --start 2026-03-01T00:00 --end 2026-04-05T23:59 --outdir .
    Fetches in batches (--batch-days, default 5), merges in memory, no cache JSON.

  python pnlTimeSeries.py --input-json delta_report_....json --outdir .
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

DEFAULT_REPORT_URL = "https://poly-pnl.it9.win/delta-report-v3"
PUBLIC_REPORT_URL = "https://poly-pnl.it9.win/trade-markout"
DEFAULT_HTTP_AUTH_USERNAME = "mm"
DEFAULT_HTTP_AUTH_PASSWORD = "2047"
# Trade-markout "algo0" wallet alias in JSON payloads.
PUBLIC_ALGO0_USER_ID = 1771
# Filename excluded when picking newest `delta_report*.json` (legacy full export name).
PUBLIC_FULL_JSON_NAME = "delta_report_public_full.json"
# Wallet hex prefix (lowercase, with 0x) -> display name (six accounts incl. algo0).
WALLET_PREFIX_ACCOUNT_NAMES: tuple[tuple[str, str], ...] = (
    ("0x507e", "NXY"),
    ("0x2005", "RN1"),
    ("0x204f", "swissTony"),
    ("0xee61", "Sovereign2013"),
    ("0xa6a8", "lhtSports"),
    ("0x2652", "algo0"),
)
# HTML dashboard (--use-public): one full chart block per named account (same layout each).
DASHBOARD_PUBLIC_VARIANT_ACCOUNTS: tuple[str, ...] = ("algo0", "NXY", "RN1")
SPORT_ID_TO_NAME = {
    "1": "Baseball",
    "2": "Tennis",
    "3": "Basketball",
    "4": "Esports",
    "5": "American Football",
    "6": "Soccer",
    "7": "Hockey",
    "8": "MMA",
}


# --- JSON discovery (for --input-json / wrappers) ---


def find_latest_report_json(folder: Path) -> Path:
    """Newest `delta_report*.json` except the unfiltered public cache (for wrappers)."""
    candidates = [
        p
        for p in folder.glob("delta_report*.json")
        if p.name != PUBLIC_FULL_JSON_NAME
    ]
    files = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No delta_report*.json in {folder}")
    return files[0]


# --- Browser: fetch report JSON ---


def run_report_and_download_json(
    username: str,
    password: str,
    start_dt: str,
    end_dt: str,
    outdir: Path,
    report_url: str = DEFAULT_REPORT_URL,
    group_by_labels: list[str] | None = None,
    wallets_filter: str = "",
    headless: bool = True,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            http_credentials={"username": username, "password": password},
            accept_downloads=True,
        )
        page = context.new_page()
        page.goto(report_url, wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(1500)

        date_inputs = page.locator("input[type='datetime-local']")
        if date_inputs.count() < 2:
            raise RuntimeError("Could not find start/end datetime inputs.")
        date_inputs.nth(0).fill(start_dt)
        date_inputs.nth(1).fill(end_dt)

        # Only enable requested group-by dimensions.
        # First clear all checkboxes, then explicitly turn on the ones we care about.
        checkboxes = page.locator("input[type='checkbox']")
        for idx in range(checkboxes.count()):
            checkbox = checkboxes.nth(idx)
            if checkbox.is_checked():
                checkbox.uncheck(force=True)

        labels = group_by_labels or ["Sport", "Date", "Role", "TIF", "P/I"]
        for label in labels:
            checked = False
            try:
                control = page.get_by_label(label, exact=False)
                if control.count() > 0:
                    control.first.check(force=True)
                    checked = True
            except Exception:
                checked = False
            if checked:
                continue
            fallback = page.locator(
                f"label:has-text('{label}') input[type='checkbox'], "
                f"label:has-text('{label}') + input[type='checkbox']"
            )
            if fallback.count() > 0:
                fallback.first.check(force=True)
                checked = True
            if not checked:
                try:
                    clickable = page.get_by_text(label, exact=False)
                    if clickable.count() > 0:
                        clickable.first.click(force=True)
                        checked = True
                except Exception:
                    checked = False
            if not checked:
                try:
                    chip = page.get_by_role("button", name=label, exact=False)
                    if chip.count() > 0:
                        chip.first.click(force=True)
                        checked = True
                except Exception:
                    checked = False
            if not checked:
                print(f"Warning: could not enable group-by control: {label}")

        if wallets_filter:
            wallet_candidates = [
                page.get_by_placeholder("address...", exact=False),
                page.get_by_label("Wallets", exact=False),
                page.get_by_placeholder("Wallets", exact=False),
                page.locator("input[name*='wallet']"),
                page.locator("input[id*='wallet']"),
                page.locator("input[type='text']"),
            ]
            wallet_set = False
            for candidate in wallet_candidates:
                try:
                    if candidate.count() > 0:
                        field = candidate.first
                        field.fill(wallets_filter)
                        field.press("Enter")
                        wallet_set = True
                        break
                except Exception:
                    continue
            if not wallet_set:
                raise RuntimeError(
                    "Could not set wallets filter. Expected to set Wallets = 'algo 0'."
                )

        run_button = None
        for name in ("Run Report", "FETCH"):
            b = page.get_by_role("button", name=name, exact=False)
            if b.count() > 0:
                run_button = b.first
                break
        if run_button is None:
            raise RuntimeError("Could not find report execution button (Run Report/FETCH).")
        run_button.click()
        # Large date ranges can exceed a few minutes; wait up to 20 minutes for rows.
        page.wait_for_selector("table tbody tr", timeout=1_200_000)
        page.wait_for_timeout(2000)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_json = outdir / f"delta_report_{ts}.json"

        with page.expect_download(timeout=600_000) as download_info:
            download_button = None
            for name in ("Download JSON", "DOWNLOAD JSON"):
                b = page.get_by_role("button", name=name, exact=False)
                if b.count() > 0:
                    download_button = b.first
                    break
            if download_button is None:
                raise RuntimeError(
                    "Could not find JSON download button (Download JSON/DOWNLOAD JSON)."
                )
            download_button.click()
        download = download_info.value
        download.save_as(str(out_json))

        browser.close()
        return out_json


def _parse_cli_datetime_boundary(s: str, *, is_end: bool) -> datetime:
    """Parse ``--start`` / ``--end``: date-only end defaults to 23:59 that day."""
    s = s.strip()
    if "T" not in s:
        d = datetime.strptime(s[:10], "%Y-%m-%d")
        if is_end:
            return d.replace(hour=23, minute=59, second=0, microsecond=0)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)
    body = s.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            ln = 19 if fmt.endswith("%S") else 16
            return datetime.strptime(body[:ln], fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime: {s!r}")


def split_report_batches(start_arg: str, end_arg: str, max_days: int) -> list[tuple[str, str]]:
    """Split [start, end] into windows of at most ``max_days`` inclusive calendar days.

    Returns ``(start_dt, end_dt)`` strings for ``datetime-local`` inputs on the page.
    """
    if max_days < 1:
        raise ValueError("max_days must be >= 1")
    ts = _parse_cli_datetime_boundary(start_arg, is_end=False)
    te = _parse_cli_datetime_boundary(end_arg, is_end=True)
    if te < ts:
        raise ValueError("end is before start")
    batches: list[tuple[str, str]] = []
    cur_d = ts.date()
    end_d = te.date()
    while cur_d <= end_d:
        chunk_end_d = min(cur_d + timedelta(days=max_days - 1), end_d)
        st_comb = datetime.combine(
            cur_d,
            ts.time() if cur_d == ts.date() else time.min,
        )
        en_comb = datetime.combine(
            chunk_end_d,
            te.time() if chunk_end_d == te.date() else time(23, 59),
        )
        batches.append(
            (
                st_comb.strftime("%Y-%m-%dT%H:%M"),
                en_comb.strftime("%Y-%m-%dT%H:%M"),
            )
        )
        cur_d = chunk_end_d + timedelta(days=1)
    return batches


def download_merged_report_batches(
    *,
    start_arg: str,
    end_arg: str,
    batch_days: int,
    username: str,
    password: str,
    outdir: Path,
    use_public: bool,
    report_url: str,
    headful: bool,
) -> dict:
    """Fetch each batch from the UI, merge rows, delete per-batch files. Returns merged JSON root."""
    batches = split_report_batches(start_arg, end_arg, batch_days)
    merged: dict | None = None
    for i, (st, en) in enumerate(batches, start=1):
        print(f"Download batch {i}/{len(batches)}: {st} .. {en}", flush=True)
        path = run_report_and_download_json(
            username=username,
            password=password,
            start_dt=st,
            end_dt=en,
            outdir=outdir,
            report_url=(PUBLIC_REPORT_URL if use_public else report_url),
            group_by_labels=(
                ["Sport", "mkt Type", "role", "stage", "price bucket", "date bucket"]
                if use_public
                else ["Sport", "Date", "Role", "TIF", "P/I"]
            ),
            wallets_filter=("algo 0" if use_public else ""),
            headless=not headful,
        )
        raw_pl = json.loads(path.read_text(encoding="utf-8"))
        reps = raw_pl.get("reports") or []
        if not isinstance(reps, list):
            reps = []
        if not reps:
            reps = groups_to_reports(raw_pl.get("groups") or [])
        if not reps:
            raise ValueError(f"Batch {st} .. {en} returned no report rows.")
        merged = merge_report_payload(merged, raw_pl, reps)
        try:
            path.unlink()
        except OSError:
            pass
    assert merged is not None
    return merged


# --- Matplotlib + CSV (overall cumulative settle PnL) ---


def _report_pnl_value(report: dict) -> float | None:
    """Prefer API PNL column; then markout total_actual_pnl; then settle timepoint pnl."""
    g = report.get("group") or {}
    for obj in (report, g):
        for key in ("PNL", "pnl"):
            if key not in obj:
                continue
            v = obj[key]
            if v is None:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    for obj in (report, g):
        if "total_actual_pnl" not in obj or obj["total_actual_pnl"] is None:
            continue
        try:
            return float(obj["total_actual_pnl"])
        except (TypeError, ValueError):
            continue
    settle = (
        (report.get("implied_prob") or {})
        .get("timepoints", {})
        .get("settle")
        or {}
    )
    v = settle.get("pnl")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def _report_turnover_value(report: dict) -> float:
    g = report.get("group") or {}
    for obj in (report, g):
        for key in ("total_notional", "total_risk"):
            if key not in obj or obj[key] is None:
                continue
            try:
                return float(obj[key])
            except (TypeError, ValueError):
                continue
    settle = (
        (report.get("implied_prob") or {})
        .get("timepoints", {})
        .get("settle")
        or {}
    )
    try:
        return float(settle.get("total_notional") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def report_pnl_turnover(report: dict) -> tuple[float, float]:
    pnl = _report_pnl_value(report)
    if pnl is None:
        pnl = 0.0
    return pnl, _report_turnover_value(report)


def groups_to_reports(groups: list) -> list[dict]:
    """Trade-markout JSON often has aggregated rows under `groups` instead of `reports`."""
    out: list[dict] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        grp = dict(g)
        if grp.get("trade_role") is None and "is_maker" in g:
            grp["trade_role"] = "maker" if g.get("is_maker") else "taker"
        out.append({"group": grp})
    return out


def _group_user_id(group: dict) -> int | None:
    v = group.get("user_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def filter_public_algo0_user(payload: dict) -> dict:
    """Keep only algo0 (user_id PUBLIC_ALGO0_USER_ID) in public / trade-markout payloads."""
    uid = PUBLIC_ALGO0_USER_ID
    out = dict(payload)
    if isinstance(out.get("groups"), list):
        out["groups"] = [
            g
            for g in out["groups"]
            if isinstance(g, dict) and _group_user_id(g) == uid
        ]
    if isinstance(out.get("reports"), list):
        fr: list[dict] = []
        for r in out["reports"]:
            if not isinstance(r, dict):
                continue
            g = r.get("group") or {}
            if _group_user_id(g) == uid:
                fr.append(r)
        out["reports"] = fr
    users = out.get("users")
    if isinstance(users, dict):
        key = str(uid)
        out["users"] = {k: v for k, v in users.items() if str(k) == key}
    return out


def report_row_group_key(r: dict) -> tuple:
    """Stable identity for a report row: sorted ``group`` entries (fine-grained)."""
    g = r.get("group") or {}
    return tuple(sorted((g or {}).items()))


def _is_trade_markout_style_group(g: dict) -> bool:
    """Trade-markout exports repeat the same PnL row with different markout time columns in ``group``."""
    return "avg_markout_edge_0s" in g or "total_markout_pnl_0s" in g


def report_row_dedupe_key(r: dict) -> tuple:
    """Key for collapsing duplicates before summing PnL.

    Markout payloads encode identical ``total_actual_pnl`` slices many times (varying only
    per-time markout fields). Those must share one key or totals are inflated ~3–5×.
    Non-markout rows keep the full sorted-``group`` key.
    """
    g = r.get("group") or {}
    if _is_trade_markout_style_group(g):
        return (
            "tm",
            g.get("user_id"),
            str(g.get("date_bucket") or g.get("date") or "").strip(),
            str(g.get("unified_sport_id") or ""),
            str(g.get("unified_market_id") or ""),
            str(g.get("trade_role") or ""),
            str(g.get("price_bucket") or ""),
        )
    return ("full", report_row_group_key(r))


def dedupe_report_rows(reports: list[dict]) -> list[dict]:
    """Drop duplicate economics rows; last occurrence wins (same key as merge)."""
    by_key: dict[tuple, dict] = {}
    for r in reports:
        if isinstance(r, dict):
            by_key[report_row_dedupe_key(r)] = r
    return list(by_key.values())


def merge_report_payload(
    existing_payload: dict | None,
    fresh_pl: dict,
    fresh_reports: list[dict],
) -> dict:
    """Merge JSON roots by report rows (dedupe on economics key); union users maps."""
    if not existing_payload:
        root = dict(fresh_pl)
        root["reports"] = dedupe_report_rows(
            [r for r in fresh_reports if isinstance(r, dict)]
        )
        return root
    root = dict(existing_payload)
    merged_reports = list(existing_payload.get("reports") or [])
    if not merged_reports:
        merged_reports = groups_to_reports(existing_payload.get("groups") or [])
    by_key: dict[tuple, dict] = {}

    for r in merged_reports:
        if isinstance(r, dict):
            by_key[report_row_dedupe_key(r)] = r
    for r in fresh_reports:
        if isinstance(r, dict):
            by_key[report_row_dedupe_key(r)] = r
    root["reports"] = list(by_key.values())
    for k in ("summary", "group_by", "success"):
        if k in fresh_pl:
            root[k] = fresh_pl[k]
    eu, fu = existing_payload.get("users") or {}, fresh_pl.get("users") or {}
    if isinstance(eu, dict) and isinstance(fu, dict):
        u = dict(eu)
        u.update(fu)
        root["users"] = u
    elif isinstance(fu, dict):
        root["users"] = fu
    return root


def iter_payload_report_rows(payload: dict) -> list[dict]:
    """Flatten ``reports`` or ``groups`` into row dicts, deduped (see ``report_row_dedupe_key``)."""
    reps = payload.get("reports")
    if isinstance(reps, list) and reps:
        return dedupe_report_rows([r for r in reps if isinstance(r, dict)])
    groups = payload.get("groups")
    if isinstance(groups, list) and groups:
        return dedupe_report_rows(groups_to_reports(groups))
    return []


def extract_daily_settle_pnl(payload: dict) -> list[tuple[datetime, float]]:
    daily_totals: dict[datetime, float] = {}

    for report in iter_payload_report_rows(payload):
        date_str = report_date(report)
        row_pnl = _report_pnl_value(report)
        if date_str is None or row_pnl is None:
            continue
        try:
            day = datetime.strptime(date_str, "%Y-%m-%d")
            daily_totals[day] = daily_totals.get(day, 0.0) + row_pnl
        except (ValueError, TypeError):
            continue

    return sorted(daily_totals.items(), key=lambda x: x[0])


def write_csv_and_plot(
    daily_values: list[tuple[datetime, float]], outdir: Path
) -> tuple[Path, Path]:
    if not daily_values:
        raise ValueError("No valid date + PnL rows found in JSON payload.")

    csv_path = outdir / "daily_settle_pnl_summary.csv"
    png_path = outdir / "cumulative_settle_pnl_timeseries.png"

    rows = ["date,pnl,cumulative_pnl"]
    dates: list[str] = []
    cumulative: list[float] = []
    running = 0.0

    for day, pnl in daily_values:
        running += pnl
        day_str = day.strftime("%Y-%m-%d")
        rows.append(f"{day_str},{pnl},{running}")
        dates.append(day_str)
        cumulative.append(running)

    csv_path.write_text("\n".join(rows), encoding="utf-8")

    plt.figure(figsize=(10, 5))
    plt.plot(dates, cumulative, marker="o")
    plt.title("Cumulative PnL by Date")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return csv_path, png_path


def _normalize_users_map(users_map: dict | None) -> dict[str, object]:
    if not users_map or not isinstance(users_map, dict):
        return {}
    return {str(k): v for k, v in users_map.items()}


# --- HTML dashboard helpers ---


def parse_bucket_range(name: str) -> tuple[float, float] | None:
    m = re.match(r"^([\d.]+)-([\d.]+)$", name.strip())
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    # Markout-style "50-60" means 0.50–0.60 implied probability.
    if lo > 1.0 or hi > 1.0:
        lo, hi = lo / 100.0, hi / 100.0
    return lo, hi


def prob_bucket_from_range(low: float, high: float) -> str:
    if high < 0.4:
        return "Low"
    if low > 0.6:
        return "High"
    return "Med"


def prob_bucket_from_range_name(bucket_name: str) -> str | None:
    r = parse_bucket_range(bucket_name)
    if not r:
        return None
    return prob_bucket_from_range(r[0], r[1])


def report_date(report: dict) -> str | None:
    group = report.get("group") or {}
    raw = group.get("date") or group.get("date_bucket")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return s.split("T", 1)[0]


def request_date_bounds(start_arg: str, end_arg: str) -> tuple[str, str]:
    return start_arg.split("T", 1)[0], end_arg.split("T", 1)[0]


def clip_reports_to_request_range(
    reports: list[dict], start_arg: str, end_arg: str
) -> list[dict]:
    """Keep rows whose report date is between start/end date (inclusive), YYYY-MM-DD."""
    d0, d1 = request_date_bounds(start_arg, end_arg)
    out: list[dict] = []
    for r in reports:
        d = report_date(r)
        if d and d0 <= d <= d1:
            out.append(r)
    return out


def filter_reports_by_account_display_name(
    reports: list[dict], users_map: dict | None, name: str
) -> list[dict]:
    um = _normalize_users_map(users_map)
    out: list[dict] = []
    for r in reports:
        g = r.get("group") or {}
        if account_display_name(g, um if um else None) == name:
            out.append(r)
    return out


def report_prob_bucket(report: dict) -> str:
    group = report.get("group") or {}
    raw = group.get("price_bucket")
    if raw:
        raw_s = str(raw).strip()
        parsed = prob_bucket_from_range_name(raw_s)
        if parsed:
            return parsed
        norm = raw_s.lower()
        if norm in {"low", "med", "high"}:
            return norm.capitalize()
    return row_prob_bucket_from_price_buckets(report)


def price_bucket_pnl(m: dict) -> float:
    for key in ("PNL", "pnl", "settle_pnl"):
        if key not in m or m[key] is None:
            continue
        try:
            return float(m[key])
        except (TypeError, ValueError):
            continue
    return 0.0


def row_prob_bucket_from_price_buckets(report: dict) -> str:
    pb = (report.get("implied_prob") or {}).get("price_buckets") or {}
    if not isinstance(pb, dict):
        return "Unknown"
    weights: dict[str, float] = defaultdict(float)
    for name, m in pb.items():
        if not isinstance(m, dict):
            continue
        r = parse_bucket_range(name)
        if not r:
            continue
        try:
            u = float(m.get("usd") or 0.0)
        except (TypeError, ValueError):
            continue
        if u <= 0:
            continue
        cat = prob_bucket_from_range(r[0], r[1])
        weights[cat] += u
    if not weights:
        return "Unknown"
    return max(("Low", "Med", "High"), key=lambda k: (weights.get(k, 0.0), k))


def cumulative_by_key(
    daily: dict[tuple[str, str], float],
) -> tuple[list[str], dict[str, list[float]]]:
    dates = sorted({d for (d, _) in daily.keys()})
    keys = sorted({k for (_, k) in daily.keys()})
    raw: dict[str, dict[str, float]] = {k: {} for k in keys}
    for (d, k), v in daily.items():
        raw[k][d] = raw[k].get(d, 0.0) + v
    series: dict[str, list[float]] = {}
    for k in keys:
        run = 0.0
        out: list[float] = []
        for d in dates:
            run += raw[k].get(d, 0.0)
            out.append(round(run, 6))
        series[k] = out
    return dates, series


def cumulative_single(daily: dict[str, float]) -> tuple[list[str], list[float]]:
    dates = sorted(daily.keys())
    out: list[float] = []
    run = 0.0
    for d in dates:
        run += daily[d]
        out.append(round(run, 6))
    return dates, out


def dashboard_summary_row(pnl: float, turnover: float) -> dict[str, float | str | None]:
    rot = (pnl / turnover) if turnover else None
    return {
        "pnl": round(pnl, 6),
        "turnover_usd": round(turnover, 6),
        "return_on_turnover": round(rot, 8) if rot is not None else None,
    }


def dashboard_dim_block(
    daily_by_date_dim: dict[tuple[str, str], float],
    dim_pnl: dict[str, float],
    dim_tover: dict[str, float],
) -> dict:
    """Chart.js block: cumulative series per key + summary rows (stage/sport/bucket/account)."""
    dates, series = cumulative_by_key(dict(daily_by_date_dim))
    dp = dict(dim_pnl)
    dt = dict(dim_tover)
    summ = []
    for k in sorted(dp.keys(), key=lambda x: (-dp[x], x)):
        summ.append(
            {
                "key": k,
                **dashboard_summary_row(dp[k], float(dt.get(k, 0.0))),
            }
        )
    return {"dates": dates, "series": series, "summary": summ}


def map_sport_name(unified_sport_id: str) -> str:
    return SPORT_ID_TO_NAME.get(unified_sport_id, unified_sport_id)


def sport_chart_id(sport: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", sport)[:80].strip("_") or "sport"


def _normalize_hex_addr(addr: str) -> str:
    a = addr.strip().lower()
    if not a.startswith("0x"):
        a = "0x" + a
    return a


def wallet_address_for_group(
    group: dict, users_map: dict[str, object] | None
) -> str | None:
    if not users_map:
        return None
    uid = group.get("user_id")
    if uid is None:
        return None
    try:
        ukey = str(int(uid))
    except (TypeError, ValueError):
        ukey = str(uid).strip()
    addr = users_map.get(ukey)
    if addr is None and isinstance(uid, int):
        addr = users_map.get(str(uid))
    if isinstance(addr, str) and addr.strip():
        return _normalize_hex_addr(addr)
    return None


def account_display_name(group: dict, users_map: dict | None) -> str:
    um = _normalize_users_map(users_map)
    addr = wallet_address_for_group(group, um if um else None)
    if addr:
        for prefix, name in WALLET_PREFIX_ACCOUNT_NAMES:
            if addr.startswith(prefix.lower()):
                return name
        if len(addr) > 14:
            return f"{addr[:8]}…{addr[-6:]}"
        return addr
    return account_label(group, users_map)


def account_label(group: dict, users_map: dict | None) -> str:
    """Human-readable account key: wallet short or user_id."""
    uid = group.get("user_id")
    if uid is None:
        return "Unknown"
    try:
        ukey = str(int(uid))
    except (TypeError, ValueError):
        ukey = str(uid).strip()
    if users_map:
        addr = users_map.get(ukey)
        if addr is None:
            addr = users_map.get(int(uid)) if isinstance(uid, int) else None
        if isinstance(addr, str) and addr.startswith("0x") and len(addr) > 14:
            return f"{addr[:8]}…{addr[-6:]}"
        if addr is not None:
            return str(addr)
    return f"user_{ukey}"


def sport_name_from_report(report: dict) -> str:
    g = report.get("group") or {}
    return map_sport_name(str(g.get("unified_sport_id") or "Unknown"))


def build_by_account_block(
    acct_reports: list[dict], users_map: dict | None
) -> dict:
    """Same shape as dim_section: dates, series, summary — for Chart.js multiSection."""
    dm: dict[tuple[str, str], float] = defaultdict(float)
    dt: dict[str, float] = defaultdict(float)
    dp: dict[str, float] = defaultdict(float)
    um = _normalize_users_map(users_map)
    for report in acct_reports:
        date = report_date(report)
        if not date:
            continue
        pnl, turnover = report_pnl_turnover(report)
        g = report.get("group") or {}
        key = account_display_name(g, um if um else None)
        if pnl or turnover:
            dm[(date, key)] += pnl
        dt[key] += turnover
        dp[key] += pnl
    return dashboard_dim_block(dict(dm), dict(dp), dict(dt))


def build_dashboard_payload(
    reports: list[dict],
    *,
    account_charts_reports: list[dict] | None = None,
    users_map_accounts: dict | None = None,
    account_sport_chart_suffix: str = "",
    include_account_by_sport_charts: bool = True,
) -> dict:
    daily_overall: dict[str, float] = defaultdict(float)
    tover_overall = 0.0
    pnl_overall = 0.0

    dim_maps: dict[str, dict[tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    dim_tover: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    dim_pnl: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    daily_bucket: dict[tuple[str, str], float] = defaultdict(float)
    bucket_tover: dict[str, float] = defaultdict(float)
    bucket_pnl: dict[str, float] = defaultdict(float)

    for report in reports:
        group = report.get("group") or {}
        date = report_date(report)
        if not date:
            continue

        pnl, turnover = report_pnl_turnover(report)
        pnl_overall += pnl
        tover_overall += turnover
        if pnl or turnover:
            daily_overall[date] += pnl

        stage = str(group.get("stage") or group.get("pregame_inplay") or "Unknown")
        sport_id = str(group.get("unified_sport_id") or "Unknown")
        sport = map_sport_name(sport_id)
        role = str(group.get("trade_role") or "Unknown")
        mkt_type = str(
            group.get("mkt_type") or group.get("market_type") or group.get("order_tif") or "Unknown"
        )

        for name, dm, key in (
            ("stage", dim_maps["stage"], stage),
            ("sport", dim_maps["sport"], sport),
            ("trade_role", dim_maps["trade_role"], role),
            ("mkt_type", dim_maps["mkt_type"], mkt_type),
        ):
            if pnl or turnover:
                dm[(date, key)] += pnl
            dim_tover[name][key] += turnover
            dim_pnl[name][key] += pnl

        pb = (report.get("implied_prob") or {}).get("price_buckets") or {}
        has_price_buckets = isinstance(pb, dict) and len(pb) > 0
        if has_price_buckets:
            for bucket_name, m in pb.items():
                if not isinstance(m, dict):
                    continue
                cat = prob_bucket_from_range_name(bucket_name)
                if not cat:
                    continue
                try:
                    sp = price_bucket_pnl(m)
                    usd = float(m.get("usd") or 0.0)
                except (TypeError, ValueError):
                    continue
                daily_bucket[(date, cat)] += sp
                bucket_tover[cat] += usd
                bucket_pnl[cat] += sp

        # Trade-markout rows have no price_buckets; use group price range / prob bucket (no double-count).
        if not has_price_buckets:
            cat = report_prob_bucket(report)
            daily_bucket[(date, cat)] += pnl
            bucket_tover[cat] += turnover
            bucket_pnl[cat] += pnl

    o_dates, o_cum = cumulative_single(dict(daily_overall))

    def dim_section(name: str) -> dict:
        return dashboard_dim_block(
            dict(dim_maps[name]),
            dict(dim_pnl[name]),
            dict(dim_tover[name]),
        )

    by_prob_bucket = dashboard_dim_block(
        dict(daily_bucket),
        dict(bucket_pnl),
        dict(bucket_tover),
    )

    acct_src = (
        account_charts_reports if account_charts_reports is not None else []
    )
    by_account = (
        build_by_account_block(acct_src, users_map_accounts)
        if acct_src
        else {"dates": [], "series": {}, "summary": []}
    )
    by_account_by_sport: dict[str, dict] = {}
    by_account_sport_meta: list[dict[str, str]] = []
    if acct_src and include_account_by_sport_charts:
        for sp in sorted({sport_name_from_report(r) for r in acct_src}):
            sub = [r for r in acct_src if sport_name_from_report(r) == sp]
            by_account_by_sport[sp] = build_by_account_block(sub, users_map_accounts)
            sid = sport_chart_id(sp)
            if account_sport_chart_suffix:
                sid = f"{sid}__{account_sport_chart_suffix}"
            by_account_sport_meta.append({"sport": sp, "id": sid})

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": {
            "dates": o_dates,
            "cumulative": o_cum,
            "summary": [
                {"key": "Total", **dashboard_summary_row(pnl_overall, tover_overall)}
            ],
        },
        "by_stage": dim_section("stage"),
        "by_sport": dim_section("sport"),
        "by_trade_role": dim_section("trade_role"),
        "by_mkt_type": dim_section("mkt_type"),
        "by_prob_bucket": by_prob_bucket,
        "by_account": by_account,
        "by_account_by_sport": by_account_by_sport,
        "by_account_sport_meta": by_account_sport_meta,
    }


def _pad_by_account_block_for_accounts(
    blk: dict, account_order: tuple[str, ...]
) -> dict:
    """Ensure ``series`` and ``summary`` include every name in ``account_order`` (zeros if missing)."""
    dates: list[str] = list(blk.get("dates") or [])
    nd = len(dates)
    series = dict(blk.get("series") or {})
    new_series: dict[str, list[float]] = {}
    for acct in account_order:
        s = series.get(acct)
        if s is not None and len(s) == nd:
            new_series[acct] = s
        else:
            new_series[acct] = [0.0] * nd
    summ_by_key = {str(r.get("key")): r for r in (blk.get("summary") or [])}
    new_summary = []
    for acct in account_order:
        row = summ_by_key.get(acct)
        if row:
            new_summary.append(dict(row))
        else:
            new_summary.append(
                {
                    "key": acct,
                    "pnl": 0.0,
                    "turnover_usd": 0.0,
                    "return_on_turnover": None,
                }
            )
    return {**blk, "dates": dates, "series": new_series, "summary": new_summary}


def build_combined_three_account_by_sport(
    reports_by_account: dict[str, list[dict]],
    users_map: dict | None,
    account_order: tuple[str, ...] = DASHBOARD_PUBLIC_VARIANT_ACCOUNTS,
) -> dict:
    """One block per sport: cumulative PnL series for algo0, NXY, RN1 on the same chart."""
    combined: list[dict] = []
    for acct in account_order:
        combined.extend(reports_by_account.get(acct) or [])
    um = _normalize_users_map(users_map)
    allowed = set(account_order)
    combined = [
        r
        for r in combined
        if account_display_name(r.get("group") or {}, um if um else None) in allowed
    ]
    by_sport: dict[str, dict] = {}
    meta: list[dict[str, str]] = []
    for sp in sorted({sport_name_from_report(r) for r in combined}):
        sub = [r for r in combined if sport_name_from_report(r) == sp]
        blk = build_by_account_block(sub, users_map)
        blk = _pad_by_account_block_for_accounts(blk, account_order)
        by_sport[sp] = blk
        meta.append({"sport": sp, "id": f"{sport_chart_id(sp)}__combo3"})
    return {"by_account_by_sport": by_sport, "by_account_sport_meta": meta}


def _dashboard_variant_body_html(dash: str, payload: dict) -> str:
    """One copy of all sections; dash is '' or '-algo0' style suffix for element ids."""
    meta_list = payload.get("by_account_sport_meta") or []
    sport_sections_lines: list[str] = []
    for item in meta_list:
        sp = item.get("sport") or ""
        sid = item.get("id") or sport_chart_id(str(sp))
        es = html.escape(str(sp))
        sport_sections_lines.append(
            f"""  <section id="sec-acct-sport-{sid}">
    <h2>By account — {es}</h2>
    <canvas id="chart-acct-sport-{sid}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by account</h3>
    <div id="table-acct-sport-{sid}"></div>
  </section>"""
        )
    sport_sections_html = "\n\n".join(sport_sections_lines)
    acct_caption = (
        "By account (all wallets)" if dash == "" else "By account (this filter)"
    )
    return f"""  <section id="sec-overall{dash}">
    <h2>Overall — cumulative PnL by date</h2>
    <canvas id="chart-overall{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary</h3>
    <div id="table-overall{dash}"></div>
  </section>

  <section id="sec-stage{dash}">
    <h2>By stage</h2>
    <canvas id="chart-stage{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by stage</h3>
    <div id="table-stage{dash}"></div>
  </section>

  <section id="sec-sport{dash}">
    <h2>By sport (<code>unified_sport_id</code>)</h2>
    <p class="note">Sport ids are mapped to readable sport names when known.</p>
    <canvas id="chart-sport{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by sport id</h3>
    <div id="table-sport{dash}"></div>
  </section>

  <section id="sec-role{dash}">
    <h2>By role</h2>
    <canvas id="chart-role{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by trade role</h3>
    <div id="table-role{dash}"></div>
  </section>

  <section id="sec-mkt-type{dash}">
    <h2>By market type</h2>
    <canvas id="chart-mkt-type{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Statistics by market type</h3>
    <div id="table-mkt-type{dash}"></div>
  </section>

  <section id="sec-bucket{dash}">
    <h2>By prob bucket (Low / Med / High)</h2>
    <p class="note">
      From <code>implied_prob.price_buckets</code> when present; otherwise each row&rsquo;s <code>group.price_bucket</code> range maps to Low/Med/High:
      Low &lt; 0.4, Med 0.4&ndash;0.6, High &gt; 0.6 (or overlap rule for range vs bands).
    </p>
    <canvas id="chart-bucket{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by prob bucket</h3>
    <div id="table-bucket{dash}"></div>
  </section>

  <section id="sec-account{dash}">
    <h2>{html.escape(acct_caption)}</h2>
    <canvas id="chart-account{dash}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by account</h3>
    <div id="table-account{dash}"></div>
  </section>

{sport_sections_html}"""


def _combined_acct_sport_sections_html(meta: list[dict]) -> str:
    lines: list[str] = []
    for item in meta:
        sp = item.get("sport") or ""
        sid = item.get("id") or sport_chart_id(str(sp))
        es = html.escape(str(sp))
        lines.append(
            f"""  <section id="sec-combo-acct-sport-{sid}">
    <h2>By account — {es}</h2>
    <p class="note">Cumulative PnL: algo0, NXY, and RN1 in one chart.</p>
    <canvas id="chart-combo-acct-sport-{sid}"></canvas>
    <h3 style="font-size:1rem;margin-top:16px;">Summary by account</h3>
    <div id="table-combo-acct-sport-{sid}"></div>
  </section>"""
        )
    return "\n\n".join(lines)


def render_dashboard_html_variants(
    variants: list[tuple[str, str, dict]],
    source_name: str,
    date_range_note: str = "",
    *,
    combined_account_sport: dict | None = None,
) -> str:
    """variants: (id_suffix, heading, payload). id_suffix '' → unsuffixed ids (single dashboard)."""
    multi = len(variants) > 1
    blocks: list[str] = []
    for suffix, title, payload in variants:
        dash = f"-{suffix}" if suffix else ""
        head = (
            f'  <h2 class="variant-title">{html.escape(title)}</h2>\n'
            if multi
            else ""
        )
        blocks.append(head + _dashboard_variant_body_html(dash, payload))
    variants_blob = "\n\n".join(blocks)
    variants_js = json.dumps(
        [{"suffix": s, "title": t, "data": p} for s, t, p in variants],
        separators=(",", ":"),
    )
    gen0 = variants[0][2]["generated_at"] if variants else ""
    note_line = (
        f"{html.escape(date_range_note)}<br/>\n    " if date_range_note.strip() else ""
    )
    combo = combined_account_sport or {
        "by_account_by_sport": {},
        "by_account_sport_meta": [],
    }
    combo_meta = combo.get("by_account_sport_meta") or []
    if combo_meta:
        combo_heading = """  <h2 class="variant-title">By account per sport — algo0, NXY, RN1 together</h2>
  <p class="note">Below: one section per sport. Each chart plots all three accounts (lines) for that sport only.</p>

"""
        combo_sections = _combined_acct_sport_sections_html(combo_meta)
    else:
        combo_heading = ""
        combo_sections = ""
    combo_js = json.dumps(combo, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PnL dashboard — cumulative PnL</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      background: #f3f4f6;
      color: #111827;
    }}
    h1 {{
      font-size: 1.5rem;
      margin: 0 0 8px;
    }}
    .meta {{
      color: #6b7280;
      font-size: 0.9rem;
      margin-bottom: 20px;
    }}
    h2.variant-title {{
      font-size: 1.35rem;
      margin: 2rem 0 12px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e5e7eb;
      max-width: 1200px;
    }}
    section {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 16px 18px;
      margin-bottom: 18px;
      max-width: 1200px;
    }}
    h2 {{
      font-size: 1.15rem;
      margin: 0 0 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      font-weight: 600;
    }}
    canvas {{
      max-height: 340px;
    }}
    .note {{
      font-size: 0.82rem;
      color: #6b7280;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <h1>PnL time series &amp; turnover summary</h1>
  <div class="meta">
    Source JSON: {source_name}<br/>
    Generated: {gen0}<br/>
    {note_line}Row PnL prefers top-level <code>PNL</code> / <code>pnl</code>, then markout <code>total_actual_pnl</code>, then
    <code>implied_prob.timepoints.settle.pnl</code>. Turnover uses <code>total_notional</code> / <code>total_risk</code> / settle notional.
    Prob bucket: <code>implied_prob.price_buckets</code> when present; otherwise row <code>group.price_bucket</code> &rarr; Low/Med/High.
  </div>
  <br /><br />

{variants_blob}
{combo_heading}{combo_sections}

  <script>
    const VARIANTS = {variants_js};
    const COMBINED = {combo_js};

    const colors = [
      '#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed',
      '#db2777', '#0d9488', '#ca8a04', '#4f46e5', '#64748b'
    ];

    function rotCell(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return '&mdash;';
      return (100 * v).toFixed(2) + '%';
    }}

    function usdCell(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return '&mdash;';
      return Math.round(v).toLocaleString('en-US');
    }}

    function summaryTable(containerId, rows) {{
      const el = document.getElementById(containerId);
      if (!el) return;
      let h = '<table><thead><tr><th>Group</th><th>PnL (USD)</th><th>Turnover (USD)</th><th>Return on turnover</th></tr></thead><tbody>';
      for (const r of rows) {{
        h += `<tr><td>${{String(r.key)}}</td><td>${{usdCell(r.pnl)}}</td><td>${{usdCell(r.turnover_usd)}}</td><td>${{rotCell(r.return_on_turnover)}}</td></tr>`;
      }}
      h += '</tbody></table>';
      el.innerHTML = h;
    }}

    function lineChart(canvasId, labels, datasets) {{
      const ctx = document.getElementById(canvasId);
      if (!ctx) return null;
      return new Chart(ctx, {{
        type: 'line',
        data: {{ labels, datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: true,
          interaction: {{ mode: 'index', intersect: false }},
          scales: {{
            x: {{ ticks: {{ maxRotation: 45, minRotation: 30 }} }},
            y: {{
              title: {{ display: true, text: 'Cumulative PnL (USD)' }}
            }}
          }},
          plugins: {{ legend: {{ display: datasets.length > 0 }} }}
        }}
      }});
    }}

    function initVariant(V) {{
      const S = V.suffix ? ('-' + V.suffix) : '';
      const DATA = V.data;

      if (DATA.overall.dates && DATA.overall.dates.length) {{
        lineChart('chart-overall' + S, DATA.overall.dates, [{{
          label: 'Cumulative PnL',
          data: DATA.overall.cumulative,
          borderColor: colors[0],
          backgroundColor: 'rgba(37, 99, 235, 0.12)',
          tension: 0.2,
          fill: true,
          pointRadius: 2
        }}]);
      }}
      summaryTable('table-overall' + S, DATA.overall.summary || []);

      function multiSection(prefix, block) {{
        const series = block.series || {{}};
        const keys = Object.keys(series).sort();
        if (!block.dates || block.dates.length === 0 || keys.length === 0) {{
          summaryTable('table-' + prefix, block.summary || []);
          return;
        }}
        const ds = keys.map((k, i) => ({{
          label: k,
          data: series[k],
          borderColor: colors[i % colors.length],
          backgroundColor: 'transparent',
          tension: 0.2,
          pointRadius: 2
        }}));
        lineChart('chart-' + prefix, block.dates, ds);
        summaryTable('table-' + prefix, block.summary || []);
      }}

      multiSection('stage' + S, DATA.by_stage);
      multiSection('sport' + S, DATA.by_sport);
      multiSection('role' + S, DATA.by_trade_role);
      multiSection('mkt-type' + S, DATA.by_mkt_type);
      multiSection('bucket' + S, DATA.by_prob_bucket);
      multiSection('account' + S, DATA.by_account);

      function multiSectionAcctSport(safeId, block) {{
        const series = block.series || {{}};
        const keys = Object.keys(series).sort();
        if (!block.dates || block.dates.length === 0 || keys.length === 0) {{
          summaryTable('table-acct-sport-' + safeId, block.summary || []);
          return;
        }}
        const ds = keys.map((k, i) => ({{
          label: k,
          data: series[k],
          borderColor: colors[i % colors.length],
          backgroundColor: 'transparent',
          tension: 0.2,
          pointRadius: 2
        }}));
        lineChart('chart-acct-sport-' + safeId, block.dates, ds);
        summaryTable('table-acct-sport-' + safeId, block.summary || []);
      }}

      const accMeta = DATA.by_account_sport_meta || [];
      for (let i = 0; i < accMeta.length; i++) {{
        const row = accMeta[i];
        const sport = row.sport;
        const safeId = row.id;
        const block = (DATA.by_account_by_sport || {{}})[sport];
        if (block) multiSectionAcctSport(safeId, block);
      }}
    }}

    VARIANTS.forEach(initVariant);

    const COMBO_ORDER = ['algo0', 'NXY', 'RN1'];
    function multiSectionComboAcctSport(safeId, block) {{
      const series = block.series || {{}};
      const keys = COMBO_ORDER.filter((k) => Object.prototype.hasOwnProperty.call(series, k));
      if (!block.dates || block.dates.length === 0 || keys.length === 0) {{
        summaryTable('table-combo-acct-sport-' + safeId, block.summary || []);
        return;
      }}
      const ds = keys.map((k, i) => ({{
        label: k,
        data: series[k],
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        tension: 0.2,
        pointRadius: 2
      }}));
      lineChart('chart-combo-acct-sport-' + safeId, block.dates, ds);
      summaryTable('table-combo-acct-sport-' + safeId, block.summary || []);
    }}

    const comboMeta = COMBINED.by_account_sport_meta || [];
    for (let i = 0; i < comboMeta.length; i++) {{
      const row = comboMeta[i];
      const sport = row.sport;
      const safeId = row.id;
      const block = (COMBINED.by_account_by_sport || {{}})[sport];
      if (block) multiSectionComboAcctSport(safeId, block);
    }}
  </script>
</body>
</html>
"""


def render_dashboard_html(payload: dict, source_name: str) -> str:
    return render_dashboard_html_variants([("", "Dashboard", payload)], source_name, "")


def write_dashboard_html(
    reports: list[dict],
    source_label: str,
    out_path: Path,
    *,
    account_charts_reports: list[dict] | None = None,
    users_map_accounts: dict | None = None,
) -> Path:
    payload = build_dashboard_payload(
        reports,
        account_charts_reports=account_charts_reports,
        users_map_accounts=users_map_accounts,
    )
    out_path.write_text(
        render_dashboard_html(payload, source_label),
        encoding="utf-8",
    )
    return out_path


def write_dashboard_html_variants(
    variants: list[tuple[str, str, list[dict], dict | None]],
    source_label: str,
    out_path: Path,
    *,
    date_range_note: str = "",
    combined_account_sport: dict | None = None,
) -> Path:
    """Each tuple is (id_suffix, page_heading, report_rows, users_map)."""
    built: list[tuple[str, str, dict]] = []
    for suffix, title, reps, users_map in variants:
        payload = build_dashboard_payload(
            reps,
            account_charts_reports=reps,
            users_map_accounts=users_map,
            account_sport_chart_suffix=suffix,
            include_account_by_sport_charts=False,
        )
        built.append((suffix, title, payload))
    out_path.write_text(
        render_dashboard_html_variants(
            built,
            source_label,
            date_range_note,
            combined_account_sport=combined_account_sport,
        ),
        encoding="utf-8",
    )
    return out_path


# --- CLI ---


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download delta report JSON (or load from file), then CSV + PNG + HTML dashboard."
    )
    p.add_argument(
        "--input-json",
        default="",
        help="Skip browser: load this JSON. If omitted, use Playwright to download.",
    )
    p.add_argument(
        "--username",
        default=DEFAULT_HTTP_AUTH_USERNAME,
        help=f"HTTP auth username for download (default: {DEFAULT_HTTP_AUTH_USERNAME!r}).",
    )
    p.add_argument(
        "--password",
        default=DEFAULT_HTTP_AUTH_PASSWORD,
        help="HTTP auth password for download (default: set in script).",
    )
    p.add_argument(
        "--url",
        default=DEFAULT_REPORT_URL,
        help=f"Report page URL (default: {DEFAULT_REPORT_URL}).",
    )
    p.add_argument(
        "--use-public",
        "--usePublic",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            f"Trade-markout flow ({PUBLIC_REPORT_URL}), group-by for public UI, wallets field "
            f"'algo 0'. Dashboard algo0 slice uses user_id {PUBLIC_ALGO0_USER_ID}; full merge "
            "keeps all wallets for NXY/RN1. Default: on. Use --no-use-public for delta-report-v3."
        ),
    )
    p.add_argument(
        "--batch-days",
        type=int,
        default=5,
        help="Download window size in inclusive calendar days per Playwright run (default: 5).",
    )
    p.add_argument(
        "--start",
        default="2026-03-27T00:00",
        help="Start datetime-local (for download).",
    )
    p.add_argument(
        "--end",
        default="2026-04-03T23:59",
        help="End datetime-local (for download).",
    )
    p.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory.",
    )
    p.add_argument(
        "--html-name",
        default="pnl_dashboard.html",
        help="Dashboard HTML filename inside outdir.",
    )
    p.add_argument(
        "--headful",
        action="store_true",
        help="Show browser window during download.",
    )
    p.add_argument(
        "--no-matplotlib",
        action="store_true",
        help="Skip PNG + daily_settle_pnl_summary.csv.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip pnl_dashboard.html.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.username or not args.password:
        raise SystemExit("Provide --username and --password for download.")

    if args.input_json:
        in_path = Path(args.input_json).resolve()
        if not in_path.is_file():
            raise FileNotFoundError(in_path)
        raw_full = json.loads(in_path.read_text(encoding="utf-8"))
        source_label = in_path.name
    else:
        if args.batch_days < 1:
            raise SystemExit("--batch-days must be >= 1")
        nb = len(split_report_batches(args.start, args.end, args.batch_days))
        raw_full = download_merged_report_batches(
            start_arg=args.start,
            end_arg=args.end,
            batch_days=args.batch_days,
            username=args.username,
            password=args.password,
            outdir=outdir,
            use_public=args.use_public,
            report_url=args.url,
            headful=args.headful,
        )
        source_label = f"live fetch ({nb} batch(es), up to {args.batch_days}d each)"

    users_full = (
        raw_full.get("users") if isinstance(raw_full.get("users"), dict) else None
    )

    if args.use_public:
        raw_algo = filter_public_algo0_user(dict(raw_full))
    else:
        raw_algo = raw_full

    reports = iter_payload_report_rows(raw_algo)
    reports_full = iter_payload_report_rows(raw_full)
    if not reports:
        raise ValueError(
            "No usable report rows (empty after load/filter, or missing reports/groups)."
        )

    reports_clip = clip_reports_to_request_range(reports, args.start, args.end)
    reports_full_clip = clip_reports_to_request_range(
        reports_full, args.start, args.end
    )
    req_lo, req_hi = request_date_bounds(args.start, args.end)
    dashboard_date_note = (
        f"Charts and tables include only rows dated {req_lo} through {req_hi} (inclusive)."
    )

    if not args.no_matplotlib:
        daily = extract_daily_settle_pnl(raw_algo)
        csv_path, png_path = write_csv_and_plot(daily, outdir)
        print(f"Daily CSV: {csv_path}")
        print(f"Matplotlib PNG: {png_path}")

    if not args.no_html:
        if args.use_public:
            v0 = DASHBOARD_PUBLIC_VARIANT_ACCOUNTS[0]
            variants_html: list[tuple[str, str, list[dict], dict | None]] = [
                (v0, f"Account filter: {v0}", reports_clip, users_full),
            ]
            for acct in DASHBOARD_PUBLIC_VARIANT_ACCOUNTS[1:]:
                variants_html.append(
                    (
                        acct,
                        f"Account filter: {acct}",
                        filter_reports_by_account_display_name(
                            reports_full_clip, users_full, acct
                        ),
                        users_full,
                    )
                )
            combined_sport = build_combined_three_account_by_sport(
                {
                    "algo0": reports_clip,
                    "NXY": filter_reports_by_account_display_name(
                        reports_full_clip, users_full, "NXY"
                    ),
                    "RN1": filter_reports_by_account_display_name(
                        reports_full_clip, users_full, "RN1"
                    ),
                },
                users_full,
            )
            html_path = write_dashboard_html_variants(
                variants_html,
                source_label,
                outdir / args.html_name,
                date_range_note=dashboard_date_note,
                combined_account_sport=combined_sport,
            )
        else:
            html_path = write_dashboard_html(
                reports_clip,
                source_label,
                outdir / args.html_name,
                account_charts_reports=reports_full_clip,
                users_map_accounts=users_full,
            )
        print(f"Dashboard HTML: {html_path}")

    print(f"Data source: {source_label}")


if __name__ == "__main__":
    main()
