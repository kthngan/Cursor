"""
Build PnL / turnover HTML report from daily private-trade-analysis JSON files.

The generated report mirrors the deltaReportNew account summary layout and adds
sections for:
  - TIF
  - Size Factor
  - Lat ACK - TradeMatchWS
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "deltaReportPrivate"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltaReportNew.analytics import (
    COL_LEAGUE,
    COL_LONGSHOT,
    COL_MKT_TYPE,
    COL_PROBABILITY,
    COL_ROLE,
    COL_SPORTS,
    COL_STAGE,
    COL_TIER,
    GRID_LITE_CSS,
    GRID_LITE_JS,
    HIGHCHARTS_ACCESSIBILITY,
    HIGHCHARTS_EXPORT_DATA,
    HIGHCHARTS_EXPORTING,
    HIGHCHARTS_JS,
    SOCCER_NAME,
    SPORT_ID_TO_NAME,
    UNIFIED_LEAGUES_CSV,
    _json_for_inline_script,
    _league_key,
    load_unified_league_maps,
    longshot_bucket_from_any_bucket,
    probability_bucket_from_price_bucket,
    run_analyses_for_df,
)

COL_TIF = "TIF"
COL_SIZE_FACTOR = "Size Factor"
COL_LAT_ACK = "Lat ACK - TradeMatchWS"
COL_MAIN_BOOK = "Main Book"
COL_ROI_CLV = "ROI CLV%"
COL_TAG = "Tag"
COL_FROM_START = "From Start"
COL_EDGE_CLV = "Edge CLV"

# Coarse lat-ACK buckets shown in summary / desktop app (ms).
LAT_ACK_BUCKETS = ("0-100", "100-1000", "1000-3000", "3000-10000", ">10000")
ROI_CLV_BUCKETS = (
    "<-10%",
    "-10% to -5%",
    "-5% to 0%",
    "0% to 5%",
    "5% - 10%",
    ">10%",
)
FROM_START_BUCKETS = (
    "<-24h",
    "-24h to -12h",
    "-12h to -6h",
    "-6h to -4h",
    "-4h to -2h",
    "-2h to 0h",
    "0 to 30m",
    "30m to 1h",
    "1h to 1h30",
    "1h30 to 2h",
    "2h to 3h",
    "> 3h",
)
EDGE_CLV_BUCKETS = (
    "<-5",
    "-5 to -3",
    "-3 to -1",
    "-1 to 0",
    "0 - 1",
    "1-3",
    "3-5",
    "> 5",
)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def daterange_inclusive(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def _to_label(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    # Keep digits, sign, decimal point and parentheses negatives.
    s = s.replace(",", "").replace("%", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_row_date(row_date: Any, fallback: dt.date) -> dt.date:
    if row_date is None:
        return fallback
    s = str(row_date).strip()
    if not s:
        return fallback
    s = s[:10]
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return fallback


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and row.get(k) is not None:
            return row.get(k)
    return None


def _normalize_sport_label(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            sid = int(value)
            if sid in SPORT_ID_TO_NAME:
                return SPORT_ID_TO_NAME[sid]
        except (TypeError, ValueError):
            pass
    s = str(value).strip()
    if not s:
        return "N/A"
    low = s.casefold()
    for name in SPORT_ID_TO_NAME.values():
        if low == name.casefold():
            return name
    return s


def _tag_label(value: Any) -> str:
    s = _to_label(value)
    if s in {"—", "–", "-", "N/A"}:
        return "N/A"
    return s


def _role_label(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv == 1:
            return "M"
        if iv == -1:
            return "T"
    s = str(value).strip()
    if not s:
        return "N/A"
    low = s.casefold()
    if low in {"maker", "m"}:
        return "M"
    if low in {"taker", "t"}:
        return "T"
    return s


def _stage_label(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        iv = int(value)
        if iv == 0:
            return "P"
        if iv == 1:
            return "I"
    s = str(value).strip()
    if not s:
        return "N/A"
    return s


def _norm_col_key(name: str) -> str:
    s = str(name).strip().casefold().replace("→", "-")
    return re.sub(r"[^a-z0-9]+", "", s)


def _parse_lat_ack_bounds(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "N/A":
        return None
    if s.startswith(">"):
        try:
            lo = float(s[1:].strip())
            return lo, float("inf")
        except ValueError:
            return None
    if s.endswith("+"):
        try:
            lo = float(s[:-1].strip())
            return lo, float("inf")
        except ValueError:
            return None
    parts = s.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def lat_ack_bucket_from_raw(raw: Any) -> str:
    """
    Combine fine lat-ACK buckets (e.g. 0-10, 100-500, 10000+) into:
    0-100, 100-1000, 1000-3000, 3000-10000, >10000.
    """
    bounds = _parse_lat_ack_bounds(raw)
    if bounds is None:
        return "N/A"
    lo, _hi = bounds
    if lo >= 10000:
        return ">10000"
    if lo >= 3000:
        return "3000-10000"
    if lo >= 1000:
        return "1000-3000"
    if lo >= 100:
        return "100-1000"
    return "0-100"


def _normalize_range_text(s: str) -> str:
    for ch in ("\u2013", "\u2014", "\u2212", "~"):
        s = s.replace(ch, "-")
    return s


def _parse_percent_bounds(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if abs(v) <= 2.0:
            v *= 100.0
        return v, v
    s = str(raw).strip()
    if not s or s.upper() == "N/A":
        return None
    if s in ROI_CLV_BUCKETS:
        return ROI_CLV_BUCKETS.index(s), ROI_CLV_BUCKETS.index(s)
    s = _normalize_range_text(s.replace("%", "").replace(" ", ""))
    if s.startswith("<"):
        try:
            return float("-inf"), float(s[1:].strip())
        except ValueError:
            return None
    if s.startswith(">"):
        try:
            return float(s[1:].strip()), float("inf")
        except ValueError:
            return None
    if s.endswith("+"):
        try:
            return float(s[:-1].strip()), float("inf")
        except ValueError:
            return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    try:
        v = float(s)
        if abs(v) <= 2.0:
            v *= 100.0
        return v, v
    except ValueError:
        return None


def roi_clv_bucket_from_raw(raw: Any) -> str:
    """Combine fine ROI CLV% buckets into the six coarse labels used in the app."""
    if raw is None:
        return "N/A"
    s = str(raw).strip()
    if s in ROI_CLV_BUCKETS:
        return s
    bounds = _parse_percent_bounds(raw)
    if bounds is None:
        return "N/A"
    lo, _hi = bounds
    if lo < -10:
        return "<-10%"
    if lo < -5:
        return "-10% to -5%"
    if lo < 0:
        return "-5% to 0%"
    if lo < 5:
        return "0% to 5%"
    if lo < 10:
        return "5% - 10%"
    return ">10%"


def _parse_numeric_bounds(raw: Any, *, known_buckets: tuple[str, ...] = ()) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), float(raw)
    s = str(raw).strip()
    if not s or s.upper() == "N/A":
        return None
    if s in known_buckets:
        return float(known_buckets.index(s)), float(known_buckets.index(s))
    s = _normalize_range_text(s.replace("%", "").replace(" ", ""))
    if s.startswith("<"):
        try:
            return float("-inf"), float(s[1:].strip())
        except ValueError:
            return None
    if s.startswith(">"):
        try:
            return float(s[1:].strip()), float("inf")
        except ValueError:
            return None
    if s.endswith("+"):
        try:
            return float(s[:-1].strip()), float("inf")
        except ValueError:
            return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    try:
        v = float(s)
        return v, v
    except ValueError:
        return None


def edge_clv_bucket_from_raw(raw: Any) -> str:
    """Combine fine Edge CLV buckets into the eight coarse labels used in the app."""
    if raw is None:
        return "N/A"
    s = str(raw).strip()
    if s in EDGE_CLV_BUCKETS:
        return s
    bounds = _parse_numeric_bounds(raw, known_buckets=EDGE_CLV_BUCKETS)
    if bounds is None:
        return "N/A"
    lo, _hi = bounds
    if lo < -5:
        return "<-5"
    if lo < -3:
        return "-5 to -3"
    if lo < -1:
        return "-3 to -1"
    if lo < 0:
        return "-1 to 0"
    if lo < 1:
        return "0 - 1"
    if lo < 3:
        return "1-3"
    if lo < 5:
        return "3-5"
    return "> 5"


def _duration_token_to_hours(token: str) -> float | None:
    token = token.strip().casefold()
    if not token:
        return None
    sign = 1.0
    if token.startswith("-"):
        sign = -1.0
        token = token[1:]
    elif token.startswith("+"):
        token = token[1:]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)h(\d+(?:\.\d+)?)?", token)
    if m:
        hours = float(m.group(1))
        if m.group(2) is not None:
            hours += float(m.group(2)) / 60.0
        return sign * hours
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(h|m|s|ms)", token)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == "h":
            return sign * val
        if unit == "m":
            return sign * val / 60.0
        if unit == "s":
            return sign * val / 3600.0
        return sign * val / 3_600_000.0
    try:
        val = float(token)
        if abs(val) > 500:
            return val / 3_600_000.0
        return sign * val
    except ValueError:
        return None


def _parse_duration_bounds(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        val = float(raw)
        if abs(val) > 500:
            val /= 3_600_000.0
        return val, val
    s = str(raw).strip()
    if not s or s.upper() == "N/A":
        return None
    if s in FROM_START_BUCKETS:
        idx = FROM_START_BUCKETS.index(s)
        return float(idx), float(idx)
    s = _normalize_range_text(s.casefold().replace(" to ", "-").replace(" ", ""))
    if s.startswith("<"):
        hi = _duration_token_to_hours(s[1:])
        if hi is None:
            return None
        return float("-inf"), hi
    if s.startswith(">"):
        lo = _duration_token_to_hours(s[1:])
        if lo is None:
            return None
        return lo, float("inf")
    if s.endswith("+"):
        lo = _duration_token_to_hours(s[:-1])
        if lo is None:
            return None
        return lo, float("inf")
    m = re.fullmatch(r"(.+?)-(.+)", s)
    if m:
        lo = _duration_token_to_hours(m.group(1))
        hi = _duration_token_to_hours(m.group(2))
        if lo is None or hi is None:
            return None
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    single = _duration_token_to_hours(s)
    if single is not None:
        return single, single
    return None


def from_start_bucket_from_raw(raw: Any) -> str:
    """Combine fine From Start buckets into the twelve coarse labels used in the app."""
    if raw is None:
        return "N/A"
    s = str(raw).strip()
    if s in FROM_START_BUCKETS:
        return s
    bounds = _parse_duration_bounds(raw)
    if bounds is None:
        return "N/A"
    lo, _hi = bounds
    if lo < -24:
        return "<-24h"
    if lo < -12:
        return "-24h to -12h"
    if lo < -6:
        return "-12h to -6h"
    if lo < -4:
        return "-6h to -4h"
    if lo < -2:
        return "-4h to -2h"
    if lo < 0:
        return "-2h to 0h"
    if lo < 0.5:
        return "0 to 30m"
    if lo < 1:
        return "30m to 1h"
    if lo < 1.5:
        return "1h to 1h30"
    if lo < 2:
        return "1h30 to 2h"
    if lo < 3:
        return "2h to 3h"
    return "> 3h"


def _extract_lat_ack_value(row: dict[str, Any]) -> Any:
    # Fast path for known spellings.
    for key in (
        "Lat ACK - TradeMatchWS",
        "Lat Ack - TradeMatchWS",
        "Lat ACK→TradeMatchWS",
        "Lat ACK->TradeMatchWS",
    ):
        if key in row:
            return row.get(key)
    # Fallback: find any column that normalizes to latacktradematchws.
    target = "latacktradematchws"
    for k, v in row.items():
        if _norm_col_key(k) == target:
            return v
    return None


def _ordered_sports_in_df(df: pd.DataFrame) -> list[str]:
    """Return sport tabs in canonical order, then any unknown labels."""
    present = [str(x).strip() for x in df[COL_SPORTS].dropna().unique() if str(x).strip()]
    if not present:
        return []
    present_set = set(present)
    ordered: list[str] = []
    for _, name in sorted(SPORT_ID_TO_NAME.items(), key=lambda kv: kv[0]):
        if name in present_set:
            ordered.append(name)
    extras = sorted([s for s in present_set if s not in set(ordered)], key=str.lower)
    ordered.extend(extras)
    return ordered


def load_private_frames(
    json_dir: Path,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    league_name_by_id, tier_by_id = load_unified_league_maps(UNIFIED_LEAGUES_CSV)
    rows: list[dict[str, Any]] = []
    for day in daterange_inclusive(start, end):
        path = json_dir / f"{day.isoformat()}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        qd_raw = payload.get("queryDate") if isinstance(payload, dict) else None
        qd_fallback = _parse_row_date(qd_raw, day)
        groups = payload.get("groups")
        if isinstance(groups, list) and groups:
            for row in groups:
                if not isinstance(row, dict):
                    continue
                qd = _parse_row_date(
                    _first_present(row, "date", "queryDate", "date_bucket"),
                    qd_fallback,
                )
                league_raw = _first_present(row, "league", "League")
                tier_raw = _first_present(row, "tier", "Tier")
                league_id_raw = _first_present(row, "unified_league_id")
                if (league_raw is None or str(league_raw).strip() == "") and league_id_raw is not None:
                    try:
                        lid = int(league_id_raw)
                    except (TypeError, ValueError):
                        lid = None
                    if lid is not None:
                        league_raw = league_name_by_id.get(lid, f"Unknown ({lid})")
                        tier_raw = tier_by_id.get(lid, tier_raw)
                rows.append(
                    {
                        "queryDate": qd,
                        COL_SPORTS: _normalize_sport_label(
                            _first_present(row, "sport", "Sport", "unified_sport_id")
                        ),
                        COL_MKT_TYPE: _to_label(_first_present(row, "mkt_type", "market_type"), "N/A"),
                        COL_LEAGUE: _to_label(league_raw, "N/A"),
                        COL_TIER: _to_label(tier_raw, "N/A"),
                        COL_STAGE: _stage_label(_first_present(row, "stage", "bet_stage", "Stage")),
                        COL_ROLE: _role_label(_first_present(row, "role", "Role")),
                        COL_TIF: _to_label(_first_present(row, "tif", "order_tif", "TIF"), "N/A"),
                        COL_MAIN_BOOK: _to_label(
                            _first_present(row, "main_book", "mainBook", "Main Book"),
                            "N/A",
                        ),
                        COL_SIZE_FACTOR: _to_label(
                            _first_present(row, "pos_size_factor_bucket", "size_factor", "Size Factor"),
                            "N/A",
                        ),
                        COL_LAT_ACK: lat_ack_bucket_from_raw(
                            _first_present(
                                row,
                                "lat_ack_to_trade_matched_ws_bucket",
                                "lat_ack_to_tradematchws_bucket",
                                "Lat ACK - TradeMatchWS",
                            )
                        ),
                        COL_ROI_CLV: roi_clv_bucket_from_raw(
                            _first_present(
                                row,
                                "roi_clv_bucket",
                                "roi_clv_pct_bucket",
                                "ROI CLV%",
                                "roi_clv",
                            )
                        ),
                        COL_TAG: _tag_label(_first_present(row, "tag", "Tag")),
                        COL_FROM_START: from_start_bucket_from_raw(
                            _first_present(
                                row,
                                "from_start_bucket",
                                "from_start",
                                "From Start",
                            )
                        ),
                        COL_EDGE_CLV: edge_clv_bucket_from_raw(
                            _first_present(
                                row,
                                "edge_clv_bucket",
                                "Edge CLV",
                                "edge_clv",
                            )
                        ),
                        COL_PROBABILITY: probability_bucket_from_price_bucket(
                            _first_present(row, "trade_price_bucket", "Trade Price")
                        ),
                        COL_LONGSHOT: longshot_bucket_from_any_bucket(
                            _first_present(row, "trade_price_bucket", "Trade Price")
                        ),
                        "total_actual_pnl": _to_float(_first_present(row, "total_actual_pnl", "PnL")),
                        "total_risk": _to_float(
                            _first_present(row, "total_risk", "total_liab", "total_notional", "Liab")
                        ),
                        "trade_count": int(
                            _to_float(_first_present(row, "count", "trade_count", "Cnt", "Count"))
                        ),
                        "fixture_count": int(
                            _to_float(_first_present(row, "fixture_count", "Fixture Count"))
                        ),
                    }
                )
            continue

        # Backward-compatible fallback for older files that only contain scraped table rows.
        table = payload.get("table") if isinstance(payload, dict) else None
        data_rows = table.get("rows") if isinstance(table, dict) else None
        if not isinstance(data_rows, list):
            continue
        for row in data_rows:
            if not isinstance(row, dict):
                continue
            qd = _parse_row_date(row.get("Date"), qd_fallback)
            rows.append(
                {
                    "queryDate": qd,
                    COL_SPORTS: _normalize_sport_label(row.get("Sport")),
                    COL_MKT_TYPE: "N/A",
                    COL_LEAGUE: _to_label(row.get("League")),
                    COL_TIER: _to_label(row.get("Tier")),
                    COL_STAGE: _to_label(row.get("Stage")),
                    COL_ROLE: _to_label(row.get("Role")),
                    COL_TIF: _to_label(row.get("TIF")),
                    COL_MAIN_BOOK: _to_label(row.get("Main Book")),
                    COL_SIZE_FACTOR: _to_label(row.get("Size Factor")),
                    COL_LAT_ACK: lat_ack_bucket_from_raw(_extract_lat_ack_value(row)),
                    COL_ROI_CLV: roi_clv_bucket_from_raw(row.get("ROI CLV%") or row.get("roi_clv")),
                    COL_TAG: _tag_label(row.get("Tag")),
                    COL_FROM_START: from_start_bucket_from_raw(row.get("From Start")),
                    COL_EDGE_CLV: edge_clv_bucket_from_raw(row.get("Edge CLV") or row.get("edge_clv")),
                    COL_PROBABILITY: probability_bucket_from_price_bucket(row.get("Trade Price")),
                    COL_LONGSHOT: longshot_bucket_from_any_bucket(row.get("Trade Price")),
                    "total_actual_pnl": _to_float(row.get("PnL")),
                    "total_risk": _to_float(row.get("Liab")),
                    "trade_count": int(_to_float(row.get("Cnt") or row.get("Count"))),
                    "fixture_count": int(_to_float(row.get("Fixture Count"))),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "queryDate",
                COL_SPORTS,
                COL_MKT_TYPE,
                COL_LEAGUE,
                COL_TIER,
                COL_STAGE,
                COL_ROLE,
                COL_TIF,
                COL_MAIN_BOOK,
                COL_SIZE_FACTOR,
                COL_LAT_ACK,
                COL_ROI_CLV,
                COL_TAG,
                COL_FROM_START,
                COL_EDGE_CLV,
                COL_PROBABILITY,
                COL_LONGSHOT,
                "total_actual_pnl",
                "total_risk",
                "trade_count",
                "fixture_count",
            ]
        )
    return pd.DataFrame(rows)


def build_private_report_html(
    df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> str:
    overall_groupbys: list[str | None] = [
        None,
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_STAGE,
        COL_ROLE,
        COL_TIF,
        COL_MAIN_BOOK,
        COL_SIZE_FACTOR,
        COL_LAT_ACK,
        COL_ROI_CLV,
        COL_TAG,
        COL_FROM_START,
        COL_EDGE_CLV,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]
    per_sport_groupbys: list[str | None] = [
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_LEAGUE,
        COL_STAGE,
        COL_ROLE,
        COL_TIF,
        COL_MAIN_BOOK,
        COL_SIZE_FACTOR,
        COL_LAT_ACK,
        COL_ROI_CLV,
        COL_TAG,
        COL_FROM_START,
        COL_EDGE_CLV,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]
    per_sport_groupbys_soccer: list[str | None] = [
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_TIER,
        COL_STAGE,
        COL_ROLE,
        COL_TIF,
        COL_MAIN_BOOK,
        COL_SIZE_FACTOR,
        COL_LAT_ACK,
        COL_ROI_CLV,
        COL_TAG,
        COL_FROM_START,
        COL_EDGE_CLV,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]

    bundles: list[dict[str, Any]] = []
    parts: list[str] = [
        f"""<!DOCTYPE html>
<html lang="en" class="highcharts-light">
<head>
<meta charset="utf-8"/>
<title>Private summary report</title>
<link rel="stylesheet" href="{GRID_LITE_CSS}"/>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #ffffff; }}
h1,h2,h3 {{ margin-top: 1.2em; }}
.block {{ margin: 1em 0 2em; }}
.block.summary-block {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.75em 1em 1em; }}
.chart-block {{ min-height: 460px; }}
.hc-chart-host {{ width: 100%; min-height: 440px; }}
.hc-grid-host {{ min-height: 120px; margin: 0.5em 0 0; width: 100%; background: #ffffff; color-scheme: light; }}
.sport-controls {{ display: flex; align-items: center; gap: 0.6em; margin: 0.25em 0 1em; }}
.sport-controls label {{ font-weight: 600; }}
.sport-controls select {{ border: 1px solid #d1d5db; border-radius: 6px; background: #ffffff; padding: 0.35em 0.5em; }}
.report-controls {{ display: flex; align-items: center; gap: 0.6em; margin: 0.25em 0 1em; }}
.report-controls label {{ font-weight: 600; }}
.report-controls select {{ border: 1px solid #d1d5db; border-radius: 6px; background: #ffffff; padding: 0.35em 0.5em; }}
.report-slice-section:not(.is-visible) {{ display: none; }}
.sport-tabs {{ display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 1em; }}
.sport-tab-btn {{ border: 1px solid #d1d5db; border-radius: 999px; background: #ffffff; padding: 0.4em 0.8em; cursor: pointer; }}
.sport-tab-btn.is-active {{ background: #111827; color: #ffffff; border-color: #111827; }}
.sport-tab-panel {{ display: none; }}
.sport-tab-panel.is-active {{ display: block; }}
.hc-grid-host .hcg-container {{
  --ig-default-color: #111827 !important;
  --ig-default-background: #ffffff !important;
  --ig-color: #111827 !important;
  --ig-background: #ffffff !important;
  --highcharts-background-color: #ffffff !important;
  --highcharts-neutral-color-100: #111827 !important;
  --ig-highlight-color-5: #f3f6fe !important;
  background: #ffffff !important;
  color: #111827 !important;
}}
.hc-grid-host .hcg-container .hcg-table {{
  background: #ffffff !important;
  color: #111827 !important;
}}
.hc-grid-host .hcg-container .hcg-table thead th {{
  --ig-header-background: #f9fafb !important;
  background: #f9fafb !important;
  color: #111827 !important;
}}
.hc-grid-host .hcg-container .hcg-table tbody td {{
  background: #ffffff !important;
}}
.hc-grid-host .hcg-container table,
.hc-grid-host .hcg-container td,
.hc-grid-host .hcg-container th {{
  border-color: #e5e7eb !important;
}}
#hc-load-errors {{ display: none; color: #991b1b; background: #fee2e2; padding: 1em; margin: 1em 0;
  white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.9rem; border: 1px solid #fecaca; }}
</style>
</head>
<body>
<pre id="hc-load-errors"></pre>
""",
        "<h1>Private PnL / turnover report</h1>\n",
    ]
    report_meta_html = (
        f"<p>Date range: {start.isoformat()} – {end.isoformat()} (inclusive). "
        "Sections include Sport/League/Tier plus Role, TIF, Size Factor, and Lat ACK - TradeMatchWS."
        "</p>\n"
    )
    tif_values = sorted({str(x) for x in df[COL_TIF].dropna().unique() if str(x).strip()})
    tif_options_html = "".join(f'<option value="{_league_key(v)}">{v}</option>' for v in tif_values)
    parts.append(
        '<div class="report-controls"><label for="global-tif-filter">TIF filter</label>'
        '<select id="global-tif-filter"><option value="all">All TIF</option>'
        + tif_options_html
        + "</select></div>\n"
    )

    tab_buttons: list[str] = [
        '<button type="button" class="sport-tab-btn is-active" data-tab-target="sport-tab-panel-overall">Overall</button>'
    ]
    overall_parts: list[str] = ['<h1 class="main-section">Overall</h1>\n']
    overall_parts.append(
        '<div class="report-slice-section is-visible" data-sport-key="overall" data-league="all" data-tif="all">'
    )
    overall_parts.append(run_analyses_for_df(df, overall_groupbys, bundles))
    overall_parts.append("</div>\n")
    for tif_value in tif_values:
        tif_key = _league_key(tif_value)
        tif_df = df[df[COL_TIF] == tif_value]
        if tif_df.empty:
            continue
        overall_parts.append(
            f'<div class="report-slice-section" data-sport-key="overall" data-league="all" data-tif="{tif_key}">'
        )
        overall_parts.append(run_analyses_for_df(tif_df, overall_groupbys, bundles))
        overall_parts.append("</div>\n")
    tab_panels: list[str] = [
        '<section id="sport-tab-panel-overall" class="sport-tab-panel is-active">\n'
        + "".join(overall_parts)
        + "</section>\n"
    ]

    for name in _ordered_sports_in_df(df):
        s_df = df[df[COL_SPORTS] == name]
        if s_df.empty:
            continue
        sport_key = _league_key(name)
        panel_id = f"sport-tab-panel-{sport_key}"
        tab_buttons.append(
            f'<button type="button" class="sport-tab-btn" data-tab-target="{panel_id}">{name}</button>'
        )
        sport_parts: list[str] = [f'<h1 class="main-section">{name}</h1>\n']
        is_soccer = name == SOCCER_NAME
        filter_col = COL_TIER if is_soccer else COL_LEAGUE
        filter_label = "Tier filter" if is_soccer else "League filter"
        all_option_label = "All tiers" if is_soccer else "All leagues"
        slice_values = sorted({str(x) for x in s_df[filter_col].dropna().unique() if str(x).strip()})
        select_id = f"sport-league-filter-{sport_key}"
        sport_parts.append(
            f'<div class="sport-controls"><label for="{select_id}">{filter_label}</label>'
            f'<select id="{select_id}" class="sport-league-filter" data-sport-key="{sport_key}">'
            f'<option value="all">{all_option_label}</option>'
            + "".join(
                f'<option value="{_league_key(v)}">{v}</option>'
                for v in slice_values
            )
            + "</select></div>\n"
        )
        sport_gbs = per_sport_groupbys_soccer if name == SOCCER_NAME else per_sport_groupbys
        sport_parts.append(
            f'<div class="report-slice-section is-visible" data-sport-key="{sport_key}" data-league="all" data-tif="all">'
        )
        sport_parts.append(run_analyses_for_df(s_df, sport_gbs, bundles))
        sport_parts.append("</div>\n")
        for tif_value in tif_values:
            tif_key = _league_key(tif_value)
            tif_df = s_df[s_df[COL_TIF] == tif_value]
            if tif_df.empty:
                continue
            sport_parts.append(
                f'<div class="report-slice-section" data-sport-key="{sport_key}" data-league="all" data-tif="{tif_key}">'
            )
            sport_parts.append(run_analyses_for_df(tif_df, sport_gbs, bundles))
            sport_parts.append("</div>\n")
        for slice_value in slice_values:
            slice_key = _league_key(slice_value)
            slice_df = s_df[s_df[filter_col] == slice_value]
            if slice_df.empty:
                continue
            sport_parts.append(
                f'<div class="report-slice-section" data-sport-key="{sport_key}" data-league="{slice_key}" data-tif="all">'
            )
            sport_parts.append(run_analyses_for_df(slice_df, sport_gbs, bundles))
            sport_parts.append("</div>\n")
            for tif_value in tif_values:
                tif_key = _league_key(tif_value)
                slice_tif_df = slice_df[slice_df[COL_TIF] == tif_value]
                if slice_tif_df.empty:
                    continue
                sport_parts.append(
                    f'<div class="report-slice-section" data-sport-key="{sport_key}" data-league="{slice_key}" data-tif="{tif_key}">'
                )
                sport_parts.append(run_analyses_for_df(slice_tif_df, sport_gbs, bundles))
                sport_parts.append("</div>\n")
        tab_panels.append(
            f'<section id="{panel_id}" class="sport-tab-panel">' + "".join(sport_parts) + "</section>\n"
        )

    parts.append('<div class="sport-tabs">' + "".join(tab_buttons) + "</div>\n")
    parts.append(report_meta_html)
    parts.append("".join(tab_panels))
    parts.append(
        f'<script src="{HIGHCHARTS_JS}"></script>\n'
        f'<script src="{HIGHCHARTS_EXPORTING}"></script>\n'
        f'<script src="{HIGHCHARTS_EXPORT_DATA}"></script>\n'
        f'<script src="{HIGHCHARTS_ACCESSIBILITY}"></script>\n'
        f'<script src="{GRID_LITE_JS}"></script>\n'
    )
    parts.append(
        '<script type="application/json" id="hc-bundles">'
        + _json_for_inline_script(bundles)
        + "</script>\n"
    )
    parts.append(
        r"""
<script>
(function () {
  var POS = "#15803d", NEG = "#dc2626";
  function showErr(msg) {
    var box = document.getElementById("hc-load-errors");
    if (box) {
      box.style.display = "block";
      box.textContent = (box.textContent || "") + msg + "\n";
    }
    if (typeof console !== "undefined" && console.error) console.error(msg);
  }
  function gridTableEl(host) {
    if (!host) return null;
    var sr = host.shadowRoot;
    if (sr) {
      var t = sr.querySelector("table");
      if (t) return t;
    }
    return host.querySelector("table");
  }
  function paintGrid(hostId, meta) {
    var host = document.getElementById(hostId);
    if (!host || !meta) return;
    var tries = 0;
    function run() {
      var tb = gridTableEl(host);
      if (!tb && tries++ < 100) {
        requestAnimationFrame(run);
        return;
      }
      if (!tb) return;
      var rows = tb.querySelectorAll("tbody tr");
      rows.forEach(function(tr, ri) {
        var cells = tr.querySelectorAll("td");
        var pc = meta.pnlCol, rc = meta.rotCol;
        if (pc >= 0 && cells[pc] && meta.pnlSigns && meta.pnlSigns[ri] != null) {
          var s = meta.pnlSigns[ri];
          if (s > 0) { cells[pc].style.color = POS; cells[pc].style.fontWeight = "600"; }
          if (s < 0) { cells[pc].style.color = NEG; cells[pc].style.fontWeight = "600"; }
        }
        if (rc >= 0 && cells[rc] && meta.rotSigns && meta.rotSigns[ri] != null) {
          var s2 = meta.rotSigns[ri];
          if (s2 > 0) { cells[rc].style.color = POS; cells[rc].style.fontWeight = "600"; }
          if (s2 < 0) { cells[rc].style.color = NEG; cells[rc].style.fontWeight = "600"; }
        }
      });
    }
    setTimeout(run, 0);
  }
  function runInit() {
    var el = document.getElementById("hc-bundles");
    if (!el) { showErr("Internal error: hc-bundles JSON script missing."); return; }
    if (typeof Highcharts === "undefined") {
      showErr("Highcharts did not load. Serve this HTML over localhost if needed.");
      return;
    }
    var gridApi = typeof Grid !== "undefined" && typeof Grid.grid === "function" ? Grid.grid : null;
    if (!gridApi) { showErr("Highcharts Grid (Grid.grid) is not available."); }
    var bundles;
    try { bundles = JSON.parse(el.textContent); }
    catch (e) { showErr("Could not parse hc-bundles JSON: " + e.message); return; }
    bundles.forEach(function (b, idx) {
      try {
        var cEl = document.getElementById(b.chartId);
        if (!cEl) showErr("Missing chart container #" + b.chartId);
        else Highcharts.chart(cEl, b.chartOptions);
        if (b.riskChartId && b.riskChartOptions) {
          var rEl = document.getElementById(b.riskChartId);
          if (!rEl) showErr("Missing risk chart container #" + b.riskChartId);
          else Highcharts.chart(rEl, b.riskChartOptions);
        }
      } catch (e) {
        showErr("Highcharts chart #" + idx + " (" + b.chartId + "): " + e.message);
      }
      if (!gridApi) return;
      try {
        var gEl = document.getElementById(b.gridId);
        if (!gEl) { showErr("Missing grid container #" + b.gridId); return; }
        gridApi(gEl, b.gridOptions);
        paintGrid(b.gridId, b.styleMeta);
      } catch (e) {
        showErr("Grid #" + idx + " (" + b.gridId + "): " + e.message);
      }
    });
    var tabButtons = document.querySelectorAll(".sport-tab-btn");
    tabButtons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var target = btn.getAttribute("data-tab-target");
        var panels = document.querySelectorAll(".sport-tab-panel");
        panels.forEach(function (p) { p.classList.toggle("is-active", p.id === target); });
        tabButtons.forEach(function (b) { b.classList.toggle("is-active", b === btn); });
      });
    });
    function applyAllFilters() {
      var tifSel = document.getElementById("global-tif-filter");
      var selectedTif = tifSel ? (tifSel.value || "all") : "all";
      var leagueMap = {};
      var filters = document.querySelectorAll(".sport-league-filter");
      filters.forEach(function (sel) {
        var sportKey = sel.getAttribute("data-sport-key");
        leagueMap[sportKey] = sel.value || "all";
      });
      var blocks = document.querySelectorAll(".report-slice-section");
      blocks.forEach(function (blk) {
        var sportKey = blk.getAttribute("data-sport-key") || "";
        var league = blk.getAttribute("data-league") || "all";
        var tif = blk.getAttribute("data-tif") || "all";
        var selectedLeague = sportKey === "overall" ? "all" : (leagueMap[sportKey] || "all");
        var leagueOk = selectedLeague === "all" ? league === "all" : league === selectedLeague;
        var tifOk = selectedTif === "all" ? tif === "all" : tif === selectedTif;
        blk.classList.toggle("is-visible", leagueOk && tifOk);
      });
    }
    var filters = document.querySelectorAll(".sport-league-filter");
    filters.forEach(function (sel) {
      sel.addEventListener("change", applyAllFilters);
    });
    var tifSel = document.getElementById("global-tif-filter");
    if (tifSel) tifSel.addEventListener("change", applyAllFilters);
    applyAllFilters();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", runInit);
  else runInit();
})();
</script>
</body>
</html>
"""
    )
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build private summary HTML report from JSON table files."
    )
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=None,
        help="Start date (ISO). Default: end minus 29 days (30-day window).",
    )
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=None,
        help="End date (ISO). Default: today.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DATA_DIR / "json",
        help="Folder with YYYY-MM-DD.json from private downloadData.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DATA_DIR / "Reports" / "summary.html",
        help="Output HTML path.",
    )
    args = parser.parse_args()

    end = args.end or dt.date.today()
    start = args.start if args.start is not None else end - dt.timedelta(days=29)
    if start > end:
        raise SystemExit("start date must be on or before end date")

    json_dir = args.json_dir.resolve()
    df = load_private_frames(json_dir, start, end)
    html = build_private_report_html(df, start, end)

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(df)} rows).")
    if df.empty:
        print("Warning: no rows found in selected JSON/date range.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
