"""
Shared analytics for delta PnL HTML reports: JSON loading, aggregation,
Highcharts / Grid Lite builders, and HTML assembly.
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = PACKAGE_DIR / "Reports"
DEFAULT_JSON_DIR = PACKAGE_DIR / "JSON"
DEFAULT_OUTPUT = DEFAULT_REPORTS_DIR / "report.html"
DEFAULT_MARKET_OUTPUT = DEFAULT_REPORTS_DIR / "market_report.html"

DEFAULT_EXCLUDE_ACCOUNTS: tuple[str, ...] = ("1983", "2082", "8482", "2000")
DEFAULT_INCLUDE_ACCOUNTS: tuple[str, ...] = ("1771", "574079", "576330")

# Account token aliases accepted by CLI args (e.g. --accounts / --exclude-accounts).
# Keys are case-insensitive.
ACCOUNT_TOKEN_ALIASES: dict[str, int] = {
    "nxy": 2082,
    "0x507e52ef684ca2dd91f90a9d26d149dd3288beae": 2082,
    "rn1": 2000,
    "0x2005d16a84ceefa912d4e380cd32e7ff827875ea": 2000,
    "sovereign": 8482,
    "0xee613b3fc183ee44f9da9c05f53e2da107e3debf": 8482,
    "tony": 1983,
    "0x204f72f35326db932158cba6adff0b9a1da95e14": 1983,
}

SPORTS: list[dict] = [
    {"id": 1, "slug": "baseball", "name": "Baseball"},
    {"id": 2, "slug": "tennis", "name": "Tennis"},
    {"id": 3, "slug": "basketball", "name": "Basketball"},
    {"id": 4, "slug": "esports", "name": "Esports"},
    {"id": 5, "slug": "football", "name": "American Football"},
    {"id": 6, "slug": "soccer", "name": "Soccer"},
    {"id": 7, "slug": "hockey", "name": "Hockey"},
    {"id": 8, "slug": "mma", "name": "MMA"},
    {"id": 9, "slug": "cricket", "name": "Cricket"},
]

SPORT_ID_TO_NAME: dict[int, str] = {s["id"]: s["name"] for s in SPORTS}

COL_SPORTS = "Sports"
COL_MKT_TYPE = "Mkt Type"
COL_LEAGUE = "League"
COL_TIER = "Tier"
COL_STAGE = "Stage"
COL_ROLE = "Role"
COL_PROBABILITY = "Probability Bucket"
COL_LONGSHOT = "Longshot Bucket"
GROUP_COLUMNS_INTERVAL_SORT: frozenset[str] = frozenset(
    {"Lat ACK - TradeMatchWS", "ROI CLV%", "From Start", "Edge CLV"}
)
ORDERED_GROUP_BUCKETS: dict[str, tuple[str, ...]] = {
    "ROI CLV%": (
        "<-10%",
        "-10% to -5%",
        "-5% to 0%",
        "0% to 5%",
        "5% - 10%",
        ">10%",
    ),
    "From Start": (
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
    ),
    "Edge CLV": (
        "<-5",
        "-5 to -3",
        "-3 to -1",
        "-1 to 0",
        "0 - 1",
        "1-3",
        "3-5",
        "> 5",
    ),
}

# Summary grid column headers (aggregate_pnl_turnover uses pnl / turnover / return_on_turnover).
SUMMARY_COL_PNL = "PnL USD"
SUMMARY_COL_TURNOVER = "Turnover USD"
SUMMARY_COL_ROT = "Return On Turnover %"

UNIFIED_LEAGUES_CSV = PACKAGE_DIR / "unified_leagues.csv"
UNIFIED_MARKETS_CSV = PACKAGE_DIR / "unified_markets.csv"
SOCCER_NAME = "Soccer"

# Market report: collapse default algo accounts into one row label.
COL_ACCOUNT = "Account"
ALGO_USER_IDS: frozenset[int] = frozenset({1771, 574079, 576330})
# nxy, rn1, sovereign, tony + default include accounts (mapped to Algo).
DEFAULT_MARKET_USER_IDS: frozenset[int] = frozenset(
    {1983, 2082, 8482, 2000, 1771, 574079, 576330}
)


def market_account_label(user_id: int) -> str:
    """Map ``user_id`` to display label: three algo IDs → ``Algo``; named accounts keep short names."""
    uid = int(user_id)
    if uid in ALGO_USER_IDS:
        return "Algo"
    if uid == 2082:
        return "nxy"
    if uid == 2000:
        return "rn1"
    if uid == 8482:
        return "sovereign"
    if uid == 1983:
        return "tony"
    return str(uid)


def add_market_account_column(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``COL_ACCOUNT`` from ``user_id`` for market summary grouping."""
    out = df.copy()
    out[COL_ACCOUNT] = out["user_id"].map(market_account_label)
    return out

_COLOR_POSITIVE = "#15803d"
_COLOR_NEGATIVE = "#dc2626"

# Pinned jsDelivr builds (same origin family; more reliable than file:// + code.highcharts.com).
_HC_VER = "12.2.0"
HIGHCHARTS_JS = f"https://cdn.jsdelivr.net/npm/highcharts@{_HC_VER}/highcharts.js"
HIGHCHARTS_EXPORTING = f"https://cdn.jsdelivr.net/npm/highcharts@{_HC_VER}/modules/exporting.js"
HIGHCHARTS_EXPORT_DATA = f"https://cdn.jsdelivr.net/npm/highcharts@{_HC_VER}/modules/export-data.js"
HIGHCHARTS_ACCESSIBILITY = f"https://cdn.jsdelivr.net/npm/highcharts@{_HC_VER}/modules/accessibility.js"
GRID_LITE_JS = "https://cdn.jsdelivr.net/npm/@highcharts/grid-lite@2.3.1/grid-lite.js"
GRID_LITE_CSS = "https://cdn.jsdelivr.net/npm/@highcharts/grid-lite@2.3.1/css/grid-lite.css"


def parse_account_token(token: str) -> int:
    t = token.strip()
    if not t:
        raise ValueError("empty account token")
    alias = ACCOUNT_TOKEN_ALIASES.get(t.lower())
    if alias is not None:
        return alias
    if t.lower().startswith("0x"):
        return int(t, 16)
    return int(t, 10)


def probability_bucket_from_price_bucket(price_bucket: str | None) -> str:
    """
    One of five implied-probability bins from the ``price_bucket`` interval ``[lo, hi]``:

    ``0-0.2``, ``0.2-0.4``, ``0.4-0.6``, ``0.6-0.8``, ``0.8-1`` — chosen by the **midpoint**
    ``(lo + hi) / 2`` after scaling to 0–1 (left-inclusive bands on the midpoint:
    ``[0,0.2)``, ``[0.2,0.4)``, …, ``[0.8,1]``).

    The API often uses **percent** bands (e.g. ``50-60``); values with ``max(lo, hi) > 1``
    are divided by 100. Pure 0–1 bands (e.g. ``0.2-0.35``) are unchanged.
    """
    unknown = "0.4-0.6"
    if price_bucket is None:
        return unknown
    raw = str(price_bucket).strip()
    if not raw:
        return unknown
    parts = raw.split("-", 1)
    if len(parts) != 2:
        return unknown
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        return unknown
    if lo > hi:
        lo, hi = hi, lo
    if max(lo, hi) > 1.0:
        lo, hi = lo / 100.0, hi / 100.0
    m = max(0.0, min(1.0, (lo + hi) / 2.0))
    if m < 0.2:
        return "0-0.2"
    if m < 0.4:
        return "0.2-0.4"
    if m < 0.6:
        return "0.4-0.6"
    if m < 0.8:
        return "0.6-0.8"
    return "0.8-1"


def _bucket_midpoint(bucket: Any) -> float | None:
    """Return normalized midpoint in [0,1] from a bucket string like ``70-80`` or ``0.7-0.8``."""
    if bucket is None:
        return None
    raw = str(bucket).strip()
    if not raw:
        return None
    parts = raw.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    if max(lo, hi) > 1.0:
        lo, hi = lo / 100.0, hi / 100.0
    return max(0.0, min(1.0, (lo + hi) / 2.0))


def longshot_bucket_from_any_bucket(*bucket_candidates: Any) -> str:
    """
    Three-bin longshot bucket from the first parsable candidate:
    ``0-0.1``, ``0.1-0.9``, ``0.9-1.0``.
    """
    for candidate in bucket_candidates:
        m = _bucket_midpoint(candidate)
        if m is None:
            continue
        if m < 0.1:
            return "0-0.1"
        if m < 0.9:
            return "0.1-0.9"
        return "0.9-1.0"
    return "0.1-0.9"


def _interval_sort_key(label: Any) -> tuple[float, float]:
    """Parse ``lo-hi`` (int or float) for stable numeric ordering; unknown last."""
    if label is None:
        return (float("inf"), float("inf"))
    if isinstance(label, float) and pd.isna(label):
        return (float("inf"), float("inf"))
    s = str(label).strip()
    if not s or s.upper() == "N/A":
        return (float("inf"), float("inf"))
    m_gt = re.fullmatch(r">\s*(-?\d+(?:\.\d+)?)", s)
    if m_gt:
        try:
            return float(m_gt.group(1)), float("inf")
        except (TypeError, ValueError):
            return (float("inf"), float("inf"))
    m_plus = re.fullmatch(r"(-?\d+(?:\.\d+)?)\+", s)
    if m_plus:
        try:
            return float(m_plus.group(1)), float("inf")
        except (TypeError, ValueError):
            return (float("inf"), float("inf"))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)", s)
    if not m:
        return (float("inf"), float("inf"))
    try:
        return float(m.group(1)), float(m.group(2))
    except (TypeError, ValueError):
        return (float("inf"), float("inf"))


def _group_sort_key(group_by: str, label: Any) -> tuple[float, float]:
    order = ORDERED_GROUP_BUCKETS.get(group_by)
    if order is not None:
        s = str(label).strip()
        try:
            idx = float(order.index(s))
            return idx, idx
        except ValueError:
            return float(len(order)), float("inf")
    return _interval_sort_key(label)


def daterange_inclusive(start: dt.date, end: dt.date) -> list[dt.date]:
    out: list[dt.date] = []
    d = start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=1)
    return out


def load_unified_league_maps(csv_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    """
    Read ``unified_leagues.csv`` (``id``, ``name``, ``tier``).
    Returns (league_id -> display name, league_id -> tier label). Missing/blank tier maps to "".
    """
    name_by_id: dict[int, str] = {}
    tier_by_id: dict[int, str] = {}
    if not csv_path.is_file():
        return name_by_id, tier_by_id
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return name_by_id, tier_by_id
    if "id" not in df.columns:
        return name_by_id, tier_by_id
    for _, row in df.iterrows():
        try:
            lid = int(row["id"])
        except (TypeError, ValueError):
            continue
        nm = row.get("name")
        if pd.notna(nm) and str(nm).strip():
            name_by_id[lid] = str(nm).strip()
        else:
            name_by_id[lid] = f"League {lid}"
        t = row.get("tier")
        if pd.isna(t) or (isinstance(t, str) and not str(t).strip()):
            tier_by_id[lid] = ""
        else:
            try:
                tier_by_id[lid] = str(int(float(t)))
            except (TypeError, ValueError):
                tier_by_id[lid] = str(t).strip()
    return name_by_id, tier_by_id


def _league_display_columns(
    unified_league_id: Any,
    name_by_id: dict[int, str],
    tier_by_id: dict[int, str],
) -> tuple[str, str]:
    """(League name / unknown label, Tier label or ``N/A``)."""
    if unified_league_id is None:
        return "N/A", "N/A"
    try:
        lid = int(unified_league_id)
    except (TypeError, ValueError):
        return "N/A", "N/A"
    league_name = name_by_id.get(lid, f"Unknown ({lid})")
    tier_raw = tier_by_id.get(lid)
    if tier_raw is None or tier_raw == "":
        tier_disp = "N/A"
    else:
        tier_disp = tier_raw
    return league_name, tier_disp


def load_unified_market_type_map(csv_path: Path) -> dict[int, str]:
    """``unified_markets.csv``: unified market ``id`` -> ``market_type`` (e.g. moneyline, spreads)."""
    out: dict[int, str] = {}
    if not csv_path.is_file():
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    if "id" not in df.columns or "market_type" not in df.columns:
        return out
    for _, row in df.iterrows():
        try:
            mid = int(row["id"])
        except (TypeError, ValueError):
            continue
        mt = row.get("market_type")
        if pd.notna(mt) and str(mt).strip():
            out[mid] = str(mt).strip()
        else:
            out[mid] = "N/A"
    return out


def _mkt_type_display(g: dict[str, Any], type_by_id: dict[int, str]) -> str:
    """Resolve display label from JSON group row; prefers explicit type strings, then ``unified_market_id``."""
    for key in ("market_type", "mkt_type"):
        v = g.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    for key in ("unified_market_id", "unified_marketId"):
        raw = g.get(key)
        if raw is None:
            continue
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            continue
        return type_by_id.get(mid, f"Unknown ({mid})")
    raw_mid = g.get("market_id")
    if raw_mid is not None:
        try:
            mid = int(raw_mid)
        except (TypeError, ValueError):
            return "N/A"
        if mid in type_by_id:
            return type_by_id[mid]
    return "N/A"


def load_frames(
    json_dir: Path,
    start: dt.date,
    end: dt.date,
    user_ids: set[int],
    leagues_csv: Path | None = None,
    markets_csv: Path | None = None,
) -> pd.DataFrame:
    leagues_path = leagues_csv if leagues_csv is not None else UNIFIED_LEAGUES_CSV
    markets_path = markets_csv if markets_csv is not None else UNIFIED_MARKETS_CSV
    name_by_id, tier_by_id = load_unified_league_maps(leagues_path)
    mkt_type_by_id = load_unified_market_type_map(markets_path)
    rows: list[dict] = []
    for day in daterange_inclusive(start, end):
        path = json_dir / f"{day.isoformat()}.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("groups") or []:
            uid = g.get("user_id")
            if uid not in user_ids:
                continue
            sid = g.get("unified_sport_id")
            sport_name = SPORT_ID_TO_NAME.get(sid, f"Sport {sid}")
            qd_raw = g.get("queryDate")
            if qd_raw:
                try:
                    qd: dt.date = dt.date.fromisoformat(str(qd_raw)[:10])
                except ValueError:
                    qd = day
            else:
                qd = day
            league_name, tier_disp = _league_display_columns(
                g.get("unified_league_id"), name_by_id, tier_by_id
            )
            mkt_type_label = _mkt_type_display(g, mkt_type_by_id)
            rows.append(
                {
                    "queryDate": qd,
                    "user_id": uid,
                    COL_SPORTS: sport_name,
                    COL_MKT_TYPE: mkt_type_label,
                    COL_LEAGUE: league_name,
                    COL_TIER: tier_disp,
                    COL_STAGE: g.get("bet_stage"),
                    COL_ROLE: "Maker" if g.get("is_maker") else "Taker",
                    COL_PROBABILITY: probability_bucket_from_price_bucket(g.get("price_bucket")),
                    COL_LONGSHOT: longshot_bucket_from_any_bucket(g.get("price_bucket")),
                    "total_actual_pnl": float(g.get("total_actual_pnl") or 0),
                    "total_risk": float(g.get("total_risk") or 0),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "queryDate",
                "user_id",
                COL_SPORTS,
                COL_MKT_TYPE,
                COL_LEAGUE,
                COL_TIER,
                COL_STAGE,
                COL_ROLE,
                COL_PROBABILITY,
                COL_LONGSHOT,
                "total_actual_pnl",
                "total_risk",
            ]
        )
    return pd.DataFrame(rows)


def collect_user_ids_in_json(json_dir: Path, start: dt.date, end: dt.date) -> list[int]:
    """Unique `user_id` values appearing in `groups` for any file in the date range."""
    found: set[int] = set()
    for day in daterange_inclusive(start, end):
        path = json_dir / f"{day.isoformat()}.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for g in data.get("groups") or []:
            uid = g.get("user_id")
            if uid is not None:
                found.add(int(uid))
    return sorted(found)


def aggregate_pnl_turnover(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Sum PNL (total_actual_pnl) and Risk (total_risk) by group_by.
    Renames to pnl and turnover; adds return_on_turnover = pnl / turnover.
    """
    if df.empty:
        cols = ["pnl", "turnover", "return_on_turnover"]
        if group_by:
            return pd.DataFrame(columns=[group_by] + cols)
        return pd.DataFrame(columns=cols)

    if group_by is None:
        totals = df[["total_actual_pnl", "total_risk"]].sum()
        out = pd.DataFrame(
            {
                "pnl": [totals["total_actual_pnl"]],
                "turnover": [totals["total_risk"]],
            }
        )
    else:
        out = (
            df.groupby(group_by, dropna=False, as_index=False)[
                ["total_actual_pnl", "total_risk"]
            ]
            .sum()
            .rename(
                columns={
                    "total_actual_pnl": "pnl",
                    "total_risk": "turnover",
                }
            )
        )
        if group_by in GROUP_COLUMNS_INTERVAL_SORT:
            k0 = out[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[0])
            k1 = out[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[1])
            out = (
                out.assign(_isk0=k0, _isk1=k1)
                .sort_values(["_isk0", "_isk1"], kind="stable")
                .drop(columns=["_isk0", "_isk1"])
            )
        else:
            out = out.sort_values(group_by, kind="stable")

    turnover_safe = out["turnover"].replace(0, pd.NA)
    out["return_on_turnover"] = out["pnl"] / turnover_safe
    return out.reset_index(drop=True)


def compute_daily_cumulative_pnl(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Build the exact daily series used for cumulative PNL charts.

    Per ``queryDate``: ``daily_pnl`` = sum of ``total_actual_pnl`` (all detail rows for that query day).
    Then ``cumulative_pnl`` = pandas ``.cumsum()`` of ``daily_pnl`` in ``queryDate`` order (within each
    ``group_by`` slice when grouping). This is a running *sum* of PNL, not a row count.
    """
    if df.empty:
        cols = ["queryDate", "daily_pnl", "cumulative_pnl", "plot_date"]
        if group_by:
            cols.insert(1, group_by)
        return pd.DataFrame(columns=cols)

    keys = ["queryDate"] + ([group_by] if group_by else [])
    daily = (
        df.groupby(keys, dropna=False, as_index=False)["total_actual_pnl"]
        .sum()
        .rename(columns={"total_actual_pnl": "daily_pnl"})
    )
    daily["daily_pnl"] = pd.to_numeric(daily["daily_pnl"], errors="coerce").fillna(0.0)
    if group_by:
        if group_by in GROUP_COLUMNS_INTERVAL_SORT:
            k0 = daily[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[0])
            k1 = daily[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[1])
            daily = (
                daily.assign(_isk0=k0, _isk1=k1)
                .sort_values(["_isk0", "_isk1", "queryDate"], kind="stable")
                .drop(columns=["_isk0", "_isk1"])
            )
        else:
            daily = daily.sort_values([group_by, "queryDate"], kind="stable")
        daily["cumulative_pnl"] = daily.groupby(group_by, dropna=False)["daily_pnl"].cumsum()
    else:
        daily = daily.sort_values("queryDate", kind="stable")
        daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    daily = daily.assign(plot_date=pd.to_datetime(daily["queryDate"]))
    return daily


def compute_daily_total_risk(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> pd.DataFrame:
    """
    Per ``queryDate``: sum of ``total_risk`` (USD), with the same ``group_by`` keys as PNL charts.
    """
    if df.empty:
        cols = ["queryDate", "daily_risk"]
        if group_by:
            cols.insert(1, group_by)
        return pd.DataFrame(columns=cols)

    keys = ["queryDate"] + ([group_by] if group_by else [])
    daily = (
        df.groupby(keys, dropna=False, as_index=False)["total_risk"]
        .sum()
        .rename(columns={"total_risk": "daily_risk"})
    )
    daily["daily_risk"] = pd.to_numeric(daily["daily_risk"], errors="coerce").fillna(0.0)
    if group_by:
        if group_by in GROUP_COLUMNS_INTERVAL_SORT:
            k0 = daily[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[0])
            k1 = daily[group_by].map(lambda x, gb=group_by: _group_sort_key(gb, x)[1])
            daily = (
                daily.assign(_isk0=k0, _isk1=k1)
                .sort_values(["_isk0", "_isk1", "queryDate"], kind="stable")
                .drop(columns=["_isk0", "_isk1"])
            )
        else:
            daily = daily.sort_values([group_by, "queryDate"], kind="stable")
    else:
        daily = daily.sort_values("queryDate", kind="stable")
    return daily


def _query_date_to_utc_ms(d: Any) -> int:
    """Unix ms at UTC midnight for a calendar queryDate (or pandas timestamp)."""
    if isinstance(d, dt.datetime):
        dd = d
        if dd.tzinfo is None:
            dd = dd.replace(tzinfo=dt.timezone.utc)
        return int(dd.timestamp() * 1000)
    if isinstance(d, dt.date):
        dd = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
        return int(dd.timestamp() * 1000)
    ts = pd.Timestamp(d)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.normalize()
    return int(ts.timestamp() * 1000)


def highcharts_cumulative_config(
    daily: pd.DataFrame,
    group_by: str | None,
    title_suffix: str,
) -> dict[str, Any]:
    """JSON-serializable options for ``Highcharts.chart`` (line, datetime x-axis)."""
    title_text = f"Cumulative PNL ({title_suffix})"
    empty = {
        "chart": {"type": "line", "zoomType": "x", "height": 450},
        "title": {"text": title_text},
        "xAxis": {"type": "datetime", "title": {"text": "queryDate"}},
        "yAxis": {"title": {"text": "Cumulative PNL"}},
        "series": [],
        "credits": {"enabled": False},
        "exporting": {"enabled": True},
    }
    if daily.empty:
        return empty

    series: list[dict[str, Any]] = []
    if group_by:
        if group_by in GROUP_COLUMNS_INTERVAL_SORT:
            names = sorted(
                daily[group_by].unique(),
                key=lambda x, gb=group_by: _group_sort_key(gb, x),
            )
        else:
            names = [n for n, _ in daily.groupby(group_by, dropna=False)]
        for name in names:
            g = daily.loc[daily[group_by] == name]
            disp = name if name is not None and not (isinstance(name, float) and pd.isna(name)) else "N/A"
            data: list[list[Any]] = []
            for _, row in g.iterrows():
                y = float(row["cumulative_pnl"])
                if pd.isna(y):
                    continue
                data.append([_query_date_to_utc_ms(row["queryDate"]), round(y, 6)])
            series.append({"type": "line", "name": str(disp), "data": data})
    else:
        data = []
        for _, row in daily.iterrows():
            y = float(row["cumulative_pnl"])
            if pd.isna(y):
                continue
            data.append([_query_date_to_utc_ms(row["queryDate"]), round(y, 6)])
        series.append({"type": "line", "name": "Cumulative PNL", "data": data})

    return {
        "chart": {"type": "line", "zoomType": "x", "height": 450},
        "title": {"text": title_text},
        "xAxis": {"type": "datetime", "title": {"text": "queryDate"}},
        "yAxis": {"title": {"text": "Cumulative PNL"}},
        "tooltip": {
            "shared": True,
            "xDateFormat": "%Y-%m-%d",
            "pointFormat": (
                '<span style="color:{series.color}">{series.name}</span>: '
                "<b>{point.y:,.0f}</b><br/>"
            ),
        },
        "legend": {"enabled": bool(group_by)},
        "plotOptions": {
            "line": {"marker": {"enabled": True, "radius": 3}},
            "series": {"connectNulls": True},
        },
        "series": series,
        "credits": {"enabled": False},
        "exporting": {"enabled": True},
    }


def highcharts_daily_risk_bar_config(
    daily_risk: pd.DataFrame,
    group_by: str | None,
    title_suffix: str,
) -> dict[str, Any]:
    """JSON-serializable options for daily sum of total_risk (USD) as grouped columns."""
    title_text = f"Daily total risk (USD) ({title_suffix})"
    base = {
        "chart": {"type": "column", "zoomType": "x", "height": 450},
        "title": {"text": title_text},
        "xAxis": {"type": "datetime", "title": {"text": "queryDate"}},
        "yAxis": {"title": {"text": "Total risk (USD)"}},
        "tooltip": {
            "shared": True,
            "xDateFormat": "%Y-%m-%d",
            "pointFormat": (
                '<span style="color:{series.color}">{series.name}</span>: '
                "<b>{point.y:,.0f}</b> USD<br/>"
            ),
        },
        "legend": {"enabled": bool(group_by)},
        "plotOptions": {
            "column": {
                "grouping": True,
                "groupPadding": 0.08,
                "pointPadding": 0.02,
                "borderWidth": 0,
            },
        },
        "series": [],
        "credits": {"enabled": False},
        "exporting": {"enabled": True},
    }
    if daily_risk.empty:
        return base

    series: list[dict[str, Any]] = []
    if group_by:
        if group_by in GROUP_COLUMNS_INTERVAL_SORT:
            names = sorted(
                daily_risk[group_by].unique(),
                key=lambda x, gb=group_by: _group_sort_key(gb, x),
            )
        else:
            names = [n for n, _ in daily_risk.groupby(group_by, dropna=False)]
        for name in names:
            g = daily_risk.loc[daily_risk[group_by] == name]
            disp = name if name is not None and not (isinstance(name, float) and pd.isna(name)) else "N/A"
            data: list[list[Any]] = []
            for _, row in g.iterrows():
                y = float(row["daily_risk"])
                if pd.isna(y):
                    continue
                data.append([_query_date_to_utc_ms(row["queryDate"]), round(y, 6)])
            series.append({"type": "column", "name": str(disp), "data": data})
    else:
        data = []
        for _, row in daily_risk.iterrows():
            y = float(row["daily_risk"])
            if pd.isna(y):
                continue
            data.append([_query_date_to_utc_ms(row["queryDate"]), round(y, 6)])
        series.append({"type": "column", "name": "Total risk (USD)", "data": data})

    out = dict(base)
    out["series"] = series
    return out


def _summary_column_header(col: str) -> str:
    if col == "pnl":
        return SUMMARY_COL_PNL
    if col == "turnover":
        return SUMMARY_COL_TURNOVER
    if col == "return_on_turnover":
        return SUMMARY_COL_ROT
    return col


def grid_lite_options_from_summary(summary: pd.DataFrame) -> dict[str, Any]:
    """
    Options for ``Grid.grid(container, options)`` — column-oriented ``data.columns`` API
    (see Highcharts Grid Lite CDN docs).
    """
    columns: dict[str, list[Any]] = {}
    for c in summary.columns:
        header = _summary_column_header(str(c))
        key = header if header != str(c) else str(c)
        col_values: list[Any] = []
        for _, row in summary.iterrows():
            v = row[c]
            if c == "pnl":
                col_values.append(
                    None if pd.isna(v) else int(round(float(v)))
                )
            elif c == "turnover":
                col_values.append(
                    None if pd.isna(v) else int(round(float(v)))
                )
            elif c == "return_on_turnover":
                col_values.append(
                    "" if pd.isna(v) else f"{float(v) * 100.0:.2f}%"
                )
            else:
                col_values.append("" if pd.isna(v) else str(v))
        columns[key] = col_values
    return {"data": {"columns": columns}}


def _json_for_inline_script(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")


def new_section_hc_ids(include_risk: bool = True) -> tuple[str, str | None, str]:
    u = uuid.uuid4().hex[:12]
    chart = f"hc-chart-{u}"
    grid = f"hc-grid-{u}"
    risk = f"hc-risk-chart-{u}" if include_risk else None
    return chart, risk, grid


def _style_meta_from_summary(summ: pd.DataFrame, gopts: dict[str, Any]) -> dict[str, Any]:
    gkeys = list(gopts["data"]["columns"].keys())
    pnl_idx = gkeys.index(SUMMARY_COL_PNL) if SUMMARY_COL_PNL in gkeys else -1
    rot_idx = gkeys.index(SUMMARY_COL_ROT) if SUMMARY_COL_ROT in gkeys else -1
    pnl_signs: list[int | None] = []
    rot_signs: list[int | None] = []
    if not summ.empty:
        if "pnl" in summ.columns:
            for x in summ["pnl"]:
                if pd.isna(x):
                    pnl_signs.append(None)
                else:
                    fv = float(x)
                    pnl_signs.append(1 if fv > 0 else (-1 if fv < 0 else 0))
        if "return_on_turnover" in summ.columns:
            for x in summ["return_on_turnover"]:
                if pd.isna(x):
                    rot_signs.append(None)
                else:
                    fv = float(x)
                    rot_signs.append(1 if fv > 0 else (-1 if fv < 0 else 0))
    return {
        "pnlCol": pnl_idx,
        "rotCol": rot_idx,
        "pnlSigns": pnl_signs,
        "rotSigns": rot_signs,
    }


def _bundle_variant_for_df(
    slice_df: pd.DataFrame,
    gb: str | None,
    label: str,
    *,
    include_risk: bool,
) -> dict[str, Any]:
    summ = aggregate_pnl_turnover(slice_df, gb)
    daily = compute_daily_cumulative_pnl(slice_df, gb)
    daily_risk = (
        compute_daily_total_risk(slice_df, gb) if include_risk else pd.DataFrame()
    )
    gopts = grid_lite_options_from_summary(summ)
    out: dict[str, Any] = {
        "chartOptions": highcharts_cumulative_config(daily, gb, label),
        "gridOptions": gopts,
        "styleMeta": _style_meta_from_summary(summ, gopts),
    }
    if include_risk:
        out["riskChartOptions"] = highcharts_daily_risk_bar_config(daily_risk, gb, label)
    else:
        out["riskChartOptions"] = None
    return out


def _league_key(league_name: Any) -> str:
    """Stable key used in HTML data attributes for per-league filtering."""
    if league_name is None:
        return "unknown"
    s = str(league_name).strip().lower()
    if not s:
        return "unknown"
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    key = "".join(out).strip("-")
    return key or "unknown"


def section_block(
    heading: str,
    chart_id: str,
    grid_id: str,
    risk_chart_id: str | None = None,
    *,
    heading_tag: str | None = None,
) -> str:
    """Single analysis block: Grid summary, cumulative PNL chart; optional daily risk chart."""
    risk_block = ""
    if risk_chart_id:
        risk_block = f"""
<div class="block chart-block">
<h3>Chart — daily total risk (USD)</h3>
<div id="{html_lib.escape(risk_chart_id)}" class="hc-chart-host"></div>
</div>
"""
    if heading_tag is not None:
        level = heading_tag
    else:
        level = "h2" if risk_chart_id else "h3"
    return f"""
<div class="block section-block">
<{level}>{html_lib.escape(heading)}</{level}>
<div class="block summary-block">
<h3>Summary</h3>
<div id="{html_lib.escape(grid_id)}" class="hc-grid-host"></div>
</div>
<div class="block chart-block">
<h3>Chart — cumulative PNL</h3>
<div id="{html_lib.escape(chart_id)}" class="hc-chart-host"></div>
</div>
{risk_block}
</div>
"""


def run_analyses_for_df(
    df: pd.DataFrame,
    groupbys: Sequence[str | None],
    bundles: list[dict[str, Any]],
    *,
    include_risk: bool = True,
    group_labels: dict[str | None, str] | None = None,
    section_heading_tag: str | None = None,
) -> str:
    """Append Highcharts/Grid bundle dicts and return HTML fragments."""
    chunks: list[str] = []
    gl = group_labels or {}
    for gb in groupbys:
        if gb is None:
            label = gl.get(None, "No group by")
        else:
            label = gl.get(gb, gb)
        chart_id, risk_chart_id, grid_id = new_section_hc_ids(include_risk=include_risk)
        variant = _bundle_variant_for_df(df, gb, label, include_risk=include_risk)
        bundle: dict[str, Any] = {
            "chartId": chart_id,
            "gridId": grid_id,
            "chartOptions": variant["chartOptions"],
            "gridOptions": variant["gridOptions"],
            "styleMeta": variant["styleMeta"],
        }
        if include_risk and risk_chart_id:
            bundle["riskChartId"] = risk_chart_id
            bundle["riskChartOptions"] = variant["riskChartOptions"]
        bundles.append(bundle)
        chunks.append(
            section_block(
                label,
                chart_id,
                grid_id,
                risk_chart_id,
                heading_tag=section_heading_tag,
            )
        )
    return "\n".join(chunks)


def _tier_sort_key(label: Any) -> tuple[int, str]:
    """Sort tier strings numerically when possible; ``N/A`` last."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return (999999, "")
    s = str(label).strip()
    if not s or s.upper() == "N/A":
        return (999999, s)
    try:
        return (int(float(s)), s)
    except (TypeError, ValueError):
        return (888888, s)


def _sorted_slice_keys(series: pd.Series, *, soccer_mode: bool) -> list[Any]:
    """Unique labels from ``series``, sorted (numeric tier order for soccer)."""
    raw = [x for x in series.dropna().unique()]
    if not raw:
        return []
    if soccer_mode:
        return sorted(raw, key=_tier_sort_key)
    return sorted(raw, key=lambda x: str(x).lower())


def build_market_report_html(
    df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> str:
    """
    Market summary: tabbed layout (Overall first, then one tab per sport with data), mirroring
    :func:`build_report_html`. Sport tabs: summary / cumulative PnL / daily risk by league (tier for
    soccer), then the same by account, then per-league (or tier) account slices. ``df`` must include
    ``COL_ACCOUNT`` from :func:`add_market_account_column`.
    """
    hc_js = html_lib.escape(HIGHCHARTS_JS)
    hc_exp = html_lib.escape(HIGHCHARTS_EXPORTING)
    hc_ed = html_lib.escape(HIGHCHARTS_EXPORT_DATA)
    hc_a11y = html_lib.escape(HIGHCHARTS_ACCESSIBILITY)
    grid_js = html_lib.escape(GRID_LITE_JS)
    grid_css = html_lib.escape(GRID_LITE_CSS)

    bundles: list[dict[str, Any]] = []
    overview_groupbys: list[str | None] = [None, COL_SPORTS]

    parts: list[str] = [
        f"""<!DOCTYPE html>
<html lang="en" class="highcharts-light">
<head>
<meta charset="utf-8"/>
<title>Market summary</title>
<link rel="stylesheet" href="{grid_css}"/>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #ffffff; }}
h1,h2,h3 {{ margin-top: 1.2em; }}
.block {{ margin: 1em 0 2em; }}
.block.summary-block {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.75em 1em 1em; }}
.chart-block {{ min-height: 460px; }}
.hc-chart-host {{ width: 100%; min-height: 440px; }}
.hc-grid-host {{ min-height: 120px; margin: 0.5em 0 0; width: 100%; background: #ffffff; color-scheme: light; }}
.sport-tabs {{ display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 1em; }}
.sport-tab-btn {{ border: 1px solid #d1d5db; border-radius: 999px; background: #ffffff; padding: 0.4em 0.8em; cursor: pointer; }}
.sport-tab-btn.is-active {{ background: #111827; color: #ffffff; border-color: #111827; }}
.sport-tab-panel {{ display: none; }}
.sport-tab-panel.is-active {{ display: block; }}
#hc-load-errors {{ display: none; color: #991b1b; background: #fee2e2; padding: 1em; margin: 1em 0;
  white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 0.9rem; border: 1px solid #fecaca; }}
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
</style>
</head>
<body>
<pre id="hc-load-errors"></pre>
""",
        "<h1>Market summary</h1>\n",
    ]
    meta = (
        f"<p>Date range: {html_lib.escape(start.isoformat())} – {html_lib.escape(end.isoformat())} "
        "(inclusive). <strong>Accounts</strong>: nxy, rn1, sovereign, tony, and default algo "
        f"users — consolidated as <strong>Algo</strong> for user_ids "
        f"{sorted(ALGO_USER_IDS)}.</p>\n"
    )

    tab_buttons: list[str] = []
    tab_panels: list[str] = []

    overall_inner = (
        '<h1 class="main-section">Overall</h1>\n'
        + run_analyses_for_df(
            df,
            overview_groupbys,
            bundles,
            include_risk=False,
            group_labels={None: "All (aggregated)", COL_SPORTS: "Sports"},
            section_heading_tag="h3",
        )
        + run_analyses_for_df(
            df,
            [COL_ACCOUNT],
            bundles,
            include_risk=True,
            group_labels={COL_ACCOUNT: "Risk and PnL by account (overall)"},
            section_heading_tag="h3",
        )
    )
    tab_buttons.append(
        '<button type="button" class="sport-tab-btn is-active" data-tab-target="sport-tab-panel-overall">'
        "Overall"
        "</button>"
    )
    tab_panels.append(
        '<section id="sport-tab-panel-overall" class="sport-tab-panel is-active">\n'
        + overall_inner
        + "</section>\n"
    )

    for sport in SPORTS:
        name = sport["name"]
        s_df = df[df[COL_SPORTS] == name]
        if s_df.empty:
            continue
        sport_key = _league_key(name)
        panel_id = f"sport-tab-panel-{sport_key}"
        sport_parts: list[str] = [
            f'<h1 class="main-section">{html_lib.escape(name)}</h1>\n',
        ]
        filter_col = COL_TIER if name == SOCCER_NAME else COL_LEAGUE
        filter_title = "Tier" if name == SOCCER_NAME else "League"
        sport_parts.append(
            run_analyses_for_df(
                s_df,
                [filter_col],
                bundles,
                include_risk=True,
                group_labels={
                    filter_col: f"{name} — summary by {filter_title.lower()}"
                },
                section_heading_tag="h3",
            )
        )
        sport_parts.append(
            run_analyses_for_df(
                s_df,
                [COL_ACCOUNT],
                bundles,
                include_risk=True,
                group_labels={COL_ACCOUNT: f"{name} — by account"},
                section_heading_tag="h3",
            )
        )

        keys = _sorted_slice_keys(
            s_df[filter_col], soccer_mode=(name == SOCCER_NAME)
        )
        for key in keys:
            sub = s_df[s_df[filter_col] == key]
            if sub.empty:
                continue
            disp = str(key) if key is not None else "N/A"
            sport_parts.append(
                run_analyses_for_df(
                    sub,
                    [COL_ACCOUNT],
                    bundles,
                    include_risk=True,
                    group_labels={
                        COL_ACCOUNT: f"{name} — {filter_title} {disp} — by account"
                    },
                    section_heading_tag="h3",
                )
            )

        tab_buttons.append(
            f'<button type="button" class="sport-tab-btn" '
            f'data-tab-target="{html_lib.escape(panel_id)}">{html_lib.escape(name)}</button>'
        )
        tab_panels.append(
            f'<section id="{html_lib.escape(panel_id)}" class="sport-tab-panel">\n'
            + "".join(sport_parts)
            + "</section>\n"
        )

    if tab_buttons:
        parts.append('<div class="sport-tabs">' + "".join(tab_buttons) + "</div>\n")
        parts.append(meta)
        parts.append("".join(tab_panels))
    else:
        parts.append(meta)

    parts.append(
        f'<script src="{hc_js}"></script>\n'
        f'<script src="{hc_exp}"></script>\n'
        f'<script src="{hc_ed}"></script>\n'
        f'<script src="{hc_a11y}"></script>\n'
        f'<script src="{grid_js}"></script>\n'
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
    if (!el) {
      showErr("Internal error: hc-bundles JSON script missing.");
      return;
    }
    if (typeof Highcharts === "undefined") {
      showErr(
        "Highcharts did not load. If you opened this file as file://, try serving the folder instead."
      );
      return;
    }
    var gridApi =
      typeof Grid !== "undefined" && typeof Grid.grid === "function" ? Grid.grid : null;
    if (!gridApi) {
      showErr("Highcharts Grid (Grid.grid) is not available after grid-lite.js.");
    }
    var bundles;
    try {
      bundles = JSON.parse(el.textContent);
    } catch (e) {
      showErr("Could not parse hc-bundles JSON: " + e.message);
      return;
    }
    bundles.forEach(function (b, idx) {
      try {
        var cEl = document.getElementById(b.chartId);
        if (!cEl) {
          showErr("Missing chart container #" + b.chartId);
        } else {
          Highcharts.chart(cEl, b.chartOptions);
        }
        if (b.riskChartId && b.riskChartOptions) {
          var rEl = document.getElementById(b.riskChartId);
          if (!rEl) {
            showErr("Missing risk chart container #" + b.riskChartId);
          } else {
            Highcharts.chart(rEl, b.riskChartOptions);
          }
        }
      } catch (e) {
        showErr("Highcharts chart #" + idx + " (" + b.chartId + "): " + e.message);
      }
      if (!gridApi) return;
      try {
        var gEl = document.getElementById(b.gridId);
        if (!gEl) {
          showErr("Missing grid container #" + b.gridId);
          return;
        }
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
        panels.forEach(function (p) {
          p.classList.toggle("is-active", p.id === target);
        });
        tabButtons.forEach(function (b) {
          b.classList.toggle("is-active", b === btn);
        });
      });
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", runInit);
  } else {
    runInit();
  }
})();
</script>
</body>
</html>
"""
    )
    return "".join(parts)


def build_report_html(
    df: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    account_tokens: Iterable[str],
) -> str:
    account_list = ", ".join(account_tokens)
    overall_groupbys: list[str | None] = [
        None,
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_STAGE,
        COL_ROLE,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]
    per_sport_groupbys: list[str | None] = [
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_LEAGUE,
        COL_STAGE,
        COL_ROLE,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]
    per_sport_groupbys_soccer: list[str | None] = [
        COL_SPORTS,
        COL_MKT_TYPE,
        COL_TIER,
        COL_STAGE,
        COL_ROLE,
        COL_PROBABILITY,
        COL_LONGSHOT,
    ]

    hc_js = html_lib.escape(HIGHCHARTS_JS)
    hc_exp = html_lib.escape(HIGHCHARTS_EXPORTING)
    hc_ed = html_lib.escape(HIGHCHARTS_EXPORT_DATA)
    hc_a11y = html_lib.escape(HIGHCHARTS_ACCESSIBILITY)
    grid_js = html_lib.escape(GRID_LITE_JS)
    grid_css = html_lib.escape(GRID_LITE_CSS)

    bundles: list[dict[str, Any]] = []
    parts: list[str] = [
        f"""<!DOCTYPE html>
<html lang="en" class="highcharts-light">
<head>
<meta charset="utf-8"/>
<title>Delta report</title>
<link rel="stylesheet" href="{grid_css}"/>
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
.sport-league-section[data-league]:not(.is-visible) {{ display: none; }}
.sport-tabs {{ display: flex; flex-wrap: wrap; gap: 0.5em; margin: 0.5em 0 1em; }}
.sport-tab-btn {{ border: 1px solid #d1d5db; border-radius: 999px; background: #ffffff; padding: 0.4em 0.8em; cursor: pointer; }}
.sport-tab-btn.is-active {{ background: #111827; color: #ffffff; border-color: #111827; }}
.sport-tab-panel {{ display: none; }}
.sport-tab-panel.is-active {{ display: block; }}
/*
 Grid Lite mounts in the light DOM (no shadow root). grid-lite.css uses prefers-color-scheme: dark on
 .hcg-container — override tokens under our host so summary tables stay white.
*/
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
        f"<h1>PNL / turnover report</h1>\n",
    ]
    report_meta_html = (
        f"<p>Date range: {html_lib.escape(start.isoformat())} – {html_lib.escape(end.isoformat())} "
        f"(inclusive). <strong>Report filter</strong> (accounts): {html_lib.escape(account_list)}.</p>\n"
    )

    tab_buttons: list[str] = []
    tab_panels: list[str] = []
    tab_buttons.append(
        '<button type="button" class="sport-tab-btn is-active" data-tab-target="sport-tab-panel-overall">'
        "Overall"
        "</button>"
    )
    tab_panels.append(
        '<section id="sport-tab-panel-overall" class="sport-tab-panel is-active">\n'
        '<h1 class="main-section">Overall (selected accounts)</h1>\n'
        + run_analyses_for_df(df, overall_groupbys, bundles)
        + "</section>\n"
    )
    for sport in SPORTS:
        name = sport["name"]
        s_df = df[df[COL_SPORTS] == name]
        if s_df.empty:
            continue
        sport_key = _league_key(name)
        panel_id = f"sport-tab-panel-{sport_key}"
        is_active = False
        tab_buttons.append(
            f'<button type="button" class="sport-tab-btn{" is-active" if is_active else ""}" '
            f'data-tab-target="{html_lib.escape(panel_id)}">{html_lib.escape(name)}</button>'
        )
        sport_parts: list[str] = [f'<h1 class="main-section">{html_lib.escape(name)}</h1>\n']
        filter_col = COL_TIER if name == SOCCER_NAME else COL_LEAGUE
        filter_label = "Tier filter" if name == SOCCER_NAME else "League filter"
        leagues = sorted({str(x) for x in s_df[filter_col].dropna().unique() if str(x).strip()})
        select_id = f"sport-league-filter-{sport_key}"
        sport_parts.append(
            f'<div class="sport-controls"><label for="{html_lib.escape(select_id)}">'
            f"{html_lib.escape(filter_label)}</label>"
            f'<select id="{html_lib.escape(select_id)}" class="sport-league-filter" '
            f'data-sport-key="{html_lib.escape(sport_key)}">'
            '<option value="all">All leagues</option>'
            + "".join(
                f'<option value="{html_lib.escape(_league_key(lg))}">{html_lib.escape(lg)}</option>'
                for lg in leagues
            )
            + "</select></div>\n"
        )
        sport_gbs = per_sport_groupbys_soccer if name == SOCCER_NAME else per_sport_groupbys
        sport_parts.append(
            f'<div class="sport-league-section is-visible" data-sport-key="{html_lib.escape(sport_key)}" '
            'data-league="all">'
        )
        sport_parts.append(run_analyses_for_df(s_df, sport_gbs, bundles))
        sport_parts.append("</div>\n")
        for league in leagues:
            league_key = _league_key(league)
            league_df = s_df[s_df[filter_col] == league]
            if league_df.empty:
                continue
            sport_parts.append(
                f'<div class="sport-league-section" data-sport-key="{html_lib.escape(sport_key)}" '
                f'data-league="{html_lib.escape(league_key)}">'
            )
            sport_parts.append(run_analyses_for_df(league_df, sport_gbs, bundles))
            sport_parts.append("</div>\n")
        tab_panels.append(
            f'<section id="{html_lib.escape(panel_id)}" class="sport-tab-panel{" is-active" if is_active else ""}">'
            + "".join(sport_parts)
            + "</section>\n"
        )

    if tab_buttons:
        parts.append('<div class="sport-tabs">' + "".join(tab_buttons) + "</div>\n")
        parts.append(report_meta_html)
        parts.append("".join(tab_panels))
    else:
        parts.append(report_meta_html)

    parts.append(
        f'<script src="{hc_js}"></script>\n'
        f'<script src="{hc_exp}"></script>\n'
        f'<script src="{hc_ed}"></script>\n'
        f'<script src="{hc_a11y}"></script>\n'
        f'<script src="{grid_js}"></script>\n'
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
  /** Resolve the summary <table> (Grid Lite uses light DOM; shadowRoot kept for forward compatibility). */
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
    if (!el) {
      showErr("Internal error: hc-bundles JSON script missing.");
      return;
    }
    if (typeof Highcharts === "undefined") {
      showErr(
        "Highcharts did not load. If you opened this file as file://, try serving the folder instead, e.g. " +
        "Python: python -m http.server 8080 then open http://localhost:8080/Reports/report.html — " +
        "or check the browser console / network tab for blocked scripts."
      );
      return;
    }
    var gridApi =
      typeof Grid !== "undefined" && typeof Grid.grid === "function" ? Grid.grid : null;
    if (!gridApi) {
      showErr("Highcharts Grid (Grid.grid) is not available after grid-lite.js — check the network tab.");
    }
    var bundles;
    try {
      bundles = JSON.parse(el.textContent);
    } catch (e) {
      showErr("Could not parse hc-bundles JSON: " + e.message);
      return;
    }
    bundles.forEach(function (b, idx) {
      try {
        var cEl = document.getElementById(b.chartId);
        if (!cEl) {
          showErr("Missing chart container #" + b.chartId);
        } else {
          Highcharts.chart(cEl, b.chartOptions);
        }
        if (b.riskChartId && b.riskChartOptions) {
          var rEl = document.getElementById(b.riskChartId);
          if (!rEl) {
            showErr("Missing risk chart container #" + b.riskChartId);
          } else {
            Highcharts.chart(rEl, b.riskChartOptions);
          }
        }
      } catch (e) {
        showErr("Highcharts chart #" + idx + " (" + b.chartId + "): " + e.message);
      }
      if (!gridApi) return;
      try {
        var gEl = document.getElementById(b.gridId);
        if (!gEl) {
          showErr("Missing grid container #" + b.gridId);
          return;
        }
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
        panels.forEach(function (p) {
          p.classList.toggle("is-active", p.id === target);
        });
        tabButtons.forEach(function (b) {
          b.classList.toggle("is-active", b === btn);
        });
      });
    });
    var filters = document.querySelectorAll(".sport-league-filter");
    filters.forEach(function (sel) {
      function applyLeagueFilter() {
        var sportKey = sel.getAttribute("data-sport-key");
        var selectedLeague = sel.value || "all";
        var blocks = document.querySelectorAll(
          '.sport-league-section[data-sport-key="' + sportKey + '"]'
        );
        blocks.forEach(function (blk) {
          var league = blk.getAttribute("data-league") || "all";
          var show = selectedLeague === "all" ? league === "all" : league === selectedLeague;
          if (show) {
            blk.classList.add("is-visible");
          } else {
            blk.classList.remove("is-visible");
          }
        });
      }
      sel.addEventListener("change", applyLeagueFilter);
      applyLeagueFilter();
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", runInit);
  } else {
    runInit();
  }
})();
</script>
</body>
</html>
"""
    )
    return "".join(parts)
