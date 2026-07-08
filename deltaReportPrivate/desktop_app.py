"""Desktop app for private summary report using native Python plotting."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QDate, QObject, QThread, Qt, QUrl, Signal, Slot, QModelIndex
from PySide6.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

def _app_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.extend([Path.cwd(), Path(sys.executable).resolve().parent])
    else:
        roots.append(Path(__file__).resolve().parent)
    roots.append(Path.cwd())
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _ensure_repo_import_path() -> None:
    """
    Make deltaReportNew imports robust across launch contexts:
    - python desktop_app.py from repo
    - launched via desktop shortcut with different cwd
    - frozen executable runs
    """
    visited: set[Path] = set()
    for seed in _app_roots():
        for p in [seed, *seed.parents]:
            if p in visited:
                continue
            visited.add(p)
            if (p / "deltaReportNew" / "analytics.py").is_file():
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                return


def _import_summary_module():
    """Prefer summary.py from disk so frozen builds pick up updates without rebuild."""
    for root in _app_roots():
        summary_path = root / "summary.py"
        if not summary_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("summary", summary_path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["summary"] = mod
        return mod

    for root in _app_roots():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    import summary

    return summary


_ensure_repo_import_path()

from deltaReportNew.analytics import (
    COL_LEAGUE,
    COL_LONGSHOT,
    COL_MKT_TYPE,
    COL_PROBABILITY,
    COL_ROLE,
    COL_SPORTS,
    COL_STAGE,
    COL_TIER,
    SOCCER_NAME,
    aggregate_pnl_turnover,
    compute_daily_cumulative_pnl,
    compute_daily_total_risk,
    HIGHCHARTS_ACCESSIBILITY,
    HIGHCHARTS_EXPORT_DATA,
    HIGHCHARTS_EXPORTING,
    HIGHCHARTS_JS,
)

_summary = _import_summary_module()
COL_LAT_ACK = _summary.COL_LAT_ACK
COL_SIZE_FACTOR = _summary.COL_SIZE_FACTOR
COL_TIF = _summary.COL_TIF
COL_MAIN_BOOK = _summary.COL_MAIN_BOOK
COL_ROI_CLV = _summary.COL_ROI_CLV
COL_TAG = _summary.COL_TAG
COL_FROM_START = _summary.COL_FROM_START
COL_EDGE_CLV = _summary.COL_EDGE_CLV
load_private_frames = _summary.load_private_frames
roi_clv_bucket_from_raw = _summary.roi_clv_bucket_from_raw
from_start_bucket_from_raw = _summary.from_start_bucket_from_raw
edge_clv_bucket_from_raw = _summary.edge_clv_bucket_from_raw


def _parse_lat_ack_bounds(raw: Any) -> tuple[float, float] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() == "N/A":
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


def apply_lat_ack_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Re-map lat ACK labels to coarse buckets (idempotent if already combined)."""
    if df.empty or COL_LAT_ACK not in df.columns:
        return df
    out = df.copy()
    out[COL_LAT_ACK] = out[COL_LAT_ACK].map(lat_ack_bucket_from_raw)
    return out


def apply_roi_clv_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_ROI_CLV not in df.columns:
        return df
    out = df.copy()
    out[COL_ROI_CLV] = out[COL_ROI_CLV].map(roi_clv_bucket_from_raw)
    return out


def apply_from_start_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_FROM_START not in df.columns:
        return df
    out = df.copy()
    out[COL_FROM_START] = out[COL_FROM_START].map(from_start_bucket_from_raw)
    return out


def apply_edge_clv_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_EDGE_CLV not in df.columns:
        return df
    out = df.copy()
    out[COL_EDGE_CLV] = out[COL_EDGE_CLV].map(edge_clv_bucket_from_raw)
    return out


APP_VERSION = "2026-07-03d"


class MultiSelectComboBox(QComboBox):
    selectionChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(True)
            line_edit.setPlaceholderText("All")
        self._block_changes = False
        self.setModel(QStandardItemModel(self))
        self.view().pressed.connect(self._handle_pressed)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)

    def _handle_pressed(self, index: QModelIndex) -> None:
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new_state)
        self._update_display()
        if not self._block_changes:
            self.selectionChanged.emit()

    def set_items(self, items: list[str], *, preserve: list[str] | None = None) -> None:
        preserve_set = set(preserve if preserve is not None else self.selected_items())
        valid_preserve = {value for value in preserve_set if value in set(items)}
        self._block_changes = True
        model = self.model()
        if not isinstance(model, QStandardItemModel):
            model = QStandardItemModel(self)
            self.setModel(model)
        model.clear()
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(
                Qt.CheckState.Checked if text in valid_preserve else Qt.CheckState.Unchecked
            )
            model.appendRow(item)
        self._block_changes = False
        self._update_display()

    def selected_items(self) -> list[str]:
        model = self.model()
        if not isinstance(model, QStandardItemModel):
            return []
        selected: list[str] = []
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def selected_label(self) -> str:
        selected = self.selected_items()
        if not selected:
            return "All"
        if len(selected) == 1:
            return selected[0]
        return f"{len(selected)} selected"

    def _update_display(self) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setText(self.selected_label())


def _filter_label(selected: list[str]) -> str:
    return "All" if not selected else ", ".join(selected)


def _apply_multi_filter(df: pd.DataFrame, column: str, selected: list[str]) -> pd.DataFrame:
    if not selected or column not in df.columns:
        return df
    return df[df[column].isin(selected)]


FILTER_COMBO_WIDTH_SCALE = 0.6
FILTER_COMBO_DROPDOWN_WIDTH = 28


def _configure_filter_combo(combo: QComboBox) -> None:
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def _resize_filter_combo(combo: QComboBox, *, min_chars: int = 12) -> None:
    fm = combo.fontMetrics()
    widest = "All"
    model = combo.model()
    if isinstance(model, QStandardItemModel):
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue
            text = item.text()
            if fm.horizontalAdvance(text) > fm.horizontalAdvance(widest):
                widest = text
    else:
        for i in range(combo.count()):
            text = combo.itemText(i)
            if fm.horizontalAdvance(text) > fm.horizontalAdvance(widest):
                widest = text
    popup_width = max(
        fm.horizontalAdvance(widest) + 48,
        fm.horizontalAdvance("M" * min_chars) + 48,
        fm.horizontalAdvance("99 selected") + 48,
    )
    combo.view().setMinimumWidth(popup_width)
    button_width = max(
        int(popup_width * FILTER_COMBO_WIDTH_SCALE) + FILTER_COMBO_DROPDOWN_WIDTH,
        fm.horizontalAdvance("No group by") + FILTER_COMBO_DROPDOWN_WIDTH + 16,
        fm.horizontalAdvance("99 selected") + FILTER_COMBO_DROPDOWN_WIDTH + 16,
    )
    combo.setFixedWidth(button_width)


def _add_filter_row(parent_layout: QVBoxLayout, label: str, combo: QComboBox) -> None:
    row = QHBoxLayout()
    label_widget = QLabel(label)
    label_widget.setMinimumWidth(130)
    row.addWidget(label_widget)
    _configure_filter_combo(combo)
    row.addWidget(combo, 0)
    row.addStretch(1)
    parent_layout.addLayout(row)


POS_COLOR = "#4ade80"
NEG_COLOR = "#f87171"
PNL_COLOR = "#60a5fa"
PLOTLY_PALETTE = [
    "#60a5fa",
    "#34d399",
    "#fbbf24",
    "#f87171",
    "#a78bfa",
    "#2dd4bf",
    "#fb923c",
    "#22d3ee",
]

DARK_BG = "#111827"
DARK_SURFACE = "#1f2937"
DARK_BORDER = "#374151"
DARK_TEXT = "#e5e7eb"
DARK_TEXT_MUTED = "#9ca3af"

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {DARK_TEXT};
}}
QPushButton {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    background-color: {DARK_BORDER};
}}
QPushButton:disabled {{
    color: #6b7280;
}}
QComboBox, QDateEdit, QLineEdit {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox {{
    padding-right: {FILTER_COMBO_DROPDOWN_WIDTH + 4}px;
}}
QComboBox::drop-down, QDateEdit::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: {FILTER_COMBO_DROPDOWN_WIDTH}px;
    border-left: 1px solid {DARK_BORDER};
    background-color: {DARK_SURFACE};
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {DARK_TEXT_MUTED};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT};
    selection-background-color: {DARK_BORDER};
    border: 1px solid #4b5563;
}}
QCheckBox {{
    color: {DARK_TEXT};
}}
QTabWidget::pane {{
    border: 1px solid {DARK_BORDER};
    background-color: {DARK_BG};
}}
QTabBar::tab {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT_MUTED};
    border: 1px solid {DARK_BORDER};
    padding: 8px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {DARK_BORDER};
    color: {DARK_TEXT};
}}
QTableWidget {{
    background-color: {DARK_SURFACE};
    alternate-background-color: {DARK_BG};
    color: {DARK_TEXT};
    gridline-color: {DARK_BORDER};
    border: 1px solid {DARK_BORDER};
}}
QHeaderView::section {{
    background-color: {DARK_BORDER};
    color: {DARK_TEXT};
    border: 1px solid #4b5563;
    padding: 4px;
}}
QScrollArea {{
    background-color: {DARK_BG};
    border: none;
}}
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: {DARK_BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar:horizontal {{
    background: {DARK_BG};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background: {DARK_BORDER};
    border-radius: 4px;
    min-width: 24px;
}}
QLabel {{
    color: {DARK_TEXT};
}}
QCalendarWidget {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT};
}}
QCalendarWidget QWidget {{
    alternate-background-color: {DARK_BG};
}}
QCalendarWidget QAbstractItemView:enabled {{
    background-color: {DARK_SURFACE};
    color: {DARK_TEXT};
    selection-background-color: {DARK_BORDER};
}}
"""

HIGHCHARTS_ASSETS: tuple[tuple[str, str], ...] = (
    (HIGHCHARTS_JS, "highcharts.js"),
    (HIGHCHARTS_EXPORTING, "highcharts-exporting.js"),
    (HIGHCHARTS_EXPORT_DATA, "highcharts-export-data.js"),
    (HIGHCHARTS_ACCESSIBILITY, "highcharts-accessibility.js"),
)

OVERALL_GROUPBYS: list[str | None] = [
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
PER_SPORT_GROUPBYS: list[str | None] = [
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
PER_SPORT_GROUPBYS_SOCCER: list[str | None] = [
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


def script_dir() -> Path:
    if getattr(sys, "frozen", False):
        # Prefer current working directory if launched via shortcut with "Start in"
        # pointing to the project folder (contains json/downloadData.py).
        cwd = Path.cwd()
        if (cwd / "downloadData.py").is_file() and (cwd / "json").is_dir():
            return cwd
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def t_minus_1() -> dt.date:
    return dt.date.today() - dt.timedelta(days=1)


def _date_to_qdate(d: dt.date) -> QDate:
    return QDate(d.year, d.month, d.day)


class ReportWorker(QObject):
    log = Signal(str)
    done = Signal(object, int, str)
    failed = Signal(str)

    def __init__(
        self,
        base_dir: Path,
        start_date: dt.date,
        end_date: dt.date,
        run_download: bool,
        download_full_range: bool,
        headed: bool,
        timeout_ms: int,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.start_date = start_date
        self.end_date = end_date
        self.run_download = run_download
        self.download_full_range = download_full_range
        self.headed = headed
        self.timeout_ms = timeout_ms

    @Slot()
    def run(self) -> None:
        try:
            if self.run_download:
                t0 = time.perf_counter()
                self._run_download()
                self.log.emit(f"[timing] downloadData.py: {time.perf_counter() - t0:.2f}s")
            self.log.emit(
                f"Loading data from {self.start_date.isoformat()} to {self.end_date.isoformat()} ..."
            )
            t1 = time.perf_counter()
            df = load_private_frames(self.base_dir / "json", self.start_date, self.end_date)
            df = apply_lat_ack_buckets(df)
            df = apply_roi_clv_buckets(df)
            df = apply_from_start_buckets(df)
            df = apply_edge_clv_buckets(df)
            self.log.emit(f"[timing] load_private_frames: {time.perf_counter() - t1:.2f}s")
            if not df.empty and COL_LAT_ACK in df.columns:
                buckets = sorted(df[COL_LAT_ACK].dropna().unique(), key=str)
                self.log.emit(f"[lat-ack] buckets after combine: {buckets}")
            self.done.emit(df, len(df), self.end_date.isoformat())
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_download(self) -> None:
        py_cmd = sys.executable if not getattr(sys, "frozen", False) else "python"
        cmd = [
            py_cmd,
            str(self.base_dir / "downloadData.py"),
            "--start",
            self.start_date.isoformat(),
            "--end",
            self.end_date.isoformat(),
            "--timeout",
            str(self.timeout_ms),
        ]
        if self.download_full_range:
            cmd.append("--full-range")
        if self.headed:
            cmd.append("--headed")
        self.log.emit(f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.base_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log.emit(line.rstrip())
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"downloadData.py failed with exit code {rc}")


def _fmt_name(value: object) -> str:
    if value is None:
        return "N/A"
    s = str(value).strip()
    return s or "N/A"


def _group_heading(group_by: str | None) -> str:
    return "No group by" if group_by is None else str(group_by)


def _aggregate_trade_fixture_counts(df: pd.DataFrame, group_by: str | None) -> pd.DataFrame:
    if df.empty:
        cols = ["trade_count", "fixture_count"]
        if group_by:
            return pd.DataFrame(columns=[group_by] + cols)
        return pd.DataFrame(columns=cols)

    work = df.copy()
    for col in ("trade_count", "fixture_count"):
        if col not in work.columns:
            work[col] = 0
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    if group_by is None:
        return pd.DataFrame(
            {
                "trade_count": [int(work["trade_count"].sum())],
                "fixture_count": [int(work["fixture_count"].sum())],
            }
        )

    return (
        work.groupby(group_by, dropna=False, as_index=False)[["trade_count", "fixture_count"]]
        .sum()
        .astype({"trade_count": int, "fixture_count": int})
    )


def _query_date_to_utc_ms(d) -> int:
    ts = pd.Timestamp(d)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    ts = ts.normalize()
    return int(ts.timestamp() * 1000)


def _ensure_highcharts_assets(base_dir: Path, log_fn=None) -> dict[str, str]:
    assets_dir = base_dir / "Reports" / "web_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for url, fname in HIGHCHARTS_ASSETS:
        p = assets_dir / fname
        if not p.is_file() or p.stat().st_size == 0:
            try:
                if log_fn is not None:
                    log_fn(f"[assets] downloading {fname}")
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                if not data:
                    raise RuntimeError(f"Empty asset download: {url}")
                p.write_bytes(data)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        out[fname] = p.as_uri()
    return out


class SectionWidget(QWidget):
    def __init__(self, df: pd.DataFrame, group_by: str | None, log_fn=None) -> None:
        super().__init__()
        if group_by == COL_LAT_ACK:
            df = apply_lat_ack_buckets(df)
        elif group_by == COL_ROI_CLV:
            df = apply_roi_clv_buckets(df)
        elif group_by == COL_FROM_START:
            df = apply_from_start_buckets(df)
        elif group_by == COL_EDGE_CLV:
            df = apply_edge_clv_buckets(df)
        layout = QVBoxLayout(self)
        heading = _group_heading(group_by)
        t_total = time.perf_counter()
        title = QLabel(heading)
        title.setStyleSheet(
            f"font-weight: 700; font-size: 15px; margin-top: 10px; color: {DARK_TEXT};"
        )
        layout.addWidget(title)

        t_compute = time.perf_counter()
        summary = aggregate_pnl_turnover(df, group_by)
        counts = _aggregate_trade_fixture_counts(df, group_by)
        if group_by is None:
            summary = summary.assign(
                trade_count=counts["trade_count"].values,
                fixture_count=counts["fixture_count"].values,
            )
        else:
            summary = summary.merge(counts, on=group_by, how="left")
        daily = compute_daily_cumulative_pnl(df, group_by)
        daily_risk = compute_daily_total_risk(df, group_by)
        compute_secs = time.perf_counter() - t_compute

        t_plot = time.perf_counter()
        web = QWebEngineView()
        web.setMinimumHeight(620)
        web.setStyleSheet(f"background-color: {DARK_BG};")
        html = self._build_highcharts_html(daily, daily_risk, group_by, heading, log_fn=log_fn)
        charts_dir = script_dir() / "Reports" / "highcharts_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_path = charts_dir / f"chart_{uuid.uuid4().hex}.html"
        chart_path.write_text(html, encoding="utf-8")
        web.load(QUrl.fromLocalFile(str(chart_path)))
        if log_fn is not None:
            web.loadFinished.connect(
                lambda ok, p=str(chart_path): log_fn(
                    f"[highcharts] load {'ok' if ok else 'failed'}: {p}"
                )
            )
        layout.addWidget(web)
        plot_secs = time.perf_counter() - t_plot

        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setDefaultSectionSize(24)
        table.setMinimumHeight(220)
        t_table = time.perf_counter()
        self._fill_table(table, summary, group_by)
        table_secs = time.perf_counter() - t_table
        layout.addWidget(table)
        total_secs = time.perf_counter() - t_total
        if log_fn is not None:
            log_fn(
                f"[timing] section={heading} compute={compute_secs:.2f}s "
                f"plot={plot_secs:.2f}s table={table_secs:.2f}s total={total_secs:.2f}s"
            )

    def _build_highcharts_html(
        self,
        daily: pd.DataFrame,
        daily_risk: pd.DataFrame,
        group_by: str | None,
        heading: str,
        log_fn=None,
    ) -> str:
        assets = _ensure_highcharts_assets(script_dir(), log_fn=log_fn)
        color_by_name: dict[str, str] = {}
        pnl_series: list[dict] = []
        if daily.empty:
            pnl_series = []
        elif group_by is None:
            data = [
                [_query_date_to_utc_ms(r["queryDate"]), float(r["cumulative_pnl"])]
                for _, r in daily.iterrows()
            ]
            pnl_series.append({"name": "Cumulative PNL", "type": "line", "data": data, "color": PNL_COLOR})
        else:
            for idx, (name, g) in enumerate(daily.groupby(group_by, dropna=False)):
                disp_name = _fmt_name(name)
                color = PLOTLY_PALETTE[idx % len(PLOTLY_PALETTE)]
                color_by_name[disp_name] = color
                data = [
                    [_query_date_to_utc_ms(r["queryDate"]), float(r["cumulative_pnl"])]
                    for _, r in g.iterrows()
                ]
                pnl_series.append({"name": disp_name, "type": "line", "data": data, "color": color})

        risk_series: list[dict] = []
        if daily_risk.empty:
            risk_series = []
        elif group_by is None:
            data = [[_query_date_to_utc_ms(r["queryDate"]), float(r["daily_risk"])] for _, r in daily_risk.iterrows()]
            risk_series.append({"name": "Daily risk", "type": "column", "data": data, "color": PNL_COLOR})
        else:
            for name, g in daily_risk.groupby(group_by, dropna=False):
                disp_name = _fmt_name(name)
                bar_color = color_by_name.get(disp_name, PNL_COLOR)
                data = [[_query_date_to_utc_ms(r["queryDate"]), float(r["daily_risk"])] for _, r in g.iterrows()]
                risk_series.append(
                    {"name": disp_name, "type": "column", "data": data, "color": bar_color}
                )

        risk_can_stack = group_by is not None and len(risk_series) > 1
        risk_plot_options_absolute = {"column": {"grouping": True, "groupPadding": 0.08, "pointPadding": 0.02}}
        risk_plot_options_percent = {"column": {"stacking": "percent", "grouping": False}}
        risk_opts_absolute = {
            "chart": {"type": "column", "zoomType": "x", "height": 300},
            "title": {"text": f"Daily total risk (USD) ({heading})"},
            "xAxis": {"type": "datetime"},
            "yAxis": {"title": {"text": "Daily risk"}},
            "legend": {"enabled": bool(group_by)},
            "plotOptions": risk_plot_options_absolute,
            "series": risk_series,
            "credits": {"enabled": False},
            "exporting": {"enabled": True},
        }
        risk_opts_percent = {
            "chart": {"type": "column", "zoomType": "x", "height": 300},
            "title": {"text": f"Daily risk share (%) ({heading})"},
            "xAxis": {"type": "datetime"},
            "yAxis": {
                "title": {"text": "Share of daily risk"},
                "min": 0,
                "max": 100,
                "labels": {"format": "{value}%"},
            },
            "legend": {"enabled": bool(group_by)},
            "plotOptions": risk_plot_options_percent,
            "tooltip": {
                "shared": True,
                "pointFormat": (
                    '<span style="color:{series.color}">\\u25CF</span> {series.name}: '
                    "<b>{point.percentage:.1f}%</b> ({point.y:,.0f})<br/>"
                ),
            },
            "series": risk_series,
            "credits": {"enabled": False},
            "exporting": {"enabled": True},
        }
        pnl_opts = {
            "chart": {"type": "line", "zoomType": "x", "height": 300},
            "title": {"text": f"Cumulative PNL ({heading})"},
            "xAxis": {"type": "datetime"},
            "yAxis": {"title": {"text": "Cumulative PNL"}},
            "legend": {"enabled": bool(group_by)},
            "series": pnl_series,
            "credits": {"enabled": False},
            "exporting": {"enabled": True},
        }
        risk_controls_style = "" if risk_can_stack else "display:none;"
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="{assets['highcharts.js']}"></script>
  <script src="{assets['highcharts-exporting.js']}"></script>
  <script src="{assets['highcharts-export-data.js']}"></script>
  <script src="{assets['highcharts-accessibility.js']}"></script>
  <style>
    body {{ margin:0; font-family: system-ui, sans-serif; background:{DARK_BG}; color:{DARK_TEXT}; }}
    #pnl {{ width:100%; height:300px; }}
    #risk-controls {{
      margin-top:8px;
      padding:6px 8px 0;
      font-size:13px;
      color:{DARK_TEXT_MUTED};
      {risk_controls_style}
    }}
    #risk-controls label {{
      margin-right:14px;
      cursor:pointer;
      user-select:none;
    }}
    #risk {{ width:100%; height:300px; }}
    #err {{ color:#fca5a5; white-space:pre-wrap; padding:8px; }}
  </style>
</head>
<body>
  <div id="err"></div>
  <div id="pnl"></div>
  <div id="risk-controls">
  <span style="font-weight:600; margin-right:8px;">Daily risk view:</span>
  <label><input type="radio" name="riskMode" value="absolute" checked> Absolute (USD)</label>
  <label><input type="radio" name="riskMode" value="percent"> Stack 100%</label>
  </div>
  <div id="risk"></div>
  <script>
    (function() {{
      function showErr(msg) {{
        var el = document.getElementById('err');
        if (el) el.textContent = (el.textContent || '') + msg + "\\n";
      }}
      if (typeof Highcharts === 'undefined') {{
        showErr('Highcharts failed to load.');
        return;
      }}
      Highcharts.setOptions({{
        chart: {{ backgroundColor: '{DARK_BG}', plotBackgroundColor: 'transparent' }},
        title: {{ style: {{ color: '{DARK_TEXT}' }} }},
        xAxis: {{
          labels: {{ style: {{ color: '{DARK_TEXT_MUTED}' }} }},
          lineColor: '{DARK_BORDER}',
          tickColor: '{DARK_BORDER}',
          gridLineColor: '#1f2937',
        }},
        yAxis: {{
          title: {{ style: {{ color: '{DARK_TEXT_MUTED}' }} }},
          labels: {{ style: {{ color: '{DARK_TEXT_MUTED}' }} }},
          gridLineColor: '#1f2937',
        }},
        legend: {{
          itemStyle: {{ color: '{DARK_TEXT}' }},
          itemHoverStyle: {{ color: '#ffffff' }},
        }},
        tooltip: {{
          backgroundColor: '{DARK_SURFACE}',
          borderColor: '{DARK_BORDER}',
          style: {{ color: '{DARK_TEXT}' }},
        }},
      }});
      try {{
        Highcharts.chart('pnl', {json.dumps(pnl_opts, ensure_ascii=False)});
        var riskOptsAbsolute = {json.dumps(risk_opts_absolute, ensure_ascii=False)};
        var riskOptsPercent = {json.dumps(risk_opts_percent, ensure_ascii=False)};
        var riskChart = Highcharts.chart('risk', riskOptsAbsolute);
        document.querySelectorAll('input[name=riskMode]').forEach(function(el) {{
          el.addEventListener('change', function() {{
            if (!this.checked) return;
            var opts = this.value === 'percent' ? riskOptsPercent : riskOptsAbsolute;
            if (riskChart) riskChart.destroy();
            riskChart = Highcharts.chart('risk', opts);
          }});
        }});
      }} catch (e) {{
        showErr(String(e));
      }}
    }})();
  </script>
</body>
</html>"""

    def _fill_table(self, table: QTableWidget, summary: pd.DataFrame, group_by: str | None) -> None:
        if summary.empty:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        display = summary.copy()
        total_turnover = float(display["turnover"].sum())
        if total_turnover and not pd.isna(total_turnover):
            display["turnover_pct"] = display["turnover"] / total_turnover
        else:
            display["turnover_pct"] = pd.NA
        turnover_safe = display["turnover"].replace(0, pd.NA)
        display["return_on_turnover"] = display["pnl"] / turnover_safe
        cols = ([group_by] if group_by is not None else []) + [
            "pnl",
            "turnover",
            "turnover_pct",
            "trade_count",
            "fixture_count",
            "return_on_turnover",
        ]
        labels = ([group_by] if group_by is not None else []) + [
            "PNL",
            "Turnover",
            "Turnover %",
            "Trades",
            "Fixtures",
            "Return On Turnover %",
        ]
        display = display[cols]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([str(x) for x in labels])
        table.setRowCount(len(display))
        for ridx, row in enumerate(display.itertuples(index=False)):
            values = list(row)
            for cidx, value in enumerate(values):
                col_name = cols[cidx]
                if col_name in {"return_on_turnover", "turnover_pct"}:
                    text = "" if pd.isna(value) else f"{float(value) * 100.0:.2f}%"
                elif col_name in {"pnl", "turnover"}:
                    text = "" if pd.isna(value) else f"{float(value):,.0f}"
                elif col_name in {"trade_count", "fixture_count"}:
                    text = "" if pd.isna(value) else f"{int(value):,}"
                else:
                    text = _fmt_name(value)
                item = QTableWidgetItem(text)
                if col_name == "pnl" and not pd.isna(value):
                    num = float(value)
                    item.setForeground(QBrush(QColor(POS_COLOR if num >= 0 else NEG_COLOR)))
                elif col_name == "return_on_turnover" and not pd.isna(value):
                    num = float(value)
                    item.setForeground(QBrush(QColor(POS_COLOR if num >= 0 else NEG_COLOR)))
                table.setItem(ridx, cidx, item)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Private Summary Desktop ({APP_VERSION})")
        self.resize(1700, 1000)

        self.base_dir = script_dir()
        self.worker_thread: QThread | None = None
        self.worker: ReportWorker | None = None
        self.current_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.tab_specs: dict[str, tuple[pd.DataFrame, list[str | None]]] = {}
        self.tab_built_for_key: dict[str, str] = {}
        self.render_key = ""
        self._rendering_tab = False

        central = QWidget(self)
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        row = QHBoxLayout()
        root.addLayout(row)
        row.addWidget(QLabel("Start date:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        default_start = t_minus_1() - dt.timedelta(days=29)
        self.start_date_edit.setDate(QDate(default_start.year, default_start.month, default_start.day))
        row.addWidget(self.start_date_edit)
        row.addWidget(QLabel("End date:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        default_end = t_minus_1()
        self.end_date_edit.setDate(QDate(default_end.year, default_end.month, default_end.day))
        row.addWidget(self.end_date_edit)
        self.last_7d_btn = QPushButton("Last 7D")
        self.last_7d_btn.clicked.connect(self._set_range_last_7d)
        row.addWidget(self.last_7d_btn)
        self.last_30d_btn = QPushButton("Last 30D")
        self.last_30d_btn.clicked.connect(self._set_range_last_30d)
        row.addWidget(self.last_30d_btn)
        self.mtd_btn = QPushButton("MTD")
        self.mtd_btn.clicked.connect(self._set_range_mtd)
        row.addWidget(self.mtd_btn)
        self.headed_check = QCheckBox("Show browser while downloading")
        row.addWidget(self.headed_check)
        self.refresh_btn = QPushButton("Refresh to T-1")
        self.refresh_btn.clicked.connect(self.on_refresh_clicked)
        row.addWidget(self.refresh_btn)
        self.download_overwrite_btn = QPushButton("Download Overwrite Range")
        self.download_overwrite_btn.clicked.connect(self.on_download_overwrite_clicked)
        row.addWidget(self.download_overwrite_btn)
        self.reload_btn = QPushButton("Reload from cache")
        self.reload_btn.clicked.connect(self.on_reload_clicked)
        row.addWidget(self.reload_btn)
        row.addStretch(1)

        filters_box = QVBoxLayout()
        root.addLayout(filters_box)
        self.sport_filter = MultiSelectComboBox()
        self.sport_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Sport filter:", self.sport_filter)
        self.league_filter = MultiSelectComboBox()
        self.league_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "League filter:", self.league_filter)
        self.tif_filter = MultiSelectComboBox()
        self.tif_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "TIF filter:", self.tif_filter)
        self.stage_filter = MultiSelectComboBox()
        self.stage_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Stage filter:", self.stage_filter)
        self.role_filter = MultiSelectComboBox()
        self.role_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Role filter:", self.role_filter)
        self.main_book_filter = MultiSelectComboBox()
        self.main_book_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Main Book filter:", self.main_book_filter)
        self.tag_filter = MultiSelectComboBox()
        self.tag_filter.selectionChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Tag filter:", self.tag_filter)
        self.section_filter = QComboBox()
        self.section_filter.currentIndexChanged.connect(self._on_filter_change)
        _add_filter_row(filters_box, "Section filter:", self.section_filter)
        load_row = QHBoxLayout()
        load_row.addStretch(1)
        self.load_view_btn = QPushButton("Load View")
        self.load_view_btn.clicked.connect(self._load_selected_view)
        load_row.addWidget(self.load_view_btn)
        filters_box.addLayout(load_row)

        self.status_label = QLabel("Ready")
        root.addWidget(self.status_label)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._render_current_tab)
        root.addWidget(self.tabs, 1)

        self._refresh_end_date_label()
        self.on_reload_clicked()

    def _set_date_range(self, start: dt.date, end: dt.date) -> None:
        self.start_date_edit.setDate(_date_to_qdate(start))
        self.end_date_edit.setDate(_date_to_qdate(end))

    @Slot()
    def _set_range_last_7d(self) -> None:
        end = t_minus_1()
        self._set_date_range(end - dt.timedelta(days=6), end)

    @Slot()
    def _set_range_last_30d(self) -> None:
        end = t_minus_1()
        self._set_date_range(end - dt.timedelta(days=29), end)

    @Slot()
    def _set_range_mtd(self) -> None:
        end = t_minus_1()
        self._set_date_range(end.replace(day=1), end)

    def _refresh_end_date_label(self) -> None:
        end = t_minus_1()
        self.end_date_edit.setDate(QDate(end.year, end.month, end.day))

    def _append_log(self, text: str) -> None:
        # Keep logs available in stdout without occupying UI space.
        print(text, flush=True)

    def _set_busy(self, busy: bool) -> None:
        self.refresh_btn.setEnabled(not busy)
        self.download_overwrite_btn.setEnabled(not busy)
        self.reload_btn.setEnabled(not busy)
        self.start_date_edit.setEnabled(not busy)
        self.end_date_edit.setEnabled(not busy)
        self.last_7d_btn.setEnabled(not busy)
        self.last_30d_btn.setEnabled(not busy)
        self.mtd_btn.setEnabled(not busy)
        self.headed_check.setEnabled(not busy)
        self.load_view_btn.setEnabled(not busy)
        self.sport_filter.setEnabled(not busy)
        self.league_filter.setEnabled(not busy)
        self.tif_filter.setEnabled(not busy)
        self.stage_filter.setEnabled(not busy)
        self.role_filter.setEnabled(not busy)
        self.main_book_filter.setEnabled(not busy)
        self.tag_filter.setEnabled(not busy)
        self.section_filter.setEnabled(not busy)

    def _selected_start_date(self) -> dt.date:
        qd = self.start_date_edit.date()
        return dt.date(qd.year(), qd.month(), qd.day())

    def _selected_end_date(self) -> dt.date:
        qd = self.end_date_edit.date()
        return dt.date(qd.year(), qd.month(), qd.day())

    def _start_worker(self, run_download: bool, *, full_range_download: bool = False) -> None:
        if self.worker_thread is not None:
            QMessageBox.warning(self, "Busy", "A job is already running.")
            return
        start_date = self._selected_start_date()
        end_date = self._selected_end_date()
        if start_date > end_date:
            QMessageBox.warning(self, "Invalid date range", "Start date must be <= end date.")
            return
        self._set_busy(True)
        self.status_label.setText("Running ...")
        self._append_log(
            f"\n=== {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{'Refresh' if run_download else 'Reload'} {start_date.isoformat()}..{end_date.isoformat()} ==="
        )
        worker = ReportWorker(
            self.base_dir,
            start_date,
            end_date,
            run_download,
            full_range_download,
            self.headed_check.isChecked(),
            120_000,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.done.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_state)
        self.worker_thread = thread
        self.worker = worker
        thread.start()

    @Slot()
    def on_refresh_clicked(self) -> None:
        end = t_minus_1()
        self.end_date_edit.setDate(QDate(end.year, end.month, end.day))
        self._start_worker(run_download=True, full_range_download=False)

    @Slot()
    def on_download_overwrite_clicked(self) -> None:
        self._start_worker(run_download=True, full_range_download=True)

    @Slot()
    def on_reload_clicked(self) -> None:
        self._start_worker(run_download=False, full_range_download=False)

    @Slot(object, int, str)
    def _on_worker_done(self, df: object, row_count: int, end_date_iso: str) -> None:
        self.current_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        self._populate_filters()
        self._on_filter_change()
        self.status_label.setText(f"Loaded {row_count} rows through {end_date_iso}")
        self._append_log(f"Done. Loaded {row_count} rows. Choose filters, then click Load View.")
        self._set_busy(False)

    @Slot(str)
    def _on_worker_failed(self, details: str) -> None:
        self._append_log(details)
        self.status_label.setText("Failed")
        self._set_busy(False)
        QMessageBox.critical(self, "Task failed", details[-2000:])

    @Slot()
    def _clear_worker_state(self) -> None:
        self.worker_thread = None
        self.worker = None

    def _league_values_for_sports(self, sports: list[str]) -> list[str]:
        df = self.current_df
        if sports and not df.empty:
            df = df[df[COL_SPORTS].isin(sports)]
        return sorted(
            {
                str(x).strip()
                for x in df.get(COL_LEAGUE, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )

    def _populate_league_filter(self, *, preserve: bool = True) -> None:
        prev_league = self.league_filter.selected_items() if preserve else []
        sports = self.sport_filter.selected_items()
        leagues = self._league_values_for_sports(sports)
        self.league_filter.set_items(leagues, preserve=prev_league if preserve else [])
        _resize_filter_combo(self.league_filter, min_chars=18)

    def _populate_filters(self) -> None:
        prev_sport = self.sport_filter.selected_items()
        prev_tif = self.tif_filter.selected_items()
        prev_stage = self.stage_filter.selected_items()
        prev_role = self.role_filter.selected_items()
        prev_main_book = self.main_book_filter.selected_items()
        prev_tag = self.tag_filter.selected_items()
        sports = sorted(
            {
                str(x).strip()
                for x in self.current_df.get(COL_SPORTS, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )
        tifs = sorted(
            {str(x).strip() for x in self.current_df.get(COL_TIF, pd.Series(dtype=str)).dropna() if str(x).strip()}
        )
        stages = sorted(
            {
                str(x).strip()
                for x in self.current_df.get(COL_STAGE, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )
        roles = sorted(
            {
                str(x).strip()
                for x in self.current_df.get(COL_ROLE, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )
        main_books = sorted(
            {
                str(x).strip()
                for x in self.current_df.get(COL_MAIN_BOOK, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )
        tags = sorted(
            {
                str(x).strip()
                for x in self.current_df.get(COL_TAG, pd.Series(dtype=str)).dropna()
                if str(x).strip()
            }
        )
        self.sport_filter.set_items(sports, preserve=prev_sport)
        self.tif_filter.set_items(tifs, preserve=prev_tif)
        self.stage_filter.set_items(stages, preserve=prev_stage)
        self.role_filter.set_items(roles, preserve=prev_role)
        self.main_book_filter.set_items(main_books, preserve=prev_main_book)
        self.tag_filter.set_items(tags, preserve=prev_tag)
        self._populate_league_filter(preserve=True)
        for combo, min_chars in (
            (self.sport_filter, 12),
            (self.tif_filter, 8),
            (self.stage_filter, 8),
            (self.role_filter, 8),
            (self.main_book_filter, 24),
            (self.tag_filter, 16),
        ):
            _resize_filter_combo(combo, min_chars=min_chars)
        self._populate_section_filter()

    def _groupbys_for_selected_sport(self) -> list[str | None]:
        sports = self.sport_filter.selected_items()
        if len(sports) == 1:
            sport = sports[0]
            return PER_SPORT_GROUPBYS_SOCCER if sport == SOCCER_NAME else PER_SPORT_GROUPBYS
        return OVERALL_GROUPBYS

    def _populate_section_filter(self) -> None:
        prev_text = self.section_filter.currentText()
        section_items = self._groupbys_for_selected_sport()
        self.section_filter.blockSignals(True)
        self.section_filter.clear()
        for gb in section_items:
            self.section_filter.addItem(_group_heading(gb), userData=gb)
        if prev_text:
            idx = self.section_filter.findText(prev_text)
            if idx >= 0:
                self.section_filter.setCurrentIndex(idx)
        self.section_filter.blockSignals(False)
        _resize_filter_combo(self.section_filter, min_chars=22)

    @Slot()
    def _on_filter_change(self) -> None:
        if self.current_df.empty:
            self.tabs.clear()
            return
        self._populate_league_filter(preserve=True)
        self._populate_section_filter()
        self.tabs.clear()
        placeholder = QWidget()
        pl = QVBoxLayout(placeholder)
        pl.addWidget(QLabel("Filters changed. Click 'Load View' to render one selected section."))
        pl.addStretch(1)
        self.tabs.addTab(placeholder, "Ready")

    @Slot()
    def _load_selected_view(self) -> None:
        t0 = time.perf_counter()
        if self.current_df.empty:
            QMessageBox.information(self, "No data", "Load data first with Reload/Refresh.")
            return
        sports = self.sport_filter.selected_items()
        leagues = self.league_filter.selected_items()
        tifs = self.tif_filter.selected_items()
        stages = self.stage_filter.selected_items()
        roles = self.role_filter.selected_items()
        main_books = self.main_book_filter.selected_items()
        tags = self.tag_filter.selected_items()
        df = self.current_df
        df = _apply_multi_filter(df, COL_SPORTS, sports)
        df = _apply_multi_filter(df, COL_LEAGUE, leagues)
        df = _apply_multi_filter(df, COL_TIF, tifs)
        df = _apply_multi_filter(df, COL_STAGE, stages)
        df = _apply_multi_filter(df, COL_ROLE, roles)
        df = _apply_multi_filter(df, COL_MAIN_BOOK, main_books)
        df = _apply_multi_filter(df, COL_TAG, tags)
        self.filtered_df = df
        selected_section = self.section_filter.currentData()
        if df.empty:
            QMessageBox.information(
                self,
                "No matching rows",
                f"No rows for Sport={_filter_label(sports)} / League={_filter_label(leagues)} / "
                f"TIF={_filter_label(tifs)} / Stage={_filter_label(stages)} / "
                f"Role={_filter_label(roles)} / Main Book={_filter_label(main_books)} / "
                f"Tag={_filter_label(tags)} between "
                f"{self._selected_start_date().isoformat()} and "
                f"{self._selected_end_date().isoformat()}.",
            )
        self.render_key = (
            f"rows={len(self.filtered_df)};sport={_filter_label(sports)};"
            f"league={_filter_label(leagues)};tif={_filter_label(tifs)};"
            f"stage={_filter_label(stages)};role={_filter_label(roles)};"
            f"main_book={_filter_label(main_books)};tag={_filter_label(tags)};"
            f"section={_group_heading(selected_section)}"
        )
        self.tab_specs.clear()
        self.tab_built_for_key.clear()
        self.tabs.clear()

        if len(sports) == 1:
            view_name = sports[0]
        elif len(sports) > 1:
            view_name = "Multi-sport"
        else:
            view_name = "Overall"
        self.tab_specs[view_name] = (self.filtered_df, [selected_section])

        placeholder = QWidget()
        pl = QVBoxLayout(placeholder)
        pl.addWidget(QLabel("Loading section view ..."))
        pl.addStretch(1)
        self.tabs.addTab(placeholder, view_name)
        self._append_log(f"[timing] build tab placeholders: {time.perf_counter() - t0:.2f}s")
        self._append_log(
            f"[filter] sport={_filter_label(sports)}, league={_filter_label(leagues)}, "
            f"tif={_filter_label(tifs)}, stage={_filter_label(stages)}, "
            f"role={_filter_label(roles)}, main_book={_filter_label(main_books)}, "
            f"tag={_filter_label(tags)}, "
            f"section={_group_heading(selected_section)}, rows={len(self.filtered_df)}"
        )
        self._set_busy(True)
        self.status_label.setText("Rendering section ...")
        self._render_current_tab(self.tabs.currentIndex())

    @Slot(int)
    def _render_current_tab(self, idx: int) -> None:
        if self._rendering_tab:
            return
        if idx < 0 or idx >= self.tabs.count():
            return
        self._rendering_tab = True
        try:
            t0 = time.perf_counter()
            tab_name = self.tabs.tabText(idx)
            spec = self.tab_specs.get(tab_name)
            if spec is None:
                self._set_busy(False)
                return
            if self.tab_built_for_key.get(tab_name) == self.render_key:
                self._append_log(f"[timing] render tab={tab_name}: reused cached widget")
                self._set_busy(False)
                return
            df_tab, groupbys = spec
            self.tab_built_for_key[tab_name] = self.render_key
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            layout = QVBoxLayout(content)
            if df_tab.empty:
                layout.addWidget(QLabel("No rows for this tab/filter."))
                layout.addStretch(1)
                self._set_busy(False)
                self.status_label.setText("No rows for selected filters")
            else:
                gb = groupbys[0] if groupbys else None
                try:
                    layout.addWidget(SectionWidget(df_tab, gb, log_fn=self._append_log))
                except Exception:
                    self._append_log(traceback.format_exc())
                    layout.addWidget(
                        QLabel("Failed to render this section. See console log for details.")
                    )
                    self.status_label.setText(f"Failed to render section {_group_heading(gb)}")
                else:
                    self._append_log(f"[timing] render tab={tab_name}: {time.perf_counter() - t0:.2f}s")
                    self.status_label.setText(f"Rendered section {_group_heading(gb)} ({tab_name})")
                layout.addStretch(1)
                self._set_busy(False)
            scroll.setWidget(content)
            self.tabs.blockSignals(True)
            try:
                self.tabs.removeTab(idx)
                self.tabs.insertTab(idx, scroll, tab_name)
                self.tabs.setCurrentIndex(idx)
            finally:
                self.tabs.blockSignals(False)
            if df_tab.empty:
                self._append_log(f"[timing] render tab={tab_name}: {time.perf_counter() - t0:.2f}s")
        finally:
            self._rendering_tab = False


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
