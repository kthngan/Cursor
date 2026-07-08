from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FillModel(str, Enum):
    FULL_NEXT_QUOTE = "full_next_quote"
    PARTIAL_BY_QUOTE_SIZE = "partial_by_quote_size"


@dataclass
class BacktestConfig:
    symbol: str = "HSI"
    data_dir: str = "historicalData"
    index_file: str = "historicalData/hsi_index_daily.csv"
    output_dir: str = "backtest_output"
    vwap_threshold: float = 0.003
    max_vwap_threshold: float = 0.007
    threshold_step: float = 0.001
    zscore_entry: float = 1.0
    rolling_window: int = 5
    range_filter_mode: str = "upper_only"
    range_filter_low_pct: float = 0.2
    range_filter_high_pct: float = 0.8
    range_filter_lookback: int = 30
    range_use_normalized: bool = False
    start_time_hkt: str = "10:30:00"
    force_exit_time_hkt: str = "15:59:00"
    fill_model: FillModel = FillModel.FULL_NEXT_QUOTE
    trade_size: int = 1
    contract_multiplier: float = 50.0
    initial_cash: float = 1_000_000.0
    tick_size: float = 1.0
    threshold_qty_map_str: str = ""
    pt_multiplier: float = 1.5
    sl_multiplier: float = 0.4

    def threshold_ladder(self) -> list[float]:
        levels: list[float] = []
        cur = self.vwap_threshold
        # Rounded stepping avoids binary float drift in labels/comparisons.
        while cur <= self.max_vwap_threshold + 1e-12:
            levels.append(round(cur, 6))
            cur += self.threshold_step
        return levels

    def pt_pct(self, threshold: float) -> float:
        return threshold * self.pt_multiplier

    def sl_pct(self, threshold: float) -> float:
        return threshold * self.sl_multiplier

    def threshold_qty_map(self) -> dict[float, int]:
        out: dict[float, int] = {}
        text = (self.threshold_qty_map_str or "").strip()
        if not text:
            return out
        for item in text.split(","):
            part = item.strip()
            if not part or ":" not in part:
                continue
            key, value = part.split(":", 1)
            out[round(float(key.strip()), 6)] = int(value.strip())
        return out

    def quantity_for_threshold(self, threshold: float) -> int:
        qty_map = self.threshold_qty_map()
        key = round(threshold, 6)
        if key in qty_map:
            return qty_map[key]
        return self.trade_size


@dataclass
class CostModel:
    """
    Pluggable cost model.

    Defaults to zero costs while preserving extension points.
    """

    commission_per_contract: float = 25.0
    slippage_ticks: float = 0.0

    def commission(self, qty: int, _price: float) -> float:
        return abs(qty) * self.commission_per_contract

    def slippage_price_adjustment(self, side: int, tick_size: float) -> float:
        return side * self.slippage_ticks * tick_size

    def slippage_cost(self, qty: int, tick_size: float, multiplier: float) -> float:
        return abs(qty) * self.slippage_ticks * tick_size * multiplier


@dataclass
class OpenRiskState:
    side: int
    entry_price: float
    take_profit: float
    stop_loss: float


@dataclass
class OpenLeg:
    leg_id: str
    side: int
    entry_threshold: float
    entry_price: float
    qty_open: int
    take_profit: float
    stop_loss: float
    opened_at: datetime


@dataclass
class PortfolioState:
    cash_balance: float
    position_qty: int = 0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    commissions: float = 0.0
    slippage: float = 0.0
    equity: float = 0.0
    last_bid: float | None = None
    last_ask: float | None = None
    last_mid: float | None = None
    open_risk: OpenRiskState | None = None
    long_legs: list[OpenLeg] = field(default_factory=list)
    short_legs: list[OpenLeg] = field(default_factory=list)


@dataclass
class EquityPoint:
    timestamp: datetime
    cash_balance: float
    realized_pnl: float
    unrealized_pnl: float
    equity: float
    position_qty: int
    mid_price: float | None


@dataclass
class TradeLogRow:
    timestamp: datetime
    symbol: str
    order_id: str
    order_kind: str
    reason: str
    leg_id: str
    entry_threshold: float
    side: int
    quantity: int
    fill_price: float
    commission: float
    slippage_cost: float
    position_after: int
    avg_entry_after: float
    realized_pnl_after: float
    cash_after: float


@dataclass
class StrategyMinuteState:
    current_minute: datetime | None = None
    last_minute_mid: float | None = None
    returns_window: list[float] = field(default_factory=list)
    cumulative_trade_value: float = 0.0
    cumulative_trade_volume: float = 0.0
    current_vwap: float | None = None
    current_mid: float | None = None
    current_signal: int = 0

