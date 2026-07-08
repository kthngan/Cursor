from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class MarketEventType(str, Enum):
    BIDASK = "TICK_BIDASK"
    TRADE = "TICK_TRADE"


@dataclass(order=True)
class QueueItem:
    """
    Wrapper for priority queue processing.

    Ordering is strictly chronological, then by priority, then by sequence id.
    """

    timestamp: datetime
    priority: int
    seq: int
    event: object = field(compare=False)


@dataclass
class MarketEvent:
    event_type: EventType
    timestamp: datetime
    symbol: str
    market_event_type: MarketEventType
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    trade_price: float | None = None
    trade_size: float | None = None
    source_file: str | None = None


@dataclass
class SignalEvent:
    event_type: EventType
    timestamp: datetime
    symbol: str
    signal: int
    vwap_dev: float | None
    rolling_zscore: float | None


@dataclass
class OrderEvent:
    event_type: EventType
    timestamp: datetime
    symbol: str
    side: int
    quantity: int
    order_kind: str
    reason: str
    leg_id: str = ""
    entry_threshold: float = 0.0
    pt_pct: float | None = None
    sl_pct: float | None = None
    remaining_qty: int = 0
    order_id: str = ""

    def __post_init__(self) -> None:
        if self.remaining_qty <= 0:
            self.remaining_qty = self.quantity


@dataclass
class FillEvent:
    event_type: EventType
    timestamp: datetime
    symbol: str
    side: int
    quantity: int
    fill_price: float
    commission: float
    slippage_cost: float
    order_id: str
    order_kind: str
    reason: str
    leg_id: str = ""
    entry_threshold: float = 0.0
    pt_pct: float | None = None
    sl_pct: float | None = None

