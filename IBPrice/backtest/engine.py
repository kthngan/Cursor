from __future__ import annotations

import heapq
from datetime import datetime

from .components import DataHandler, ExecutionHandler, OrderManager, Performance, Portfolio, VwapReversionZscoreStrategy
from .events import EventType, MarketEvent, QueueItem
from .models import BacktestConfig


class BacktestEngine:
    """
    Orchestrates event flow in strict chronological order.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_handler: DataHandler,
        strategy: VwapReversionZscoreStrategy,
        portfolio: Portfolio,
        order_manager: OrderManager,
        execution_handler: ExecutionHandler,
        performance: Performance,
    ) -> None:
        self.config = config
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.order_manager = order_manager
        self.execution_handler = execution_handler
        self.performance = performance
        self._queue: list[QueueItem] = []
        self._seq = 0

    def _priority_for_event(self, event_type: EventType) -> int:
        if event_type == EventType.MARKET:
            return 0
        if event_type == EventType.SIGNAL:
            return 1
        if event_type == EventType.ORDER:
            return 2
        return 3

    def _push_event(self, event: object, timestamp: datetime, event_type: EventType) -> None:
        self._seq += 1
        heapq.heappush(
            self._queue,
            QueueItem(
                timestamp=timestamp,
                priority=self._priority_for_event(event_type),
                seq=self._seq,
                event=event,
            ),
        )

    def _enqueue_market_events(self, events: list[MarketEvent]) -> None:
        for event in events:
            self._push_event(event, event.timestamp, EventType.MARKET)

    def run(self, start_date=None, end_date=None) -> None:
        days = self.data_handler.selected_days(start_date=start_date, end_date=end_date)
        for day in days:
            day_events = self.data_handler.load_day_events(day)
            if not day_events:
                continue
            self._enqueue_market_events(day_events)
            self._drain_queue()
            # End-of-day equity snapshot only.
            self.performance.record_equity(day_events[-1].timestamp, self.portfolio.state)

    def _drain_queue(self) -> None:
        while self._queue:
            item = heapq.heappop(self._queue)
            event = item.event

            if event.event_type == EventType.MARKET:
                self.portfolio.on_market(event)

                signal = self.strategy.on_market(event)
                if signal is not None:
                    self._push_event(signal, signal.timestamp, EventType.SIGNAL)

                exit_orders = self.order_manager.on_market_for_exit(event, self.portfolio.state)
                for exit_order in exit_orders:
                    self._push_event(exit_order, exit_order.timestamp, EventType.ORDER)

                fills = self.execution_handler.on_market(event, self.portfolio.state)
                for fill in fills:
                    self._push_event(fill, fill.timestamp, EventType.FILL)

            elif event.event_type == EventType.SIGNAL:
                order = self.order_manager.on_signal(event, self.portfolio.state)
                if order is not None:
                    self._push_event(order, order.timestamp, EventType.ORDER)

            elif event.event_type == EventType.ORDER:
                if event.order_kind == "ENTRY":
                    if self.portfolio.has_entry_threshold(event.side, event.entry_threshold):
                        continue
                elif event.order_kind == "EXIT":
                    if not self.portfolio.has_leg(event.leg_id):
                        continue
                self.execution_handler.on_order(event)

            elif event.event_type == EventType.FILL:
                trade_row = self.portfolio.on_fill(event)
                self.performance.record_trade(trade_row)

