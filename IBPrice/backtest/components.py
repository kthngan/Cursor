from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .events import EventType, FillEvent, MarketEvent, MarketEventType, OrderEvent, SignalEvent
from .models import (
    BacktestConfig,
    CostModel,
    EquityPoint,
    FillModel,
    OpenLeg,
    PortfolioState,
    StrategyMinuteState,
    TradeLogRow,
)


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _parse_date(text: str) -> datetime.date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _float_or_none(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    return float(t)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile on empty list")
    xs = sorted(values)
    k = (len(xs) - 1) * percentile
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _minute_floor(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


class DataHandler:
    """
    Reads tick CSV files and enforces day-level eligibility filters.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.index_file = Path(config.index_file)
        self._index_ranges, self._index_ranges_norm = self._load_index_ranges()

    def _load_index_ranges(self) -> tuple[dict[datetime.date, float], dict[datetime.date, float]]:
        out_raw: dict[datetime.date, float] = {}
        out_norm: dict[datetime.date, float] = {}
        with self.index_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                day = _parse_date(row["date"])
                daily_range = float(row["high"]) - float(row["low"])
                out_raw[day] = daily_range
                close = float(row["close"])
                out_norm[day] = (daily_range / close) if close != 0 else daily_range
        return out_raw, out_norm

    def _all_tick_days(self) -> list[datetime.date]:
        days: list[datetime.date] = []
        for p in sorted(self.data_dir.glob("*.csv")):
            if p.name == "hsi_index_daily.csv":
                continue
            try:
                day = _parse_date(p.stem)
            except ValueError:
                continue
            days.append(day)
        return days

    def _is_day_eligible(self, day: datetime.date, index_days_sorted: list[datetime.date]) -> bool:
        """
        Skip rule:
        - previous business day range >= 80th percentile of prior 30 business day ranges
          excluding the previous business day itself.
        """

        mode = (self.config.range_filter_mode or "upper_only").lower()
        if mode == "none":
            return True

        if day not in index_days_sorted:
            return True

        day_idx = index_days_sorted.index(day)
        if day_idx <= 0:
            return True

        prev_day = index_days_sorted[day_idx - 1]
        range_map = self._index_ranges_norm if self.config.range_use_normalized else self._index_ranges
        prev_range = range_map.get(prev_day)
        if prev_range is None:
            return True

        history_candidates = index_days_sorted[: day_idx - 1]
        history_days = history_candidates[-max(1, self.config.range_filter_lookback) :]
        history_ranges = [range_map[d] for d in history_days if d in range_map]
        if len(history_ranges) < 5:
            return True

        high_q = _percentile(history_ranges, self.config.range_filter_high_pct)
        if mode == "upper_only":
            return prev_range < high_q
        if mode == "two_sided":
            low_q = _percentile(history_ranges, self.config.range_filter_low_pct)
            return low_q <= prev_range < high_q
        return True

    def selected_days(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> list[datetime.date]:
        tick_days = self._all_tick_days()
        if start_date is not None:
            tick_days = [d for d in tick_days if d >= start_date]
        if end_date is not None:
            tick_days = [d for d in tick_days if d <= end_date]

        index_days_sorted = sorted(self._index_ranges.keys())
        return [d for d in tick_days if self._is_day_eligible(d, index_days_sorted)]

    def day_file(self, day: datetime.date) -> Path:
        return self.data_dir / f"{day:%Y-%m-%d}.csv"

    def load_day_events(self, day: datetime.date) -> list[MarketEvent]:
        path = self.day_file(day)
        events: list[MarketEvent] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row["timestamp_hkt"])
                ev_type = MarketEventType(row["event_type"])
                event = MarketEvent(
                    event_type=EventType.MARKET,
                    timestamp=ts,
                    symbol=self.config.symbol,
                    market_event_type=ev_type,
                    bid=_float_or_none(row.get("bid_price")),
                    ask=_float_or_none(row.get("offer_price")),
                    bid_size=_float_or_none(row.get("bid_size")),
                    ask_size=_float_or_none(row.get("ask_size")),
                    trade_price=_float_or_none(row.get("last_price")),
                    trade_size=_float_or_none(row.get("last_size")),
                    source_file=path.name,
                )
                events.append(event)

        def sort_key(ev: MarketEvent) -> tuple[datetime, int]:
            # Process quote updates before trade updates at same timestamp.
            pr = 0 if ev.market_event_type == MarketEventType.BIDASK else 1
            return (ev.timestamp, pr)

        events.sort(key=sort_key)
        return events


class VwapReversionZscoreStrategy:
    """
    Minimal working event-driven strategy implementation.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.state = StrategyMinuteState()

    def _update_mid(self, event: MarketEvent) -> None:
        if event.bid is None or event.ask is None:
            return
        self.state.current_mid = 0.5 * (event.bid + event.ask)

    def _update_vwap(self, event: MarketEvent) -> None:
        if event.trade_price is None or event.trade_size is None:
            return
        if event.trade_size <= 0:
            return
        self.state.cumulative_trade_value += event.trade_price * event.trade_size
        self.state.cumulative_trade_volume += event.trade_size
        if self.state.cumulative_trade_volume > 0:
            self.state.current_vwap = (
                self.state.cumulative_trade_value / self.state.cumulative_trade_volume
            )

    def _on_new_minute(self, ts: datetime) -> tuple[float | None, float | None, int]:
        if self.state.current_mid is None:
            return None, None, 0
        if self.state.current_vwap is None or self.state.current_vwap == 0:
            return None, None, 0

        if self.state.last_minute_mid is None:
            self.state.last_minute_mid = self.state.current_mid
            return None, None, 0

        minute_ret = (self.state.current_mid / self.state.last_minute_mid) - 1.0
        self.state.last_minute_mid = self.state.current_mid
        self.state.returns_window.append(minute_ret)
        if len(self.state.returns_window) > self.config.rolling_window:
            self.state.returns_window.pop(0)

        if len(self.state.returns_window) < self.config.rolling_window:
            return None, None, 0

        mu = statistics.fmean(self.state.returns_window)
        sigma = statistics.pstdev(self.state.returns_window)
        if sigma <= 0:
            zscore = 0.0
        else:
            zscore = (minute_ret - mu) / sigma

        vwap_dev = (self.state.current_mid - self.state.current_vwap) / self.state.current_vwap
        if vwap_dev >= self.config.vwap_threshold and zscore >= self.config.zscore_entry:
            signal = -1
        elif vwap_dev <= -self.config.vwap_threshold and zscore <= -self.config.zscore_entry:
            signal = 1
        else:
            signal = 0

        return vwap_dev, zscore, signal

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        if event.market_event_type == MarketEventType.BIDASK:
            self._update_mid(event)
        elif event.market_event_type == MarketEventType.TRADE:
            self._update_vwap(event)

        minute = _minute_floor(event.timestamp)
        if self.state.current_minute is None:
            self.state.current_minute = minute
            return None
        if minute == self.state.current_minute:
            return None

        self.state.current_minute = minute
        vwap_dev, zscore, signal = self._on_new_minute(event.timestamp)
        self.state.current_signal = signal
        if vwap_dev is None or zscore is None:
            return None
        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=event.timestamp,
            symbol=event.symbol,
            signal=signal,
            vwap_dev=vwap_dev,
            rolling_zscore=zscore,
        )


class OrderManager:
    """
    Converts signals and quote-based risk checks into orders.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._order_counter = 0
        self._leg_counter = 0
        self.start_time = datetime.strptime(config.start_time_hkt, "%H:%M:%S").time()
        self.force_exit_time = datetime.strptime(config.force_exit_time_hkt, "%H:%M:%S").time()

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"O{self._order_counter:08d}"

    def _next_leg_id(self) -> str:
        self._leg_counter += 1
        return f"L{self._leg_counter:08d}"

    def _can_trade(self, ts: datetime) -> bool:
        return ts.time() > self.start_time

    def _can_enter(self, ts: datetime) -> bool:
        return self._can_trade(ts) and ts.time() < self.force_exit_time

    def _is_force_exit_time(self, ts: datetime) -> bool:
        return ts.time() >= self.force_exit_time

    def _next_threshold_for_side(self, side: int, portfolio: PortfolioState) -> float | None:
        ladder = self.config.threshold_ladder()
        side_legs = portfolio.long_legs if side > 0 else portfolio.short_legs
        active_thresholds = sorted({round(x.entry_threshold, 6) for x in side_legs})
        if not active_thresholds:
            return ladder[0] if ladder else None
        highest = active_thresholds[-1]
        for lv in ladder:
            if lv > highest + 1e-12:
                return lv
        return None

    def on_signal(self, signal: SignalEvent, portfolio: PortfolioState) -> OrderEvent | None:
        if not self._can_enter(signal.timestamp):
            return None
        if signal.signal == 0 or signal.vwap_dev is None:
            return None

        side = signal.signal
        next_threshold = self._next_threshold_for_side(side, portfolio)
        if next_threshold is None:
            return None

        abs_dev = abs(signal.vwap_dev)
        if abs_dev + 1e-12 < next_threshold:
            return None
        qty = self.config.quantity_for_threshold(next_threshold)
        if qty <= 0:
            return None

        return OrderEvent(
            event_type=EventType.ORDER,
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            quantity=qty,
            order_kind="ENTRY",
            reason="signal_entry",
            leg_id=self._next_leg_id(),
            entry_threshold=next_threshold,
            pt_pct=self.config.pt_pct(next_threshold),
            sl_pct=self.config.sl_pct(next_threshold),
            order_id=self._next_order_id(),
        )

    def _exit_order_for_leg(
        self,
        *,
        event: MarketEvent,
        leg: OpenLeg,
        side: int,
        reason: str,
    ) -> OrderEvent:
        return OrderEvent(
            event_type=EventType.ORDER,
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=side,
            quantity=leg.qty_open,
            order_kind="EXIT",
            reason=reason,
            leg_id=leg.leg_id,
            entry_threshold=leg.entry_threshold,
            order_id=self._next_order_id(),
        )

    def on_market_for_exit(self, event: MarketEvent, portfolio: PortfolioState) -> list[OrderEvent]:
        out: list[OrderEvent] = []
        if event.market_event_type != MarketEventType.BIDASK:
            return out
        if not self._can_trade(event.timestamp):
            return out

        if self._is_force_exit_time(event.timestamp):
            for leg in portfolio.long_legs:
                if event.bid is not None:
                    out.append(
                        self._exit_order_for_leg(event=event, leg=leg, side=-1, reason="time_out")
                    )
            for leg in portfolio.short_legs:
                if event.ask is not None:
                    out.append(
                        self._exit_order_for_leg(event=event, leg=leg, side=1, reason="time_out")
                    )
            return out

        if event.bid is not None:
            for leg in portfolio.long_legs:
                if event.bid >= leg.take_profit:
                    out.append(
                        self._exit_order_for_leg(
                            event=event,
                            leg=leg,
                            side=-1,
                            reason="long_profit_take",
                        )
                    )
                elif event.bid <= leg.stop_loss:
                    out.append(
                        self._exit_order_for_leg(
                            event=event,
                            leg=leg,
                            side=-1,
                            reason="long_stop_loss",
                        )
                    )
        if event.ask is not None:
            for leg in portfolio.short_legs:
                if event.ask <= leg.take_profit:
                    out.append(
                        self._exit_order_for_leg(
                            event=event,
                            leg=leg,
                            side=1,
                            reason="short_profit_take",
                        )
                    )
                elif event.ask >= leg.stop_loss:
                    out.append(
                        self._exit_order_for_leg(
                            event=event,
                            leg=leg,
                            side=1,
                            reason="short_stop_loss",
                        )
                    )
        return out


class ExecutionHandler:
    """
    Simulated execution handler with switchable fill model.
    """

    def __init__(self, config: BacktestConfig, cost_model: CostModel) -> None:
        self.config = config
        self.cost_model = cost_model
        self.pending_orders: dict[str, OrderEvent] = {}

    def on_order(self, order: OrderEvent) -> None:
        for existing in self.pending_orders.values():
            if order.order_kind == "ENTRY" and existing.order_kind == "ENTRY":
                if (
                    existing.side == order.side
                    and abs(existing.entry_threshold - order.entry_threshold) < 1e-12
                ):
                    return
            if order.order_kind == "EXIT" and existing.order_kind == "EXIT":
                if existing.leg_id == order.leg_id:
                    return
        self.pending_orders[order.order_id] = order

    def reconcile_pending_orders(self, state: PortfolioState) -> None:
        active_leg_ids = {leg.leg_id for leg in state.long_legs} | {leg.leg_id for leg in state.short_legs}
        active_long_thresholds = {round(leg.entry_threshold, 6) for leg in state.long_legs}
        active_short_thresholds = {round(leg.entry_threshold, 6) for leg in state.short_legs}
        for order_id in list(self.pending_orders.keys()):
            order = self.pending_orders[order_id]
            if order.order_kind == "ENTRY":
                if order.side > 0 and round(order.entry_threshold, 6) in active_long_thresholds:
                    del self.pending_orders[order_id]
                elif order.side < 0 and round(order.entry_threshold, 6) in active_short_thresholds:
                    del self.pending_orders[order_id]
            elif order.order_kind == "EXIT":
                if order.leg_id not in active_leg_ids:
                    del self.pending_orders[order_id]

    def _quote_for_side(self, event: MarketEvent, side: int) -> tuple[float | None, float | None]:
        if side > 0:
            return event.ask, event.ask_size
        return event.bid, event.bid_size

    def on_market(self, event: MarketEvent, state: PortfolioState) -> list[FillEvent]:
        if event.market_event_type != MarketEventType.BIDASK:
            return []
        self.reconcile_pending_orders(state)
        if not self.pending_orders:
            return []

        fills: list[FillEvent] = []
        for order_id in list(self.pending_orders.keys()):
            order = self.pending_orders[order_id]
            quote_price, quote_size = self._quote_for_side(event, order.side)
            if quote_price is None:
                continue

            if self.config.fill_model == FillModel.FULL_NEXT_QUOTE:
                qty = order.remaining_qty
            else:
                if quote_size is None or quote_size <= 0:
                    continue
                qty = min(order.remaining_qty, int(quote_size))
                if qty <= 0:
                    continue

            slip_adjust = self.cost_model.slippage_price_adjustment(order.side, self.config.tick_size)
            fill_price = quote_price + slip_adjust
            commission = self.cost_model.commission(qty, fill_price)
            slippage_cost = self.cost_model.slippage_cost(
                qty,
                self.config.tick_size,
                self.config.contract_multiplier,
            )
            fills.append(
                FillEvent(
                    event_type=EventType.FILL,
                    timestamp=event.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=qty,
                    fill_price=fill_price,
                    commission=commission,
                    slippage_cost=slippage_cost,
                    order_id=order.order_id,
                    order_kind=order.order_kind,
                    reason=order.reason,
                    leg_id=order.leg_id,
                    entry_threshold=order.entry_threshold,
                    pt_pct=order.pt_pct,
                    sl_pct=order.sl_pct,
                )
            )
            order.remaining_qty -= qty
            if order.remaining_qty <= 0:
                del self.pending_orders[order_id]

        return fills


class Portfolio:
    """
    Full trade accounting and mark-to-market state.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.state = PortfolioState(
            cash_balance=config.initial_cash,
            equity=config.initial_cash,
        )

    def _all_legs(self) -> list[OpenLeg]:
        return self.state.long_legs + self.state.short_legs

    def has_leg(self, leg_id: str) -> bool:
        return any(leg.leg_id == leg_id for leg in self._all_legs())

    def has_entry_threshold(self, side: int, threshold: float) -> bool:
        legs = self.state.long_legs if side > 0 else self.state.short_legs
        return any(abs(leg.entry_threshold - threshold) < 1e-12 for leg in legs)

    def _find_leg(self, leg_id: str) -> tuple[list[OpenLeg] | None, OpenLeg | None]:
        for bucket in (self.state.long_legs, self.state.short_legs):
            for leg in bucket:
                if leg.leg_id == leg_id:
                    return bucket, leg
        return None, None

    def _recompute_position_snapshot(self) -> None:
        long_qty = sum(x.qty_open for x in self.state.long_legs)
        short_qty = sum(x.qty_open for x in self.state.short_legs)
        self.state.position_qty = long_qty - short_qty
        if self.state.position_qty > 0 and long_qty > 0:
            self.state.avg_entry_price = sum(x.entry_price * x.qty_open for x in self.state.long_legs) / long_qty
        elif self.state.position_qty < 0 and short_qty > 0:
            self.state.avg_entry_price = sum(x.entry_price * x.qty_open for x in self.state.short_legs) / short_qty
        else:
            self.state.avg_entry_price = 0.0

    def _mark_to_market(self) -> None:
        if self.state.last_mid is None:
            self.state.unrealized_pnl = 0.0
        else:
            mult = self.config.contract_multiplier
            long_upnl = sum(
                (self.state.last_mid - leg.entry_price) * leg.qty_open * mult
                for leg in self.state.long_legs
            )
            short_upnl = sum(
                (leg.entry_price - self.state.last_mid) * leg.qty_open * mult
                for leg in self.state.short_legs
            )
            self.state.unrealized_pnl = long_upnl + short_upnl

        position_value = (
            0.0
            if self.state.last_mid is None
            else self.state.position_qty * self.state.last_mid * self.config.contract_multiplier
        )
        self.state.equity = self.state.cash_balance + position_value

    def on_market(self, event: MarketEvent) -> None:
        if event.market_event_type == MarketEventType.BIDASK:
            self.state.last_bid = event.bid
            self.state.last_ask = event.ask
            if event.bid is not None and event.ask is not None:
                self.state.last_mid = 0.5 * (event.bid + event.ask)
        self._mark_to_market()

    def on_fill(self, fill: FillEvent) -> TradeLogRow:
        signed_qty = fill.side * fill.quantity
        px = fill.fill_price
        mult = self.config.contract_multiplier

        # Cash movement from trade + explicit transaction costs.
        self.state.cash_balance -= signed_qty * px * mult
        self.state.cash_balance -= fill.commission
        self.state.commissions += fill.commission
        self.state.slippage += fill.slippage_cost

        if fill.order_kind == "ENTRY":
            pt_pct = fill.pt_pct or 0.0
            sl_pct = fill.sl_pct or 0.0
            if fill.side > 0:
                take_profit = px * (1.0 + pt_pct)
                stop_loss = px * (1.0 - sl_pct)
                self.state.long_legs.append(
                    OpenLeg(
                        leg_id=fill.leg_id,
                        side=fill.side,
                        entry_threshold=fill.entry_threshold,
                        entry_price=px,
                        qty_open=fill.quantity,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        opened_at=fill.timestamp,
                    )
                )
            else:
                take_profit = px * (1.0 - pt_pct)
                stop_loss = px * (1.0 + sl_pct)
                self.state.short_legs.append(
                    OpenLeg(
                        leg_id=fill.leg_id,
                        side=fill.side,
                        entry_threshold=fill.entry_threshold,
                        entry_price=px,
                        qty_open=fill.quantity,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        opened_at=fill.timestamp,
                    )
                )
        elif fill.order_kind == "EXIT":
            bucket, leg = self._find_leg(fill.leg_id)
            if leg is not None and bucket is not None:
                close_qty = min(fill.quantity, leg.qty_open)
                if leg.side > 0:
                    realized = (px - leg.entry_price) * close_qty * mult
                else:
                    realized = (leg.entry_price - px) * close_qty * mult
                self.state.realized_pnl += realized
                leg.qty_open -= close_qty
                if leg.qty_open <= 0:
                    bucket.remove(leg)

        self._recompute_position_snapshot()
        self._mark_to_market()

        return TradeLogRow(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            order_id=fill.order_id,
            order_kind=fill.order_kind,
            reason=fill.reason,
            leg_id=fill.leg_id,
            entry_threshold=fill.entry_threshold,
            side=fill.side,
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            commission=fill.commission,
            slippage_cost=fill.slippage_cost,
            position_after=self.state.position_qty,
            avg_entry_after=self.state.avg_entry_price,
            realized_pnl_after=self.state.realized_pnl,
            cash_after=self.state.cash_balance,
        )


class Performance:
    def __init__(self) -> None:
        self.equity_curve: list[EquityPoint] = []
        self.trade_log: list[TradeLogRow] = []
        self._last_equity_bucket: datetime | None = None

    def record_equity(self, timestamp: datetime, state: PortfolioState) -> None:
        bucket = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
        if self._last_equity_bucket == bucket:
            return
        self._last_equity_bucket = bucket
        self.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                cash_balance=state.cash_balance,
                realized_pnl=state.realized_pnl,
                unrealized_pnl=state.unrealized_pnl,
                equity=state.equity,
                position_qty=state.position_qty,
                mid_price=state.last_mid,
            )
        )

    def record_trade(self, row: TradeLogRow) -> None:
        self.trade_log.append(row)

    def summary(self) -> dict[str, float | int]:
        if not self.equity_curve:
            return {
                "num_trades": 0,
                "final_equity": 0.0,
                "realized_pnl": 0.0,
                "max_drawdown": 0.0,
            }
        eq = [x.equity for x in self.equity_curve]
        peak = eq[0]
        max_dd = 0.0
        for value in eq:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        return {
            "num_trades": len(self.trade_log),
            "final_equity": self.equity_curve[-1].equity,
            "realized_pnl": self.equity_curve[-1].realized_pnl,
            "max_drawdown": max_dd,
        }

    def write_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        eq_path = output_dir / "equity_curve.csv"
        with eq_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.equity_curve[0]).keys()) if self.equity_curve else [
                "timestamp",
                "cash_balance",
                "realized_pnl",
                "unrealized_pnl",
                "equity",
                "position_qty",
                "mid_price",
            ])
            writer.writeheader()
            for row in self.equity_curve:
                writer.writerow(asdict(row))

        trade_path = output_dir / "trade_log.csv"
        trade_fields = [
            "timestamp",
            "symbol",
            "order_id",
            "order_kind",
            "reason",
            "leg_id",
            "entry_threshold",
            "side",
            "quantity",
            "fill_price",
            "commission",
            "slippage_cost",
            "position_after",
            "avg_entry_after",
            "realized_pnl_after",
            "cash_after",
        ]
        with trade_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trade_fields)
            writer.writeheader()
            for row in self.trade_log:
                writer.writerow(asdict(row))

        summary_path = output_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            for k, v in self.summary().items():
                f.write(f"{k}: {v}\n")

