"""
Run event-driven backtest for HSI tick data.

Example:
    python -m backtest.main --data-dir historicalData --index-file historicalData/hsi_index_daily.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .components import DataHandler, ExecutionHandler, OrderManager, Performance, Portfolio, VwapReversionZscoreStrategy
from .engine import BacktestEngine
from .models import BacktestConfig, CostModel, FillModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-driven tick backtest for IBPrice data.")
    parser.add_argument("--symbol", default="HSI")
    parser.add_argument("--data-dir", default="historicalData")
    parser.add_argument("--index-file", default="historicalData/hsi_index_daily.csv")
    parser.add_argument("--output-dir", default="backtest_output")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--vwap-threshold", type=float, default=0.003)
    parser.add_argument("--max-vwap-threshold", type=float, default=0.007)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--pt-multiplier", type=float, default=1.5, help="PT ratio vs entry threshold.")
    parser.add_argument("--sl-multiplier", type=float, default=0.4, help="SL ratio vs entry threshold.")
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--range-filter-mode", choices=["none", "upper_only", "two_sided"], default="upper_only")
    parser.add_argument("--range-filter-low-pct", type=float, default=0.2)
    parser.add_argument("--range-filter-high-pct", type=float, default=0.8)
    parser.add_argument("--range-filter-lookback", type=int, default=30)
    parser.add_argument("--range-use-normalized", action="store_true")
    parser.add_argument(
        "--threshold-qty-map",
        default="",
        help="Comma-separated threshold:qty map, e.g. 0.002:2,0.003:2,0.006:0",
    )
    parser.add_argument("--force-exit-time-hkt", default="15:59:00", help="Force close open position at/after HH:MM:SS")
    parser.add_argument("--fill-model", choices=[x.value for x in FillModel], default=FillModel.FULL_NEXT_QUOTE.value)
    parser.add_argument("--trade-size", type=int, default=1)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--commission-per-contract",
        "--commission-per-fill",
        dest="commission_per_contract",
        type=float,
        default=25.0,
        help="Commission in HKD charged per filled contract.",
    )
    parser.add_argument("--slippage-ticks", type=float, default=0.0)
    parser.add_argument("--tick-size", type=float, default=1.0)
    parser.add_argument("--contract-multiplier", type=float, default=50.0)
    return parser.parse_args()


def _parse_date_or_none(text: str | None):
    if not text:
        return None
    return datetime.strptime(text, "%Y-%m-%d").date()


def main() -> None:
    args = _parse_args()

    config = BacktestConfig(
        symbol=args.symbol,
        data_dir=args.data_dir,
        index_file=args.index_file,
        output_dir=args.output_dir,
        vwap_threshold=args.vwap_threshold,
        max_vwap_threshold=args.max_vwap_threshold,
        threshold_step=args.threshold_step,
        pt_multiplier=args.pt_multiplier,
        sl_multiplier=args.sl_multiplier,
        rolling_window=args.rolling_window,
        range_filter_mode=args.range_filter_mode,
        range_filter_low_pct=args.range_filter_low_pct,
        range_filter_high_pct=args.range_filter_high_pct,
        range_filter_lookback=args.range_filter_lookback,
        range_use_normalized=args.range_use_normalized,
        force_exit_time_hkt=args.force_exit_time_hkt,
        fill_model=FillModel(args.fill_model),
        trade_size=args.trade_size,
        initial_cash=args.initial_cash,
        tick_size=args.tick_size,
        contract_multiplier=args.contract_multiplier,
        threshold_qty_map_str=args.threshold_qty_map,
    )
    cost_model = CostModel(
        commission_per_contract=args.commission_per_contract,
        slippage_ticks=args.slippage_ticks,
    )

    data_handler = DataHandler(config)
    strategy = VwapReversionZscoreStrategy(config)
    portfolio = Portfolio(config)
    order_manager = OrderManager(config)
    execution_handler = ExecutionHandler(config, cost_model)
    performance = Performance()

    engine = BacktestEngine(
        config=config,
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        order_manager=order_manager,
        execution_handler=execution_handler,
        performance=performance,
    )
    engine.run(
        start_date=_parse_date_or_none(args.start_date),
        end_date=_parse_date_or_none(args.end_date),
    )

    output_dir = Path(config.output_dir)
    performance.write_outputs(output_dir)

    summary = performance.summary()
    print("Backtest complete.")
    print(f"Output directory: {output_dir.resolve()}")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

