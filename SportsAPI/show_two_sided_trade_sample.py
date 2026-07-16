#!/usr/bin/env python3
"""Show a sample match with both P1 and P2 backtest trades in detail."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_calibrated_probability_model import REPORTS_DIR, fmt  # noqa: E402
from calibrate_adjusted_probability import (  # noqa: E402
    TARGET_COL,
    add_factor_features,
    fit_probability_adjustment,
    load_enriched_rows,
    predict_adjusted_probs,
    probability_adjustment_stats,
    run_backtest,
    select_top_factors_per_group,
)
from train_xgboost_factor_models import fill_matrix  # noqa: E402

OUTPUT_PATH = REPORTS_DIR / "two_sided_trade_sample.html"
THRESHOLD = 0.005


def prepare_test_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_enriched_rows()
    df, factor_features, factor_groups = add_factor_features(df)
    df = df[df["has_odds"] == 1].copy()
    df["devig_p1"] = pd.to_numeric(df["odds_no_vig_p1"], errors="coerce")
    df["devig_p2"] = pd.to_numeric(df["odds_no_vig_p2"], errors="coerce").fillna(1.0 - df["devig_p1"])
    df = df[(df["odds_age_seconds"] >= 0)].copy() if "odds_age_seconds" in df.columns else df

    events = (
        df[["event_id", "event_start_date"]]
        .drop_duplicates("event_id")
        .sort_values(["event_start_date", "event_id"], na_position="last")
    )
    event_ids = events["event_id"].tolist()
    n = len(event_ids)
    train_set = set(event_ids[: int(n * 0.60)])
    test_set = set(event_ids[int(n * 0.60):])
    train = df.loc[df["event_id"].isin(train_set)].copy()
    test = df.loc[df["event_id"].isin(test_set)].copy()

    x_train = fill_matrix(train, train, factor_features)
    y_train = train[TARGET_COL].to_numpy(dtype=float)
    devig_p1_train = train["devig_p1"].to_numpy(dtype=float)
    devig_p2_train = train["devig_p2"].to_numpy(dtype=float)

    model_pass1 = fit_probability_adjustment(
        x_train, y_train, devig_p1_train, devig_p2_train,
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    stats_pass1 = probability_adjustment_stats(
        model_pass1, train, "devig_p1", "devig_p2", TARGET_COL, factor_features,
    )
    selected = select_top_factors_per_group(stats_pass1, factor_groups, max_per_group=2)

    model = fit_probability_adjustment(
        fill_matrix(train, train, selected), y_train, devig_p1_train, devig_p2_train,
        l2=0.02, learning_rate=0.06, epochs=500,
    )
    adjusted_p1, adjusted_p2, _ = predict_adjusted_probs(
        model,
        fill_matrix(train, test, selected),
        test["devig_p1"].to_numpy(dtype=float),
        test["devig_p2"].to_numpy(dtype=float),
    )
    return test, adjusted_p1, adjusted_p2


def pick_sample_event(bt_df: pd.DataFrame) -> str:
    """Pick a test match with both P1 and P2 trades."""
    bets = bt_df.loc[bt_df["side"] != ""]
    summary = bets.groupby("event_id").agg(
        p1=("side", lambda s: (s == "P1").sum()),
        p2=("side", lambda s: (s == "P2").sum()),
        rows=("side", "count"),
    )
    both = summary[(summary["p1"] >= 1) & (summary["p2"] >= 1)].copy()
    both["total"] = both["p1"] + both["p2"]
    both = both.sort_values(["p1", "p2", "total"], ascending=False)
    if both.empty:
        raise RuntimeError("No match found with both P1 and P2 trades at this threshold.")
    return str(both.index[0])


def build_trade_table(match_df: pd.DataFrame) -> pd.DataFrame:
    trades = match_df.loc[match_df["side"] != ""].copy()
    if "utc_time" not in trades.columns and "ut" in trades.columns:
        trades["utc_time"] = pd.to_datetime(trades["ut"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    trades["trade_price"] = trades["bet_cost"]
    trades["signal_raw_p1"] = pd.to_numeric(trades.get("signal_raw_implied_p1"), errors="coerce")
    trades["signal_raw_p2"] = pd.to_numeric(trades.get("signal_raw_implied_p2"), errors="coerce")
    trades["exec_raw_p1"] = pd.to_numeric(trades.get("next_raw_implied_p1"), errors="coerce")
    trades["exec_raw_p2"] = pd.to_numeric(trades.get("next_raw_implied_p2"), errors="coerce")
    trades["pinnacle_p1"] = pd.to_numeric(trades.get("odds_p1_price"), errors="coerce")
    trades["pinnacle_p2"] = pd.to_numeric(trades.get("odds_p2_price"), errors="coerce")
    trades["exec_pinnacle_p1"] = pd.to_numeric(trades.get("next_pinnacle_p1"), errors="coerce")
    trades["exec_pinnacle_p2"] = pd.to_numeric(trades.get("next_pinnacle_p2"), errors="coerce")
    trades["edge"] = np.where(
        trades["side"] == "P1",
        trades["adjusted_prob_p1"] - trades["trade_price"],
        trades["adjusted_prob_p2"] - trades["trade_price"],
    )
    trades["won"] = trades["payout"] == 1.0
    trades["match_winner"] = np.where(trades[TARGET_COL] == 1, "P1", "P2")

    cols = [
        "seq", "utc_time", "side", "sets_after", "game_score_after", "point_score_state",
        "pinnacle_p1", "pinnacle_p2", "exec_pinnacle_p1", "exec_pinnacle_p2",
        "raw_implied_p1", "raw_implied_p2", "signal_raw_p1", "signal_raw_p2",
        "exec_raw_p1", "exec_raw_p2", "devig_p1", "devig_p2",
        "adjusted_prob_p1", "adjusted_prob_p2", "trade_price", "edge",
        "won", "pnl", "match_winner",
    ]
    available = [c for c in cols if c in trades.columns]
    return trades[available].sort_values("seq")


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> int:
    print("Preparing test predictions and backtest...")
    test, adjusted_p1, adjusted_p2 = prepare_test_predictions()
    bt = run_backtest(test, adjusted_p1, adjusted_p2, threshold=THRESHOLD)
    bt_df = bt["df"]

    event_id = pick_sample_event(bt_df)
    match_all = bt_df.loc[bt_df["event_id"] == event_id].copy()
    trades = build_trade_table(match_all)

    event_name = str(match_all["event_name"].iloc[0]) if "event_name" in match_all.columns else event_id
    p1_count = int((trades["side"] == "P1").sum())
    p2_count = int((trades["side"] == "P2").sum())
    total_pnl = float(trades["pnl"].sum())

    print(f"\nSample match: {event_name} (event_id={event_id})")
    print(f"Threshold: {THRESHOLD} | P1 trades: {p1_count} | P2 trades: {p2_count} | Match PnL: {total_pnl:+.3f}")
    print()

    display_cols = [
        "seq", "utc_time", "side", "sets_after", "game_score_after",
        "raw_implied_p1", "raw_implied_p2", "exec_raw_p1", "exec_raw_p2",
        "adjusted_prob_p1", "adjusted_prob_p2", "trade_price", "edge", "won", "pnl",
    ]
    show = trades[display_cols]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(show.to_string(index=False))

    rows = []
    for _, r in show.iterrows():
        rows.append([
            str(int(r["seq"])),
            str(r["utc_time"]),
            str(r["side"]),
            str(r["sets_after"]),
            str(r["game_score_after"]),
            fmt(r["pinnacle_p1"], 2),
            fmt(r["pinnacle_p2"], 2),
            fmt(r["raw_implied_p1"], 4),
            fmt(r["raw_implied_p2"], 4),
            fmt(r["devig_p1"], 4),
            fmt(r["devig_p2"], 4),
            fmt(r["adjusted_prob_p1"], 4),
            fmt(r["adjusted_prob_p2"], 4),
            fmt(r["trade_price"], 4),
            fmt(r["edge"], 4),
            "Yes" if r["won"] else "No",
            fmt(r["pnl"], 4),
        ])

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Two-Sided Trade Sample</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2328; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #57606a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 7px 9px; text-align: right; }}
    th {{ background: #f6f8fa; text-align: center; }}
    td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5) {{ text-align: left; }}
    .note {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>Two-Sided Backtest Trade Sample</h1>
  <p class="muted">{html.escape(event_name)} &mdash; event_id {html.escape(event_id)}</p>

  <div class="note">
    <b>Rules (threshold = {THRESHOLD}):</b>
    <ul>
      <li>adjusted_prob_p1 = devig_p1 + adjustment; adjusted_prob_p2 = devig_p2 - adjustment</li>
      <li>Buy <b>P1</b> when adjusted_prob_p1 &gt; raw implied P1 + {THRESHOLD}</li>
      <li>Buy <b>P2</b> when adjusted_prob_p2 &gt; raw implied P2 + {THRESHOLD}</li>
      <li>Execution price = raw implied from the <b>next odds update</b> in the same match</li>
    </ul>
    <p>P1 trades: {p1_count} | P2 trades: {p2_count} | Total match PnL: {total_pnl:+.3f}</p>
  </div>

  {html_table([
    "Seq", "Time (UTC)", "Side", "Sets", "Games",
    "Pinnacle P1", "Pinnacle P2", "Raw P1", "Raw P2", "De-vig P1", "De-vig P2",
    "Adj P1", "Adj P2", "Trade price", "Edge", "Won", "PnL",
  ], rows)}

  <p class="muted">All {len(trades)} trades for this match shown above.</p>
</body>
</html>
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(page, encoding="utf-8")
    print(f"\nReport: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
