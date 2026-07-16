#!/usr/bin/env python3
"""Summarize backtest PnL by side (P1 vs P2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_adjusted_probability import (  # noqa: E402
    BACKTEST_THRESHOLDS,
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


def prepare_test_predictions():
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
    test_set = set(event_ids[int(n * 0.60) :])
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


def side_summary(bt_df: pd.DataFrame, side: str) -> dict:
    bets = bt_df.loc[bt_df["side"] == side]
    if bets.empty:
        return {
            "bets": 0, "wagered": 0.0, "pnl": 0.0, "roi": 0.0,
            "win_rate": 0.0, "avg_edge": 0.0,
        }
    wagered = float(bets["bet_cost"].sum())
    pnl = float(bets["pnl"].sum())
    if side == "P1":
        avg_edge = float((bets["adjusted_prob_p1"] - bets["trade_price_p1"]).mean())
    else:
        avg_edge = float((bets["adjusted_prob_p2"] - bets["trade_price_p2"]).mean())
    return {
        "bets": len(bets),
        "wagered": wagered,
        "pnl": pnl,
        "roi": pnl / wagered if wagered else 0.0,
        "win_rate": float(bets["payout"].mean()),
        "avg_edge": avg_edge,
    }


def main() -> int:
    print("Preparing test predictions...")
    test, adjusted_p1, adjusted_p2 = prepare_test_predictions()

    thresholds = BACKTEST_THRESHOLDS
    print()
    print("BACKTEST PnL SUMMARY")
    print("adj_p1 = devig_p1 + adj | adj_p2 = devig_p2 - adj | symmetric raw-implied signals")
    print("=" * 92)

    for th in thresholds:
        bt = run_backtest(test, adjusted_p1, adjusted_p2, threshold=th)
        bt_df = bt["df"]
        p1 = side_summary(bt_df, "P1")
        p2 = side_summary(bt_df, "P2")
        for label, s in [("TOTAL", {
            "bets": bt["n_bets"], "wagered": bt["total_wagered"], "pnl": bt["total_pnl"],
            "roi": bt["roi"], "win_rate": bt["win_rate"], "avg_edge": bt["avg_edge"],
        }), ("P1", p1), ("P2", p2)]:
            print(
                f"thresh={th:.3f} {label:5s}  "
                f"bets={s['bets']:>6,}  wagered={s['wagered']:>10.1f}  "
                f"PnL={s['pnl']:>+9.2f}  ROI={s['roi']*100:>+7.2f}%  "
                f"win={s['win_rate']*100:>5.1f}%  edge={s['avg_edge']:>+.4f}"
            )
        print()

    th = 0.010
    bt = run_backtest(test, adjusted_p1, adjusted_p2, threshold=th)
    bets = bt["df"].loc[bt["df"]["side"] != ""]
    match_pnl = bets.groupby("event_id")["pnl"].sum()
    print(f"At threshold {th:.3f} (primary):")
    print(f"  Matches with trades: {bets['event_id'].nunique()}")
    print(f"  Profitable matches: {(match_pnl > 0).sum()} ({(match_pnl > 0).mean()*100:.1f}%)")
    print(f"  Avg PnL per match: {match_pnl.mean():+.3f}")
    print(f"  Median PnL per match: {match_pnl.median():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
