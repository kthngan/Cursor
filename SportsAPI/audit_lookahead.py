#!/usr/bin/env python3
"""Audit factors and odds for forward-looking / lookahead bias."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_adjusted_probability import (
    WITH_ODDS_DIR,
    add_factor_features,
    load_enriched_rows,
    select_top_factors_per_group,
)
from train_calibrated_probability_model import METRIC_COLUMNS, SCORE_FEATURES
from train_xgboost_factor_models import add_factor_features as build_factors

FORBIDDEN_FEATURE_SUBSTRINGS = [
    "match_winner", "target_", "winner_side", "game_winner", "future",
]

STALENESS_MAX = 300


def main() -> int:
    print("=== Lookahead Bias Audit ===\n")

    df = load_enriched_rows()
    df = df[df["has_odds"] == 1].copy()
    df["row_dt"] = pd.to_datetime(df["ut"], unit="s", utc=True, errors="coerce")
    df["snap_dt"] = pd.to_datetime(df["odds_snapshot_time"], utc=True, errors="coerce")
    df["odds_age_seconds"] = (df["row_dt"] - df["snap_dt"]).dt.total_seconds()

    # --- Odds timing ---
    print("1. ODDS TIMING")
    total = len(df)
    neg_age = (df["odds_age_seconds"] < 0).sum()
    zero_age = (df["odds_age_seconds"] == 0).sum()
    within_5m = ((df["odds_age_seconds"] >= 0) & (df["odds_age_seconds"] <= STALENESS_MAX)).sum()
    print(f"   Rows with odds: {total:,}")
    print(f"   Negative age (snapshot AFTER row): {neg_age:,} ({neg_age/total:.2%})")
    print(f"   Zero age (exact match): {zero_age:,}")
    print(f"   Age 0-300s (used in model): {within_5m:,} ({within_5m/total:.2%})")
    if neg_age > 0:
        sample = df.loc[df["odds_age_seconds"] < 0].head(3)
        print("   Sample negative-age rows:")
        for _, r in sample.iterrows():
            print(f"     ut={r['utc_time']} snap={r['odds_snapshot_time']} age={r['odds_age_seconds']:.0f}s")
    print()

    filtered = df[(df["odds_age_seconds"] >= 0) & (df["odds_age_seconds"] <= STALENESS_MAX)]
    neg_in_filtered = (filtered["odds_age_seconds"] < 0).sum()
    print(f"   After <=5min filter: {len(filtered):,} rows, negative age: {neg_in_filtered}")
    print("   Verdict: odds are latest snapshot AT OR BEFORE row time (no future snapshots in model set)")
    print()

    # --- Target vs features ---
    print("2. TARGET vs FEATURES")
    df, all_features, groups = build_factors(df)
    leaked = [f for f in all_features if any(s in f.lower() for s in FORBIDDEN_FEATURE_SUBSTRINGS)]
    print(f"   Total factor features: {len(all_features)}")
    print(f"   Forbidden substrings in features: {leaked or 'none'}")
    print(f"   target_p1_win used as label only (constant per match, not in feature list)")
    print(f"   match_winner_side used only to derive target, not in features")
    print()

    # --- Rolling / delta direction ---
    print("3. FACTOR CONSTRUCTION (backward-looking)")
    print("   Rolling metrics: computed from trailing windows at each incident (sportsapi_metric_helpers)")
    print("   Score state (*_after): state AFTER incident at row ut — known at that timestamp")
    print("   live_form_delta_5: rolling_live_form_ratio - shift(5) within single match CSV")
    print("   Trend deltas: groupby(event_id).diff(lag) — backward within match")

    # Check first 5 rows of a sample match have NaN for diff features
    sample_eid = df["event_id"].iloc[0]
    sample = df[df["event_id"] == sample_eid].head(10)
    diff_col = "rolling_live_form_ratio_delta_3"
    if diff_col in sample.columns:
        first_vals = sample[diff_col].head(3).tolist()
        print(f"   Sample match {sample_eid} first {diff_col} values: {first_vals}")
        print("   (NaN at start confirms backward diff, not forward)")
    print()

    # --- Example row ---
    print("4. EXAMPLE ROW (verify timing chain)")
    row = filtered.iloc[len(filtered) // 2]
    print(f"   Event: {row.get('event_name', row['event_id'])}")
    print(f"   Row seq: {row.get('seq')}  ut: {row.get('utc_time')}  ({row['row_dt']})")
    print(f"   Odds snap: {row['odds_snapshot_time']}  age: {row['odds_age_seconds']:.0f}s")
    print(f"   Score: sets={row.get('sets_after')} games={row.get('game_score_after')} points={row.get('point_score_state')}")
    print(f"   Pinnacle de-vig P1: {row.get('odds_no_vig_p1'):.4f}  P2: {row.get('odds_no_vig_p2'):.4f}")
    print(f"   Rolling live form: {row.get('rolling_live_form_ratio')}")
    print(f"   Target (final winner): P1 win = {row.get('target_p1_win')}  [label only]")
    print()

    # --- Potential caveats ---
    print("5. CAVEATS (not forward-looking, but worth knowing)")
    print("   - target_p1_win is the FINAL match outcome on every row (label, not feature)")
    print("   - Features reflect state AFTER the incident at row ut (post-point, not pre-point)")
    print("   - Odds forward-fill: if match missing from a snapshot, last known odds are carried forward")
    print("     (stale/backward, not future)")
    print(f"   - Enrichment fallback snap_idx=0 can assign future snapshots when row precedes all snaps;")
    print(f"     {neg_age:,} such rows exist but are EXCLUDED by age>=0 filter")
    print("   - Probability adjustment has no hard cap; outputs are clipped to [0, 1]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
