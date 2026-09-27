#!/usr/bin/env python3
"""Chronological data-scaling curve for the established numerical model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluate_prospective_entry_baseline import read_observations, stratified_sample  # noqa: E402
from prospective_entry_outcomes import horizon_slug  # noqa: E402
from train_prospective_entry_model import (  # noqa: E402
    CONFIGS,
    build_feature_matrix,
    evaluate_bundle,
    fit_bundle,
)


HORIZONS = (1_440, 10_080, 43_200)
THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
MAXIMUM_HORIZON_MS = max(HORIZONS) * 60_000


def episode_balanced_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = frame.loc[np.isfinite(frame[column].to_numpy(dtype=float)), ["signal_id", column]]
    if values.empty:
        return None
    return float(values.groupby("signal_id", sort=False)[column].mean().mean())


def summarize(metrics: pd.DataFrame, lookup: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    joined = metrics.merge(lookup, on="observation_id", how="left", validate="many_to_one")
    output: dict[str, dict[str, float | int | None]] = {}
    for horizon in HORIZONS:
        rows = joined.loc[joined["horizon_minutes"].eq(horizon)]
        fill_rows = rows.loc[rows["fill_brier"].notna()]
        risk_rows = rows.loc[rows["risk_interval_error_pct"].notna()]
        wait_rows = rows.loc[rows["time_absolute_error_minutes"].notna()]
        outcome_2 = rows.loc[rows.get("adverse_threshold_pct", pd.Series(index=rows.index)).eq(2.0)]
        outcome_5 = rows.loc[rows.get("adverse_threshold_pct", pd.Series(index=rows.index)).eq(5.0)]
        output[horizon_slug(horizon)] = {
            "queries": int(fill_rows["observation_id"].nunique()),
            "episodes": int(fill_rows["signal_id"].nunique()),
            "fill_brier": episode_balanced_mean(fill_rows, "fill_brier"),
            "adverse_interval_mae_pct": episode_balanced_mean(risk_rows, "risk_interval_error_pct"),
            "wait_mae_minutes": episode_balanced_mean(wait_rows, "time_absolute_error_minutes"),
            "target_vs_2pct_brier": episode_balanced_mean(outcome_2, "outcome_brier"),
            "target_vs_5pct_brier": episode_balanced_mean(outcome_5, "outcome_brier"),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "data" / "prospective_entry_outcomes_v1",
    )
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--maximum-queries", type=int, default=1_500)
    parser.add_argument("--configuration", default="moderate")
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(","))
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must be in (0, 1]")
    if not 0.05 <= args.test_fraction <= 0.40:
        raise ValueError("test-fraction must be in [0.05, 0.40]")
    config = next((value for value in CONFIGS if value.name == args.configuration), None)
    if config is None:
        raise ValueError(f"unknown configuration: {args.configuration}")

    started = perf_counter()
    observations, metadata = read_observations(args.dataset_dir, args.timeframe)
    observations = observations.reset_index(drop=True)
    features = build_feature_matrix(args.root, observations)
    signal_times = np.sort(observations["signal_open_time_ms"].unique()).astype(np.int64)
    test_start_position = int(len(signal_times) * (1.0 - args.test_fraction))
    test_start_position = min(max(test_start_position, 1), len(signal_times) - 1)
    test_start_ms = int(signal_times[test_start_position])
    maximum_slug = horizon_slug(max(HORIZONS))
    query_window = observations.loc[
        observations["signal_open_time_ms"].ge(test_start_ms)
        & observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
    ].copy()
    queries = stratified_sample(query_window, args.maximum_queries).sort_index()
    if queries.empty:
        raise RuntimeError("no fully matured queries in fixed test window")
    first_query_close_ms = int(queries["entry_close_time_ms"].min())
    label_available_ms = observations["entry_close_time_ms"].to_numpy(dtype=np.int64) + MAXIMUM_HORIZON_MS
    eligible = observations.loc[
        observations["signal_open_time_ms"].lt(test_start_ms)
        & observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool).to_numpy()
        & (label_available_ms <= first_query_close_ms)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no label-mature history before fixed query window")
    eligible_times = np.sort(eligible["signal_open_time_ms"].unique()).astype(np.int64)
    lookup = observations[["observation_id", "signal_id"]].drop_duplicates("observation_id")

    curve = []
    for fraction in fractions:
        keep = max(1, int(np.ceil(len(eligible_times) * fraction)))
        oldest_included = int(eligible_times[-keep])
        train = eligible.loc[eligible["signal_open_time_ms"].ge(oldest_included)].copy()
        fit_started = perf_counter()
        bundle = fit_bundle(
            train,
            features.loc[train.index],
            HORIZONS,
            THRESHOLDS,
            config,
            args.timeframe,
        )
        fit_seconds = perf_counter() - fit_started
        score_started = perf_counter()
        metrics = evaluate_bundle(bundle, queries, features)
        score_seconds = perf_counter() - score_started
        record = {
            "fraction": fraction,
            "training_rows": int(bundle["training_rows"]),
            "training_filled_rows": int(bundle["training_filled_rows"]),
            "training_episodes": int(train["signal_id"].nunique()),
            "oldest_signal_open_time_ms": oldest_included,
            "fit_seconds": fit_seconds,
            "score_seconds": score_seconds,
            "metrics": summarize(metrics, lookup),
        }
        curve.append(record)
        print(json.dumps({"stage": "fraction_complete", **record}, separators=(",", ":")), flush=True)

    print(
        json.dumps(
            {
                "stage": "complete",
                "schema_version": "wick-data-learning-curve-v1",
                "timeframe": args.timeframe,
                "source_metadata_generated_at_utc": metadata.get("generated_at_utc"),
                "eligible_rows": len(eligible),
                "eligible_episodes": int(eligible["signal_id"].nunique()),
                "query_rows": len(queries),
                "query_episodes": int(queries["signal_id"].nunique()),
                "test_start_ms": test_start_ms,
                "total_seconds": perf_counter() - started,
                "curve": curve,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
