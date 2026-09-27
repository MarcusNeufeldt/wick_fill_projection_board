#!/usr/bin/env python3
"""Multi-origin chronological learning curve for the established A model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from evaluate_prospective_entry_baseline import read_observations  # noqa: E402
from evaluate_rolling_forecast_systems import (  # noqa: E402
    eligible_history_and_queries,
    rolling_signal_folds,
    signal_window,
)
from learning_curve import HORIZONS, THRESHOLDS, summarize  # noqa: E402
from train_prospective_entry_model import (  # noqa: E402
    CONFIGS,
    build_feature_matrix,
    evaluate_bundle,
    fit_bundle,
)


METRICS = (
    "fill_brier",
    "adverse_interval_mae_pct",
    "wait_mae_minutes",
    "target_vs_2pct_brier",
    "target_vs_5pct_brier",
)


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
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.40)
    parser.add_argument("--maximum-queries", type=int, default=750)
    parser.add_argument("--configuration", default="moderate")
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(","))
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must be in (0, 1]")
    config = next((value for value in CONFIGS if value.name == args.configuration), None)
    if config is None:
        raise ValueError(f"unknown configuration: {args.configuration}")

    started = perf_counter()
    observations, metadata = read_observations(args.dataset_dir, args.timeframe)
    observations = observations.reset_index(drop=True)
    features = build_feature_matrix(args.root, observations)
    lookup = observations[["observation_id", "signal_id"]].drop_duplicates("observation_id")
    folds = rolling_signal_folds(
        observations,
        args.folds,
        args.initial_train_fraction,
    )
    results = []
    aggregate_values: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for fold in folds:
        before_test = observations.loc[
            observations["signal_open_time_ms"].lt(fold.validation_end_signal_ms)
        ].copy()
        test_window = signal_window(
            observations,
            fold.validation_end_signal_ms,
            fold.test_end_signal_ms,
        )
        eligible, queries, availability = eligible_history_and_queries(
            before_test,
            test_window,
            args.maximum_queries,
        )
        eligible_times = np.sort(eligible["signal_open_time_ms"].unique()).astype(np.int64)
        for fraction in fractions:
            keep = max(1, int(np.ceil(len(eligible_times) * fraction)))
            oldest_included = int(eligible_times[-keep])
            train = eligible.loc[
                eligible["signal_open_time_ms"].ge(oldest_included)
            ].copy()
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
            metrics = evaluate_bundle(bundle, queries, features)
            summary = summarize(metrics, lookup)
            record = {
                "fold": fold.number,
                "fraction": fraction,
                "training_rows": int(bundle["training_rows"]),
                "training_filled_rows": int(bundle["training_filled_rows"]),
                "training_episodes": int(train["signal_id"].nunique()),
                "query_rows": int(len(queries)),
                "query_episodes": int(queries["signal_id"].nunique()),
                "oldest_signal_open_time_ms": oldest_included,
                "first_query_close_ms": availability["first_query_close_ms"],
                "fit_seconds": fit_seconds,
                "metrics": summary,
            }
            results.append(record)
            for horizon, values in summary.items():
                for metric in METRICS:
                    value = values[metric]
                    if value is not None:
                        aggregate_values[(fraction, horizon, metric)].append(float(value))
            print(json.dumps({"stage": "fold_fraction_complete", **record}, separators=(",", ":")), flush=True)

    aggregate = {}
    for fraction in fractions:
        fraction_result = {}
        for horizon in ("1440m", "10080m", "43200m"):
            horizon_result = {}
            for metric in METRICS:
                values = np.asarray(
                    aggregate_values[(fraction, horizon, metric)], dtype=float
                )
                horizon_result[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "folds": int(len(values)),
                }
            fraction_result[horizon] = horizon_result
        aggregate[f"{fraction:g}"] = fraction_result
    print(
        json.dumps(
            {
                "stage": "complete",
                "schema_version": "wick-data-rolling-learning-curve-v1",
                "timeframe": args.timeframe,
                "source_metadata_generated_at_utc": metadata.get("generated_at_utc"),
                "folds": args.folds,
                "fractions": fractions,
                "total_seconds": perf_counter() - started,
                "aggregate": aggregate,
                "results": results,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
