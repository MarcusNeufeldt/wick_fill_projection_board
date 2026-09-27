#!/usr/bin/env python3
"""Reference benchmark for the complete Python prospective-outcome kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from prospective_entry_outcomes import (  # noqa: E402
    DEFAULT_ADVERSE_THRESHOLDS_PCT,
    DEFAULT_ENTRY_AGES_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    build_series_dataset,
    horizon_slug,
    parse_float_list,
    parse_int_list,
    threshold_slug,
)
from python_wick_scan import read_candles  # noqa: E402


def finite_sum(values: pd.Series) -> float:
    numeric = values.to_numpy(dtype=float)
    return float(np.sum(numeric[np.isfinite(numeric)]))


def aggregate_observations(
    observations: pd.DataFrame,
    horizons: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    first_adverse = {}
    for threshold in thresholds:
        slug = threshold_slug(threshold)
        values = observations[f"first_adverse_{slug}pct_bars_from_entry"]
        finite = values.loc[values.notna()]
        first_adverse[slug] = {
            "hit_count": int(len(finite)),
            "hit_bars_sum": int(finite.sum()),
        }
    horizon_output = {}
    for horizon in horizons:
        slug = horizon_slug(horizon)
        outcomes = {}
        for threshold in thresholds:
            threshold_key = threshold_slug(threshold)
            counts = observations[
                f"outcome_{slug}_vs_{threshold_key}pct"
            ].value_counts()
            outcomes[threshold_key] = {
                str(key): int(value) for key, value in sorted(counts.items())
            }
        horizon_output[slug] = {
            "fully_observed_count": int(
                observations[f"horizon_{slug}_fully_observed"].astype(bool).sum()
            ),
            "target_hit_count": int(
                observations[f"target_hit_{slug}"].astype(bool).sum()
            ),
            "adverse_lower_sum_pct": finite_sum(
                observations[f"max_adverse_pre_target_lower_{slug}_pct"]
            ),
            "adverse_upper_sum_pct": finite_sum(
                observations[f"max_adverse_pre_target_upper_{slug}_pct"]
            ),
            "outcomes": outcomes,
        }
    target = observations["target_touch_bars_from_entry"]
    finite_target = target.loc[target.notna()]
    return {
        "observation_count": int(len(observations)),
        "omitted_entry_counts": {},
        "observations_by_age": {
            f"{int(key)}m": int(value)
            for key, value in sorted(observations["entry_age_minutes"].value_counts().items())
        },
        "entry_distance_sum_pct": finite_sum(
            observations["entry_distance_from_target_pct"]
        ),
        "peak_distance_sum_pct": finite_sum(
            observations["peak_distance_from_target_pct"]
        ),
        "drawdown_sum_pct": finite_sum(observations["drawdown_from_peak_pct"]),
        "target_touch_count": int(len(finite_target)),
        "target_touch_bars_sum": int(finite_target.sum()),
        "first_adverse": first_adverse,
        "horizons": horizon_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("1m", "5m"), required=True)
    parser.add_argument("--maximum-fill-days", type=int, default=180)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument(
        "--entry-ages-minutes",
        type=parse_int_list,
        default=DEFAULT_ENTRY_AGES_MINUTES,
    )
    parser.add_argument(
        "--horizons-minutes",
        type=parse_int_list,
        default=DEFAULT_HORIZONS_MINUTES,
    )
    parser.add_argument(
        "--adverse-thresholds-pct",
        type=parse_float_list,
        default=DEFAULT_ADVERSE_THRESHOLDS_PCT,
    )
    args = parser.parse_args()
    interval_minutes = int(args.timeframe.removesuffix("m"))
    total_start = perf_counter()
    read_start = perf_counter()
    frame = read_candles(args.input, interval_minutes * 60_000, args.maximum_rows)
    read_seconds = perf_counter() - read_start
    dataset_start = perf_counter()
    signals, observations, summary = build_series_dataset(
        frame,
        "BENCHMARK",
        args.timeframe,
        args.entry_ages_minutes,
        args.horizons_minutes,
        args.adverse_thresholds_pct,
        args.maximum_fill_days,
    )
    dataset_seconds = perf_counter() - dataset_start
    outcomes = aggregate_observations(
        observations,
        args.horizons_minutes,
        args.adverse_thresholds_pct,
    )
    outcomes["omitted_entry_counts"] = summary["omitted_entry_counts"]
    direction_counts = signals["direction_sign"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "engine": "python",
                "mode": "outcomes",
                "rows": len(frame),
                "strict_signals": len(signals),
                "lower_signals": int(direction_counts.get(1, 0)),
                "upper_signals": int(direction_counts.get(-1, 0)),
                "signal_index_checksum": int(signals["signal_index"].sum()),
                "statuses": summary["resolution_status_counts"],
                "outcomes": outcomes,
                "read_seconds": read_seconds,
                "outcome_seconds": dataset_seconds,
                "total_seconds": perf_counter() - total_start,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
