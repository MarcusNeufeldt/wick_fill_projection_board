#!/usr/bin/env python3
"""Fast regression checks for the generated conditional wick path artifact.

This intentionally recalculates the historical eligibility cutoff outside the
scenario builder. It protects against datetime-unit regressions that could
silently put future completed paths into a pinned signal's analogue cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from conditional_wick_assets import default_library_dir


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=root / "data" / "current_conditional_path_scenarios.json")
    parser.add_argument("--library", type=Path, default=default_library_dir(root))
    parser.add_argument("--chart", type=Path, default=root / "conditional-wick-path-dashboard.html")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    episodes = pd.read_csv(args.library / "episodes.csv")
    fill_open_ms = pd.to_datetime(episodes["fill_open_time_utc"], utc=True).map(lambda value: int(value.timestamp() * 1000))
    episode_interval_ms = episodes["interval_minutes"].astype(int) * 60_000
    fill_close_ms = fill_open_ms + episode_interval_ms
    snapshot_close_ms = int(pd.Timestamp(scenario["current_state"]["as_of_close_time_utc"]).timestamp() * 1000)
    expected_all = int((fill_close_ms <= snapshot_close_ms).sum())
    expected_timeframe = int(
        ((fill_close_ms <= snapshot_close_ms) & episodes["timeframe"].eq(scenario["pinned_signal"]["timeframe"])).sum()
    )
    actual = scenario["actual_candles"]
    interval_seconds = int(scenario["pinned_signal"]["timeframe"].replace("m", "")) * 60
    target = float(scenario["pinned_signal"]["wick_target"])
    lower = scenario["pinned_signal"]["direction"] == "lower_wick"
    current_move = float(scenario["current_state"]["current_move_pct"])
    path_checks: list[dict[str, object]] = []
    for item in scenario["scenarios"]:
        candles = item["projected_candles"]
        last = candles[-1]
        valid_ohlc = all(
            float(candle["high"]) >= max(float(candle["open"]), float(candle["close"]))
            and float(candle["low"]) <= min(float(candle["open"]), float(candle["close"]))
            and float(candle["high"]) >= float(candle["low"])
            for candle in candles
        )
        touches = float(last["low"]) <= target if lower else float(last["high"]) >= target
        contiguous = int(candles[0]["time"]) == int(actual[-1]["time"]) + interval_seconds
        same_timeframe = item["historical_timeframe"] == scenario["pinned_signal"]["timeframe"]
        episode = episodes.loc[episodes["episode_id"].eq(item["episode_id"])]
        if len(episode) != 1:
            raise RuntimeError(f"Scenario episode is absent from the local library: {item['episode_id']}")
        departure_ok = int(item["alignment_offset_bars"]) >= int(episode["signal_to_departure_bars"].iat[0])
        directional_values = [
            (float(candle["high"]) / target - 1.0) * 100.0
            if lower
            else (1.0 - float(candle["low"]) / target) * 100.0
            for candle in candles
        ]
        displayed_peak = max(0.0, max(directional_values))
        displayed_additional = max(0.0, displayed_peak - current_move)
        projected_metric_matches = math.isclose(
            displayed_peak,
            float(item["projected_future_max_away_move_pct"]),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        additional_metric_matches = math.isclose(
            displayed_additional,
            float(item["projected_additional_adverse_move_pct"]),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        path_checks.append(
            {
                "name": item["name"],
                "valid_ohlc": valid_ohlc,
                "terminal_touches_target": touches,
                "starts_after_actual": contiguous,
                "same_timeframe": same_timeframe,
                "alignment_after_departure": departure_ok,
                "displayed_projected_metric_matches": projected_metric_matches,
                "displayed_additional_metric_matches": additional_metric_matches,
            }
        )

    chart = args.chart.read_text(encoding="utf-8")
    match = re.search(r'<script id="scenario-data" type="application/json">(.*?)</script>', chart, re.DOTALL)
    if match is None:
        raise RuntimeError("Generated chart does not embed a scenario-data payload")
    embedded = json.loads(match.group(1))
    chart_matches = (
        embedded["pinned_signal"]["signal_open_time_utc"] == scenario["pinned_signal"]["signal_open_time_utc"]
        and [item["episode_id"] for item in embedded["scenarios"]]
        == [item["episode_id"] for item in scenario["scenarios"]]
    )
    checks = {
        "eligibility_all_matches": expected_all
        == scenario["library"]["eligible_completed_episodes_before_snapshot"],
        "eligibility_same_timeframe_matches": expected_timeframe
        == scenario["library"]["same_timeframe_trajectory_candidates_before_snapshot"],
        "all_path_checks_pass": all(
            bool(item["valid_ohlc"])
            and bool(item["terminal_touches_target"])
            and bool(item["starts_after_actual"])
            and bool(item["same_timeframe"])
            and bool(item["alignment_after_departure"])
            and bool(item["displayed_projected_metric_matches"])
            and bool(item["displayed_additional_metric_matches"])
            for item in path_checks
        ),
        "embedded_chart_matches_scenario": chart_matches,
    }
    print(
        json.dumps(
            {
                "checks": checks,
                "expected_eligible_all": expected_all,
                "expected_eligible_same_timeframe": expected_timeframe,
                "path_checks": path_checks,
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
