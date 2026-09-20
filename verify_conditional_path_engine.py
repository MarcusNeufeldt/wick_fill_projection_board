#!/usr/bin/env python3
"""Fast regression checks for the generated conditional wick path artifact.

This intentionally recalculates the historical eligibility cutoff outside the
scenario builder. It protects against datetime-unit regressions that could
silently put future completed paths into a pinned signal's analogue cohort.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=root / "data" / "current_conditional_path_scenarios.json")
    parser.add_argument("--library", type=Path, default=root / "data" / "conditional_path_library_5y")
    parser.add_argument("--chart", type=Path, default=root / "conditional-wick-path-dashboard.html")
    args = parser.parse_args()

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    episodes = pd.read_csv(args.library / "episodes.csv")
    fill_open_ms = pd.to_datetime(episodes["fill_open_time_utc"], utc=True).map(lambda value: int(value.timestamp() * 1000))
    signal_ms = int(pd.Timestamp(scenario["pinned_signal"]["signal_open_time_utc"]).timestamp() * 1000)
    expected_all = int((fill_open_ms < signal_ms).sum())
    expected_timeframe = int(
        ((fill_open_ms < signal_ms) & episodes["timeframe"].eq(scenario["pinned_signal"]["timeframe"])).sum()
    )
    actual = scenario["actual_candles"]
    interval_seconds = int(scenario["pinned_signal"]["timeframe"].replace("m", "")) * 60
    target = float(scenario["pinned_signal"]["wick_target"])
    lower = scenario["pinned_signal"]["direction"] == "lower_wick"
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
        path_checks.append(
            {
                "name": item["name"],
                "valid_ohlc": valid_ohlc,
                "terminal_touches_target": touches,
                "starts_after_actual": contiguous,
                "same_timeframe": same_timeframe,
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
        == scenario["library"]["eligible_completed_episodes_before_signal"],
        "eligibility_same_timeframe_matches": expected_timeframe
        == scenario["library"]["same_timeframe_trajectory_candidates_before_signal"],
        "all_path_checks_pass": all(
            bool(item["valid_ohlc"])
            and bool(item["terminal_touches_target"])
            and bool(item["starts_after_actual"])
            and bool(item["same_timeframe"])
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
