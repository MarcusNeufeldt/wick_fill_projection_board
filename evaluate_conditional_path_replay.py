"""Chronological replay validation for the V1 conditional wick-path engine.

This evaluates the product question rather than an unconditional fill label:
at fixed, observable offsets after a strict wick signal, do the three selected
historical trajectories form a useful envelope for the remaining time-to-fill
and counter-direction excursion of later, unseen clean-fill episodes?

Every candidate trajectory must have completed before the replay snapshot bar
closed. That prevents a path that was still unresolved at the historical
decision time from leaking into the candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conditional_wick_assets import default_library_dir
from build_conditional_path_scenarios import (
    prepare_path_states,
    project_at,
    read_library_paths,
    with_fill_close_time_ms,
)


def utc_iso(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_five_minute_states(library_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 5m cross-asset library using the serving engine's state semantics."""
    episodes = pd.read_csv(library_dir / "episodes.csv")
    events = with_fill_close_time_ms(episodes.loc[episodes["timeframe"].eq("5m")].copy())
    if events.empty:
        raise RuntimeError("The trajectory library has no 5m completed episodes")
    events["signal_open_time_ms"] = pd.to_numeric(events["signal_open_time_ms"], errors="raise").astype("int64")
    paths = read_library_paths(library_dir / "paths", events["path_file"].tolist())
    paths = paths.loc[paths["episode_id"].astype(str).isin(set(events["episode_id"].astype(str)))].copy()
    states = prepare_path_states(events, paths)
    return events.reset_index(drop=True), paths.reset_index(drop=True), states.reset_index(drop=True)


def evenly_spaced(values: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(values) <= maximum:
        return values
    positions = np.unique(np.linspace(0, len(values) - 1, num=maximum, dtype=int))
    return values.iloc[positions]


def metric_rate(cases: pd.DataFrame, name: str) -> float | None:
    if cases.empty:
        return None
    return round(float(cases[name].astype(float).mean()), 4)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path, default=default_library_dir(root))
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--offsets", default="12,60,240,960", help="Comma-separated observable 5m offsets after signal.")
    parser.add_argument("--cases-per-offset", type=int, default=24)
    parser.add_argument("--min-candidates", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_library_dir(root) / "replay_validation_5m.json",
    )
    parser.add_argument("--markdown-output", type=Path, default=root / "PATH_REPLAY_VALIDATION.md")
    args = parser.parse_args()
    offsets = sorted({int(value.strip()) for value in args.offsets.split(",") if value.strip()})
    if not offsets or any(value <= 0 for value in offsets):
        parser.error("--offsets must contain positive integers")
    if args.holdout_months < 1 or args.cases_per_offset < 1 or args.min_candidates < 12 or args.top_k < 12:
        parser.error("holdout/case counts must be positive; min-candidates and top-k must be at least 12")

    events, paths, states = load_five_minute_states(args.library_dir.resolve())
    last_signal = pd.to_datetime(events["signal_open_time_ms"].max(), unit="ms", utc=True)
    holdout_start = last_signal - pd.DateOffset(months=args.holdout_months)
    holdout_start_ms = int(holdout_start.timestamp() * 1000)
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {
        "no_observable_state": 0,
        "not_confirmed_departure": 0,
        "not_away_from_wick": 0,
        "too_few_candidates": 0,
    }

    for offset in offsets:
        target_pool = events.loc[
            events["signal_open_time_ms"].ge(holdout_start_ms)
            & events["signal_to_fill_bars"].gt(offset)
        ].sort_values("signal_open_time_ms", kind="stable")
        target_pool = evenly_spaced(target_pool, args.cases_per_offset)
        for target in target_pool.itertuples(index=False):
            target_series = pd.Series(target._asdict())
            target_path = states.loc[states["episode_id"].eq(target_series["episode_id"])]
            snapshot = target_path.loc[target_path["offset_bars"].eq(offset)]
            if len(snapshot) != 1:
                skipped["no_observable_state"] += 1
                continue
            row = snapshot.iloc[0]
            if not bool(row["candidate_after_departure"]):
                skipped["not_confirmed_departure"] += 1
                continue
            if float(row["alignment_current_move_pct"]) <= 0:
                skipped["not_away_from_wick"] += 1
                continue
            state = {
                "elapsed_bars": float(offset),
                "current_move_pct": float(row["alignment_current_move_pct"]),
                "peak_move_pct": float(row["alignment_peak_move_pct"]),
                "drawdown_from_peak_pct": float(row["alignment_drawdown_pct"]),
            }
            observation_open_time_ms = int(target_series["signal_open_time_ms"]) + offset * int(
                target_series["interval_minutes"]
            ) * 60_000
            observation_close_time_ms = observation_open_time_ms + int(target_series["interval_minutes"]) * 60_000
            projection = project_at(
                events,
                paths,
                target_series,
                state,
                observation_close_time_ms,
                float(target_series["wick_target"]),
                int(target_series["direction_sign"]),
                observation_close_time_ms,
                int(target_series["interval_minutes"]),
                args.top_k,
                path_states=states,
            )
            matches = projection["matched_states"]
            eligible_episode_count = int(len(projection["eligible_events"]))
            if len(matches) < args.min_candidates:
                skipped["too_few_candidates"] += 1
                continue
            scenarios = projection["scenarios"]
            by_name = {scenario["name"]: scenario for scenario in scenarios}
            duration_values = np.asarray([scenario["remaining_to_fill_bars"] for scenario in scenarios], dtype=float)
            excursion_values = np.asarray(
                [scenario["projected_future_max_away_move_pct"] for scenario in scenarios], dtype=float
            )
            actual_remaining = int(target_series["signal_to_fill_bars"]) - offset
            actual_future_peak = max(0.0, float(row["future_peak_move_pct"]))
            duration_covered = float(duration_values.min()) <= actual_remaining <= float(duration_values.max())
            excursion_covered = float(excursion_values.min()) <= actual_future_peak <= float(excursion_values.max())
            normal_duration = float(by_name["normal"]["remaining_to_fill_bars"])
            normal_excursion = float(by_name["normal"]["projected_future_max_away_move_pct"])
            records.append(
                {
                    "episode_id": str(target_series["episode_id"]),
                    "asset": str(target_series["asset"]),
                    "direction": str(target_series["direction"]),
                    "signal_open_time_utc": str(target_series["signal_open_time_utc"]),
                    "observation_offset_bars": offset,
                    "observation_open_time_utc": utc_iso(observation_open_time_ms),
                    "actual_remaining_bars": actual_remaining,
                    "actual_future_peak_move_pct": actual_future_peak,
                    "eligible_completed_episode_count": eligible_episode_count,
                    "duration_envelope_covered": duration_covered,
                    "excursion_envelope_covered": excursion_covered,
                    "joint_envelope_covered": duration_covered and excursion_covered,
                    "normal_duration_abs_log_error": abs(np.log1p(normal_duration) - np.log1p(actual_remaining)),
                    "normal_excursion_abs_error_pct": abs(normal_excursion - actual_future_peak),
                    "fast_remaining_bars": int(by_name["fast"]["remaining_to_fill_bars"]),
                    "normal_remaining_bars": int(normal_duration),
                    "extreme_remaining_bars": int(by_name["extreme"]["remaining_to_fill_bars"]),
                    "fast_projected_future_peak_move_pct": float(
                        by_name["fast"]["projected_future_max_away_move_pct"]
                    ),
                    "normal_projected_future_peak_move_pct": normal_excursion,
                    "extreme_projected_future_peak_move_pct": float(
                        by_name["extreme"]["projected_future_max_away_move_pct"]
                    ),
                }
            )
        print(json.dumps({"stage": "offset_complete", "offset_bars": offset, "cases_so_far": len(records)}), flush=True)

    cases = pd.DataFrame(records)
    per_offset: dict[str, dict[str, Any]] = {}
    for offset in offsets:
        subset = cases.loc[cases["observation_offset_bars"].eq(offset)] if not cases.empty else cases
        per_offset[str(offset)] = {
            "case_count": int(len(subset)),
            "duration_envelope_coverage": metric_rate(subset, "duration_envelope_covered"),
            "excursion_envelope_coverage": metric_rate(subset, "excursion_envelope_covered"),
            "joint_envelope_coverage": metric_rate(subset, "joint_envelope_covered"),
            "median_normal_duration_abs_log_error": (
                round(float(subset["normal_duration_abs_log_error"].median()), 4) if not subset.empty else None
            ),
            "median_normal_excursion_abs_error_pct": (
                round(float(subset["normal_excursion_abs_error_pct"].median()), 4) if not subset.empty else None
            ),
        }
    aggregate = {
        "case_count": int(len(cases)),
        "duration_envelope_coverage": metric_rate(cases, "duration_envelope_covered"),
        "excursion_envelope_coverage": metric_rate(cases, "excursion_envelope_covered"),
        "joint_envelope_coverage": metric_rate(cases, "joint_envelope_covered"),
        "median_normal_duration_abs_log_error": (
            round(float(cases["normal_duration_abs_log_error"].median()), 4) if not cases.empty else None
        ),
        "median_normal_excursion_abs_error_pct": (
            round(float(cases["normal_excursion_abs_error_pct"].median()), 4) if not cases.empty else None
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": "1.1.0",
        "purpose": "Chronological V1 replay of conditional wick-fill path envelopes; not a fill-probability backtest.",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "library": str(args.library_dir.resolve()),
        "timeframe": "5m",
        "assets": sorted(events["asset"].unique().tolist()),
        "holdout_start_utc": utc_iso(holdout_start_ms),
        "last_signal_utc": utc_iso(int(events["signal_open_time_ms"].max())),
        "observation_offsets_bars": offsets,
        "top_k": args.top_k,
        "minimum_resolved_candidates": args.min_candidates,
        "leakage_guard": "Each candidate terminal fill candle had to close at or before the replay snapshot candle closed.",
        "projection_semantics": {
            "departure": "Target and analogue states must be at or after the confirmed close-at-or-beyond-opposite-extreme departure.",
            "future_excursion_window": "The projected and actual excursion windows begin after the observable snapshot and include the terminal fill candle OHLC envelope.",
        },
        "conditional_population": "Only strict signals with a clean departure followed by a complete wick fill within the library cap.",
        "aggregate": aggregate,
        "per_offset": per_offset,
        "skipped": skipped,
        "cases_csv": str(args.output.with_name(args.output.stem + "_cases.csv").resolve()),
    }
    atomic_write_json(args.output, summary)
    cases_path = args.output.with_name(args.output.stem + "_cases.csv")
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(cases_path, index=False)

    lines = [
        "# V1 Conditional Path Replay Validation",
        "",
        f"Generated: `{summary['generated_at_utc']}`.",
        "",
        "This is not a binary fill-probability test. It replays later clean-fill episodes at fixed, observable 5-minute offsets and asks whether the three historical path scenarios cover the eventual remaining time and maximum move away from the wick.",
        "",
        "Leakage guard: a candidate analogue is eligible only when its own terminal fill candle has closed at or before the replay snapshot. Both target and analogue states must be after confirmed departure. The evaluation population is therefore conditional on strict signals that later made a clean departure and fully filled within the 180-day library cap.",
        "",
        "## Aggregate",
        "",
        f"- Cases: **{aggregate['case_count']}**",
        f"- Duration-envelope coverage: **{aggregate['duration_envelope_coverage']}**",
        f"- Excursion-envelope coverage: **{aggregate['excursion_envelope_coverage']}**",
        f"- Joint coverage: **{aggregate['joint_envelope_coverage']}**",
        f"- Median normal duration log-error: **{aggregate['median_normal_duration_abs_log_error']}**",
        f"- Median normal excursion absolute error: **{aggregate['median_normal_excursion_abs_error_pct']} percentage points**",
        "",
        "## By snapshot offset",
        "",
        "| Offset | Cases | Duration coverage | Excursion coverage | Joint coverage |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for offset in offsets:
        item = per_offset[str(offset)]
        lines.append(
            f"| {offset} bars | {item['case_count']} | {item['duration_envelope_coverage']} | {item['excursion_envelope_coverage']} | {item['joint_envelope_coverage']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The three displayed paths are deliberately real historical episodes, not calibrated confidence intervals. Coverage is a usefulness diagnostic for risk sizing, not a probability guarantee. The next model gate is to compare this V1 replay against survival/quantile/conformal envelopes on the same frozen replay protocol before introducing a neural trajectory generator.",
            "",
            f"Detailed rows: `{cases_path.name}`. Machine-readable summary: `{args.output.name}`.",
        ]
    )
    atomic_write_text(args.markdown_output, "\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "markdown_output": str(args.markdown_output), "aggregate": aggregate}))


if __name__ == "__main__":
    main()
