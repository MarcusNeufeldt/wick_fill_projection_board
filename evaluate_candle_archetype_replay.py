#!/usr/bin/env python3
"""Replay a pinned candle archetype against later, target-like clean fills.

The default reference is the ETHUSDT 5m candle at 2026-09-16 18:35 UTC.
Upper and lower wick signals are mirrored.  Each replay query compares the
existing adaptive matcher with the experimental archetype-gated matcher while
using only historical episodes that had resolved before that query snapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_conditional_path_library import detect_strict_signals, read_five_minute_file
from build_conditional_path_scenarios import (
    prepare_path_states,
    project_at,
    read_library_paths,
    with_fill_close_time_ms,
)
from candle_archetype import (
    ARCHETYPE_BLEND_WEIGHT,
    ARCHETYPE_COMPONENT_SCALES,
    ARCHETYPE_POOL_SIZE,
    archetype_distance,
    archetype_profile,
)
from conditional_wick_assets import SUPPORTED_ASSETS, default_library_dir
from evaluate_conditional_path_replay import (
    atomic_write_json,
    atomic_write_text,
    evenly_spaced,
    utc_iso,
)
from native_wick_matcher import NativeStateIndex, load_runtime_cache


DEFAULT_CHECKPOINTS = (1, 3, 6, 12, 24, 48, 96, 192, 576, 1_440)


def row_spans(states: pd.DataFrame) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    for episode_id, positions in states.groupby("episode_id", sort=False).indices.items():
        ordered = np.sort(np.asarray(positions, dtype=np.int64))
        if len(ordered) and int(ordered[-1] - ordered[0] + 1) != len(ordered):
            raise RuntimeError(f"Runtime rows are not contiguous for {episode_id}")
        spans[str(episode_id)] = (int(ordered[0]), int(ordered[-1]) + 1)
    return spans


def load_library(
    library_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, NativeStateIndex, dict[str, tuple[int, int]]]:
    episodes = with_fill_close_time_ms(pd.read_csv(library_dir / "episodes.csv"))
    events = episodes.loc[episodes["timeframe"].eq("5m")].copy().reset_index(drop=True)
    events["signal_open_time_ms"] = pd.to_numeric(
        events["signal_open_time_ms"], errors="raise"
    ).astype("int64")
    runtime = load_runtime_cache(library_dir, "5m")
    if runtime is None:
        paths = read_library_paths(library_dir / "paths", events["path_file"].tolist())
        states = prepare_path_states(events, paths).reset_index(drop=True)
        native_index = NativeStateIndex.from_frames(events, states, "5m")
    else:
        states, native_index = runtime
        states = states.reset_index(drop=True)
    return events, states, native_index, row_spans(states)


def reference_signal(source: Path, asset: str, signal_time_ms: int) -> pd.Series:
    frame = read_five_minute_file(source)
    signals = detect_strict_signals(frame, asset, "5m")
    match = signals.loc[signals["open_time"].eq(signal_time_ms)]
    if len(match) != 1:
        raise RuntimeError("Reference candle is not one unique strict 5m wick signal")
    return match.iloc[0]


def near_clone_audit(
    events: pd.DataFrame, reference: pd.Series, reference_time_ms: int
) -> dict[str, Any]:
    """Count progressively stricter mirrored matches known before the pin."""
    past = events.loc[
        events["signal_open_time_ms"].lt(reference_time_ms)
        & events["fill_close_time_ms"].le(reference_time_ms)
    ].copy()
    shape = (
        past["body_pct_of_range"]
        .sub(float(reference["body_pct_of_range"]))
        .abs()
        .le(0.005)
        & past["dominant_wick_pct_of_range"]
        .sub(float(reference["dominant_wick_pct_of_range"]))
        .abs()
        .le(0.02)
        & past["opposite_wick_pct_of_range"]
        .sub(float(reference["opposite_wick_pct_of_range"]))
        .abs()
        .le(0.02)
    )
    absolute_range = past["range_pct_of_close"].div(
        float(reference["range_pct_of_close"])
    ).between(0.75, 1.25)
    relative_range = past["range_vs_prior_20_median"].div(
        float(reference["range_vs_prior_20_median"])
    ).between(0.75, 1.25)
    relative_volume = past["volume_vs_prior_20_mean"].div(
        float(reference["volume_vs_prior_20_mean"])
    ).between(2.0 / 3.0, 1.5)
    prior_return = past["aligned_prior_1h_return_pct"].sub(
        float(reference["aligned_prior_1h_return_pct"])
    ).abs().le(1.0)
    by_asset = {
        str(asset): int(count)
        for asset, count in past.loc[shape].groupby("asset").size().items()
    }
    return {
        "eligible_completed_episodes_before_reference": int(len(past)),
        "shape_close_count": int(shape.sum()),
        "shape_close_by_asset": by_asset,
        "shape_plus_absolute_range_count": int((shape & absolute_range).sum()),
        "shape_plus_absolute_and_relative_range_count": int(
            (shape & absolute_range & relative_range).sum()
        ),
        "full_configuration_count": int(
            (shape & absolute_range & relative_range & relative_volume & prior_return).sum()
        ),
        "shape_tolerances": {
            "body_share_percentage_points": 0.5,
            "dominant_wick_share_percentage_points": 2.0,
            "opposite_wick_share_percentage_points": 2.0,
        },
        "absolute_and_relative_range_tolerance": "within 25% of the reference",
    }


def close_distance(candle: dict[str, Any], target: float, direction_sign: int) -> float:
    close = float(candle["close"])
    value = (close / target - 1.0) * 100.0 if direction_sign == 1 else (1.0 - close / target) * 100.0
    return max(0.0, value)


def checkpoint_path_mae(
    scenario: dict[str, Any],
    target_states: pd.DataFrame,
    observation_offset: int,
    fill_offset: int,
    target: float,
    direction_sign: int,
    checkpoints: tuple[int, ...],
) -> float:
    actual_by_offset = dict(
        zip(
            target_states["offset_bars"].astype(int),
            target_states["alignment_current_move_pct"].astype(float),
        )
    )
    projected = scenario["projected_candles"]
    errors: list[float] = []
    for horizon in checkpoints:
        actual_offset = observation_offset + horizon
        actual = 0.0 if actual_offset >= fill_offset else max(0.0, float(actual_by_offset[actual_offset]))
        predicted = 0.0 if horizon >= len(projected) else close_distance(
            projected[horizon - 1], target, direction_sign
        )
        errors.append(abs(predicted - actual))
    return float(np.mean(errors))


def summarize(cases: pd.DataFrame) -> dict[str, Any]:
    if cases.empty:
        return {"case_count": 0}
    return {
        "case_count": int(len(cases)),
        "distinct_target_episodes": int(cases["episode_id"].nunique()),
        "mean_checkpoint_path_mae_pct": round(float(cases["checkpoint_path_mae_pct"].mean()), 6),
        "median_checkpoint_path_mae_pct": round(float(cases["checkpoint_path_mae_pct"].median()), 6),
        "median_duration_abs_log_error": round(float(cases["duration_abs_log_error"].median()), 6),
        "median_excursion_abs_error_pct": round(float(cases["excursion_abs_error_pct"].median()), 6),
        "duration_envelope_coverage": round(float(cases["duration_envelope_covered"].mean()), 6),
        "excursion_envelope_coverage": round(float(cases["excursion_envelope_covered"].mean()), 6),
        "joint_envelope_coverage": round(float(cases["joint_envelope_covered"].mean()), 6),
    }


def clustered_bootstrap(
    paired: pd.DataFrame, challenger: str, repetitions: int = 2_000
) -> dict[str, Any]:
    if paired.empty:
        return {"paired_case_count": 0}
    grouped = {
        key: group.reset_index(drop=True)
        for key, group in paired.groupby("episode_id", sort=False)
    }
    episode_ids = np.asarray(list(grouped), dtype=object)
    rng = np.random.default_rng(16_2026)

    def improvement(frame: pd.DataFrame) -> float:
        adaptive = float(frame["adaptive"].mean())
        challenger_error = float(frame[challenger].mean())
        return 1.0 - challenger_error / adaptive if adaptive > 0 else 0.0

    observed = improvement(paired)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        frame = pd.concat([grouped[item] for item in sampled], ignore_index=True)
        draws[index] = improvement(frame)
    return {
        "paired_case_count": int(len(paired)),
        "distinct_target_episodes": int(len(episode_ids)),
        "challenger": challenger,
        "relative_mean_path_mae_improvement": round(observed, 6),
        "episode_clustered_95pct_interval": [
            round(float(np.quantile(draws, 0.025)), 6),
            round(float(np.quantile(draws, 0.975)), 6),
        ],
        "lower_error_case_fraction": round(
            float((paired[challenger] < paired["adaptive"]).mean()), 6
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=SUPPORTED_ASSETS, default="ETHUSDT")
    parser.add_argument("--signal-time", default="2026-09-16T18:35:00Z")
    parser.add_argument("--source", type=Path, default=root / "data" / "ETHUSDT_5m_5y.csv")
    parser.add_argument("--library-dir", type=Path, default=default_library_dir(root))
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--max-archetype-distance", type=float, default=2.0)
    parser.add_argument("--offsets", default="12,60,240,960,1440,2880,8640")
    parser.add_argument("--cases-per-offset", type=int, default=48)
    parser.add_argument("--minimum-matches", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "candle_archetype_replay_5m.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=root / "CANDLE_ARCHETYPE_EXPERIMENT.md",
    )
    args = parser.parse_args()
    offsets = tuple(sorted({int(value) for value in args.offsets.split(",") if value.strip()}))
    if not offsets or min(offsets) < 1:
        parser.error("--offsets must contain positive bar counts")
    if args.top_k < 12 or args.minimum_matches < 12 or args.cases_per_offset < 1:
        parser.error("top-k and minimum-matches must be >=12; cases-per-offset must be positive")
    if args.minimum_matches > ARCHETYPE_POOL_SIZE:
        parser.error(f"--minimum-matches cannot exceed archetype pool size {ARCHETYPE_POOL_SIZE}")

    signal_time_ms = int(pd.Timestamp(args.signal_time).timestamp() * 1_000)
    reference = reference_signal(args.source.resolve(), args.asset, signal_time_ms)
    reference_profile = archetype_profile(reference)
    events, states, native_index, spans = load_library(args.library_dir.resolve())
    clone_audit = near_clone_audit(events, reference, signal_time_ms)
    events["reference_archetype_distance"] = archetype_distance(events, reference)
    holdout_start_ms = int(
        (pd.Timestamp(args.signal_time) - pd.DateOffset(months=args.holdout_months)).timestamp()
        * 1_000
    )
    target_events = events.loc[
        events["signal_open_time_ms"].between(holdout_start_ms, signal_time_ms - 1)
        & events["reference_archetype_distance"].le(args.max_archetype_distance)
    ].sort_values("signal_open_time_ms", kind="stable")

    records: list[dict[str, Any]] = []
    skipped = {"no_state": 0, "not_departed": 0, "too_few_matches": 0}
    for offset in offsets:
        targets = target_events.loc[target_events["signal_to_fill_bars"].gt(offset)]
        targets = evenly_spaced(targets, args.cases_per_offset)
        for target in targets.itertuples(index=False):
            target_series = pd.Series(target._asdict())
            target_states = states.loc[states["episode_id"].eq(target_series["episode_id"])]
            snapshot = target_states.loc[target_states["offset_bars"].eq(offset)]
            if len(snapshot) != 1:
                skipped["no_state"] += 1
                continue
            state_row = snapshot.iloc[0]
            if not bool(state_row["candidate_after_departure"]):
                skipped["not_departed"] += 1
                continue
            current_state = {
                "elapsed_bars": float(offset),
                "current_move_pct": float(state_row["alignment_current_move_pct"]),
                "peak_move_pct": float(state_row["alignment_peak_move_pct"]),
                "drawdown_from_peak_pct": float(state_row["alignment_drawdown_pct"]),
            }
            observation_close_ms = int(target_series["signal_open_time_ms"]) + (offset + 1) * 300_000
            for mode in ("adaptive", "blended", "archetype"):
                projection = project_at(
                    events,
                    states,
                    target_series,
                    current_state,
                    observation_close_ms,
                    float(target_series["wick_target"]),
                    int(target_series["direction_sign"]),
                    observation_close_ms,
                    5,
                    args.top_k,
                    path_states=states,
                    native_state_index=native_index,
                    episode_row_spans=spans,
                    matching_mode=mode,
                    archetype_pool_size=ARCHETYPE_POOL_SIZE,
                    archetype_blend_weight=ARCHETYPE_BLEND_WEIGHT,
                )
                if projection["insufficient_matches"] or len(projection["matched_states"]) < args.minimum_matches:
                    skipped["too_few_matches"] += 1
                    continue
                scenarios = {item["name"]: item for item in projection["scenarios"]}
                normal = scenarios["normal"]
                actual_remaining = int(target_series["signal_to_fill_bars"]) - offset
                actual_excursion = max(0.0, float(state_row["future_peak_move_pct"]))
                durations = np.asarray([item["remaining_to_fill_bars"] for item in scenarios.values()], dtype=float)
                excursions = np.asarray([item["projected_future_max_away_move_pct"] for item in scenarios.values()], dtype=float)
                duration_covered = float(durations.min()) <= actual_remaining <= float(durations.max())
                excursion_covered = float(excursions.min()) <= actual_excursion <= float(excursions.max())
                records.append(
                    {
                        "matching_mode": mode,
                        "episode_id": str(target_series["episode_id"]),
                        "asset": str(target_series["asset"]),
                        "direction": str(target_series["direction"]),
                        "signal_open_time_utc": str(target_series["signal_open_time_utc"]),
                        "reference_archetype_distance": float(target_series["reference_archetype_distance"]),
                        "observation_offset_bars": offset,
                        "actual_remaining_bars": actual_remaining,
                        "actual_future_peak_move_pct": actual_excursion,
                        "normal_episode_id": str(normal["episode_id"]),
                        "checkpoint_path_mae_pct": checkpoint_path_mae(
                            normal,
                            target_states,
                            offset,
                            int(target_series["signal_to_fill_bars"]),
                            float(target_series["wick_target"]),
                            int(target_series["direction_sign"]),
                            DEFAULT_CHECKPOINTS,
                        ),
                        "duration_abs_log_error": abs(
                            np.log1p(float(normal["remaining_to_fill_bars"]))
                            - np.log1p(actual_remaining)
                        ),
                        "excursion_abs_error_pct": abs(
                            float(normal["projected_future_max_away_move_pct"])
                            - actual_excursion
                        ),
                        "duration_envelope_covered": duration_covered,
                        "excursion_envelope_covered": excursion_covered,
                        "joint_envelope_covered": duration_covered and excursion_covered,
                    }
                )
        print(json.dumps({"stage": "offset_complete", "offset_bars": offset, "rows": len(records)}), flush=True)

    cases = pd.DataFrame(records)
    if cases.empty:
        raise RuntimeError("No valid replay cases were produced")
    per_mode = {
        mode: summarize(cases.loc[cases["matching_mode"].eq(mode)])
        for mode in ("adaptive", "blended", "archetype")
    }
    paired = cases.pivot_table(
        index=["episode_id", "observation_offset_bars"],
        columns="matching_mode",
        values="checkpoint_path_mae_pct",
        aggfunc="first",
    ).dropna().reset_index()
    comparisons = {
        challenger: clustered_bootstrap(paired, challenger)
        for challenger in ("blended", "archetype")
    }
    per_offset: dict[str, Any] = {}
    for offset in offsets:
        subset = cases.loc[cases["observation_offset_bars"].eq(offset)]
        per_offset[str(offset)] = {
            mode: summarize(subset.loc[subset["matching_mode"].eq(mode)])
            for mode in ("adaptive", "blended", "archetype")
        }

    cases_path = args.output.with_name(args.output.stem + "_cases.csv")
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(cases_path, index=False)
    summary = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "purpose": "Direction-mirrored replay of signals resembling the pinned September 16 ETH 5m candle.",
        "reference": {
            "asset": args.asset,
            "timeframe": "5m",
            "signal_time_utc": args.signal_time,
            "profile": reference_profile,
            "component_scales": ARCHETYPE_COMPONENT_SCALES,
            "maximum_distance": args.max_archetype_distance,
        },
        "target_population": {
            "holdout_start_utc": utc_iso(holdout_start_ms),
            "holdout_end_exclusive_utc": args.signal_time,
            "target_like_completed_episode_count": int(len(target_events)),
            "assets": sorted(target_events["asset"].unique().tolist()),
            "directions": sorted(target_events["direction"].unique().tolist()),
        },
        "historical_clone_audit": clone_audit,
        "observation_offsets_bars": offsets,
        "checkpoints_after_snapshot_bars": DEFAULT_CHECKPOINTS,
        "archetype_pool_size": ARCHETYPE_POOL_SIZE,
        "archetype_blend_weight": ARCHETYPE_BLEND_WEIGHT,
        "per_mode": per_mode,
        "comparisons_vs_adaptive": comparisons,
        "per_offset": per_offset,
        "skipped": skipped,
        "cases_csv": str(cases_path.resolve()),
        "leakage_guard": "Every analogue terminal fill candle closed before the replay snapshot; target futures were used only for scoring.",
    }
    atomic_write_json(args.output.resolve(), summary)

    blend_comparison = comparisons["blended"]
    hard_comparison = comparisons["archetype"]
    blend_relative = blend_comparison.get("relative_mean_path_mae_improvement")
    blend_interval = blend_comparison.get(
        "episode_clustered_95pct_interval", [None, None]
    )
    hard_relative = hard_comparison.get("relative_mean_path_mae_improvement")
    hard_interval = hard_comparison.get(
        "episode_clustered_95pct_interval", [None, None]
    )
    lines = [
        "# Pinned Candle Archetype Experiment",
        "",
        f"Reference: `{args.asset} 5m {args.signal_time}`. Upper and lower wicks are direction-mirrored.",
        "",
        "## Reference candle configuration",
        "",
        f"- Body: **{reference_profile['body_pct_of_range'] * 100:.3f}%** of total candle range",
        f"- Dominant wick: **{reference_profile['dominant_wick_pct_of_range'] * 100:.3f}%** of range",
        f"- Opposite wick: **{reference_profile['opposite_wick_pct_of_range'] * 100:.3f}%** of range",
        f"- Total range: **{reference_profile['range_pct_of_close']:.3f}%** of close and **{reference_profile['range_vs_prior_20_median']:.3f}x** the prior 20-candle median range",
        f"- Volume: **{reference_profile['volume_vs_prior_20_mean']:.3f}x** the prior 20-candle mean",
        f"- Direction-aligned prior one-hour return: **{reference_profile['aligned_prior_1h_return_pct']:.3f}%**",
        "",
        "Target-like cases have root-mean-square normalized archetype distance <= 2.0. One distance unit is 0.75 percentage points of body share, 3 points of either wick share, a 1.5x range factor, a 2x volume factor, or 1 percentage point of aligned prior return.",
        "",
        "## How many close historical candles existed before the pin?",
        "",
        f"Among **{clone_audit['eligible_completed_episodes_before_reference']}** five-asset 5m clean-fill episodes resolved before the reference candle:",
        "",
        f"- **{clone_audit['shape_close_count']}** matched the body and mirrored wick shares within 0.5 / 2 / 2 percentage points.",
        f"- **{clone_audit['shape_plus_absolute_range_count']}** also had absolute candle range within 25% of the reference.",
        f"- **{clone_audit['shape_plus_absolute_and_relative_range_count']}** also matched the relative-volatility regime within 25%.",
        f"- **{clone_audit['full_configuration_count']}** matched the full range, volume, and prior-return configuration.",
        "",
        "The exact full configuration therefore has no historical clone; the matcher must use graded similarity rather than pretend identical examples exist.",
        "",
        f"Target-like signals from the prior {args.holdout_months} months with clean fills available in the current library: **{len(target_events)}**.",
        f"Paired replay cases: **{blend_comparison.get('paired_case_count', 0)}** across **{blend_comparison.get('distinct_target_episodes', 0)}** distinct target episodes.",
        "",
        "## Result",
        "",
        f"- Adaptive mean checkpoint path MAE: **{per_mode['adaptive'].get('mean_checkpoint_path_mae_pct')} percentage points**",
        f"- Soft {ARCHETYPE_BLEND_WEIGHT:.0%} blend mean checkpoint path MAE: **{per_mode['blended'].get('mean_checkpoint_path_mae_pct')} percentage points**",
        f"- Soft-blend relative improvement: **{None if blend_relative is None else round(blend_relative * 100, 2)}%**",
        f"- Soft-blend episode-clustered 95% interval: **{None if blend_interval[0] is None else round(blend_interval[0] * 100, 2)}% to {None if blend_interval[1] is None else round(blend_interval[1] * 100, 2)}%**",
        f"- Soft blend lower-error case fraction: **{blend_comparison.get('lower_error_case_fraction')}**",
        f"- Hard archetype mean checkpoint path MAE: **{per_mode['archetype'].get('mean_checkpoint_path_mae_pct')} percentage points**",
        f"- Hard-archetype relative improvement: **{None if hard_relative is None else round(hard_relative * 100, 2)}%**",
        f"- Hard-archetype episode-clustered 95% interval: **{None if hard_interval[0] is None else round(hard_interval[0] * 100, 2)}% to {None if hard_interval[1] is None else round(hard_interval[1] * 100, 2)}%**",
        "",
        "## By signal age",
        "",
        "| Snapshot age | Adaptive path MAE | Soft blend path MAE | Hard archetype path MAE | Paired cases |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for offset in offsets:
        item = per_offset[str(offset)]
        lines.append(
            f"| {offset} bars ({offset * 5 / 60:.1f}h) | {item['adaptive'].get('mean_checkpoint_path_mae_pct')} | {item['blended'].get('mean_checkpoint_path_mae_pct')} | {item['archetype'].get('mean_checkpoint_path_mae_pct')} | {item['adaptive'].get('case_count')} |"
        )
    lines.extend(
        [
            "",
            "This is a conditional clean-fill path experiment, not an eventual-fill probability test. The dashboard keeps this matcher opt-in until the chronological evidence is clearly useful.",
            "",
            f"Detailed rows: `{cases_path.name}`. Machine summary: `{args.output.name}`.",
        ]
    )
    atomic_write_text(args.markdown_output.resolve(), "\n".join(lines) + "\n")
    print(json.dumps({"summary": str(args.output.resolve()), "report": str(args.markdown_output.resolve()), "comparisons_vs_adaptive": comparisons}))


if __name__ == "__main__":
    main()
