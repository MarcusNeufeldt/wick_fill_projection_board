#!/usr/bin/env python3
"""Generate fast, normal, and extreme wick-fill scenarios for one pinned signal.

The output contains rescaled *real* historical OHLC trajectories.  It is
conditional on the pinned wick eventually filling and must never be presented
as an unconditional forecast or a trade instruction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_conditional_path_library import (
    FEATURE_COLUMNS,
    FIVE_MINUTES_MS,
    detect_strict_signals,
    read_five_minute_file,
    resample_to_fifteen_minutes,
    utc_iso,
)


FEATURE_WEIGHTS = {
    "body_pct_of_range": 0.8,
    "dominant_wick_pct_of_range": 0.8,
    "opposite_wick_pct_of_range": 0.8,
    "range_pct_of_close": 0.6,
    "range_vs_prior_20_median": 0.6,
    "volume_vs_prior_20_mean": 0.5,
    "aligned_prior_1h_return_pct": 0.5,
}
STATE_WEIGHTS = {
    "current_move_pct": 1.8,
    "peak_move_pct": 1.2,
    "drawdown_from_peak_pct": 1.2,
    "elapsed_bars": 0.4,
}
CATEGORY_PENALTIES = {"asset": 0.35, "timeframe": 0.25, "direction": 0.10}


def parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp required: {value}")
    return int(parsed.timestamp() * 1000)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalized_price(price: float | np.ndarray, target: float, direction_sign: int | np.ndarray) -> float | np.ndarray:
    return np.asarray(direction_sign) * (np.asarray(price) / target - 1.0) * 100.0


def log_distance(left: np.ndarray | float, right: float, floor: float) -> np.ndarray:
    safe_left = np.maximum(np.asarray(left, dtype=float), 0.0) + floor
    safe_right = max(float(right), 0.0) + floor
    return np.abs(np.log(safe_left / safe_right))


def robust_scales(events: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        values = events[name].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        fallback = float(np.std(values))
        scales[name] = mad if mad > 1e-12 else (fallback if fallback > 1e-12 else 1.0)
    return scales


def state_for_live_path(
    frame: pd.DataFrame,
    signal_index: int,
    current_index: int,
    direction_sign: int,
    target: float,
    interval_minutes: int,
) -> dict[str, float]:
    observed = frame.iloc[signal_index : current_index + 1]
    current_move = float(normalized_price(float(frame["close"].iat[current_index]), target, direction_sign))
    if direction_sign == 1:
        peak = float(np.max(normalized_price(observed["high"].to_numpy(dtype=float), target, direction_sign)))
    else:
        peak = float(np.max(normalized_price(observed["low"].to_numpy(dtype=float), target, direction_sign)))
    return {
        "elapsed_bars": float(current_index - signal_index),
        "elapsed_minutes": float((current_index - signal_index) * interval_minutes),
        "current_move_pct": current_move,
        "peak_move_pct": peak,
        "drawdown_from_peak_pct": max(0.0, peak - current_move),
    }


def read_library_paths(paths_dir: Path, path_files: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fields = [
        "episode_id",
        "asset",
        "timeframe",
        "direction",
        "direction_sign",
        "offset_bars",
        "normalized_open_pct",
        "normalized_high_pct",
        "normalized_low_pct",
        "normalized_close_pct",
    ]
    for path_file in sorted(set(path_files)):
        path = paths_dir / path_file
        if not path.exists():
            raise FileNotFoundError(f"Missing path file referenced by library: {path}")
        print(json.dumps({"stage": "read_paths", "path_file": path_file}), flush=True)
        frames.append(pd.read_csv(path, compression="gzip", usecols=fields))
    if not frames:
        raise RuntimeError("No trajectory path files are available")
    return pd.concat(frames, ignore_index=True)


def choose_best_alignment(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    current_signal: pd.Series,
    current_state: dict[str, float],
    target_asset: str,
    target_timeframe: str,
    target_direction: str,
) -> pd.DataFrame:
    scales = robust_scales(events)
    target_features = {field: float(current_signal[field]) for field in FEATURE_COLUMNS}
    event_scores = events[["episode_id", "asset", "timeframe", "direction", "signal_to_fill_bars", *FEATURE_COLUMNS]].copy()
    feature_score = np.zeros(len(event_scores), dtype=float)
    for field, weight in FEATURE_WEIGHTS.items():
        feature_score += weight * np.abs(event_scores[field].to_numpy(dtype=float) - target_features[field]) / scales[field]
    event_scores["feature_distance"] = feature_score
    category_distance = (
        (event_scores["asset"] != target_asset).astype(float) * CATEGORY_PENALTIES["asset"]
        + (event_scores["timeframe"] != target_timeframe).astype(float) * CATEGORY_PENALTIES["timeframe"]
        + (event_scores["direction"] != target_direction).astype(float) * CATEGORY_PENALTIES["direction"]
    )
    event_scores["category_distance"] = category_distance
    value = paths.merge(
        event_scores[["episode_id", "signal_to_fill_bars", "feature_distance", "category_distance"]],
        on="episode_id",
        how="inner",
        validate="many_to_one",
    )
    value = value.loc[(value["offset_bars"] > 0) & (value["offset_bars"] < value["signal_to_fill_bars"])].copy()
    if value.empty:
        raise RuntimeError("No historical path states are available for alignment")
    directional_peak_value = np.where(
        value["direction_sign"].to_numpy(dtype=int) == 1,
        value["normalized_high_pct"].to_numpy(dtype=float),
        value["normalized_low_pct"].to_numpy(dtype=float),
    )
    value["directional_peak_at_bar"] = directional_peak_value
    value = value.sort_values(["episode_id", "offset_bars"], kind="stable")
    value["alignment_peak_move_pct"] = value.groupby("episode_id", sort=False)["directional_peak_at_bar"].cummax()
    value["alignment_current_move_pct"] = value["normalized_close_pct"].astype(float)
    value["alignment_drawdown_pct"] = np.maximum(
        0.0, value["alignment_peak_move_pct"] - value["alignment_current_move_pct"]
    )
    value["remaining_to_fill_bars"] = value["signal_to_fill_bars"] - value["offset_bars"]
    value["state_distance"] = (
        STATE_WEIGHTS["current_move_pct"]
        * log_distance(value["alignment_current_move_pct"].to_numpy(), current_state["current_move_pct"], 0.30)
        + STATE_WEIGHTS["peak_move_pct"]
        * log_distance(value["alignment_peak_move_pct"].to_numpy(), current_state["peak_move_pct"], 0.30)
        + STATE_WEIGHTS["drawdown_from_peak_pct"]
        * log_distance(value["alignment_drawdown_pct"].to_numpy(), current_state["drawdown_from_peak_pct"], 0.30)
        + STATE_WEIGHTS["elapsed_bars"]
        * log_distance(value["offset_bars"].to_numpy(), current_state["elapsed_bars"], 3.0)
    )
    value["match_score"] = value["state_distance"] + 0.5 * value["feature_distance"] + value["category_distance"]

    # Future counter-direction excursion, calculated within the same historical episode.
    value["future_peak_move_pct"] = value.groupby("episode_id", sort=False)["directional_peak_at_bar"].transform(
        lambda series: series.iloc[::-1].cummax().iloc[::-1]
    )
    best_indices = value.groupby("episode_id", sort=False)["match_score"].idxmin()
    return value.loc[best_indices].sort_values("match_score", kind="stable").reset_index(drop=True)


def empirical_percentile(values: np.ndarray, value: float) -> float:
    if len(values) == 0:
        return 0.5
    return float(np.mean(values <= value))


def select_scenarios(matches: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    cohort = matches.head(min(top_k, len(matches))).copy().reset_index(drop=True)
    if len(cohort) < 12:
        raise RuntimeError("Fewer than 12 comparable historical states; scenario selection would be too unstable")
    duration = np.log1p(cohort["remaining_to_fill_bars"].to_numpy(dtype=float))
    excursion = cohort["future_peak_move_pct"].to_numpy(dtype=float)
    duration_rank = pd.Series(duration).rank(pct=True, method="average").to_numpy(dtype=float)
    excursion_rank = pd.Series(excursion).rank(pct=True, method="average").to_numpy(dtype=float)
    cohort["joint_risk_percentile"] = 0.55 * duration_rank + 0.45 * excursion_rank
    cohort["scenario_match_percentile"] = pd.Series(cohort["match_score"]).rank(pct=True, method="average").to_numpy(dtype=float)

    median_duration = float(np.median(duration))
    median_excursion = float(np.median(excursion))
    duration_scale = max(float(np.median(np.abs(duration - median_duration)) * 1.4826), float(np.std(duration)), 1e-6)
    excursion_scale = max(float(np.median(np.abs(excursion - median_excursion)) * 1.4826), float(np.std(excursion)), 1e-6)

    choices: list[tuple[str, str, pd.Series]] = []
    used: set[str] = set()
    definitions = [
        ("fast", "A real comparable episode near the lower joint duration/excursion risk percentile.", 0.25),
        ("normal", "The closest joint medoid of remaining duration and future adverse excursion among comparable episodes.", None),
        ("extreme", "A real comparable episode near the upper joint duration/excursion risk percentile; it is a stress reference, not a worst-case guarantee.", 0.90),
    ]
    for name, description, target_percentile in definitions:
        available = cohort.loc[~cohort["episode_id"].isin(used)].copy()
        if target_percentile is None:
            scenario_score = (
                np.abs(np.log1p(available["remaining_to_fill_bars"].to_numpy(dtype=float)) - median_duration) / duration_scale
                + np.abs(available["future_peak_move_pct"].to_numpy(dtype=float) - median_excursion) / excursion_scale
                + 0.20 * available["scenario_match_percentile"].to_numpy(dtype=float)
            )
        else:
            scenario_score = (
                np.abs(available["joint_risk_percentile"].to_numpy(dtype=float) - target_percentile)
                + 0.18 * available["scenario_match_percentile"].to_numpy(dtype=float)
            )
        selected = available.iloc[int(np.argmin(scenario_score))]
        used.add(str(selected["episode_id"]))
        choices.append((name, description, selected))
    scenarios: list[dict[str, Any]] = []
    for name, description, selected in choices:
        selected_joint_risk = float(selected["joint_risk_percentile"])
        scenarios.append(
            {
                "name": name,
                "description": description,
                "episode_id": str(selected["episode_id"]),
                "historical_asset": str(selected["asset"]),
                "historical_timeframe": str(selected["timeframe"]),
                "historical_direction": str(selected["direction"]),
                "alignment_offset_bars": int(selected["offset_bars"]),
                "remaining_to_fill_bars": int(selected["remaining_to_fill_bars"]),
                "historical_alignment_current_move_pct": float(selected["alignment_current_move_pct"]),
                "historical_alignment_peak_move_pct": float(selected["alignment_peak_move_pct"]),
                "historical_alignment_drawdown_pct": float(selected["alignment_drawdown_pct"]),
                "future_max_away_move_pct": float(selected["future_peak_move_pct"]),
                "joint_risk_percentile": selected_joint_risk,
                "matched_cohort_size": int(len(cohort)),
                "matched_cohort_tail_at_or_above_fraction": float(
                    np.mean(cohort["joint_risk_percentile"].to_numpy(dtype=float) >= selected_joint_risk)
                ),
                "match_score": float(selected["match_score"]),
            }
        )
    return scenarios


def projected_candles(
    paths: pd.DataFrame,
    scenario: dict[str, Any],
    current_target: float,
    current_direction_sign: int,
    current_move_pct: float,
    projection_start_open_time_ms: int,
    interval_minutes: int,
) -> tuple[list[dict[str, float | int]], float]:
    source = paths.loc[paths["episode_id"].eq(scenario["episode_id"])].copy()
    source = source.sort_values("offset_bars", kind="stable")
    alignment_offset = int(scenario["alignment_offset_bars"])
    aligned = source.loc[source["offset_bars"].eq(alignment_offset)]
    if len(aligned) != 1:
        raise RuntimeError(f"Expected exactly one alignment row for {scenario['episode_id']}")
    historical_move = float(aligned["normalized_close_pct"].iat[0])
    if historical_move <= 0:
        raise RuntimeError("Cannot rescale a historical alignment state at or beyond the wick target")
    scale = current_move_pct / historical_move
    future = source.loc[source["offset_bars"].gt(alignment_offset)].copy()
    if len(future) != int(scenario["remaining_to_fill_bars"]):
        raise RuntimeError("Scenario path length does not agree with selected remaining duration")

    def project(normalized: float) -> float:
        return current_target * (1.0 + current_direction_sign * scale * normalized / 100.0)

    output: list[dict[str, float | int]] = []
    for step, row in enumerate(future.itertuples(index=False), start=1):
        # An upper-wick analogue can be direction-normalized into a lower-wick
        # projection (and vice versa). That reflection reverses the numerical
        # order of the historical high and low. Project every OHLC value first,
        # then restore the invariant high >= open/close >= low for chart data.
        projected_open = project(float(row.normalized_open_pct))
        projected_high = project(float(row.normalized_high_pct))
        projected_low = project(float(row.normalized_low_pct))
        projected_close = project(float(row.normalized_close_pct))
        output.append(
            {
                "time": int(projection_start_open_time_ms // 1000 + (step - 1) * interval_minutes * 60),
                "open": round(projected_open, 6),
                "high": round(max(projected_open, projected_high, projected_low, projected_close), 6),
                "low": round(min(projected_open, projected_high, projected_low, projected_close), 6),
                "close": round(projected_close, 6),
            }
        )
    return output, scale


def actual_candles(frame: pd.DataFrame, start: int, end: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for candle in frame.iloc[start : end + 1].itertuples(index=False):
        rows.append(
            {
                "time": int(candle.open_time // 1000),
                "open": round(float(candle.open), 6),
                "high": round(float(candle.high), 6),
                "low": round(float(candle.low), 6),
                "close": round(float(candle.close), 6),
            }
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="ETHUSDT", choices=("ETHUSDT", "BTCUSDT"))
    parser.add_argument("--timeframe", default="5m", choices=("5m", "15m"))
    parser.add_argument("--source", type=Path, help="5m source CSV; defaults to the selected five-year asset file")
    parser.add_argument("--library-dir", type=Path, default=root / "data" / "conditional_path_library_5y")
    parser.add_argument("--signal-time", default="2026-09-16T18:35:00Z")
    parser.add_argument("--as-of", help="Last closed candle's open time in UTC; default is the latest source candle")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--actual-lookback-bars", type=int, default=480)
    parser.add_argument(
        "--signal-context-bars",
        type=int,
        default=96,
        help="Completed candles to include before the pinned signal, even when the signal predates the normal lookback",
    )
    parser.add_argument("--output", type=Path, default=root / "data" / "current_conditional_path_scenarios.json")
    args = parser.parse_args()
    if args.top_k < 12:
        parser.error("--top-k must be at least 12")
    if args.actual_lookback_bars < 1 or args.signal_context_bars < 0:
        parser.error("--actual-lookback-bars must be positive and --signal-context-bars cannot be negative")

    source_path = args.source or (root / "data" / f"{args.asset}_5m_5y.csv")
    raw = read_five_minute_file(source_path.resolve())
    frame = raw if args.timeframe == "5m" else resample_to_fifteen_minutes(raw)
    signal_time_ms = parse_utc(args.signal_time)
    matching_signal = frame.loc[frame["open_time"].eq(signal_time_ms)]
    if matching_signal.empty:
        raise RuntimeError(f"Signal candle does not exist in {args.timeframe} source: {args.signal_time}")
    signals = detect_strict_signals(frame, args.asset, args.timeframe)
    signal_rows = signals.loc[signals["open_time"].eq(signal_time_ms)]
    if len(signal_rows) != 1:
        raise RuntimeError("Pinned candle is not a strict wick signal with enough preceding context")
    signal = signal_rows.iloc[0]
    signal_index = int(signal["bar_index"])
    as_of_index = len(frame) - 1
    if args.as_of:
        as_of_time_ms = parse_utc(args.as_of)
        matching_as_of = frame.index[frame["open_time"].eq(as_of_time_ms)]
        if len(matching_as_of) != 1:
            raise RuntimeError(f"As-of candle does not exist: {args.as_of}")
        as_of_index = int(matching_as_of[0])
    if as_of_index <= signal_index:
        raise RuntimeError("As-of candle must be later than the pinned signal")
    direction_sign = int(signal["direction_sign"])
    target = float(signal["wick_target"])
    later = frame.iloc[signal_index + 1 : as_of_index + 1]
    filled = (later["low"] <= target).any() if direction_sign == 1 else (later["high"] >= target).any()
    if filled:
        raise RuntimeError("Pinned wick has already been fully touched by the supplied as-of candle")
    departed = (later["close"] > float(signal["high"])).any() if direction_sign == 1 else (
        later["close"] < float(signal["low"])
    ).any()
    if not departed:
        raise RuntimeError(
            "Pinned wick has not yet made the confirmed close beyond the opposite signal extreme required for a conditional path projection"
        )
    current_state = state_for_live_path(
        frame,
        signal_index,
        as_of_index,
        direction_sign,
        target,
        int(signal["interval_minutes"]),
    )
    if current_state["current_move_pct"] <= 0:
        raise RuntimeError("Current close is not on the away-from-wick side required for conditional projection")
    # A fixed tail-only chart window can omit an older pinned signal completely.
    # Always retain a small lead-in before the signal plus every observed candle
    # through the current snapshot, so the dashboard makes the actual signal and
    # its observed move-away leg inspectable.
    tail_start_index = max(0, as_of_index - args.actual_lookback_bars + 1)
    signal_context_start_index = max(0, signal_index - args.signal_context_bars)
    actual_start_index = min(tail_start_index, signal_context_start_index)

    library_dir = args.library_dir.resolve()
    episodes_path = library_dir / "episodes.csv"
    summary_path = library_dir / "summary.json"
    if not episodes_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Conditional path library has not been built")
    episodes = pd.read_csv(episodes_path)
    episodes["signal_open_time_ms"] = pd.to_numeric(episodes["signal_open_time_ms"], errors="raise").astype("int64")
    # Do not infer the backing unit of Pandas datetime int64 values. Recent
    # Windows/Pandas builds can retain microsecond resolution here, which would
    # incorrectly place completed episodes in 1970 and leak future paths into
    # a historical analogue cohort.
    episodes["fill_open_time_ms"] = pd.to_datetime(episodes["fill_open_time_utc"], utc=True).map(
        lambda value: int(value.timestamp() * 1000)
    )
    # At the live signal time, only historical episodes already resolved are allowed in the cohort.
    eligible = episodes.loc[episodes["fill_open_time_ms"] < signal_time_ms].copy()
    if eligible.empty:
        raise RuntimeError("No historical completed episodes resolved before the pinned signal")
    # A scenario is rendered as real candles on the pinned chart timeframe.  Mixing
    # a 15m path into a 5m candle chart would invent intrabar candles and distort
    # elapsed time, so cross-timeframe observations remain available to later
    # quantile models but are not direct V1 trajectory candidates.
    trajectory_eligible = eligible.loc[eligible["timeframe"].eq(args.timeframe)].copy()
    if trajectory_eligible.empty:
        raise RuntimeError(f"No historical completed {args.timeframe} episodes resolved before the pinned signal")
    print(
        json.dumps(
            {
                "stage": "eligible_episodes",
                "all_completed_before_signal": int(len(eligible)),
                "same_timeframe_trajectory_candidates": int(len(trajectory_eligible)),
            }
        ),
        flush=True,
    )
    paths = read_library_paths(library_dir / "paths", trajectory_eligible["path_file"].tolist())
    paths = paths.loc[paths["episode_id"].isin(set(trajectory_eligible["episode_id"]))].copy()
    matched = choose_best_alignment(
        trajectory_eligible,
        paths,
        signal,
        current_state,
        args.asset,
        args.timeframe,
        str(signal["direction"]),
    )
    scenarios = select_scenarios(matched, args.top_k)
    interval_minutes = int(signal["interval_minutes"])
    projection_start = int(frame["open_time"].iat[as_of_index]) + interval_minutes * 60_000
    for scenario in scenarios:
        candles, scale = projected_candles(
            paths,
            scenario,
            target,
            direction_sign,
            current_state["current_move_pct"],
            projection_start,
            interval_minutes,
        )
        scenario["normalization_scale"] = round(scale, 8)
        scenario["projected_candles"] = candles
        scenario["projected_terminal_fill_candle_utc"] = utc_iso(
            (candles[-1]["time"] * 1000 + interval_minutes * 60_000) if candles else projection_start
        )

    top_matches = []
    for row in matched.head(min(10, len(matched))).itertuples(index=False):
        top_matches.append(
            {
                "episode_id": str(row.episode_id),
                "asset": str(row.asset),
                "timeframe": str(row.timeframe),
                "direction": str(row.direction),
                "alignment_offset_bars": int(row.offset_bars),
                "remaining_to_fill_bars": int(row.remaining_to_fill_bars),
                "future_max_away_move_pct": float(row.future_peak_move_pct),
                "match_score": float(row.match_score),
            }
        )
    cohort = matched.head(min(args.top_k, len(matched)))
    remaining_minutes = cohort["remaining_to_fill_bars"].to_numpy(dtype=float) * interval_minutes
    future_move = cohort["future_peak_move_pct"].to_numpy(dtype=float)
    output = {
        "schema_version": "1.0.0",
        "method": "state-conditioned empirical historical trajectory scenarios",
        "conditionality": "Every projected path is a rescaled real historical episode that eventually fully fills its wick; this is not an unconditional fill probability or a trade recommendation.",
        "library": {
            "directory": str(library_dir),
            "eligible_completed_episodes_before_signal": int(len(eligible)),
            "same_timeframe_trajectory_candidates_before_signal": int(len(trajectory_eligible)),
            "top_k_state_matched_episodes": int(len(cohort)),
            "matching_features": FEATURE_WEIGHTS,
            "matching_state_weights": STATE_WEIGHTS,
            "category_penalties": CATEGORY_PENALTIES,
        },
        "pinned_signal": {
            "asset": args.asset,
            "timeframe": args.timeframe,
            "signal_open_time_utc": utc_iso(signal_time_ms),
            "direction": str(signal["direction"]),
            "direction_sign": direction_sign,
            "wick_target": target,
            "signal_open": float(signal["open"]),
            "signal_high": float(signal["high"]),
            "signal_low": float(signal["low"]),
            "signal_close": float(signal["close"]),
            "signal_features": {field: float(signal[field]) for field in FEATURE_COLUMNS},
        },
        "current_state": {
            **{key: round(float(value), 8) for key, value in current_state.items()},
            "as_of_open_time_utc": utc_iso(int(frame["open_time"].iat[as_of_index])),
            "as_of_close": float(frame["close"].iat[as_of_index]),
        },
        "cohort_distribution": {
            "remaining_time_to_fill_minutes": {
                "p25": float(np.quantile(remaining_minutes, 0.25)),
                "p50": float(np.quantile(remaining_minutes, 0.50)),
                "p90": float(np.quantile(remaining_minutes, 0.90)),
            },
            "future_max_away_move_pct": {
                "p50": float(np.quantile(future_move, 0.50)),
                "p90": float(np.quantile(future_move, 0.90)),
            },
        },
        "actual_window": {
            "start_open_time_utc": utc_iso(int(frame["open_time"].iat[actual_start_index])),
            "end_open_time_utc": utc_iso(int(frame["open_time"].iat[as_of_index])),
            "signal_context_bars": int(args.signal_context_bars),
            "signal_candle_index_in_window": int(signal_index - actual_start_index),
        },
        "actual_candles": actual_candles(frame, actual_start_index, as_of_index),
        "scenarios": scenarios,
        "top_matches": top_matches,
        "limitations": [
            "Scenario paths are conditional on eventual fill and are not unconditional price forecasts.",
            "Direct projected candles use only the pinned timeframe; mixing historical 15m bars into a 5m candle path would invent intrabar detail and distort time.",
            "The current feature weights are deliberately transparent starting values; they must be learned or tuned only inside chronological replay folds.",
            "The scenarios preserve real joint historical trajectories but still need path-coverage walk-forward validation before risk use.",
        ],
    }
    output_path = args.output.resolve()
    atomic_write_json(output_path, output)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "scenarios": [
                    {
                        "name": scenario["name"],
                        "episode_id": scenario["episode_id"],
                        "remaining_bars": scenario["remaining_to_fill_bars"],
                        "future_max_away_move_pct": scenario["future_max_away_move_pct"],
                    }
                    for scenario in scenarios
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
