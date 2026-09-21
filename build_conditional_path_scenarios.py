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
from typing import Any, Mapping

import numpy as np
import pandas as pd

from conditional_wick_assets import (
    SUPPORTED_ASSETS,
    default_library_dir,
    one_minute_library_dir,
)
from build_conditional_path_library import (
    FEATURE_COLUMNS,
    FIVE_MINUTES_MS,
    detect_strict_signals,
    read_five_minute_file,
    read_one_minute_file,
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
ONE_MINUTE_MATCH_EXACT_BARS = 240
ONE_MINUTE_MATCH_FIVE_MINUTE_BARS = 1_440


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
        values = values[np.isfinite(values)]
        if not len(values):
            scales[name] = 1.0
            continue
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


def with_fill_close_time_ms(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an explicit candle-close timestamp for each completed episode."""
    value = events.copy()
    if "fill_close_time_ms" in value.columns:
        value["fill_close_time_ms"] = pd.to_numeric(value["fill_close_time_ms"], errors="raise").astype("int64")
        return value
    required = {"fill_open_time_utc", "interval_minutes"}
    missing = sorted(required.difference(value.columns))
    if missing:
        raise RuntimeError(f"Episode records are missing fields required for fill-close eligibility: {', '.join(missing)}")
    fill_open_time_ms = pd.to_datetime(value["fill_open_time_utc"], utc=True).map(
        lambda timestamp: int(timestamp.timestamp() * 1000)
    )
    interval_minutes = pd.to_numeric(value["interval_minutes"], errors="raise").astype("int64")
    value["fill_close_time_ms"] = (fill_open_time_ms + interval_minutes * 60_000).astype("int64")
    return value


def eligible_episodes_at_snapshot(events: pd.DataFrame, snapshot_close_time_ms: int) -> pd.DataFrame:
    """Keep only episodes whose terminal fill candle was known when the snapshot closed."""
    value = with_fill_close_time_ms(events)
    return value.loc[value["fill_close_time_ms"].le(int(snapshot_close_time_ms))].copy()


def has_confirmed_departure(later: pd.DataFrame, direction_sign: int, opposite_extreme: float) -> bool:
    """Use the library's inclusive close-at-or-beyond-opposite-extreme departure rule."""
    if direction_sign == 1:
        return bool(later["close"].ge(float(opposite_extreme)).any())
    return bool(later["close"].le(float(opposite_extreme)).any())


def prepare_path_states(events: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Build observable historical states using the same signal, phase, and future-window semantics."""
    required_event_fields = {"episode_id", "signal_to_departure_bars", "signal_to_fill_bars"}
    missing_event_fields = sorted(required_event_fields.difference(events.columns))
    if missing_event_fields:
        raise RuntimeError(
            "Episode records are missing fields required for post-departure path states: "
            + ", ".join(missing_event_fields)
        )
    event_fields = ["episode_id", "signal_to_departure_bars", "signal_to_fill_bars"]
    value = paths.merge(events[event_fields], on="episode_id", how="inner", validate="many_to_one")
    value = value.loc[
        value["offset_bars"].ge(0) & value["offset_bars"].le(value["signal_to_fill_bars"])
    ].copy()
    if value.empty:
        raise RuntimeError("No trajectory rows remain through the terminal fill candle")
    value = value.sort_values(["episode_id", "offset_bars"], kind="stable").reset_index(drop=True)
    value["directional_peak_at_bar"] = np.where(
        value["direction_sign"].to_numpy(dtype=int) == 1,
        value["normalized_high_pct"].to_numpy(dtype=float),
        value["normalized_low_pct"].to_numpy(dtype=float),
    )
    value["alignment_peak_move_pct"] = value.groupby("episode_id", sort=False)[
        "directional_peak_at_bar"
    ].cummax()
    value["alignment_current_move_pct"] = value["normalized_close_pct"].astype(float)
    value["alignment_drawdown_pct"] = np.maximum(
        0.0, value["alignment_peak_move_pct"] - value["alignment_current_move_pct"]
    )
    value["remaining_to_fill_bars"] = value["signal_to_fill_bars"] - value["offset_bars"]
    # The displayed projection begins after the observable snapshot and includes
    # the terminal fill candle.  Use the same forward candle-envelope window for
    # matching, rendering, and replay scoring.
    value["future_peak_move_pct"] = value.groupby("episode_id", sort=False)["directional_peak_at_bar"].transform(
        lambda series: series.iloc[::-1].cummax().iloc[::-1].shift(-1)
    )
    value["candidate_after_departure"] = (
        value["offset_bars"].gt(0)
        & value["offset_bars"].lt(value["signal_to_fill_bars"])
        & value["offset_bars"].ge(value["signal_to_departure_bars"])
        & value["alignment_current_move_pct"].gt(0)
        & value["future_peak_move_pct"].notna()
    )
    return value


def sampled_matching_states(path_states: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Keep an adaptive, past-only 1m alignment grid without altering render paths.

    One-minute candles remain intact in every returned scenario.  Only the
    historical *alignment snapshots* are thinned after the first four hours:
    every minute through 240 bars, every five minutes through one day, then
    every fifteen minutes.  This prevents a sparse set of very long 1m paths
    from making each five-minute live recalculation impractically slow.
    """
    if timeframe != "1m":
        return path_states
    offsets = path_states["offset_bars"].to_numpy(dtype=np.int64)
    candidate = path_states["candidate_after_departure"].to_numpy(dtype=bool)
    keep = candidate & (
        (offsets <= ONE_MINUTE_MATCH_EXACT_BARS)
        | ((offsets <= ONE_MINUTE_MATCH_FIVE_MINUTE_BARS) & (offsets % 5 == 0))
        | (offsets % 15 == 0)
    )
    return path_states.loc[keep].copy()


def choose_best_alignment(
    events: pd.DataFrame,
    path_states: pd.DataFrame,
    current_signal: pd.Series,
    current_state: dict[str, float],
    target_asset: str,
    target_timeframe: str,
    target_direction: str,
    native_state_index: Any | None = None,
    snapshot_close_time_ms: int | None = None,
    top_k: int | None = None,
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
    if native_state_index is not None and snapshot_close_time_ms is not None and top_k is not None:
        native = native_state_index.choose(
            events=events,
            path_states=path_states,
            event_scores=event_scores,
            snapshot_close_time_ms=snapshot_close_time_ms,
            current_state=current_state,
            state_weights=STATE_WEIGHTS,
            top_k=top_k,
        )
        if native is not None:
            return native
    value = path_states.loc[path_states["candidate_after_departure"]].copy().merge(
        event_scores[["episode_id", "signal_to_fill_bars", "feature_distance", "category_distance"]],
        on="episode_id",
        how="inner",
        validate="many_to_one",
    )
    if value.empty:
        raise RuntimeError("No post-departure historical path states are available for alignment")
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
    best_indices = value.groupby("episode_id", sort=False)["match_score"].idxmin()
    return value.loc[best_indices].sort_values("match_score", kind="stable").reset_index(drop=True)


def empirical_percentile(values: np.ndarray, value: float) -> float:
    if len(values) == 0:
        return 0.5
    return float(np.mean(values <= value))


def projected_coordinate_matches(matches: pd.DataFrame, current_move_pct: float) -> pd.DataFrame:
    """Map each comparable future excursion into the pinned wick's price coordinates before ranking it."""
    value = matches.copy()
    historical_move = value["alignment_current_move_pct"].to_numpy(dtype=float)
    if np.any(historical_move <= 0):
        raise RuntimeError("Historical alignment states must remain on the away-from-wick side before rescaling")
    scale = float(current_move_pct) / historical_move
    historical_future = np.maximum(0.0, value["future_peak_move_pct"].to_numpy(dtype=float))
    projected_future = np.maximum(0.0, historical_future * scale)
    value["normalization_scale"] = scale
    value["historical_future_max_away_move_pct"] = historical_future
    value["projected_future_max_away_move_pct"] = projected_future
    value["projected_additional_adverse_move_pct"] = np.maximum(0.0, projected_future - float(current_move_pct))
    return value


def select_scenarios(matches: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    cohort = matches.head(min(top_k, len(matches))).copy().reset_index(drop=True)
    if len(cohort) < 12:
        raise RuntimeError("Fewer than 12 comparable historical states; scenario selection would be too unstable")
    duration = np.log1p(cohort["remaining_to_fill_bars"].to_numpy(dtype=float))
    excursion = cohort["projected_future_max_away_move_pct"].to_numpy(dtype=float)
    duration_rank = pd.Series(duration).rank(pct=True, method="average").to_numpy(dtype=float)
    excursion_rank = pd.Series(excursion).rank(pct=True, method="average").to_numpy(dtype=float)
    cohort["joint_risk_score"] = 0.55 * duration_rank + 0.45 * excursion_rank
    score_values = cohort["joint_risk_score"].to_numpy(dtype=float)
    cohort["joint_risk_score_percentile"] = np.asarray(
        [empirical_percentile(score_values, value) for value in score_values], dtype=float
    )
    cohort["scenario_match_percentile"] = pd.Series(cohort["match_score"]).rank(pct=True, method="average").to_numpy(dtype=float)

    median_duration = float(np.median(duration))
    median_excursion = float(np.median(excursion))
    duration_scale = max(float(np.median(np.abs(duration - median_duration)) * 1.4826), float(np.std(duration)), 1e-6)
    excursion_scale = max(float(np.median(np.abs(excursion - median_excursion)) * 1.4826), float(np.std(excursion)), 1e-6)

    choices: list[tuple[str, str, pd.Series]] = []
    used: set[str] = set()
    definitions = [
        ("fast", "A real comparable episode near the lower joint duration/projected-adverse-excursion score percentile.", 0.25),
        ("normal", "The closest joint medoid of remaining duration and projected future adverse excursion among comparable episodes.", None),
        ("extreme", "A real comparable episode near the upper joint duration/projected-adverse-excursion score percentile; it is a stress reference, not a worst-case guarantee.", 0.90),
    ]
    for name, description, target_percentile in definitions:
        available = cohort.loc[~cohort["episode_id"].isin(used)].copy()
        if target_percentile is None:
            scenario_score = (
                np.abs(np.log1p(available["remaining_to_fill_bars"].to_numpy(dtype=float)) - median_duration) / duration_scale
                + np.abs(available["projected_future_max_away_move_pct"].to_numpy(dtype=float) - median_excursion)
                / excursion_scale
                + 0.20 * available["scenario_match_percentile"].to_numpy(dtype=float)
            )
        else:
            scenario_score = (
                np.abs(available["joint_risk_score_percentile"].to_numpy(dtype=float) - target_percentile)
                + 0.18 * available["scenario_match_percentile"].to_numpy(dtype=float)
            )
        selected = available.iloc[int(np.argmin(scenario_score))]
        used.add(str(selected["episode_id"]))
        choices.append((name, description, selected))
    scenarios: list[dict[str, Any]] = []
    for name, description, selected in choices:
        selected_joint_risk_score = float(selected["joint_risk_score"])
        selected_joint_risk_score_percentile = float(selected["joint_risk_score_percentile"])
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
                "historical_future_max_away_move_pct": float(selected["historical_future_max_away_move_pct"]),
                "projected_future_max_away_move_pct": float(selected["projected_future_max_away_move_pct"]),
                "projected_additional_adverse_move_pct": float(selected["projected_additional_adverse_move_pct"]),
                "joint_risk_score": selected_joint_risk_score,
                "joint_risk_score_percentile": selected_joint_risk_score_percentile,
                "matched_cohort_size": int(len(cohort)),
                "matched_cohort_tail_at_or_above_fraction": float(
                    np.mean(cohort["joint_risk_score"].to_numpy(dtype=float) >= selected_joint_risk_score)
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
    episode_row_spans: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, float | int]], float]:
    episode_id = str(scenario["episode_id"])
    span = episode_row_spans.get(episode_id) if episode_row_spans is not None else None
    if span is None:
        source = paths.loc[paths["episode_id"].eq(episode_id)].copy()
        source = source.sort_values("offset_bars", kind="stable")
    else:
        source = paths.iloc[span[0] : span[1]]
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


def projected_path_metrics(
    candles: list[dict[str, float | int]],
    current_target: float,
    current_direction_sign: int,
    current_move_pct: float,
) -> dict[str, float]:
    """Measure exactly the rounded candle envelope returned to the chart client."""
    if not candles:
        raise RuntimeError("A projected path must contain at least one future candle")
    if current_direction_sign == 1:
        directional_values = [(float(candle["high"]) / current_target - 1.0) * 100.0 for candle in candles]
    else:
        directional_values = [(1.0 - float(candle["low"]) / current_target) * 100.0 for candle in candles]
    future_peak = max(0.0, max(directional_values))
    return {
        "projected_future_max_away_move_pct": round(float(future_peak), 8),
        "projected_additional_adverse_move_pct": round(max(0.0, float(future_peak) - current_move_pct), 8),
    }


def project_at(
    episodes: pd.DataFrame,
    paths: pd.DataFrame,
    target_signal: pd.Series,
    current_state: dict[str, float],
    snapshot_close_time_ms: int,
    current_target: float,
    current_direction_sign: int,
    projection_start_open_time_ms: int,
    interval_minutes: int,
    top_k: int,
    path_states: pd.DataFrame | None = None,
    native_state_index: Any | None = None,
    episode_row_spans: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Select and render one shared, snapshot-safe V1 projection result.

    Both serving and chronological replay call this function.  It applies the
    terminal-fill-close availability rule, excludes pre-departure analogue
    states, ranks projected-coordinate risk before route selection, and then
    derives each displayed metric from the exact candles returned to the client.
    """
    eligible_events = eligible_episodes_at_snapshot(episodes, snapshot_close_time_ms)
    if eligible_events.empty:
        empty_states = pd.DataFrame()
        return {
            "eligible_events": eligible_events,
            "trajectory_events": eligible_events.copy(),
            "matched_states": empty_states,
            "cohort": empty_states.copy(),
            "scenarios": [],
            "insufficient_matches": True,
        }
    target_timeframe = str(target_signal["timeframe"])
    trajectory_events = eligible_events.loc[eligible_events["timeframe"].eq(target_timeframe)].copy()
    if trajectory_events.empty:
        empty_states = pd.DataFrame()
        return {
            "eligible_events": eligible_events,
            "trajectory_events": trajectory_events,
            "matched_states": empty_states,
            "cohort": empty_states.copy(),
            "scenarios": [],
            "insufficient_matches": True,
        }
    trajectory_ids = set(trajectory_events["episode_id"].astype(str))
    if path_states is None:
        trajectory_paths = paths.loc[paths["episode_id"].astype(str).isin(trajectory_ids)].copy()
        states = prepare_path_states(trajectory_events, trajectory_paths)
    else:
        all_library_episodes_are_eligible = len(eligible_events) == len(episodes) and len(trajectory_events) == len(episodes)
        states = (
            path_states
            if all_library_episodes_are_eligible
            else path_states.loc[path_states["episode_id"].astype(str).isin(trajectory_ids)].copy()
        )
    # The native index already contains exactly the sampled, post-departure
    # candidate grid and retains original full-state row IDs for the selected
    # suffixes. Rebuilding that 1m sample from every state on each live request
    # copies millions of rows without affecting an exact native selection.
    native_ready = native_state_index is not None and getattr(native_state_index, "library", None) is not None
    if not native_ready:
        states = sampled_matching_states(states, target_timeframe)
    matched_states = choose_best_alignment(
        trajectory_events,
        states,
        target_signal,
        current_state,
        str(target_signal["asset"]),
        target_timeframe,
        str(target_signal["direction"]),
        native_state_index=native_state_index,
        snapshot_close_time_ms=snapshot_close_time_ms,
        top_k=top_k,
    )
    matched_states = projected_coordinate_matches(matched_states, current_state["current_move_pct"])
    cohort = matched_states.head(min(top_k, len(matched_states))).copy()
    if len(cohort) < 12:
        return {
            "eligible_events": eligible_events,
            "trajectory_events": trajectory_events,
            "matched_states": matched_states,
            "cohort": cohort,
            "scenarios": [],
            "insufficient_matches": True,
        }
    scenarios = select_scenarios(matched_states, top_k)
    for scenario in scenarios:
        candles, scale = projected_candles(
            paths,
            scenario,
            current_target,
            current_direction_sign,
            current_state["current_move_pct"],
            projection_start_open_time_ms,
            interval_minutes,
            episode_row_spans,
        )
        scenario["normalization_scale"] = round(float(scale), 8)
        scenario["projected_candles"] = candles
        scenario.update(
            projected_path_metrics(candles, current_target, current_direction_sign, current_state["current_move_pct"])
        )
        scenario["projected_terminal_fill_candle_utc"] = utc_iso(
            candles[-1]["time"] * 1000 + interval_minutes * 60_000
        )
    return {
        "eligible_events": eligible_events,
        "trajectory_events": trajectory_events,
        "matched_states": matched_states,
        "cohort": cohort,
        "scenarios": scenarios,
        "insufficient_matches": False,
    }


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
    parser.add_argument("--asset", default="ETHUSDT", choices=SUPPORTED_ASSETS)
    parser.add_argument("--timeframe", default="5m", choices=("1m", "5m", "15m"))
    parser.add_argument("--source", type=Path, help="Raw source CSV; defaults to the selected five-year asset file")
    parser.add_argument("--library-dir", type=Path, help="Trajectory library; defaults to the matching timeframe library")
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
    source_interval = "1m" if args.timeframe == "1m" else "5m"
    source_path = args.source or (root / "data" / f"{args.asset}_{source_interval}_5y.csv")
    raw = (read_one_minute_file if args.timeframe == "1m" else read_five_minute_file)(source_path.resolve())
    frame = raw if args.timeframe in {"1m", "5m"} else resample_to_fifteen_minutes(raw)
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
    departed = has_confirmed_departure(later, direction_sign, float(signal["opposite_extreme"]))
    if not departed:
        raise RuntimeError(
            "Pinned wick has not yet made the confirmed close at or beyond the opposite signal extreme required for a conditional path projection"
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

    library_dir = (
        args.library_dir
        or (one_minute_library_dir(root, args.asset) if args.timeframe == "1m" else default_library_dir(root))
    ).resolve()
    episodes_path = library_dir / "episodes.csv"
    summary_path = library_dir / "summary.json"
    if not episodes_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Conditional path library has not been built")
    episodes = with_fill_close_time_ms(pd.read_csv(episodes_path))
    episodes["signal_open_time_ms"] = pd.to_numeric(episodes["signal_open_time_ms"], errors="raise").astype("int64")
    interval_minutes = int(signal["interval_minutes"])
    snapshot_close_time_ms = int(frame["open_time"].iat[as_of_index]) + interval_minutes * 60_000
    # Resolve the same snapshot-close candidate population that replay uses.
    eligible_for_paths = eligible_episodes_at_snapshot(episodes, snapshot_close_time_ms)
    if eligible_for_paths.empty:
        raise RuntimeError("No historical completed episodes had closed by the current snapshot")
    # A scenario is rendered as real candles on the pinned chart timeframe.  Mixing
    # a 15m path into a 5m candle chart would invent intrabar candles and distort
    # elapsed time, so cross-timeframe observations remain available to later
    # quantile models but are not direct V1 trajectory candidates.
    trajectory_for_paths = eligible_for_paths.loc[eligible_for_paths["timeframe"].eq(args.timeframe)].copy()
    if trajectory_for_paths.empty:
        raise RuntimeError(f"No historical completed {args.timeframe} episodes had closed by the current snapshot")
    print(
        json.dumps(
            {
                "stage": "eligible_episodes",
                "all_completed_before_snapshot": int(len(eligible_for_paths)),
                "same_timeframe_trajectory_candidates_before_snapshot": int(len(trajectory_for_paths)),
            }
        ),
        flush=True,
    )
    paths = read_library_paths(library_dir / "paths", trajectory_for_paths["path_file"].tolist())
    paths = paths.loc[paths["episode_id"].isin(set(trajectory_for_paths["episode_id"]))].copy()
    projection_start = snapshot_close_time_ms
    projection = project_at(
        episodes,
        paths,
        signal,
        current_state,
        snapshot_close_time_ms,
        target,
        direction_sign,
        projection_start,
        interval_minutes,
        args.top_k,
    )
    eligible = projection["eligible_events"]
    trajectory_eligible = projection["trajectory_events"]
    matched = projection["matched_states"]
    scenarios = projection["scenarios"]
    if projection["insufficient_matches"]:
        raise RuntimeError("Fewer than 12 comparable historical states; scenario selection would be too unstable")

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
                "historical_future_max_away_move_pct": float(row.historical_future_max_away_move_pct),
                "projected_future_max_away_move_pct": float(row.projected_future_max_away_move_pct),
                "projected_additional_adverse_move_pct": float(row.projected_additional_adverse_move_pct),
                "match_score": float(row.match_score),
            }
        )
    cohort = projection["cohort"]
    remaining_minutes = cohort["remaining_to_fill_bars"].to_numpy(dtype=float) * interval_minutes
    projected_future_move = cohort["projected_future_max_away_move_pct"].to_numpy(dtype=float)
    historical_future_move = cohort["historical_future_max_away_move_pct"].to_numpy(dtype=float)
    output = {
        "schema_version": "1.1.0",
        "method": "state-conditioned empirical historical trajectory scenarios",
        "conditionality": "Every projected path is a rescaled real historical episode that eventually fully fills its wick; this is not an unconditional fill probability or a trade recommendation.",
        "projection_semantics": {
            "candidate_availability": "An analogue is eligible only when its terminal fill candle closed at or before the pinned observation snapshot closed.",
            "departure": "The pinned signal and analogue alignment must have a close at or beyond the opposite signal extreme; analogue alignment offsets before that departure are excluded.",
            "future_excursion_window": "Future move-away is the direction-normalized high/low candle envelope after the snapshot through and including the terminal fill candle.",
            "intrabar_note": "OHLC cannot establish whether a terminal fill-candle extreme happened before or after the wick touch; this is a consistent candle-envelope measurement.",
        },
        "library": {
            "directory": str(library_dir),
            "availability_cutoff_close_utc": utc_iso(snapshot_close_time_ms),
            "eligible_completed_episodes_before_snapshot": int(len(eligible)),
            "same_timeframe_trajectory_candidates_before_snapshot": int(len(trajectory_eligible)),
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
            "as_of_close_time_utc": utc_iso(snapshot_close_time_ms),
            "as_of_close": float(frame["close"].iat[as_of_index]),
        },
        "cohort_distribution": {
            "remaining_time_to_fill_minutes": {
                "p25": float(np.quantile(remaining_minutes, 0.25)),
                "p50": float(np.quantile(remaining_minutes, 0.50)),
                "p90": float(np.quantile(remaining_minutes, 0.90)),
            },
            "projected_future_max_away_move_pct": {
                "p50": float(np.quantile(projected_future_move, 0.50)),
                "p90": float(np.quantile(projected_future_move, 0.90)),
            },
            "historical_future_max_away_move_pct": {
                "p50": float(np.quantile(historical_future_move, 0.50)),
                "p90": float(np.quantile(historical_future_move, 0.90)),
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
            "The terminal fill-candle risk uses a complete OHLC envelope; finer data is required to order an intrabar wick touch and extreme exactly.",
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
                        "projected_future_max_away_move_pct": scenario["projected_future_max_away_move_pct"],
                        "projected_additional_adverse_move_pct": scenario["projected_additional_adverse_move_pct"],
                    }
                    for scenario in scenarios
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
