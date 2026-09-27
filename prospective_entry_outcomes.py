#!/usr/bin/env python3
"""Build all-outcome labels for observable prospective wick entries.

This dataset is deliberately separate from the completed-route library.  It
retains every strict signal and creates an observation only when, at the close
of an explicit age bar, the signal has departed and its wick remains unfilled.
Future risk is measured from that observable entry close.

For a bar that touches both the target and an adverse threshold, OHLC cannot
establish which happened first.  Such cases are labelled ``ambiguous_intrabar``
and adverse excursion is stored as a strict pre-touch lower bound plus an
including-touch-bar upper bound.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from build_conditional_path_library import (
    FEATURE_COLUMNS,
    detect_strict_signals,
    read_five_minute_file,
    read_one_minute_file,
    trace_clean_path,
    utc_iso,
)
from conditional_wick_assets import SUPPORTED_ASSETS


DATASET_SCHEMA_VERSION = "prospective-entry-outcomes-v1.0.0"
SEQUENCE_CONTRACT_VERSION = "v3-observable-sequences-v1.0.0"
DEFAULT_ENTRY_AGES_MINUTES = (
    60,
    600,
    1_440,
    4_320,
    10_080,
    20_160,
    43_200,
    86_400,
    129_600,
)
DEFAULT_HORIZONS_MINUTES = (60, 240, 1_440, 4_320, 10_080, 20_160, 43_200)
DEFAULT_ADVERSE_THRESHOLDS_PCT = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
PRE_SIGNAL_CONTEXT_MINUTES = 8 * 60
RECENT_CONTEXT_MINUTES = 256 * 5
DEFAULT_RUST_BINARY = Path("experiments/rust_training_throughput/target/release") / (
    "wick-throughput-poc.exe" if os.name == "nt" else "wick-throughput-poc"
)


class RustKernelError(RuntimeError):
    """The optional Rust kernel could not prove parity with the Python frame."""


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return result


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive comma-separated numbers")
    return result


def threshold_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def horizon_slug(minutes: int) -> str:
    return f"{int(minutes)}m"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".parquet.tmp") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def first_true_offset(values: np.ndarray) -> int | None:
    hits = np.flatnonzero(values)
    return None if not len(hits) else int(hits[0]) + 1


def competing_outcome(
    target_bars: int | None,
    adverse_bars: int | None,
    horizon_bars: int,
    available_future_bars: int,
) -> str:
    """Resolve target/adverse ordering, preserving censoring and same-bar ambiguity."""
    target_within = target_bars is not None and target_bars <= horizon_bars
    adverse_within = adverse_bars is not None and adverse_bars <= horizon_bars
    if target_within and adverse_within:
        if target_bars == adverse_bars:
            return "ambiguous_intrabar"
        return "target_first" if target_bars < adverse_bars else "adverse_first"
    if target_within:
        return "target_first"
    if adverse_within:
        return "adverse_first"
    if available_future_bars >= horizon_bars:
        return "neither"
    return "right_censored"


def _target_hits(
    highs: np.ndarray,
    lows: np.ndarray,
    direction_sign: int,
    target: float,
) -> np.ndarray:
    return lows <= target if direction_sign == 1 else highs >= target


def _adverse_from_entry(
    highs: np.ndarray,
    lows: np.ndarray,
    direction_sign: int,
    entry_price: float,
) -> np.ndarray:
    if direction_sign == 1:
        values = (highs / entry_price - 1.0) * 100.0
    else:
        values = (1.0 - lows / entry_price) * 100.0
    return np.maximum(values, 0.0)


def _distance_from_target(close: float, target: float, direction_sign: int) -> float:
    return max(0.0, direction_sign * (close / target - 1.0) * 100.0)


def _signal_record(
    frame: pd.DataFrame,
    signal: Mapping[str, Any],
    status: str,
    departure_index: int | None,
    fill_index: int | None,
    maximum_future_bars: int,
) -> dict[str, Any]:
    signal_index = int(signal["bar_index"])
    interval_minutes = int(signal["interval_minutes"])
    available = len(frame) - signal_index - 1
    record: dict[str, Any] = {
        "signal_id": f"{signal['asset']}_{signal['timeframe']}_{signal['direction']}_{int(signal['open_time'])}",
        "asset": str(signal["asset"]),
        "timeframe": str(signal["timeframe"]),
        "direction": str(signal["direction"]),
        "direction_sign": int(signal["direction_sign"]),
        "signal_open_time_ms": int(signal["open_time"]),
        "signal_open_time_utc": str(signal["signal_open_time_utc"]),
        "signal_index": signal_index,
        "wick_target": float(signal["wick_target"]),
        "opposite_extreme": float(signal["opposite_extreme"]),
        "signal_open": float(signal["open"]),
        "signal_high": float(signal["high"]),
        "signal_low": float(signal["low"]),
        "signal_close": float(signal["close"]),
        "signal_volume": float(signal["volume"]),
        "resolution_status": status,
        "resolved_within_followup": status in {"filled", "filled_before_departure"},
        "target_unresolved_at_followup_end": status in {"unfilled", "right_censored"},
        "censoring_reason": (
            "source_ended_before_followup_horizon"
            if status == "right_censored"
            else (
                "target_not_touched_within_followup_horizon"
                if status == "unfilled"
                else None
            )
        ),
        "departure_index": departure_index,
        "fill_index": fill_index,
        "signal_to_departure_bars": (
            None if departure_index is None else int(departure_index - signal_index)
        ),
        "signal_to_fill_bars": (
            None if fill_index is None else int(fill_index - signal_index)
        ),
        "departure_open_time_utc": (
            None if departure_index is None else utc_iso(int(frame["open_time"].iat[departure_index]))
        ),
        "fill_open_time_utc": (
            None if fill_index is None else utc_iso(int(frame["open_time"].iat[fill_index]))
        ),
        "available_followup_bars": int(available),
        "requested_followup_bars": int(maximum_future_bars),
        "followup_fully_observed": bool(available >= maximum_future_bars),
        "interval_minutes": interval_minutes,
    }
    for field in FEATURE_COLUMNS:
        record[field] = float(signal[field])
    return record


def _observation_record(
    frame: pd.DataFrame,
    signal: Mapping[str, Any],
    departure_index: int,
    entry_index: int,
    age_minutes: int,
    horizons_minutes: Iterable[int],
    adverse_thresholds_pct: Iterable[float],
) -> dict[str, Any]:
    interval_minutes = int(signal["interval_minutes"])
    signal_index = int(signal["bar_index"])
    direction_sign = int(signal["direction_sign"])
    target = float(signal["wick_target"])
    entry_price = float(frame["close"].iat[entry_index])
    maximum_horizon_bars = max(int(minutes) // interval_minutes for minutes in horizons_minutes)
    future_end = min(len(frame) - 1, entry_index + maximum_horizon_bars)
    future = frame.iloc[entry_index + 1 : future_end + 1]
    future_highs = future["high"].to_numpy(dtype=float)
    future_lows = future["low"].to_numpy(dtype=float)
    available_future_bars = len(future)
    target_bars = first_true_offset(
        _target_hits(future_highs, future_lows, direction_sign, target)
    )
    adverse = _adverse_from_entry(
        future_highs, future_lows, direction_sign, entry_price
    )
    first_adverse = {
        float(threshold): first_true_offset(adverse >= float(threshold))
        for threshold in adverse_thresholds_pct
    }

    observed = frame.iloc[signal_index : entry_index + 1]
    if direction_sign == 1:
        away_envelope = (observed["high"].to_numpy(dtype=float) / target - 1.0) * 100.0
    else:
        away_envelope = (1.0 - observed["low"].to_numpy(dtype=float) / target) * 100.0
    peak_move = max(0.0, float(np.max(away_envelope)))
    current_move = _distance_from_target(entry_price, target, direction_sign)
    pre_signal_bars = max(1, PRE_SIGNAL_CONTEXT_MINUTES // interval_minutes)
    recent_bars = max(1, RECENT_CONTEXT_MINUTES // interval_minutes)
    signal_id = f"{signal['asset']}_{signal['timeframe']}_{signal['direction']}_{int(signal['open_time'])}"
    observation_id = f"{signal_id}_entry_{age_minutes}m"
    record: dict[str, Any] = {
        "observation_id": observation_id,
        "signal_id": signal_id,
        "asset": str(signal["asset"]),
        "timeframe": str(signal["timeframe"]),
        "direction": str(signal["direction"]),
        "direction_sign": direction_sign,
        "signal_open_time_ms": int(signal["open_time"]),
        "signal_index": signal_index,
        "departure_index": int(departure_index),
        "entry_index": int(entry_index),
        "entry_open_time_ms": int(frame["open_time"].iat[entry_index]),
        "entry_close_time_ms": int(frame["close_time"].iat[entry_index]),
        "entry_open_time_utc": utc_iso(int(frame["open_time"].iat[entry_index])),
        "entry_age_minutes": int(age_minutes),
        "entry_age_bars": int(entry_index - signal_index),
        "entry_price": entry_price,
        "entry_price_field": "close",
        "wick_target": target,
        "signal_volume": float(
            signal["signal_volume"] if "signal_volume" in signal else signal["volume"]
        ),
        "entry_distance_from_target_pct": current_move,
        "peak_distance_from_target_pct": peak_move,
        "drawdown_from_peak_pct": max(0.0, peak_move - current_move),
        "departure_to_entry_bars": int(entry_index - departure_index),
        "available_future_bars": int(available_future_bars),
        "target_touch_bars_from_entry": target_bars,
        "sequence_contract_version": SEQUENCE_CONTRACT_VERSION,
        "pre_signal_start_index": max(0, signal_index - pre_signal_bars),
        "pre_signal_end_index": signal_index - 1,
        "recent_start_index": max(0, entry_index - recent_bars + 1),
        "recent_end_index": entry_index,
        "signal_to_entry_start_index": signal_index,
        "signal_to_entry_end_index": entry_index,
    }
    for field in FEATURE_COLUMNS:
        record[field] = float(signal[field])
    for threshold, bars in first_adverse.items():
        record[f"first_adverse_{threshold_slug(threshold)}pct_bars_from_entry"] = bars

    for horizon_minutes in horizons_minutes:
        horizon_bars = int(horizon_minutes) // interval_minutes
        slug = horizon_slug(int(horizon_minutes))
        visible_bars = min(horizon_bars, available_future_bars)
        target_within = target_bars is not None and target_bars <= horizon_bars
        strict_cutoff = min(
            visible_bars,
            (target_bars - 1) if target_within else visible_bars,
        )
        envelope_cutoff = min(
            visible_bars,
            target_bars if target_within else visible_bars,
        )
        strict_values = adverse[: max(strict_cutoff, 0)]
        envelope_values = adverse[: max(envelope_cutoff, 0)]
        record[f"horizon_{slug}_fully_observed"] = bool(
            available_future_bars >= horizon_bars
        )
        record[f"target_hit_{slug}"] = bool(target_within)
        record[f"max_adverse_pre_target_lower_{slug}_pct"] = (
            0.0 if not len(strict_values) else float(np.max(strict_values))
        )
        record[f"max_adverse_pre_target_upper_{slug}_pct"] = (
            0.0 if not len(envelope_values) else float(np.max(envelope_values))
        )
        for threshold, adverse_bars in first_adverse.items():
            record[
                f"outcome_{slug}_vs_{threshold_slug(threshold)}pct"
            ] = competing_outcome(
                target_bars,
                adverse_bars,
                horizon_bars,
                available_future_bars,
            )
    return record


def _observation_record_from_kernel(
    frame: pd.DataFrame,
    signal: Mapping[str, Any],
    kernel: Mapping[str, Any],
    horizons_minutes: Iterable[int],
    adverse_thresholds_pct: Iterable[float],
) -> dict[str, Any]:
    """Assemble the authoritative Python schema from a validated Rust scan row."""
    interval_minutes = int(signal["interval_minutes"])
    signal_index = int(signal["bar_index"])
    entry_index = int(kernel["entry_index"])
    departure_index = int(kernel["departure_index"])
    age_minutes = int(kernel["entry_age_minutes"])
    direction_sign = int(signal["direction_sign"])
    target = float(signal["wick_target"])
    entry_price = float(frame["close"].iat[entry_index])
    pre_signal_bars = max(1, PRE_SIGNAL_CONTEXT_MINUTES // interval_minutes)
    recent_bars = max(1, RECENT_CONTEXT_MINUTES // interval_minutes)
    signal_id = (
        f"{signal['asset']}_{signal['timeframe']}_{signal['direction']}_"
        f"{int(signal['open_time'])}"
    )
    record: dict[str, Any] = {
        "observation_id": f"{signal_id}_entry_{age_minutes}m",
        "signal_id": signal_id,
        "asset": str(signal["asset"]),
        "timeframe": str(signal["timeframe"]),
        "direction": str(signal["direction"]),
        "direction_sign": direction_sign,
        "signal_open_time_ms": int(signal["open_time"]),
        "signal_index": signal_index,
        "departure_index": departure_index,
        "entry_index": entry_index,
        "entry_open_time_ms": int(frame["open_time"].iat[entry_index]),
        "entry_close_time_ms": int(frame["close_time"].iat[entry_index]),
        "entry_open_time_utc": utc_iso(int(frame["open_time"].iat[entry_index])),
        "entry_age_minutes": age_minutes,
        "entry_age_bars": int(entry_index - signal_index),
        "entry_price": entry_price,
        "entry_price_field": "close",
        "wick_target": target,
        "signal_volume": float(
            signal["signal_volume"] if "signal_volume" in signal else signal["volume"]
        ),
        "entry_distance_from_target_pct": float(
            kernel["entry_distance_from_target_pct"]
        ),
        "peak_distance_from_target_pct": float(
            kernel["peak_distance_from_target_pct"]
        ),
        "drawdown_from_peak_pct": float(kernel["drawdown_from_peak_pct"]),
        "departure_to_entry_bars": int(entry_index - departure_index),
        "available_future_bars": int(kernel["available_future_bars"]),
        "target_touch_bars_from_entry": kernel["target_touch_bars_from_entry"],
        "sequence_contract_version": SEQUENCE_CONTRACT_VERSION,
        "pre_signal_start_index": max(0, signal_index - pre_signal_bars),
        "pre_signal_end_index": signal_index - 1,
        "recent_start_index": max(0, entry_index - recent_bars + 1),
        "recent_end_index": entry_index,
        "signal_to_entry_start_index": signal_index,
        "signal_to_entry_end_index": entry_index,
    }
    for field in FEATURE_COLUMNS:
        record[field] = float(signal[field])

    thresholds = tuple(float(value) for value in adverse_thresholds_pct)
    first_adverse = tuple(kernel["first_adverse_bars_from_entry"])
    if len(first_adverse) != len(thresholds):
        raise RustKernelError("Rust first-adverse width does not match thresholds")
    for threshold, bars in zip(thresholds, first_adverse, strict=True):
        record[f"first_adverse_{threshold_slug(threshold)}pct_bars_from_entry"] = bars

    horizons = tuple(int(value) for value in horizons_minutes)
    kernel_horizons = tuple(kernel["horizons"])
    if len(kernel_horizons) != len(horizons):
        raise RustKernelError("Rust horizon width does not match requested horizons")
    for horizon_minutes, values in zip(horizons, kernel_horizons, strict=True):
        if int(values["horizon_minutes"]) != horizon_minutes:
            raise RustKernelError("Rust horizon order does not match requested horizons")
        slug = horizon_slug(horizon_minutes)
        outcomes = tuple(values["outcomes"])
        if len(outcomes) != len(thresholds):
            raise RustKernelError("Rust outcome width does not match thresholds")
        record[f"horizon_{slug}_fully_observed"] = bool(values["fully_observed"])
        record[f"target_hit_{slug}"] = bool(values["target_hit"])
        record[f"max_adverse_pre_target_lower_{slug}_pct"] = float(
            values["adverse_lower_pct"]
        )
        record[f"max_adverse_pre_target_upper_{slug}_pct"] = float(
            values["adverse_upper_pct"]
        )
        for threshold, outcome in zip(thresholds, outcomes, strict=True):
            record[f"outcome_{slug}_vs_{threshold_slug(threshold)}pct"] = str(outcome)
    return record


def materialize_v3_observable_inputs(
    frame: pd.DataFrame,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the existing V3 numerical sequences without future candles.

    The import is intentionally lazy so dataset construction does not require
    PyTorch.  The returned candle geometry remains explicit beside the learned
    sequence inputs.
    """
    import train_neural_path_v3 as v3

    timeframe = str(observation["timeframe"])
    previous_timeframe = f"{v3.BAR_MINUTES}m"
    v3.configure_timeframe(timeframe)
    try:
        return _materialize_v3_observable_inputs_configured(
            frame, observation, v3
        )
    finally:
        v3.configure_timeframe(previous_timeframe)


def _materialize_v3_observable_inputs_configured(
    frame: pd.DataFrame,
    observation: Mapping[str, Any],
    v3: Any,
) -> dict[str, Any]:
    timeframe = str(observation["timeframe"])
    raw = {
        field: frame[field].to_numpy(
            dtype=np.int64 if field == "open_time" else np.float64
        )
        for field in ("open_time", "open", "high", "low", "close", "volume")
    }
    signal_ms = int(observation["signal_open_time_ms"])
    entry_ms = int(observation["entry_open_time_ms"])
    target = float(observation["wick_target"])
    direction_sign = int(observation["direction_sign"])
    signal_volume = float(observation["signal_volume"])
    interval_ms = int(timeframe.removesuffix("m")) * 60_000
    pre_signal = v3.raw_window(
        raw,
        signal_ms - interval_ms,
        v3.PRE_SIGNAL_LENGTH,
        target,
        direction_sign,
        signal_volume,
        signal_ms,
        source_length=v3.PRE_SIGNAL_SOURCE_BARS,
    )
    recent = v3.raw_window(
        raw,
        entry_ms,
        v3.RECENT_LENGTH,
        target,
        direction_sign,
        signal_volume,
        signal_ms,
        source_length=v3.RECENT_SOURCE_BARS,
    )
    signal_index = int(observation["signal_index"])
    entry_index = int(observation["entry_index"])
    source = frame.iloc[signal_index : entry_index + 1]
    episode_path = pd.DataFrame(
        {
            "offset_bars": np.arange(len(source), dtype=np.int64),
            "volume": source["volume"].to_numpy(dtype=float),
        }
    )
    for field in ("open", "high", "low", "close"):
        episode_path[f"normalized_{field}_pct"] = v3.direction_normalized_pct(
            source[field].to_numpy(dtype=float), target, direction_sign
        )
    episode = v3.episode_window(
        episode_path,
        int(observation["entry_age_bars"]),
        signal_volume,
    )
    return {
        "pre_signal": pre_signal,
        "recent": recent,
        "signal_to_entry": episode,
        "candle_geometry": {
            field: float(observation[field]) for field in FEATURE_COLUMNS
        },
        "latest_observable_open_time_ms": entry_ms,
    }


def _build_series_dataset_python(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    entry_ages_minutes: Iterable[int] = DEFAULT_ENTRY_AGES_MINUTES,
    horizons_minutes: Iterable[int] = DEFAULT_HORIZONS_MINUTES,
    adverse_thresholds_pct: Iterable[float] = DEFAULT_ADVERSE_THRESHOLDS_PCT,
    maximum_followup_days: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return all strict signals plus all observable active-entry snapshots."""
    interval_minutes = int(timeframe.removesuffix("m"))
    ages = tuple(int(value) for value in entry_ages_minutes)
    horizons = tuple(int(value) for value in horizons_minutes)
    thresholds = tuple(float(value) for value in adverse_thresholds_pct)
    if any(value % interval_minutes for value in (*ages, *horizons)):
        raise ValueError("entry ages and horizons must be exact multiples of the timeframe")
    if maximum_followup_days * 24 * 60 < max(ages) + max(horizons):
        raise ValueError("maximum follow-up must cover the oldest entry age plus longest horizon")
    signals = detect_strict_signals(frame, asset, timeframe)
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    maximum_future_bars = maximum_followup_days * 24 * 60 // interval_minutes
    signal_records: list[dict[str, Any]] = []
    observation_records: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    omitted_entries: Counter[str] = Counter()

    for row in signals.itertuples(index=False):
        signal = row._asdict()
        status, departure_index, fill_index = trace_clean_path(
            closes,
            highs,
            lows,
            int(signal["bar_index"]),
            int(signal["direction_sign"]),
            float(signal["wick_target"]),
            float(signal["opposite_extreme"]),
            maximum_future_bars,
        )
        statuses[status] += 1
        signal_records.append(
            _signal_record(
                frame,
                signal,
                status,
                departure_index,
                fill_index,
                maximum_future_bars,
            )
        )
        for age_minutes in ages:
            entry_index = int(signal["bar_index"]) + age_minutes // interval_minutes
            if entry_index >= len(frame):
                omitted_entries["entry_not_observed"] += 1
                continue
            if departure_index is None or departure_index > entry_index:
                omitted_entries["not_departed_by_entry"] += 1
                continue
            if fill_index is not None and fill_index <= entry_index:
                omitted_entries["filled_by_entry"] += 1
                continue
            observation_records.append(
                _observation_record(
                    frame,
                    signal,
                    int(departure_index),
                    entry_index,
                    age_minutes,
                    horizons,
                    thresholds,
                )
            )

    signal_frame = pd.DataFrame.from_records(signal_records)
    observation_frame = pd.DataFrame.from_records(observation_records)
    summary = {
        "asset": asset,
        "timeframe": timeframe,
        "signal_count": int(len(signal_frame)),
        "observation_count": int(len(observation_frame)),
        "resolution_status_counts": dict(sorted(statuses.items())),
        "omitted_entry_counts": dict(sorted(omitted_entries.items())),
        "source_start_utc": utc_iso(int(frame["open_time"].iat[0])),
        "source_end_utc": utc_iso(int(frame["open_time"].iat[-1])),
    }
    return signal_frame, observation_frame, summary


def _csv_values(values: Iterable[int | float]) -> str:
    return ",".join(f"{value:g}" if isinstance(value, float) else str(value) for value in values)


def _run_rust_kernel(
    source_path: Path,
    source_rows: int,
    timeframe: str,
    entry_ages_minutes: tuple[int, ...],
    horizons_minutes: tuple[int, ...],
    adverse_thresholds_pct: tuple[float, ...],
    maximum_followup_days: int,
    rust_binary: Path,
) -> dict[str, Any]:
    binary = Path(rust_binary)
    if not binary.is_file():
        raise RustKernelError(f"Rust kernel binary not found: {binary}")
    source = Path(source_path)
    if not source.is_file():
        raise RustKernelError(f"Rust kernel source file not found: {source}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
        export_path = Path(handle.name)
    command = [
        str(binary),
        "--mode",
        "export",
        "--input",
        str(source),
        "--output",
        str(export_path),
        "--timeframe",
        timeframe,
        "--maximum-fill-days",
        str(maximum_followup_days),
        "--maximum-rows",
        str(source_rows),
        "--entry-ages-minutes",
        _csv_values(entry_ages_minutes),
        "--horizons-minutes",
        _csv_values(horizons_minutes),
        "--adverse-thresholds-pct",
        _csv_values(adverse_thresholds_pct),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RustKernelError("Rust kernel returned invalid summary JSON") from error
        try:
            with export_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RustKernelError("Rust kernel export could not be read") from error
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RustKernelError(f"Rust kernel execution failed: {detail.strip()}") from error
    finally:
        export_path.unlink(missing_ok=True)
    if not isinstance(payload, dict):
        raise RustKernelError("Rust kernel export is not an object")
    return payload


def _validate_rust_kernel(
    payload: Mapping[str, Any],
    signals: pd.DataFrame,
    source_rows: int,
    timeframe: str,
    entry_ages_minutes: tuple[int, ...],
    horizons_minutes: tuple[int, ...],
    adverse_thresholds_pct: tuple[float, ...],
) -> None:
    interval_minutes = int(timeframe.removesuffix("m"))
    checks = {
        "schema version": payload.get("schema_version") == "rust-prospective-kernel-v1",
        "timeframe": payload.get("timeframe") == timeframe,
        "interval": int(payload.get("interval_minutes", -1)) == interval_minutes,
        "source rows": int(payload.get("source_rows", -1)) == source_rows,
        "entry ages": tuple(payload.get("entry_ages_minutes", ())) == entry_ages_minutes,
        "horizons": tuple(payload.get("horizons_minutes", ())) == horizons_minutes,
        "thresholds": tuple(float(value) for value in payload.get("adverse_thresholds_pct", ()))
        == adverse_thresholds_pct,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RustKernelError(f"Rust kernel contract mismatch: {', '.join(failed)}")

    expected_indexes = [int(value) for value in signals["bar_index"].tolist()]
    traces = payload.get("traces")
    if not isinstance(traces, list):
        raise RustKernelError("Rust kernel traces are missing")
    actual_indexes = [int(row["signal_index"]) for row in traces]
    if actual_indexes != expected_indexes:
        raise RustKernelError("Rust/Python strict-signal order mismatch")
    if int(payload.get("strict_signals", -1)) != len(expected_indexes):
        raise RustKernelError("Rust/Python strict-signal count mismatch")
    if int(payload.get("signal_index_checksum", -1)) != sum(expected_indexes):
        raise RustKernelError("Rust/Python strict-signal checksum mismatch")
    observations = payload.get("observations")
    aggregate = payload.get("aggregate")
    if not isinstance(observations, list) or not isinstance(aggregate, dict):
        raise RustKernelError("Rust observation export is incomplete")
    if int(aggregate.get("observation_count", -1)) != len(observations):
        raise RustKernelError("Rust observation count mismatch")


def _build_series_dataset_rust(
    frame: pd.DataFrame,
    source_path: Path,
    asset: str,
    timeframe: str,
    entry_ages_minutes: Iterable[int],
    horizons_minutes: Iterable[int],
    adverse_thresholds_pct: Iterable[float],
    maximum_followup_days: int,
    rust_binary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    interval_minutes = int(timeframe.removesuffix("m"))
    ages = tuple(int(value) for value in entry_ages_minutes)
    horizons = tuple(int(value) for value in horizons_minutes)
    thresholds = tuple(float(value) for value in adverse_thresholds_pct)
    if any(value % interval_minutes for value in (*ages, *horizons)):
        raise ValueError("entry ages and horizons must be exact multiples of the timeframe")
    if maximum_followup_days * 24 * 60 < max(ages) + max(horizons):
        raise ValueError("maximum follow-up must cover the oldest entry age plus longest horizon")

    signals = detect_strict_signals(frame, asset, timeframe)
    payload = _run_rust_kernel(
        source_path,
        len(frame),
        timeframe,
        ages,
        horizons,
        thresholds,
        maximum_followup_days,
        rust_binary,
    )
    _validate_rust_kernel(payload, signals, len(frame), timeframe, ages, horizons, thresholds)

    maximum_future_bars = maximum_followup_days * 24 * 60 // interval_minutes
    signal_maps = [row._asdict() for row in signals.itertuples(index=False)]
    signal_by_index = {int(signal["bar_index"]): signal for signal in signal_maps}
    trace_by_index = {
        int(trace["signal_index"]): trace for trace in payload["traces"]
    }
    statuses: Counter[str] = Counter()
    signal_records: list[dict[str, Any]] = []
    for signal in signal_maps:
        signal_index = int(signal["bar_index"])
        trace = trace_by_index[signal_index]
        status = str(trace["status"])
        statuses[status] += 1
        signal_records.append(
            _signal_record(
                frame,
                signal,
                status,
                trace["departure_index"],
                trace["fill_index"],
                maximum_future_bars,
            )
        )

    observation_records: list[dict[str, Any]] = []
    for kernel in payload["observations"]:
        signal_index = int(kernel["signal_index"])
        signal = signal_by_index.get(signal_index)
        if signal is None:
            raise RustKernelError("Rust observation references an unknown signal")
        age_minutes = int(kernel["entry_age_minutes"])
        if age_minutes not in ages:
            raise RustKernelError("Rust observation references an unknown entry age")
        if int(kernel["entry_index"]) != signal_index + age_minutes // interval_minutes:
            raise RustKernelError("Rust observation entry index is inconsistent")
        trace = trace_by_index[signal_index]
        if kernel["departure_index"] != trace["departure_index"]:
            raise RustKernelError("Rust observation departure index is inconsistent")
        observation_records.append(
            _observation_record_from_kernel(frame, signal, kernel, horizons, thresholds)
        )

    aggregate = payload["aggregate"]
    signal_frame = pd.DataFrame.from_records(signal_records)
    observation_frame = pd.DataFrame.from_records(observation_records)
    summary = {
        "asset": asset,
        "timeframe": timeframe,
        "signal_count": int(len(signal_frame)),
        "observation_count": int(len(observation_frame)),
        "resolution_status_counts": dict(sorted(statuses.items())),
        "omitted_entry_counts": dict(sorted(aggregate["omitted_entry_counts"].items())),
        "source_start_utc": utc_iso(int(frame["open_time"].iat[0])),
        "source_end_utc": utc_iso(int(frame["open_time"].iat[-1])),
        "label_engine": "rust",
    }
    return signal_frame, observation_frame, summary


def build_series_dataset(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    entry_ages_minutes: Iterable[int] = DEFAULT_ENTRY_AGES_MINUTES,
    horizons_minutes: Iterable[int] = DEFAULT_HORIZONS_MINUTES,
    adverse_thresholds_pct: Iterable[float] = DEFAULT_ADVERSE_THRESHOLDS_PCT,
    maximum_followup_days: int = 180,
    *,
    engine: str = "python",
    source_path: Path | None = None,
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one series with Python or the parity-checked optional Rust kernel."""
    if engine not in {"python", "rust", "auto"}:
        raise ValueError("engine must be python, rust, or auto")
    if engine != "python" and source_path is not None:
        try:
            return _build_series_dataset_rust(
                frame,
                source_path,
                asset,
                timeframe,
                entry_ages_minutes,
                horizons_minutes,
                adverse_thresholds_pct,
                maximum_followup_days,
                rust_binary,
            )
        except RustKernelError as error:
            if engine == "rust":
                raise
            print(f"Rust label kernel unavailable; using Python: {error}", file=sys.stderr)
    elif engine == "rust":
        raise RustKernelError("explicit Rust engine requires source_path")
    signals, observations, summary = _build_series_dataset_python(
        frame,
        asset,
        timeframe,
        entry_ages_minutes,
        horizons_minutes,
        adverse_thresholds_pct,
        maximum_followup_days,
    )
    summary["label_engine"] = "python"
    return signals, observations, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prospective_entry_outcomes_v1"),
    )
    parser.add_argument("--assets", default=",".join(SUPPORTED_ASSETS))
    parser.add_argument("--timeframes", default="1m,5m")
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
    parser.add_argument("--maximum-followup-days", type=int, default=180)
    parser.add_argument(
        "--engine",
        choices=("auto", "python", "rust"),
        default="auto",
        help="label scan engine; auto validates Rust parity and falls back to Python",
    )
    parser.add_argument("--rust-binary", type=Path, default=DEFAULT_RUST_BINARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = tuple(item.strip().upper() for item in args.assets.split(",") if item.strip())
    timeframes = tuple(item.strip() for item in args.timeframes.split(",") if item.strip())
    unknown_assets = sorted(set(assets).difference(SUPPORTED_ASSETS))
    if unknown_assets:
        raise ValueError(f"unsupported assets: {unknown_assets}")
    if not timeframes or set(timeframes).difference({"1m", "5m"}):
        raise ValueError("timeframes must contain only 1m and/or 5m")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for timeframe in timeframes:
        reader = read_one_minute_file if timeframe == "1m" else read_five_minute_file
        for asset in assets:
            source = args.data_dir / f"{asset}_{timeframe}_5y.csv"
            frame = reader(source)
            signals, observations, summary = build_series_dataset(
                frame,
                asset,
                timeframe,
                args.entry_ages_minutes,
                args.horizons_minutes,
                args.adverse_thresholds_pct,
                args.maximum_followup_days,
                engine=args.engine,
                source_path=source,
                rust_binary=args.rust_binary,
            )
            stem = f"{asset}_{timeframe}"
            signal_path = args.output_dir / "signals" / f"{stem}.parquet"
            observation_path = args.output_dir / "observations" / f"{stem}.parquet"
            atomic_write_parquet(signal_path, signals)
            atomic_write_parquet(observation_path, observations)
            summaries.append(summary)
            files.extend(
                [
                    {
                        "kind": "signals",
                        "asset": asset,
                        "timeframe": timeframe,
                        "path": signal_path.relative_to(args.output_dir).as_posix(),
                    },
                    {
                        "kind": "observations",
                        "asset": asset,
                        "timeframe": timeframe,
                        "path": observation_path.relative_to(args.output_dir).as_posix(),
                    },
                ]
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "purpose": "prospective-entry waiting-time and entry-relative adverse-risk evaluation",
        "entry_price": "close of the observable entry-age candle",
        "population": "all strict signals; observations require departure and an unfilled target at entry",
        "entry_ages_minutes": list(args.entry_ages_minutes),
        "horizons_minutes": list(args.horizons_minutes),
        "adverse_thresholds_pct": list(args.adverse_thresholds_pct),
        "maximum_followup_days": int(args.maximum_followup_days),
        "requested_label_engine": args.engine,
        "rust_binary": str(args.rust_binary),
        "status_semantics": "unfilled means not touched within the explicit follow-up horizon, not never; right_censored means the source ended before that horizon",
        "intrabar_policy": "same-bar target/adverse events are ambiguous; adverse lower bound excludes and upper bound includes the target-touch bar",
        "sequence_contract": {
            "version": SEQUENCE_CONTRACT_VERSION,
            "pre_signal_context_minutes": PRE_SIGNAL_CONTEXT_MINUTES,
            "recent_context_minutes": RECENT_CONTEXT_MINUTES,
            "materialization": "raw candle indexes are retained so the existing V3 direction-normalized OHLCV sequence encoder can materialize only candles observable at entry",
        },
        "series": summaries,
        "files": files,
    }
    atomic_write_json(args.output_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
