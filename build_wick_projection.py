#!/usr/bin/env python3
"""Build a state-matched historical-analogue projection for one unfilled wick.

The model is deliberately non-parametric: it finds completed historical wick
paths whose signal anatomy and in-progress state resemble the current path.
Their post-match candles are direction-normalized, scaled to the current wick,
and summarized as a median candle path plus a 10-90% historical envelope.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


INTERVAL_MS = 5 * 60 * 1000
FEATURES = (
    "body_pct_of_range",
    "dominant_wick_pct_of_range",
    "opposite_wick_pct_of_range",
    "range_vs_prior_20_median",
    "volume_vs_prior_20_mean",
    "prior_1h_return_pct",
)


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int


def utc_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--signal-time must include a timezone")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def read_candles(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            candle = Candle(
                open_time=int(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                quote_volume=float(row["quote_asset_volume"]),
                trades=int(row["number_of_trades"]),
            )
            if candles and candle.open_time - candles[-1].open_time != INTERVAL_MS:
                raise RuntimeError(f"Input timestamp gap before {utc_iso(candle.open_time)}")
            candles.append(candle)
    return candles


def quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def robust_scale(values: Iterable[float]) -> float:
    ordered = list(values)
    median = statistics.median(ordered)
    mad = statistics.median(abs(value - median) for value in ordered)
    if mad > 1e-9:
        return mad
    deviation = statistics.pstdev(ordered)
    return deviation if deviation > 1e-9 else 1.0


def read_events(path: Path) -> dict[str, dict[str, object]]:
    events: dict[str, dict[str, object]] = {}
    with path.open("r", newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            event: dict[str, object] = dict(raw)
            for field in FEATURES + (
                "signal_to_fill_minutes",
                "signal_open",
                "signal_high",
                "signal_low",
                "signal_close",
            ):
                event[field] = float(raw[field]) if raw[field] else None
            event["direction_sign"] = int(raw["direction_sign"])
            events[raw["event_id"]] = event
    if not events:
        raise RuntimeError(f"No events in {path}")
    return events


def signal_shape_and_features(candles: list[Candle], index: int) -> tuple[int, float, dict[str, float]]:
    signal = candles[index]
    span = signal.high - signal.low
    if span <= 0:
        raise RuntimeError("The requested signal candle has no range")
    lower_wick = min(signal.open, signal.close) - signal.low
    upper_wick = signal.high - max(signal.open, signal.close)
    direction_sign = 1 if lower_wick >= upper_wick else -1
    wick_extreme = signal.low if direction_sign == 1 else signal.high
    prior20 = candles[index - 20:index]
    if len(prior20) != 20 or index < 12:
        raise RuntimeError("Not enough pre-signal data to calculate the current signal features")
    prior_ranges = [candle.high - candle.low for candle in prior20]
    prior_volumes = [candle.volume for candle in prior20]
    dominant = lower_wick if direction_sign == 1 else upper_wick
    opposite = upper_wick if direction_sign == 1 else lower_wick
    features = {
        "body_pct_of_range": abs(signal.close - signal.open) / span * 100,
        "dominant_wick_pct_of_range": dominant / span * 100,
        "opposite_wick_pct_of_range": opposite / span * 100,
        "range_vs_prior_20_median": span / statistics.median(prior_ranges),
        "volume_vs_prior_20_mean": signal.volume / statistics.mean(prior_volumes),
        "prior_1h_return_pct": (candles[index - 1].close / candles[index - 12].open - 1) * 100,
    }
    return direction_sign, wick_extreme, features


def normalized_price(price: float, wick_extreme: float, direction_sign: int) -> float:
    return direction_sign * (price / wick_extreme - 1) * 100


def log_distance(left: float, right: float, floor: float) -> float:
    return abs(math.log((max(left, 0.0) + floor) / (max(right, 0.0) + floor)))


def event_feature_distance(
    event: dict[str, object], target: dict[str, float], scales: dict[str, float]
) -> float:
    weights = {
        "body_pct_of_range": 0.8,
        "dominant_wick_pct_of_range": 0.8,
        "opposite_wick_pct_of_range": 0.8,
        "range_vs_prior_20_median": 0.6,
        "volume_vs_prior_20_mean": 0.6,
        "prior_1h_return_pct": 0.4,
    }
    distance = 0.0
    for field in FEATURES:
        value = event[field]
        if value is None:
            continue
        distance += weights[field] * abs(float(value) - target[field]) / scales[field]
    return distance


def first_pass_select_analogues(
    path: Path,
    events: dict[str, dict[str, object]],
    target_features: dict[str, float],
    state: dict[str, float],
    top_k: int,
) -> list[dict[str, object]]:
    scales = {
        field: robust_scale(float(event[field]) for event in events.values() if event[field] is not None)
        for field in FEATURES
    }
    candidates: list[dict[str, object]] = []
    current_id: str | None = None
    peak = float("-inf")
    best: dict[str, object] | None = None

    def finish_current() -> None:
        nonlocal best
        if best is not None:
            candidates.append(best)
        best = None

    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            event_id = row["event_id"]
            if event_id != current_id:
                finish_current()
                current_id = event_id
                peak = float("-inf")
            event = events[event_id]
            normalized_values = [
                float(row["normalized_open_pct"]),
                float(row["normalized_high_pct"]),
                float(row["normalized_low_pct"]),
                float(row["normalized_close_pct"]),
            ]
            peak = max(peak, *normalized_values)
            offset = int(row["offset_candles"])
            fill_offset = int(round(float(event["signal_to_fill_minutes"]) / 5))
            close = float(row["normalized_close_pct"])
            if offset >= fill_offset or close <= 0:
                continue
            drawdown = peak - close
            state_distance = (
                1.8 * log_distance(close, state["current_move_pct"], 0.3)
                + 1.2 * log_distance(peak, state["peak_move_pct"], 0.3)
                + 1.2 * log_distance(drawdown, state["drawdown_from_peak_pct"], 0.3)
                + 0.4 * log_distance(float(offset), state["elapsed_bars"], 3.0)
            )
            feature_distance = event_feature_distance(event, target_features, scales)
            score = state_distance + 0.5 * feature_distance
            if best is None or score < float(best["score"]):
                best = {
                    "event_id": event_id,
                    "score": score,
                    "alignment_offset_bars": offset,
                    "alignment_close_move_pct": close,
                    "alignment_peak_move_pct": peak,
                    "alignment_drawdown_pct": drawdown,
                    "remaining_to_fill_bars": fill_offset - offset,
                    "historical_signal_utc": event["signal_open_time_utc"],
                    "historical_direction": event["direction"],
                }
    finish_current()
    candidates.sort(key=lambda candidate: float(candidate["score"]))
    return candidates[:top_k]


def aggregate_actual(candles: list[Candle], start: int, end: int, bucket_size: int = 6) -> list[dict[str, object]]:
    aggregated: list[dict[str, object]] = []
    for bucket_start in range(start, end + 1, bucket_size):
        bucket = candles[bucket_start:min(end + 1, bucket_start + bucket_size)]
        aggregated.append(
            {
                "open_time_utc": utc_iso(bucket[0].open_time),
                "open": bucket[0].open,
                "high": max(candle.high for candle in bucket),
                "low": min(candle.low for candle in bucket),
                "close": bucket[-1].close,
            }
        )
    return aggregated


def second_pass_projection(
    path: Path,
    selected: list[dict[str, object]],
    wick_extreme: float,
    current_direction_sign: int,
    current_move_pct: float,
) -> tuple[list[dict[str, object]], int]:
    selected_by_id = {str(candidate["event_id"]): candidate for candidate in selected}
    remaining = [int(candidate["remaining_to_fill_bars"]) for candidate in selected]
    p75_remaining = quantile(remaining, 0.75) or 288
    horizon = min(2016, max(288, int(math.ceil(p75_remaining))))
    by_offset: dict[int, list[dict[str, float]]] = {}

    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            candidate = selected_by_id.get(row["event_id"])
            if candidate is None:
                continue
            original_offset = int(row["offset_candles"])
            relative_offset = original_offset - int(candidate["alignment_offset_bars"])
            if relative_offset < 0 or relative_offset > horizon:
                continue
            alignment_close = float(candidate["alignment_close_move_pct"])
            scale = current_move_pct / alignment_close
            normalized = {
                key: float(row[key]) * scale
                for key in (
                    "normalized_open_pct",
                    "normalized_high_pct",
                    "normalized_low_pct",
                    "normalized_close_pct",
                )
            }
            open_price = wick_extreme * (1 + current_direction_sign * normalized["normalized_open_pct"] / 100)
            close_price = wick_extreme * (1 + current_direction_sign * normalized["normalized_close_pct"] / 100)
            all_prices = [
                wick_extreme * (1 + current_direction_sign * normalized[key] / 100)
                for key in normalized
            ]
            by_offset.setdefault(relative_offset, []).append(
                {
                    "open": open_price,
                    "high": max(all_prices),
                    "low": min(all_prices),
                    "close": close_price,
                }
            )

    minimum_active = max(6, len(selected) // 4)
    projection: list[dict[str, object]] = []
    for offset in range(horizon + 1):
        samples = by_offset.get(offset, [])
        if len(samples) < minimum_active:
            break
        open_price = quantile((sample["open"] for sample in samples), 0.50)
        close_price = quantile((sample["close"] for sample in samples), 0.50)
        high_price = max(open_price or 0, close_price or 0, quantile((sample["high"] for sample in samples), 0.50) or 0)
        low_price = min(open_price or 0, close_price or 0, quantile((sample["low"] for sample in samples), 0.50) or 0)
        projection.append(
            {
                "offset_bars": offset,
                "minutes_from_now": offset * 5,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "envelope_low_p10": quantile((sample["low"] for sample in samples), 0.10),
                "envelope_high_p90": quantile((sample["high"] for sample in samples), 0.90),
                "active_analogues": len(samples),
            }
        )
    return projection, horizon


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_dir / "data" / "ETHUSDT_5m_3y.csv")
    parser.add_argument("--events", type=Path, default=project_dir / "data" / "filled_wick_study" / "filled_events.csv")
    parser.add_argument(
        "--paths", type=Path, default=project_dir / "data" / "filled_wick_study" / "filled_event_candles.csv.gz"
    )
    parser.add_argument("--signal-time", default="2026-09-16T18:35:00Z")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output", type=Path, default=project_dir / "data" / "current_wick_projection.json")
    args = parser.parse_args()
    if args.top_k < 10:
        parser.error("--top-k must be at least 10")

    candles = read_candles(args.input)
    signal_time = parse_utc(args.signal_time)
    timestamps = [candle.open_time for candle in candles]
    signal_index = bisect.bisect_left(timestamps, signal_time)
    if signal_index == len(candles) or candles[signal_index].open_time != signal_time:
        raise RuntimeError(f"Signal candle not present: {args.signal_time}")
    current_index = len(candles) - 1
    direction_sign, wick_extreme, target_features = signal_shape_and_features(candles, signal_index)
    signal = candles[signal_index]
    observed = candles[signal_index:current_index + 1]
    if direction_sign == 1 and min(candle.low for candle in observed[1:]) <= wick_extreme:
        raise RuntimeError("The lower wick has already been filled in the available current data")
    if direction_sign == -1 and max(candle.high for candle in observed[1:]) >= wick_extreme:
        raise RuntimeError("The upper wick has already been filled in the available current data")

    normalized_observed = [
        normalized_price(price, wick_extreme, direction_sign)
        for candle in observed
        for price in (candle.open, candle.high, candle.low, candle.close)
    ]
    current_move = normalized_price(candles[current_index].close, wick_extreme, direction_sign)
    peak_move = max(normalized_observed)
    current_state = {
        "elapsed_bars": float(current_index - signal_index),
        "elapsed_minutes": float((current_index - signal_index) * 5),
        "current_move_pct": current_move,
        "peak_move_pct": peak_move,
        "drawdown_from_peak_pct": peak_move - current_move,
    }

    events = read_events(args.events)
    selected = first_pass_select_analogues(args.paths, events, target_features, current_state, args.top_k)
    if len(selected) < 10:
        raise RuntimeError("Not enough completed historical analogues were found")
    projection, requested_horizon = second_pass_projection(
        args.paths, selected, wick_extreme, direction_sign, current_move
    )
    remaining_minutes = [int(candidate["remaining_to_fill_bars"]) * 5 for candidate in selected]
    result = {
        "method": "state_matched_historical_analogue_ensemble",
        "generated_at_utc": utc_iso(int(datetime.now(timezone.utc).timestamp() * 1000)),
        "source_data_ends_utc": utc_iso(candles[current_index].open_time + INTERVAL_MS - 1),
        "signal": {
            "open_time_utc": utc_iso(signal.open_time),
            "direction": "lower_wick" if direction_sign == 1 else "upper_wick",
            "wick_extreme_price": wick_extreme,
            "open": signal.open,
            "high": signal.high,
            "low": signal.low,
            "close": signal.close,
        },
        "current": {
            "open_time_utc": utc_iso(candles[current_index].open_time),
            "close": candles[current_index].close,
            **current_state,
        },
        "signal_features": target_features,
        "analogue_selection": {
            "count": len(selected),
            "state_match": "signal anatomy plus current move, peak, drawdown, and elapsed time",
            "remaining_to_fill_minutes": {
                "median": quantile(remaining_minutes, 0.50),
                "p25": quantile(remaining_minutes, 0.25),
                "p75": quantile(remaining_minutes, 0.75),
                "p90": quantile(remaining_minutes, 0.90),
                "max": max(remaining_minutes),
            },
            "matches": selected,
        },
        "actual_candles_30m": aggregate_actual(candles, signal_index, current_index),
        "projection_candles_5m": projection,
        "projection_requested_horizon_bars": requested_horizon,
        "projection_rendered_horizon_bars": projection[-1]["offset_bars"] if projection else 0,
        "limitations": [
            "This is a nearest-historical-analogue projection, not a guaranteed price forecast.",
            "Only historical paths already known to fill are used, so it estimates the conditional path if the current wick eventually fills.",
            "A walk-forward backtest with unresolved and early-fill signals is required before treating it as a trading model.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "selected_analogues": len(selected),
        "current_move_pct_from_wick": current_move,
        "projection_candles": len(projection),
        "remaining_to_fill_median_minutes": quantile(remaining_minutes, 0.50),
    }))


if __name__ == "__main__":
    main()
