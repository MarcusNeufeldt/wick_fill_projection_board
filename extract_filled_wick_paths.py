#!/usr/bin/env python3
"""Extract completed lower- or upper-wick signal-to-fill paths from ETHUSDT candles.

This is an event-study extractor, not a predictor.  It uses future candles only
to label historical examples as completed paths.  All signal features are
computed strictly from candles available at the signal close.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


INTERVAL_MS = 5 * 60 * 1000


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
    taker_buy_base: float
    taker_buy_quote: float


@dataclass(frozen=True)
class Shape:
    direction: str
    span: float
    body_pct: float
    dominant_wick_pct: float
    opposite_wick_pct: float


@dataclass(frozen=True)
class Trace:
    status: str
    departure_index: int | None
    fill_index: int | None


def utc_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone, for example 2026-09-16T18:35:00Z")
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
                taker_buy_base=float(row["taker_buy_base_asset_volume"]),
                taker_buy_quote=float(row["taker_buy_quote_asset_volume"]),
            )
            if candles and candle.open_time - candles[-1].open_time != INTERVAL_MS:
                raise RuntimeError(f"Input has a timestamp gap before {utc_iso(candle.open_time)}")
            candles.append(candle)
    if not candles:
        raise RuntimeError(f"Input has no candles: {path}")
    return candles


def classify_shape(
    candle: Candle,
    body_max_pct: float,
    dominant_wick_min_pct: float,
    opposite_wick_max_pct: float,
) -> Shape | None:
    span = candle.high - candle.low
    if span <= 0:
        return None
    body_pct = abs(candle.close - candle.open) / span
    lower_wick_pct = (min(candle.open, candle.close) - candle.low) / span
    upper_wick_pct = (candle.high - max(candle.open, candle.close)) / span
    if body_pct > body_max_pct:
        return None
    if lower_wick_pct >= dominant_wick_min_pct and upper_wick_pct <= opposite_wick_max_pct:
        return Shape("lower_wick", span, body_pct, lower_wick_pct, upper_wick_pct)
    if upper_wick_pct >= dominant_wick_min_pct and lower_wick_pct <= opposite_wick_max_pct:
        return Shape("upper_wick", span, body_pct, upper_wick_pct, lower_wick_pct)
    return None


def trace_completed_path(
    candles: list[Candle],
    signal_index: int,
    direction: str,
    usable_end_index: int,
    departure_bars: int | None,
    fill_bars: int | None,
) -> Trace:
    """Find clean signal -> departure -> full-fill paths without OHLC ordering leaks.

    A candle that both touches the wick extreme and closes beyond the opposite
    extreme is conservatively treated as an early fill: five-minute OHLC data
    cannot prove which happened first inside that one candle.
    """
    signal = candles[signal_index]
    departure_limit = (
        usable_end_index
        if departure_bars is None
        else min(usable_end_index, signal_index + departure_bars)
    )
    departure_index: int | None = None

    for index in range(signal_index + 1, departure_limit + 1):
        candle = candles[index]
        if direction == "lower_wick":
            if candle.low <= signal.low:
                return Trace("filled_before_departure", None, None)
            if candle.close >= signal.high:
                departure_index = index
                break
        else:
            if candle.high >= signal.high:
                return Trace("filled_before_departure", None, None)
            if candle.close <= signal.low:
                departure_index = index
                break

    if departure_index is None:
        status = (
            "right_censored"
            if departure_bars is None or departure_limit < signal_index + departure_bars
            else "no_departure"
        )
        return Trace(status, None, None)

    fill_limit = (
        usable_end_index
        if fill_bars is None
        else min(usable_end_index, departure_index + fill_bars)
    )
    for index in range(departure_index + 1, fill_limit + 1):
        candle = candles[index]
        is_filled = (
            candle.low <= signal.low if direction == "lower_wick" else candle.high >= signal.high
        )
        if is_filled:
            return Trace("filled", departure_index, index)

    status = (
        "right_censored"
        if fill_bars is None or fill_limit < departure_index + fill_bars
        else "unfilled"
    )
    return Trace(status, departure_index, None)


def percent_change(value: float, base: float) -> float | None:
    return (value / base - 1) * 100 if base else None


def signal_features(candles: list[Candle], index: int, shape: Shape) -> dict[str, float | None]:
    signal = candles[index]
    result: dict[str, float | None] = {
        "range_usdt": shape.span,
        "range_pct_of_open": shape.span / signal.open * 100 if signal.open else None,
        "body_pct_of_range": shape.body_pct * 100,
        "dominant_wick_pct_of_range": shape.dominant_wick_pct * 100,
        "opposite_wick_pct_of_range": shape.opposite_wick_pct * 100,
        "taker_buy_share_pct": signal.taker_buy_base / signal.volume * 100 if signal.volume else None,
        "prior_1h_return_pct": None,
        "range_vs_prior_20_median": None,
        "volume_vs_prior_20_mean": None,
    }
    if index >= 12:
        result["prior_1h_return_pct"] = percent_change(candles[index - 1].close, candles[index - 12].open)
    if index >= 20:
        prior_ranges = [candle.high - candle.low for candle in candles[index - 20:index]]
        prior_volumes = [candle.volume for candle in candles[index - 20:index]]
        median_range = statistics.median(prior_ranges)
        mean_volume = statistics.mean(prior_volumes)
        result["range_vs_prior_20_median"] = shape.span / median_range if median_range else None
        result["volume_vs_prior_20_mean"] = signal.volume / mean_volume if mean_volume else None
    return result


def event_row(
    candles: list[Candle], signal_index: int, shape: Shape, trace: Trace
) -> dict[str, object]:
    if trace.status != "filled" or trace.departure_index is None or trace.fill_index is None:
        raise ValueError("Only completed traces can become event rows")
    signal = candles[signal_index]
    departure = candles[trace.departure_index]
    fill = candles[trace.fill_index]
    between_departure_and_fill = candles[trace.departure_index:trace.fill_index + 1]
    if shape.direction == "lower_wick":
        max_index_relative, max_away_price = max(
            enumerate(candle.high for candle in between_departure_and_fill), key=lambda item: item[1]
        )
        fill_extreme = fill.low
        max_away_move_pct = percent_change(max_away_price, signal.close)
        max_away_from_wick_extreme_pct = percent_change(max_away_price, signal.low)
    else:
        max_index_relative, max_away_price = min(
            enumerate(candle.low for candle in between_departure_and_fill), key=lambda item: item[1]
        )
        fill_extreme = fill.high
        max_away_move_pct = percent_change(signal.close, max_away_price)
        max_away_from_wick_extreme_pct = percent_change(signal.high, max_away_price)

    features = signal_features(candles, signal_index, shape)
    event_id = f"{shape.direction}_{signal.open_time}"
    return {
        "event_id": event_id,
        "direction": shape.direction,
        "direction_sign": 1 if shape.direction == "lower_wick" else -1,
        "signal_open_time_utc": utc_iso(signal.open_time),
        "departure_open_time_utc": utc_iso(departure.open_time),
        "fill_open_time_utc": utc_iso(fill.open_time),
        "signal_open": signal.open,
        "signal_high": signal.high,
        "signal_low": signal.low,
        "signal_close": signal.close,
        "signal_volume_eth": signal.volume,
        "signal_quote_volume_usdt": signal.quote_volume,
        "signal_trades": signal.trades,
        "fill_extreme_price": fill_extreme,
        "signal_to_departure_minutes": (trace.departure_index - signal_index) * 5,
        "signal_to_fill_minutes": (trace.fill_index - signal_index) * 5,
        "departure_to_fill_minutes": (trace.fill_index - trace.departure_index) * 5,
        "max_away_price_before_fill": max_away_price,
        "max_away_move_pct_from_signal_close": max_away_move_pct,
        "max_away_move_pct_from_wick_extreme": max_away_from_wick_extreme_pct,
        "time_to_max_away_minutes": (trace.departure_index + max_index_relative - signal_index) * 5,
        **features,
    }


def path_rows(
    candles: list[Candle], event: dict[str, object], signal_index: int, trace: Trace
) -> Iterable[dict[str, object]]:
    assert trace.departure_index is not None and trace.fill_index is not None
    direction_sign = int(event["direction_sign"])
    wick_extreme = float(event["signal_low"] if direction_sign == 1 else event["signal_high"])

    def normalized_price_move(price: float) -> float:
        return direction_sign * (price / wick_extreme - 1) * 100

    for index in range(signal_index, trace.fill_index + 1):
        candle = candles[index]
        if index == signal_index:
            phase = "signal"
        elif index < trace.departure_index:
            phase = "pre_departure"
        elif index == trace.departure_index:
            phase = "departure"
        elif index < trace.fill_index:
            phase = "return_path"
        else:
            phase = "fill"
        yield {
            "event_id": event["event_id"],
            "direction": event["direction"],
            "direction_sign": direction_sign,
            "signal_open_time_utc": event["signal_open_time_utc"],
            "departure_open_time_utc": event["departure_open_time_utc"],
            "fill_open_time_utc": event["fill_open_time_utc"],
            "event_open_time_utc": utc_iso(candle.open_time),
            "offset_candles": index - signal_index,
            "phase": phase,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume_eth": candle.volume,
            "quote_volume_usdt": candle.quote_volume,
            "trades": candle.trades,
            "taker_buy_base_eth": candle.taker_buy_base,
            "taker_buy_quote_usdt": candle.taker_buy_quote,
            "normalized_open_pct": normalized_price_move(candle.open),
            "normalized_high_pct": normalized_price_move(candle.high),
            "normalized_low_pct": normalized_price_move(candle.low),
            "normalized_close_pct": normalized_price_move(candle.close),
        }


def write_csv_atomic(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    if path.suffix == ".gz":
        destination_handle = gzip.open(temporary, "wt", newline="", encoding="utf-8")
    else:
        destination_handle = temporary.open("w", newline="", encoding="utf-8")
    with destination_handle as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    os.replace(temporary, path)
    return count


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p75": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=project_dir / "data" / "ETHUSDT_5m_3y.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "data" / "filled_wick_study"
    )
    parser.add_argument(
        "--as-of",
        help="Exclusive UTC cutoff. Only data before this point may form historical events.",
    )
    parser.add_argument("--direction", choices=("lower", "upper", "both"), default="both")
    parser.add_argument("--body-max-pct", type=float, default=0.05)
    parser.add_argument("--dominant-wick-min-pct", type=float, default=0.75)
    parser.add_argument("--opposite-wick-max-pct", type=float, default=0.20)
    parser.add_argument(
        "--departure-bars",
        type=int,
        default=0,
        help="Maximum bars to confirm the move away; 0 searches all available history",
    )
    parser.add_argument(
        "--fill-bars",
        type=int,
        default=0,
        help="Maximum bars after departure to reach the wick extreme; 0 searches all available history",
    )
    args = parser.parse_args()

    if not 0 <= args.body_max_pct <= 1:
        parser.error("--body-max-pct must be between 0 and 1")
    if not 0 <= args.opposite_wick_max_pct <= 1:
        parser.error("--opposite-wick-max-pct must be between 0 and 1")
    if not 0 <= args.dominant_wick_min_pct <= 1:
        parser.error("--dominant-wick-min-pct must be between 0 and 1")
    if args.departure_bars < 0 or args.fill_bars < 0:
        parser.error("--departure-bars and --fill-bars cannot be negative")

    candles = read_candles(args.input)
    timestamps = [candle.open_time for candle in candles]
    as_of_ms = parse_utc(args.as_of) if args.as_of else None
    usable_end_index = (bisect.bisect_left(timestamps, as_of_ms) - 1) if as_of_ms else len(candles) - 1
    if usable_end_index < 1:
        raise RuntimeError("The as-of cutoff precedes the available data")

    allowed_directions = {
        "lower": {"lower_wick"},
        "upper": {"upper_wick"},
        "both": {"lower_wick", "upper_wick"},
    }[args.direction]
    departure_bars = args.departure_bars or None
    fill_bars = args.fill_bars or None

    statuses: dict[str, int] = {
        "filled": 0,
        "filled_before_departure": 0,
        "no_departure": 0,
        "unfilled": 0,
        "right_censored": 0,
    }
    candidates = 0
    events: list[dict[str, object]] = []
    traces: list[tuple[int, Trace, dict[str, object]]] = []
    for index in range(usable_end_index + 1):
        shape = classify_shape(
            candles[index], args.body_max_pct, args.dominant_wick_min_pct, args.opposite_wick_max_pct
        )
        if shape is None or shape.direction not in allowed_directions:
            continue
        candidates += 1
        trace = trace_completed_path(
            candles,
            index,
            shape.direction,
            usable_end_index,
            departure_bars,
            fill_bars,
        )
        statuses[trace.status] += 1
        if trace.status == "filled":
            event = event_row(candles, index, shape, trace)
            events.append(event)
            traces.append((index, trace, event))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_fields = list(events[0].keys()) if events else [
        "event_id", "direction", "signal_open_time_utc", "departure_open_time_utc", "fill_open_time_utc"
    ]
    event_path = args.output_dir / "filled_events.csv"
    path_path = args.output_dir / "filled_event_candles.csv.gz"
    events_written = write_csv_atomic(event_path, events, event_fields)

    path_fields = [
        "event_id", "direction", "direction_sign", "signal_open_time_utc", "departure_open_time_utc", "fill_open_time_utc",
        "event_open_time_utc", "offset_candles", "phase", "open", "high", "low", "close",
        "volume_eth", "quote_volume_usdt", "trades", "taker_buy_base_eth", "taker_buy_quote_usdt",
        "normalized_open_pct", "normalized_high_pct", "normalized_low_pct", "normalized_close_pct",
    ]

    def all_path_rows() -> Iterable[dict[str, object]]:
        for signal_index, trace, event in traces:
            yield from path_rows(candles, event, signal_index, trace)

    path_rows_written = write_csv_atomic(path_path, all_path_rows(), path_fields)
    summary = {
        "input": str(args.input),
        "output": {
            "filled_events_csv": str(event_path),
            "filled_event_candles_csv": str(path_path),
        },
        "analysis_cutoff_utc_exclusive": utc_iso(as_of_ms) if as_of_ms else None,
        "data_available_through_utc": utc_iso(candles[usable_end_index].open_time),
        "definition": {
            "direction": args.direction,
            "body_max_pct_of_range": args.body_max_pct * 100,
            "dominant_wick_min_pct_of_range": args.dominant_wick_min_pct * 100,
            "opposite_wick_max_pct_of_range": args.opposite_wick_max_pct * 100,
            "departure": "first later close beyond the signal high for lower wicks, or below the signal low for upper wicks",
            "full_fill": "first later touch of the signal wick extreme after departure",
            "departure_window_minutes": args.departure_bars * 5 if args.departure_bars else None,
            "fill_window_minutes_after_departure": args.fill_bars * 5 if args.fill_bars else None,
            "uncapped_windows_search_to": "the historical analysis cutoff" if not args.departure_bars or not args.fill_bars else None,
            "same_bar_ambiguity": "a bar that touches the wick extreme before a provable departure is excluded as filled_before_departure",
        },
        "candidate_signals": candidates,
        "outcomes": statuses,
        "completed_filled_events": events_written,
        "completed_path_candles": path_rows_written,
        "signal_to_fill_minutes": distribution([float(event["signal_to_fill_minutes"]) for event in events]),
        "max_away_move_pct_from_wick_extreme": distribution(
            [float(event["max_away_move_pct_from_wick_extreme"]) for event in events]
        ),
    }
    summary_path = args.output_dir / "summary.json"
    write_text_atomic(summary_path, json.dumps(summary, indent=2) + "\n")
    definition_path = args.output_dir / "DEFINITION.md"
    write_text_atomic(
        definition_path,
        "# Filled wick-path study\n\n"
        "Each output episode is a strict chronological sequence: signal candle, confirmed move away, then the first later touch of the full wick extreme. "
        "Lower- and upper-wick episodes are direction-normalized with `direction_sign`: the signal wick extreme is zero, and positive movement means moving away from that wick for both orientations. "
        "Only completed historical episodes are exported. The feature columns in `filled_events.csv` use data no later than the signal candle; the path and fill columns are labels for retrospective study, never prediction inputs.\n",
    )
    print(json.dumps({
        "summary": str(summary_path),
        "filled_events": events_written,
        "path_candles": path_rows_written,
        "outcomes": statuses,
    }))


if __name__ == "__main__":
    main()
