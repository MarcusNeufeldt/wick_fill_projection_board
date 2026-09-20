#!/usr/bin/env python3
"""Build a five-year, multi-asset library of completed wick-fill trajectories.

The output is a *conditional* path library: only strict events that first
depart and later fully touch the dominant wick are exported as trajectories.
All signal features are known at the signal close; post-signal paths are labels
for historical matching only.  Early fills, non-departures, and censored events
are counted in the summary instead of being silently treated as evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FIVE_MINUTES_MS = 5 * 60 * 1000
FIFTEEN_MINUTES_MS = 15 * 60 * 1000
BODY_MAX_PCT = 0.05
DOMINANT_WICK_MIN_PCT = 0.75
OPPOSITE_WICK_MAX_PCT = 0.20
FEATURE_COLUMNS = [
    "body_pct_of_range",
    "dominant_wick_pct_of_range",
    "opposite_wick_pct_of_range",
    "range_pct_of_close",
    "range_vs_prior_20_median",
    "volume_vs_prior_20_mean",
    "aligned_prior_1h_return_pct",
]
REQUIRED_COLUMNS = {"open_time", "open", "high", "low", "close", "volume", "close_time"}


def utc_iso(timestamp_ms: int) -> str:
    return pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def read_five_minute_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw candle file: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ["open_time", "open", "high", "low", "close", "volume", "close_time"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    frame = frame.sort_values("open_time", kind="stable").reset_index(drop=True)
    if frame["open_time"].duplicated().any():
        raise RuntimeError(f"{path} contains duplicate candle timestamps")
    deltas = np.diff(frame["open_time"].to_numpy(dtype=np.int64))
    if len(deltas) and not np.all(deltas == FIVE_MINUTES_MS):
        raise RuntimeError(f"{path} contains a gap or non-5m timestamp")
    return frame


def restrict_to_common_window(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int, int]:
    start_ms = max(int(frame["open_time"].iat[0]) for frame in frames.values())
    end_exclusive_ms = min(int(frame["open_time"].iat[-1]) + FIVE_MINUTES_MS for frame in frames.values())
    if end_exclusive_ms <= start_ms:
        raise RuntimeError("The BTC and ETH source files do not overlap")
    expected_rows = (end_exclusive_ms - start_ms) // FIVE_MINUTES_MS
    trimmed: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        value = frame.loc[(frame["open_time"] >= start_ms) & (frame["open_time"] < end_exclusive_ms)].copy()
        value = value.reset_index(drop=True)
        if len(value) != expected_rows:
            raise RuntimeError(f"Common window for {symbol} is not contiguous")
        trimmed[symbol] = value
    return trimmed, start_ms, end_exclusive_ms


def resample_to_fifteen_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    value["bucket"] = (value["open_time"] // FIFTEEN_MINUTES_MS) * FIFTEEN_MINUTES_MS
    grouped = value.groupby("bucket", sort=True, observed=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        close_time=("close_time", "last"),
        _count=("open_time", "size"),
        _first_open=("open_time", "min"),
        _last_open=("open_time", "max"),
    ).reset_index(names="open_time")
    out = out.loc[
        (out["_count"] == 3)
        & (out["_last_open"] - out["_first_open"] == 2 * FIVE_MINUTES_MS)
    ].copy()
    out = out.drop(columns=["_count", "_first_open", "_last_open"]).reset_index(drop=True)
    deltas = np.diff(out["open_time"].to_numpy(dtype=np.int64))
    if len(deltas) and not np.all(deltas == FIFTEEN_MINUTES_MS):
        raise RuntimeError("15m resample contains an incomplete or discontinuous bucket")
    return out


def detect_strict_signals(frame: pd.DataFrame, asset: str, timeframe: str) -> pd.DataFrame:
    interval_ms = FIVE_MINUTES_MS if timeframe == "5m" else FIFTEEN_MINUTES_MS
    interval_minutes = interval_ms // 60_000
    bars_per_hour = 60 // interval_minutes
    value = frame.copy()
    value["bar_index"] = np.arange(len(value), dtype=np.int64)
    value["range"] = value["high"] - value["low"]
    valid_range = value["range"] > 0
    value["body_pct_of_range"] = np.where(
        valid_range, (value["close"] - value["open"]).abs() / value["range"], np.nan
    )
    value["lower_wick_pct_of_range"] = np.where(
        valid_range, (np.minimum(value["open"], value["close"]) - value["low"]) / value["range"], np.nan
    )
    value["upper_wick_pct_of_range"] = np.where(
        valid_range, (value["high"] - np.maximum(value["open"], value["close"])) / value["range"], np.nan
    )
    value["range_pct_of_close"] = np.where(valid_range, value["range"] / value["close"] * 100.0, np.nan)
    value["range_vs_prior_20_median"] = value["range"] / value["range"].rolling(20, min_periods=20).median().shift(1)
    value["volume_vs_prior_20_mean"] = value["volume"] / value["volume"].rolling(20, min_periods=20).mean().shift(1)
    value["prior_1h_return_pct"] = (value["close"] / value["close"].shift(bars_per_hour) - 1.0) * 100.0
    lower = (
        valid_range
        & (value["body_pct_of_range"] <= BODY_MAX_PCT)
        & (value["lower_wick_pct_of_range"] >= DOMINANT_WICK_MIN_PCT)
        & (value["upper_wick_pct_of_range"] <= OPPOSITE_WICK_MAX_PCT)
    )
    upper = (
        valid_range
        & (value["body_pct_of_range"] <= BODY_MAX_PCT)
        & (value["upper_wick_pct_of_range"] >= DOMINANT_WICK_MIN_PCT)
        & (value["lower_wick_pct_of_range"] <= OPPOSITE_WICK_MAX_PCT)
    )
    value["direction"] = np.select([lower, upper], ["lower_wick", "upper_wick"], default=None)
    value["direction_sign"] = np.select([lower, upper], [1, -1], default=0).astype("int8")
    value["dominant_wick_pct_of_range"] = np.where(
        value["direction"] == "lower_wick", value["lower_wick_pct_of_range"], value["upper_wick_pct_of_range"]
    )
    value["opposite_wick_pct_of_range"] = np.where(
        value["direction"] == "lower_wick", value["upper_wick_pct_of_range"], value["lower_wick_pct_of_range"]
    )
    value["aligned_prior_1h_return_pct"] = value["direction_sign"] * value["prior_1h_return_pct"]
    value["wick_target"] = np.where(value["direction"] == "lower_wick", value["low"], value["high"])
    value["opposite_extreme"] = np.where(value["direction"] == "lower_wick", value["high"], value["low"])
    signals = value.loc[(value["direction_sign"] != 0) & value[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    signals["asset"] = asset
    signals["timeframe"] = timeframe
    signals["interval_minutes"] = interval_minutes
    signals["signal_open_time_utc"] = pd.to_datetime(signals["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return signals


def trace_clean_path(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    signal_index: int,
    direction_sign: int,
    wick_target: float,
    opposite_extreme: float,
    maximum_future_bars: int,
) -> tuple[str, int | None, int | None]:
    """Return status, departure index, fill index with conservative OHLC ordering."""
    end_index = min(len(closes) - 1, signal_index + maximum_future_bars)
    fully_observed = signal_index + maximum_future_bars < len(closes)
    if end_index <= signal_index:
        return "right_censored", None, None
    if direction_sign == 1:
        fill_hits = np.flatnonzero(lows[signal_index + 1 : end_index + 1] <= wick_target)
        departure_hits = np.flatnonzero(closes[signal_index + 1 : end_index + 1] >= opposite_extreme)
    else:
        fill_hits = np.flatnonzero(highs[signal_index + 1 : end_index + 1] >= wick_target)
        departure_hits = np.flatnonzero(closes[signal_index + 1 : end_index + 1] <= opposite_extreme)
    first_fill_relative = int(fill_hits[0]) if len(fill_hits) else None
    first_departure_relative = int(departure_hits[0]) if len(departure_hits) else None
    if first_fill_relative is not None and (
        first_departure_relative is None or first_fill_relative <= first_departure_relative
    ):
        return "filled_before_departure", None, None
    if first_departure_relative is None:
        return ("no_departure" if fully_observed else "right_censored"), None, None
    departure_index = signal_index + 1 + first_departure_relative
    if direction_sign == 1:
        post_departure_fills = np.flatnonzero(lows[departure_index + 1 : end_index + 1] <= wick_target)
    else:
        post_departure_fills = np.flatnonzero(highs[departure_index + 1 : end_index + 1] >= wick_target)
    if len(post_departure_fills):
        fill_index = departure_index + 1 + int(post_departure_fills[0])
        return "filled", departure_index, fill_index
    return ("unfilled" if fully_observed else "right_censored"), departure_index, None


def episode_record(
    frame: pd.DataFrame,
    signal: dict[str, Any],
    departure_index: int,
    fill_index: int,
    path_file: str,
) -> dict[str, Any]:
    signal_index = int(signal["bar_index"])
    direction_sign = int(signal["direction_sign"])
    target = float(signal["wick_target"])
    path = frame.iloc[signal_index : fill_index + 1]
    if direction_sign == 1:
        away_values = (path["high"].to_numpy(dtype=float) / target - 1.0) * 100.0
    else:
        away_values = (1.0 - path["low"].to_numpy(dtype=float) / target) * 100.0
    peak_relative = int(np.argmax(away_values))
    episode_id = f"{signal['asset']}_{signal['timeframe']}_{signal['direction']}_{int(signal['open_time'])}"
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "path_file": path_file,
        "asset": signal["asset"],
        "timeframe": signal["timeframe"],
        "interval_minutes": int(signal["interval_minutes"]),
        "direction": signal["direction"],
        "direction_sign": direction_sign,
        "signal_open_time_ms": int(signal["open_time"]),
        "signal_open_time_utc": signal["signal_open_time_utc"],
        "signal_index": signal_index,
        "departure_index": departure_index,
        "fill_index": fill_index,
        "departure_open_time_utc": utc_iso(int(frame["open_time"].iat[departure_index])),
        "fill_open_time_utc": utc_iso(int(frame["open_time"].iat[fill_index])),
        "wick_target": target,
        "signal_open": float(signal["open"]),
        "signal_high": float(signal["high"]),
        "signal_low": float(signal["low"]),
        "signal_close": float(signal["close"]),
        "signal_volume": float(signal["volume"]),
        "signal_to_departure_bars": departure_index - signal_index,
        "signal_to_fill_bars": fill_index - signal_index,
        "signal_to_departure_minutes": (departure_index - signal_index) * int(signal["interval_minutes"]),
        "signal_to_fill_minutes": (fill_index - signal_index) * int(signal["interval_minutes"]),
        "max_away_move_pct_from_wick": float(away_values[peak_relative]),
        "time_to_max_away_bars": peak_relative,
        "time_to_max_away_minutes": peak_relative * int(signal["interval_minutes"]),
    }
    for field in FEATURE_COLUMNS:
        record[field] = float(signal[field])
    return record


def path_rows(frame: pd.DataFrame, event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    signal_index = int(event["signal_index"])
    departure_index = int(event["departure_index"])
    fill_index = int(event["fill_index"])
    direction_sign = int(event["direction_sign"])
    target = float(event["wick_target"])
    for index in range(signal_index, fill_index + 1):
        candle = frame.iloc[index]
        if index == signal_index:
            phase = "signal"
        elif index < departure_index:
            phase = "pre_departure"
        elif index == departure_index:
            phase = "departure"
        elif index < fill_index:
            phase = "return_path"
        else:
            phase = "fill"

        def normalized(value: float) -> float:
            return direction_sign * (value / target - 1.0) * 100.0

        yield {
            "episode_id": event["episode_id"],
            "asset": event["asset"],
            "timeframe": event["timeframe"],
            "direction": event["direction"],
            "direction_sign": direction_sign,
            "event_open_time_utc": utc_iso(int(candle["open_time"])),
            "offset_bars": index - signal_index,
            "phase": phase,
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
            "normalized_open_pct": normalized(float(candle["open"])),
            "normalized_high_pct": normalized(float(candle["high"])),
            "normalized_low_pct": normalized(float(candle["low"])),
            "normalized_close_pct": normalized(float(candle["close"])),
        }


PATH_FIELDS = [
    "episode_id",
    "asset",
    "timeframe",
    "direction",
    "direction_sign",
    "event_open_time_utc",
    "offset_bars",
    "phase",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "normalized_open_pct",
    "normalized_high_pct",
    "normalized_low_pct",
    "normalized_close_pct",
]


def write_paths(path: Path, frame: pd.DataFrame, events: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows_written = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATH_FIELDS)
        writer.writeheader()
        for event in events:
            for row in path_rows(frame, event):
                writer.writerow(row)
                rows_written += 1
    os.replace(temporary, path)
    return rows_written


def process_series(
    frame: pd.DataFrame,
    asset: str,
    timeframe: str,
    maximum_fill_days: int,
    paths_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    signals = detect_strict_signals(frame, asset, timeframe)
    interval_minutes = int(signals["interval_minutes"].iat[0])
    maximum_future_bars = maximum_fill_days * 24 * 60 // interval_minutes
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    path_file = f"{asset}_{timeframe}_paths.csv.gz"
    statuses: Counter[str] = Counter()
    events: list[dict[str, Any]] = []
    total = len(signals)
    for position, row in enumerate(signals.itertuples(index=False), start=1):
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
        if status == "filled" and departure_index is not None and fill_index is not None:
            events.append(episode_record(frame, signal, departure_index, fill_index, path_file))
        if position % 2_000 == 0 or position == total:
            print(
                json.dumps(
                    {
                        "stage": "trace_series",
                        "asset": asset,
                        "timeframe": timeframe,
                        "signals_processed": position,
                        "signals_total": total,
                        "completed_paths": len(events),
                    }
                ),
                flush=True,
            )
    path_rows_written = write_paths(paths_dir / path_file, frame, events)
    return events, dict(statuses), path_rows_written


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eth-5m", type=Path, default=root / "data" / "ETHUSDT_5m_5y.csv")
    parser.add_argument("--btc-5m", type=Path, default=root / "data" / "BTCUSDT_5m_5y.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "conditional_path_library_5y")
    parser.add_argument(
        "--max-fill-days",
        type=int,
        default=180,
        help="Require clean departure then fill within this many days; protects the path library from unbounded tails.",
    )
    args = parser.parse_args()
    if args.max_fill_days < 1:
        parser.error("--max-fill-days must be positive")

    source_paths = {"ETHUSDT": args.eth_5m.resolve(), "BTCUSDT": args.btc_5m.resolve()}
    raw: dict[str, pd.DataFrame] = {}
    for asset, path in source_paths.items():
        print(json.dumps({"stage": "read_source", "asset": asset}), flush=True)
        raw[asset] = read_five_minute_file(path)
    common, common_start_ms, common_end_exclusive_ms = restrict_to_common_window(raw)
    print(
        json.dumps(
            {
                "stage": "common_window",
                "start_open_time_utc": utc_iso(common_start_ms),
                "end_exclusive_utc": utc_iso(common_end_exclusive_ms),
            }
        ),
        flush=True,
    )
    out_dir = args.out_dir.resolve()
    paths_dir = out_dir / "paths"
    all_events: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for asset, source in common.items():
        for timeframe, series in (("5m", source), ("15m", resample_to_fifteen_minutes(source))):
            print(json.dumps({"stage": "begin_series", "asset": asset, "timeframe": timeframe}), flush=True)
            events, statuses, path_rows_written = process_series(
                series, asset, timeframe, args.max_fill_days, paths_dir
            )
            all_events.extend(events)
            strata.append(
                {
                    "asset": asset,
                    "timeframe": timeframe,
                    "interval_minutes": 5 if timeframe == "5m" else 15,
                    "strict_signal_count": int(sum(statuses.values())),
                    "outcomes": {key: int(value) for key, value in sorted(statuses.items())},
                    "completed_clean_paths": len(events),
                    "path_rows_written": path_rows_written,
                }
            )
    events_frame = pd.DataFrame.from_records(all_events)
    if events_frame.empty:
        raise RuntimeError("No completed clean paths were found")
    events_frame = events_frame.sort_values(
        ["signal_open_time_ms", "asset", "interval_minutes", "direction"], kind="stable"
    ).reset_index(drop=True)
    events_path = out_dir / "episodes.csv"
    atomic_write_csv(events_path, events_frame)
    summary = {
        "schema_version": "1.0.0",
        "purpose": "Conditional historical trajectory library for strict wick events; not an unconditional fill-probability model.",
        "market": "Binance USD-M Futures perpetual contracts",
        "sources": {asset: {"path": str(path), "sha256": sha256_file(path)} for asset, path in source_paths.items()},
        "common_source_window": {
            "start_open_time_utc": utc_iso(common_start_ms),
            "end_exclusive_utc": utc_iso(common_end_exclusive_ms),
        },
        "strict_signal_definition": {
            "body_pct_of_range_max": BODY_MAX_PCT,
            "dominant_wick_pct_of_range_min": DOMINANT_WICK_MIN_PCT,
            "opposite_wick_pct_of_range_max": OPPOSITE_WICK_MAX_PCT,
        },
        "trajectory_definition": {
            "clean_path": "A later close first crosses the opposite signal extreme; a still-later bar then touches the wick target.",
            "same_bar_policy": "If a bar touches the wick target before a provable earlier departure, it is classified as filled_before_departure and excluded from the conditional trajectory library.",
            "maximum_fill_days": args.max_fill_days,
            "direction_normalization": "The wick target is zero; positive values mean movement away from the target for both lower- and upper-wick events.",
        },
        "feature_columns": FEATURE_COLUMNS,
        "episodes_csv": str(events_path),
        "paths_directory": str(paths_dir),
        "completed_episode_count": int(len(events_frame)),
        "strata": strata,
    }
    summary_path = out_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    definition_path = out_dir / "DEFINITION.md"
    definition_path.write_text(
        "# Conditional wick-fill path library\n\n"
        "This library contains only historically completed strict paths: signal candle, provable departure, then first later full wick touch. "
        "All feature values come from the signal close. Normalised path values are relative to the wick target, with positive direction meaning price is away from that target. "
        f"The maximum signal-to-fill window is {args.max_fill_days} days; observations that cannot be fully evaluated by the source cutoff are counted as right-censored in `summary.json`.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episodes_csv": str(events_path),
                "summary_json": str(summary_path),
                "completed_episode_count": int(len(events_frame)),
                "strata": [
                    {
                        "asset": row["asset"],
                        "timeframe": row["timeframe"],
                        "completed_clean_paths": row["completed_clean_paths"],
                    }
                    for row in strata
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
