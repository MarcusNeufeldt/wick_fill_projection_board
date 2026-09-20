#!/usr/bin/env python3
"""Build a leak-safe BTC/ETH, 5m/15m panel for strict wick-fill research.

The panel is deliberately a labelled research dataset, not a trading signal.
Each row is a completed strict-wick candle with features available when that
candle closed.  Its future labels are calculated separately for 1d, 7d, and
30d horizons.  Rows without enough subsequent candles retain blank labels so
they cannot accidentally enter a backtest as mature observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FIVE_MINUTES_MS = 5 * 60 * 1000
FIFTEEN_MINUTES_MS = 15 * 60 * 1000
BODY_MAX_PCT = 0.05
DOMINANT_WICK_MIN_PCT = 0.75
OPPOSITE_WICK_MAX_PCT = 0.20
HORIZON_MINUTES = {"1d": 24 * 60, "7d": 7 * 24 * 60, "30d": 30 * 24 * 60}
REQUIRED_COLUMNS = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
}


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
    numeric_columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    for column in numeric_columns:
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


def build_signal_rows(frame: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    interval_ms = FIVE_MINUTES_MS if timeframe == "5m" else FIFTEEN_MINUTES_MS
    interval_minutes = interval_ms // 60_000
    bars_per_hour = 60 // interval_minutes
    value = frame.copy()
    value["bar_index"] = np.arange(len(value), dtype=np.int64)
    value["range"] = value["high"] - value["low"]
    valid_range = value["range"] > 0
    value["body_pct_of_range"] = np.where(
        valid_range,
        (value["close"] - value["open"]).abs() / value["range"],
        np.nan,
    )
    value["lower_wick_pct_of_range"] = np.where(
        valid_range,
        (np.minimum(value["open"], value["close"]) - value["low"]) / value["range"],
        np.nan,
    )
    value["upper_wick_pct_of_range"] = np.where(
        valid_range,
        (value["high"] - np.maximum(value["open"], value["close"])) / value["range"],
        np.nan,
    )
    value["range_pct_of_close"] = np.where(valid_range, value["range"] / value["close"] * 100.0, np.nan)
    value["range_vs_prior_20_median"] = value["range"] / value["range"].rolling(20, min_periods=20).median().shift(1)
    value["volume_vs_prior_20_mean"] = value["volume"] / value["volume"].rolling(20, min_periods=20).mean().shift(1)
    value["prior_1h_return_pct"] = (value["close"] / value["close"].shift(bars_per_hour) - 1.0) * 100.0

    lower_signal = (
        valid_range
        & (value["body_pct_of_range"] <= BODY_MAX_PCT)
        & (value["lower_wick_pct_of_range"] >= DOMINANT_WICK_MIN_PCT)
        & (value["upper_wick_pct_of_range"] <= OPPOSITE_WICK_MAX_PCT)
    )
    upper_signal = (
        valid_range
        & (value["body_pct_of_range"] <= BODY_MAX_PCT)
        & (value["upper_wick_pct_of_range"] >= DOMINANT_WICK_MIN_PCT)
        & (value["lower_wick_pct_of_range"] <= OPPOSITE_WICK_MAX_PCT)
    )
    value["direction"] = np.select([lower_signal, upper_signal], ["lower_wick", "upper_wick"], default=None)
    value["direction_sign"] = np.select([lower_signal, upper_signal], [1, -1], default=0).astype("int8")
    value["dominant_wick_pct_of_range"] = np.where(
        value["direction"] == "lower_wick",
        value["lower_wick_pct_of_range"],
        value["upper_wick_pct_of_range"],
    )
    value["opposite_wick_pct_of_range"] = np.where(
        value["direction"] == "lower_wick",
        value["upper_wick_pct_of_range"],
        value["lower_wick_pct_of_range"],
    )
    value["aligned_prior_1h_return_pct"] = value["direction_sign"] * value["prior_1h_return_pct"]
    needed_features = [
        "body_pct_of_range",
        "dominant_wick_pct_of_range",
        "opposite_wick_pct_of_range",
        "range_pct_of_close",
        "range_vs_prior_20_median",
        "volume_vs_prior_20_mean",
        "prior_1h_return_pct",
        "aligned_prior_1h_return_pct",
    ]
    signals = value.loc[(value["direction_sign"] != 0) & value[needed_features].notna().all(axis=1)].copy()
    signals["asset"] = symbol
    signals["timeframe"] = timeframe
    signals["interval_minutes"] = interval_minutes
    signals["signal_open_time_utc"] = pd.to_datetime(signals["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return signals


def outcome_for_horizon(fill_bars: int | None, departure_bars: int | None, horizon_bars: int) -> str:
    fill_observed = fill_bars is not None and fill_bars <= horizon_bars
    departure_observed = departure_bars is not None and departure_bars <= horizon_bars
    if fill_observed:
        if departure_observed and departure_bars < fill_bars:
            return "departure_then_fill"
        return "filled_before_or_with_departure"
    if departure_observed:
        return "departed_not_filled"
    return "no_departure_no_fill"


def label_signal_rows(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    interval_minutes = int(signals["interval_minutes"].iat[0])
    horizon_bars = {name: minutes // interval_minutes for name, minutes in HORIZON_MINUTES.items()}
    maximum_horizon_bars = max(horizon_bars.values())
    records: list[dict[str, Any]] = []
    base_columns = [
        "asset",
        "timeframe",
        "interval_minutes",
        "bar_index",
        "open_time",
        "signal_open_time_utc",
        "direction",
        "direction_sign",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "body_pct_of_range",
        "dominant_wick_pct_of_range",
        "opposite_wick_pct_of_range",
        "range_pct_of_close",
        "range_vs_prior_20_median",
        "volume_vs_prior_20_mean",
        "prior_1h_return_pct",
        "aligned_prior_1h_return_pct",
    ]
    for row in signals[base_columns].itertuples(index=False):
        record = row._asdict()
        index = int(record["bar_index"])
        available_bars = min(maximum_horizon_bars, len(closes) - index - 1)
        future_end = index + 1 + available_bars
        direction = record["direction"]
        if direction == "lower_wick":
            fill_hits = np.flatnonzero(lows[index + 1 : future_end] <= float(record["low"]))
            departure_hits = np.flatnonzero(closes[index + 1 : future_end] >= float(record["high"]))
        else:
            fill_hits = np.flatnonzero(highs[index + 1 : future_end] >= float(record["high"]))
            departure_hits = np.flatnonzero(closes[index + 1 : future_end] <= float(record["low"]))
        fill_bars = int(fill_hits[0] + 1) if len(fill_hits) else None
        departure_bars = int(departure_hits[0] + 1) if len(departure_hits) else None
        record["first_fill_bars_within_30d"] = fill_bars
        record["first_departure_bars_within_30d"] = departure_bars
        record["first_fill_minutes_within_30d"] = fill_bars * interval_minutes if fill_bars is not None else None
        record["first_departure_minutes_within_30d"] = departure_bars * interval_minutes if departure_bars is not None else None
        for horizon_name, bars in horizon_bars.items():
            label_is_mature = index + bars < len(closes)
            if not label_is_mature:
                record[f"fill_{horizon_name}"] = None
                record[f"departure_{horizon_name}"] = None
                record[f"clean_departure_then_fill_{horizon_name}"] = None
                record[f"outcome_{horizon_name}"] = None
                continue
            fill_observed = fill_bars is not None and fill_bars <= bars
            departure_observed = departure_bars is not None and departure_bars <= bars
            clean_fill = fill_observed and departure_observed and departure_bars < fill_bars
            record[f"fill_{horizon_name}"] = int(fill_observed)
            record[f"departure_{horizon_name}"] = int(departure_observed)
            record[f"clean_departure_then_fill_{horizon_name}"] = int(clean_fill)
            record[f"outcome_{horizon_name}"] = outcome_for_horizon(fill_bars, departure_bars, bars)
        records.append(record)
    panel = pd.DataFrame.from_records(records)
    return panel.sort_values(["open_time", "asset", "interval_minutes", "direction"], kind="stable").reset_index(drop=True)


def rate(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def summarize_panel(panel: pd.DataFrame, common_start_ms: int, common_end_exclusive_ms: int, source_paths: dict[str, Path]) -> dict[str, Any]:
    strata: list[dict[str, Any]] = []
    for (asset, timeframe), group in panel.groupby(["asset", "timeframe"], sort=True):
        result: dict[str, Any] = {
            "asset": asset,
            "timeframe": timeframe,
            "interval_minutes": int(group["interval_minutes"].iat[0]),
            "strict_signal_count": int(len(group)),
            "lower_wick_count": int((group["direction"] == "lower_wick").sum()),
            "upper_wick_count": int((group["direction"] == "upper_wick").sum()),
        }
        for horizon in HORIZON_MINUTES:
            mature = group[f"fill_{horizon}"].notna()
            result[f"mature_{horizon}_count"] = int(mature.sum())
            result[f"fill_{horizon}_rate"] = rate(group.loc[mature, f"fill_{horizon}"])
            result[f"clean_departure_then_fill_{horizon}_rate"] = rate(group.loc[mature, f"clean_departure_then_fill_{horizon}"])
            outcomes = group.loc[mature, f"outcome_{horizon}"].value_counts().to_dict()
            result[f"outcome_{horizon}_counts"] = {str(key): int(value) for key, value in outcomes.items()}
        strata.append(result)
    return {
        "schema_version": "1.0.0",
        "market": "Binance USD-M Futures perpetual contracts",
        "strict_signal_definition": {
            "body_pct_of_range_max": BODY_MAX_PCT,
            "dominant_wick_pct_of_range_min": DOMINANT_WICK_MIN_PCT,
            "opposite_wick_pct_of_range_max": OPPOSITE_WICK_MAX_PCT,
            "directions": ["lower_wick", "upper_wick"],
        },
        "label_definition": {
            "fill": "A later candle touches the signal wick extreme.",
            "departure": "A later candle closes beyond the signal candle's opposite extreme.",
            "clean_departure_then_fill": "Departure happens in an earlier candle than the first full wick touch; a same-bar tie is conservatively not clean.",
            "horizons_minutes": HORIZON_MINUTES,
        },
        "common_source_window": {
            "start_open_time_utc": utc_iso(common_start_ms),
            "end_exclusive_utc": utc_iso(common_end_exclusive_ms),
        },
        "source_files": {
            symbol: {"path": str(path), "sha256": sha256_file(path)} for symbol, path in source_paths.items()
        },
        "row_count": int(len(panel)),
        "strata": strata,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eth-5m", type=Path, default=root / "data" / "ETHUSDT_5m_3y.csv")
    parser.add_argument("--btc-5m", type=Path, default=root / "data" / "BTCUSDT_5m_3y.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "multi_asset_panel")
    args = parser.parse_args()

    source_paths = {"ETHUSDT": args.eth_5m.resolve(), "BTCUSDT": args.btc_5m.resolve()}
    raw: dict[str, pd.DataFrame] = {}
    for symbol, path in source_paths.items():
        print(json.dumps({"stage": "read_source", "asset": symbol}), flush=True)
        raw[symbol] = read_five_minute_file(path)
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
    parts: list[pd.DataFrame] = []
    for symbol, source in common.items():
        print(json.dumps({"stage": "label", "asset": symbol, "timeframe": "5m"}), flush=True)
        parts.append(label_signal_rows(source, build_signal_rows(source, symbol, "5m")))
        fifteen_minute = resample_to_fifteen_minutes(source)
        print(json.dumps({"stage": "label", "asset": symbol, "timeframe": "15m"}), flush=True)
        parts.append(label_signal_rows(fifteen_minute, build_signal_rows(fifteen_minute, symbol, "15m")))
    panel = pd.concat(parts, ignore_index=True)
    output_dir = args.out_dir.resolve()
    csv_path = output_dir / "panel_signals.csv"
    summary_path = output_dir / "panel_summary.json"
    atomic_write_csv(csv_path, panel)
    summary = summarize_panel(panel, common_start_ms, common_end_exclusive_ms, source_paths)
    summary["panel_csv"] = str(csv_path)
    summary["panel_csv_sha256"] = sha256_file(csv_path)
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "panel_csv": str(csv_path),
                "summary_json": str(summary_path),
                "rows": int(len(panel)),
                "common_window": summary["common_source_window"],
                "strata": [
                    {
                        "asset": item["asset"],
                        "timeframe": item["timeframe"],
                        "strict_signal_count": item["strict_signal_count"],
                    }
                    for item in summary["strata"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
