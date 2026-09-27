#!/usr/bin/env python3
"""Reference benchmark for the current Python wick scan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build_conditional_path_library import (  # noqa: E402
    detect_strict_signals,
    trace_clean_path,
)


def read_candles(path: Path, interval_ms: int, maximum_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=maximum_rows)
    for column in ("open_time", "open", "high", "low", "close", "volume", "close_time"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    frame = frame.sort_values("open_time", kind="stable").reset_index(drop=True)
    if frame["open_time"].duplicated().any():
        raise RuntimeError("duplicate candle timestamps")
    deltas = np.diff(frame["open_time"].to_numpy(dtype=np.int64))
    if len(deltas) and not np.all(deltas == interval_ms):
        raise RuntimeError("gap or unexpected timestamp interval")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("1m", "5m", "15m"), required=True)
    parser.add_argument("--maximum-fill-days", type=int, default=180)
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args()
    interval_minutes = int(args.timeframe.removesuffix("m"))
    interval_ms = interval_minutes * 60_000

    total_start = perf_counter()
    read_start = perf_counter()
    frame = read_candles(args.input, interval_ms, args.maximum_rows)
    read_seconds = perf_counter() - read_start

    detect_start = perf_counter()
    signals = detect_strict_signals(frame, "BENCHMARK", args.timeframe)
    detect_seconds = perf_counter() - detect_start

    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    maximum_future_bars = args.maximum_fill_days * 24 * 60 // interval_minutes
    trace_start = perf_counter()
    statuses: Counter[str] = Counter()
    for signal in signals.itertuples(index=False):
        status, _, _ = trace_clean_path(
            closes,
            highs,
            lows,
            int(signal.bar_index),
            int(signal.direction_sign),
            float(signal.wick_target),
            float(signal.opposite_extreme),
            maximum_future_bars,
        )
        statuses[status] += 1
    trace_seconds = perf_counter() - trace_start
    direction_counts = signals["direction_sign"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "engine": "python",
                "rows": len(frame),
                "strict_signals": len(signals),
                "lower_signals": int(direction_counts.get(1, 0)),
                "upper_signals": int(direction_counts.get(-1, 0)),
                "signal_index_checksum": int(signals["bar_index"].sum()),
                "statuses": dict(sorted(statuses.items())),
                "read_seconds": read_seconds,
                "detect_seconds": detect_seconds,
                "trace_seconds": trace_seconds,
                "total_seconds": perf_counter() - total_start,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
