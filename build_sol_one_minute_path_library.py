#!/usr/bin/env python3
"""Build an isolated one-minute clean-fill trajectory library for one asset.

The five-asset library deliberately contains only 5m and 15m trajectories.
Keeping each 1m asset separate prevents minute data from changing the existing
path population, cache footprint, or 5m V2 calibration.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_conditional_path_library import (
    BODY_MAX_PCT,
    DOMINANT_WICK_MIN_PCT,
    FEATURE_COLUMNS,
    ONE_MINUTE_MS,
    OPPOSITE_WICK_MAX_PCT,
    atomic_write_csv,
    atomic_write_json,
    detect_strict_signals,
    episode_record,
    read_one_minute_file,
    sha256_file,
    trace_clean_path,
    utc_iso,
)
from conditional_wick_assets import DEFAULT_ONE_MINUTE_ASSET, SUPPORTED_ASSETS, one_minute_library_dir


COMPACT_PATH_COLUMNS = [
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


def trace_events(
    source: pd.DataFrame, asset: str, maximum_fill_days: int, path_file: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Find the same clean completed episodes as the shared builder, without its unused raw-column export."""
    signals = detect_strict_signals(source, asset, "1m")
    closes = source["close"].to_numpy(dtype=float)
    highs = source["high"].to_numpy(dtype=float)
    lows = source["low"].to_numpy(dtype=float)
    maximum_future_bars = maximum_fill_days * 24 * 60
    statuses: Counter[str] = Counter()
    events: list[dict[str, Any]] = []
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
            events.append(episode_record(source, signal, departure_index, fill_index, path_file))
        if position % 2_000 == 0 or position == len(signals):
            print(
                json.dumps(
                    {
                        "stage": "trace_series",
                        "asset": asset,
                        "timeframe": "1m",
                        "signals_processed": position,
                        "signals_total": int(len(signals)),
                        "completed_paths": int(len(events)),
                    }
                ),
                flush=True,
            )
    return events, dict(statuses)


def write_compact_paths(
    path: Path, source: pd.DataFrame, events: list[dict[str, Any]], asset: str = DEFAULT_ONE_MINUTE_ASSET
) -> int:
    """Write only the normalized candle fields used by matching and chart rendering.

    The generic library retains diagnostic raw OHLCV fields.  At one-minute
    density those unused columns dominate build time and cache size, so this
    isolated library intentionally stores the server's exact read contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows_written = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        for position, event in enumerate(events, start=1):
            start = int(event["signal_index"])
            end = int(event["fill_index"])
            candles = source.iloc[start : end + 1]
            count = len(candles)
            target = float(event["wick_target"])
            direction_sign = int(event["direction_sign"])

            def normalised(column: str) -> np.ndarray:
                values = candles[column].to_numpy(dtype=float)
                return direction_sign * (values / target - 1.0) * 100.0

            compact = pd.DataFrame(
                {
                    "episode_id": str(event["episode_id"]),
                    "asset": str(event["asset"]),
                    "timeframe": str(event["timeframe"]),
                    "direction": str(event["direction"]),
                    "direction_sign": direction_sign,
                    "offset_bars": np.arange(count, dtype=np.int32),
                    "normalized_open_pct": normalised("open"),
                    "normalized_high_pct": normalised("high"),
                    "normalized_low_pct": normalised("low"),
                    "normalized_close_pct": normalised("close"),
                },
                columns=COMPACT_PATH_COLUMNS,
            )
            compact.to_csv(handle, index=False, header=position == 1, lineterminator="\n")
            rows_written += count
            if position % 1_000 == 0 or position == len(events):
                handle.flush()
                print(
                    json.dumps(
                        {
                            "stage": "write_paths",
                            "asset": asset,
                            "timeframe": "1m",
                            "paths_written": position,
                            "paths_total": len(events),
                            "path_rows_written": rows_written,
                        }
                    ),
                    flush=True,
                )
    os.replace(temporary, path)
    return rows_written


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=SUPPORTED_ASSETS, default=DEFAULT_ONE_MINUTE_ASSET)
    parser.add_argument(
        "--source",
        "--sol-1m",
        dest="source",
        type=Path,
        help="Contiguous 1m source CSV; --sol-1m remains a compatibility alias.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--max-fill-days",
        type=int,
        default=180,
        help="Require clean departure then fill within this many days; protects the path library from unbounded tails.",
    )
    args = parser.parse_args()
    if args.max_fill_days < 1:
        parser.error("--max-fill-days must be positive")

    asset = str(args.asset)
    source_path = (args.source or root / "data" / f"{asset}_1m_5y.csv").resolve()
    print(json.dumps({"stage": "read_source", "asset": asset, "timeframe": "1m"}), flush=True)
    source = read_one_minute_file(source_path)
    start_ms = int(source["open_time"].iat[0])
    end_exclusive_ms = int(source["open_time"].iat[-1]) + ONE_MINUTE_MS

    out_dir = (args.out_dir or one_minute_library_dir(root, asset)).resolve()
    paths_dir = out_dir / "paths"
    print(json.dumps({"stage": "begin_series", "asset": asset, "timeframe": "1m"}), flush=True)
    path_file = f"{asset}_1m_paths.csv.gz"
    events, statuses = trace_events(source, asset, args.max_fill_days, path_file)
    path_rows_written = write_compact_paths(paths_dir / path_file, source, events, asset)
    events_frame = pd.DataFrame.from_records(events)
    if events_frame.empty:
        raise RuntimeError(f"No completed clean {asset} 1m paths were found")
    events_frame = events_frame.sort_values(
        ["signal_open_time_ms", "asset", "interval_minutes", "direction"], kind="stable"
    ).reset_index(drop=True)
    events_path = out_dir / "episodes.csv"
    atomic_write_csv(events_path, events_frame)

    strata = [
        {
            "asset": asset,
            "timeframe": "1m",
            "interval_minutes": 1,
            "strict_signal_count": int(sum(statuses.values())),
            "outcomes": {key: int(value) for key, value in sorted(statuses.items())},
            "completed_clean_paths": int(len(events)),
            "path_rows_written": int(path_rows_written),
        }
    ]
    summary = {
        "schema_version": "1.0.0",
        "purpose": f"Standalone {asset} 1m conditional historical trajectory library for strict wick events; not an unconditional fill-probability model.",
        "market": "Binance USD-M Futures perpetual contracts",
        "sources": {asset: {"path": str(source_path), "sha256": sha256_file(source_path)}},
        "common_source_window": {
            "start_open_time_utc": utc_iso(start_ms),
            "end_exclusive_utc": utc_iso(end_exclusive_ms),
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
    (out_dir / "DEFINITION.md").write_text(
        f"# {asset} 1m conditional wick-fill path library\n\n"
        f"This standalone library contains only historically completed {asset} one-minute strict paths: signal candle, provable departure, then first later full wick touch. "
        "It does not alter the five-asset 5m/15m library or its V2 calibration. "
        "All normalised path values are relative to the wick target, with positive direction meaning price is away from that target. "
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
                        "asset": asset,
                        "timeframe": "1m",
                        "completed_clean_paths": int(len(events)),
                    }
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
