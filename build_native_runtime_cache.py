#!/usr/bin/env python3
"""Materialize a fast local dashboard cache for one immutable path library.

The cache is derived only from the existing completed-path library.  It does
not change signal definitions, source candles, episode eligibility, or model
weights.  The dashboard validates the library fingerprint before using it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_conditional_path_scenarios import prepare_path_states, read_library_paths, with_fill_close_time_ms
from conditional_wick_assets import DEFAULT_ONE_MINUTE_ASSET, SUPPORTED_ASSETS, default_library_dir, one_minute_library_dir
from native_wick_matcher import RUNTIME_STATE_COLUMNS, NativeStateIndex, write_runtime_cache


def default_library(root: Path, timeframe: str, asset: str) -> Path:
    return one_minute_library_dir(root, asset) if timeframe == "1m" else default_library_dir(root)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", choices=("1m", "5m", "15m"), default="5m")
    parser.add_argument(
        "--asset",
        choices=SUPPORTED_ASSETS,
        default=DEFAULT_ONE_MINUTE_ASSET,
        help="One-minute asset whose isolated library should be cached.",
    )
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing cache only after rebuilding it successfully.")
    args = parser.parse_args()

    library_dir = (args.library_dir or default_library(root, args.timeframe, args.asset)).resolve()
    episodes_path = library_dir / "episodes.csv"
    episodes = with_fill_close_time_ms(pd.read_csv(episodes_path))
    episodes["signal_open_time_ms"] = pd.to_numeric(episodes["signal_open_time_ms"], errors="raise").astype("int64")
    trajectory_events = episodes.loc[episodes["timeframe"].eq(args.timeframe)].copy()
    if trajectory_events.empty:
        raise RuntimeError(f"No completed {args.timeframe} trajectory episodes are available")
    print(json.dumps({"stage": "read_paths", "timeframe": args.timeframe, "episode_count": len(trajectory_events)}), flush=True)
    paths = read_library_paths(library_dir / "paths", trajectory_events["path_file"].tolist())
    print(json.dumps({"stage": "prepare_states", "path_rows": len(paths)}), flush=True)
    states = prepare_path_states(trajectory_events, paths).loc[:, RUNTIME_STATE_COLUMNS].reset_index(drop=True)
    native_state_index = NativeStateIndex.from_frames(trajectory_events, states, args.timeframe)
    output = write_runtime_cache(
        library_dir,
        args.timeframe,
        states,
        native_state_index,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "timeframe": args.timeframe,
                "cache": str(output),
                "state_rows": len(states),
                "candidate_state_rows": len(native_state_index.state_rows),
                "matching_backend": native_state_index.backend,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
