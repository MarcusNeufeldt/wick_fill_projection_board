#!/usr/bin/env python3
"""Run both scanners, reject parity differences, and report measured speedup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARITY_FIELDS = (
    "rows",
    "strict_signals",
    "lower_signals",
    "upper_signals",
    "signal_index_checksum",
    "statuses",
)


def run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON: {command}")
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("1m", "5m", "15m"), required=True)
    parser.add_argument("--maximum-fill-days", type=int, default=180)
    parser.add_argument("--maximum-rows", type=int)
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=HERE / "target" / "release" / "wick-throughput-poc.exe",
    )
    args = parser.parse_args()
    shared = [
        "--input",
        str(args.input),
        "--timeframe",
        args.timeframe,
        "--maximum-fill-days",
        str(args.maximum_fill_days),
    ]
    if args.maximum_rows is not None:
        shared.extend(("--maximum-rows", str(args.maximum_rows)))
    python_result = run([sys.executable, str(HERE / "python_wick_scan.py"), *shared])
    rust_result = run([str(args.rust_binary), *shared])
    differences = {
        field: {"python": python_result[field], "rust": rust_result[field]}
        for field in PARITY_FIELDS
        if python_result[field] != rust_result[field]
    }
    if differences:
        print(json.dumps({"parity": "failed", "differences": differences}, indent=2))
        raise SystemExit(1)
    python_seconds = float(python_result["total_seconds"])
    rust_seconds = float(rust_result["total_seconds"])
    print(
        json.dumps(
            {
                "parity": "passed",
                "timeframe": args.timeframe,
                "rows": python_result["rows"],
                "strict_signals": python_result["strict_signals"],
                "statuses": python_result["statuses"],
                "python_seconds": python_seconds,
                "rust_seconds": rust_seconds,
                "end_to_end_speedup": python_seconds / rust_seconds,
                "trace_speedup": float(python_result["trace_seconds"])
                / float(rust_result["trace_seconds"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
