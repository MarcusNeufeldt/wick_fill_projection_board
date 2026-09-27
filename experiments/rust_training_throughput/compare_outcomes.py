#!/usr/bin/env python3
"""Compare the full Python and Rust outcome kernels on identical candles."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_FIELDS = (
    "rows",
    "strict_signals",
    "lower_signals",
    "upper_signals",
    "signal_index_checksum",
    "statuses",
)


def run(command: list[str]) -> dict[str, Any]:
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


def compare(left: Any, right: Any, path: str, differences: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append(
                f"{path}: key mismatch {sorted(left)} != {sorted(right)}"
            )
            return
        for key in sorted(left):
            compare(left[key], right[key], f"{path}.{key}", differences)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-9):
                differences.append(f"{path}: {left} != {right}")
        elif left != right:
            differences.append(f"{path}: {left} != {right}")
        return
    if left != right:
        differences.append(f"{path}: {left!r} != {right!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timeframe", choices=("1m", "5m"), required=True)
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
    python_result = run([sys.executable, str(HERE / "python_outcome_scan.py"), *shared])
    rust_result = run([str(args.rust_binary), "--mode", "outcomes", *shared])
    differences = []
    for field in BASE_FIELDS:
        compare(python_result[field], rust_result[field], field, differences)
    compare(python_result["outcomes"], rust_result["outcomes"], "outcomes", differences)
    if differences:
        print(json.dumps({"parity": "failed", "differences": differences[:50]}, indent=2))
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
                "observations": python_result["outcomes"]["observation_count"],
                "python_seconds": python_seconds,
                "rust_seconds": rust_seconds,
                "end_to_end_speedup": python_seconds / rust_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
