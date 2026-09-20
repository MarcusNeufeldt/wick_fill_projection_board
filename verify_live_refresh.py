#!/usr/bin/env python3
"""Small non-network integrity check for the incremental kline updater."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from download_futures_klines import CSV_HEADER, INTERVAL_MS
from refresh_futures_klines import _append_rows_atomically, _validated_append_rows, source_snapshot


def kline(open_time_ms: int) -> list[str]:
    return [
        str(open_time_ms),
        "100.00",
        "102.00",
        "99.00",
        "101.00",
        "10.0",
        str(open_time_ms + INTERVAL_MS["5m"] - 1),
        "1000.0",
        "42",
        "5.0",
        "500.0",
        "0",
    ]


def expect_rejected(rows: list[list[str]], start_ms: int, end_exclusive_ms: int) -> None:
    try:
        _validated_append_rows(rows, start_ms=start_ms, end_exclusive_ms=end_exclusive_ms, interval_ms=INTERVAL_MS["5m"])
    except RuntimeError:
        return
    raise AssertionError("Expected discontinuous or duplicate rows to be rejected")


def main() -> None:
    interval_ms = INTERVAL_MS["5m"]
    start = 1_700_000_000_000 - 1_700_000_000_000 % interval_ms
    with tempfile.TemporaryDirectory(prefix="conditional-wick-refresh-") as directory:
        source = Path(directory) / "ETHUSDT_5m_5y.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            writer.writerow(kline(start))

        before = source_snapshot(source)
        appended = _validated_append_rows(
            [kline(start + interval_ms), kline(start + 2 * interval_ms)],
            start_ms=start + interval_ms,
            end_exclusive_ms=start + 3 * interval_ms,
            interval_ms=interval_ms,
        )
        _append_rows_atomically(source, appended)
        after = source_snapshot(source)
        if after.last_open_time_ms != start + 2 * interval_ms:
            raise AssertionError("Atomic append did not advance the final candle")
        expect_rejected(
            [kline(start + interval_ms), kline(start + interval_ms)],
            start + interval_ms,
            start + 3 * interval_ms,
        )
        expect_rejected(
            [kline(start + interval_ms), kline(start + 3 * interval_ms)],
            start + interval_ms,
            start + 3 * interval_ms,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "initial_last_open_time_ms": before.last_open_time_ms,
                    "final_last_open_time_ms": after.last_open_time_ms,
                    "appended_rows": len(appended),
                    "rejected_invalid_merges": 2,
                }
            )
        )


if __name__ == "__main__":
    main()
