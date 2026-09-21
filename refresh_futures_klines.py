#!/usr/bin/env python3
"""Incrementally append completed Binance USD-M Futures klines.

The historical downloader deliberately creates a fixed, auditable snapshot.
This companion keeps that snapshot current without replacing its history: it
asks Binance for only the candles after the local last completed bar, rejects
incomplete or discontinuous responses, and atomically swaps in the appended
CSV.  It is safe to run repeatedly; an already-current source is left
untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from download_futures_klines import (
    CSV_HEADER,
    FUTURES_API,
    INTERVAL_MS,
    current_month_rows,
    normalize_timestamp,
    request_json,
    utc_iso,
)
from conditional_wick_assets import SUPPORTED_ASSETS


DEFAULT_SYMBOLS = SUPPORTED_ASSETS


@dataclass(frozen=True)
class SourceSnapshot:
    """The small, stable portion of a CSV needed to schedule a refresh."""

    path: str
    mtime_ns: int
    size_bytes: int
    last_open_time_ms: int
    last_close_time_ms: int

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["last_closed_candle_open_time_utc"] = utc_iso(self.last_open_time_ms)
        payload["last_closed_candle_close_time_utc"] = utc_iso(self.last_close_time_ms)
        return payload


@dataclass(frozen=True)
class RefreshResult:
    """A serialisable record of one non-destructive source refresh attempt."""

    symbol: str
    interval: str
    source_path: str
    metadata_path: str
    changed: bool
    rows_added: int
    api_requests: int
    server_time_utc: str
    completed_end_exclusive_utc: str
    prior_last_closed_candle_open_time_utc: str
    last_closed_candle_open_time_utc: str
    source_mtime_ns: int

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def _tail_row(path: Path) -> list[str]:
    """Read the final CSV row without loading the five-year file into memory."""

    if not path.exists():
        raise FileNotFoundError(f"Missing candle source: {path}")
    size = path.stat().st_size
    if size == 0:
        raise RuntimeError(f"Candle source is empty: {path}")
    # Kline rows are short.  A generous tail keeps this O(1) even for years of
    # history while still producing a useful error if the file is malformed.
    with path.open("rb") as handle:
        handle.seek(max(0, size - 131_072))
        tail = handle.read()
    for raw in reversed(tail.splitlines()):
        if raw.strip():
            try:
                return next(csv.reader([raw.decode("utf-8")]))
            except (UnicodeDecodeError, StopIteration, csv.Error) as error:
                raise RuntimeError(f"Could not parse the final candle row in {path}") from error
    raise RuntimeError(f"Candle source has no data rows: {path}")


def source_snapshot(path: Path) -> SourceSnapshot:
    """Return source generation details without parsing the full candle file."""

    row = _tail_row(path)
    if len(row) < len(CSV_HEADER):
        raise RuntimeError(f"Final candle row in {path} has {len(row)} columns; expected {len(CSV_HEADER)}")
    try:
        last_open_time_ms = normalize_timestamp(row[0])
        last_close_time_ms = normalize_timestamp(row[6])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Final candle row in {path} has invalid timestamps") from error
    if last_close_time_ms <= last_open_time_ms:
        raise RuntimeError(f"Final candle row in {path} closes before it opens")
    stat = path.stat()
    return SourceSnapshot(
        path=str(path),
        mtime_ns=stat.st_mtime_ns,
        size_bytes=stat.st_size,
        last_open_time_ms=last_open_time_ms,
        last_close_time_ms=last_close_time_ms,
    )


def _normalise_api_row(row: Sequence[object], interval_ms: int) -> list[str]:
    if len(row) < len(CSV_HEADER):
        raise RuntimeError(f"Binance returned a malformed kline with {len(row)} columns")
    normalised = [str(value) for value in row[: len(CSV_HEADER)]]
    open_time_ms = normalize_timestamp(normalised[0])
    close_time_ms = normalize_timestamp(normalised[6])
    expected_close_time_ms = open_time_ms + interval_ms - 1
    if close_time_ms != expected_close_time_ms:
        raise RuntimeError(
            "Binance returned a kline whose close timestamp does not match the requested interval "
            f"({open_time_ms=} {close_time_ms=} {expected_close_time_ms=})"
        )
    normalised[0] = str(open_time_ms)
    normalised[6] = str(close_time_ms)
    return normalised


def _validated_append_rows(
    raw_rows: Sequence[Sequence[object]],
    *,
    start_ms: int,
    end_exclusive_ms: int,
    interval_ms: int,
) -> list[list[str]]:
    """Require a complete, ordered, non-overlapping run before writing it."""

    rows = [_normalise_api_row(row, interval_ms) for row in raw_rows]
    rows.sort(key=lambda row: int(row[0]))
    timestamps = [int(row[0]) for row in rows]
    expected = list(range(start_ms, end_exclusive_ms, interval_ms))
    if timestamps != expected:
        if len(timestamps) != len(set(timestamps)):
            problem = "duplicate timestamps"
        elif timestamps and (timestamps[0] < start_ms or timestamps[-1] >= end_exclusive_ms):
            problem = "timestamps outside the completed refresh window"
        else:
            problem = "a gap or missing completed candle"
        raise RuntimeError(
            f"Binance refresh response has {problem}: expected {len(expected)} contiguous candles, got {len(timestamps)}"
        )
    return rows


def _replace_with_retry(temporary: Path, destination: Path, attempts: int = 6) -> None:
    """Retry a Windows sharing violation without weakening the atomic-swap contract."""
    for attempt in range(attempts):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.15 * (attempt + 1))


def _append_rows_atomically(path: Path, rows: Sequence[Sequence[str]]) -> None:
    """Copy the verified source, append rows, and replace it in one operation."""

    if not rows:
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with path.open("rb") as source:
            source.seek(-1, os.SEEK_END)
            needs_newline = source.read(1) not in {b"\n", b"\r"}
            source.seek(0)
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        with temporary.open("a", newline="", encoding="utf-8") as destination:
            if needs_newline:
                destination.write("\n")
            csv.writer(destination).writerows(rows)
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def exchange_server_time_ms() -> int:
    """Read Binance's clock once so multi-asset refreshes share one cutoff."""
    payload = request_json(f"{FUTURES_API}/time")
    if not isinstance(payload, dict) or "serverTime" not in payload:
        raise RuntimeError("Unexpected Binance Futures server-time response")
    return normalize_timestamp(payload["serverTime"])


def completed_end_exclusive_ms(server_time_ms: int, interval_ms: int) -> int:
    return server_time_ms - server_time_ms % interval_ms


def refresh_source(
    symbol: str,
    path: Path,
    *,
    interval: str = "5m",
    max_catchup_bars: int = 50_000,
    server_time_ms: int | None = None,
) -> RefreshResult:
    """Append every missing fully closed candle for one existing local source."""

    symbol = symbol.upper()
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported Binance kline interval: {interval}")
    if max_catchup_bars < 1:
        raise ValueError("max_catchup_bars must be positive")
    interval_ms = INTERVAL_MS[interval]
    before = source_snapshot(path)
    start_ms = before.last_open_time_ms + interval_ms
    if server_time_ms is None:
        server_time_ms = exchange_server_time_ms()
    else:
        server_time_ms = normalize_timestamp(server_time_ms)
    end_exclusive_ms = completed_end_exclusive_ms(server_time_ms, interval_ms)
    pending_bars = max(0, (end_exclusive_ms - start_ms) // interval_ms)
    if pending_bars > max_catchup_bars:
        raise RuntimeError(
            f"{symbol} is {pending_bars:,} candles behind, above the safe {max_catchup_bars:,}-bar refresh limit. "
            "Re-run the historical downloader to repair this source."
        )

    rows: list[list[str]] = []
    api_requests = 0
    if start_ms < end_exclusive_ms:
        raw_rows, api_requests = current_month_rows(symbol, interval, interval_ms, start_ms, end_exclusive_ms)
        rows = _validated_append_rows(
            raw_rows,
            start_ms=start_ms,
            end_exclusive_ms=end_exclusive_ms,
            interval_ms=interval_ms,
        )
        _append_rows_atomically(path, rows)

    after = source_snapshot(path)
    metadata_path = path.with_suffix(".refresh.json")
    result = RefreshResult(
        symbol=symbol,
        interval=interval,
        source_path=str(path),
        metadata_path=str(metadata_path),
        changed=bool(rows),
        rows_added=len(rows),
        api_requests=api_requests,
        server_time_utc=utc_iso(server_time_ms),
        completed_end_exclusive_utc=utc_iso(end_exclusive_ms),
        prior_last_closed_candle_open_time_utc=utc_iso(before.last_open_time_ms),
        last_closed_candle_open_time_utc=utc_iso(after.last_open_time_ms),
        source_mtime_ns=after.mtime_ns,
    )
    _atomic_write_json(
        metadata_path,
        {
            "market": "Binance USD-M Futures",
            "endpoint": f"{FUTURES_API}/klines",
            "refreshed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_before": before.as_payload(),
            "source_after": after.as_payload(),
            "result": result.as_payload(),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", choices=DEFAULT_SYMBOLS, help="Symbol to refresh; repeat for both")
    parser.add_argument("--interval", default="5m", choices=tuple(INTERVAL_MS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--max-catchup-bars", type=int, default=50_000)
    args = parser.parse_args()
    symbols = tuple(args.symbol or DEFAULT_SYMBOLS)
    server_time_ms = exchange_server_time_ms()
    for symbol in symbols:
        path = args.data_dir / f"{symbol}_{args.interval}_5y.csv"
        result = refresh_source(
            symbol,
            path,
            interval=args.interval,
            max_catchup_bars=args.max_catchup_bars,
            server_time_ms=server_time_ms,
        )
        print(json.dumps(result.as_payload(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
