#!/usr/bin/env python3
"""Download an exact rolling window of Binance USD-M ETHUSDT 5-minute klines.

The script uses Binance Vision monthly archives for completed months and the
official USD-M Futures REST API for the current partial month.  It writes a
single, sorted CSV plus a metadata file that records the frozen UTC window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
FUTURES_API = "https://fapi.binance.com/fapi/v1"
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000
CSV_HEADER = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def utc_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def subtract_calendar_years(value: datetime, years: int) -> datetime:
    """Subtract whole calendar years while safely handling February 29."""
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, month=2, day=28)


def request_bytes(url: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "candle-projection-algo/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as error:  # pragma: no cover - depends on remote service
            last_error = error
            if attempt == attempts:
                break
            time.sleep(attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def request_json(url: str) -> object:
    return json.loads(request_bytes(url).decode("utf-8"))


def normalize_timestamp(value: object) -> int:
    timestamp = int(value)
    # Binance Vision currently documents microseconds for some datasets.  The
    # futures files used here are milliseconds, but accepting either prevents
    # a future timestamp-unit change from silently corrupting the output.
    return timestamp // 1000 if timestamp >= 100_000_000_000_000 else timestamp


def normalize_row(row: Iterable[object]) -> list[str] | None:
    values = [str(value) for value in row]
    if len(values) < len(CSV_HEADER):
        return None
    try:
        open_time = normalize_timestamp(values[0])
        close_time = normalize_timestamp(values[6])
    except ValueError:
        return None  # CSV header row
    values = values[: len(CSV_HEADER)]
    values[0] = str(open_time)
    values[6] = str(close_time)
    return values


def archive_rows(url: str) -> list[list[str]]:
    blob = request_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        csv_name = next((name for name in archive.namelist() if name.lower().endswith(".csv")), None)
        if csv_name is None:
            raise RuntimeError(f"Archive contains no CSV: {url}")
        with archive.open(csv_name) as raw_csv:
            reader = csv.reader(io.TextIOWrapper(raw_csv, encoding="utf-8", newline=""))
            return [normalized for row in reader if (normalized := normalize_row(row)) is not None]


def current_month_rows(start_ms: int, end_exclusive_ms: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    cursor = start_ms
    requests = 0
    while cursor < end_exclusive_ms:
        # Keep each request within Binance's 1,500-row limit and exclude the
        # in-progress candle by making endTime one millisecond before the
        # exclusive boundary.
        batch_end_exclusive = min(cursor + 1500 * INTERVAL_MS, end_exclusive_ms)
        parameters = urllib.parse.urlencode(
            {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": batch_end_exclusive - 1,
                "limit": 1500,
            }
        )
        payload = request_json(f"{FUTURES_API}/klines?{parameters}")
        requests += 1
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Binance returned no kline data beginning at {utc_iso(cursor)}: {payload!r}")
        batch = [normalized for row in payload if (normalized := normalize_row(row)) is not None]
        rows.extend(batch)
        next_cursor = normalize_timestamp(batch[-1][0]) + INTERVAL_MS
        if next_cursor <= cursor:
            raise RuntimeError("Binance REST pagination did not advance")
        cursor = next_cursor
    return rows, requests


def atomic_write_csv(destination: Path, rows: list[list[str]]) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    os.replace(temporary, destination)
    return digest


def atomic_write_json(destination: Path, payload: dict[str, object]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=3, help="Rolling calendar years to retrieve (default: 3)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for the combined CSV and metadata (default: ./data)",
    )
    args = parser.parse_args()
    if args.years < 1:
        parser.error("--years must be at least 1")

    server_time_payload = request_json(f"{FUTURES_API}/time")
    if not isinstance(server_time_payload, dict) or not isinstance(server_time_payload.get("serverTime"), int):
        raise RuntimeError(f"Unexpected Binance server time response: {server_time_payload!r}")
    server_time_ms = server_time_payload["serverTime"]
    end_exclusive_ms = server_time_ms - (server_time_ms % INTERVAL_MS)
    end_exclusive = datetime.fromtimestamp(end_exclusive_ms / 1000, tz=timezone.utc)
    start = subtract_calendar_years(end_exclusive, args.years)
    start_ms = int(start.timestamp() * 1000)

    current_month = month_start(end_exclusive)
    sources: list[str] = []
    collected: dict[int, list[str]] = {}
    month = month_start(start)
    while month < current_month:
        stamp = month.strftime("%Y-%m")
        url = f"{ARCHIVE_BASE}/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{stamp}.zip"
        sources.append(url)
        for row in archive_rows(url):
            open_time = int(row[0])
            if start_ms <= open_time < end_exclusive_ms:
                collected[open_time] = row
        month = next_month(month)

    api_start_ms = int(current_month.timestamp() * 1000)
    current_rows, api_requests = current_month_rows(api_start_ms, end_exclusive_ms)
    for row in current_rows:
        open_time = int(row[0])
        if start_ms <= open_time < end_exclusive_ms:
            collected[open_time] = row

    expected_rows = (end_exclusive_ms - start_ms) // INTERVAL_MS
    timestamps = sorted(collected)
    missing = []
    expected = start_ms
    for timestamp in timestamps:
        while expected < timestamp and len(missing) < 10:
            missing.append(expected)
            expected += INTERVAL_MS
        expected = timestamp + INTERVAL_MS
    while expected < end_exclusive_ms and len(missing) < 10:
        missing.append(expected)
        expected += INTERVAL_MS
    if len(timestamps) != expected_rows or missing:
        missing_text = ", ".join(utc_iso(value) for value in missing) or "none sampled"
        raise RuntimeError(
            f"Coverage validation failed: got {len(timestamps):,} rows; expected {expected_rows:,}; "
            f"first missing timestamps: {missing_text}"
        )
    if any((later - earlier) != INTERVAL_MS for earlier, later in zip(timestamps, timestamps[1:])):
        raise RuntimeError("Coverage validation failed: non-contiguous 5-minute timestamps")

    rows = [collected[timestamp] for timestamp in timestamps]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ETHUSDT_5m_3y.csv"
    metadata_path = args.output_dir / "ETHUSDT_5m_3y.metadata.json"
    sha256 = atomic_write_csv(csv_path, rows)
    metadata = {
        "symbol": SYMBOL,
        "market": "Binance USD-M Futures perpetual",
        "contract_type": "PERPETUAL",
        "margin_asset": "USDT",
        "interval": INTERVAL,
        "timestamp_unit": "milliseconds since Unix epoch",
        "timezone": "UTC",
        "generated_at_utc": utc_iso(int(time.time() * 1000)),
        "binance_server_time_utc": utc_iso(server_time_ms),
        "window": {
            "start_open_time_utc": utc_iso(start_ms),
            "end_exclusive_utc": utc_iso(end_exclusive_ms),
            "last_closed_candle_open_time_utc": utc_iso(end_exclusive_ms - INTERVAL_MS),
            "calendar_years": args.years,
        },
        "rows": len(rows),
        "expected_rows": expected_rows,
        "csv_sha256": sha256,
        "sources": {
            "monthly_archives": sources,
            "current_month_api": f"{FUTURES_API}/klines",
            "current_month_api_requests": api_requests,
        },
    }
    atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                "rows": len(rows),
                "start_utc": metadata["window"]["start_open_time_utc"],
                "end_exclusive_utc": metadata["window"]["end_exclusive_utc"],
                "sha256": sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
