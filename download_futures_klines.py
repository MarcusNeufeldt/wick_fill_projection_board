"""Download an exact rolling Binance USD-M perpetual kline window.

Completed months come from Binance Vision monthly archives. The current partial
month comes from the official USD-M Futures REST API so the output includes the
latest fully closed candle. The result is a sorted CSV plus metadata recording
the exact UTC window, source URLs, and checksum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
FUTURES_API = "https://fapi.binance.com/fapi/v1"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
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
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(value: datetime) -> datetime:
    return value.replace(year=value.year + 1, month=1) if value.month == 12 else value.replace(month=value.month + 1)


def subtract_calendar_years(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, month=2, day=28)


def request_bytes(url: str, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "candle-projection-algo/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # Network failures are retried deterministically.
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {error}")


def request_json(url: str) -> object:
    return json.loads(request_bytes(url).decode("utf-8"))


def normalize_timestamp(value: str | int) -> int:
    timestamp = int(value)
    while timestamp >= 10**14:  # Some Vision datasets use microseconds.
        timestamp //= 1000
    return timestamp


def archive_rows(url: str) -> list[list[str]]:
    blob = request_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not names:
            raise RuntimeError(f"Archive contains no CSV: {url}")
        with archive.open(names[0]) as source:
            rows = list(csv.reader(io.TextIOWrapper(source, encoding="utf-8")))
    if rows and rows[0] and rows[0][0].lower() == "open_time":
        rows.pop(0)
    return rows


def api_rows(symbol: str, interval: str, interval_ms: int, start_ms: int, end_exclusive_ms: int) -> tuple[list[list[str]], int]:
    """Read one exact candle range from the official USD-M Futures API."""
    rows: list[list[str]] = []
    cursor = start_ms
    requests = 0
    while cursor < end_exclusive_ms:
        parameters = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_exclusive_ms - 1,
                "limit": 1500,
            }
        )
        batch = request_json(f"{FUTURES_API}/klines?{parameters}")
        requests += 1
        if not isinstance(batch, list) or not batch:
            break
        rows.extend([[str(value) for value in row] for row in batch])
        next_cursor = normalize_timestamp(batch[-1][0]) + interval_ms
        if next_cursor <= cursor:
            raise RuntimeError("Futures API did not advance the candle cursor")
        cursor = next_cursor
    return rows, requests


def current_month_rows(symbol: str, interval: str, interval_ms: int, start_ms: int, end_exclusive_ms: int) -> tuple[list[list[str]], int]:
    """Backward-compatible name for callers that previously requested only the current month."""
    return api_rows(symbol, interval, interval_ms, start_ms, end_exclusive_ms)


def missing_candle_ranges(
    timestamps: list[int], start_ms: int, end_exclusive_ms: int, interval_ms: int
) -> list[tuple[int, int]]:
    """Return contiguous missing [start, end) ranges without masking duplicate timestamps."""
    cursor = start_ms
    ranges: list[tuple[int, int]] = []
    for timestamp in sorted(set(timestamps)):
        if timestamp < start_ms or timestamp >= end_exclusive_ms:
            continue
        if timestamp > cursor:
            ranges.append((cursor, timestamp))
        cursor = max(cursor, timestamp + interval_ms)
    if cursor < end_exclusive_ms:
        ranges.append((cursor, end_exclusive_ms))
    return ranges


def validate_contract(symbol: str) -> dict[str, object]:
    payload = request_json(f"{FUTURES_API}/exchangeInfo")
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected exchangeInfo response")
    contract = next((item for item in payload.get("symbols", []) if item.get("symbol") == symbol), None)
    if contract is None:
        raise RuntimeError(f"No USD-M futures contract named {symbol}")
    if contract.get("contractType") != "PERPETUAL" or contract.get("marginAsset") != "USDT":
        raise RuntimeError(f"{symbol} is not the expected USDT-margined perpetual contract")
    return {
        "symbol": contract.get("symbol"),
        "status": contract.get("status"),
        "contract_type": contract.get("contractType"),
        "base_asset": contract.get("baseAsset"),
        "quote_asset": contract.get("quoteAsset"),
        "margin_asset": contract.get("marginAsset"),
    }


def atomic_write_csv(path: Path, rows: list[list[str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="USD-M perpetual symbol, for example BTCUSDT")
    parser.add_argument("--interval", default="5m", choices=tuple(INTERVAL_MS))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if args.years < 1:
        parser.error("--years must be at least one")
    interval_ms = INTERVAL_MS[args.interval]
    contract = validate_contract(symbol)
    print(
        json.dumps(
            {
                "stage": "validated_contract",
                "symbol": symbol,
                "interval": args.interval,
                "contract_type": contract.get("contractType", contract.get("contract_type")),
            }
        ),
        flush=True,
    )
    server_time = request_json(f"{FUTURES_API}/time")
    if not isinstance(server_time, dict) or "serverTime" not in server_time:
        raise RuntimeError("Unexpected futures time response")
    end_exclusive_ms = normalize_timestamp(server_time["serverTime"])
    end_exclusive_ms -= end_exclusive_ms % interval_ms
    end_exclusive = datetime.fromtimestamp(end_exclusive_ms / 1000, tz=timezone.utc)
    start = subtract_calendar_years(end_exclusive, args.years)
    start_ms = int(start.timestamp() * 1000)

    current_month = month_start(end_exclusive)
    sources: list[str] = []
    selected_rows: list[list[str]] = []
    month = month_start(start)
    while month < current_month:
        stamp = month.strftime("%Y-%m")
        url = f"{ARCHIVE_BASE}/{symbol}/{args.interval}/{symbol}-{args.interval}-{stamp}.zip"
        sources.append(url)
        print(json.dumps({"stage": "download_month", "month": stamp}), flush=True)
        for row in archive_rows(url):
            if len(row) < len(CSV_HEADER):
                raise RuntimeError(f"Malformed archive row from {url}")
            open_time = normalize_timestamp(row[0])
            if start_ms <= open_time < end_exclusive_ms:
                normalized = list(row[: len(CSV_HEADER)])
                normalized[0] = str(open_time)
                normalized[6] = str(normalize_timestamp(normalized[6]))
                selected_rows.append(normalized)
        month = next_month(month)

    print(json.dumps({"stage": "download_current_month", "month": current_month.strftime("%Y-%m")}), flush=True)
    current_rows, current_month_api_requests = api_rows(
        symbol,
        args.interval,
        interval_ms,
        int(current_month.timestamp() * 1000),
        end_exclusive_ms,
    )
    for row in current_rows:
        if len(row) < len(CSV_HEADER):
            raise RuntimeError("Malformed current-month API row")
        open_time = normalize_timestamp(row[0])
        if start_ms <= open_time < end_exclusive_ms:
            normalized = list(row[: len(CSV_HEADER)])
            normalized[0] = str(open_time)
            normalized[6] = str(normalize_timestamp(normalized[6]))
            selected_rows.append(normalized)

    # Vision archives occasionally omit historical candles even when the official
    # Futures API still serves them. Repair only the detected gaps and retain a
    # precise record in metadata; the exact-row and contiguous-time checks below
    # remain the final contract.
    archive_gap_repairs: list[dict[str, object]] = []
    archive_gap_api_requests = 0
    for gap_start_ms, gap_end_exclusive_ms in missing_candle_ranges(
        [int(row[0]) for row in selected_rows], start_ms, end_exclusive_ms, interval_ms
    ):
        gap_rows, gap_api_requests = api_rows(
            symbol, args.interval, interval_ms, gap_start_ms, gap_end_exclusive_ms
        )
        archive_gap_api_requests += gap_api_requests
        repaired_rows = 0
        for row in gap_rows:
            if len(row) < len(CSV_HEADER):
                raise RuntimeError("Malformed Futures API row while repairing an archive gap")
            open_time = normalize_timestamp(row[0])
            if gap_start_ms <= open_time < gap_end_exclusive_ms:
                normalized = list(row[: len(CSV_HEADER)])
                normalized[0] = str(open_time)
                normalized[6] = str(normalize_timestamp(normalized[6]))
                selected_rows.append(normalized)
                repaired_rows += 1
        archive_gap_repairs.append(
            {
                "start_open_time_utc": utc_iso(gap_start_ms),
                "end_exclusive_utc": utc_iso(gap_end_exclusive_ms),
                "expected_rows": (gap_end_exclusive_ms - gap_start_ms) // interval_ms,
                "api_rows_returned": repaired_rows,
                "api_requests": gap_api_requests,
            }
        )

    selected_rows.sort(key=lambda row: int(row[0]))
    expected_rows = (end_exclusive_ms - start_ms) // interval_ms
    if len(selected_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} candles but collected {len(selected_rows)}")
    timestamps = [int(row[0]) for row in selected_rows]
    if len(timestamps) != len(set(timestamps)):
        raise RuntimeError("Duplicate candle timestamps in combined data")
    if any(later - earlier != interval_ms for earlier, later in zip(timestamps, timestamps[1:])):
        raise RuntimeError("Timestamp gap in combined data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{args.interval}_{args.years}y"
    csv_path = args.output_dir / f"{stem}.csv"
    metadata_path = args.output_dir / f"{stem}.metadata.json"
    atomic_write_csv(csv_path, selected_rows)
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "market": "Binance USD-M Futures",
        "contract": contract,
        "symbol": symbol,
        "interval": args.interval,
        "interval_ms": interval_ms,
        "window": {
            "start_open_time_utc": utc_iso(start_ms),
            "end_exclusive_utc": utc_iso(end_exclusive_ms),
            "last_closed_candle_open_time_utc": utc_iso(end_exclusive_ms - interval_ms),
        },
        "row_count": len(selected_rows),
        "sha256": checksum,
        "monthly_archives": sources,
        "current_month_api": f"{FUTURES_API}/klines",
        "current_month_api_requests": current_month_api_requests,
        "archive_gap_api_requests": archive_gap_api_requests,
        "archive_gap_repairs": archive_gap_repairs,
        "generated_at_utc": utc_iso(int(datetime.now(timezone.utc).timestamp() * 1000)),
    }
    atomic_write_json(metadata_path, metadata)
    print(json.dumps({"csv": str(csv_path), "metadata": str(metadata_path), "rows": len(selected_rows), "window": metadata["window"]}))


if __name__ == "__main__":
    main()
