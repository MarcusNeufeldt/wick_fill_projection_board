#!/usr/bin/env python3
"""Verify the dashboard exposes its non-browser live-refresh contract."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any


def get_json(url: str, *, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local verification endpoint only
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8793")
    parser.add_argument("--expect-enabled", action="store_true")
    parser.add_argument("--verify-projection", action="store_true", help="Also run the default pin through the dashboard API.")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    last_error: Exception | None = None
    for _ in range(20):
        try:
            health = get_json(f"{base}/api/health")
            refresh = get_json(f"{base}/api/refresh-status")
            break
        except Exception as error:  # Server startup can race this lightweight probe.
            last_error = error
            time.sleep(0.25)
    else:
        raise RuntimeError(f"Dashboard never became reachable: {last_error}")

    if health.get("ok") is not True:
        raise AssertionError("Health endpoint did not report ok")
    if bool(refresh.get("enabled")) != args.expect_enabled:
        raise AssertionError(f"Unexpected refresh enabled state: {refresh.get('enabled')}")
    if not refresh.get("data_version"):
        raise AssertionError("Refresh status has no data generation")
    sources = refresh.get("sources", {})
    for asset in ("ETHUSDT", "BTCUSDT"):
        source = sources.get(asset, {})
        if not source.get("available") or not source.get("last_closed_candle_open_time_utc"):
            raise AssertionError(f"Refresh status has no usable {asset} source")
    projection_as_of = None
    if args.verify_projection:
        projection = get_json(
            f"{base}/api/scenarios?asset=ETHUSDT&timeframe=5m&signal_time=2026-09-16T18%3A35%3A00Z",
            timeout=240,
        )
        projection_as_of = projection.get("current_state", {}).get("as_of_open_time_utc")
        if projection_as_of != sources["ETHUSDT"]["last_closed_candle_open_time_utc"]:
            raise AssertionError("Projection did not use the current ETH source generation")
        if [item.get("name") for item in projection.get("scenarios", [])] != ["fast", "normal", "extreme"]:
            raise AssertionError("Projection did not return the expected three route categories")
    print(
        json.dumps(
            {
                "ok": True,
                "enabled": refresh["enabled"],
                "in_progress": refresh["in_progress"],
                "next_refresh_utc": refresh["next_refresh_utc"],
                "data_version": refresh["data_version"],
                "eth_last_closed": sources["ETHUSDT"]["last_closed_candle_open_time_utc"],
                "btc_last_closed": sources["BTCUSDT"]["last_closed_candle_open_time_utc"],
                "projection_as_of": projection_as_of,
            }
        )
    )


if __name__ == "__main__":
    main()
