#!/usr/bin/env python3
"""Verify the dashboard exposes its non-browser live-refresh contract."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from conditional_wick_assets import (
    SUPPORTED_ASSETS,
    SUPPORTED_DASHBOARD_TIMEFRAMES,
    assets_for_timeframe,
)


def get_json(url: str, *, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local verification endpoint only
        return json.loads(response.read().decode("utf-8"))


def expected_projection_as_of(last_closed_open_time_utc: str | None, timeframe: str) -> str | None:
    """Map a source generation to the latest complete selected-timeframe candle."""
    if not last_closed_open_time_utc:
        return None
    if timeframe in {"1m", "5m"}:
        return last_closed_open_time_utc
    parsed = datetime.fromisoformat(last_closed_open_time_utc.replace("Z", "+00:00"))
    open_seconds = int(parsed.timestamp())
    aligned_seconds = open_seconds - open_seconds % (15 * 60)
    return datetime.fromtimestamp(aligned_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def source_key(asset: str, timeframe: str) -> str:
    return f"{asset}_1m" if timeframe == "1m" else asset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8793")
    parser.add_argument("--expect-enabled", action="store_true")
    parser.add_argument(
        "--expect-1m",
        "--expect-sol-1m",
        dest="expect_one_minute",
        action="store_true",
        help="Require a usable live 1m source for every configured asset.",
    )
    parser.add_argument("--verify-projection", action="store_true", help="Also run the default pin through the dashboard API.")
    parser.add_argument("--projection-asset", default="ETHUSDT", choices=SUPPORTED_ASSETS)
    parser.add_argument("--projection-timeframe", default="5m", choices=SUPPORTED_DASHBOARD_TIMEFRAMES)
    parser.add_argument("--projection-signal-time", default="2026-09-16T18:35:00Z")
    args = parser.parse_args()
    if args.projection_asset not in assets_for_timeframe(args.projection_timeframe):
        parser.error(f"{args.projection_timeframe} projection verification does not support {args.projection_asset}")
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
    for asset in SUPPORTED_ASSETS:
        source = sources.get(asset, {})
        if not source.get("available") or not source.get("last_closed_candle_open_time_utc"):
            raise AssertionError(f"Refresh status has no usable {asset} source")
    if args.expect_one_minute or args.projection_timeframe == "1m":
        for asset in SUPPORTED_ASSETS:
            one_minute = sources.get(source_key(asset, "1m"), {})
            if not one_minute.get("available") or not one_minute.get("last_closed_candle_open_time_utc"):
                raise AssertionError(f"Refresh status has no usable {asset} 1m source")
    projection_as_of = None
    projection_elapsed_seconds = None
    if args.verify_projection:
        projection_started = time.monotonic()
        signal_time = urllib.parse.quote(args.projection_signal_time, safe="")
        projection = get_json(
            f"{base}/api/scenarios?asset={args.projection_asset}&timeframe={args.projection_timeframe}&signal_time={signal_time}",
            timeout=240,
        )
        projection_elapsed_seconds = round(time.monotonic() - projection_started, 3)
        projection_as_of = projection.get("current_state", {}).get("as_of_open_time_utc")
        # A five-minute refresh can atomically replace the source while a slower
        # projection is in flight. Re-run once against that new generation
        # instead of falsely comparing the valid older response to newer data.
        refresh_after = get_json(f"{base}/api/refresh-status")
        sources_after = refresh_after.get("sources", {})
        projection_source_key = source_key(args.projection_asset, args.projection_timeframe)
        source_last_after = sources_after.get(projection_source_key, {}).get("last_closed_candle_open_time_utc")
        expected_as_of = expected_projection_as_of(source_last_after, args.projection_timeframe)
        source_advanced = source_last_after != sources[projection_source_key]["last_closed_candle_open_time_utc"]
        if projection_as_of != expected_as_of and source_advanced:
            retry_started = time.monotonic()
            projection = get_json(
                f"{base}/api/scenarios?asset={args.projection_asset}&timeframe={args.projection_timeframe}&signal_time={signal_time}",
                timeout=240,
            )
            projection_elapsed_seconds = round(projection_elapsed_seconds + time.monotonic() - retry_started, 3)
            projection_as_of = projection.get("current_state", {}).get("as_of_open_time_utc")
            refresh_after = get_json(f"{base}/api/refresh-status")
            sources_after = refresh_after.get("sources", {})
            expected_as_of = expected_projection_as_of(
                sources_after.get(projection_source_key, {}).get("last_closed_candle_open_time_utc"),
                args.projection_timeframe,
            )
        refresh = refresh_after
        sources = sources_after
        if projection_as_of != expected_as_of:
            raise AssertionError(f"Projection did not use the current {args.projection_asset} source generation")
        if [item.get("name") for item in projection.get("scenarios", [])] != ["fast", "normal", "extreme"]:
            raise AssertionError("Projection did not return the expected three route categories")
        if projection.get("schema_version") != "1.1.0":
            raise AssertionError("Projection did not return the corrected V1 schema")
        if projection.get("library", {}).get("availability_cutoff_close_utc") != projection.get(
            "current_state", {}
        ).get("as_of_close_time_utc"):
            raise AssertionError("Projection availability cutoff does not match its observation snapshot close")
        for scenario in projection["scenarios"]:
            for field in (
                "historical_future_max_away_move_pct",
                "projected_future_max_away_move_pct",
                "projected_additional_adverse_move_pct",
                "joint_risk_score",
                "joint_risk_score_percentile",
            ):
                if field not in scenario:
                    raise AssertionError(f"Projection scenario is missing corrected field: {field}")
        v2 = projection.get("v2_risk", {})
        if v2.get("available"):
            if v2.get("age_support", {}).get("status") not in {
                "exact_sampled_age",
                "between_sampled_ages",
                "outside_sampled_age",
                "unknown",
            }:
                raise AssertionError("V2 diagnostic has no valid snapshot-age support status")
            if "artifact" not in v2:
                raise AssertionError("V2 diagnostic has no static artifact metadata")
    print(
        json.dumps(
            {
                "ok": True,
                "enabled": refresh["enabled"],
                "in_progress": refresh["in_progress"],
                "next_refresh_utc": refresh["next_refresh_utc"],
                "data_version": refresh["data_version"],
                "source_last_closed": {
                    asset: sources[asset]["last_closed_candle_open_time_utc"] for asset in SUPPORTED_ASSETS
                },
                "one_minute_last_closed": {
                    asset: sources.get(source_key(asset, "1m"), {}).get("last_closed_candle_open_time_utc")
                    for asset in SUPPORTED_ASSETS
                },
                "projection_asset": args.projection_asset if args.verify_projection else None,
                "projection_timeframe": args.projection_timeframe if args.verify_projection else None,
                "projection_as_of": projection_as_of,
                "projection_elapsed_seconds": projection_elapsed_seconds,
                "projection_matching_backend": projection.get("library", {}).get("matching_backend")
                if args.verify_projection
                else None,
                "projection_performance": projection.get("performance") if args.verify_projection else None,
            }
        )
    )


if __name__ == "__main__":
    main()
