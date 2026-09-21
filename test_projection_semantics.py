"""Small deterministic regressions for the corrected conditional-path contract."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from build_conditional_path_library import FEATURE_COLUMNS, detect_strict_signals, trace_clean_path
from build_conditional_path_scenarios import (
    eligible_episodes_at_snapshot,
    has_confirmed_departure,
    prepare_path_states,
    project_at,
    projected_candles,
    projected_coordinate_matches,
    read_library_paths,
    projected_path_metrics,
    sampled_matching_states,
    select_scenarios,
)
from build_sol_one_minute_path_library import write_compact_paths
from conditional_wick_assets import (
    ASSET_CATEGORICAL_FEATURE_COLUMNS,
    SOL_ONE_MINUTE_ASSET,
    SUPPORTED_ASSETS,
    asset_indicator_column,
    assets_for_timeframe,
)
from download_futures_klines import missing_candle_ranges
from native_wick_matcher import NativeStateIndex, load_runtime_cache, write_runtime_cache
from refresh_futures_klines import RefreshResult, _replace_with_retry
from serve_conditional_wick_dashboard import (
    HTML,
    Engine,
    aggregate_visual_candles,
    appended_candle_rows,
    display_timeframe_minutes,
    episode_row_spans,
    incremental_strict_signals,
    snapshot_age_support,
    v2_artifact_metadata,
    validate_asset_timeframe,
)
from train_conditional_wick_v2 import (
    FUTURE_AWAY_TARGET_SEMANTICS,
    MODEL_SCHEMA_VERSION,
    OBSERVABLE_STATE_INPUT_COLUMNS,
    observable_state_frame,
    observable_state_schema,
)
from verify_live_refresh_service import expected_projection_as_of


def path_rows(episode_id: str, values: list[tuple[float, float, float, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, (opened, high, low, closed) in enumerate(values):
        rows.append(
            {
                "episode_id": episode_id,
                "asset": "ETHUSDT",
                "timeframe": "5m",
                "direction": "lower_wick",
                "direction_sign": 1,
                "offset_bars": offset,
                "normalized_open_pct": opened,
                "normalized_high_pct": high,
                "normalized_low_pct": low,
                "normalized_close_pct": closed,
            }
        )
    return rows


def event_row(episode_id: str, fill_close_time_ms: int, fill_bars: int = 3) -> dict[str, object]:
    row: dict[str, object] = {
        "episode_id": episode_id,
        "asset": "ETHUSDT",
        "timeframe": "5m",
        "direction": "lower_wick",
        "direction_sign": 1,
        "interval_minutes": 5,
        "signal_open_time_ms": 100_000,
        "fill_close_time_ms": fill_close_time_ms,
        "signal_to_departure_bars": 1,
        "signal_to_fill_bars": fill_bars,
        "wick_target": 100.0,
    }
    for feature in FEATURE_COLUMNS:
        row[feature] = 0.5
    return row


class ProjectionSemanticsTests(unittest.TestCase):
    def test_live_refresh_preserves_the_selected_route_and_chart_viewport(self) -> None:
        self.assertIn("function captureChartViewport()", HTML)
        self.assertIn("function restoreChartViewport(viewport)", HTML)
        self.assertIn(
            "const preferredScenario=data.scenarios.some(item=>item.name===selectedScenario)?selectedScenario:'normal';",
            HTML,
        )
        self.assertIn("selectScenario(preferredScenario,{viewport});", HTML)
        self.assertIn("render(payload,{preserveViewport:sourceRefresh});", HTML)
        self.assertIn("if(!restoreChartViewport(viewport))fitChartToFullRange(display.barCount);", HTML)

    def test_browser_uses_the_current_binance_futures_kline_stream(self) -> None:
        self.assertIn(
            "wss://fstream.binance.com/market/ws/${asset.toLowerCase()}@kline_${timeframe}",
            HTML,
        )
        self.assertIn("liveCandleSeries.update(liveCandle.candle);", HTML)
        self.assertIn("if(kline.x)scheduleClosedCandleRefresh();", HTML)
        self.assertIn("window.addEventListener('beforeunload',stopLiveFeed);", HTML)
        live_section = HTML[HTML.index("function clearLiveCandle()"):HTML.index("function aggregateVisualCandles")]
        self.assertNotIn("request(", live_section)

    def test_server_visual_aggregation_preserves_ohlc_and_transition(self) -> None:
        actual = [
            {"time": 60, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0},
            {"time": 120, "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0},
        ]
        projected = [
            {"time": 120, "open": 12.0, "high": 14.0, "low": 11.0, "close": 13.0},
            {"time": 180, "open": 13.0, "high": 15.0, "low": 12.0, "close": 14.0},
        ]

        candles, has_transition = aggregate_visual_candles(actual, projected, 5)

        self.assertTrue(has_transition)
        self.assertEqual(candles, [{"time": 0, "open": 10.0, "high": 15.0, "low": 9.0, "close": 14.0, "phase": "transition"}])

    def test_episode_row_spans_preserve_projected_candles(self) -> None:
        paths = pd.DataFrame(
            [
                {"episode_id": "a", "offset_bars": 0, "normalized_open_pct": 1.0, "normalized_high_pct": 1.5, "normalized_low_pct": 0.5, "normalized_close_pct": 1.0},
                {"episode_id": "a", "offset_bars": 1, "normalized_open_pct": 0.8, "normalized_high_pct": 1.2, "normalized_low_pct": 0.2, "normalized_close_pct": 0.4},
                {"episode_id": "b", "offset_bars": 0, "normalized_open_pct": 2.0, "normalized_high_pct": 2.5, "normalized_low_pct": 1.5, "normalized_close_pct": 2.0},
                {"episode_id": "b", "offset_bars": 1, "normalized_open_pct": 1.0, "normalized_high_pct": 1.5, "normalized_low_pct": 0.5, "normalized_close_pct": 0.8},
            ]
        )
        scenario = {"episode_id": "a", "alignment_offset_bars": 0, "remaining_to_fill_bars": 1}
        fallback = projected_candles(paths, scenario, 100.0, 1, 1.0, 0, 1)
        indexed = projected_candles(paths, scenario, 100.0, 1, 1.0, 0, 1, episode_row_spans(paths))

        self.assertEqual(indexed, fallback)

    def test_verified_small_source_append_reuses_the_cached_one_minute_frame(self) -> None:
        columns = [
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
        rows = []
        for index in range(3):
            open_time = index * 60_000
            rows.append(
                [open_time, 100.0, 101.0, 99.0, 100.5, 10.0, open_time + 59_999, 0.0, 1, 0.0, 0.0, 0]
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "SOLUSDT_1m_5y.csv"
            source.parent.mkdir(parents=True)
            full = pd.DataFrame(rows, columns=columns)
            full.to_csv(source, index=False)
            engine = Engine(root)
            previous = full.iloc[:2, :7].copy()
            engine.frame_cache[("SOLUSDT", "1m", 1)] = previous
            result = RefreshResult(
                symbol="SOLUSDT",
                interval="1m",
                source_path=str(source),
                metadata_path=str(source.with_suffix(".refresh.json")),
                changed=True,
                rows_added=1,
                api_requests=1,
                server_time_utc="2026-01-01T00:00:00Z",
                completed_end_exclusive_utc="2026-01-01T00:03:00Z",
                prior_last_closed_candle_open_time_utc="2026-01-01T00:01:00Z",
                last_closed_candle_open_time_utc="2026-01-01T00:02:00Z",
                source_mtime_ns=source.stat().st_mtime_ns,
            )

            self.assertTrue(engine.apply_source_append("SOLUSDT", "1m", result))
            updated = engine._frame("SOLUSDT", "1m")
            self.assertEqual(appended_candle_rows(source, 1, 1)["open_time"].tolist(), [120_000])

        self.assertEqual(updated["open_time"].tolist(), [0, 60_000, 120_000])

    def test_display_timeframe_cannot_invent_finer_candles(self) -> None:
        self.assertEqual(display_timeframe_minutes("15m", "5m"), ("15m", 15))
        with self.assertRaisesRegex(ValueError, "smaller"):
            display_timeframe_minutes("1m", "5m")

    def test_incremental_signal_detection_resumes_from_the_cached_generation(self) -> None:
        frame = pd.DataFrame({"open_time": np.arange(200, dtype=np.int64) * 60_000})
        previous = pd.DataFrame(
            [
                {"open_time": 10 * 60_000, "bar_index": 10, "marker": "preserved"},
                {"open_time": 150 * 60_000, "bar_index": 150, "marker": "old-tail"},
            ]
        )
        refreshed_tail = pd.DataFrame(
            [{"open_time": 150 * 60_000, "bar_index": 74, "marker": "new-tail"}]
        )

        with patch("serve_conditional_wick_dashboard.detect_strict_signals", return_value=refreshed_tail) as detector:
            result = incremental_strict_signals(frame, previous, "SOLUSDT", "1m", previous_row_count=140)

        detector.assert_called_once()
        self.assertEqual(int(detector.call_args.args[0]["open_time"].iat[0]), 76 * 60_000)
        self.assertEqual(result[["bar_index", "marker"]].to_dict("records"), [{"bar_index": 10, "marker": "preserved"}, {"bar_index": 150, "marker": "new-tail"}])

    def test_incremental_signal_detection_keeps_signals_created_during_a_delayed_refresh(self) -> None:
        rows = [
            {
                "open_time": index * 60_000,
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 10.0,
                "close_time": (index + 1) * 60_000 - 1,
            }
            for index in range(300)
        ]
        rows[130].update({"high": 101.0, "low": 99.9})
        frame = pd.DataFrame(rows)
        previous = detect_strict_signals(frame.iloc[:100].copy(), "SOLUSDT", "1m")

        full = detect_strict_signals(frame, "SOLUSDT", "1m")
        incremental = incremental_strict_signals(
            frame, previous, "SOLUSDT", "1m", previous_row_count=100
        )

        self.assertEqual(full["bar_index"].tolist(), [130])
        self.assertEqual(incremental["bar_index"].tolist(), [130])

    def test_snapshot_close_eligibility_includes_boundary(self) -> None:
        events = pd.DataFrame(
            [
                {"episode_id": "before", "fill_close_time_ms": 1_000},
                {"episode_id": "boundary", "fill_close_time_ms": 1_300},
                {"episode_id": "after", "fill_close_time_ms": 1_600},
            ]
        )
        eligible = eligible_episodes_at_snapshot(events, 1_300)
        self.assertEqual(eligible["episode_id"].tolist(), ["before", "boundary"])

    def test_post_departure_state_gate_and_inclusive_departure(self) -> None:
        events = pd.DataFrame([event_row("one", 1_000, fill_bars=5) | {"signal_to_departure_bars": 3}])
        paths = pd.DataFrame(
            path_rows(
                "one",
                [
                    (0.1, 2.0, 0.0, 0.2),
                    (0.4, 2.5, 0.1, 0.6),
                    (0.7, 3.0, 0.2, 1.0),
                    (1.5, 4.0, 0.4, 2.0),
                    (1.0, 5.0, 0.1, 1.5),
                    (0.2, 7.0, -0.1, 0.0),
                ],
            )
        )
        states = prepare_path_states(events, paths)
        candidate_offsets = states.loc[states["candidate_after_departure"], "offset_bars"].tolist()
        self.assertEqual(candidate_offsets, [3, 4])
        self.assertTrue(has_confirmed_departure(pd.DataFrame({"close": [110.0]}), 1, 110.0))
        self.assertTrue(has_confirmed_departure(pd.DataFrame({"close": [90.0]}), -1, 90.0))

    def test_terminal_fill_candle_is_in_canonical_displayed_risk(self) -> None:
        paths = pd.DataFrame(
            path_rows(
                "one",
                [
                    (0.1, 2.0, 0.0, 0.2),
                    (1.8, 2.2, 1.2, 2.0),
                    (3.0, 4.0, 1.0, 2.5),
                    (0.5, 7.0, -0.2, 0.0),
                ],
            )
        )
        scenario = {"episode_id": "one", "alignment_offset_bars": 1, "remaining_to_fill_bars": 2}
        candles, scale = projected_candles(
            paths,
            scenario,
            current_target=100.0,
            current_direction_sign=1,
            current_move_pct=6.0,
            projection_start_open_time_ms=2_000_000,
            interval_minutes=5,
        )
        self.assertAlmostEqual(scale, 3.0)
        metrics = projected_path_metrics(candles, 100.0, 1, 6.0)
        self.assertAlmostEqual(metrics["projected_future_max_away_move_pct"], 21.0, places=5)
        self.assertAlmostEqual(metrics["projected_additional_adverse_move_pct"], 15.0, places=5)

    def test_projected_coordinates_reverse_the_historical_risk_order(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(12):
            row: dict[str, object] = {
                "episode_id": f"episode-{index}",
                "asset": "ETHUSDT",
                "timeframe": "5m",
                "direction": "lower_wick",
                "offset_bars": 1,
                "remaining_to_fill_bars": index + 2,
                "alignment_current_move_pct": 2.0,
                "alignment_peak_move_pct": 3.0,
                "alignment_drawdown_pct": 1.0,
                "future_peak_move_pct": float(index + 2),
                "match_score": float(index),
            }
            rows.append(row)
        rows[0]["episode_id"] = "A"
        rows[0]["alignment_current_move_pct"] = 1.0
        rows[0]["future_peak_move_pct"] = 4.0
        rows[1]["episode_id"] = "B"
        rows[1]["alignment_current_move_pct"] = 3.0
        rows[1]["future_peak_move_pct"] = 6.0
        corrected = projected_coordinate_matches(pd.DataFrame(rows), current_move_pct=3.0)
        by_id = corrected.set_index("episode_id")
        self.assertLess(
            float(by_id.loc["B", "projected_future_max_away_move_pct"]),
            float(by_id.loc["A", "projected_future_max_away_move_pct"]),
        )
        scenarios = select_scenarios(corrected, top_k=12)
        self.assertTrue(
            all("projected_future_max_away_move_pct" in scenario for scenario in scenarios)
        )
        self.assertTrue(all("joint_risk_score_percentile" in scenario for scenario in scenarios))
        self.assertTrue(all("joint_risk_percentile" not in scenario for scenario in scenarios))
        self.assertTrue(all("future_max_away_move_pct" not in scenario for scenario in scenarios))

    def test_project_at_uses_snapshot_availability_for_its_only_candidate_pool(self) -> None:
        events = pd.DataFrame(
            [event_row(f"eligible-{index}", 1_300, fill_bars=3) for index in range(12)]
            + [event_row("future", 1_600, fill_bars=3)]
        )
        values = [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (1.0, 4.0, 0.2, 1.5), (0.2, 5.0, -0.1, 0.0)]
        paths = pd.DataFrame(
            [row for episode_id in events["episode_id"] for row in path_rows(str(episode_id), values)]
        )
        target = pd.Series(event_row("target", 9_999, fill_bars=3))
        arguments = {
            "current_state": {
                "elapsed_bars": 1.0,
                "current_move_pct": 2.0,
                "peak_move_pct": 3.0,
                "drawdown_from_peak_pct": 1.0,
            },
            "snapshot_close_time_ms": 1_300,
            "current_target": 100.0,
            "current_direction_sign": 1,
            "projection_start_open_time_ms": 1_300,
            "interval_minutes": 5,
            "top_k": 12,
        }
        result = project_at(events, paths, target, **arguments)
        self.assertEqual(len(result["eligible_events"]), 12)
        self.assertNotIn("future", set(result["eligible_events"]["episode_id"]))
        self.assertEqual(len(result["scenarios"]), 3)
        self.assertFalse(result["insufficient_matches"])

        precomputed = prepare_path_states(events, paths)
        with_precomputed = project_at(events, paths, target, path_states=precomputed, **arguments)
        with_cached_render_states = project_at(
            events, precomputed, target, path_states=precomputed, **arguments
        )
        self.assertEqual(
            result["matched_states"][["episode_id", "offset_bars"]].to_dict("records"),
            with_precomputed["matched_states"][["episode_id", "offset_bars"]].to_dict("records"),
        )
        self.assertEqual(
            [
                (item["name"], item["episode_id"], item["remaining_to_fill_bars"])
                for item in result["scenarios"]
            ],
            [
                (item["name"], item["episode_id"], item["remaining_to_fill_bars"])
                for item in with_precomputed["scenarios"]
            ],
        )
        self.assertEqual(
            [
                (item["name"], item["episode_id"], item["remaining_to_fill_bars"])
                for item in result["scenarios"]
            ],
            [
                (item["name"], item["episode_id"], item["remaining_to_fill_bars"])
                for item in with_cached_render_states["scenarios"]
            ],
        )

    def test_native_matcher_preserves_python_route_selection(self) -> None:
        events = pd.DataFrame(
            [event_row(f"eligible-{index:02d}", 1_300, fill_bars=3) for index in range(12)]
            + [event_row("future", 1_600, fill_bars=3)]
        )
        values = [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (1.0, 4.0, 0.2, 1.5), (0.2, 5.0, -0.1, 0.0)]
        paths = pd.DataFrame(
            [row for episode_id in events["episode_id"] for row in path_rows(str(episode_id), values)]
        )
        target = pd.Series(event_row("target", 9_999, fill_bars=3))
        arguments = {
            "current_state": {
                "elapsed_bars": 1.0,
                "current_move_pct": 2.0,
                "peak_move_pct": 3.0,
                "drawdown_from_peak_pct": 1.0,
            },
            "snapshot_close_time_ms": 1_300,
            "current_target": 100.0,
            "current_direction_sign": 1,
            "projection_start_open_time_ms": 1_300,
            "interval_minutes": 5,
            "top_k": 12,
        }
        states = prepare_path_states(events, paths)
        reference = project_at(events, states, target, path_states=states, **arguments)
        native_index = NativeStateIndex.from_frames(events, states, "5m")
        accelerated = project_at(
            events,
            states,
            target,
            path_states=states,
            native_state_index=native_index,
            **arguments,
        )
        self.assertEqual(
            reference["matched_states"][["episode_id", "offset_bars"]].to_dict("records"),
            accelerated["matched_states"][["episode_id", "offset_bars"]].to_dict("records"),
        )
        self.assertEqual(
            [(item["name"], item["episode_id"]) for item in reference["scenarios"]],
            [(item["name"], item["episode_id"]) for item in accelerated["scenarios"]],
        )

    def test_native_runtime_cache_round_trips_state_rows_and_arrays(self) -> None:
        events = pd.DataFrame([event_row(f"episode-{index:02d}", 1_300, fill_bars=3) for index in range(12)])
        paths = pd.DataFrame(
            [
                row
                for episode_id in events["episode_id"]
                for row in path_rows(
                    str(episode_id),
                    [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (1.0, 4.0, 0.2, 1.5), (0.2, 5.0, -0.1, 0.0)],
                )
            ]
        )
        states = prepare_path_states(events, paths).reset_index(drop=True)
        index = NativeStateIndex.from_frames(events, states, "5m")
        with TemporaryDirectory() as directory:
            library = Path(directory)
            (library / "paths").mkdir()
            (library / "episodes.csv").touch()
            for asset in SUPPORTED_ASSETS:
                (library / "paths" / f"{asset}_5m_paths.csv.gz").touch()
            write_runtime_cache(library, "5m", states, index)
            loaded = load_runtime_cache(library, "5m")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            cached_states, cached_index = loaded
            self.assertEqual(len(cached_states), len(states))
            self.assertEqual(cached_index.episode_ids.astype(str).tolist(), index.episode_ids.astype(str).tolist())
            self.assertEqual(cached_index.state_rows.tolist(), index.state_rows.tolist())
            cached_index.close()

    def test_one_minute_runtime_cache_fingerprints_only_its_isolated_path(self) -> None:
        events = pd.DataFrame([event_row(f"episode-{index:02d}", 1_300, fill_bars=3) for index in range(12)])
        paths = pd.DataFrame(
            [
                row
                for episode_id in events["episode_id"]
                for row in path_rows(
                    str(episode_id),
                    [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (0.2, 5.0, -0.1, 0.0)],
                )
            ]
        )
        states = prepare_path_states(events, paths).reset_index(drop=True)
        index = NativeStateIndex.from_frames(events, states, "1m")
        with TemporaryDirectory() as directory:
            library = Path(directory)
            (library / "paths").mkdir()
            (library / "episodes.csv").touch()
            (library / "paths" / "ETHUSDT_1m_paths.csv.gz").touch()
            write_runtime_cache(library, "1m", states, index)
            loaded = load_runtime_cache(library, "1m")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            _, cached_index = loaded
            self.assertEqual(cached_index.episode_ids.astype(str).tolist(), index.episode_ids.astype(str).tolist())
            cached_index.close()

    def test_project_at_returns_insufficient_state_without_selecting_scenarios(self) -> None:
        events = pd.DataFrame([event_row(f"eligible-{index}", 1_300, fill_bars=2) for index in range(11)])
        values = [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (0.2, 5.0, -0.1, 0.0)]
        paths = pd.DataFrame(
            [row for episode_id in events["episode_id"] for row in path_rows(str(episode_id), values)]
        )
        result = project_at(
            events,
            paths,
            pd.Series(event_row("target", 9_999, fill_bars=2)),
            {
                "elapsed_bars": 1.0,
                "current_move_pct": 2.0,
                "peak_move_pct": 3.0,
                "drawdown_from_peak_pct": 1.0,
            },
            snapshot_close_time_ms=1_300,
            current_target=100.0,
            current_direction_sign=1,
            projection_start_open_time_ms=1_300,
            interval_minutes=5,
            top_k=12,
        )
        self.assertTrue(result["insufficient_matches"])
        self.assertEqual(len(result["matched_states"]), 11)
        self.assertEqual(result["scenarios"], [])

    def test_project_at_returns_insufficient_state_when_no_history_is_available(self) -> None:
        events = pd.DataFrame([event_row("future", 1_600, fill_bars=2)])
        paths = pd.DataFrame(
            path_rows("future", [(0.1, 2.0, 0.0, 0.2), (1.7, 3.0, 0.5, 2.0), (0.2, 5.0, -0.1, 0.0)])
        )
        result = project_at(
            events,
            paths,
            pd.Series(event_row("target", 9_999, fill_bars=2)),
            {
                "elapsed_bars": 1.0,
                "current_move_pct": 2.0,
                "peak_move_pct": 3.0,
                "drawdown_from_peak_pct": 1.0,
            },
            snapshot_close_time_ms=1_300,
            current_target=100.0,
            current_direction_sign=1,
            projection_start_open_time_ms=1_300,
            interval_minutes=5,
            top_k=12,
        )
        self.assertTrue(result["insufficient_matches"])
        self.assertTrue(result["eligible_events"].empty)
        self.assertTrue(result["trajectory_events"].empty)
        self.assertTrue(result["matched_states"].empty)
        self.assertEqual(result["scenarios"], [])

    def test_same_candle_fill_and_departure_is_not_clean_departure(self) -> None:
        status, departure, fill = trace_clean_path(
            closes=np.asarray([100.0, 110.0]),
            highs=np.asarray([110.0, 111.0]),
            lows=np.asarray([100.0, 100.0]),
            signal_index=0,
            direction_sign=1,
            wick_target=100.0,
            opposite_extreme=110.0,
            maximum_future_bars=1,
        )
        self.assertEqual((status, departure, fill), ("filled_before_departure", None, None))

    def test_missing_candle_ranges_leave_exact_repair_boundaries(self) -> None:
        self.assertEqual(
            missing_candle_ranges([0, 5, 15, 30], start_ms=0, end_exclusive_ms=40, interval_ms=5),
            [(10, 15), (20, 30), (35, 40)],
        )

    def test_refresh_verifier_aligns_selected_timeframe_as_of(self) -> None:
        self.assertEqual(
            expected_projection_as_of("2026-09-20T19:40:00Z", "1m"),
            "2026-09-20T19:40:00Z",
        )
        self.assertEqual(
            expected_projection_as_of("2026-09-20T19:40:00Z", "5m"),
            "2026-09-20T19:40:00Z",
        )
        self.assertEqual(
            expected_projection_as_of("2026-09-20T19:40:00Z", "15m"),
            "2026-09-20T19:30:00Z",
        )

    def test_all_assets_are_configured_for_one_minute(self) -> None:
        self.assertEqual(set(assets_for_timeframe("1m")), set(SUPPORTED_ASSETS))
        self.assertEqual(set(assets_for_timeframe("5m")), set(SUPPORTED_ASSETS))

    def test_dashboard_routes_each_one_minute_asset_to_its_own_source_and_library(self) -> None:
        engine = Engine(Path("fixture-root"))
        for asset in SUPPORTED_ASSETS:
            self.assertEqual(engine._source_path(asset, "1m").name, f"{asset}_1m_5y.csv")
            self.assertEqual(
                engine._library_dir_for_timeframe(asset, "1m").name,
                f"conditional_path_library_{asset.removesuffix('USDT').lower()}_1m_5y",
            )
            validate_asset_timeframe(asset, "1m")

    def test_compact_sol_one_minute_paths_match_the_dashboard_read_contract(self) -> None:
        source = pd.DataFrame(
            {
                "open_time": [0, 60_000, 120_000],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 103.0, 104.0],
                "low": [99.0, 100.0, 98.0],
                "close": [100.5, 102.0, 100.0],
            }
        )
        event = {
            "episode_id": "SOLUSDT_1m_lower_wick_0",
            "asset": SOL_ONE_MINUTE_ASSET,
            "timeframe": "1m",
            "direction": "lower_wick",
            "direction_sign": 1,
            "signal_index": 0,
            "fill_index": 2,
            "wick_target": 100.0,
        }
        with TemporaryDirectory() as directory:
            paths_dir = Path(directory)
            filename = "SOLUSDT_1m_paths.csv.gz"
            self.assertEqual(write_compact_paths(paths_dir / filename, source, [event]), 3)
            loaded = read_library_paths(paths_dir, [filename])
        self.assertEqual(loaded["offset_bars"].tolist(), [0, 1, 2])
        self.assertAlmostEqual(float(loaded["normalized_high_pct"].iat[1]), 3.0)

    def test_strict_signal_detector_excludes_nonfinite_context_features(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": np.arange(61, dtype=np.int64) * 60_000,
                "open": np.full(61, 100.0),
                "high": np.full(61, 100.0),
                "low": np.full(61, 100.0),
                "close": np.full(61, 100.0),
                "volume": np.ones(61),
            }
        )
        frame.loc[60, ["high", "low"]] = [101.0, 90.0]
        signals = detect_strict_signals(frame, SOL_ONE_MINUTE_ASSET, "1m")
        self.assertTrue(signals.empty)

    def test_one_minute_matching_snapshot_grid_preserves_early_and_adaptive_offsets(self) -> None:
        states = pd.DataFrame(
            {
                "offset_bars": [1, 239, 240, 241, 245, 1_440, 1_441, 1_455, 1_456],
                "candidate_after_departure": [True, True, True, True, True, True, True, True, False],
            }
        )
        sampled = sampled_matching_states(states, "1m")
        self.assertEqual(sampled["offset_bars"].tolist(), [1, 239, 240, 245, 1_440, 1_455])
        self.assertIs(sampled_matching_states(states, "5m"), states)

    def test_refresh_swap_retries_a_transient_windows_reader_lock(self) -> None:
        with (
            patch("refresh_futures_klines.os.replace", side_effect=[PermissionError("locked"), None]) as replace,
            patch("refresh_futures_klines.time.sleep") as sleep,
        ):
            _replace_with_retry(Path("temporary.csv"), Path("source.csv"), attempts=2)
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once_with(0.15)

    def test_v2_support_and_static_artifact_metadata(self) -> None:
        self.assertIn("after the snapshot", FUTURE_AWAY_TARGET_SEMANTICS)
        self.assertIn("terminal fill candle", FUTURE_AWAY_TARGET_SEMANTICS)
        self.assertNotIn("includes the observed", FUTURE_AWAY_TARGET_SEMANTICS)
        self.assertEqual(snapshot_age_support(12, [12, 24])["status"], "exact_sampled_age")
        self.assertEqual(snapshot_age_support(13, [12, 24])["status"], "between_sampled_ages")
        self.assertEqual(snapshot_age_support(25, [12, 24])["status"], "outside_sampled_age")
        self.assertEqual(snapshot_age_support(12, [])["status"], "unknown")
        metadata = v2_artifact_metadata(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "generated_at_utc": "2026-09-20T09:00:00Z",
                "conditional_population": "clean fills only",
                "chronological_split": {
                    "max_train_fill_close_utc": "2025-09-19T03:25:00Z",
                    "holdout_start_utc": "2025-09-20T00:00:00Z",
                    "train_episode_count": 42,
                },
                "data": {"train_rows_used": 100},
            }
        )
        self.assertEqual(metadata["training_label_cutoff_utc"], "2025-09-19T03:25:00Z")
        self.assertEqual(metadata["training_snapshot_rows"], 100)
        old_metadata = v2_artifact_metadata({})
        self.assertIsNone(old_metadata["generated_at_utc"])
        self.assertIsNone(old_metadata["training_label_cutoff_utc"])

    def test_v2_asset_universe_and_one_hot_live_state(self) -> None:
        schema = observable_state_schema()
        self.assertEqual(schema["schema_version"], MODEL_SCHEMA_VERSION)
        self.assertEqual(tuple(schema["required_categorical_fields"]["asset"]), SUPPORTED_ASSETS)
        self.assertTrue(set(ASSET_CATEGORICAL_FEATURE_COLUMNS).issubset(schema["derived_inside_v2"]))

        state: dict[str, object] = {"asset": "SOLUSDT", "direction": "lower_wick"}
        state.update({name: 1.0 for name in OBSERVABLE_STATE_INPUT_COLUMNS})
        state["drawdown_from_peak_pct"] = 0.0
        frame = observable_state_frame(state)
        row = frame.iloc[0]
        self.assertEqual(float(row[asset_indicator_column("SOLUSDT")]), 1.0)
        self.assertEqual(
            sum(float(row[asset_indicator_column(asset)]) for asset in SUPPORTED_ASSETS),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
