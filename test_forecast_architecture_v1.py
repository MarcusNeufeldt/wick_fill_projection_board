from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from forecast_architecture_v1 import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SCHEMA_VERSION,
    predict_architecture,
)
from forecast_artifact_manifest import resolve_active_artifact, write_active_manifest
from prospective_entry_model import MODEL_SCHEMA_VERSION, live_observation
from prospective_forecast_ledger import record_forecast
from serve_conditional_wick_dashboard import Engine
from train_forecast_architecture_v1 import train as train_forecast_architecture


class ConstantClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.tile(
            np.asarray([[1.0 - self.probability, self.probability]]),
            (len(matrix), 1),
        )


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(len(matrix), self.value, dtype=float)


def base_prediction(fill: float, risk: float, wait: float) -> dict:
    return {
        "fill_probability": {"1440m": fill},
        "additional_adverse_pct": {
            "1440m": {"p50": risk, "p80": risk + 1.0, "p90": risk + 2.0}
        },
        "remaining_time_minutes_if_filled_within_horizon": {
            "1440m": {"p10": wait / 2.0, "p50": wait, "p90": wait * 2.0}
        },
        "competing_outcomes": {},
    }


class ForecastArchitectureTests(unittest.TestCase):
    def test_prospective_ledger_is_append_only_per_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecasts.sqlite"
            values = dict(
                architecture_version=ARCHITECTURE_ID,
                artifact_hash="a" * 64,
                forecast_source="forecast_v1",
                asset="ETHUSDT",
                timeframe="5m",
                signal_open_time_ms=100,
                observation_close_time_ms=200,
                direction="lower_wick",
                wick_target=99.0,
                entry_price=105.0,
                payload={"fill_probability": {"1440m": 0.7}},
            )
            self.assertTrue(record_forecast(path, **values))
            self.assertFalse(record_forecast(path, **values))
            fallback_values = dict(values)
            fallback_values.update(
                architecture_version="legacy_5m_deadbeef",
                artifact_hash="b" * 64,
                forecast_source="legacy_fallback",
            )
            self.assertTrue(record_forecast(path, **fallback_values))
            connection = sqlite3.connect(path)
            try:
                count = connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
                evaluation_count = connection.execute(
                    "SELECT COUNT(*) FROM forecast_v1_evaluation"
                ).fetchone()[0]
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(forecasts)")
                }
            finally:
                connection.close()
            self.assertEqual(count, 2)
            self.assertEqual(evaluation_count, 1)
            self.assertTrue(
                {"architecture_version", "artifact_hash", "forecast_source"}.issubset(
                    columns
                )
            )

    def test_live_observation_contains_sequence_coordinates(self) -> None:
        frame = pd.DataFrame(
            {
                "close": np.linspace(100.0, 110.0, 20),
                "close_time": np.arange(20, dtype=np.int64) * 300_000,
                "high": np.linspace(101.0, 111.0, 20),
                "low": np.linspace(99.0, 109.0, 20),
                "open": np.linspace(100.0, 110.0, 20),
                "volume": np.ones(20),
            }
        )
        signal = {
            "bar_index": 5,
            "direction": "lower_wick",
            "direction_sign": 1,
            "wick_target": 99.0,
        }
        from build_conditional_path_library import FEATURE_COLUMNS

        signal.update({name: 0.0 for name in FEATURE_COLUMNS})
        query, _ = live_observation(
            frame, signal, "ETHUSDT", "5m", 19, 6, 10.0, 12.0, 2.0
        )
        self.assertEqual(query["wick_target"], 99.0)
        self.assertEqual(query["signal_to_entry_start_index"], 5)
        self.assertEqual(query["signal_to_entry_end_index"], 19)
        self.assertEqual(query["recent_end_index"], 19)

    def test_composite_uses_e0_for_fill_and_p50_but_a_for_wait_and_tail(self) -> None:
        support = {
            "fingerprint_name": "state_only",
            "block_order": ["state"],
            "blocks": {
                "state": {
                    "columns": ["x"],
                    "scaler": {"median": [0.0], "scale": [1.0]},
                    "weight": 1.0,
                }
            },
            "neighborhood": {
                "k": 3,
                "weighting": "uniform",
                "radius_quantile": None,
                "maximum_distance": None,
            },
            "history_matrix": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            "history_signal_ids": np.asarray(["a", "b", "c"]),
            "history_assets": np.asarray(["ETHUSDT", "BTCUSDT", "ETHUSDT"]),
            "history_fill": np.asarray([1.0, 1.0, 0.0]),
            "history_adverse_pct": np.asarray([1.0, 2.0, 4.0]),
            "history_wait_minutes": np.asarray([60.0, 120.0, np.nan]),
            "calibration_nearest_neighbor_distances": np.asarray([0.0, 0.5, 1.0]),
            "calibration_median_neighbor_distances": np.asarray([0.5, 1.0, 2.0]),
        }
        bundle = {
            "schema_version": ARCHITECTURE_SCHEMA_VERSION,
            "architecture_id": ARCHITECTURE_ID,
            "horizons_minutes": [1_440],
            "base_a": {},
            "base_b": {},
            "e0_heads": {
                "1440m": {
                    "fill_model": ConstantClassifier(0.7),
                    "fill_constant": None,
                    "risk_model": ConstantRegressor(4.0),
                }
            },
            "support_artifacts": {"1440m": support},
            "ownership": {
                "fill_probability": "E0",
                "adverse_p50": "E0",
                "adverse_p80_p90": "A",
                "waiting_time": "A",
                "historical_support": "C2",
                "routes": "separate",
            },
        }
        a = base_prediction(0.4, 1.0, 100.0)
        b = base_prediction(0.6, 1.5, 80.0)
        with patch(
            "forecast_architecture_v1.predict_prospective_bundle",
            side_effect=[a, b],
        ):
            prediction = predict_architecture(
                bundle,
                pd.DataFrame([{"x": 0.25}]),
                pd.DataFrame(index=[0]),
                "ETHUSDT",
            )
        self.assertAlmostEqual(prediction["fill_probability"]["1440m"], 0.7)
        self.assertAlmostEqual(
            prediction["additional_adverse_pct"]["1440m"]["p50"], 4.0
        )
        self.assertGreaterEqual(
            prediction["additional_adverse_pct"]["1440m"]["p90"], 4.0
        )
        self.assertEqual(
            prediction["remaining_time_minutes_if_filled_within_horizon"]["1440m"][
                "p50"
            ],
            100.0,
        )
        self.assertEqual(prediction["ownership"]["waiting_time"], "A")
        self.assertIn(
            prediction["historical_support"]["1440m"]["level"],
            {"very_high", "high", "medium", "low", "very_low"},
        )
        route_candidates = prediction["historical_route_candidates"]["1440m"]
        self.assertEqual([item["episode_id"] for item in route_candidates], ["a", "b"])
        self.assertEqual(route_candidates[0]["neighbor_rank"], 1)
        self.assertEqual(route_candidates[0]["wait_minutes"], 60.0)

    def test_c2_route_engine_selects_four_real_historical_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = Engine(Path(directory))
            frame = pd.DataFrame(
                {
                    "close_time": [10_000],
                    "close": [102.0],
                }
            )
            signal = pd.DataFrame(
                [
                    {
                        "open_time": 1_000,
                        "wick_target": 100.0,
                        "direction_sign": 1,
                        "interval_minutes": 5,
                    }
                ]
            )
            horizon_waits = {
                "1440m": [60.0, 90.0, 120.0, 150.0],
                "10080m": [300.0, 450.0, 600.0, 750.0],
                "43200m": [1_500.0, 2_500.0, 3_500.0, 4_500.0],
            }
            candidates: dict[str, list[dict]] = {}
            episode_rows = []
            state_rows = []
            spans = {}
            for horizon_index, (slug, waits) in enumerate(horizon_waits.items()):
                candidates[slug] = []
                for item_index, wait in enumerate(waits):
                    episode_id = f"episode_{horizon_index}_{item_index}"
                    candidates[slug].append(
                        {
                            "episode_id": episode_id,
                            "asset": "ETHUSDT",
                            "distance": 0.1 + item_index * 0.01,
                            "neighbor_rank": item_index + 1,
                            "wait_minutes": wait,
                            "adverse_pct": 1.0,
                        }
                    )
                    episode_rows.append(
                        {"episode_id": episode_id, "fill_close_time_ms": 9_000}
                    )
                    row_index = len(state_rows)
                    state_rows.append(
                        {
                            "episode_id": episode_id,
                            "asset": "ETHUSDT",
                            "timeframe": "5m",
                            "direction": "lower_wick",
                            "offset_bars": 1,
                            "remaining_to_fill_bars": int(wait / 5.0),
                            "alignment_current_move_pct": 2.0,
                            "alignment_peak_move_pct": 3.0,
                            "alignment_drawdown_pct": 1.0,
                            "future_peak_move_pct": 3.0,
                        }
                    )
                    state_rows.append(
                        {
                            "episode_id": episode_id,
                            "offset_bars": 2,
                            "normalized_open_pct": 4.1,
                            "normalized_high_pct": 4.2,
                            "normalized_low_pct": 4.0,
                            "normalized_close_pct": 4.1,
                        }
                    )
                    spans[episode_id] = (row_index, row_index + 2)
            risk = {
                "artifact": {
                    "forecast_source": "forecast_v1",
                    "architecture_version": ARCHITECTURE_ID,
                    "artifact_sha256": "a" * 64,
                },
                "remaining_time_minutes_if_filled_within_horizon": {
                    "1440m": {"p10": 90.0, "p50": 105.0},
                    "10080m": {"p50": 450.0},
                    "43200m": {"p90": 2_500.0},
                },
                "additional_adverse_pct": {
                    "1440m": {"p50": 1.0, "p80": 2.0},
                    "10080m": {"p50": 1.0},
                    "43200m": {"p90": 1.0},
                },
            }
            projection = {
                "current_state": {"current_move_pct": 2.0},
                "library": {},
                "scenarios": [],
            }
            candles = [
                {"time": 1, "open": 102.0, "high": 105.0, "low": 102.0, "close": 104.0},
                {"time": 301, "open": 104.0, "high": 104.0, "low": 100.0, "close": 100.0},
            ]
            with (
                patch.object(engine, "_frame", return_value=frame),
                patch.object(engine, "_signals", return_value=signal),
                patch.object(
                    engine,
                    "_library_states",
                    return_value=(
                        pd.DataFrame(episode_rows),
                        pd.DataFrame(state_rows),
                        None,
                        spans,
                    ),
                ),
                patch(
                    "serve_conditional_wick_dashboard.projected_candles",
                    return_value=(candles, 1.0),
                ),
                patch(
                    "serve_conditional_wick_dashboard.projected_path_metrics",
                    return_value={
                        "projected_future_max_away_move_pct": 3.0,
                        "projected_additional_adverse_move_pct": 1.0,
                    },
                ),
            ):
                result = engine._apply_c2_numerical_routes(
                    "ETHUSDT", "5m", 1_000, projection, risk, candidates
                )

        self.assertTrue(result["active"])
        self.assertEqual(
            [scenario["name"] for scenario in projection["scenarios"]],
            ["fast", "normal", "adverse", "extreme"],
        )
        self.assertTrue(
            all(
                scenario["selector"] == "c2_e0_a_numerical_aligned_real_path"
                for scenario in projection["scenarios"]
            )
        )
        adverse = next(
            scenario
            for scenario in projection["scenarios"]
            if scenario["name"] == "adverse"
        )
        self.assertEqual(adverse["adverse_first_threshold_pct"], 2.0)
        self.assertEqual(adverse["projected_adverse_threshold_bars"], 1)
        self.assertEqual(adverse["projected_adverse_ordering"], "adverse_before_fill")

    def test_dashboard_prefers_frozen_5m_artifact_without_affecting_1m(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "models"
            (model_root / "5m").mkdir(parents=True)
            artifact = model_root / "5m" / f"{ARCHITECTURE_ID}.joblib"
            joblib.dump(
                {
                    "schema_version": ARCHITECTURE_SCHEMA_VERSION,
                    "architecture_id": ARCHITECTURE_ID,
                },
                artifact,
            )
            manifest = write_active_manifest(
                model_root / "5m",
                artifact,
                architecture_version=ARCHITECTURE_ID,
                training_labels_through="2026-09-22T10:39:59.999000Z",
                frozen=True,
            )
            engine = Engine(root)
            engine.prospective_artifact_root = model_root
            model, error = engine._prospective_bundle("5m")
            self.assertIsNone(error)
            self.assertEqual(model["schema_version"], ARCHITECTURE_SCHEMA_VERSION)
            self.assertEqual(model["_forecast_source"], "forecast_v1")
            self.assertEqual(model["_artifact_sha256"], manifest["artifact_sha256"])
            _, one_minute_error = engine._prospective_bundle("1m")
            self.assertIn("unavailable", one_minute_error)

    def test_hash_mismatch_fails_closed_to_legacy_and_is_not_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models" / "5m"
            model_dir.mkdir(parents=True)
            artifact = model_dir / f"{ARCHITECTURE_ID}.joblib"
            joblib.dump(
                {
                    "schema_version": ARCHITECTURE_SCHEMA_VERSION,
                    "architecture_id": ARCHITECTURE_ID,
                },
                artifact,
            )
            write_active_manifest(
                model_dir,
                artifact,
                architecture_version=ARCHITECTURE_ID,
                training_labels_through="2026-09-22T10:39:59.999000Z",
                frozen=True,
            )
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                resolve_active_artifact(model_dir)
            joblib.dump(
                {"schema_version": MODEL_SCHEMA_VERSION},
                model_dir / "model.joblib",
            )
            engine = Engine(root)
            engine.prospective_artifact_root = root / "models"
            model, error = engine._prospective_bundle("5m")
            self.assertIsNone(error)
            self.assertEqual(model["_forecast_source"], "legacy_fallback")
            self.assertIn("hash mismatch", model["_active_forecast_error"])

    def test_trainer_refuses_to_overwrite_active_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models" / "5m"
            model_dir.mkdir(parents=True)
            artifact = model_dir / f"{ARCHITECTURE_ID}.joblib"
            artifact.write_bytes(b"frozen")
            write_active_manifest(
                model_dir,
                artifact,
                architecture_version=ARCHITECTURE_ID,
                training_labels_through="2026-09-22T10:39:59.999000Z",
                frozen=True,
            )
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                train_forecast_architecture(root, root / "missing", artifact)
            candidate = model_dir / "candidate.joblib"
            candidate.write_bytes(b"candidate")
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
                write_active_manifest(
                    model_dir,
                    candidate,
                    architecture_version="candidate",
                    training_labels_through="2026-09-27T00:00:00Z",
                    frozen=True,
                )


if __name__ == "__main__":
    unittest.main()
