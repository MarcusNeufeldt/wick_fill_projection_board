from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import train_neural_path_v3 as training
from neural_path_v3_live import live_age_bounds
from train_neural_path_v3 import (
    EPISODE_LENGTH,
    EPISODE_SEQUENCE_CHANNELS,
    FUTURE_HORIZON_BARS,
    PRE_SIGNAL_LENGTH,
    RAW_SEQUENCE_CHANNELS,
    RECENT_LENGTH,
    NeuralPathModel,
    age_bucket_labels,
    age_episode_balanced_weights,
    age_stratified_cap,
    apply_sequence_scaler,
    decode_curve_distances,
    direction_normalized_pct,
    episode_balanced_bootstrap_delta,
    fit_sequence_scaler,
    future_targets,
    inner_chronological_split,
    path_metrics,
)


class NeuralPathV3Tests(unittest.TestCase):
    def test_one_minute_context_keeps_wall_clock_span_and_tensor_shape(self) -> None:
        try:
            training.configure_timeframe("1m")
            self.assertEqual(training.BAR_MINUTES, 1)
            self.assertEqual(training.PRE_SIGNAL_SOURCE_BARS, 8 * 60)
            self.assertEqual(training.RECENT_SOURCE_BARS, 1_280)
            self.assertEqual(training.PRE_SIGNAL_LENGTH, 96)
            self.assertEqual(training.RECENT_LENGTH, 256)

            count = 500
            times = np.arange(count, dtype=np.int64) * 60_000
            close = 100.0 + np.arange(count, dtype=np.float32) * 0.01
            raw = {
                "open_time": times,
                "open": close - 0.01,
                "high": close + 0.03,
                "low": close - 0.03,
                "close": close,
                "volume": np.full(count, 10.0, dtype=np.float32),
            }
            window = training.raw_window(
                raw,
                end_open_time_ms=int(times[-1]),
                length=training.PRE_SIGNAL_LENGTH,
                source_length=training.PRE_SIGNAL_SOURCE_BARS,
                target=100.0,
                direction_sign=1,
                signal_volume=10.0,
                signal_open_time_ms=int(times[-1]),
            )
            self.assertEqual(
                window.shape,
                (training.PRE_SIGNAL_LENGTH, len(training.RAW_SEQUENCE_CHANNELS)),
            )
            self.assertTrue(np.all(window[:, -1] == 1.0))
            expected_first = training.signed_log1p(
                training.direction_normalized_pct(
                    np.asarray([close[-training.PRE_SIGNAL_SOURCE_BARS]]), 100.0, 1
                )
            )[0]
            expected_last = training.signed_log1p(
                training.direction_normalized_pct(
                    np.asarray([close[-1]]), 100.0, 1
                )
            )[0]
            self.assertAlmostEqual(float(window[0, 3]), float(expected_first), places=5)
            self.assertAlmostEqual(float(window[-1, 3]), float(expected_last), places=5)
        finally:
            training.configure_timeframe("5m")

    def test_age_balanced_weights_give_each_regime_equal_mass(self) -> None:
        frame = pd.DataFrame(
            {
                "episode_id": ["young-a", "young-a", "young-b", "old-a"],
                "offset_bars": [1, 3, 12, 1_440],
            }
        )
        weights = age_episode_balanced_weights(frame)
        labels = age_bucket_labels(frame["offset_bars"])
        young_mass = float(weights[labels == "1-24"].sum())
        old_mass = float(weights[labels == "961-1440"].sum())
        self.assertAlmostEqual(young_mass, old_mass, places=6)

    def test_age_stratified_cap_preserves_old_rows(self) -> None:
        rows = []
        for index in range(100):
            rows.append(
                {
                    "episode_id": f"young-{index}",
                    "offset_bars": 12,
                    "asset": "ETHUSDT",
                    "direction": "lower_wick",
                    "snapshot_close_time_ms": index,
                }
            )
        for index in range(5):
            rows.append(
                {
                    "episode_id": f"old-{index}",
                    "offset_bars": 1_440,
                    "asset": "ETHUSDT",
                    "direction": "lower_wick",
                    "snapshot_close_time_ms": 1_000 + index,
                }
            )
        selected = age_stratified_cap(pd.DataFrame(rows), maximum=20)
        self.assertEqual(int((selected["offset_bars"] == 1_440).sum()), 5)
        self.assertEqual(len(selected), 20)

    def test_direction_normalization_mirrors_upper_and_lower_wicks(self) -> None:
        lower = direction_normalized_pct(np.asarray([105.0]), 100.0, 1)
        upper = direction_normalized_pct(np.asarray([95.0]), 100.0, -1)
        np.testing.assert_allclose(lower, upper, rtol=0, atol=1e-8)

    def test_sequence_scaler_keeps_padding_zero(self) -> None:
        values = np.zeros((2, 4, 3), dtype=np.float32)
        values[0, 2:, 0] = [2.0, 4.0]
        values[0, 2:, 1] = [10.0, 14.0]
        values[0, 2:, 2] = 1.0
        values[1, 1:, 0] = [1.0, 3.0, 5.0]
        values[1, 1:, 1] = [8.0, 12.0, 16.0]
        values[1, 1:, 2] = 1.0
        transformed = apply_sequence_scaler(values, fit_sequence_scaler(values))
        self.assertTrue(
            np.all(transformed[:, :, :-1][transformed[:, :, -1] == 0.0] == 0.0)
        )
        self.assertTrue(np.all(transformed[:, :, -1] == values[:, :, -1]))

    def test_future_targets_stop_at_fill_touch(self) -> None:
        path = pd.DataFrame(
            {
                "offset_bars": np.arange(6),
                "normalized_close_pct": [0.2, 1.0, 1.4, 0.8, 0.3, 0.1],
            }
        )
        curve, raw_distances, direction = future_targets(
            path, snapshot_offset=2, current_move_pct=1.4
        )
        distances = decode_curve_distances(curve[None, :], np.asarray([1.4]))[0]
        self.assertAlmostEqual(float(distances[0]), 0.8, places=5)
        np.testing.assert_allclose(distances, raw_distances, rtol=0, atol=1e-5)
        self.assertEqual(float(distances[1]), 0.0)
        self.assertTrue(np.all(distances[2:] == 0.0))
        self.assertTrue(np.all(direction == 1.0))

    def test_future_targets_preserve_raw_truth_below_ratio_floor(self) -> None:
        path = pd.DataFrame(
            {
                "offset_bars": np.arange(8),
                "normalized_close_pct": [
                    0.01,
                    0.015,
                    0.018,
                    0.02,
                    0.02,
                    0.02,
                    0.02,
                    0.0,
                ],
            }
        )
        curve, raw_distances, direction = future_targets(
            path, snapshot_offset=0, current_move_pct=0.01
        )
        decoded = decode_curve_distances(curve[None, :], np.asarray([0.01]))[0]
        self.assertAlmostEqual(float(raw_distances[0]), 0.015, places=6)
        self.assertAlmostEqual(float(decoded[0]), 0.015, places=6)
        self.assertEqual(float(direction[0]), 0.0)

    def test_path_metrics_score_raw_truth_beyond_training_clip(self) -> None:
        samples = SimpleNamespace(
            current_move_pct=np.asarray([0.1], dtype=np.float32),
            curve_distance_pct=np.full((1, len(FUTURE_HORIZON_BARS)), 3.0),
        )
        clipped_prediction = np.full(
            (1, len(FUTURE_HORIZON_BARS)), np.log1p(25.0), dtype=np.float32
        )
        result = path_metrics(samples, clipped_prediction)
        self.assertAlmostEqual(
            result["mean_absolute_distance_error_pct_points"], 0.5, places=6
        )

    def test_live_age_gate_is_narrower_than_artifact_support(self) -> None:
        bundle = SimpleNamespace(
            snapshot_offsets_bars=(1, 3, 120, 480, 2880),
            live_min_age_bars=1,
            live_max_age_bars=120,
        )
        self.assertEqual(live_age_bounds(bundle), (1, 120))

    def test_live_age_gate_can_select_a_mature_expert(self) -> None:
        bundle = SimpleNamespace(
            snapshot_offsets_bars=(480, 600, 2_880),
            live_min_age_bars=481,
            live_max_age_bars=2_880,
        )
        self.assertEqual(live_age_bounds(bundle), (481, 2_880))

    def test_inner_split_is_episode_disjoint_and_label_available(self) -> None:
        rows = []
        day_ms = 24 * 60 * 60 * 1000
        for episode in range(20):
            signal = episode * 10 * day_ms
            rows.append(
                {
                    "episode_id": f"episode-{episode}",
                    "signal_open_time_ms": signal,
                    "snapshot_close_time_ms": signal + day_ms,
                    "fill_close_time_ms": signal + 2 * day_ms,
                }
            )
        fit, validation, details = inner_chronological_split(
            pd.DataFrame(rows), validation_fraction=0.2, embargo_bars=1
        )
        self.assertFalse(set(fit["episode_id"]).intersection(validation["episode_id"]))
        self.assertLessEqual(
            int(fit["fill_close_time_ms"].max()),
            int(validation["snapshot_close_time_ms"].min()),
        )
        self.assertTrue(
            details["all_fit_labels_resolved_before_every_validation_snapshot"]
        )

    def test_model_shapes(self) -> None:
        model = NeuralPathModel(
            static_features=7, sequence_width=8, embedding_dim=16, dropout=0.0
        )
        batch = 3
        output = model(
            torch.zeros(batch, len(RAW_SEQUENCE_CHANNELS), PRE_SIGNAL_LENGTH),
            torch.zeros(batch, len(RAW_SEQUENCE_CHANNELS), RECENT_LENGTH),
            torch.zeros(batch, len(EPISODE_SEQUENCE_CHANNELS), EPISODE_LENGTH),
            torch.zeros(batch, 7),
        )
        self.assertEqual(tuple(output["embedding"].shape), (batch, 16))
        self.assertEqual(tuple(output["risk"].shape), (batch, 2, 3))
        self.assertEqual(
            tuple(output["curve"].shape), (batch, len(FUTURE_HORIZON_BARS))
        )
        norms = torch.linalg.vector_norm(output["embedding"], dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_episode_bootstrap_detects_uniform_improvement(self) -> None:
        baseline = np.asarray([2.0, 4.0, 3.0, 5.0], dtype=float)
        challenger = baseline - 1.0
        episodes = np.asarray(["a", "a", "b", "c"])
        result = episode_balanced_bootstrap_delta(
            baseline, challenger, episodes, seed=7, repetitions=200
        )
        self.assertAlmostEqual(result["delta_challenger_minus_scalar"], -1.0)
        self.assertEqual(result["probability_challenger_better"], 1.0)
        self.assertLess(result["bootstrap_95pct_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
